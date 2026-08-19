# -*- coding: utf-8 -*-
"""
GPU-resident, batched multislice ePIE (3PIE) for ptypy, backed by PyTorch.

Design
------
* All slices, the probe, the diffraction data and the masks live on the GPU for
  the whole reconstruction. Nothing crosses the PCIe bus in the inner loop.
* The Fourier modulus projection is reimplemented in torch, so there is no
  round trip to ptypy's CPU projection.
* Scan positions are processed in mini-batches. The multislice forward pass is
  inherently sequential in z and cannot be batched over slices, but it batches
  perfectly over positions, which is where the speedup is for small N.

IMPORTANT -- THIS IS NOT SEQUENTIAL ePIE
----------------------------------------
With batch_size > 1 the object is updated once per batch from accumulated
gradients, rather than once per view. This is the mini-batch generalisation used
by fold_slice / PtyRAD and is generally better conditioned, but it is a
different algorithm from 3PIE and its per-iteration convergence differs. Set
batch_size = 1 to recover sequential semantics (and lose most of the speedup).

The per-view ePIE denominator max|P|^2 is replaced by the overlap-accumulated
sum_v |P_v|^2 + reg, which is the correct preconditioner for a batched update.

PROPAGATOR EXTRACTION
---------------------
Rather than reimplementing ptypy's FFT conventions (normalisation, fftshift,
chirp factors) and hoping they match, each ptypy propagator is probed
numerically at initialisation:

  * shift-invariant (near-field, slice-to-slice):  fw(w) = IDFT(H * DFT(w))
    recovered exactly from H = DFT(fw(delta_00)).

  * chirp-DFT (far-field, sample-to-detector):     fw(w) = A * DFT(R * w)
    recovered exactly from A = fw(delta_00) and R = IDFT(fw(ones) / A).

Both are then verified against the ptypy operator on random data. A convention
mismatch is a startup error, not a silently wrong reconstruction.

LIMITATIONS
-----------
* Single object storage, single object mode. Multiple probe modes supported.
* No position refinement, no probe recentering in the batched path.
* MPI is not supported (inherited from ptypy's stochastic engines).
* The reported error values follow ptypy's definitions in spirit but the
  normalisation may differ; compare against the CPU engine before quoting them.
"""

import numpy as np

from ptypy.engines import stochastic
from ptypy.engines import register
from ptypy.core import geometry
from ptypy.utils import Param
from ptypy.utils.verbose import logger
from ptypy import io
from matplotlib import pyplot as plt
import os
from datetime import datetime
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------#
#  Numerical extraction of a ptypy propagator                                 #
# ---------------------------------------------------------------------------#
def _op_call(op, arr):
    """Call a ptypy propagator, tolerating single- or double-precision input."""
    last = None
    for dt in (np.complex64, np.complex128):
        try:
            return np.asarray(op(arr.astype(dt)), dtype=np.complex128)
        except Exception as e:      # noqa: BLE001 - propagator dtype pickiness
            last = e
    raise RuntimeError("could not call propagator: %s" % last)


def _apply_numpy(kind, d, w):
    if kind == 'lsi':                       # ifft2(fft2(w) * H)
        return np.fft.ifft2(np.fft.fft2(w) * d['H'])
    if kind == 'ilsi':                      # fft2(ifft2(w) * H)
        return np.fft.fft2(np.fft.ifft2(w) * d['H'])
    if kind == 'chirp':                     # A * fft2(R * w)
        return d['A'] * np.fft.fft2(d['R'] * w)
    if kind == 'ichirp':                    # A * ifft2(R * w)
        return d['A'] * np.fft.ifft2(d['R'] * w)
    raise ValueError(kind)


def _extract_operator(op, shape, name="propagator", rtol=1e-5):
    """
    Recover a ptypy propagator exactly, without assuming its internals.

    Four candidate forms are built from two probes and the one that reproduces
    the operator on random data is kept:

        lsi     op(w) = ifft2(fft2(w) * H)       near-field forward
        ilsi    op(w) = fft2(ifft2(w) * H)       near-field reverse pairing
        chirp   op(w) = A * fft2(R * w)          far-field forward
        ichirp  op(w) = A * ifft2(R * w)         far-field reverse

    numpy's convention is used throughout (fft2 unnormalised, ifft2 carrying
    1/N), which torch reproduces with the default norm='backward'.
    """
    ny, nx = shape
    Ntot = float(ny * nx)

    d00 = np.zeros(shape, dtype=np.complex128)
    d00[0, 0] = 1.0
    ones = np.ones(shape, dtype=np.complex128)

    out00 = _op_call(op, d00)
    out1 = _op_call(op, ones)

    cands = [
        ('lsi',  {'H': np.fft.fft2(out00)}),
        ('ilsi', {'H': Ntot * np.fft.ifft2(out00)}),
    ]
    if np.abs(out00).min() > 0:
        A = out00
        cands.append(('chirp',  {'A': A, 'R': np.fft.ifft2(out1 / A)}))
        cands.append(('ichirp', {'A': A, 'R': np.fft.fft2(out1 / A)}))

    rng = np.random.default_rng(0)
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    ref = _op_call(op, w)
    scale = max(np.abs(ref).max(), 1e-30)

    best_kind, best_d, best_err = None, None, np.inf
    errs = {}
    for kind, d in cands:
        err = np.abs(_apply_numpy(kind, d, w) - ref).max() / scale
        errs[kind] = err
        if err < best_err:
            best_kind, best_d, best_err = kind, d, err

    if best_err > rtol:
        raise RuntimeError(
            "%s: numerical extraction failed. Errors by form: %s. The propagator "
            "is none of ifft2(fft2(w)*H), fft2(ifft2(w)*H), A*fft2(R*w), "
            "A*ifft2(R*w). Inspect it and hard-code the transfer function."
            % (name, {k: '%.2e' % v for k, v in errs.items()}))

    spec = {'kind': best_kind}
    spec.update(best_d)
    logger.info("%s extracted as '%s', verification error %.2e" % (name, best_kind, best_err))
    return spec


@register()
class ThreePIETorch(stochastic.EPIE):
    """
    Batched GPU multislice ePIE.

    Defaults:

    [name]
    default = ThreePIETorch
    type = str
    help = Engine identifier

    [number_of_slices]
    default = 2
    type = int
    help = The number of slices

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = Single float, or a list of length number_of_slices - 1.

    [slice_start_iteration]
    default = 0
    type = int, list, tuple
    help = Iteration at which each slice starts being used

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the slice data

    [device]
    default = cuda
    type = str
    help = Torch device
    doc = 'cuda', 'cuda:0', or 'cpu'. Falls back to cpu with a warning if CUDA
          is unavailable.

    [batch_size]
    default = 32
    type = int
    help = Number of scan positions per update
    doc = 1 reproduces sequential 3PIE semantics. Larger values are faster but
          change the algorithm to a mini-batch method. 16-64 is usually a good
          trade for 512x512 frames.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

    [denom_reg]
    default = 0.01
    type = float
    help = Tikhonov regularisation on update denominators
    doc = denom -> denom + denom_reg * max(denom). Replaces ePIE's max-based
          normalisation, which is ill-suited to accumulated batch gradients.

    [object_step]
    default = 1.0
    type = float
    help = Step size for the object slice updates

    [probe_step]
    default = 0.2
    type = float
    help = Step size for the illumination update

    [sync_interval]
    default = 1
    type = int
    help = Iterations between copying GPU state back into ptypy containers
    doc = Only needed for plotting and autosave. Set higher to save a little time.

    [shuffle]
    default = True
    type = bool
    help = Reshuffle the position order every iteration
    """

    def __init__(self, ptycho_parent, pars=None):
        super(ThreePIETorch, self).__init__(ptycho_parent, pars)
        self.article = dict(
            title='Ptychographic transmission microscopy in three dimensions using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29, year=2012, page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        self.ptycho.citations.add_article(**self.article)

        self.plot_progress_dir = f'iteration_progress{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.plot_progress_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()

        if torch is None:
            raise ImportError("ThreePIETorch requires PyTorch")

        N = self.p.number_of_slices

        # ---- device and dtype ----
        want = str(self.p.device)
        if want.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            want = 'cpu'
        self.dev = torch.device(want)
        double = str(self.p.precision).lower().startswith('d')
        self.cdt = torch.complex128 if double else torch.complex64
        self.rdt = torch.float64 if double else torch.float32
        logger.info("ThreePIETorch on %s, %s precision" % (self.dev, 'double' if double else 'single'))

        # ---- slice_start_iteration ----
        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = np.ones(N) * self.p.slice_start_iteration
        self.p.slice_start_iteration = np.asarray(self.p.slice_start_iteration)

        # ---- ptypy containers, kept for saving and plotting ----
        self._object = [self.ob.copy(self.ob.ID + "_o_" + str(i)) for i in range(N)]

        if len(self.ob.storages) != 1:
            raise NotImplementedError("ThreePIETorch supports a single object storage")
        self._ob_storage = list(self.ob.storages.values())[0]
        if self._ob_storage.data.shape[0] != 1:
            raise NotImplementedError("ThreePIETorch supports a single object mode")

        if len(self.pr.storages) != 1:
            raise NotImplementedError("ThreePIETorch supports a single probe storage")
        self._pr_storage = list(self.pr.storages.values())[0]

        obny, obnx = self._ob_storage.data.shape[-2:]
        self.M, self.fny, self.fnx = self._pr_storage.data.shape

        # ---- GPU state ----
        ob0 = self._ob_storage.data[0]
        self.ob_g = torch.stack([
            torch.as_tensor(np.ascontiguousarray(ob0), dtype=self.cdt, device=self.dev)
            for _ in range(N)])                                     # (N, obny, obnx)
        self.pr_g = torch.as_tensor(np.ascontiguousarray(self._pr_storage.data),
                                    dtype=self.cdt, device=self.dev)  # (M, fny, fnx)

        # ---- propagators ----
        self._build_propagators()

        # ---- scan geometry and data ----
        self._collect_views()

        # ---- optional circular probe support ----
        self.pr_support = None
        ps = getattr(self.p, 'probe_support', None)
        if ps is not None and 0 < float(ps) < 1:
            yy, xx = np.mgrid[:self.fny, :self.fnx]
            r = np.hypot(yy - self.fny / 2.0, xx - self.fnx / 2.0)
            mask = (r < float(ps) * min(self.fny, self.fnx) / 2.0).astype(np.float64)
            self.pr_support = torch.as_tensor(mask, dtype=self.rdt, device=self.dev)
            logger.info("probe support: circular, radius fraction %.3f" % float(ps))

        self._report_memory()

    def _build_propagators(self):
        """Extract the slice-to-slice near-field and the far-field propagators."""
        N = self.p.number_of_slices
        scan = list(self.ptycho.model.scans.values())[0]
        geom = scan.geometries[0]

        # --- near-field, slice to slice: build the same ptypy Geo objects the
        #     CPU engine does, then extract them numerically ---
        g = Param()
        g.energy = geom.energy
        g.psize = geom.resolution
        g.shape = geom.shape
        g.propagation = "nearfield"

        if type(self.p.slice_thickness) in [list, tuple]:
            if len(self.p.slice_thickness) != N - 1:
                raise ValueError("slice_thickness list must have length number_of_slices - 1")
            thicknesses = list(self.p.slice_thickness)
        else:
            thicknesses = [self.p.slice_thickness] * (N - 1)

        shape = tuple(geom.shape)
        self._fw_spec, self._bw_spec = [], []
        for i, t in enumerate(thicknesses):
            g.distance = t
            G = geometry.Geo(owner=None, pars=g)
            self._fw_spec.append(_extract_operator(G.propagator.fw, shape, "nearfield fw[%d]" % i))
            self._bw_spec.append(_extract_operator(G.propagator.bw, shape, "nearfield bw[%d]" % i))

        # --- far-field, sample to detector: take it from a live pod ---
        pod = None
        for name, view in self.di.views.items():
            if view.active:
                pod = list(view.pods.values())[0]
                break
        if pod is None:
            raise RuntimeError("no active diffraction views")

        self._ff_fw = _extract_operator(pod.fw, shape, "farfield fw")
        self._ff_bw = _extract_operator(pod.bw, shape, "farfield bw")

        # move to GPU
        def to_g(spec):
            out = {'kind': spec['kind']}
            for k in ('H', 'A', 'R'):
                if k in spec:
                    out[k] = torch.as_tensor(spec[k], dtype=self.cdt, device=self.dev)
            return out

        self._fw_g = [to_g(s) for s in self._fw_spec]
        self._bw_g = [to_g(s) for s in self._bw_spec]
        self._ff_fw_g = to_g(self._ff_fw)
        self._ff_bw_g = to_g(self._ff_bw)

    @staticmethod
    def _apply(spec, w):
        """Apply an extracted operator to a stack with trailing spatial dims."""
        k = spec['kind']
        if k == 'lsi':
            return torch.fft.ifft2(torch.fft.fft2(w) * spec['H'])
        if k == 'ilsi':
            return torch.fft.fft2(torch.fft.ifft2(w) * spec['H'])
        if k == 'chirp':
            return spec['A'] * torch.fft.fft2(spec['R'] * w)
        if k == 'ichirp':
            return spec['A'] * torch.fft.ifft2(spec['R'] * w)
        raise ValueError(k)

    def _collect_views(self):
        """Pull positions, intensities and masks onto the GPU once."""
        names = sorted([n for n, v in self.di.views.items() if v.active])
        if not names:
            raise RuntimeError("no active diffraction views")

        rows, cols, data, masks, layers = [], [], [], [], None

        for n in names:
            view = self.di.views[n]
            pods = list(view.pods.values())

            ov = pods[0].ob_view
            try:
                lo = np.asarray(ov.dlow, dtype=np.int64)
            except AttributeError:
                raise RuntimeError(
                    "This ptypy version's View has no .dlow; adapt _collect_views "
                    "to however object patches are indexed here.")
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))

            lay = [int(p.pr_view.dlayer) for p in pods]
            if layers is None:
                layers = lay
            elif lay != layers:
                raise NotImplementedError("inconsistent probe-mode layout across views")

            data.append(np.asarray(view.data, dtype=np.float64))
            m = pods[0].mask
            masks.append(np.ones_like(data[-1]) if m is None else np.asarray(m, dtype=np.float64))

        self.n_pos = len(names)
        self.view_names = names
        self.rows_g = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols_g = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.I_g = torch.as_tensor(np.stack(data), dtype=self.rdt, device=self.dev)
        self.mask_g = torch.as_tensor(np.stack(masks), dtype=self.rdt, device=self.dev)
        self.amp_g = torch.sqrt(self.I_g.clamp_min(0))

        if list(layers) != list(range(self.M)):
            logger.warning("probe layers %s do not match 0..%d; check mode ordering"
                           % (layers, self.M - 1))
        logger.info("collected %d scan positions, %d probe mode(s)" % (self.n_pos, self.M))

    def _report_memory(self):
        el = 8 if self.cdt == torch.complex64 else 16
        obn = self.ob_g.numel() * el
        dat = (self.I_g.numel() + self.mask_g.numel() + self.amp_g.numel()) * (4 if self.rdt == torch.float32 else 8)
        work = self.p.batch_size * self.M * self.fny * self.fnx * el * (2 * self.p.number_of_slices + 4)
        logger.info("GPU memory estimate: objects %.2f GB, data %.2f GB, workspace %.2f GB"
                    % (obn / 1e9, dat / 1e9, work / 1e9))

    # ------------------------------------------------------------------ #
    #  Patch gather / scatter                                             #
    # ------------------------------------------------------------------ #
    def _gather(self, arr2d, idx):
        r = self.rows_g[idx]                    # (B, fny)
        c = self.cols_g[idx]                    # (B, fnx)
        return arr2d[r[:, :, None], c[:, None, :]]

    def _scatter_add(self, dst, idx, vals):
        B = idx.numel()
        r = self.rows_g[idx][:, :, None].expand(B, self.fny, self.fnx)
        c = self.cols_g[idx][:, None, :].expand(B, self.fny, self.fnx)
        dst.index_put_((r.reshape(-1), c.reshape(-1)), vals.reshape(-1), accumulate=True)

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        N = self.p.number_of_slices
        B = int(self.p.batch_size)

        for _ in range(num):
            order = (torch.randperm(self.n_pos, device=self.dev)
                     if self.p.shuffle else torch.arange(self.n_pos, device=self.dev))

            e_fmag = e_phot = e_exit = 0.0
            for s in range(0, self.n_pos, B):
                idx = order[s:s + B]
                a, b, c = self._batch_update(idx)
                e_fmag += a
                e_phot += b
                e_exit += c

            n = float(self.n_pos)
            self._last_error = np.array([e_fmag / n, e_phot / n, e_exit / n])

            self.curiter += 1
            if self.curiter % max(1, int(self.p.sync_interval)) == 0:
                self._sync_to_ptypy()
            if self.curiter % 5 == 0:
                self._plot_iteration_progress()

        return {self.view_names[0]: self._last_error}

    def _batch_update(self, idx):
        N = self.p.number_of_slices
        B = idx.numel()
        reg = float(self.p.denom_reg)

        active = [self.curiter >= self.p.slice_start_iteration[i] for i in range(N)]

        # ---------------- forward ----------------
        ob_p = [self._gather(self.ob_g[i], idx) for i in range(N)]      # (B, ny, nx)
        psi = self.pr_g[None].expand(B, self.M, self.fny, self.fnx)     # incident at slice 0

        psis, exits = [], []
        for i in range(N):
            psis.append(psi)
            e = psi * ob_p[i][:, None] if active[i] else psi
            exits.append(e)
            if i < N - 1:
                psi = self._apply(self._fw_g[i], e)

        # ---------------- Fourier modulus projection ----------------
        F = self._apply(self._ff_fw_g, exits[-1])                       # (B, M, ny, nx)
        Imod = (F.real ** 2 + F.imag ** 2).sum(dim=1)                   # (B, ny, nx)
        amod = torch.sqrt(Imod.clamp_min(0))

        am = self.amp_g[idx]
        mk = self.mask_g[idx]

        eps = torch.finfo(self.rdt).eps
        scale = torch.where(mk > 0, am / amod.clamp_min(eps), torch.ones_like(amod))
        scale = torch.where(amod > eps, scale, torch.ones_like(scale))

        e_proj = self._apply(self._ff_bw_g, F * scale[:, None].to(self.cdt))

        d = amod - am
        err_fmag = float((mk * d * d).sum() / mk.sum().clamp_min(1))
        err_exit = float(((e_proj - exits[-1]).abs() ** 2).sum())
        err_phot = float((mk * (Imod - self.I_g[idx]) ** 2 / (self.I_g[idx] + 1.0)).sum()
                         / mk.sum().clamp_min(1))

        # ---------------- backward ----------------
        num_o = [torch.zeros_like(self.ob_g[0]) for _ in range(N)]
        den_o = [torch.zeros(self.ob_g[0].shape, dtype=self.rdt, device=self.dev) for _ in range(N)]

        num_P = torch.zeros_like(self.pr_g)                             # (M, ny, nx)
        den_P = torch.zeros((self.fny, self.fnx), dtype=self.rdt, device=self.dev)

        target = e_proj
        for i in range(N - 1, -1, -1):
            delta = target - exits[i]                                   # (B, M, ny, nx)

            if active[i]:
                # object gradient, summed over probe modes, accumulated over batch
                self._scatter_add(num_o[i], idx, (psis[i].conj() * delta).sum(dim=1))
                self._scatter_add(den_o[i], idx, (psis[i].abs() ** 2).sum(dim=1))

                o = ob_p[i][:, None]                                    # (B, 1, ny, nx)
                if i > 0:
                    # local correction to the incident wave, per view, used only
                    # to back-propagate; matches 3PIE's sequential structure
                    den = o.abs() ** 2
                    den = den + reg * den.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
                    dpsi = o.conj() * delta / den.to(self.cdt)
                else:
                    # slice 0: this IS the illumination update, so accumulate the
                    # raw numerator and denominator across the batch rather than
                    # normalising each view separately
                    num_P += (o.conj() * delta).sum(dim=0)
                    den_P += (ob_p[0].abs() ** 2).sum(dim=0)
                    dpsi = None
            else:
                dpsi = delta
                if i == 0:
                    num_P += delta.sum(dim=0)
                    den_P += torch.ones_like(den_P) * B

            if i > 0:
                target = self._apply(self._bw_g[i - 1], psis[i] + dpsi)

        # ---------------- apply updates ----------------
        for i in range(N):
            if not active[i]:
                continue
            dn = den_o[i] + reg * den_o[i].max().clamp_min(eps)
            self.ob_g[i] += float(self.p.object_step) * num_o[i] / dn.to(self.cdt)

        if self.curiter >= self.p.probe_update_start:
            dn = den_P + reg * den_P.max().clamp_min(eps)
            self.pr_g += float(self.p.probe_step) * num_P / dn[None].to(self.cdt)
            if self.pr_support is not None:
                self.pr_g *= self.pr_support

        return err_fmag, err_phot, err_exit

    # ------------------------------------------------------------------ #
    #  Sync and finalize                                                  #
    # ------------------------------------------------------------------ #
    def _sync_to_ptypy(self):
        """Copy GPU state into the ptypy containers for plotting and autosave."""
        N = self.p.number_of_slices
        for i in range(N):
            arr = self.ob_g[i].detach().cpu().numpy()
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._pr_storage.data[:] = self.pr_g.detach().cpu().numpy()

        self.ob.fill(self._object[0])
        for i in range(1, N):
            self.ob *= self._object[i]

    def engine_finalize(self):
        self._sync_to_ptypy()

        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness
        slices_info.slice_start_iteration = self.p.slice_start_iteration
        slices_info.batch_size = self.p.batch_size
        slices_info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                               for ob in self._object}

        header = {'description': 'multi-slice result details (torch engine).'}
        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info(f'Saving to {self.p.fslices}')
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()

    def _plot_iteration_progress(self):
        try:
            probe = (self.pr_g[0].abs() ** 2).cpu().numpy()
            first = self.ob_g[0].angle().cpu().numpy()
            final = self.ob_g[-1].angle().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, im, title, cmap in (
                    (axes[0], probe, f"Probe Intensity (Iter {self.curiter})", 'viridis'),
                    (axes[1], first, f"Slice 0 (First) Phase (Iter {self.curiter})", 'gray'),
                    (axes[2], final,
                        f"Slice {self.p.number_of_slices - 1} (Final) Phase (Iter {self.curiter})", 'gray')):
                h = ax.imshow(im, cmap=cmap)
                ax.set_title(title)
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(f"{self.plot_progress_dir}/iter_progress_{self.curiter:04d}.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not plot iteration progress: {e}")