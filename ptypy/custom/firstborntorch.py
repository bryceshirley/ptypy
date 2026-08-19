# -*- coding: utf-8 -*-
"""
First Born multislice engine for ptychography (GPU, PyTorch).

Same forward model and same update as the CPU engine:

    phi_j = F^-1[ H_j F[P] ]                        illumination at slice j
    psi   = c * ( P + sum_j F^-1[ Hbar_j F[ s_j phi_j ] ] )

with H_0 = 1 and, optionally, O_j = Obar_j (1 + s_j), c = prod_j Obar_j.

At batch_size = 1 this reproduces the CPU engine: same denominators, same step
convention, same modulus projection, one object update per scan position.
Larger batches accumulate the gradient over several positions before updating,
which is faster but is a mini-batch method rather than sequential ePIE.

The whole dataset, both object slices and the probe stay resident on the GPU.
Because phi_j carries no object and the slice sum has no sequential dependency
in z, the entire slice stack goes through one batched FFT in both the forward
and the adjoint pass.
"""
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from ptypy.engines import stochastic
from ptypy.engines import register
from ptypy.utils import Param
from ptypy.utils.verbose import logger
from ptypy import utils as u
from ptypy import io

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------#
#  Numerical extraction of ptypy's far-field propagator                       #
# ---------------------------------------------------------------------------#
def _op_call(op, arr):
    last = None
    for dt in (np.complex64, np.complex128):
        try:
            return np.asarray(op(arr.astype(dt)), dtype=np.complex128)
        except Exception as e:      # noqa: BLE001
            last = e
    raise RuntimeError("could not call propagator: %s" % last)


def _apply_numpy(kind, d, w):
    if kind == 'lsi':
        return np.fft.ifft2(np.fft.fft2(w) * d['H'])
    if kind == 'ilsi':
        return np.fft.fft2(np.fft.ifft2(w) * d['H'])
    if kind == 'chirp':
        return d['A'] * np.fft.fft2(d['R'] * w)
    return d['A'] * np.fft.ifft2(d['R'] * w)


def _extract_operator(op, shape, name, rtol=1e-5):
    """
    Recover a ptypy propagator exactly from two probes, then verify it on random
    data, so that a convention mismatch is a startup error rather than a wrong
    reconstruction. Candidates: ifft2(fft2(w) H), fft2(ifft2(w) H),
    A fft2(R w), A ifft2(R w). numpy convention, which torch matches with the
    default norm='backward'.
    """
    ny, nx = shape
    d00 = np.zeros(shape, dtype=np.complex128)
    d00[0, 0] = 1.0
    out00 = _op_call(op, d00)
    out1 = _op_call(op, np.ones(shape, dtype=np.complex128))

    cands = [('lsi', {'H': np.fft.fft2(out00)}),
             ('ilsi', {'H': float(ny * nx) * np.fft.ifft2(out00)})]
    if np.abs(out00).min() > 0:
        cands += [('chirp', {'A': out00, 'R': np.fft.ifft2(out1 / out00)}),
                  ('ichirp', {'A': out00, 'R': np.fft.fft2(out1 / out00)})]

    rng = np.random.default_rng(0)
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    ref = _op_call(op, w)
    scale = max(np.abs(ref).max(), 1e-30)

    errs = {k: np.abs(_apply_numpy(k, d, w) - ref).max() / scale for k, d in cands}
    kind = min(errs, key=errs.get)
    if errs[kind] > rtol:
        raise RuntimeError("%s: propagator extraction failed, errors %s"
                           % (name, {k: '%.2e' % v for k, v in errs.items()}))

    spec = {'kind': kind}
    spec.update(dict(cands)[kind])
    logger.info("%s extracted as '%s' (error %.2e)" % (name, kind, errs[kind]))
    return spec


@register()
class FirstBornTorch(stochastic.EPIE):
    """
    First Born multislice ptychography on the GPU.

    Defaults:

    [name]
    default = FirstBornTorch
    type = str
    help = Engine identifier

    [number_of_slices]
    default = 2
    type = int
    help = Number of object slices

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Slice separation in meters
    doc = A single float, or a list of length number_of_slices - 1.

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the reconstructed slices

    [alpha]
    default = 0.2
    type = float
    help = Object update step size
    doc = Divided by number_of_slices internally. Per slice the normal operator
          A_j^dag A_j = |phi_j|^2 is exactly diagonal, so a full step is correct
          for one slice alone; updating N slices simultaneously is a Jacobi
          iteration and needs step < 2/N.

    [beta]
    default = 1.0
    type = float
    help = Probe update step size
    doc = Divided by number_of_slices internally, as for alpha.

    [object_denom]
    default = max
    type = str
    help = Preconditioner: 'max' or 'local'
    doc = 'max' uses the scalar maximum of the accumulated denominator (ePIE
          convention, and what reproduces the CPU engine at batch_size = 1).
          'local' uses the per-pixel accumulated denominator plus denom_reg
          times its maximum, which is the exact diagonal of A^dag A.

    [denom_reg]
    default = 0.01
    type = float
    help = Floor on the 'local' denominator, as a fraction of its maximum

    [mean_field]
    default = False
    type = bool
    help = Expand each slice about its complex mean instead of about vacuum

    [mean_field_relax]
    default = 0.5
    type = float
    help = Relaxation on the Obar update; Obar <- (1-r) Obar + r <O>

    [device]
    default = cuda
    type = str
    help = Torch device: 'cuda', 'cuda:0' or 'cpu'

    [batch_size]
    default = 32
    type = int
    help = Scan positions per object update
    doc = 1 reproduces the CPU engine. Larger values accumulate the gradient
          over several positions before updating, which is faster but is a
          mini-batch method rather than sequential ePIE. Peak workspace scales
          as number_of_slices * batch_size * probe_modes.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)
    doc = Use 'double' to compare against the CPU engine numerically.
    """

    def __init__(self, ptycho_parent, pars=None):
        super(FirstBornTorch, self).__init__(ptycho_parent, pars)
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
    #  Setup                                                              #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()
        if torch is None:
            raise ImportError("FirstBornTorch requires PyTorch")

        N = self.p.number_of_slices

        dev = str(self.p.device)
        if dev.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            dev = 'cpu'
        self.dev = torch.device(dev)
        dbl = str(self.p.precision).lower().startswith('d')
        self.cdt = torch.complex128 if dbl else torch.complex64
        self.rdt = torch.float64 if dbl else torch.float32

        if len(self.ob.storages) != 1 or len(self.pr.storages) != 1:
            raise NotImplementedError("single object and probe storage only")
        self._obs = list(self.ob.storages.values())[0]
        self._prs = list(self.pr.storages.values())[0]
        if self._obs.data.shape[0] != 1:
            raise NotImplementedError("single object mode only")
        self.M, self.fny, self.fnx = self._prs.data.shape

        # ptypy containers kept only so the result can be saved
        self._object = [self.ob.copy(self.ob.ID + "_o_%d" % i) for i in range(N)]

        # object slices, vacuum plus a little noise to break symmetry
        rng = np.random.default_rng()
        ob0 = np.ones_like(self._obs.data[0])
        self.ob_g = torch.stack([
            torch.as_tensor(ob0 + 1e-3 * ((rng.random(ob0.shape) - 0.5)
                                          + 1j * (rng.random(ob0.shape) - 0.5)),
                            dtype=self.cdt, device=self.dev) for _ in range(N)])
        self.pr_g = torch.as_tensor(np.ascontiguousarray(self._prs.data),
                                    dtype=self.cdt, device=self.dev)

        # slice propagators for j = 1 .. N-1; H_0 = 1 identically and is never
        # stored or applied, which removes 1/N of the transforms
        geom = list(self.ptycho.model.scans.values())[0].geometries[0]
        shape, lam = geom.shape, geom.lam
        V, W = u.grids(shape, lam / shape / geom.resolution, 'fft')
        a2 = V**2 + W**2

        th = self.p.slice_thickness
        z = [sum(th[:i]) for i in range(1, N)] if type(th) in (list, tuple) \
            else [i * th for i in range(1, N)]

        H = np.empty((max(N - 1, 1), *shape), dtype=np.complex128)
        for i, zi in enumerate(z):
            # clamp the evanescent band so |H| == 1 exactly; referring the model
            # to the entrance plane by factoring out H^N depends on it
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            Hi = np.exp(2j * np.pi * (zi / lam) * kz)
            Hi[a2 > 1.0] = 0.0
            H[i] = Hi
        self.H = torch.as_tensor(H[:N - 1], dtype=self.cdt, device=self.dev)
        self.Hb = self.H.conj()

        self._build_farfield()
        self._collect_views()

        self._Obar = torch.ones(N, dtype=self.cdt, device=self.dev)
        self._mf_c = torch.ones((), dtype=self.cdt, device=self.dev)

        # phi depends only on the probe, so it is reused until the probe moves
        self._phi_cache = None

    

    def _build_farfield(self):
        pod = next(list(v.pods.values())[0] for v in self.di.views.values() if v.active)
        shape = (self.fny, self.fnx)
        for tag, spec in (('fw', _extract_operator(pod.fw, shape, "farfield fw")),
                          ('bw', _extract_operator(pod.bw, shape, "farfield bw"))):
            g = {'kind': spec['kind']}
            for k in ('H', 'A', 'R'):
                if k in spec:
                    g[k] = torch.as_tensor(spec[k], dtype=self.cdt, device=self.dev)
            setattr(self, '_ff_' + tag, g)

    @staticmethod
    def _apply(spec, w):
        k = spec['kind']
        if k == 'lsi':
            return torch.fft.ifft2(torch.fft.fft2(w) * spec['H'])
        if k == 'ilsi':
            return torch.fft.fft2(torch.fft.ifft2(w) * spec['H'])
        if k == 'chirp':
            return spec['A'] * torch.fft.fft2(spec['R'] * w)
        return spec['A'] * torch.fft.ifft2(spec['R'] * w)

    def _collect_views(self):
        """Upload positions, intensities and masks once."""
        names = sorted(n for n, v in self.di.views.items() if v.active)
        rows, cols, data, masks = [], [], [], []
        for n in names:
            pods = list(self.di.views[n].pods.values())
            lo = np.asarray(pods[0].ob_view.dlow, dtype=np.int64)
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))
            data.append(np.asarray(self.di.views[n].data, dtype=np.float64))
            m = pods[0].mask
            masks.append(np.ones_like(data[-1]) if m is None else np.asarray(m, dtype=np.float64))

        self.n_pos = len(names)
        self.rows = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.I = torch.as_tensor(np.stack(data), dtype=self.rdt, device=self.dev)
        self.mask = torch.as_tensor(np.stack(masks), dtype=self.rdt, device=self.dev)
        self.amp = torch.sqrt(self.I.clamp_min(0))

        # illuminated region, so <O_j> is not diluted by the untouched border
        cov = torch.zeros(self.ob_g[0].shape, dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            self._scatter(cov, idx, torch.ones((idx.numel(), self.fny, self.fnx),
                                               dtype=self.rdt, device=self.dev))
        self.cov = cov > 0
        frac = float(self.cov.to(self.rdt).mean())
        logger.info("%d positions, %d probe mode(s), illuminated fraction %.4f"
                    % (self.n_pos, self.M, frac))
        if frac < 1e-3:
            raise RuntimeError("coverage mask empty; check View.dlow indexing")

    # ------------------------------------------------------------------ #
    #  Gather / scatter                                                   #
    # ------------------------------------------------------------------ #
    def _gather(self, arr, idx):
        return arr[self.rows[idx][:, :, None], self.cols[idx][:, None, :]]

    def _scatter(self, dst, idx, vals):
        B = idx.numel()
        r = self.rows[idx][:, :, None].expand(B, self.fny, self.fnx)
        c = self.cols[idx][:, None, :].expand(B, self.fny, self.fnx)
        dst.index_put_((r.reshape(-1), c.reshape(-1)), vals.reshape(-1), accumulate=True)

    # ------------------------------------------------------------------ #
    #  Mean-field background                                              #
    # ------------------------------------------------------------------ #
    def _update_mean_field(self):
        """Refresh Obar and c once per iteration, before the sweep, so the
        gradients are exact for the whole sweep."""
        N = self.p.number_of_slices
        if not self.p.mean_field:
            self._Obar = torch.ones(N, dtype=self.cdt, device=self.dev)
            self._mf_c = torch.ones((), dtype=self.cdt, device=self.dev)
            return

        r = float(self.p.mean_field_relax)
        new = torch.stack([self.ob_g[j][self.cov].mean() for j in range(N)])
        new = torch.where(new.abs() < 1e-6, self._Obar, new)
        self._Obar = (1.0 - r) * self._Obar + r * new
        self._mf_c = torch.prod(self._Obar)

    # ------------------------------------------------------------------ #
    #  Forward, projection, adjoint                                       #
    # ------------------------------------------------------------------ #
    def _phi(self):
        """Unscattered illumination at every slice, (N, M, ny, nx).

        phi_0 = P exactly, since H_0 = 1. Contains no object, so it is neither
        position- nor batch-dependent and is cached until the probe is updated.
        """
        if self._phi_cache is None:
            N = self.p.number_of_slices
            ph = torch.empty((N, self.M, self.fny, self.fnx),
                             dtype=self.cdt, device=self.dev)
            ph[0] = self.pr_g
            if N > 1:
                ph[1:] = torch.fft.ifft2(self.H[:, None] * torch.fft.fft2(self.pr_g))
            self._phi_cache = (ph, (ph.abs() ** 2).sum(dim=1))
        return self._phi_cache

    def _batch_update(self, idx):
        N = self.p.number_of_slices
        B = idx.numel()
        eps = torch.finfo(self.rdt).eps
        use_max = str(self.p.object_denom).lower().startswith('m')
        reg = float(self.p.denom_reg)

        phi, den_phi = self._phi()

        O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(N)])  # (N,B,ny,nx)
        s = O / self._Obar.view(N, 1, 1, 1) - 1.0

        # --- forward ---
        scat = s[0][:, None] * phi[0][None]                                   # H_0 = 1
        if N > 1:
            Fs = torch.fft.fft2(s[1:, :, None] * phi[1:, None])               # (N-1,B,M,ny,nx)
            scat = scat + torch.fft.ifft2((self.Hb[:, None, None] * Fs).sum(0))
        psi = self._mf_c * (self.pr_g[None] + scat)                           # (B,M,ny,nx)

        # --- modulus projection, intensities summed over probe modes ---
        F = self._apply(self._ff_fw, psi)
        Imod = (F.real ** 2 + F.imag ** 2).sum(dim=1)
        amod = torch.sqrt(Imod.clamp_min(0))
        am, mk = self.amp[idx], self.mask[idx]
        scale = torch.where(mk > 0, am / amod.clamp_min(1e-12), torch.ones_like(amod))
        psi_p = self._apply(self._ff_bw, F * scale[:, None].to(self.cdt))

        msum = mk.sum().clamp_min(1)
        err = np.array([float((mk * (amod - am) ** 2).sum() / msum),
                        float((mk * (Imod - self.I[idx]) ** 2 / (self.I[idx] + 1.0)).sum() / msum),
                        float((psi_p - psi).abs().sum())])

        # --- adjoint ---
        d = psi_p - psi
        dj = torch.empty((N, B, self.M, self.fny, self.fnx), dtype=self.cdt, device=self.dev)
        dj[0] = d                                                             # H_0 = 1
        if N > 1:
            dj[1:] = torch.fft.ifft2(self.H[:, None, None] * torch.fft.fft2(d))

        # chain must be 5-D: dj is (N,B,M,ny,nx) and torch right-aligns, so a
        # 4-D chain would put the slice axis onto the batch axis -- silently at
        # batch_size = 1, and as a shape error otherwise
        chain = (self._mf_c / self._Obar).conj().view(N, 1, 1, 1, 1)
        num = (chain * phi[:, None].conj() * dj).sum(dim=2)                   # (N,B,ny,nx)
        den = (self._mf_c / self._Obar).abs().view(N, 1, 1) ** 2 * den_phi    # (N,ny,nx)

        step_o = float(self.p.alpha) / N
        step_p = float(self.p.beta) / N

        # --- probe ---
        if self.curiter >= self.p.probe_update_start:
            g = s[0][:, None].conj() * dj[0]
            if N > 1:
                Fg = torch.fft.fft2(s[1:, :, None].conj() * dj[1:])
                g = g + torch.fft.ifft2((self.Hb[:, None, None] * Fg).sum(0))
            num_p = (self._mf_c.conj() * (d + g)).sum(dim=0)                  # (M,ny,nx)
            den_p = self._mf_c.abs() ** 2 * (O.abs() ** 2).sum(0).sum(0)      # (ny,nx)
            dn = den_p.max() if use_max else den_p + reg * den_p.max()
            self.pr_g += step_p * num_p / dn.clamp_min(eps).to(self.cdt)
            self._phi_cache = None

        # --- object ---
        for j in range(N):
            acc_n = torch.zeros_like(self.ob_g[j])
            acc_d = torch.zeros(self.ob_g[j].shape, dtype=self.rdt, device=self.dev)
            self._scatter(acc_n, idx, num[j])
            self._scatter(acc_d, idx, den[j].expand(B, self.fny, self.fnx))
            dn = acc_d.max() if use_max else acc_d + reg * acc_d.max()
            self.ob_g[j] += step_o * acc_n / dn.clamp_min(eps).to(self.cdt)

        return err

    # ------------------------------------------------------------------ #
    #  Iteration                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        B = int(self.p.batch_size)

        for _ in range(num):
            self._update_mean_field()
            order = torch.randperm(self.n_pos, device=self.dev)

            err = np.zeros(3)
            for s in range(0, self.n_pos, B):
                err += self._batch_update(order[s:s + B])
            err /= float(self.n_pos)

            logger.info("iter %d: |Obar| = %s, arg(Obar) = %s"
                        % (self.curiter,
                           np.array2string(self._Obar.abs().cpu().numpy(), precision=4),
                           np.array2string(self._Obar.angle().cpu().numpy(), precision=4)))
            self._plot_iteration_progress()
            self.curiter += 1

        return {'all': err}

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

            ob = self._Obar.cpu().numpy()
            plt.suptitle(f"|c| = {abs(complex(self._mf_c.cpu())):.4f}   "
                         f"|Obar| = {np.array2string(np.abs(ob), precision=3)}   "
                         f"arg(Obar) = {np.array2string(np.angle(ob), precision=3)}")
            plt.tight_layout()
            plt.savefig(f"{self.plot_progress_dir}/iter_progress_{self.curiter:04d}.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not plot iteration progress: {e}")

    def engine_finalize(self):
        for i in range(self.p.number_of_slices):
            arr = self.ob_g[i].cpu().numpy()
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._prs.data[:] = self.pr_g.cpu().numpy()

        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        info = Param()
        info.number_of_slices = self.p.number_of_slices
        info.slice_thickness = self.p.slice_thickness
        info.mean_field = self.p.mean_field
        info.object_denom = self.p.object_denom
        info.batch_size = self.p.batch_size
        info.Obar = self._Obar.cpu().numpy()
        info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                        for ob in self._object}

        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info("saving slices to %s" % self.p.fslices)
        io.h5write(self.p.fslices,
                   header={'description': 'First Born multislice reconstruction'},
                   content=info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()