"""
ISS (Inter-slice Single Scattering) engine for ptychography, CuPy backend.

A line-for-line translation of ptypy.custom.torch.iss_torch: the same forward
model, the same preconditioned Jacobi update, the same parameter tree. The
numpy-level plumbing (propagator extraction, slice containers, the progress
figures) is shared with the torch engines through base_torch, which imports
cleanly without torch installed; nothing here touches torch at runtime.

FORWARD MODEL
-------------
    \\Delta O_j   = O_j - 1                            (scattering perturbation)
    P_j = F^-1[ H_{z_j} F[P] ]                        (unscattered illumination)
    psi   = P + sum_j F^-1[ H_{-z_j} F[ \\Delta O_j P_j ] ] (exit wave)

with H_{z_j} = exp(2i pi (z_j/lam)(sqrt(1 - (lam f)^2) - 1)) and H_0 = 1, the
whole model referred to the entrance plane.

The slices enter as an affine SUM, not an ordered product. Because P_j
depends only on the probe, it is computed exactly once per mini-batch.

INVERSION & PRECONDITIONING
---------------------------
fw : Forward far-field propagator (sample exit plane -> detector plane)
bw : Backward far-field propagator (detector plane -> sample exit plane)
fac : The amplitude projection factor ( sqrt(I_data) / sqrt(I_model) - 1 )

Because the forward map is exactly affine, there is zero object-to-object
cross-curvature. The per-slice normal operator (A_j^dag A_j) is exactly
diagonal, reducing to elementwise multiplication by |phi_j|^2 (denoted as K_j
in the code).

Updating N slices simultaneously is a Jacobi iteration. The theoretically
stable step size is alpha < 2/N. The code internally divides by N, so
alpha = 1.0 is a safe, aggressive default.

CUDA GRAPHS
-----------
With cuda_graphs = True the whole per-position update -- gather, forward,
modulus projection, adjoint, scatter, object and probe step -- is captured
once as a CUDA graph and replayed for every following position, removing the
per-batch kernel-launch overhead that dominates at small frames.

What makes the update capturable as written:
  * the batch is fixed at 1, so every array shape is static;
  * the position index is read from a persistent device buffer the host
    refills before each launch, so the replay follows the scan;
  * nothing in the update path synchronises or copies to the host -- the
    error terms accumulate into a persistent device vector read once per
    iteration, outside the graph.
Each update variant (with and without the probe gradient) runs eagerly on its
first visit, which builds the cuFFT plans and warms CuPy's memory pool so the
capture itself allocates nothing from the driver; the variant is captured on
its second visit. The capture bakes in alpha, beta, the mask, the probe
support and the schedule constants, so change nothing mid-run. If a capture
fails the engine logs a warning and continues eagerly.
"""

import os

import numpy as np

from ptypy import io
from ptypy import utils as u
from ptypy.custom.torch.base_torch import (
    TorchMixin,
    extract_operator,
    plot_error_history,
    plot_progress,
    slice_containers,
)
from ptypy.engines import register, stochastic
from ptypy.utils import Param
from ptypy.utils.verbose import logger

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


# ===========================================================================
#  Pure Mathematical Core (Native CuPy)
# ===========================================================================
def first_born_step(
    O, P, I_dat, H_fwd, H_rev, ff_fw, ff_bw, mask=None, calc_probe_grad=True,
    phi=None,
):
    """
    Pure mathematical core for ISS Multislice Ptychography.

    Returns raw gradients and preconditioners for external scattering
    into a global object space (used by ptypy).

    O     : Object slices [N, B, Ny, Nx] or [N, Ny, Nx] (Complex)
    P     : Probe modes [M, Ny, Nx] (Complex)
    I_dat : Measured Diffraction Intensity [B, Ny, Nx] or [Ny, Nx] (Real)
    H_fwd : Forward free-space propagators H_{z_j}  [N, 1, 1, Ny, Nx] (Complex)
    H_rev : Reverse free-space propagators H_{-z_j} [N, 1, 1, Ny, Nx] (Complex)
    phi   : Optional precomputed unscattered illumination [N, 1, M, Ny, Nx];
            pass it once the probe is frozen so it is not recomputed per batch.
    """
    # Handle single-FOV inputs seamlessly by expanding a dummy batch dimension
    has_batch = O.ndim == 4
    if not has_batch:
        O = O[:, None]
        I_dat = I_dat[None]
        if mask is not None:
            mask = mask[None]

    # ---------------------------------------------------------
    # 0. PERTURBATION & ILLUMINATION
    # ---------------------------------------------------------
    g = (O - 1.0)[:, :, None]  # [N, B, 1, Ny, Nx]
    if phi is None:
        P_f = cp.fft.fft2(P)[None]  # [1, M, Ny, Nx]
        phi = cp.fft.ifft2(H_fwd * P_f)  # [N, 1, M, Ny, Nx]

    # ---------------------------------------------------------
    # 1. FORWARD PASS
    # ---------------------------------------------------------
    scattered_f = cp.fft.fft2(g * phi)
    psi = P[None] + cp.fft.ifft2((H_rev * scattered_f).sum(axis=0))

    # ---------------------------------------------------------
    # 2. DETECTOR PROJECTION
    # ---------------------------------------------------------
    F_psi = ff_fw(psi)

    # real**2 + imag**2 avoids the redundant sqrt inside abs()
    I_mod = (F_psi.real**2 + F_psi.imag**2).sum(axis=1)  # [B, Ny, Nx]

    # Single sqrt pass
    fac = cp.sqrt(I_dat / cp.maximum(I_mod, 1e-12)) - 1.0

    if mask is not None:
        fac = cp.where(mask > 0, fac, cp.zeros_like(fac))

    r_field = ff_bw(F_psi * fac[:, None])  # [B, M, Ny, Nx]

    # ---------------------------------------------------------
    # 3. ADJOINT PASS & GRADIENTS
    # ---------------------------------------------------------
    r_field_f = cp.fft.fft2(r_field)[None]
    r_j = cp.fft.ifft2(H_fwd * r_field_f)  # [N, B, M, Ny, Nx]

    grad_g = (cp.conj(phi) * r_j).sum(axis=2)  # [N, B, Ny, Nx]

    # Fast exact diagonal normal operator: sum(|phi|^2) without abs() roots
    K_O = (phi.real**2 + phi.imag**2).sum(axis=2)[:, 0]  # [N, Ny, Nx]

    # Probe gradient
    if calc_probe_grad:
        grad_P_raw = r_field + cp.fft.ifft2(
            (H_rev * cp.fft.fft2(cp.conj(g) * r_j)).sum(axis=0)
        )
        grad_P = grad_P_raw.sum(axis=0)  # [M, Ny, Nx]
        K_P = (O.real**2 + O.imag**2).sum(axis=(0, 1))  # [Ny, Nx]
    else:
        grad_P, K_P = None, None

    # Return gradients for global scattering
    if not has_batch:
        grad_g = grad_g[:, 0]
        r_field = r_field[0]
        I_mod = I_mod[0]

    return grad_g, K_O, grad_P, K_P, r_field, I_mod


# ---------------------------------------------------------------------------#
#  Operator / support helpers (cupy counterparts of base_torch's)            #
# ---------------------------------------------------------------------------#
def to_cupy(spec, dtype):
    """Move an extracted propagator spec onto the device."""
    out = {"kind": spec["kind"]}
    for k in ("H", "A", "R"):
        if k in spec:
            out[k] = cp.asarray(spec[k], dtype=dtype)
    return out


def apply_cupy(spec, w):
    """Apply an extracted operator to a stack with trailing spatial dims."""
    k = spec["kind"]
    if k == "lsi":
        return cp.fft.ifft2(cp.fft.fft2(w) * spec["H"])
    if k == "ilsi":
        return cp.fft.fft2(cp.fft.ifft2(w) * spec["H"])
    if k == "chirp":
        return spec["A"] * cp.fft.fft2(spec["R"] * w)
    if k == "ichirp":
        return spec["A"] * cp.fft.ifft2(spec["R"] * w)
    raise ValueError(k)


def circular_support(fraction, shape, dtype):
    """ptypy's circular probe support as a cupy mask, or None if disabled."""
    if fraction is None or not 0 < float(fraction) < 1:
        return None
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - ny / 2.0, xx - nx / 2.0)
    mask = r < float(fraction) * min(ny, nx) / 2.0
    return cp.asarray(mask.astype(np.float64), dtype=dtype)


class _HostView:
    """The thin tensor surface base_torch's plotting helpers touch
    (indexing, .abs(), .angle(), **, .cpu().numpy(), .shape), wrapped
    around a host numpy array so the figures need neither torch nor a
    second copy of the plotting code."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, k):
        return _HostView(self.a[k])

    def __pow__(self, p):
        return _HostView(self.a ** p)

    def abs(self):
        return _HostView(np.abs(self.a))

    def angle(self):
        return _HostView(np.angle(self.a))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


# ---------------------------------------------------------------------------#
#  CuPy infrastructure (cupy counterpart of base_torch.ISSEngineInfra)       #
# ---------------------------------------------------------------------------#
class CupyISSEngineInfra(TorchMixin):
    """ISS-family infrastructure on TorchMixin: object/probe arrays with
    warm starts, the branch propagators, view collection with flat-index
    gather/scatter, and the slice write-out -- all on CuPy. TorchMixin
    contributes only its torch-free members (plot/terminal-log plumbing
    and their Defaults); every torch-touching method is overridden here.
    """

    def _setup_cupy_infra(self):
        self._setup_cupy_device()

        N = self.p.number_of_slices
        self._obs = next(iter(self.ob.storages.values()))
        self._prs = next(iter(self.pr.storages.values()))
        self.M, self.fny, self.fnx = self._prs.data.shape

        # Initialize object slices and probe.
        #
        # The start is 1 + 1e-3 * uniform, symmetry-broken so the slices are
        # not degenerate. It is drawn from self.p.seed rather than from OS
        # entropy: a paired comparison whose two runs begin from different
        # objects has a twin-run difference that is not zero at iteration 0,
        # which is usually the quantity being measured. seed = 0 leaves the
        # draw free.
        self._object = slice_containers(self.ob, N)
        seed = int(self.p.seed)
        ob0 = np.ones_like(self._obs.data[0])
        rng = np.random.default_rng(seed if seed else None)
        self.ob_g = cp.stack(
            [
                cp.asarray(
                    ob0
                    + 1e-3
                    * (
                        (rng.random(ob0.shape) - 0.5)
                        + 1j * (rng.random(ob0.shape) - 0.5)
                    ),
                    dtype=self.cdt,
                )
                for _ in range(N)
            ]
        )
        logger.info(
            "object start: 1 + 1e-3 noise, seed %s"
            % (seed if seed else "free (not reproducible)")
        )
        self.pr_g = cp.asarray(
            np.ascontiguousarray(self._prs.data), dtype=self.cdt
        )

        # Warm start: overwrite the fresh object/probe init with a previous
        # reconstruction so a run can continue where another stopped.
        ws_ob = getattr(self.p, "warm_start_h5", None)
        if ws_ob:
            import h5py as _h5
            with _h5.File(ws_ob, "r") as _f:
                objs = _f["content/objects"]
                keys = sorted(objs.keys(), key=lambda s: int(s.split("_")[-1]))
                if len(keys) != N:
                    raise ValueError(
                        "warm_start_h5 has %d slices, engine wants %d"
                        % (len(keys), N))
                for i, k in enumerate(keys):
                    arr = objs[k]["Sscan00G00"]["data"][()][0]
                    self.ob_g[i] = cp.asarray(arr, dtype=self.cdt)
            logger.info("warm-start: loaded %d object slices from %s"
                        % (N, ws_ob))
        ws_pr = getattr(self.p, "warm_start_ptyr", None)
        if ws_pr:
            import h5py as _h5
            with _h5.File(ws_pr, "r") as _f:
                g = _f["content/probe"]
                pk = list(g.keys())[0]
                Pw = g[pk]["data"][()]
            if Pw.shape != tuple(self.pr_g.shape):
                raise ValueError("warm_start_ptyr probe shape %s != %s"
                                 % (Pw.shape, tuple(self.pr_g.shape)))
            self.pr_g = cp.asarray(Pw, dtype=self.cdt)
            logger.info("warm-start: loaded probe %s from %s"
                        % (str(Pw.shape), ws_pr))

        self._order_rng = None
        if seed:
            self._order_rng = np.random.default_rng(seed + 1)
            logger.info(f"seed = {seed}: the position order is reproducible")
        self.ony, self.onx = self.ob_g.shape[-2:]

        # Precompute Free-Space Propagators (H_z and H_{-z})
        geom = next(iter(self.ptycho.model.scans.values())).geometries[0]
        V, W = u.grids(geom.shape, geom.lam / geom.shape[0] / geom.resolution,
                       "fft")
        a2 = V**2 + W**2
        z_depths = [
            sum(self.p.slice_thickness[:i])
            if isinstance(self.p.slice_thickness, (list, tuple))
            else i * self.p.slice_thickness
            for i in range(1, N)
        ]

        H = np.empty((max(N - 1, 1), *geom.shape), dtype=np.complex128)
        for i, zi in enumerate(z_depths):
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            H[i] = np.exp(2j * np.pi * (zi / geom.lam) * kz) * (a2 <= 1.0)

        self.H = cp.asarray(H[: N - 1], dtype=self.cdt)
        self.Hb = cp.conj(self.H) if N > 1 else self.H

        # Build fully broadcastable H arrays for the pure math function
        self.H_fwd = cp.ones((N, 1, 1, *geom.shape), dtype=self.cdt)
        self.H_rev = cp.ones((N, 1, 1, *geom.shape), dtype=self.cdt)
        if N > 1:
            self.H_fwd[1:, 0, 0] = self.H
            self.H_rev[1:, 0, 0] = self.Hb

        self._build_farfield()
        self._collect_views()

        self._setup_probe_support()
        self._setup_plot_state()

    def _setup_cupy_device(self):
        """Resolve device and dtypes from p.device / p.precision."""
        if cp is None:
            raise ImportError("ISSCupy requires CuPy")
        want = str(self.p.device)
        if not want.startswith("cuda"):
            raise ValueError(
                "ISSCupy runs on CUDA only (device=%r); use the torch engine "
                "for a CPU run" % want)
        devno = int(want.split(":")[1]) if ":" in want else 0
        cp.cuda.Device(devno).use()
        self.dev = devno
        dbl = str(self.p.precision).lower().startswith("d")
        self.cdt = cp.complex128 if dbl else cp.complex64
        self.rdt = cp.float64 if dbl else cp.float32
        self._eps = float(np.finfo(np.float64 if dbl else np.float32).eps)

    def _setup_probe_support(self):
        """Circular probe support whenever p.probe_support is set."""
        self.pr_support = circular_support(
            getattr(self.p, "probe_support", None),
            (self.fny, self.fnx),
            self.rdt,
        )
        if self.pr_support is not None:
            logger.info(
                f"probe support: circular, radius fraction "
                f"{float(self.p.probe_support):.3f}"
            )

    def _build_farfield(self):
        """Extracts fw and bw far-field propagators from ptypy."""
        pod = next(iter(next(iter(self.di.views.values())).pods.values()))
        shape = (self.fny, self.fnx)
        for tag, op in (("fw", pod.fw), ("bw", pod.bw)):
            spec = extract_operator(op, shape, f"farfield {tag}")
            setattr(self, f"_ff_{tag}", to_cupy(spec, self.cdt))

    @staticmethod
    def _apply(spec, w):
        return apply_cupy(spec, w)

    def _collect_views(self):
        names = sorted(n for n, v in self.di.views.items() if v.active)
        rows, cols, amps, masks = [], [], [], []
        trivial = True
        for n in names:
            pod = next(iter(self.di.views[n].pods.values()))
            lo = np.asarray(pod.ob_view.dlow, dtype=np.int64)
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))
            I = np.asarray(self.di.views[n].data, dtype=np.float64)
            amps.append(np.sqrt(np.maximum(I, 0.0)))
            m = (
                np.ones_like(I, dtype=bool)
                if pod.mask is None
                else np.asarray(pod.mask) > 0
            )
            trivial = trivial and bool(m.all())
            masks.append(m)

        self.n_pos = len(names)
        rows_np, cols_np = np.stack(rows), np.stack(cols)
        self.rows = cp.asarray(rows_np, dtype=cp.int64)
        self.cols = cp.asarray(cols_np, dtype=cp.int64)
        self.amp_dat = cp.asarray(np.stack(amps), dtype=self.rdt)
        self.maskb = (
            None if trivial else cp.asarray(np.stack(masks), dtype=cp.bool_)
        )
        self.view_names = names

        # Illuminated region.  Any statistic taken over a slice is meaningless
        # outside it: most of the object array is never touched and stays at
        # exactly 1, which would drag every rms towards zero by an arbitrary
        # factor.  Read from outside the engine, by diagnostics. Built on the
        # host: overlapping patches need accumulate-scatter, which numpy's
        # add.at gives without a device kernel, chunked to bound the index
        # arrays it materialises.
        cov = np.zeros((self.ony, self.onx))
        for s in range(0, self.n_pos, 128):
            np.add.at(
                cov,
                (rows_np[s:s + 128, :, None], cols_np[s:s + 128, None, :]),
                1.0,
            )
        self.cov = cp.asarray(cov > 0)

    def _flat(self, idx):
        return (
            self.rows[idx][:, :, None] * self.onx + self.cols[idx][:, None, :]
        ).reshape(-1)

    def _gather(self, arr, flat, B):
        return arr.reshape(-1)[flat].reshape(B, self.fny, self.fnx)

    def _scatter(self, dst, flat, vals):
        # Accumulating scatter. The engine only ever scatters one view at a
        # time (batch 1), whose patch indices are unique, so the plain
        # advanced-indexed add is exact -- and, unlike cupyx.scatter_add, it
        # works for complex values and inside a CUDA graph capture.
        dst.reshape(-1)[flat] += vals.ravel()

    def _model_object(self, O_patches):
        """What the forward core sees as the slice transmissions.

        The chord engine stores the transmissions O_j themselves, so this is
        the identity and the core's g = O - 1 is the chord perturbation. The
        tangent subclass stores the scattering potential s_j instead and
        returns 1 + i s_j, so the same core computes g = i s_j: the
        linearised transmission.
        """
        return O_patches

    def _grad_to_param(self, grad):
        """Chain rule from the model perturbation g to the stored parameter.

        The chord engine's parameter is O with g = O - 1, so dg/dO = 1 and
        this is the identity. The tangent subclass stores s with g = i s and
        overrides this with the conjugate factor, -i.
        """
        return grad

    def _display_object(self):
        """Slice transmissions for plotting; the tangent subclass, which
        stores the potential, overrides this with the exponential."""
        return self.ob_g

    def _plot_progress(self):
        self._stream.synchronize()
        plot_progress(
            _HostView(cp.asnumpy(self.pr_g)),
            _HostView(cp.asnumpy(self._display_object())),
            self.curiter,
            self.plot_progress_dir,
            psize=getattr(self.p, "psize", (1e-6, 1e-6)),
            engine_name=type(self).__name__,
            cmaps=self.PLOT_CMAPS,
            zoom_center=self.p.get("plot_zoom_center", None),
            phase_clim=self.p.get("plot_phase_clim", None),
            scalebar=self.p.get("plot_scalebar", None),
            phase_zero_top=bool(self.p.get("plot_phase_zero_top", False)),
        )
        plot_error_history(
            self._err_history,
            self.plot_progress_dir,
            engine_name=type(self).__name__,
            channels=self.p.get("plot_error_channels", None),
        )

    def _write_slices(self):
        self._stream.synchronize()
        for i in range(self.p.number_of_slices):
            arr = cp.asnumpy(self.ob_g[i])
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._prs.data[:] = cp.asnumpy(self.pr_g)

        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        info = Param(
            number_of_slices=self.p.number_of_slices,
            slice_thickness=self.p.slice_thickness,
            alpha=self.p.get("alpha", None),
            objects={
                ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                for ob in self._object
            },
        )

        # Tell ptypy to ignore its own custom class types when saving to HDF5
        h5opt = io.h5options["UNSUPPORTED"]
        io.h5options["UNSUPPORTED"] = "ignore"

        logger.info(f"saving slices to {self.p.fslices}")
        io.h5write(
            self.p.fslices,
            header={"description": "ISS multislice"},
            content=info,
        )

        io.h5options["UNSUPPORTED"] = h5opt


# ---------------------------------------------------------------------------#
#  ISS Engine wrapper                                                        #
# ---------------------------------------------------------------------------#
@register()
class ISSCupy(CupyISSEngineInfra, stochastic.EPIE):
    """
    ISS multislice ptychography on the GPU, CuPy backend.

    Defaults:

    [name]
    default = ISSCupy
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

    [warm_start_h5]
    default = None
    type = str
    help = Path to a previous fslices h5 whose object slices seed this run
    doc = When set, the object slices (Cobj_o_i) replace the 1+noise init after
          engine_initialize. Must have number_of_slices slices. Used to continue
          a prior reconstruction without re-running its iterations.

    [warm_start_ptyr]
    default = None
    type = str
    help = Path to a previous .ptyr whose probe seeds this run
    doc = When set, the probe replaces the illumination init after
          engine_initialize. Shape must match the probe storage.

    [alpha]
    default = 1.0
    type = float
    help = Object update step size
    doc = Divided by number_of_slices internally. Values near 1 are safe and
          aggressive; 0.2 is roughly five times slower than it needs to be.

    [beta]
    default = 1.0
    type = float
    help = Probe update step size
    doc = Divided by number_of_slices internally, as for alpha.

    [probe_update_interval]
    default = 1
    type = int
    help = Batches between probe updates
    doc = The probe gradient is a quarter of the transforms in a batch and the
          probe converges long before the object, so 4 costs almost nothing in
          quality and saves close to 20 percent of the runtime.

    [cuda_graphs]
    default = False
    type = bool
    help = Capture the per-position update as a CUDA graph and replay it
    doc = The first visit of each update variant (with / without the probe
          gradient) runs eagerly to build the cuFFT plans and warm the memory
          pool; the variant is captured on its second visit and replayed for
          every following position, removing the per-batch kernel-launch
          overhead. The capture bakes in alpha, beta, the probe schedule, the
          mask and every array shape, so change nothing mid-run. If a capture
          fails the engine logs a warning and continues eagerly.

    [device]
    default = cuda
    type = str
    help = CUDA device: 'cuda' or 'cuda:0'
    doc = CuPy is CUDA-only; use the torch engine for a CPU run.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

    [shuffle]
    default = False
    type = bool
    help = Reshuffle the position order every iteration
    doc = Set False to walk views in sorted order, as the CPU engine does with
          its shuffle disabled.

    [seed]
    default = 123
    type = int
    help = Seed for the position order and the object start; 0 leaves them free
    doc = Set the same non-zero seed for both runs of a paired comparison, and
          for both engines, so that nothing but the thing under test differs.

    [plot_interval]
    default = 10
    type = int
    help = Iterations between diagnostic PNGs, 0 disables
    doc = Same name and meaning as in the torch engines, so a paired
          comparison plots on the same cadence from one setting. On by
          default: with no explicit progress directory the figures and the
          terminal log land in iteration_progress_<engine>_<stamp>/ in the
          working directory, so every run leaves a browsable record.

    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.article = dict(
            title='An inter-slice Born series for parallel-in-depth '
                  'ptychography of thick samples',
            author='B. Shirley et al.',
            journal='in preparation',
            volume=0,
            year=2026,
            page=0,
            doi='',
            comment='The inter-slice Born series reconstruction algorithm',
        )
        ptycho_parent.citations.add_article(**self.article)

    def engine_initialize(self):
        super().engine_initialize()
        self._stamp_plot_dir()
        self._setup_cupy_infra()
        N = self.p.number_of_slices

        # Persistent Gradient / Preconditioner Accumulators
        self._acc_grad = cp.zeros((N, self.ony, self.onx), dtype=self.cdt)
        self._acc_K = cp.zeros((N, self.ony, self.onx), dtype=self.rdt)

        self._err = cp.zeros(3, dtype=cp.float64)
        self._nbatch = 0
        self._pint = max(1, int(self.p.probe_update_interval))

        # All engine work runs on one non-default stream: CUDA graph capture
        # is illegal on the legacy default stream, and keeping the eager path
        # on the same stream means the two modes are ordered identically.
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._use_graphs = bool(self.p.cuda_graphs)
        self._graphs = {}        # do_probe -> captured cupy.cuda.Graph
        self._graph_warm = set() # variants that have run eagerly once
        self._idx_buf = cp.zeros((1,), dtype=cp.int64)
        if self._use_graphs:
            logger.info("cuda_graphs on: each update variant runs eagerly "
                        "once, is captured on its second visit, and replays "
                        "thereafter")

    def _step(self, idx):
        """One scan position: probe scheduling on the host, then the update
        eagerly or as a CUDA-graph replay."""
        self._nbatch += 1
        do_probe = (
            self.curiter >= self.p.probe_update_start
            and self._nbatch % self._pint == 0
        )

        if not self._use_graphs:
            self._batch_update(idx, do_probe)
            return

        # The graph reads the position index from this persistent buffer;
        # refilling it is the only per-position host work after capture.
        cp.copyto(self._idx_buf, idx)
        graph = self._graphs.get(do_probe)
        if graph is None:
            if do_probe not in self._graph_warm:
                # eager warm-up: builds the cuFFT plans and pools the
                # workspaces this variant will ask for during capture
                self._graph_warm.add(do_probe)
                self._batch_update(self._idx_buf, do_probe)
                return
            try:
                self._stream.begin_capture()
                self._batch_update(self._idx_buf, do_probe)
                graph = self._stream.end_capture()
            except Exception as e:  # noqa: BLE001 - driver/capture errors
                try:
                    self._stream.end_capture()
                except Exception:  # noqa: BLE001
                    pass
                logger.warning(
                    "CUDA graph capture failed (%s); continuing eagerly" % e)
                self._use_graphs = False
                self._batch_update(self._idx_buf, do_probe)
                return
            self._graphs[do_probe] = graph
            logger.info("captured CUDA graph for the %s variant"
                        % ("probe-update" if do_probe else "object-only"))
        graph.launch()

    def _batch_update(self, idx, do_probe):
        """
        Executes a single position update: extracts local object patches,
        computes gradients via the pure mathematical core, tracks convergence
        errors, and applies the preconditioned Jacobi update. Device work
        only -- no host transfers or synchronisation -- so the whole body is
        CUDA-graph capturable.
        """
        B = int(idx.size)
        flat = self._flat(idx)  # Flattened indices for memory mapping

        # =====================================================================
        # 1. EXTRACT DATA & PREPARE INPUTS
        # =====================================================================
        O_patches = cp.stack(
            [
                self._gather(self.ob_g[j], flat, B)
                for j in range(self.p.number_of_slices)
            ]
        )
        O_patches = self._model_object(O_patches)
        amp_dat = self.amp_dat[idx]
        I_dat = amp_dat**2
        mask = self.maskb[idx].astype(self.rdt) if self.maskb is not None \
            else None

        # Explicit wrapper functions for the far-field propagators
        def forward_prop(x):
            return self._apply(self._ff_fw, x)

        def backward_prop(x):
            return self._apply(self._ff_bw, x)

        # =====================================================================
        # 2. RUN PURE MATHEMATICAL CORE
        # =====================================================================
        grad_g, K_O, grad_P, K_P, r_field, I_mod = first_born_step(
            O=O_patches,
            P=self.pr_g,
            I_dat=I_dat,
            H_fwd=self.H_fwd,
            H_rev=self.H_rev,
            ff_fw=forward_prop,
            ff_bw=backward_prop,
            mask=mask,
            calc_probe_grad=do_probe,
        )

        # =====================================================================
        # 3. CALCULATE CONVERGENCE METRICS (Errors)
        # =====================================================================
        amp_mod = cp.sqrt(cp.maximum(I_mod, 0))

        # Unified mask logic to avoid repeated branching
        m_weight = 1.0 if mask is None else mask
        m_sum = (
            float(B * self.fny * self.fnx)
            if mask is None
            else cp.maximum(mask.sum(), 1)
        )

        e0 = (m_weight * (amp_mod - amp_dat) ** 2).sum() / m_sum
        e1 = (m_weight * (I_mod - I_dat) ** 2 / (I_dat + 1.0)).sum() / m_sum
        self._err += cp.stack(
            (e0, e1, cp.abs(r_field).sum())
        ).astype(cp.float64)

        # =====================================================================
        # 4. SCATTER GRADIENTS & APPLY PRECONDITIONED UPDATE
        # =====================================================================
        self._acc_grad.fill(0)
        self._acc_K.fill(0)

        # Map the local batch gradients back into the global object array
        for j in range(self.p.number_of_slices):
            self._scatter(self._acc_grad[j], flat, grad_g[j])
            self._scatter(
                self._acc_K[j], flat,
                cp.broadcast_to(K_O[j], (B, self.fny, self.fnx)),
            )

        N = self.p.number_of_slices
        eps = self._eps

        # --- Update Probe (if scheduled) ---
        if do_probe:
            dn_p = K_P.max()
            self.pr_g += (float(self.p.beta) / N) * (
                grad_P / cp.maximum(dn_p, eps)
            )
            if self.pr_support is not None:
                self.pr_g *= self.pr_support

        # --- Update Object Slices ---
        dn_o = self._acc_K.max(axis=(-2, -1), keepdims=True)
        upd = self._grad_to_param(self._acc_grad / cp.maximum(dn_o, eps))
        self.ob_g += (float(self.p.alpha) / N) * upd

    def engine_iterate(self, num=1):
        B = 1
        # Order anything still pending on other streams (initialisation
        # transfers run on the default stream) before the engine stream
        # touches the same arrays.
        cp.cuda.Device().synchronize()
        for _ in range(num):

            with self._stream:
                if self.p.shuffle:
                    perm = (
                        self._order_rng.permutation(self.n_pos)
                        if self._order_rng is not None
                        else np.random.permutation(self.n_pos)
                    )
                    order = cp.asarray(perm, dtype=cp.int64)
                else:
                    order = cp.arange(self.n_pos, dtype=cp.int64)

                self._err.fill(0)
                for s in range(0, int(order.size), B):
                    self._step(order[s : s + B])

            self._stream.synchronize()
            err = cp.asnumpy(self._err / float(self.n_pos))
            self._err_history.append(err.copy())

            # Increment first, then plot: the frame is named for the number of
            # iterations behind it, and the last one is saved, matching the
            # torch engine so a paired comparison reads frames from the same
            # iteration.
            self.curiter += 1
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        return {"all": err}

    def engine_finalize(self):
        self._write_slices()
        return super().engine_finalize()


@register()
class ISSTangentCupy(ISSCupy):
    """
    ISS with the tangent (linearised-transmission) perturbation.

    The chord engine expands the cascade in g_j = O_j - 1, the full slice
    transmission. This variant uses the weak-phase convention of the prior
    ISS treatments instead: O_j ~ 1 + i s_j. The reconstructed unknown
    is the scattering potential s_j itself -- phase in its real part,
    absorption in its imaginary part -- and the factor i belongs to the
    forward model, which is handed g_j = i s_j. Updates are additive in s_j
    with the same preconditioner and step sizes (the chain rule through
    g = i s rotates the gradient by -i), and the slices are mapped back to
    transmissions O_j = exp(i s_j) for plotting and saving. Everything else
    -- batching, propagators, probe update, seeding, graph capture -- is
    inherited unchanged, so a paired chord/tangent comparison differs in the
    perturbation alone.

    Defaults:

    [name]
    default = ISSTangentCupy
    type = str
    help = Engine identifier

    [potential_real]
    default = False
    type = bool
    help = Constrain the potential s to be real, i.e. a pure phase object
    doc = The strict weak-phase-object convention: g = i s with s real
          forbids absorption. False reconstructs a complex s, whose imaginary
          part carries the absorption.
    """

    def engine_initialize(self):
        super().engine_initialize()
        # the working object becomes the potential s = -i log O; the start
        # 1 + 1e-3 noise maps to s ~ 1e-3 noise, the same symmetry breaking
        self.ob_g = -1j * cp.log(self.ob_g)
        if bool(self.p.potential_real):
            self.ob_g = self.ob_g.real.astype(self.cdt)
            logger.info("tangent potential constrained to be real (pure phase)")

    def _model_object(self, O_patches):
        # the core subtracts 1, so 1 + i s makes the perturbation g = i s
        return 1.0 + 1j * O_patches

    def _grad_to_param(self, grad):
        # dL/ds* = conj(dg/ds) dL/dg* = -i dL/dg*; for real s keep the
        # real part, which is the Wirtinger descent direction on the reals
        g = -1j * grad
        return g.real.astype(self.cdt) if bool(self.p.potential_real) else g

    def _display_object(self):
        return cp.exp(1j * self.ob_g)

    def engine_finalize(self):
        # hand transmissions, not potentials, to the containers and file
        self.ob_g = cp.exp(1j * self.ob_g)
        return super().engine_finalize()
