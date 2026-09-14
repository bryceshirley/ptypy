"""
ISS multislice engine for ptychography.

FORWARD MODEL
-------------
    g_j   = O_j - 1                                   (scattering perturbation)
    phi_j = F^-1[ H_{z_j} F[P] ]                      (unscattered illumination)
    psi   = P + sum_j F^-1[ H_{-z_j} F[ g_j phi_j ] ] (exit wave)

with H_{z_j} = exp(2i pi (z_j/lam)(sqrt(1 - (lam f)^2) - 1)) and H_0 = 1, the whole
model referred to the entrance plane.

The slices enter as an affine SUM, not an ordered product. Because phi_j
depends only on the probe, it is computed exactly once per mini-batch.

INVERSION & PRECONDITIONING
---------------------------
fw : Forward far-field propagator (sample exit plane -> detector plane)
bw : Backward far-field propagator (detector plane -> sample exit plane)
fac : The amplitude projection factor ( sqrt(I_data) / sqrt(I_model) - 1 )

Because the forward map is exactly affine, there is zero object-to-object
cross-curvature. The per-slice normal operator (A_j^dag A_j) is exactly diagonal,
reducing to elementwise multiplication by |phi_j|^2 (denoted as K_j in the code).

Updating N slices simultaneously is a Jacobi iteration. The theoretically stable
step size is alpha < 2/N. The code internally divides by N, so alpha = 1.0 is a
safe, aggressive default.

THE OVERLAP PROPERTY (WHY THIS WORKS)
-------------------------------------
Any higher-order multiple scattering terms neglected by this model involve products
like g_k * P[g_j * phi_j]. Because the perturbation g = 0 in free space (vacuum),
these terms vanish completely wherever the illuminated objects on different slices
do not overlap in the beam path.

Therefore, for sparse samples or layers separated by transverse shifts, this model
is not an approximation: it is mathematically EXACT to multislice, but computes
in O(1) depth rather than O(N).
"""

import os
from datetime import datetime

import numpy as np

from ptypy import io
from ptypy import utils as u
from ptypy.custom import torch_helpers
from ptypy.custom.engine_hooks import EngineHooks
from ptypy.engines import register, stochastic
from ptypy.utils import Param
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:
    torch = None


# ===========================================================================
#  Pure Mathematical Core (Native PyTorch)
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
        O = O.unsqueeze(1)
        I_dat = I_dat.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    # ---------------------------------------------------------
    # 0. PERTURBATION & ILLUMINATION
    # ---------------------------------------------------------
    g = (O - 1.0).unsqueeze(2)  # [N, B, 1, Ny, Nx]
    if phi is None:
        P_f = torch.fft.fft2(P).unsqueeze(0)  # [1, M, Ny, Nx]
        phi = torch.fft.ifft2(H_fwd * P_f)  # [N, 1, M, Ny, Nx]

    # ---------------------------------------------------------
    # 1. FORWARD PASS
    # ---------------------------------------------------------
    scattered_f = torch.fft.fft2(g * phi)
    psi = P.unsqueeze(0) + torch.fft.ifft2(torch.sum(H_rev * scattered_f, dim=0))

    # ---------------------------------------------------------
    # 2. DETECTOR PROJECTION
    # ---------------------------------------------------------
    F_psi = ff_fw(psi)

    # Use .square() to avoid redundant sqrt inside .abs()
    I_mod = torch.sum(F_psi.real.square() + F_psi.imag.square(), dim=1)  # [B, Ny, Nx]

    # Single sqrt pass
    fac = torch.sqrt(I_dat / I_mod.clamp_min(1e-12)) - 1.0

    if mask is not None:
        fac = torch.where(mask > 0, fac, torch.zeros_like(fac))

    r_field = ff_bw(F_psi * fac.unsqueeze(1))  # [B, M, Ny, Nx]

    # ---------------------------------------------------------
    # 3. ADJOINT PASS & GRADIENTS
    # ---------------------------------------------------------
    r_field_f = torch.fft.fft2(r_field).unsqueeze(0)
    r_j = torch.fft.ifft2(H_fwd * r_field_f)  # [N, B, M, Ny, Nx]

    grad_g = torch.sum(phi.conj() * r_j, dim=2)  # [N, B, Ny, Nx]

    # Fast exact diagonal normal operator: sum(|phi|^2) without .abs() roots
    K_O = torch.sum(phi.real.square() + phi.imag.square(), dim=2).squeeze(
        1
    )  # [N, Ny, Nx]

    # Probe gradient
    if calc_probe_grad:
        grad_P_raw = r_field + torch.fft.ifft2(
            torch.sum(H_rev * torch.fft.fft2(g.conj() * r_j), dim=0)
        )
        grad_P = torch.sum(grad_P_raw, dim=0)  # [M, Ny, Nx]
        K_P = torch.sum(O.real.square() + O.imag.square(), dim=(0, 1))  # [Ny, Nx]
    else:
        grad_P, K_P = None, None

    # Return gradients for global scattering
    if not has_batch:
        grad_g = grad_g.squeeze(1)
        r_field = r_field.squeeze(0)
        I_mod = I_mod.squeeze(0)

    return grad_g, K_O, grad_P, K_P, r_field, I_mod


def first_born_step_lowmem(
    O, P, I_dat, H_fwd, H_rev, ff_fw, ff_bw, mask=None, calc_probe_grad=True,
    phi=None,
):
    """
    first_born_step evaluated slice by slice: same arithmetic, same returns,
    but the (N, B, M, Ny, Nx) workspaces of the parallel version never exist.
    Peak memory is (B, M, Ny, Nx) plus the (N, B, Ny, Nx) gradient output, so
    the batch can grow by roughly a factor N on the same GPU, paid for with a
    Python loop over slices.

    phi : optional precomputed unscattered illumination [N, M, Ny, Nx]; pass
    it once the probe is frozen so it is not recomputed per batch.
    """
    has_batch = O.ndim == 4
    if not has_batch:
        O = O.unsqueeze(1)
        I_dat = I_dat.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    N, B = O.shape[0], O.shape[1]
    M, ny, nx = P.shape

    # Unscattered illumination at every slice: no batch dimension, so this
    # stays small at any N.
    if phi is None:
        P_f = torch.fft.fft2(P)
        phi = torch.fft.ifft2(H_fwd[:, 0] * P_f.unsqueeze(0))  # [N, M, Ny, Nx]

    # ---- forward: accumulate the scattered field in the frequency domain ---
    acc = torch.zeros((B, M, ny, nx), dtype=O.dtype, device=O.device)
    for j in range(N):
        acc += H_rev[j, 0] * torch.fft.fft2((O[j] - 1.0).unsqueeze(1) * phi[j])
    psi = P.unsqueeze(0) + torch.fft.ifft2(acc)
    del acc

    # ---- detector projection: identical to the parallel core ---------------
    F_psi = ff_fw(psi)
    I_mod = torch.sum(F_psi.real.square() + F_psi.imag.square(), dim=1)
    fac = torch.sqrt(I_dat / I_mod.clamp_min(1e-12)) - 1.0
    if mask is not None:
        fac = torch.where(mask > 0, fac, torch.zeros_like(fac))
    r_field = ff_bw(F_psi * fac.unsqueeze(1))  # [B, M, Ny, Nx]

    # ---- adjoint, slice by slice -------------------------------------------
    r_field_f = torch.fft.fft2(r_field)
    grad_g = torch.empty((N, B, ny, nx), dtype=O.dtype, device=O.device)
    K_O = torch.empty((N, ny, nx), dtype=I_dat.dtype, device=O.device)
    acc_P = (torch.zeros((B, M, ny, nx), dtype=O.dtype, device=O.device)
             if calc_probe_grad else None)
    for j in range(N):
        r_j = torch.fft.ifft2(H_fwd[j, 0] * r_field_f)
        grad_g[j] = (phi[j].conj().unsqueeze(0) * r_j).sum(dim=1)
        K_O[j] = (phi[j].real.square() + phi[j].imag.square()).sum(dim=0)
        if calc_probe_grad:
            acc_P += H_rev[j, 0] * torch.fft.fft2(
                (O[j] - 1.0).conj().unsqueeze(1) * r_j)
        del r_j

    if calc_probe_grad:
        grad_P = (r_field + torch.fft.ifft2(acc_P)).sum(dim=0)
        K_P = torch.sum(O.real.square() + O.imag.square(), dim=(0, 1))
    else:
        grad_P, K_P = None, None

    if not has_batch:
        grad_g = grad_g.squeeze(1)
        r_field = r_field.squeeze(0)
        I_mod = I_mod.squeeze(0)

    return grad_g, K_O, grad_P, K_P, r_field, I_mod


# ---------------------------------------------------------------------------#
#  ISS Engine wrapper                                                 #
# ---------------------------------------------------------------------------#
@register()
class ISSTorch(stochastic.EPIE, EngineHooks):
    """
    ISS multislice ptychography on the GPU.

    Defaults:

    [name]
    default = ISSTorch
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
    doc = Divided by number_of_slices internally. With object_denom = 'local'
          the preconditioner is the exact diagonal of A^dag A, so the per-slice
          operator is the identity and the Jacobi bound is alpha < 2. Values
          near 1 are safe and aggressive; 0.2 is roughly five times slower than
          it needs to be.

    [beta]
    default = 1.0
    type = float
    help = Probe update step size
    doc = Divided by number_of_slices internally, as for alpha.

    [born_mode]
    default = low_mem
    type = str
    help = How the Born sum is evaluated: 'parallel' or 'lowmem'
    doc = 'parallel' pushes the whole slice stack through batched FFTs and
          holds (N, batch, modes, ny, nx) workspaces -- fastest while it
          fits. 'lowmem' loops over slices and keeps only (batch, modes,
          ny, nx) workspaces, so the batch can grow by roughly a factor N
          on the same GPU. Same arithmetic, same result.

    [momentum]
    default = 0.0
    type = float
    help = Heavy-ball momentum on the object update
    doc = The increment is scaled by (1 - momentum) so the effective step
          stays alpha/N and momentum can be changed without retuning alpha;
          0 disables. Accelerates exactly the slow, low-spatial-frequency
          directions that carry the depth information.

    [probe_update_interval]
    default = 1
    type = int
    help = Batches between probe updates
    doc = The probe gradient is a quarter of the transforms in a batch and the
          probe converges long before the object, so 4 costs almost nothing in
          quality and saves close to 20 percent of the runtime.

    [probe_converged_rtol]
    default = 0.0
    type = float
    help = Freeze the probe once its per-iteration relative change drops below this
    doc = Checked once per iteration as |P_new - P_old| / |P_old| over the whole
          probe stack. Below the tolerance the probe update stops for good and
          the unscattered illumination phi, which depends only on the probe, is
          computed once and cached, saving both the probe-gradient transforms
          and the per-batch phi FFTs for the rest of the run. 0 (the default)
          disables the check and the probe is updated to the end: auto-freezing
          could strand a probe that had only partially converged (e.g. a tight
          high-NA probe still building structure), so it is opt-in.

    [object_denom]
    default = 'local'
    type = str
    help = Preconditioner: 'local' or 'max'
    doc = 'local' uses the per-pixel accumulated denominator plus denom_reg
          times its maximum, which is the exact diagonal of A^dag A. 'max' uses
          the scalar maximum (ePIE convention); at batch_size > 1 that divides
          everything by the most-overlapped pixel and starves sparsely covered
          regions, so it is only for reproducing the CPU engine at batch_size 1.

    [denom_reg]
    default = 0.01
    type = float
    help = Floor on the 'local' denominator, as a fraction of its maximum

    [apply_probe_support]
    default = False
    type = bool
    help = Apply the circular probe support from p.probe_support
    doc = This engine runs its own iteration loop and so never reaches
          BaseEngine.support_constraint; without this flag p.probe_support is
          accepted and ignored. Same name and meaning as in ThreePIETorch.
          Leave False when comparing against an engine that has no support.

    [device]
    default = cuda
    type = str
    help = Torch device: 'cuda', 'cuda:0' or 'cpu'

    [batch_size]
    default = 1
    type = int
    help = Scan positions per object update
    doc = Peak workspace scales as batch_size * probe_modes. Larger batches
          use the FFT hardware better but give fewer updates per epoch;
          32 to 128 is usually the sweet spot.

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
    default = 3
    type = int
    help = Iterations between diagnostic PNGs, 0 disables
    doc = Same name and meaning as in ThreePIETorch, so a paired comparison
          plots on the same cadence from one setting.
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.ptycho.citations.add_article(
            title="Ptychographic transmission microscopy in three dimensions using a multi-slice approach",
            author="A. M. Maiden et al.",
            journal="J. Opt. Soc. Am. A",
            volume=29,
            year=2012,
            page=1606,
            doi="10.1364/JOSAA.29.001606",
            comment="The 3PIE reconstruction algorithm",
        )
        # Local time on purpose: this names a run directory the user reads.
        # The class name is in it so a folder of runs identifies its solver
        # without the log (type(self) so subclasses label themselves).
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        self.plot_progress_dir = f"iteration_progress_{type(self).__name__}_{stamp}"

    def engine_initialize(self):
        super().engine_initialize()
        if torch is None:
            raise ImportError("ISSTorch requires PyTorch")

        N = self.p.number_of_slices
        want = str(self.p.device)
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            want = "cpu"
        self.dev = torch.device(want)
        dbl = str(self.p.precision).lower().startswith("d")
        self.cdt = torch.complex128 if dbl else torch.complex64
        self.rdt = torch.float64 if dbl else torch.float32

        self._obs = next(iter(self.ob.storages.values()))
        self._prs = next(iter(self.pr.storages.values()))
        self.M, self.fny, self.fnx = self._prs.data.shape

        # Initialize object slices and probe.
        #
        # The start is 1 + 1e-3 * uniform, symmetry-broken so the slices are not
        # degenerate.  It is drawn from self.p.seed rather than from OS entropy:
        # a paired comparison whose two runs begin from different objects has a
        # twin-run difference that is not zero at iteration 0, which is usually
        # the quantity being measured.  seed = 0 leaves the draw free, which is
        # what the original engine did.
        self._object = torch_helpers.slice_containers(self.ob, N)
        seed = int(self.p.seed)
        ob0 = np.ones_like(self._obs.data[0])
        rng = np.random.default_rng(seed if seed else None)
        self.ob_g = torch.stack(
            [
                torch.as_tensor(
                    ob0
                    + 1e-3
                    * (
                        (rng.random(ob0.shape) - 0.5)
                        + 1j * (rng.random(ob0.shape) - 0.5)
                    ),
                    dtype=self.cdt,
                    device=self.dev,
                )
                for _ in range(N)
            ]
        )
        logger.info(
            "object start: 1 + 1e-3 noise, seed %s"
            % (seed if seed else "free (not reproducible)")
        )
        self.pr_g = torch.as_tensor(
            np.ascontiguousarray(self._prs.data), dtype=self.cdt, device=self.dev
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
                    self.ob_g[i] = torch.as_tensor(
                        arr, dtype=self.cdt, device=self.dev)
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
            self.pr_g = torch.as_tensor(Pw, dtype=self.cdt, device=self.dev)
            logger.info("warm-start: loaded probe %s from %s"
                        % (str(Pw.shape), ws_pr))

        self._order_gen = None
        if seed:
            self._order_gen = torch.Generator(device=self.dev)
            self._order_gen.manual_seed(seed + 1)
            logger.info(f"seed = {seed}: the position order is reproducible")
        self.ony, self.onx = self.ob_g.shape[-2:]

        # Precompute Free-Space Propagators (H_z and H_{-z})
        geom = next(iter(self.ptycho.model.scans.values())).geometries[0]
        V, W = u.grids(geom.shape, geom.lam / geom.shape[0] / geom.resolution, "fft")
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

        self.H = torch.as_tensor(H[: N - 1], dtype=self.cdt, device=self.dev)
        self.Hb = self.H.conj().contiguous() if N > 1 else self.H

        # Build fully broadcastable H arrays for pure math function
        self.H_fwd = torch.ones((N, 1, 1, *geom.shape), dtype=self.cdt, device=self.dev)
        self.H_rev = torch.ones((N, 1, 1, *geom.shape), dtype=self.cdt, device=self.dev)
        if N > 1:
            self.H_fwd[1:, 0, 0] = self.H
            self.H_rev[1:, 0, 0] = self.Hb

        self._build_farfield()
        self._collect_views()

        # Probe-freeze machinery: once the probe's per-iteration relative
        # change falls below probe_converged_rtol its updates stop and phi,
        # the unscattered illumination (which depends only on the probe), is
        # computed once into _phi_cache and reused every batch.
        self._phi_cache = None
        self._probe_frozen = False
        self._pr_prev = None
        self._probe_tol = float(self.p.probe_converged_rtol)

        self.pr_support = None
        if bool(self.p.apply_probe_support):
            self.pr_support = torch_helpers.circular_support(
                getattr(self.p, "probe_support", None),
                (self.fny, self.fnx),
                self.rdt,
                self.dev,
            )
            if self.pr_support is None:
                logger.warning(
                    "apply_probe_support is set but probe_support = %s is not "
                    "between 0 and 1, so no support is applied"
                    % (getattr(self.p, "probe_support", None),)
                )
            else:
                logger.info(
                    f"probe support: circular, radius fraction "
                    f"{float(self.p.probe_support):.3f}"
                )

        # Persistent Gradient / Preconditioner Accumulators
        self._acc_grad = torch.zeros(
            (N, self.ony, self.onx), dtype=self.cdt, device=self.dev
        )
        self._acc_K = torch.zeros(
            (N, self.ony, self.onx), dtype=self.rdt, device=self.dev
        )

        self._err = torch.zeros(3, dtype=torch.float64, device=self.dev)
        self._err_history = []
        self._z0 = torch.zeros((), dtype=self.rdt, device=self.dev)
        self._nbatch = 0
        self._pint = max(1, int(self.p.probe_update_interval))
        self._plot_int = max(0, int(self.p.plot_interval))
        self._use_max = str(self.p.object_denom).lower().startswith("m")

        self._core = (first_born_step_lowmem
                      if str(self.p.born_mode).lower().startswith("l")
                      else first_born_step)
        self._mom = None
        if float(self.p.momentum) > 0:
            self._mom = torch.zeros_like(self.ob_g)
        logger.info("born_mode = %s, momentum = %.3g"
                    % (self._core.__name__.replace("first_born_step", "")
                       .lstrip("_") or "parallel", float(self.p.momentum)))

        self._step = self._batch_update

        self._pad = False
        # Here, not in __init__: a caller may point plot_progress_dir somewhere
        # else between construction and the run, and making the default at
        # construction time left an empty timestamped directory behind.
        # Only when plotting is on: with plot_interval = 0 nothing is ever
        # written there and the empty timestamped directory is pure litter.
        if self._plot_int > 0:
            os.makedirs(self.plot_progress_dir, exist_ok=True)
        self.post_initialize()

    def _build_farfield(self):
        """Extracts fw (forward) and bw (backward) far-field propagators from ptypy."""
        pod = next(iter(next(iter(self.di.views.values())).pods.values()))
        shape = (self.fny, self.fnx)
        for tag, op in (("fw", pod.fw), ("bw", pod.bw)):
            spec = torch_helpers.extract_operator(op, shape, f"farfield {tag}")
            setattr(
                self, f"_ff_{tag}", torch_helpers.to_torch(spec, self.cdt, self.dev)
            )

    @staticmethod
    def _apply(spec, w):
        return torch_helpers.apply_torch(spec, w)

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
        self.rows = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.amp_dat = torch.as_tensor(np.stack(amps), dtype=self.rdt, device=self.dev)
        self.maskb = (
            None
            if trivial
            else torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.dev)
        )
        self.view_names = names

        # Illuminated region.  Any statistic taken over a slice is meaningless
        # outside it: most of the object array is never touched and stays at
        # exactly 1, which would drag every rms towards zero by an arbitrary
        # factor.  Read from outside the engine, by diagnostics.
        cov = torch.zeros((self.ony, self.onx), dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            B = idx.numel()
            self._scatter(
                cov,
                self._flat(idx),
                torch.ones((B, self.fny, self.fnx), dtype=self.rdt, device=self.dev),
            )
        self.cov = cov > 0

    def _flat(self, idx):
        return (
            self.rows[idx][:, :, None] * self.onx + self.cols[idx][:, None, :]
        ).reshape(-1)

    def _gather(self, arr, flat, B):
        return arr.reshape(-1)[flat].view(B, self.fny, self.fnx)

    def _scatter(self, dst, flat, vals):
        dst.view(-1).index_put_((flat,), vals.reshape(-1), accumulate=True)

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

    def _batch_update(self, idx):
        """
        Executes a single mini-batch update: extracts local object patches,
        computes gradients via the pure mathematical core, tracks convergence
        errors, and applies the preconditioned Jacobi update.
        """
        B = idx.numel()
        flat = self._flat(idx)  # Flattened indices for memory mapping

        # =====================================================================
        # 1. EXTRACT DATA & PREPARE INPUTS
        # =====================================================================
        O_patches = torch.stack(
            [
                self._gather(self.ob_g[j], flat, B)
                for j in range(self.p.number_of_slices)
            ]
        )
        O_patches = self._model_object(O_patches)
        I_dat = self.amp_dat[idx].square()
        mask = self.maskb[idx].to(self.rdt) if self.maskb is not None else None

        self._nbatch += 1
        do_probe = (
            not self._probe_frozen
            and self.curiter >= self.p.probe_update_start
            and self._nbatch % self._pint == 0
        )

        # Explicit wrapper functions for the far-field propagators
        def forward_prop(x):
            return self._apply(self._ff_fw, x)

        def backward_prop(x):
            return self._apply(self._ff_bw, x)

        # =====================================================================
        # 2. RUN PURE MATHEMATICAL CORE
        # =====================================================================
        grad_g, K_O, grad_P, K_P, r_field, I_mod = self._core(
            O=O_patches,
            P=self.pr_g,
            I_dat=I_dat,
            H_fwd=self.H_fwd,
            H_rev=self.H_rev,
            ff_fw=forward_prop,
            ff_bw=backward_prop,
            mask=mask,
            calc_probe_grad=do_probe,
            phi=self._phi_cache,
        )

        # =====================================================================
        # 3. CALCULATE CONVERGENCE METRICS (Errors)
        # =====================================================================
        amp_mod = I_mod.clamp_min(0).sqrt()
        amp_dat = self.amp_dat[idx]

        # Unified mask logic to avoid repeated branching
        m_weight = 1.0 if mask is None else mask
        m_sum = (
            float(B * self.fny * self.fnx) if mask is None else mask.sum().clamp_min(1)
        )

        e0 = (m_weight * (amp_mod - amp_dat).square()).sum() / m_sum
        e1 = (m_weight * (I_mod - I_dat).square() / (I_dat + 1.0)).sum() / m_sum
        self._err.add_(torch.stack([e0, e1, r_field.abs().sum()]).to(torch.float64))

        # =====================================================================
        # 4. SCATTER GRADIENTS & APPLY PRECONDITIONED UPDATE
        # =====================================================================
        self._acc_grad.zero_()
        self._acc_K.zero_()

        # Map the local batch gradients back into the global object array
        for j in range(self.p.number_of_slices):
            self._scatter(self._acc_grad[j], flat, grad_g[j])
            self._scatter(self._acc_K[j], flat, K_O[j].expand(B, self.fny, self.fnx))

        N = self.p.number_of_slices
        eps = torch.finfo(self.rdt).eps
        reg = float(self.p.denom_reg)

        # object_denom is honoured as given, at every batch size. ePIE at B=1
        # conventionally divides by the MAX incident intensity, because the
        # local denominator amplifies noise in dark regions; that is what
        # object_denom='max' selects. Forcing it at B=1 regardless made
        # object_denom='local' silently mean 'max' there, so a comparison of
        # preconditioners at B=1 compared nothing. denom_reg floors the local
        # denominator at a fraction of its own maximum, which is the intended
        # guard against the dark-region blow-up.
        use_max = self._use_max

        # --- Update Probe (if scheduled) ---
        if do_probe:
            peak_P = K_P.amax()
            dn_p = peak_P if use_max else K_P + (peak_P * reg)
            self.pr_g.add_(grad_P / dn_p.clamp_min(eps), alpha=float(self.p.beta) / N)
            if self.pr_support is not None:
                self.pr_g.mul_(self.pr_support)

        # --- Update Object Slices ---
        peak_K = self._acc_K.amax(dim=(-2, -1), keepdim=True)
        dn_o = peak_K if use_max else self._acc_K + (peak_K * reg)
        upd = self._grad_to_param(self._acc_grad / dn_o.clamp_min(eps))
        if self._mom is not None:
            # heavy ball, scaled by (1 - mu) so the effective step stays
            # alpha / N whatever the momentum
            mu = float(self.p.momentum)
            self._mom.mul_(mu).add_(upd, alpha=float(self.p.alpha) / N * (1.0 - mu))
            self.ob_g += self._mom
        else:
            self.ob_g.add_(upd, alpha=float(self.p.alpha) / N)

    def _freeze_probe(self, rel):
        """Stop probe updates for good and pin the unscattered illumination."""
        self._probe_frozen = True
        self._pr_prev = None
        P_f = torch.fft.fft2(self.pr_g).unsqueeze(0)
        if self._core is first_born_step_lowmem:
            self._phi_cache = torch.fft.ifft2(self.H_fwd[:, 0] * P_f)
        else:
            self._phi_cache = torch.fft.ifft2(self.H_fwd * P_f)
        msg = (
            f"probe converged at iteration {self.curiter}: relative change "
            f"{rel:.3e} < {self._probe_tol:.3e}. Probe updates stopped and "
            f"its illumination field is now cached."
        )
        logger.info(msg)
        print(msg)

    def _check_probe_converged(self):
        """Once per iteration: freeze the probe when its update stalls."""
        if (
            self._probe_frozen
            or self._probe_tol <= 0
            or self.curiter <= self.p.probe_update_start
        ):
            return
        if self._pr_prev is not None:
            base = torch.linalg.vector_norm(self._pr_prev)
            diff = torch.linalg.vector_norm(self.pr_g - self._pr_prev)
            rel = float(diff / base.clamp_min(torch.finfo(self.rdt).eps))
            msg = (
                f"iteration {self.curiter}: probe relative change "
                f"{rel:.3e} (freezes below {self._probe_tol:.3e})"
            )
            logger.info(msg)
            print(msg)
            if rel < self._probe_tol:
                self._freeze_probe(rel)
                return
        self._pr_prev = self.pr_g.clone()

    def engine_iterate(self, num=1):
        B = int(self.p.batch_size)
        for _ in range(num):
            self.pre_iteration()

            if self.p.shuffle:
                order = (
                    torch.randperm(self.n_pos, device=self.dev)
                    if self._order_gen is None
                    else torch.randperm(
                        self.n_pos, generator=self._order_gen, device=self.dev
                    )
                )
            else:
                order = torch.arange(self.n_pos, device=self.dev)

            self._err.zero_()
            for s in range(0, order.numel(), B):
                self._step(order[s : s + B])
            err = (self._err / float(self.n_pos)).cpu().numpy()
            self._err_history.append(err.copy())

            # Increment first, then plot: the frame is named for the number of
            # iterations behind it, and the last one is saved. Plotting before
            # the increment named the state after iteration 1 "0000" and never
            # reached numiter -- ThreePIETorch counts the other way round, so a
            # paired comparison was reading frames one iteration apart.
            self.curiter += 1
            self._check_probe_converged()
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        return {"all": err}

    def _display_object(self):
        """Slice transmissions for plotting; the tangent subclass, which
        stores log O, overrides this with the exponential."""
        return self.ob_g

    def _plot_progress(self):
        torch_helpers.plot_progress(
            self.pr_g,
            self._display_object(),
            self.curiter,
            self.plot_progress_dir,
            psize=getattr(self.p, "psize", (1e-6, 1e-6)),
            engine_name=type(self).__name__,
        )
        torch_helpers.plot_error_history(
            self._err_history,
            self.plot_progress_dir,
            engine_name=type(self).__name__,
        )

    def engine_finalize(self):
        for i in range(self.p.number_of_slices):
            arr = self.ob_g[i].cpu().numpy()
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._prs.data[:] = self.pr_g.cpu().numpy()

        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        info = Param(
            number_of_slices=self.p.number_of_slices,
            slice_thickness=self.p.slice_thickness,
            batch_size=self.p.batch_size,
            alpha=self.p.alpha,
            objects={
                ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                for ob in self._object
            },
        )
        info.update(self.extra_slices_info())

        # We must tell ptypy to ignore its own custom class types when saving to HDF5
        h5opt = io.h5options["UNSUPPORTED"]
        io.h5options["UNSUPPORTED"] = "ignore"

        logger.info(f"saving slices to {self.p.fslices}")
        io.h5write(
            self.p.fslices,
            header={"description": "ISS multislice"},
            content=info,
        )

        io.h5options["UNSUPPORTED"] = h5opt

        return super().engine_finalize()


@register()
class ISSTangentTorch(ISSTorch):
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
    -- batching, propagators, probe update, seeding -- is inherited
    unchanged, so a paired chord/tangent comparison differs in the
    perturbation alone.

    Defaults:

    [name]
    default = ISSTangentTorch
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
        self.ob_g = -1j * torch.log(self.ob_g)
        if bool(self.p.potential_real):
            self.ob_g = self.ob_g.real.to(self.cdt)
            logger.info("tangent potential constrained to be real (pure phase)")

    def _model_object(self, O_patches):
        # the core subtracts 1, so 1 + i s makes the perturbation g = i s
        return 1.0 + 1j * O_patches

    def _grad_to_param(self, grad):
        # dL/ds* = conj(dg/ds) dL/dg* = -i dL/dg*; for real s keep the
        # real part, which is the Wirtinger descent direction on the reals
        g = -1j * grad
        return g.real.to(self.cdt) if bool(self.p.potential_real) else g

    def _display_object(self):
        return torch.exp(1j * self.ob_g)

    def engine_finalize(self):
        # hand transmissions, not potentials, to the containers and file
        self.ob_g = torch.exp(1j * self.ob_g)
        return super().engine_finalize()
