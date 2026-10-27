"""
ISS (Inter-slice Single Scattering) engine for ptychography.

FORWARD MODEL
-------------
    \Delta O_j   = O_j - 1                            (scattering perturbation)
    P_j = F^-1[ H_{z_j} F[P] ]                        (unscattered illumination)
    psi   = P + sum_j F^-1[ H_{-z_j} F[ \Delta O_j P_j ] ] (exit wave)

with H_{z_j} = exp(2i pi (z_j/lam)(sqrt(1 - (lam f)^2) - 1)) and H_0 = 1, the whole
model referred to the entrance plane.

The slices enter as an affine SUM, not an ordered product. Because P_j
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
from ptypy.custom.torch.base_torch import ISSEngineInfra
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


# ---------------------------------------------------------------------------#
#  ISS Engine wrapper                                                 #
# ---------------------------------------------------------------------------#
@register()
class ISSTorch(ISSEngineInfra, stochastic.EPIE):
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

    [device]
    default = cuda
    type = str
    help = Torch device: 'cuda', 'cuda:0' or 'cpu'

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
    doc = Same name and meaning as in ThreePIETorch, so a paired comparison
          plots on the same cadence from one setting. On by default: with no
          explicit progress directory the figures and the terminal log land
          in iteration_progress_<engine>_<stamp>/ in the working directory,
          so every run leaves a browsable record.

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
        self._setup_torch_infra()
        N = self.p.number_of_slices

        # Persistent Gradient / Preconditioner Accumulators
        self._acc_grad = torch.zeros(
            (N, self.ony, self.onx), dtype=self.cdt, device=self.dev
        )
        self._acc_K = torch.zeros(
            (N, self.ony, self.onx), dtype=self.rdt, device=self.dev
        )

        self._err = torch.zeros(3, dtype=torch.float64, device=self.dev)
        self._z0 = torch.zeros((), dtype=self.rdt, device=self.dev)
        self._nbatch = 0
        self._pint = max(1, int(self.p.probe_update_interval))

        self._step = self._batch_update

        self._pad = False
        # Here, not in __init__: a caller may point plot_progress_dir somewhere
        # else between construction and the run, and making the default at
        # construction time left an empty timestamped directory behind.
        # Only when plotting is on: with plot_interval = 0 nothing is ever
        # written there and the empty timestamped directory is pure litter.
        if self._plot_int > 0:
            os.makedirs(self.plot_progress_dir, exist_ok=True)

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
            self.curiter >= self.p.probe_update_start
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

        # --- Update Probe (if scheduled) ---
        if do_probe:
            dn_p = K_P.amax()
            self.pr_g.add_(grad_P / dn_p.clamp_min(eps), alpha=float(self.p.beta) / N)
            if self.pr_support is not None:
                self.pr_g.mul_(self.pr_support)

        # --- Update Object Slices ---
        dn_o = self._acc_K.amax(dim=(-2, -1), keepdim=True)
        upd = self._grad_to_param(self._acc_grad / dn_o.clamp_min(eps))
        self.ob_g.add_(upd, alpha=float(self.p.alpha) / N)

    def engine_iterate(self, num=1):
        B = 1
        for _ in range(num):

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
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        return {"all": err}

    def engine_finalize(self):
        self._write_slices()
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
