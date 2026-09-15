"""
MLISS: maximum-likelihood conjugate gradients for the ISS model.

WHY THIS CAN EXIST
------------------
The ISS exit field is AFFINE in the object and LINEAR in the probe, so along
a JOINT direction (object direction h, probe direction q) the field is an
exact quadratic in the scalar step t,

    psi(t) = psi0 + t*a + t^2*b,
    a = q + S(h~_j phi_j + g_j phiq_j),   b = S(h~_j phiq_j),

with S(x) = sum_j Prop_{-z_j}[x_j], phi_j the unscattered illumination and
phiq_j the propagated probe direction. The mode-summed intensity is then an
exact per-pixel QUARTIC in t, and every noise model of the thin-sample
maximum-likelihood engine (ptypy.engines.ML: Thibault & Guizar-Sicairos)
ports verbatim with (f, a, b) -> (F0, A, Bf):

    gaussian : w (I(t) - I_dat)^2            exact degree-8 polynomial
    poisson  : m (I(t) - I_dat log I(t))     Taylor coefficients as in ML.py
    euclid   : w (sqrt(I(t)) - sqrt(I_dat))^2  Taylor coefficients as in ML.py

with the gaussian weights w = mask/(1 + I_dat) (identical to ML.py's
GaussianModel at intensity_renormalization = 1 AND to LISSTorch's omega).
Directions are Polak-Ribiere+ conjugate gradients over the joint
(object, probe) variable, preconditioned by the exact diagonal normal
operator sum_j |phi_j|^2 (object) and sum |O|^2 (probe), with the ML.py
scale_probe_object coupling. The step is the global minimiser of the
polynomial (np.roots of its derivative), damped by ls_damp.

Frozen probe => b = 0 => the gaussian objective is an exact quartic and the
step reduces to LISSTorch's solve_step_cubic on (e, v, w) = (A0 - I_dat,
A1/2, A2): with debug_lsborn_equiv (euclid gradient direction +
gaussian-weighted line search + beta = 0) this engine reproduces
LISSTorch's global-line-search mode step for step.

MODES
-----
full_batch = True (default): one CG iteration = one exact global step from
the full-dataset gradient; batch_size only sets the memory CHUNK. This is
the mathematically clean CG (conjugacy needs a consistent objective).
full_batch = False: per-batch steps; beta is forced to 0 whenever the batch
composition changes, so with shuffling this degrades gracefully towards
exactly-stepped steepest descent per batch (LSISS-like). ls_damp guidance:
1.0 for full batch (the global exact step needs no cyclic under-relaxation),
0.5 for the stochastic mode.

COST per full-batch iteration: one gradient pass (forward + adjoint, as the
parent) plus one coefficient pass (one forward + two direction responses),
roughly 1.4-1.6x the transforms of one LISSTorch global-LS epoch at equal
chunk size -- but taking one global exact CG step instead of n_pos/batch
greedy local ones.

Error channels: slots 0 (Fourier amplitude error) and 1 (photon chi^2) keep
the parent's definitions so "Errors :: Fourier" stays comparable across
engines and noise models; slot 2 is the model negative log-likelihood per
position (deviation from the parent's exit-residual sum, documented here).

Inert inherited parameters: alpha, beta (no fixed-step fallback here --
a degenerate polynomial takes t = 0 and forces a CG restart), momentum,
probe_update_interval, born_mode. A warning is logged if any is set away
from its default.
"""

import numpy as np

from ptypy.custom.torch.iss_torch import ISSTorch
from ptypy.engines import register
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ===========================================================================
#  Pure math (tensor in / tensor out; no ptypy imports - ptyrad-portable)
# ===========================================================================
def ml_rho(ml_type, u, I_dat, mask):
    """Detector-plane factor rho = dL/du of the NLL, so that the exact
    Wirtinger field gradient is dL/dF* = rho * F (and torch autograd's
    .grad, which is 2*dL/dz*, equals twice the manual gradient built from
    this rho -- asserted in the unit tests).

    Note ML.py's per-model prefactors are NOT reproduced here: its gaussian
    gradient is 1x the NLL gradient while its poisson/euclid are 2x (an
    internal inconsistency that is harmless there because every step comes
    from a scale-invariant line search, as it does here). We normalise all
    three to the true NLL gradient.

    u     : mode-summed model intensity  (B, ny, nx) real
    I_dat : measured intensity           (B, ny, nx) real
    mask  : detector mask or None        (B, ny, nx) real
    """
    if ml_type == "gaussian":
        w = 1.0 / (1.0 + I_dat) if mask is None else mask / (1.0 + I_dat)
        return 2.0 * w * (u - I_dat)
    if ml_type == "poisson":
        rho = 1.0 - I_dat / (u + 1e-6)
        return rho if mask is None else rho * mask
    # euclid: Amodel = sqrt(u + 1e-12), the same regularisation as the
    # line-search coefficients (ML.py's new_grad uses Amodel + 1e-6 while
    # its poly coefficients use sqrt(A0 + 1e-12); we keep the two
    # consistent by adopting the coefficient convention everywhere)
    amod = (u.clamp_min(0) + 1e-12).sqrt()
    rho = 1.0 - I_dat.clamp_min(0).sqrt() / amod
    return rho if mask is None else rho * mask


def ml_LL(ml_type, u, I_dat, mask):
    """Model negative log-likelihood of one chunk (scalar, accumulated in
    float64). Poisson EXCLUDES the lgamma base term (added by the caller,
    which knows the chunk's measured counts)."""
    m = 1.0 if mask is None else mask.double()
    ud = u.double()
    Id = I_dat.double()
    if ml_type == "gaussian":
        w = m / (1.0 + Id)
        return float((w * (ud - Id).square()).sum())
    if ml_type == "poisson":
        up = ud + 1e-6
        return float((m * (up - Id * torch.log(up))).sum())
    amod = (ud.clamp_min(0) + 1e-12).sqrt()
    return float((m * (amod - Id.clamp_min(0).sqrt()).square()).sum())


def iss_poly_Ak(F0, A, Bf=None):
    """Per-pixel coefficients of the intensity quartic I(t) = A0 + A1 t +
    A2 t^2 + A3 t^3 + A4 t^4 for the far field F(t) = F0 + t A + t^2 Bf,
    mode-summed over dim 1, in float64 (overflow guard). Bf = None (frozen
    probe) returns A3 = A4 = None."""
    A0 = (F0.real.square() + F0.imag.square()).sum(1).double()
    A1 = 2.0 * (F0.real * A.real + F0.imag * A.imag).sum(1).double()
    if Bf is None:
        A2 = (A.real.square() + A.imag.square()).sum(1).double()
        return A0, A1, A2, None, None
    A2 = ((A.real.square() + A.imag.square())
          + 2.0 * (F0.real * Bf.real + F0.imag * Bf.imag)).sum(1).double()
    A3 = 2.0 * (A.real * Bf.real + A.imag * Bf.imag).sum(1).double()
    A4 = (Bf.real.square() + Bf.imag.square()).sum(1).double()
    return A0, A1, A2, A3, A4


def _zero_like(x):
    return torch.zeros_like(x)


def ml_poly_coeffs_gaussian(A0, A1, A2, A3, A4, I_dat, mask):
    """B[0..8] of the exact degree-8 gaussian LL(t); ML.py:745-753."""
    if A3 is None:
        A3 = _zero_like(A2)
        A4 = _zero_like(A2)
    I = I_dat.double()
    w = (1.0 / (1.0 + I)) if mask is None else mask.double() / (1.0 + I)
    d = A0 - I
    B = np.zeros(9, dtype=np.float64)
    B[0] = float((w * d * d).sum())
    B[1] = float((w * 2.0 * d * A1).sum())
    B[2] = float((w * (A1 * A1 + 2.0 * d * A2)).sum())
    B[3] = float((w * (2.0 * d * A3 + 2.0 * A1 * A2)).sum())
    B[4] = float((w * (A2 * A2 + 2.0 * A1 * A3 + 2.0 * d * A4)).sum())
    B[5] = float((w * (2.0 * A1 * A4 + 2.0 * A2 * A3)).sum())
    B[6] = float((w * (A3 * A3 + 2.0 * A2 * A4)).sum())
    B[7] = float((w * 2.0 * A3 * A4).sum())
    B[8] = float((w * A4 * A4).sum())
    return B


def ml_poly_coeffs_poisson(A0, A1, A2, A3, A4, I_dat, mask, llbase):
    """B[0..8] of the poisson NLL(t) Taylor expansion; ML.py:997-1005."""
    if A3 is None:
        A3 = _zero_like(A2)
        A4 = _zero_like(A2)
    I = I_dat.double()
    m = 1.0 if mask is None else mask.double()
    A0 = A0 + 1e-6
    DI = 1.0 - I / A0
    B = np.zeros(9, dtype=np.float64)
    B[0] = llbase + float((m * (A0 - I * torch.log(A0))).sum())
    B[1] = float((m * (A1 * DI)).sum())
    B[2] = float((m * (A2 * DI)).sum()
                 + 0.5 * float((m * (I * (A1 / A0) ** 2)).sum()))
    B[3] = float((m * (A3 * DI)).sum()
                 + 0.5 * float((m * (I * (2.0 * A1 * A2) / A0**2)).sum()))
    B[4] = float((m * (A4 * DI)).sum()
                 + 0.5 * float((m * (I * (A2**2 + 2.0 * A1 * A3) / A0**2)).sum()))
    B[5] = 0.5 * float((m * (I * (2.0 * A1 * A4 + 2.0 * A2 * A3) / A0**2)).sum())
    B[6] = 0.5 * float((m * (I * (A3**2 + 2.0 * A2 * A4) / A0**2)).sum())
    B[7] = 0.5 * float((m * (I * (2.0 * A3 * A4) / A0**2)).sum())
    B[8] = 0.5 * float((m * (I * (A4 / A0) ** 2)).sum())
    return B


def ml_poly_coeffs_euclid(A0, A1, A2, A3, A4, I_dat, mask):
    """B[0..8] of the euclid (amplitude) NLL(t) Taylor expansion;
    ML.py:1265-1278, mode-summed amplitude convention."""
    if A3 is None:
        A3 = _zero_like(A2)
        A4 = _zero_like(A2)
    Amp = I_dat.clamp_min(0).sqrt().double()
    w = 1.0 if mask is None else mask.double()
    A0 = A0 + 1e-12
    rA0 = A0.sqrt()
    DA = 1.0 - Amp / rA0
    DA32 = Amp / A0.pow(1.5)
    B = np.zeros(9, dtype=np.float64)
    B[0] = float((w * (rA0 - Amp).square()).sum())
    B[1] = float((w * (A1 * DA)).sum())
    B[2] = float((w * (A2 * DA)).sum()
                 + 0.25 * float((w * (A1**2 * DA32)).sum()))
    B[3] = float((w * (A3 * DA)).sum()
                 + 0.25 * float((w * (2.0 * A1 * A2 * DA32)).sum())
                 - 0.125 * float((w * (A1**3 / A0**2)).sum()))
    B[4] = float((w * (A4 * DA)).sum()
                 + 0.25 * float((w * ((A2**2 + 2.0 * A1 * A3) * DA32)).sum())
                 - 0.125 * float((w * ((3.0 * A1**2 * A2) / A0**2)).sum())
                 + 0.015625 * float((w * (A1**4 / A0**3)).sum()))
    B[5] = (0.25 * float((w * ((2.0 * A2 * A3 + 2.0 * A1 * A4) * DA32)).sum())
            - 0.125 * float((w * ((3.0 * A1 * A2**2 + 3.0 * A1**2 * A3) / A0**2)).sum())
            + 0.015625 * float((w * ((4.0 * A1**3 * A2) / A0**3)).sum()))
    B[6] = (0.25 * float((w * ((A3**2 + 2.0 * A2 * A4) * DA32)).sum())
            - 0.125 * float((w * ((A2**3 + 3.0 * A1**2 * A4
                                   + 6.0 * A1 * A2 * A3) / A0**2)).sum())
            + 0.015625 * float((w * ((6.0 * A1**2 * A2**2
                                      + 4.0 * A1**3 * A3) / A0**3)).sum()))
    B[7] = (0.25 * float((w * (2.0 * A3 * A4 * DA32)).sum())
            - 0.125 * float((w * ((3.0 * A2**2 * A3 + 3.0 * A1 * A3**2
                                   + 6.0 * A1 * A2 * A4) / A0**2)).sum())
            + 0.015625 * float((w * ((4.0 * A1 * A2**3 + 12.0 * A1**2 * A2 * A3
                                      + 4.0 * A1**3 * A4) / A0**3)).sum()))
    B[8] = (0.25 * float((w * (A4**2 * DA32)).sum())
            - 0.125 * float((w * ((3.0 * A2 * A3**2 + 3.0 * A2**2 * A4
                                   + 6.0 * A1 * A3 * A4) / A0**2)).sum())
            + 0.015625 * float((w * ((A2**4 + 12.0 * A1 * A2**2 * A3
                                      + 6.0 * A1**2 * A3**2
                                      + 12.0 * A1**2 * A2 * A4) / A0**3)).sum()))
    return B


def poly_global_min(B, ls_damp=1.0, max_step=0.0):
    """Global minimiser of the polynomial sum_k B[k] t^k: real roots of the
    derivative, keep the one with the lowest value if it improves on t = 0.
    Returns (t, degenerate): degenerate = True means no improving real root
    was found (t = 0) and the caller should restart CG."""
    B = np.asarray(B, dtype=np.float64)
    if not np.all(np.isfinite(B)):
        return 0.0, True
    dB = np.arange(1, len(B)) * B[1:]
    co = np.flip(dB)                       # highest power first
    co = list(co)
    while co and abs(co[0]) < 1e-300:
        co = co[1:]
    if len(co) < 2:
        return 0.0, True
    roots = np.roots(co)
    cands = [float(r.real) for r in roots
             if abs(r.imag) <= 1e-8 * (1.0 + abs(r.real))]
    if not cands:
        return 0.0, True
    Bflip = np.flip(B)
    q0 = float(np.polyval(Bflip, 0.0))
    best, bestq = None, q0
    for t in cands:
        qt = float(np.polyval(Bflip, t))
        if np.isfinite(qt) and qt < bestq:
            best, bestq = t, qt
    if best is None:
        return 0.0, True
    t = best * float(ls_damp)
    if max_step and max_step > 0:
        t = float(np.clip(t, -max_step, max_step))
    return t, False


# ===========================================================================
#  Engine
# ===========================================================================
@register()
class MLISSTorch(ISSTorch):
    """
    MLISS: maximum-likelihood conjugate gradients on the affine ISS model.

    DESIGN NOTE: this engine subclasses ISSTorch, the stable base, NOT its
    sibling LISSTorch -- LISSTorch is hardwired to one objective (the
    gaussian-weighted quartic on the projection residual) and evolves
    independently, so inheriting from it would couple two engines that are
    semantically peers. The three small affine-model helpers the two share
    (_phi_fast, _scattered, _dir_to_model) are duplicated here by design,
    following the codebase's tangent-variant precedent; keep them in sync
    by hand if the physics ever changes. LISSTorch's optional probe gauge
    projections (fourier/modulus/rsamp/power) are NOT duplicated in this
    version; the plain circular probe support of ISSTorch is available.

    Defaults:

    [name]
    default = MLISSTorch
    type = str
    help = Engine identifier

    [ML_type]
    default = gaussian
    type = str
    help = Noise model, one of gaussian | poisson | euclid
    doc = gaussian is intensity least squares with weights mask/(1+I),
          identical to ML.py's GaussianModel at intensity_renormalization = 1
          and to the parent's line-search weights. poisson is the counting
          NLL; euclid is amplitude least squares on the mode-summed
          amplitude, the objective closest to the parent's projection
          residual.

    [poly_line_coeffs]
    default = all
    type = str
    help = Line-search polynomial order, all | quadratic
    doc = all evaluates the full polynomial (degree 8 with a moving probe,
          degree 4 frozen; gaussian is exact, poisson/euclid are the ML.py
          Taylor expansions) and takes the global minimiser of its
          derivative. quadratic keeps B[0..2] and steps to the parabola
          vertex, ML.py's faster default.

    [full_batch]
    default = True
    type = bool
    help = One CG step per iteration from the full-dataset gradient
    doc = batch_size becomes the memory chunk only. False runs per-batch
          steps with beta forced to 0 whenever the batch composition
          changes (stochastic mode, LSISS-like but noise-model aware).

    [restart_interval]
    default = 20
    type = int
    help = Scheduled CG restart (beta = 0) every this many iterations; 0 disables

    [scale_probe_object]
    default = 1.0
    type = float
    help = Relative scale of the probe vs object direction (ML.py convention)

    [scale_precond]
    default = False
    type = bool
    help = Adapt scale_probe_object from the gradient norms with 0.9 memory
    doc = ML.py's scale_precond / scale_p_o_memory behaviour.

    [ls_damp]
    default = 1.0
    type = float
    help = Damping of the polynomial line-search step
    doc = LISSTorch uses 0.5 because its per-batch exact steps are a cyclic
          solver that needs under-relaxation; the full-batch CG step here
          minimises the GLOBAL objective, so 1.0 is the natural default.
          For full_batch = False keep 0.5.

    [max_step]
    default = 0.0
    type = float
    help = Hard cap |t| on the line-search step; 0 disables

    [probe_denom]
    default = max
    type = str
    help = Probe preconditioner denominator, max or local
    doc = Same convention as LISSTorch: 'max' divides the probe gradient by
          the peak of sum|O|^2 (robust against hot pixels), 'local' by the
          per-pixel map regularised with denom_reg.

    [reg_del2]
    default = False
    type = bool
    help = Placeholder for the Gaussian-prior regularizer (not implemented)
    doc = ML.py's Regul_del2 port is planned; setting True logs a warning
          and is otherwise ignored in this version.

    [reg_del2_amplitude]
    default = 0.01
    type = float
    help = Amplitude of the (not yet implemented) del2 regularizer

    [debug_lsborn_equiv]
    default = False
    type = bool
    help = Test hook, reproduces LISSTorch's global line-search mode
    doc = Uses the euclid (projection-residual) gradient direction with the
          gaussian mask/(1+I) line-search weights and beta = 0: with a
          frozen probe, full batch and equal ls_damp/denominators this is
          step-for-step LISSTorch with per_slice_ls = False.
    """

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()
        # affine-model helper state (kept in sync with LISSTorch by hand,
        # see the class docstring)
        self._phi_f = None
        # start inside the probe support so the direction-space projection
        # (see _cg_update) keeps every iterate supported from the start
        if self.pr_support is not None:
            self.pr_g = self.pr_g * self.pr_support
        self._ls_damp = float(self.p.ls_damp)
        self._ls_cap = float(self.p.max_step)
        self._use_max_probe = str(self.p.probe_denom).lower().startswith("m")

        self._ml_type = str(self.p.ML_type).lower()
        if self._ml_type not in ("gaussian", "poisson", "euclid"):
            raise ValueError("ML_type must be gaussian | poisson | euclid, "
                             "got %r" % self.p.ML_type)
        self._poly_all = str(self.p.poly_line_coeffs).lower().startswith("a")
        self._full_batch = bool(self.p.full_batch)
        self._restart_int = int(self.p.restart_interval)
        self._spo = float(self.p.scale_probe_object)
        self._spo_adapt = bool(self.p.scale_precond)
        self._spo_mem = 0.9
        self._lsb_equiv = bool(self.p.debug_lsborn_equiv)

        # CG state: current joint direction and previous preconditioned grads
        self._cg_oh = torch.zeros_like(self.ob_g)
        self._cg_ph = torch.zeros_like(self.pr_g)
        self._Do_prev = None
        self._Dp_prev = None
        self._force_restart = True
        self._it_since_restart = 0
        self._last_batch_key = None

        # probe-gradient accumulators (the parent only has object ones)
        self._acc_gP = torch.zeros_like(self.pr_g)
        self._acc_KP = torch.zeros((self.fny, self.fnx), dtype=self.rdt,
                                   device=self.dev)
        self._LL = 0.0
        if self.maskb is not None:
            self._tot_meas = float(self.maskb.sum())
        else:
            self._tot_meas = float(self.n_pos * self.fny * self.fnx)

        # inert inherited parameters: loudly documented, silently unused
        inert = []
        if float(self.p.momentum) > 0:
            inert.append("momentum")
        if int(self.p.probe_update_interval) > 1:
            inert.append("probe_update_interval")
        if inert:
            logger.warning("MLISSTorch ignores: %s (see engine docstring)"
                           % ", ".join(inert))
        if bool(self.p.reg_del2):
            logger.warning("reg_del2 is not implemented in this version "
                           "and is ignored")
        logger.info(
            "MLISS: ML_type=%s, poly=%s, full_batch=%s, restart=%d, "
            "ls_damp=%.3g%s"
            % (self._ml_type,
               "all" if self._poly_all else "quadratic",
               self._full_batch, self._restart_int, self._ls_damp,
               ", debug_lsborn_equiv" if self._lsb_equiv else ""))

    # ------------------------------------------------------------------ #
    #  Affine-model helpers (duplicated from LISSTorch by design; see the #
    #  class docstring -- keep in sync by hand)                           #
    # ------------------------------------------------------------------ #
    def _phi_fast(self):
        if self._phi_f is None:
            P_f = torch.fft.fft2(self.pr_g)                    # (M, ny, nx)
            self._phi_f = torch.fft.ifft2(self.H_fwd[:, 0] * P_f.unsqueeze(0))
        return self._phi_f                                     # (N, M, ny, nx)

    def _scattered(self, per_slice_fields):
        """sum_j Prop_{-z_j}[ x_j ] for x_j of shape (B, M, ny, nx)."""
        acc = None
        for j, x in enumerate(per_slice_fields):
            t = self.H_rev[j, 0] * torch.fft.fft2(x)
            acc = t if acc is None else acc + t
        return torch.fft.ifft2(acc)

    def _dir_to_model(self, d):
        """Model-perturbation direction for a parameter-space direction:
        identity for the chord; the tangent subclass overrides with i*d."""
        return d

    # ------------------------------------------------------------------ #
    #  Pass A: model gradient of one chunk into the accumulators          #
    # ------------------------------------------------------------------ #
    def _ml_grad_chunk(self, idx, do_probe, phi):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        flat = self._flat(idx)

        O = torch.stack([self._gather(self.ob_g[j], flat, B) for j in range(N)])
        O = self._model_object(O)
        g = O - 1.0
        amp_dat = self.amp_dat[idx]
        I_dat = amp_dat.square()
        mask = self.maskb[idx].to(self.rdt) if self.maskb is not None else None

        psi = self.pr_g.unsqueeze(0) + self._scattered(
            g[j].unsqueeze(1) * phi[j] for j in range(N))
        F = self._apply(self._ff_fw, psi)                  # (B, M, ny, nx)
        u = torch.sum(F.real.square() + F.imag.square(), dim=1)

        rho_type = "euclid" if self._lsb_equiv else self._ml_type
        rho = ml_rho(rho_type, u, I_dat, mask)
        LL = ml_LL(self._ml_type, u, I_dat, mask)
        if self._ml_type == "poisson":
            m = 1.0 if mask is None else mask
            LL += float((m * torch.lgamma(I_dat.double() + 1.0)).sum())
        self._LL += LL

        r = self._apply(self._ff_bw, F * rho.unsqueeze(1).to(self.cdt))
        r_f = torch.fft.fft2(r)
        acc_P = None
        for j in range(N):
            r_j = torch.fft.ifft2(self.H_fwd[j, 0] * r_f)
            self._scatter(self._acc_grad[j], flat,
                          (phi[j].conj().unsqueeze(0) * r_j).sum(dim=1))
            K_j = (phi[j].real.square() + phi[j].imag.square()).sum(dim=0)
            self._scatter(self._acc_K[j], flat,
                          K_j.expand(B, self.fny, self.fnx))
            if do_probe:
                t = self.H_rev[j, 0] * torch.fft.fft2(
                    g[j].conj().unsqueeze(1) * r_j)
                acc_P = t if acc_P is None else acc_P + t
            del r_j
        if do_probe:
            self._acc_gP += (r + torch.fft.ifft2(acc_P)).sum(dim=0)
            self._acc_KP += torch.sum(O.real.square() + O.imag.square(),
                                      dim=(0, 1))

        # error channels: 0/1 keep the parent's definitions (per-position
        # means, so the "Errors :: Fourier" column is engine-comparable at
        # any chunk size); 2 is the model NLL of the chunk.
        amp_mod = u.clamp_min(0).sqrt()
        m_weight = 1.0 if mask is None else mask
        m_sum = (float(B * self.fny * self.fnx) if mask is None
                 else mask.sum().clamp_min(1))
        e0 = B * (m_weight * (amp_mod - amp_dat).square()).sum() / m_sum
        e1 = B * (m_weight * (u - I_dat).square() / (I_dat + 1.0)).sum() / m_sum
        self._err.add_(torch.stack(
            [e0, e1, torch.as_tensor(LL, device=self.dev)]
        ).to(torch.float64))

    # ------------------------------------------------------------------ #
    #  Pass B: polynomial coefficients of one chunk along the direction   #
    # ------------------------------------------------------------------ #
    def _ml_coeff_chunk(self, idx, phi, phiq, do_probe):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        flat = self._flat(idx)

        O = torch.stack([self._gather(self.ob_g[j], flat, B) for j in range(N)])
        O = self._model_object(O)
        g = O - 1.0
        amp_dat = self.amp_dat[idx]
        I_dat = amp_dat.square()
        mask = self.maskb[idx].to(self.rdt) if self.maskb is not None else None

        h = torch.stack([self._gather(self._cg_oh[j], flat, B)
                         for j in range(N)])
        hm = self._dir_to_model(h)                        # model perturbation

        F0 = self._apply(self._ff_fw, self.pr_g.unsqueeze(0) + self._scattered(
            g[j].unsqueeze(1) * phi[j] for j in range(N)))

        if do_probe:
            a_field = self._cg_ph.unsqueeze(0) + self._scattered(
                hm[j].unsqueeze(1) * phi[j] + g[j].unsqueeze(1) * phiq[j]
                for j in range(N))
            b_field = self._scattered(
                hm[j].unsqueeze(1) * phiq[j] for j in range(N))
            A = self._apply(self._ff_fw, a_field)
            Bf = self._apply(self._ff_fw, b_field)
        else:
            A = self._apply(self._ff_fw, self._scattered(
                hm[j].unsqueeze(1) * phi[j] for j in range(N)))
            Bf = None

        A0, A1, A2, A3, A4 = iss_poly_Ak(F0, A, Bf)

        coeff_type = "gaussian" if self._lsb_equiv else self._ml_type
        if coeff_type == "gaussian":
            return ml_poly_coeffs_gaussian(A0, A1, A2, A3, A4, I_dat, mask)
        if coeff_type == "poisson":
            m = 1.0 if mask is None else mask
            llbase = float((m * torch.lgamma(I_dat.double() + 1.0)).sum())
            return ml_poly_coeffs_poisson(A0, A1, A2, A3, A4, I_dat, mask,
                                          llbase)
        return ml_poly_coeffs_euclid(A0, A1, A2, A3, A4, I_dat, mask)

    # ------------------------------------------------------------------ #
    #  One CG update over the given chunks (full batch: all of them)      #
    # ------------------------------------------------------------------ #
    def _cg_update(self, chunks, do_probe, phi, restart):
        eps = torch.finfo(self.rdt).eps
        reg = float(self.p.denom_reg)

        # --- preconditioned gradients in parameter space ------------------
        peak_K = self._acc_K.amax(dim=(-2, -1), keepdim=True)
        dn_o = peak_K if self._use_max else self._acc_K + peak_K * reg
        Do = self._grad_to_param(self._acc_grad / dn_o.clamp_min(eps))
        Dp = None
        if do_probe:
            peak_P = self._acc_KP.amax()
            dn_p = (peak_P if self._use_max_probe
                    else self._acc_KP + peak_P * reg)
            Dp = self._acc_gP / dn_p.clamp_min(eps)
            if self.pr_support is not None:
                # the support is a LINEAR projector: constraining the
                # DIRECTION keeps every iterate inside the support, makes
                # the post-step projection a no-op and preserves conjugacy
                # (CG on the supported subspace).
                Dp = Dp * self.pr_support

        # --- probe/object scale (ML.py scale_precond) ---------------------
        if self._spo_adapt and Dp is not None:
            cn2_p = float(Dp.abs().square().double().sum())
            cn2_o = float(Do.abs().square().double().sum())
            spo_new = (float(self.p.scale_probe_object) * cn2_o / cn2_p
                       if cn2_p > 1e-5 else float(self.p.scale_probe_object))
            if self._scale_p_o is None:
                self._scale_p_o = spo_new
            else:
                self._scale_p_o = (self._scale_p_o ** self._spo_mem
                                   * spo_new ** (1.0 - self._spo_mem))
            spo = self._scale_p_o
        else:
            spo = self._spo

        # --- Polak-Ribiere+ ------------------------------------------------
        bt = 0.0
        if (not restart and self._Do_prev is not None
                and not self._lsb_equiv):
            def _cn2(x):
                return float(x.abs().square().double().sum())

            def _redot(x, y):
                return float(torch.sum(
                    (x.conj() * y).real.double()))

            num = _cn2(Do) - _redot(Do, self._Do_prev)
            den = _cn2(self._Do_prev)
            if Dp is not None and self._Dp_prev is not None:
                num += spo * (_cn2(Dp) - _redot(Dp, self._Dp_prev))
                den += spo * _cn2(self._Dp_prev)
            bt = max(0.0, num / den) if den > 0 else 0.0

        if bt == 0.0:
            self._it_since_restart = 0
        self._cg_oh.mul_(bt).sub_(Do)
        if do_probe and Dp is not None:
            self._cg_ph.mul_(bt).sub_(spo * Dp)
        else:
            self._cg_ph.zero_()

        # --- polynomial coefficients along the joint direction ------------
        phiq = None
        if do_probe:
            phiq = torch.fft.ifft2(
                self.H_fwd[:, 0] * torch.fft.fft2(self._cg_ph).unsqueeze(0))

        def _coeffs():
            Bc = np.zeros(9, dtype=np.float64)
            for idx in chunks:
                Bc += self._ml_coeff_chunk(idx, phi, phiq, do_probe)
            return Bc

        Bc = _coeffs()
        if Bc[1] >= 0.0 and bt > 0.0:
            # stale conjugacy: uphill at t = 0 -> steepest descent, once
            logger.info("CG: non-descent direction (B1 = %.3e), restarting"
                        % Bc[1])
            self._cg_oh.zero_().sub_(Do)
            if do_probe and Dp is not None:
                self._cg_ph.zero_().sub_(spo * Dp)
            bt = 0.0
            self._it_since_restart = 0
            if do_probe:
                phiq = torch.fft.ifft2(
                    self.H_fwd[:, 0]
                    * torch.fft.fft2(self._cg_ph).unsqueeze(0))
            Bc = _coeffs()

        # renormalise for np.roots conditioning only (does not move roots)
        scale = abs(Bc[0]) if abs(Bc[0]) > 0 else 1.0
        Bn = Bc / scale

        # --- the step ------------------------------------------------------
        if self._poly_all:
            tmin, degenerate = poly_global_min(Bn, self._ls_damp,
                                               self._ls_cap)
        else:
            degenerate = not (np.isfinite(Bn[2]) and Bn[2] > 0)
            tmin = 0.0 if degenerate else float(-0.5 * Bn[1] / Bn[2])
            tmin *= self._ls_damp
            if self._ls_cap > 0:
                tmin = float(np.clip(tmin, -self._ls_cap, self._ls_cap))
        if degenerate:
            self._force_restart = True

        self.ob_g += tmin * self._cg_oh
        probe_moved = False
        if do_probe and Dp is not None and tmin != 0.0:
            self.pr_g = self.pr_g + tmin * self._cg_ph
            if self.pr_support is not None:
                # the support projection is a non-linear jump ONLY when it
                # actually moves the probe; a probe already inside its
                # support passes through unchanged and conjugacy survives.
                pre = self.pr_g
                post = pre * self.pr_support
                delta = float((post - pre).abs().square().sum())
                norm = float(pre.abs().square().sum())
                self.pr_g = post
                if delta > 1e-12 * max(norm, 1e-30):
                    self._force_restart = True
            self._phi_f = None
            probe_moved = True

        self._Do_prev = Do
        self._Dp_prev = Dp
        self._it_since_restart += 1
        return bt, tmin, probe_moved

    # ------------------------------------------------------------------ #
    #  Iteration loop                                                     #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        import time as _time
        B = int(self.p.batch_size)
        err = np.zeros(3)
        for _ in range(num):
            t0 = _time.time()
            self._err.zero_()
            self._LL = 0.0

            do_probe = (not self._probe_frozen
                        and self.curiter >= int(self.p.probe_update_start))

            if self._full_batch:
                order = torch.arange(self.n_pos, device=self.dev)
                chunks = [order[s:s + B]
                          for s in range(0, self.n_pos, B)]
                self._acc_grad.zero_()
                self._acc_K.zero_()
                self._acc_gP.zero_()
                self._acc_KP.zero_()
                phi = self._phi_fast()
                for idx in chunks:
                    self._ml_grad_chunk(idx, do_probe, phi)
                restart = (self._force_restart
                           or self._Do_prev is None
                           or (self._restart_int > 0
                               and self._it_since_restart
                               >= self._restart_int))
                self._force_restart = False
                bt, tmin, probe_moved = self._cg_update(
                    chunks, do_probe, phi, restart)
                if probe_moved:
                    phi = None
            else:
                # stochastic mode: per-batch CG with restart on any change
                # of the batch composition
                if self.p.shuffle:
                    order = (torch.randperm(self.n_pos, device=self.dev)
                             if self._order_gen is None
                             else torch.randperm(self.n_pos,
                                                 generator=self._order_gen,
                                                 device=self.dev))
                else:
                    order = torch.arange(self.n_pos, device=self.dev)
                bt = tmin = 0.0
                for s in range(0, order.numel(), B):
                    idx = order[s:s + B]
                    self._acc_grad.zero_()
                    self._acc_K.zero_()
                    self._acc_gP.zero_()
                    self._acc_KP.zero_()
                    phi = self._phi_fast()
                    self._ml_grad_chunk(idx, do_probe, phi)
                    key = (int(idx.numel()), int(idx.min()),
                           int(idx.max()), int(idx.sum()))
                    restart = (self._force_restart
                               or self._Do_prev is None
                               or key != self._last_batch_key
                               or (self._restart_int > 0
                                   and self._it_since_restart
                                   >= self._restart_int))
                    self._force_restart = False
                    bt, tmin, _ = self._cg_update([idx], do_probe, phi,
                                                  restart)
                    self._last_batch_key = key

            err = (self._err / float(self.n_pos)).cpu().numpy()
            self._err_history.append(err.copy())
            logger.info("CG: beta %.3g, tmin %.3g, LL %.6e, %.2fs"
                        % (bt, tmin, self._LL, _time.time() - t0))

            self.curiter += 1
            self._check_probe_converged()
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        return {"all": err}


@register()
class MLISSTangentTorch(MLISSTorch):
    """
    MLISS with the tangent (linearised-transmission) perturbation g = i s.

    Same conventions as ISSTangentTorch / LSISSTangentTorch: the working
    object is the scattering potential s, the forward model receives
    g = i s, gradients are rotated by -i (and projected real when
    potential_real is set -- BEFORE the conjugacy inner products, so the
    imaginary leakage cannot bias beta), and the slices are exponentiated
    back to transmissions for plotting and saving.

    Defaults:

    [name]
    default = MLISSTangentTorch
    type = str
    help = Engine identifier

    [potential_real]
    default = False
    type = bool
    help = Constrain the potential s to be real, i.e. a pure phase object
    doc = The strict weak-phase-object convention: g = i s with s real
          forbids absorption. False reconstructs a complex s, whose
          imaginary part carries the absorption.
    """

    def engine_initialize(self):
        super().engine_initialize()
        self.ob_g = -1j * torch.log(self.ob_g)
        if bool(self.p.potential_real):
            self.ob_g = self.ob_g.real.to(self.cdt)
            logger.info("tangent potential constrained to be real "
                        "(pure phase)")

    def _model_object(self, O_patches):
        return 1.0 + 1j * O_patches

    def _grad_to_param(self, grad):
        g = -1j * grad
        return g.real.to(self.cdt) if bool(self.p.potential_real) else g

    def _dir_to_model(self, d):
        return 1j * d

    def _display_object(self):
        return torch.exp(1j * self.ob_g)

    def engine_finalize(self):
        self.ob_g = torch.exp(1j * self.ob_g)
        return super().engine_finalize()
