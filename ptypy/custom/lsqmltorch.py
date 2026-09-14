"""
Multislice LSQ-ML: the 3PIE cascade with per-view, per-slice optimal step
lengths solved from the linearised least-squares problem.

    Odstrcil, Menzel & Guizar-Sicairos, "Iterative least-squares solver for
    generalized maximum-likelihood ptychography", Opt. Express 26(3), 3108
    (2018) -- the step-length solve of Eqs. (14)-(16), applied at every
    slice of the multislice cascade exactly as the multilayer engines built
    on it do (PtychoShelves 'MLs', fold_slice).

WHAT IS TAKEN FROM THE PAPER
----------------------------
At each plane the exit-wave residual chi is linearised jointly in the
object and the incident field,

    chi  ~=  alpha * (dO * psi)  +  beta * (dpsi * O),

with the preconditioned directions

    dO   = psi* chi / (sum_m |psi|^2 + delta * max),
    dpsi =   O* chi / (M |O|^2      + delta * max),

and (alpha, beta) are the REAL solution of the 2x2 normal equations over
the view's pixels and modes, Tikhonov-regularised. This replaces the fixed
ePIE steps of ThreePIE(Torch): the step is computed per view, per slice,
per iteration -- but from a LINEARISED model, unlike FastThreePIETorch's
exact quartic at the detector. That difference is the point of this
engine: it is the published adaptive-step baseline the exact line search
is compared against.

The multislice structure is the parent's sequential backward pass: at
slice i the incident-field correction beta * dpsi becomes, backpropagated,
the residual for slice i-1; at slice 0 it is the probe update.

NOT IMPLEMENTED (out of scope for the step-rule comparison): the momentum
-accelerated aggregation ('MLc'), variable intensity correction, position
refinement, and the illumination-weighted compact batch update -- run at
batch_size = 1, where sequential LSQ-ML is exactly what this is.

The directions are evaluated jointly at the CURRENT object (the paper
linearises both variables about the same point); the parent's Maiden-style
convention of using the already-updated object for the psi correction does
not apply here.
"""

import numpy as np

from ptypy.custom.threepietorch import ThreePIETorch
from ptypy.engines import register
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@register()
class LSQMLTorch(ThreePIETorch):
    """
    Multislice LSQ-ML: 3PIE cascade with per-view optimal steps.

    Defaults:

    [name]
    default = LSQMLTorch
    type = str
    help = Engine identifier

    [lsq_delta]
    default = 0.1
    type = float
    help = Relative floor of the direction denominators
    doc = The paper's delta: directions divide by the local intensity
          floored at lsq_delta times its per-view maximum. Only used with
          object_denom = 'local'; 'max' divides by the scalar maximum
          (raw-gradient direction) as in the parent.

    [lsq_reg]
    default = 0.1
    type = float
    help = Relative Tikhonov regularisation of the 2x2 step solve
    doc = Added to the diagonal as lsq_reg * mean(diagonal); stabilises
          the solve when the object and probe responses become nearly
          parallel.

    [step_damp]
    default = 1.0
    type = float
    help = Multiplier on every LSQ step
    doc = The published method runs the full LSQ step; exposed for
          ablation against the fast engines' ls_damp.

    [max_step]
    default = 0.0
    type = float
    help = Cap on the magnitude of the LSQ steps, 0 disables
    """

    def engine_initialize(self):
        super().engine_initialize()
        if any(self.p.slice_start_iteration > 0):
            raise NotImplementedError(
                "LSQMLTorch requires all slices active from iteration 0"
            )
        self._lsq_delta = float(self.p.lsq_delta)
        self._lsq_reg = float(self.p.lsq_reg)
        self._damp = float(self.p.step_damp)
        self._cap = float(self.p.max_step)
        self._steps_o, self._steps_p = [], []
        logger.info(
            "LSQ-ML steps: direction denom '%s' (delta %.3g), solve reg "
            "%.3g, damp %.3g, cap %s"
            % (self._denom, self._lsq_delta, self._lsq_reg, self._damp,
               self._cap if self._cap > 0 else "none"))

    def _lsq_steps(self, Af, Bf, delta, joint):
        """
        Real (alpha, beta) per view from the normal equations of
        ||delta - alpha Af - beta Bf||^2, summed over modes and pixels.
        joint=False solves the alpha-only 1D problem (probe frozen at
        slice 0). Sums in float64 -- early residuals overflow float32.
        """
        dims = (1, 2, 3)
        a11 = (Af.real.square() + Af.imag.square()).sum(dim=dims).double()
        b1 = ((Af.conj() * delta).real).sum(dim=dims).double()
        if joint:
            a22 = (Bf.real.square() + Bf.imag.square()).sum(dim=dims).double()
            a12 = ((Af.conj() * Bf).real).sum(dim=dims).double()
            b2 = ((Bf.conj() * delta).real).sum(dim=dims).double()
            lam = self._lsq_reg * 0.5 * (a11 + a22)
            det = (a11 + lam) * (a22 + lam) - a12 * a12
            tiny = torch.finfo(torch.float64).tiny
            ok = det.abs() > tiny
            det = torch.where(ok, det, torch.ones_like(det))
            alpha = torch.where(ok, (b1 * (a22 + lam) - b2 * a12) / det,
                                torch.zeros_like(det))
            beta = torch.where(ok, (b2 * (a11 + lam) - b1 * a12) / det,
                               torch.zeros_like(det))
        else:
            lam = self._lsq_reg * a11
            alpha = b1 / (a11 + lam).clamp_min(
                torch.finfo(torch.float64).tiny)
            beta = torch.zeros_like(alpha)
        alpha = torch.nan_to_num(alpha) * self._damp
        beta = torch.nan_to_num(beta) * self._damp
        if self._cap > 0:
            alpha = alpha.clamp(-self._cap, self._cap)
            beta = beta.clamp(-self._cap, self._cap)
        return alpha, beta

    def _batch_update(self, idx):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        M = self.M
        eps = torch.finfo(self.rdt).eps
        use_max = self._denom == "max"

        do_pr = self.curiter >= self.p.probe_update_start

        # ---------------- forward cascade (parent's conventions) ----------
        ob_p = [self._gather(self.ob_g[i], idx) for i in range(N)]
        psi = self.pr_g[None].expand(B, M, self.fny, self.fnx)

        psis, exits = [], []
        for i in range(N):
            psis.append(psi)
            e = psi * ob_p[i][:, None]
            exits.append(e)
            if i < N - 1:
                psi = self._apply(self._fw_g[i], e)

        # ---------------- Fourier modulus projection ----------------------
        F = self._apply(self._ff_fw_g, exits[-1])
        Imod = (F.real**2 + F.imag**2).sum(dim=1)
        amod = torch.sqrt(Imod.clamp_min(0))

        am = self.amp_g[idx]
        mk = self.mask_g[idx]
        I_meas = self.I_g[idx]

        fm = (1.0 - mk) + mk * am / (amod + 1e-10)
        e_proj = self._apply(self._ff_bw_g, F * fm[:, None].to(self.cdt))

        # ---------------- per-view errors (parent conventions) ------------
        npix = float(self.fny * self.fnx)
        msum = mk.sum(dim=(-2, -1)).clamp_min(1.0)
        d = amod - am
        err_fmag = (mk * d * d).sum(dim=(-2, -1)) / msum
        err_exit = ((e_proj - exits[-1]).abs() ** 2).mean(
            dim=(-2, -1)).sum(dim=1)
        if self.p.compute_log_likelihood:
            err_phot = (mk * (Imod - I_meas) ** 2 / (I_meas + 1.0)).sum(
                dim=(-2, -1)) / npix
        else:
            err_phot = torch.zeros_like(err_fmag)

        # ---------------- backward: LSQ steps at every slice ---------------
        num_o = [torch.zeros_like(self.ob_g[0]) for _ in range(N)]
        cnt_o = [torch.zeros(self.ob_g[0].shape, dtype=self.rdt,
                             device=self.dev) for _ in range(N)]
        ones_patch = torch.ones((B, self.fny, self.fnx), dtype=self.rdt,
                                device=self.dev)
        num_pr = torch.zeros_like(self.pr_g)

        target = e_proj
        for i in range(N - 1, -1, -1):
            delta = target - exits[i]                       # (B, M, ny, nx)

            # object direction: psi* chi over the local incident intensity
            w = (psis[i].real.square()
                 + psis[i].imag.square()).sum(dim=1)        # (B, ny, nx)
            grad = (psis[i].conj() * delta).sum(dim=1)      # (B, ny, nx)
            if use_max:
                dn_o = w.amax(dim=(-2, -1))[:, None, None].clamp_min(eps)
            else:
                dn_o = (w + self._lsq_delta
                        * w.amax(dim=(-2, -1))[:, None, None]).clamp_min(eps)
            dO = grad / dn_o.to(self.cdt)

            # incident-field direction: O* chi over the object intensity,
            # both evaluated at the current point (joint linearisation)
            solve_joint = do_pr or i > 0
            if solve_joint:
                ow = M * (ob_p[i].real.square()
                          + ob_p[i].imag.square())          # (B, ny, nx)
                if use_max:
                    dn_p = ow.amax(dim=(-2, -1))[:, None, None].clamp_min(eps)
                else:
                    dn_p = (ow + self._lsq_delta
                            * ow.amax(dim=(-2, -1))[:, None, None]
                            ).clamp_min(eps)
                dpsi = ob_p[i].conj()[:, None] * delta / dn_p[:, None].to(
                    self.cdt)
                Bf = dpsi * ob_p[i][:, None]
            else:
                dpsi = None
                Bf = None

            Af = dO[:, None] * psis[i]
            alpha, beta = self._lsq_steps(Af, Bf, delta, solve_joint)
            self._steps_o.extend(alpha.detach().cpu().numpy().tolist())
            if solve_joint:
                self._steps_p.extend(beta.detach().cpu().numpy().tolist())

            a_c = alpha.to(self.rdt)[:, None, None].to(self.cdt)
            self._scatter_add(num_o[i], idx, a_c * dO)
            self._scatter_add(cnt_o[i], idx, ones_patch)

            if solve_joint:
                b_c = beta.to(self.rdt)[:, None, None, None].to(self.cdt)
                psi_new = psis[i] + b_c * dpsi
                if i == 0:
                    num_pr = num_pr + (b_c * dpsi).sum(dim=0)
            else:
                psi_new = psis[i]

            if i > 0:
                target = self._apply(self._bw_g[i - 1], psi_new)

        # ---------------- apply updates ------------------------------------
        # each view's increment already carries its own LSQ step; overlapping
        # views in a batch are averaged (identical to a single view at B = 1)
        for i in range(N):
            self.ob_g[i] += num_o[i] / cnt_o[i].clamp_min(1.0).to(self.cdt)

        if do_pr:
            self.pr_g = self.pr_g + num_pr / float(B)
            if self.pr_support is not None:
                self.pr_g = self.pr_g * self.pr_support
            self._phi_cache = None

        # ---------------- pack per-view errors -----------------------------
        ii = idx.detach().cpu().numpy()
        ef = err_fmag.detach().cpu().numpy()
        ep = err_phot.detach().cpu().numpy()
        ee = err_exit.detach().cpu().numpy()
        return {
            self.view_names[int(k)]: np.array([ef[j], ep[j], ee[j]])
            for j, k in enumerate(ii)
        }

    def engine_iterate(self, num=1):
        out = super().engine_iterate(num)
        if self._steps_o:
            so = np.asarray(self._steps_o[-(self.n_pos
                                            * int(self.p.number_of_slices)):])
            msg = ("LSQ steps: slice median %.3g (p5 %.3g, p95 %.3g)"
                   % (np.median(so), np.percentile(so, 5),
                      np.percentile(so, 95)))
            if self._steps_p:
                sp = np.asarray(self._steps_p[-(self.n_pos * int(
                    self.p.number_of_slices)):])
                msg += ", incident median %.3g" % np.median(sp)
            logger.info(msg)
        return out
