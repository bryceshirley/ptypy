"""
Multislice (3PIE cascade) with exact per-slice line searches: the
LSISSTorch idea carried to the full ordered-product model.

WHY THIS CAN EXIST
------------------
The multislice exit field is MULTILINEAR in the slices: linear in each
slice O_j separately (holding the others fixed) and linear in the probe.
So although a joint step over all slices moves the far field on a
degree-N curve, a step confined to ONE slice moves it exactly linearly,

    F(a) = F + a D_j,   D_j = far field of (d_j psi_j^in) pushed through
                              the remaining cascade,

and the Gaussian intensity cost along a is the same exact quartic as in
the Born engines -- one cubic root, no trial steps. The same holds for
the probe through the full cascade.

THE SWEEP
---------
Slices are stepped sequentially, j = 0..N-1. After slice j's exact step
the far field is EXACTLY F + a_j D_j (elementwise update, no forward
re-evaluation), and the incident fields at every downstream slice change
by exactly the intermediates already computed for D_j -- so they are
maintained exactly too. Each slice's line search therefore minimises the
true cost given every update already applied in this batch; only the
DIRECTIONS (preconditioned gradients from the initial adjoint pass) are
held fixed across the sweep. The probe line search runs last, against
the fully updated cascade.

Directions are the multislice analogue of the parent's: the residual
r = bw(F fac) backpropagated up the cascade, preconditioned per slice by
the incident intensity sum_m |psi_j^in|^2 (the 'local' denominator).

MOTIVATION
----------
ThreePIE's 10-slice failure is cascade overshoot at low spatial
frequencies with the fixed ePIE step -- a step-size pathology, not a
gradient bug. The exact quartic knows the true curvature along each
slice's direction, including the low-f directions where the fixed step
overshoots; ls_damp then under-relaxes for the cyclic sweep exactly as
in the Born engines.

COST
----
Per batch: forward cascade N propagations, adjoint N, slice responses
sum_j (N - j) ~ N^2/2, probe response N. For N = 10 roughly 3-4x a
ThreePIETorch iteration.
"""

import numpy as np

from ptypy.custom.lsisstorch import solve_step_cubic
from ptypy.custom.threepietorch import ThreePIETorch
from ptypy.engines import register
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@register()
class FastThreePIETorch(ThreePIETorch):
    """
    Multislice ptychography with closed-form exact per-slice line searches.

    Defaults:

    [name]
    default = FastThreePIETorch
    type = str
    help = Engine identifier

    [ls_damp]
    default = 0.5
    type = float
    help = Multiplier on every exact step
    doc = Same role as in LSISSTorch: the exact minimiser of each
          batch's cost is too greedy across overlapping views, and half
          the exact step is the classic under-relaxation for cyclic
          solvers.

    [max_step]
    default = 0.0
    type = float
    help = Cap on the magnitude of the line-search steps, 0 disables

    [probe_update_interval]
    default = 1
    type = int
    help = Batches between probe updates
    doc = 1 (probe line search every batch) measured decisively better in
          the Born engines at batch 1; kept as a knob for cadence tests.

    [response_mode]
    default = exact
    type = str
    help = How the per-slice far-field responses are built
    doc = 'exact': responses built sequentially with within-view
          incident-field maintenance -- slice j's step updates every
          downstream psi before slice j+1's response is formed (N^2/2
          propagations per view, launched one by one). 'wavefront': all
          N responses are built from the pre-sweep state and marched
          through the cascade as ONE stacked tensor -- stage k advances
          every response born at slice <= k together, then slice k+1's
          own perturbation joins the stack (N batched stages, one
          batched far field). The per-slice line searches still run
          sequentially against the exactly updated far field F + a D,
          so the update cadence (batch 1, probe every view) and the
          dominant cross-slice coupling at the detector are unchanged;
          only the within-view incident-field refresh is given up
          (second order in the applied steps). Response stack holds
          N*B*M frames -- intended for batch_size 1.
    """

    def engine_initialize(self):
        super().engine_initialize()
        if any(self.p.slice_start_iteration > 0):
            raise NotImplementedError(
                "FastThreePIETorch requires all slices active from iteration 0"
            )
        # 'max' under an exact line search reduces to the raw-gradient
        # direction (any scalar normalisation is absorbed by the step);
        # 'local' divides per pixel by the incident intensity. They differ
        # in direction SHAPE only, which is exactly what a preconditioner
        # A/B on this engine tests.
        self._ls_damp = float(self.p.ls_damp)
        self._ls_cap = float(self.p.max_step)
        self._pint = max(1, int(self.p.probe_update_interval))
        self._resp_mode = str(self.p.response_mode).lower()
        if self._resp_mode not in ("exact", "wavefront"):
            raise ValueError("response_mode must be 'exact' or 'wavefront'")
        self._nbatch = 0
        self._steps_o, self._steps_p = [], []
        logger.info("exact per-slice line search: damp %.3g, cap %s, "
                    "probe every %d batch(es), responses '%s'"
                    % (self._ls_damp,
                       self._ls_cap if self._ls_cap > 0 else "none",
                       self._pint, self._resp_mode))

    def _exact_step(self, e, v, w, omega, fallback, tag):
        """Exact quartic minimiser along one direction, damped and capped."""
        # float64: the first sweeps see residuals of ~1e10, so w^2 in the
        # cubic coefficients reaches ~1e40, past float32's 3.4e38
        e, v, w, omega = e.double(), v.double(), w.double(), omega.double()
        c0 = float((omega * e * v).sum())
        c1 = float((omega * (e * w + 2.0 * v * v)).sum())
        c2 = 3.0 * float((omega * v * w).sum())
        c3 = float((omega * w * w).sum())

        def Q(a):
            r = e + (2.0 * a) * v + (a * a) * w
            return float((omega * r * r).sum())

        a = solve_step_cubic(c0, c1, c2, c3, Q, fallback)
        a *= self._ls_damp
        if self._ls_cap > 0:
            a = float(np.clip(a, -self._ls_cap, self._ls_cap))
        (self._steps_o if tag == "o" else self._steps_p).append(a)
        return a

    def _batch_update(self, idx):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        M = self.M
        eps = torch.finfo(self.rdt).eps

        self._nbatch += 1
        do_pr = (self.curiter >= self.p.probe_update_start
                 and self._nbatch % self._pint == 0)

        # ---------------- forward cascade ----------------
        ob_p = [self._gather(self.ob_g[i], idx) for i in range(N)]  # (B, ny, nx)
        psis = []                       # incident field at each slice
        psi = self.pr_g[None].expand(B, M, self.fny, self.fnx).clone()
        for i in range(N):
            psis.append(psi)
            e_i = psi * ob_p[i][:, None]
            if i < N - 1:
                psi = self._apply(self._fw_g[i], e_i)
        exit_field = e_i

        F = self._apply(self._ff_fw_g, exit_field)          # (B, M, ny, nx)
        u = (F.real.square() + F.imag.square()).sum(dim=1)  # (B, ny, nx)

        am = self.amp_g[idx]
        mk = self.mask_g[idx]
        I_meas = self.I_g[idx]

        # ---------------- residual (parent's projection factor) ----------
        amod = torch.sqrt(u.clamp_min(0))
        fm = (1.0 - mk) + mk * am / (amod + 1e-10)
        r = self._apply(self._ff_bw_g, F * (fm - 1.0)[:, None].to(self.cdt))

        # ---------------- per-view errors, parent conventions -------------
        npix = float(self.fny * self.fnx)
        msum = mk.sum(dim=(-2, -1)).clamp_min(1.0)
        d_am = amod - am
        err_fmag = (mk * d_am * d_am).sum(dim=(-2, -1)) / msum
        err_exit = (r.abs() ** 2).mean(dim=(-2, -1)).sum(dim=1)
        if self.p.compute_log_likelihood:
            err_phot = (mk * (u - I_meas) ** 2 / (I_meas + 1.0)).sum(
                dim=(-2, -1)) / npix
        else:
            err_phot = torch.zeros_like(err_fmag)

        # ---------------- adjoint cascade: gradients per slice ------------
        phi2 = self._unscattered() if self._denom == "unscattered" else None
        grad_p, den_p = [None] * N, [None] * N
        chi = r                                           # at the exit plane
        for j in range(N - 1, -1, -1):
            grad_p[j] = (psis[j].conj() * chi).sum(dim=1)  # (B, ny, nx)
            den_p[j] = (phi2[j].expand(B, self.fny, self.fnx)
                        if phi2 is not None
                        else (psis[j].real.square()
                              + psis[j].imag.square()).sum(dim=1))
            chi_in = ob_p[j].conj()[:, None] * chi
            if j > 0:
                chi = self._apply(self._bw_g[j - 1], chi_in)
        grad_P = chi_in.sum(dim=0) if do_pr else None      # (M, ny, nx)

        # ---------------- object directions: preconditioned, on the canvas
        d_canvas, d_p = [], []
        for j in range(N):
            num = torch.zeros_like(self.ob_g[0])
            den = torch.zeros(self.ob_g[0].shape, dtype=self.rdt,
                              device=self.dev)
            self._scatter_add(num, idx, grad_p[j])
            self._scatter_add(den, idx, den_p[j])
            if self._denom == "max":
                dc = num / den.amax().clamp_min(eps).to(self.cdt)
            else:
                dn = (den + self._reg * den.amax()).clamp_min(eps)
                dc = num / dn.to(self.cdt)
            d_canvas.append(dc)
            d_p.append(self._gather(dc, idx))

        omega = mk / (I_meas + 1.0)
        e_res = u - I_meas

        if self._resp_mode == "wavefront":
            # ---------------- batched responses, sequential exact steps ----
            # all N responses from the pre-sweep state, marched through the
            # cascade as one stacked tensor: stage k advances every response
            # born at slice <= k together, then slice k+1's perturbation
            # joins the stack
            S = (d_p[0][:, None] * psis[0])[None]        # (1, B, M, ny, nx)
            for k in range(N - 1):
                S = self._apply(self._fw_g[k], S)
                S = S * ob_p[k + 1][None, :, None]
                S = torch.cat(
                    [S, (d_p[k + 1][:, None] * psis[k + 1])[None]], dim=0)
            D_all = self._apply(self._ff_fw_g, S)        # (N, B, M, ny, nx)
            del S

            # the line searches still run sequentially against the exactly
            # updated far field; only the within-view incident-field refresh
            # of 'exact' mode is given up
            for j in range(N):
                D = D_all[j]
                v = (F.real * D.real + F.imag * D.imag).sum(dim=1)
                w = (D.real.square() + D.imag.square()).sum(dim=1)
                a = self._exact_step(e_res, v, w, omega,
                                     fallback=self.ob_step, tag="o")
                self.ob_g[j] += a * d_canvas[j]
                ob_p[j] = ob_p[j] + a * d_p[j]
                F = F + a * D
                u = u + (2.0 * a) * v + (a * a) * w
                e_res = u - I_meas
            del D_all
        else:
            # ---------------- sequential exact per-slice steps -------------
            for j in range(N):
                # response of slice j through the remaining cascade, saving
                # the incident-field perturbations it creates downstream
                eps_f = d_p[j][:, None] * psis[j]          # exit perturbation
                incs = []
                for k in range(j, N - 1):
                    inc = self._apply(self._fw_g[k], eps_f)  # incident at k+1
                    incs.append(inc)
                    eps_f = inc * ob_p[k + 1][:, None]
                D = self._apply(self._ff_fw_g, eps_f)
                v = (F.real * D.real + F.imag * D.imag).sum(dim=1)
                w = (D.real.square() + D.imag.square()).sum(dim=1)

                a = self._exact_step(e_res, v, w, omega,
                                     fallback=self.ob_step, tag="o")

                self.ob_g[j] += a * d_canvas[j]
                ob_p[j] = ob_p[j] + a * d_p[j]
                # the far field and every downstream incident field move
                # exactly linearly: update them elementwise, no re-evaluation
                F = F + a * D
                u = u + (2.0 * a) * v + (a * a) * w
                e_res = u - I_meas
                for k, inc in enumerate(incs, start=j + 1):
                    psis[k] = psis[k] + a * inc

        # ---------------- probe: exact line search through the cascade ----
        if do_pr:
            K_P = torch.zeros((self.fny, self.fnx), dtype=self.rdt,
                              device=self.dev)
            K_P += (M * (ob_p[0].real.square()
                         + ob_p[0].imag.square())).sum(dim=0)
            if self._denom == "max":
                q = grad_P / K_P.amax().clamp_min(eps).to(self.cdt)
            else:
                dn = (K_P + self._reg * K_P.amax()).clamp_min(eps)
                q = grad_P / dn.to(self.cdt)

            eps_f = q[None].expand(B, M, self.fny, self.fnx) * 1.0
            for k in range(N):
                eps_f = eps_f * ob_p[k][:, None]
                if k < N - 1:
                    eps_f = self._apply(self._fw_g[k], eps_f)
            D_P = self._apply(self._ff_fw_g, eps_f)
            v_p = (F.real * D_P.real + F.imag * D_P.imag).sum(dim=1)
            w_p = (D_P.real.square() + D_P.imag.square()).sum(dim=1)

            b = self._exact_step(e_res, v_p, w_p, omega,
                                 fallback=self.pr_step, tag="p")
            self.pr_g = self.pr_g + b * q
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
            msg = ("exact steps: slice median %.3g (p5 %.3g, p95 %.3g)"
                   % (np.median(so), np.percentile(so, 5),
                      np.percentile(so, 95)))
            if self._steps_p:
                sp = np.asarray(self._steps_p[-self.n_pos:])
                msg += ", probe median %.3g" % np.median(sp)
            logger.info(msg)
        return out
