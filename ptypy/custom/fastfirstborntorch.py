"""
First Born multislice with exact line searches, exploiting the affine model.

WHY THIS CAN EXIST
------------------
The first Born exit field is AFFINE in the slices and LINEAR in the probe:

    psi(g + a d, P) = psi(g, P) + a * dpsi_o        (dpsi_o independent of a)
    psi(g, P + b q) = psi(g, P) + b * dpsi_p        (dpsi_p independent of b)

The second derivative of the field with respect to the object is identically
zero, so along any update direction the far field moves on a straight line,

    F(a) = F + a D,   I(a) = |F(a)|^2 = u + 2 a v + a^2 w   (per pixel),

and the Gaussian intensity cost

    Q(a) = sum  mask * (I(a) - I_dat)^2 / (I_dat + 1)

is an exact QUARTIC polynomial in the scalar a. Its minimiser is a root of a
cubic whose four coefficients are plain masked sums -- an exact, global,
closed-form line search costing one extra forward half-pass (for D) and zero
trial evaluations. The same holds for the probe, whose unscattered term
enters the field through the identity (the "direct path"): the probe
direction's response is q plus its scattered echo, and its step is the root
of another cubic.

Better still, because the object step moves the far field EXACTLY along D,
the field after the object update is F + a D elementwise -- no re-evaluation
-- so the probe line search inside the same batch runs against the already
updated field. The bilinear object-probe cross term is thereby handled
exactly, which no fixed-step simultaneous update does.

WHAT CHANGES AGAINST FirstBornTorch
-----------------------------------
Directions are the same preconditioned gradients as the parent's; only the
step sizes differ: alpha/N and beta/N are replaced by the exact minimisers.
There is nothing to tune -- the parent's alpha and beta are used only as a
fallback when a cubic degenerates (zero direction, fully masked batch).
`momentum` mixes the previous displacement into the direction BEFORE the
line search (heavy ball with an exact step, close to conjugate gradients).
Cost per batch: one forward+adjoint (as the parent) plus one forward
half-pass per line search, i.e. about 1.5x the parent's transforms for the
object step, 2x when the probe updates too.
"""

import numpy as np

from ptypy.custom.firstborntorch import FirstBornTorch
from ptypy.engines import register
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def solve_step_cubic(c0, c1, c2, c3, Qfun, fallback):
    """Minimise the quartic Q along the line: dQ/da = 4(c0 + c1 a + c2 a^2
    + c3 a^3) = 0. Returns the real root with the lowest Q; `fallback` when
    the cubic is degenerate or no root improves on a = 0."""
    if not np.all(np.isfinite([c0, c1, c2, c3])):
        return fallback
    coeffs = [c3, c2, c1, c0]
    while coeffs and abs(coeffs[0]) < 1e-300:
        coeffs = coeffs[1:]
    if len(coeffs) < 2:
        return fallback
    roots = np.roots(coeffs)
    cands = [float(r.real) for r in roots
             if abs(r.imag) <= 1e-8 * (1.0 + abs(r.real))]
    if not cands:
        return fallback
    q0 = Qfun(0.0)
    best, bestq = None, q0
    for a in cands:
        qa = Qfun(a)
        if qa < bestq:
            best, bestq = a, qa
    return fallback if best is None else best


@register()
class FastFirstBornTorch(FirstBornTorch):
    """
    First Born multislice with closed-form exact line searches.

    Defaults:

    [name]
    default = FastFirstBornTorch
    type = str
    help = Engine identifier

    [per_slice_ls]
    default = False
    type = bool
    help = One exact line search per slice instead of one global step
    doc = A single global step is dominated by the best-illuminated layer:
          the entrance layers, whose beam footprint covers the fewest
          positions, under-converge. Sequential per-slice quartic steps
          (field maintained exactly F <- F + a_j D_j between them, as in
          FastThreePIE) give each layer its own optimal step at the cost of
          one extra direction transform per slice.

    [ls_damp]
    default = 0.5
    type = float
    help = Multiplier on every exact step
    doc = 1.0 takes the exact minimiser of EACH BATCH's cost, which is too
          greedy across overlapping batches: neighbouring views partially
          undo each other and the run plateaus an order of magnitude above
          where it should (measured: stuck at 1.5-1.8 vs 0.42 and falling
          with 0.5, same 15 iterations). Half the exact step is the classic
          under-relaxation for cyclic solvers and is the default.

    [max_step]
    default = 0.0
    type = float
    help = Cap on the magnitude of the line-search steps, 0 disables
    doc = A safety rail only; the exact step needs no cap on any data seen
          so far.

    [probe_denom]
    default = max
    type = str
    help = Denominator for the probe update: 'max' (global) or 'local' (per-pixel)
    doc = Decoupled from object_denom because the object and probe want
          different preconditioners. 'local' (per-pixel K_P) is right for the
          object, but on the PROBE it reshapes the update direction by 1/|O|^2
          per pixel and amplifies low-transmission pixels into a central hot
          spike -- benign for a broad low-NA probe but a runaway divergence
          for a tight high-NA one (measured: peak/mean 1268 at iter 2 climbing
          to 3719 by iter 8, truth 595). 'max' (a single global scale, whose
          value the exact line search absorbs anyway) keeps the direction
          shape intact and the probe bounded at every NA (measured: peak/mean
          settles 623->232 over 15 iters, no spike), so it is the default.

    [probe_fourier_frac]
    default = 0.0
    type = float
    help = Bandwidth-limit the probe to this fraction of Nyquist, 0 disables
    doc = A physical aperture constraint. The illumination cannot carry
          spatial frequencies beyond the focusing aperture, so the probe's
          Fourier support is compact (the simulated high-NA probe holds
          99.95%% of its energy within 0.75 of Nyquist). A single-pixel
          real-space spike is by contrast broadband -- flat across all
          frequencies -- so re-projecting the probe onto a circular Fourier
          mask each update removes the spike's out-of-band energy while
          leaving the genuine multi-lobe structure untouched. This cures the
          high-NA central-hot-pixel divergence that neither the denominator
          choice nor real-space support can (the spike sits at the probe
          centre, inside any real-space support). 0.75-0.9 keeps all real
          structure; leave 0 for well-sampled low-NA probes.

    [probe_modulus_constraint]
    default = False
    type = bool
    help = Project the probe onto |FF{P}| = sqrt(<I>) each probe update, keep phase
    doc = The bilinear First Born model makes the average diffraction pattern
          <I> = mean_j |FF{exit_j}|^2 measure the probe far-field amplitude
          directly (object cross-terms average out), the same quantity the
          'stxm' illumination init back-propagates. A good init alone does not
          hold: under joint probe+object updates the probe far-field amplitude
          drifts off <I> and slowly over-concentrates into a spike (measured:
          stxm init peak/mean 8 climbing past truth 24.6 to 60 by iter 40,
          proj corr collapses). Re-imposing |FF{P}| = sqrt(<I>) every update --
          replacing the mode-summed far-field amplitude with the measured
          envelope while keeping the probe phase free -- pins the envelope and
          the total power (Parseval) so only the recoverable phase evolves.
          Requires the average envelope; computed once in engine_initialize.

    [probe_power_norm]
    default = False
    type = bool
    help = Renormalise probe power to the measured per-frame flux each probe update
    doc = Lighter-weight gauge fix than probe_modulus_constraint: rescales the
          probe so its total power equals the mean per-frame flux, pinning the
          probe/object scaling gauge without touching the amplitude shape.
          Subsumed by probe_modulus_constraint (which also fixes total power);
          use alone when only the scale, not the envelope, runs away.

    [probe_realspace_amp]
    default = False
    type = bool
    help = Lock the probe real-space amplitude |P| to the init each probe update
    doc = The probe_modulus_constraint pins the FAR-FIELD amplitude but leaves
          the far-field phase free, so the probe can still refocus (flatten its
          defocus phase) and collapse to a real-space spike -- the probe/object
          gauge ambiguity of the Born model (measured: modulus alone still
          climbs to peak/mean 113, proj corr ~0). Combining it with a REAL-SPACE
          amplitude lock |P| = |P_init| (keeping the real-space phase free) is a
          two-plane Fienup projection: the probe amplitude is fixed in both
          domains from the data/geometry (far-field <I>, real-space stxm+defocus
          envelope) and only the mutually-consistent aberration phase evolves,
          so the probe cannot concentrate. Requires a good |P_init|; use with
          the stxm+defocus init.

    [probe_modulus_selfcons]
    default = False
    type = bool
    help = Debias the modulus target by the scattered power: |FF{P}|=sqrt(<I>-<|S|^2>)
    doc = sqrt(<I>) overestimates the probe far-field amplitude because the
          measured mean <I> = |FF{P}|^2 + <|S_j|^2> + 2Re(U*<S_j>) carries the
          incoherent scattered power <|S_j|^2> of the object (S_j = FF{P.(T-1)},
          the scattered field; U = FF{P} the unscattered direct beam). Requires
          probe_modulus_constraint. When on, the scattered power is accumulated
          from the model far field each epoch (|F - U|^2 mode-summed) and the
          modulus target is recomputed as sqrt(max(<I> - <|S|^2>, 0)); as the
          object converges the target sharpens toward the true |FF{P}|, letting
          the probe amplitude climb past the biased ceiling. No-op until enough
          object structure exists; the fixed sqrt(<I>) is the first-epoch target.
    """

    def engine_initialize(self):
        super().engine_initialize()
        # own unscattered-illumination cache, (N, M, ny, nx); dropped on
        # every probe change. Independent of the parent's core-shaped cache.
        self._phi_f = None
        # previous displacement per slice canvas, for the heavy ball
        self._disp = None
        if float(self.p.momentum) > 0:
            self._disp = torch.zeros_like(self.ob_g)
        self._ls_damp = float(self.p.ls_damp)
        self._ls_cap = float(self.p.max_step)
        # probe denominator is decoupled from object_denom: 'local' is right
        # for the object but destabilises a tight high-NA probe (hot pixels)
        self._use_max_probe = str(self.p.probe_denom).lower().startswith("m")
        # optional circular Fourier (aperture) constraint on the probe, built
        # lazily from the probe frame size; kills broadband real-space spikes
        self._pr_ffrac = float(self.p.probe_fourier_frac)
        self._pr_fmask = None
        # Fourier-modulus / power gauge fixes (bilinear-Born tricks). The
        # average diffraction sqrt(<I>) measures the probe far-field amplitude;
        # pin it (or at least the total power) each update so the probe cannot
        # drift off the data envelope into a spike. amp_dat is (n_pos,fny,fnx).
        self._pr_modulus = bool(self.p.probe_modulus_constraint)
        self._pr_pownorm = bool(self.p.probe_power_norm)
        self._pr_rsamp = bool(self.p.probe_realspace_amp)
        self._pr_selfcons = bool(getattr(self.p, "probe_modulus_selfcons", False))
        self._pr_env = None
        self._pr_rs_env = None
        self._pr_target_power = 0.0
        # per-epoch scattered-power accumulator for the self-consistent modulus
        self._pr_Idat_mean = None
        self._pr_S2_acc = None
        self._pr_S2_n = 0
        if self._pr_modulus:
            # sqrt(<I>): the mean measured intensity envelope, (fny, fnx)
            self._pr_Idat_mean = self.amp_dat.square().mean(dim=0).clamp_min(0)
            self._pr_env = self._pr_Idat_mean.sqrt()
            if self._pr_selfcons:
                self._pr_S2_acc = torch.zeros_like(self._pr_env)
        if self._pr_rsamp:
            # |P_init|: the real-space amplitude of the (stxm+defocus) init, held
            # fixed so the probe cannot refocus. Captured before any update.
            self._pr_rs_env = self.pr_g.abs().clone()
        if self._pr_pownorm:
            # mean per-frame far-field flux -> target real-space probe power
            # via Parseval on the same far-field pair used for the modulus
            self._pr_target_power = float(self.amp_dat.square().sum()) / self.n_pos
        self._steps_o, self._steps_p = [], []
        logger.info("exact line search: damp %.3g, cap %s"
                    % (self._ls_damp,
                       self._ls_cap if self._ls_cap > 0 else "none"))

    # ------------------------------------------------------------------ #
    #  Pieces of the affine model                                          #
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
        """Model-perturbation direction for a parameter-space direction.

        The chord stores the perturbation itself (dg/dO = 1), so this is the
        identity. The tangent subclass stores the potential s with g = i s
        and overrides this with i d; the field stays affine along the step
        either way, so the same exact quartic line search applies.
        """
        return d

    def _line_search(self, e, v, w, omega, fallback, tag):
        """Exact step along a direction whose far-field response gives the
        per-pixel linear (v) and quadratic (w) intensity coefficients."""
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

    def _apply_probe_fourier(self):
        """Re-project the probe onto its physical aperture: zero all Fourier
        content beyond probe_fourier_frac * Nyquist. A no-op when disabled."""
        if self._pr_ffrac <= 0:
            return
        if self._pr_fmask is None:
            ny, nx = self.pr_g.shape[-2:]
            fy = torch.fft.fftfreq(ny, device=self.pr_g.device, dtype=self.rdt)
            fx = torch.fft.fftfreq(nx, device=self.pr_g.device, dtype=self.rdt)
            rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
            self._pr_fmask = (rr <= 0.5 * self._pr_ffrac).to(self.cdt)
        self.pr_g = torch.fft.ifft2(torch.fft.fft2(self.pr_g) * self._pr_fmask)

    def _apply_probe_modulus(self):
        """Replace the probe's mode-summed far-field amplitude with the
        measured envelope sqrt(<I>), keeping phase. Uses the detector far-field
        pair (self._ff_fw/_ff_bw) so it matches amp_dat. A no-op when disabled."""
        if not self._pr_modulus:
            return
        F = self._apply(self._ff_fw, self.pr_g)              # (M, fny, fnx)
        amp = (F.real.square() + F.imag.square()).sum(0).clamp_min(1e-20).sqrt()
        scale = (self._pr_env / amp).to(self.cdt)            # (fny, fnx)
        self.pr_g = self._apply(self._ff_bw, F * scale.unsqueeze(0))

    def _apply_probe_rsamp(self):
        """Lock the probe real-space amplitude to the init |P_init|, keeping the
        real-space phase. With the far-field modulus this is a two-plane Fienup
        projection that stops the probe refocusing. A no-op when disabled."""
        if not self._pr_rsamp:
            return
        ph = self.pr_g / self.pr_g.abs().clamp_min(1e-20)
        self.pr_g = self._pr_rs_env * ph

    def _normalize_probe_power(self):
        """Rescale the probe so its far-field power equals the mean per-frame
        flux -- pins the probe/object scaling gauge. A no-op when disabled."""
        if self._pr_target_power <= 0:
            return
        F = self._apply(self._ff_fw, self.pr_g)
        pf = float((F.real.square() + F.imag.square()).sum())
        if pf > 0:
            self.pr_g = self.pr_g * (self._pr_target_power / pf) ** 0.5

    # ------------------------------------------------------------------ #
    #  The batch update                                                    #
    # ------------------------------------------------------------------ #
    def _batch_update(self, idx):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        flat = self._flat(idx)
        eps = torch.finfo(self.rdt).eps
        reg = float(self.p.denom_reg)

        O = torch.stack([self._gather(self.ob_g[j], flat, B) for j in range(N)])
        O = self._model_object(O)                              # (N, B, ny, nx)
        g = O - 1.0
        amp_dat = self.amp_dat[idx]
        I_dat = amp_dat.square()
        mask = self.maskb[idx].to(self.rdt) if self.maskb is not None else None

        self._nbatch += 1
        do_probe = (not self._probe_frozen
                    and self.curiter >= self.p.probe_update_start
                    and self._nbatch % self._pint == 0)

        phi = self._phi_fast()

        # ---- forward ----------------------------------------------------
        psi = self.pr_g.unsqueeze(0) + self._scattered(
            g[j].unsqueeze(1) * phi[j] for j in range(N))
        F = self._apply(self._ff_fw, psi)                      # (B, M, ny, nx)
        u = torch.sum(F.real.square() + F.imag.square(), dim=1)

        # self-consistent modulus: accumulate the scattered power <|S|^2> from
        # the model far field, S = F - U with U = FF{P} the unscattered beam.
        if self._pr_selfcons and self._pr_S2_acc is not None:
            with torch.no_grad():
                U = self._apply(self._ff_fw, self.pr_g)        # (M, fny, fnx)
                S = F - U.unsqueeze(0)
                self._pr_S2_acc += (S.real.square()
                                    + S.imag.square()).sum(dim=1).sum(dim=0)
                self._pr_S2_n += B

        # ---- adjoint: same residual and gradients as the parent ----------
        fac = torch.sqrt(I_dat / u.clamp_min(1e-12)) - 1.0
        if mask is not None:
            fac = torch.where(mask > 0, fac, torch.zeros_like(fac))
        r = self._apply(self._ff_bw, F * fac.unsqueeze(1).to(self.cdt))
        r_f = torch.fft.fft2(r)

        self._acc_grad.zero_()
        self._acc_K.zero_()
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

        # ---- errors, parent conventions ----------------------------------
        amp_mod = u.clamp_min(0).sqrt()
        m_weight = 1.0 if mask is None else mask
        m_sum = (float(B * self.fny * self.fnx) if mask is None
                 else mask.sum().clamp_min(1))
        e0 = (m_weight * (amp_mod - amp_dat).square()).sum() / m_sum
        e1 = (m_weight * (u - I_dat).square() / (I_dat + 1.0)).sum() / m_sum
        self._err.add_(torch.stack([e0, e1, r.abs().sum()]).to(torch.float64))

        # ---- object direction: preconditioned gradient --------------------
        peak_K = self._acc_K.amax(dim=(-2, -1), keepdim=True)
        dn_o = peak_K if self._use_max else self._acc_K + peak_K * reg
        d = self._grad_to_param(self._acc_grad / dn_o.clamp_min(eps))
        if self._disp is not None:
            d = d + float(self.p.momentum) * self._disp

        # ---- object line search -------------------------------------------
        d_p = torch.stack([self._gather(d[j], flat, B) for j in range(N)])
        # the far field responds to the MODEL perturbation: for the chord the
        # parameter is the perturbation (identity), for the tangent dg = i ds
        dg_p = self._dir_to_model(d_p)
        omega = (1.0 / (I_dat + 1.0)) if mask is None else mask / (I_dat + 1.0)

        if bool(self.p.per_slice_ls):
            # sequential exact steps, one per slice; the field is maintained
            # exactly between them, so each solve sees the previous updates
            a_vec = torch.zeros(N, dtype=self.rdt, device=self.dev)
            for j in range(N):
                Dj = self._apply(self._ff_fw, torch.fft.ifft2(
                    self.H_rev[j, 0]
                    * torch.fft.fft2(dg_p[j].unsqueeze(1) * phi[j])))
                e = u - I_dat
                v = (F.real * Dj.real + F.imag * Dj.imag).sum(dim=1)
                w = torch.sum(Dj.real.square() + Dj.imag.square(), dim=1)
                a_j = self._line_search(e, v, w, omega,
                                        fallback=float(self.p.alpha) / N,
                                        tag="o")
                F = F + a_j * Dj
                u = u + (2.0 * a_j) * v + (a_j * a_j) * w
                self.ob_g[j] += a_j * d[j]
                a_vec[j] = a_j
                del Dj
            e = u - I_dat
            if self._disp is not None:
                self._disp = a_vec[:, None, None] * d
            g2 = g + a_vec[:, None, None, None] * dg_p
        else:
            D = self._apply(self._ff_fw, self._scattered(
                dg_p[j].unsqueeze(1) * phi[j] for j in range(N)))
            v = (F.real * D.real + F.imag * D.imag).sum(dim=1)
            w = torch.sum(D.real.square() + D.imag.square(), dim=1)
            e = u - I_dat
            a = self._line_search(e, v, w, omega,
                                  fallback=float(self.p.alpha) / N, tag="o")
            self.ob_g += a * d
            if self._disp is not None:
                self._disp = a * d
            # after the object step the far field is EXACTLY F + a D
            F = F + a * D
            u = u + (2.0 * a) * v + (a * a) * w
            e = u - I_dat
            # the perturbation moves by the MODEL image of the parameter
            # step: a * d for the chord, i a * d for the tangent
            g2 = g + a * dg_p

        # ---- probe: exact update of the field, then its own line search ---
        if do_probe:

            grad_P = (r + torch.fft.ifft2(acc_P)).sum(dim=0)   # (M, ny, nx)
            K_P = torch.sum(O.real.square() + O.imag.square(), dim=(0, 1))
            peak_P = K_P.amax()
            dn_p = peak_P if self._use_max_probe else K_P + peak_P * reg
            q = grad_P / dn_p.clamp_min(eps)

            qz = [torch.fft.ifft2(self.H_fwd[j, 0] * torch.fft.fft2(q))
                  for j in range(N)]
            dpsi_p = q.unsqueeze(0) + self._scattered(
                g2[j].unsqueeze(1) * qz[j] for j in range(N))
            D_P = self._apply(self._ff_fw, dpsi_p)
            v_p = (F.real * D_P.real + F.imag * D_P.imag).sum(dim=1)
            w_p = torch.sum(D_P.real.square() + D_P.imag.square(), dim=1)

            b = self._line_search(e, v_p, w_p, omega,
                                  fallback=float(self.p.beta) / N, tag="p")
            self.pr_g = self.pr_g + b * q
            if self.pr_support is not None:
                self.pr_g = self.pr_g * self.pr_support
            self._apply_probe_fourier()
            # bilinear-Born gauge fixes: pin the far-field amplitude to the
            # measured envelope (modulus), then the real-space amplitude to the
            # init (two-plane Fienup: stops refocusing), and/or power to flux.
            self._apply_probe_modulus()
            self._apply_probe_rsamp()
            self._normalize_probe_power()
            self._phi_f = None

    def engine_iterate(self, num=1):
        out = super().engine_iterate(num)
        # self-consistent modulus: debias the target by the epoch's mean
        # scattered power, |FF{P}| = sqrt(max(<I> - <|S|^2>, 0)), then reset.
        if (self._pr_selfcons and self._pr_S2_acc is not None
                and self._pr_S2_n > 0):
            with torch.no_grad():
                s2_mean = self._pr_S2_acc / self._pr_S2_n
                self._pr_env = (self._pr_Idat_mean - s2_mean).clamp_min(0).sqrt()
                frac = float(s2_mean.sum()
                             / self._pr_Idat_mean.sum().clamp_min(1e-30))
                logger.info("self-cons modulus: scattered power fraction %.3f"
                            % frac)
                self._pr_S2_acc.zero_()
                self._pr_S2_n = 0
        if self._steps_o:
            so = np.asarray(self._steps_o[-self.n_pos:])
            msg = "exact steps: object median %.3g (p5 %.3g, p95 %.3g)" % (
                np.median(so), np.percentile(so, 5), np.percentile(so, 95))
            if self._steps_p:
                sp = np.asarray(self._steps_p[-self.n_pos:])
                msg += ", probe median %.3g" % np.median(sp)
            logger.info(msg)
        # boundedness watchdog: peak/mean of the probe intensity. A healthy
        # probe holds near the truth value; runaway hot pixels send it up.
        with torch.no_grad():
            I = (self.pr_g.real.square() + self.pr_g.imag.square()).sum(dim=0)
            logger.info("probe intensity peak/mean %.1f (max|P| %.3g)"
                        % (float(I.max() / I.mean().clamp_min(1e-30)),
                           float(I.max().sqrt())))
        return out


@register()
class FastFirstBornTangentTorch(FastFirstBornTorch):
    """
    Exact-line-search first Born with the tangent (linearised) perturbation.

    Same relationship to FastFirstBornTorch as FirstBornTangentTorch has to
    FirstBornTorch: the reconstructed unknown is the scattering potential s
    with g = i s, the gradient direction picks up the -i chain-rule factor
    (projected onto the reals when potential_real), and the line search runs
    along the MODEL direction i d via _dir_to_model. Because the field is
    affine along any fixed direction -- including a real s-direction mapped
    through g = i s -- the intensity cost stays exactly quartic in the step
    and the same closed-form solve applies, now restricted to the constraint
    manifold when potential_real is set.

    Defaults:

    [name]
    default = FastFirstBornTangentTorch
    type = str
    help = Engine identifier

    [potential_real]
    default = False
    type = bool
    help = Constrain the potential s to be real, i.e. a pure phase object
    doc = The strict weak-phase-object convention, as in
          FirstBornTangentTorch: g = i s with s real forbids absorption and
          locks the scattered wave in quadrature with the transmitted beam.
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
        # dL/ds* = -i dL/dg*; for real s keep the real part, the Wirtinger
        # descent direction on the reals
        g = -1j * grad
        return g.real.to(self.cdt) if bool(self.p.potential_real) else g

    def _dir_to_model(self, d):
        # dg/ds = i: a parameter step a d moves the perturbation by i a d
        return 1j * d

    def _display_object(self):
        return torch.exp(1j * self.ob_g)

    def engine_finalize(self):
        # hand transmissions, not potentials, to the containers and file
        self.ob_g = torch.exp(1j * self.ob_g)
        return super().engine_finalize()
