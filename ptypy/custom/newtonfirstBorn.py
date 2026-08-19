# -*- coding: utf-8 -*-
"""
First Born multislice engine for ptychography (GPU, PyTorch).

FORWARD MODEL
-------------
    s_j   = O_j - 1                                 scattering perturbation
    phi_j = F^-1[ H_j F[P] ]                        illumination at slice j
    psi   = P + sum_j F^-1[ Hbar_j F[ s_j phi_j ] ]

with H_j = exp(2i pi (z_j/lam)(sqrt(1 - (lam f)^2) - 1)) and H_0 = 1, the whole
model referred to the entrance plane. Slices enter as a SUM, not a product, and
phi_j carries no object.

WHAT LINEARITY BUYS
-------------------
psi is exactly linear in {s_j} and, separately, exactly linear in P. The only
nonlinearity is the modulus at the detector: L = || |F psi| - sqrt(I) ||^2.
Linearity up to the detector means that for ANY set of directions d_i,

    F( psi + sum_i t_i d_i ) = F psi + sum_i t_i U_i ,     U_i = F(A d_i)

holds to machine precision. So k applications of the operator buy the loss, its
gradient AND its Hessian with respect to t at any point, for the cost of
elementwise reductions:

    a   = sqrt( sum_m |G|^2 ),   G = F psi + sum_i t_i U_i
    R_i = Re sum_m conj(G) U_i
    dL/dt_i    = 2 sum mask (1 - a_meas/a) R_i
    d2L/dt_idj = 2 sum mask [ (a_meas/a^3) R_i R_j
                              + (1 - a_meas/a) Re sum_m conj(U_i) U_j ]

This is the exact Hessian of the REAL objective restricted to the subspace, not
of a linearised surrogate, so the reduced problem can be solved by damped Newton
rather than by backtracking a single step length. Multislice cannot do any of
this: one trial step there costs a full nonlinear cascade.

CHOOSING THE SUBSPACE
---------------------
A_j^dag A_j = M_{|phi_j|^2} exactly, because the propagator is unitary, so the
per-slice preconditioner is exact. What is NOT diagonal is the cross term
A_j^dag A_k, and that is the only reason a fixed step needs step < 2/N.

Giving each slice its own coefficient puts those cross terms inside the
(N+1)x(N+1) reduced system, where they are solved exactly. The inter-slice
coupling stops being a stability constraint and becomes part of the step. This
is subspace = 'slices', and it is the setting the model structure argues for.
Adding the previous accepted increment as one more direction ('slices+prev')
turns the method into subspace-accelerated Newton and discovers its own momentum
coefficient with nothing to tune.

COST
----
Per batch, on top of one forward, one projection and one adjoint: k operator
applications and k far-field transforms, where k = 2 for 'joint', N+1 for
'slices' and N+2 for 'slices+prev'. The reduced solve itself is k x k and free.
Peak memory grows by k*B*M*ny*nx complex numbers; halve batch_size if that bites.
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
    data, so a convention mismatch is a startup error rather than a wrong
    reconstruction.
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
class NewtonFirstBorn(stochastic.EPIE):
    """
    First Born multislice ptychography with a subspace Newton step.

    Defaults:

    [name]
    default = NewtonFirstBorn
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

    [subspace]
    default = slices
    type = str
    help = 'joint', 'slices' or 'slices+prev'
    doc = 'joint' uses one combined object direction plus the probe (k = 2).
          'slices' gives every slice its own coefficient plus the probe
          (k = N+1), which puts the inter-slice cross terms A_j^dag A_k inside
          the reduced system where they are solved exactly, so no 1/N damping is
          needed. 'slices+prev' adds the previous accepted increment (k = N+2),
          giving subspace acceleration with no momentum parameter to tune.

    [newton_iters]
    default = 8
    type = int
    help = Damped Newton iterations on the reduced problem
    doc = Each costs only elementwise reductions, no transforms.

    [newton_damping]
    default = 0.005
    type = float
    help = Levenberg damping, as a fraction of the Hessian diagonal
    doc = The modulus makes the reduced Hessian possibly indefinite, and the
          object/probe scale gauge makes it near-singular, since O -> O(1+e),
          P -> P(1-e) leaves psi unchanged to first order. Damping is raised
          automatically if a Newton iterate fails to decrease the loss.

    [max_backtrack]
    default = 8
    type = int
    help = Halvings allowed before a batch takes no step

    [alpha]
    default = 1.0
    type = float
    help = Object step
    doc = Damping on the Newton step for 'newton' (1 = as solved); a fixed step
          divided by number_of_slices for 'epie'.

    [beta]
    default = 1.0
    type = float
    help = Probe step
    doc = Interpreted as alpha is, for the probe.

    [object_denom]
    default = local
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

    [device]
    default = cuda
    type = str
    help = Torch device: 'cuda', 'cuda:0' or 'cpu'

    [batch_size]
    default = 64
    type = int
    help = Scan positions per update

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)
    """

    def __init__(self, ptycho_parent, pars=None):
        super(NewtonFirstBorn, self).__init__(ptycho_parent, pars)
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
            raise ImportError("NewtonFirstBorn requires PyTorch")

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

        self._object = [self.ob.copy(self.ob.ID + "_o_%d" % i) for i in range(N)]

        rng = np.random.default_rng()
        ob0 = np.ones_like(self._obs.data[0])
        self.ob_g = torch.stack([
            torch.as_tensor(ob0 + 1e-3 * ((rng.random(ob0.shape) - 0.5)
                                          + 1j * (rng.random(ob0.shape) - 0.5)),
                            dtype=self.cdt, device=self.dev) for _ in range(N)])
        self.pr_g = torch.as_tensor(np.ascontiguousarray(self._prs.data),
                                    dtype=self.cdt, device=self.dev)
        self.ony, self.onx = int(self.ob_g.shape[-2]), int(self.ob_g.shape[-1])

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

        self._phi_cache = None
        self._acc_n = torch.zeros((N, self.ony, self.onx), dtype=self.cdt, device=self.dev)
        self._acc_d = torch.zeros((N, self.ony, self.onx), dtype=self.rdt, device=self.dev)
        self._prev = None                       # previous accepted object increment
        self._pr_ref = float((self.pr_g.abs() ** 2).sum())
        self._lam = float(self.p.newton_damping)

        logger.info("subspace = %s (k = %d), batch = %d, denom = %s"
                    % (self.p.subspace, self._k(),
                       self.p.batch_size, self.p.object_denom))

    def _k(self):
        N = self.p.number_of_slices
        sub = str(self.p.subspace).lower()
        if sub.startswith('j'):
            return 2
        return N + 2 if 'prev' in sub else N + 1

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

        if (int(self.rows.max()) >= self.ony or int(self.cols.max()) >= self.onx
                or int(self.rows.min()) < 0 or int(self.cols.min()) < 0):
            raise RuntimeError("patch extends outside the object array")

        cov = torch.zeros((self.ony, self.onx), dtype=self.rdt, device=self.dev)
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
    #  Linear operator                                                    #
    # ------------------------------------------------------------------ #
    def _phi(self):
        """Unscattered illumination at every slice, (N, M, ny, nx).

        phi_0 = P exactly since H_0 = 1, and phi carries no object, so it is
        neither position- nor batch-dependent and is cached until P moves.
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

    def _A_slices(self, sd, phi):
        """A on a full slice-space direction: sum_j P_{-j}( sd_j phi_j ).

        sd is (N, B, ny, nx); returns (B, M, ny, nx). Slice 0 needs no transform.
        """
        out = sd[0][:, None] * phi[0][None]
        if sd.shape[0] > 1:
            Fs = torch.fft.fft2(sd[1:, :, None] * phi[1:, None])
            out = out + torch.fft.ifft2((self.Hb[:, None, None] * Fs).sum(0))
        return out

    def _A_one_slice(self, j, dj, phi):
        """A on a direction supported on slice j alone: P_{-j}( dj phi_j )."""
        x = dj[:, None] * phi[j][None]
        if j == 0:
            return x                                    # H_0 = 1
        return torch.fft.ifft2(self.Hb[j - 1] * torch.fft.fft2(x))

    def _A_probe(self, dP, s):
        """A on a probe direction: dP + sum_j P_{-j}( s_j P_j dP ).

        psi is linear in P as well, so the probe direction has an exact forward
        model too.
        """
        N = self.p.number_of_slices
        pd = torch.empty((N, self.M, self.fny, self.fnx), dtype=self.cdt, device=self.dev)
        pd[0] = dP
        if N > 1:
            pd[1:] = torch.fft.ifft2(self.H[:, None] * torch.fft.fft2(dP))
        return self._A_slices(s, pd) + dP[None]

    # ------------------------------------------------------------------ #
    #  Reduced Newton                                                     #
    # ------------------------------------------------------------------ #
    def _reduced_newton(self, F, U, am, mk):
        """
        Damped Newton on L(t) = sum mask ( |F + sum_i t_i U_i| - a_meas )^2.

        Exact because psi is linear up to the detector, so G(t) = F + sum t_i U_i
        is exact and the loss, its gradient and its Hessian in t are elementwise
        reductions. Verified against finite differences.

        Returns the coefficient vector, or zeros if no decrease was found.
        """
        k = len(U)
        eps = 1e-12
        t = np.zeros(k)

        def state(tv):
            G = F.clone()
            for i in range(k):
                if tv[i] != 0.0:
                    G = G + float(tv[i]) * U[i]
            a = torch.sqrt((G.real ** 2 + G.imag ** 2).sum(dim=1).clamp_min(eps))
            return G, a

        def loss_from(a):
            return float((mk * (a - am) ** 2).sum())

        G0, a0 = state(t)
        L0 = loss_from(a0)
        best_t, best_L = t.copy(), L0
        lam = self._lam

        for _ in range(int(self.p.newton_iters)):
            G, a = state(t)
            w1 = mk * (1.0 - am / a)
            w2 = mk * am / a ** 3
            R = [(G.conj() * U[i]).real.sum(dim=1) for i in range(k)]   # (B,ny,nx)

            g = np.array([2.0 * float((w1 * R[i]).sum()) for i in range(k)])
            Hm = np.zeros((k, k))
            for i in range(k):
                for j in range(i, k):
                    S = (U[i].conj() * U[j]).real.sum(dim=1)
                    v = 2.0 * (float((w2 * R[i] * R[j]).sum()) + float((w1 * S).sum()))
                    Hm[i, j] = Hm[j, i] = v

            # Levenberg damping: the modulus can make Hm indefinite and the
            # object/probe scale gauge makes it near-singular
            d = np.maximum(np.abs(np.diag(Hm)), 1e-30)
            for _try in range(6):
                try:
                    dt = np.linalg.solve(Hm + lam * np.diag(d), -g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                cand = t + dt
                _, ac = state(cand)
                Lc = loss_from(ac)
                if np.isfinite(Lc) and Lc < best_L:
                    t, best_t, best_L = cand, cand.copy(), Lc
                    lam = max(lam * 0.5, 1e-6)
                    break
                lam *= 10.0
            else:
                break

        if best_L >= L0:
            # no Newton iterate helped; fall back to backtracked steepest
            # descent inside the same subspace
            G, a = state(np.zeros(k))
            w1 = mk * (1.0 - am / a)
            g = np.array([2.0 * float((w1 * (G.conj() * U[i]).real.sum(dim=1)).sum())
                          for i in range(k)])
            nrm = float(np.linalg.norm(g))
            if nrm == 0.0:
                return np.zeros(k)
            step = 1.0 / nrm
            for _ in range(int(self.p.max_backtrack)):
                cand = -step * g
                _, ac = state(cand)
                if loss_from(ac) < L0:
                    return cand
                step *= 0.5
            return np.zeros(k)

        self._lam = lam
        return best_t

    # ------------------------------------------------------------------ #
    #  One batch                                                          #
    # ------------------------------------------------------------------ #
    def _batch_update(self, idx):
        N = self.p.number_of_slices
        B = idx.numel()
        eps = torch.finfo(self.rdt).eps
        sub = str(self.p.subspace).lower()
        use_max = str(self.p.object_denom).lower().startswith('m')
        reg = float(self.p.denom_reg)
        do_probe = self.curiter >= self.p.probe_update_start

        phi, den_phi = self._phi()

        O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(N)])   # (N,B,ny,nx)
        s = O - 1.0

        # --- forward ---
        psi = self.pr_g[None] + self._A_slices(s, phi)                         # (B,M,ny,nx)

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
        r = psi_p - psi
        dj = torch.empty((N, B, self.M, self.fny, self.fnx), dtype=self.cdt, device=self.dev)
        dj[0] = r                                                              # H_0 = 1
        if N > 1:
            dj[1:] = torch.fft.ifft2(self.H[:, None, None] * torch.fft.fft2(r))

        num = (phi[:, None].conj() * dj).sum(dim=2)                            # (N,B,ny,nx)

        # --- preconditioned object direction in object space ---
        self._acc_n.zero_()
        self._acc_d.zero_()
        for j in range(N):
            self._scatter(self._acc_n[j], idx, num[j])
            self._scatter(self._acc_d[j], idx, den_phi[j].expand(B, self.fny, self.fnx))
        peak = self._acc_d.amax(dim=(-2, -1), keepdim=True)
        dn = peak if use_max else self._acc_d + reg * peak
        gobj = self._acc_n / dn.clamp_min(eps).to(self.cdt)                    # (N,ony,onx)

        # --- preconditioned probe direction ---
        dP = None
        if do_probe:
            gp = s[0][:, None].conj() * dj[0]
            if N > 1:
                Fg = torch.fft.fft2(s[1:, :, None].conj() * dj[1:])
                gp = gp + torch.fft.ifft2((self.Hb[:, None, None] * Fg).sum(0))
            num_p = (r + gp).sum(dim=0)                                        # (M,ny,nx)
            den_p = (O.abs() ** 2).sum(0).sum(0)                               # (ny,nx)
            pk = den_p.amax()
            dnp = pk if use_max else den_p + reg * pk
            dP = num_p / dnp.clamp_min(eps).to(self.cdt)

        # ---------------- subspace Newton ----------------
        # Build the direction set and its image under A, then F. Each direction
        # costs one operator application and one far-field transform; the reduced
        # solve is k x k and free.
        Gp = self._gather_stack(gobj, idx)                                     # (N,B,ny,nx)
        dirs, U = [], []

        if sub.startswith('j'):
            dirs.append(('obj', None, gobj))
            U.append(self._apply(self._ff_fw, self._A_slices(Gp, phi)))
        else:
            for j in range(N):
                dirs.append(('obj', j, gobj))
                U.append(self._apply(self._ff_fw, self._A_one_slice(j, Gp[j], phi)))
            if 'prev' in sub and self._prev is not None:
                Pp = self._gather_stack(self._prev, idx)
                dirs.append(('obj_full', None, self._prev))
                U.append(self._apply(self._ff_fw, self._A_slices(Pp, phi)))

        if do_probe:
            dirs.append(('probe', None, dP))
            U.append(self._apply(self._ff_fw, self._A_probe(dP, s)))

        t = self._reduced_newton(F, U, am, mk)

        inc = torch.zeros_like(self.ob_g)
        for ti, (kind, j, payload) in zip(t, dirs):
            if ti == 0.0:
                continue
            if kind == 'obj':
                if j is None:
                    inc += float(ti) * float(self.p.alpha) * payload
                else:
                    inc[j] += float(ti) * float(self.p.alpha) * payload[j]
            elif kind == 'obj_full':
                inc += float(ti) * float(self.p.alpha) * payload
            else:
                self.pr_g += float(ti) * float(self.p.beta) * payload
                self._phi_cache = None

        self.ob_g += inc
        self._prev = inc
        return err

    def _gather_stack(self, field, idx):
        return torch.stack([self._gather(field[j], idx)
                            for j in range(self.p.number_of_slices)])

    # ------------------------------------------------------------------ #
    #  Iteration                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        B = int(self.p.batch_size)

        for _ in range(num):
            order = torch.randperm(self.n_pos, device=self.dev)

            err = np.zeros(3)
            for s in range(0, self.n_pos, B):
                err += self._batch_update(order[s:s + B])
            err /= float(self.n_pos)

            logger.info("iter %d: %s  lam = %.2e"
                        % (self.curiter, self._slice_means_str(), self._lam))
            self._plot_iteration_progress()
            self.curiter += 1

        return {'all': err}

    def _slice_means_str(self):
        """
        <O_j> over the illuminated region.

        Diagnostic only: for N slices a constant a_j on slice j sends psi to
        psi + a_j P, so only sum_j a_j is observable and the differences are
        exactly null. These numbers therefore report the gauge, not the sample;
        judge the reconstruction on contrast within a slice.
        """
        m = torch.stack([self.ob_g[j][self.cov].mean()
                         for j in range(self.p.number_of_slices)])
        return ("|<O>| = %s, arg(<O>) = %s"
                % (np.array2string(m.abs().cpu().numpy(), precision=4),
                   np.array2string(m.angle().cpu().numpy(), precision=4)))

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

            plt.suptitle(self._slice_means_str())
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
        info.subspace = self.p.subspace
        info.object_denom = self.p.object_denom
        info.batch_size = self.p.batch_size
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