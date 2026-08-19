# -*- coding: utf-8 -*-
"""
First Born multislice engine for ptychography (GPU, PyTorch).

FORWARD MODEL
-------------
    phi_j = F^-1[ H_j F[P] ]                        illumination at slice j
    psi   = c * ( P + sum_j F^-1[ Hbar_j F[ s_j phi_j ] ] )

with H_0 = 1 and, optionally, O_j = Obar_j (1 + s_j), c = prod_j Obar_j.

WHAT LINEARITY BUYS, AND WHAT IT DOES NOT
-----------------------------------------
psi is exactly linear in {s_j} and, separately, exactly linear in P. The
NONLINEARITY IS THE MODULUS at the detector: the loss is || |F psi| - sqrt(I) ||^2,
so the problem as a whole is not linear least squares. What is linear is
everything up to the detector, and that is enough for three things a multislice
engine cannot do:

1. EXACT STEP FOR THE GAUSS-NEWTON SUBPROBLEM. Freeze the modulus projection
   psi_proj at the current iterate. Because psi(s + t D) = psi + t A D holds to
   machine precision, || psi + t A D - psi_proj ||^2 is exactly quadratic in t
   and the minimiser is closed form after ONE application of A -- no trial
   evaluations. Multislice would have to re-run the full nonlinear cascade for
   every candidate step. Note what this is not: psi_proj moves as the iterate
   moves, so the step minimises the linearised subproblem, not the modulus loss.
   It can therefore overshoot the true objective, which is why the step is
   clamped non-negative and alpha is kept as a damping factor.

2. NO JACOBI DAMPING. A_j^dag A_j = M_{|phi_j|^2} exactly, since the propagator
   is unitary, so the per-slice preconditioner is exact; but the cross terms
   A_j^dag A_k are not diagonal, and a fixed step therefore needs step < 2/N.
   The line search sees all N slices inside one direction and accounts for their
   coupling exactly, so the 1/N damping is unnecessary.

3. JOINT OBJECT AND PROBE STEP. psi is linear in both, so the two directions are
   stepped together by a 2x2 solve that also removes their cross term.

PARALLELISM
-----------
phi_j carries no object, so it is neither position- nor batch-dependent and is
cached until the probe moves. The slice sum has no sequential dependency in z,
so the whole stack goes through one batched FFT in the forward and the adjoint.
H_0 = 1 is structural: slice 0 is never transformed.

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
    reconstruction. Candidates: ifft2(fft2(w) H), fft2(ifft2(w) H),
    A fft2(R w), A ifft2(R w), in numpy convention, which torch matches with the
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
class JointLineSearchFirstBorn(stochastic.EPIE):
    """
    First Born multislice ptychography on the GPU.

    Defaults:

    [name]
    default = JointLineSearchFirstBorn
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
    default = 1.0
    type = float
    help = Object step
    doc = Damping on the exact step for 'linesearch' (1 = exact), learning rate
          for 'adam', and a fixed step divided by number_of_slices for 'epie'.

    [beta]
    default = 1.0
    type = float
    help = Probe step
    doc = Interpreted as alpha is, for the probe.

    [lm_damping]
    default = 0.01
    type = float
    help = Levenberg damping on the 2x2 line-search solve
    doc = The object/probe scale gauge makes the 2x2 system near-singular, since
          O -> O(1+e), P -> P(1-e) leaves psi unchanged to first order. Damping
          keeps the initial guess sane; backtracking then guarantees descent.

    [max_backtrack]
    default = 8
    type = int
    help = Halvings allowed before a batch takes no step

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
    default = 32
    type = int
    help = Scan positions per update
    doc = Peak workspace scales as number_of_slices * batch_size * probe_modes.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)
    """

    def __init__(self, ptycho_parent, pars=None):
        super(JointLineSearchFirstBorn, self).__init__(ptycho_parent, pars)
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

        # persistent buffers, so the inner loop allocates nothing object-sized
        self._acc_n = torch.zeros((N, self.ony, self.onx), dtype=self.cdt, device=self.dev)
        self._acc_d = torch.zeros((N, self.ony, self.onx), dtype=self.rdt, device=self.dev)
        self._nstep = 0
        self._pr_ref = float((self.pr_g.abs() ** 2).sum())
        logger.info("batch_size = %d, denom = %s"
                    % (self.p.batch_size, self.p.object_denom))

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

        cov = torch.zeros((self.ony, self.onx), dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            flat = self._flat_index(idx)
            cov.view(-1).index_put_((flat,),
                                    torch.ones(flat.shape, dtype=self.rdt, device=self.dev),
                                    accumulate=True)
        self.cov = cov > 0
        frac = float(self.cov.to(self.rdt).mean())
        logger.info("%d positions, %d probe mode(s), illuminated fraction %.4f"
                    % (self.n_pos, self.M, frac))
        if frac < 1e-3:
            raise RuntimeError("coverage mask empty; check View.dlow indexing")

        # _scatter_slices works on a flattened (N, ony, onx) view, so an index
        # past the end of one object plane would silently wrap into the next
        # slice (or, for a column overflow, into the next row) instead of
        # raising. That shows up as a displaced ghost copy of real features.
        if (int(self.rows.max()) >= self.ony or int(self.cols.max()) >= self.onx
                or int(self.rows.min()) < 0 or int(self.cols.min()) < 0):
            raise RuntimeError(
                "patch extends outside the object array: rows [%d, %d] of %d, "
                "cols [%d, %d] of %d. Flat scatter would wrap into the adjacent "
                "slice." % (int(self.rows.min()), int(self.rows.max()), self.ony,
                            int(self.cols.min()), int(self.cols.max()), self.onx))

    # ------------------------------------------------------------------ #
    #  Gather / scatter                                                   #
    # ------------------------------------------------------------------ #
    def _flat_index(self, idx):
        """Flat indices into one object plane for every pixel of every patch."""
        return (self.rows[idx][:, :, None] * self.onx
                + self.cols[idx][:, None, :]).reshape(-1)

    def _gather(self, arr, idx):
        return arr[self.rows[idx][:, :, None], self.cols[idx][:, None, :]]

    def _scatter_slices(self, dst, flat, vals):
        """Accumulate (N, B, ny, nx) into (N, ony, onx), reusing one index tensor."""
        stride = self.ony * self.onx
        flat_dst = dst.view(-1)
        for j in range(dst.shape[0]):
            flat_dst.index_put_((flat + j * stride,), vals[j].reshape(-1), accumulate=True)

    # ------------------------------------------------------------------ #
    #  Linear operator pieces                                             #
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

    def _A_object(self, sd, phi):
        """
        A applied to a slice-space direction: c sum_j P_{-j}( sd_j phi_j ).

        sd is (N, B, ny, nx) in s units; returns (B, M, ny, nx). This is what
        makes the exact line search possible: the forward model of a direction
        costs one application and is exact, because psi is linear in s.
        """
        out = sd[0][:, None] * phi[0][None]                        # H_0 = 1
        if sd.shape[0] > 1:
            F = torch.fft.fft2(sd[1:, :, None] * phi[1:, None])
            out = out + torch.fft.ifft2((self.Hb[:, None, None] * F).sum(0))
        return out

    def _A_probe(self, dP, s):
        """
        A applied to a probe direction: c ( dP + sum_j P_{-j}( s_j P_j dP ) ).

        psi is linear in P as well, so the probe direction also has an exact
        forward model. (M, ny, nx) -> (B, M, ny, nx).
        """
        N = self.p.number_of_slices
        pd = torch.empty((N, self.M, self.fny, self.fnx), dtype=self.cdt, device=self.dev)
        pd[0] = dP
        if N > 1:
            pd[1:] = torch.fft.ifft2(self.H[:, None] * torch.fft.fft2(dP))
        return self._A_object(s, pd) + dP[None]

    # ------------------------------------------------------------------ #
    #  One batch                                                          #
    # ------------------------------------------------------------------ #
    def _batch_update(self, idx):
        N = self.p.number_of_slices
        B = idx.numel()
        eps = torch.finfo(self.rdt).eps
        use_max = str(self.p.object_denom).lower().startswith('m')
        reg = float(self.p.denom_reg)
        do_probe = self.curiter >= self.p.probe_update_start

        phi, den_phi = self._phi()
        flat = self._flat_index(idx)

        O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(N)])   # (N,B,ny,nx)
        s = O - 1.0

        # --- forward ---
        psi = self.pr_g[None] + self._A_object(s, phi)            # (B,M,ny,nx)

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

        # chain must be 5-D: dj is (N,B,M,ny,nx) and torch right-aligns, so a
        # 4-D chain would put the slice axis onto the batch axis
        num = (phi[:, None].conj() * dj).sum(dim=2)                    # (N,B,ny,nx)
        den = den_phi     # (N,ny,nx)

        # --- preconditioned object direction, in object space ---
        self._acc_n.zero_()
        self._acc_d.zero_()
        self._scatter_slices(self._acc_n, flat, num)
        self._scatter_slices(self._acc_d, flat,
                             den[:, None].expand(N, B, self.fny, self.fnx))
        peak = self._acc_d.amax(dim=(-2, -1), keepdim=True)
        dn = peak if use_max else self._acc_d + reg * peak
        g = self._acc_n / dn.clamp_min(eps).to(self.cdt)                       # (N,ony,onx)

        # --- preconditioned probe direction ---
        if do_probe:
            gp = s[0][:, None].conj() * dj[0]
            if N > 1:
                Fg = torch.fft.fft2(s[1:, :, None].conj() * dj[1:])
                gp = gp + torch.fft.ifft2((self.Hb[:, None, None] * Fg).sum(0))
            num_p = ( (r + gp)).sum(dim=0)                  # (M,ny,nx)
            den_p = (O.abs() ** 2).sum(0).sum(0)       # (ny,nx)
            pk = den_p.amax()
            dnp = pk if use_max else den_p + reg * pk
            dP = num_p / dnp.clamp_min(eps).to(self.cdt)
        else:
            dP = None

        # --- step ---
        t_o, t_p = self._line_search(g, dP, s, phi, r, idx, F, am, mk)
        step_o = float(self.p.alpha) * t_o
        step_p = float(self.p.beta) * t_p

        inc = step_o * g
        self.ob_g += inc

        if do_probe:
            self.pr_g += step_p * dP
            self._phi_cache = None

        return err

    def _line_search(self, g, dP, s, phi, r, idx, F, am, mk):
        """
        Exact line search on the TRUE modulus loss over the 2-D plane spanned by
        the object and probe directions.

        Because psi is linear in both arguments,

            F( psi + t_o u_o + t_p u_p ) = F psi + t_o F u_o + t_p F u_p

        holds exactly. So two extra far-field transforms buy the loss at ANY
        (t_o, t_p) for the cost of elementwise arithmetic, and the search is on
        the real objective rather than on a frozen projection. Multislice cannot
        do this: evaluating a trial step there needs the whole nonlinear cascade.

        The damped Gauss-Newton solve supplies the initial guess. It is only a
        guess, because the object/probe scale gauge -- O -> O(1+e), P -> P(1-e)
        leaves psi unchanged to first order -- makes u_o and u_p nearly
        anti-parallel and the 2x2 system near-singular in exactly that
        direction. Levenberg damping plus backtracking on the true loss makes the
        step monotone regardless.
        """
        N = self.p.number_of_slices
        G = torch.stack([self._gather(g[j], idx) for j in range(N)])
        u_o = self._A_object(G, phi)
        F_o = self._apply(self._ff_fw, u_o)

        oo = float((u_o.abs() ** 2).sum())
        ro = float((u_o.conj() * r).real.sum())
        lam = float(self.p.lm_damping)

        if dP is None:
            F_p = None
            t_o = ro / (oo * (1.0 + lam)) if oo > 0 else 0.0
            t_p = 0.0
        else:
            u_p = self._A_probe(dP, s)
            F_p = self._apply(self._ff_fw, u_p)
            pp = float((u_p.abs() ** 2).sum())
            rp = float((u_p.conj() * r).real.sum())
            op = float((u_o.conj() * u_p).real.sum())
            ood, ppd = oo * (1.0 + lam), pp * (1.0 + lam)
            det = ood * ppd - op * op
            if det > 0:
                t_o = (ppd * ro - op * rp) / det
                t_p = (ood * rp - op * ro) / det
            else:
                t_o = ro / ood if ood > 0 else 0.0
                t_p = 0.0

        def loss(to, tp):
            Ft = F + to * F_o if F_p is None else F + to * F_o + tp * F_p
            a = torch.sqrt((Ft.real ** 2 + Ft.imag ** 2).sum(dim=1).clamp_min(0))
            return float((mk * (a - am) ** 2).sum())

        L0 = loss(0.0, 0.0)
        for _ in range(int(self.p.max_backtrack)):
            if loss(t_o, t_p) < L0:
                return t_o, t_p
            t_o *= 0.5
            t_p *= 0.5
        return 0.0, 0.0          # no decrease found; take no step this batch

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