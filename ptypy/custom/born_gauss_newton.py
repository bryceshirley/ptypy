# -*- coding: utf-8 -*-
"""
Alternating Gauss-Newton ptychography on the first Born multislice model.

WHY THIS IS A DIFFERENT ALGORITHM, NOT A FASTER FORWARD MODEL
-------------------------------------------------------------
The Born forward model

    s_j   = O_j - 1                                  exact, no expansion in s
    phi_j = F^-1[ H_j F[P] ]                         illumination at slice j
    Psi   = F_ff[ P + sum_j F^-1[ Hbar_j F[ s_j phi_j ] ] ]

is AFFINE in the object: Psi = A s + b with A linear and b = F_ff[P] carrying
no object at all. Every cross derivative vanishes,

    d^2 Psi / d s_i d s_j = 0    for all i, j,

so the Gauss-Newton approximation to the amplitude objective discards ONLY the
curvature of the modulus. Nothing is dropped from the object itself, and the
Gauss-Newton subproblem is therefore an exact convex quadratic with a FIXED
operator, solvable by conjugate gradients.

Multislice cannot be written this way. There Psi is a degree-N multilinear
polynomial in {O_j}, so Gauss-Newton additionally throws away all the
slice-coupling curvature, which is the dominant part; the subproblem is neither
exact nor fixed. The comparison this engine supports is therefore not "same
solver, cheaper transforms" but "a solver class that exists only because the
model is affine". That is a structural claim about the model, not a claim about
tuning.

THE OBJECTIVE AND THE TWO BLOCKS
--------------------------------
    a = sqrt( sum_m |Psi_m|^2 )        coherent model amplitude
    t = sqrt( max(I - B, 0) )          measured amplitude, background removed
    f = sum_k w_k ( a_k - t_k )^2

The model is BILINEAR: affine in s for fixed P, and linear in P for fixed s
(phi_j is itself a linear function of P). So both blocks admit exact quadratic
subproblems in principle. Only the object block gets one here:

  * OBJECT: Gauss-Newton. With r = t - a and Psihat = Psi / a,

        J d      = sum_m Re( conj(Psihat_m) (A d)_m )        real
        J^dag p  = A^dag ( p Psihat )                        complex

    and the step solves ( J^dag W J + mu M ) d = J^dag W r by preconditioned
    CG in the real inner product <u,v>_R = Re sum(conj(u) v). Note this is
    valid across the WHOLE detector, dark field included, because Psihat is
    defined wherever a > 0. It is not a bright-field or weak-scattering method;
    the earlier holographic linearisation, which needed |A s| << |b| and only
    saw the bright field, is the special case Psi -> b of this operator.

  * PROBE: left nonlinear. Fixing s, Psi = C P is linear in P but there is no
    reference term, so |C P|^2 is genuine phase retrieval and the subproblem is
    not convex. The affine structure buys the object block, not the probe
    block. The probe is also constrained by every frame at once, so it
    converges far faster than the object and an ePIE-style update is enough.

  * BACKGROUND: a per-pixel B >= 0, refreshed as a coordinate step. Included
    because in practice sum |b|^2 falls well short of the mean frame total, and
    that deficit otherwise gets fitted into the object.

LINE SEARCH
-----------
f(lambda) along s -> s + lambda d is not polynomial, because of the square
root. But the model INTENSITY is exactly quadratic in lambda,

    I_c(lambda) = p0 + lambda p1 + lambda^2 p2

with p0 = |Psi|^2, p1 = 2 Re<Psi, A d>, p2 = |A d|^2 summed over modes. So one
sweep computes p0, p1, p2 and then evaluates f on an entire grid of lambda with
elementwise work only, no further transforms. lambda = 0 is always in the grid,
so a step can never increase the objective, and lambda doubles as the trust
indicator that drives the Levenberg-Marquardt damping.

COST
----
One CG iteration costs two forward sweeps and one adjoint sweep, since Psihat
must be known wherever A d is applied. With gn_cache = True the unit field
Psihat is held for the CG subset and the cost drops to one forward plus one
adjoint; the cache is (positions x modes x frame) complex64 and falls back
automatically if it will not allocate. gn_positions controls the rest: the
system is heavily overdetermined and a fixed random subset changes the step
very little.
"""
import os
import time
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
class BornGaussNewton(stochastic.EPIE):
    """
    Alternating Gauss-Newton / ePIE ptychography on the affine Born model.

    Defaults:

    [name]
    default = BornGaussNewton
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

    [gn_cg]
    default = 8
    type = int
    help = CG iterations per Gauss-Newton step
    doc = The subproblem is an exact convex quadratic, so CG converges fast
          under the Jacobi preconditioner and there is little point running it
          to tolerance: the quadratic is only a local model of the modulus.
          Early stopping is also the regulariser for the rank deficiency, which
          is real (CTF zeros, and nothing beyond the double overlap band).

    [gn_positions]
    default = 0.3
    type = float
    help = Fraction of scan positions used inside the CG
    doc = The normal equations are heavily overdetermined, so a fixed random
          subset changes the step direction very little while cutting the
          dominant cost proportionally. The subset is redrawn per Gauss-Newton
          step but held fixed within one, since resampling mid-CG would change
          the quadratic being minimised.

    [gn_damping]
    default = 1e-2
    type = float
    help = Initial Levenberg-Marquardt damping, relative to the preconditioner
    doc = Adapted from the line search: a step that wants lambda near 1 means
          the quadratic model is trustworthy and damping falls, a step cut
          short means it is not and damping rises.

    [gn_damping_factor]
    default = 3.0
    type = float
    help = Multiplicative adjustment applied to the damping each step

    [gn_cache]
    default = True
    type = bool
    help = Cache the unit model field for the CG subset
    doc = Psihat = Psi / a is needed on every CG iteration and is fixed within
          a Gauss-Newton step. Caching it removes one forward sweep out of
          three. Costs positions x modes x frame in complex64 on the device and
          falls back to recomputation automatically if allocation fails.

    [probe_epochs]
    default = 1
    type = int
    help = Nonlinear probe epochs between Gauss-Newton steps
    doc = The probe sees every frame, so it is far better determined than the
          object and does not need the heavier machinery. Raise this only if
          the probe is visibly lagging.

    [warmup]
    default = 3
    type = int
    help = Probe-only epochs before the first Gauss-Newton step
    doc = The Gauss-Newton operator is built from the current probe. Starting
          from a raw illumination model wastes the step.

    [beta]
    default = 0.2
    type = float
    help = Probe update step size
    doc = Divided by number_of_slices internally, as the denominator sums
          |O_j|^2 over slices.

    [denom_reg]
    default = 0.01
    type = float
    help = Floor on the Jacobi preconditioner, as a fraction of its maximum

    [background]
    default = True
    type = bool
    help = Fit a per-pixel incoherent background
    doc = Without it, any part of the measured intensity the coherent model
          cannot produce is absorbed by the object. If sum |b|^2 falls well
          short of the mean frame total, that deficit is real and has to go
          somewhere.

    [background_interval]
    default = 5
    type = int
    help = Gauss-Newton steps between background refreshes

    [background_relax]
    default = 0.5
    type = float
    help = Relaxation on the background coordinate step

    [device]
    default = cuda
    type = str
    help = Torch device: 'cuda', 'cuda:0' or 'cpu'

    [batch_size]
    default = 32
    type = int
    help = Scan positions per sweep chunk
    doc = Only a memory knob here, not an algorithmic one: the Gauss-Newton
          step accumulates over the whole subset before updating, so unlike
          ePIE the result does not depend on it.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

    [seed]
    default = 0
    type = int
    help = RNG seed for initialisation, subsets and batch shuffling

    [flog]
    default = convergence.csv
    type = str
    help = Path for the per-iteration convergence log
    doc = Columns: iter, wall clock seconds, objective, amplitude R factor,
          damping, accepted step. Plot R against seconds.

    [log_interval]
    default = 10
    type = int
    help = Iterations between progress plots and log flushes
    """

    def __init__(self, ptycho_parent, pars=None):
        super(BornGaussNewton, self).__init__(ptycho_parent, pars)
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
            raise ImportError("BornGaussNewton requires PyTorch")

        N = self.p.number_of_slices

        dev = str(self.p.device)
        if dev.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            dev = 'cpu'
        self.dev = torch.device(dev)
        dbl = str(self.p.precision).lower().startswith('d')
        self.cdt = torch.complex128 if dbl else torch.complex64
        self.rdt = torch.float64 if dbl else torch.float32

        torch.manual_seed(int(self.p.seed))

        if len(self.ob.storages) != 1 or len(self.pr.storages) != 1:
            raise NotImplementedError("single object and probe storage only")
        self._obs = list(self.ob.storages.values())[0]
        self._prs = list(self.pr.storages.values())[0]
        if self._obs.data.shape[0] != 1:
            raise NotImplementedError("single object mode only")
        self.M, self.fny, self.fnx = self._prs.data.shape

        self._object = [self.ob.copy(self.ob.ID + "_o_%d" % i) for i in range(N)]

        rng = np.random.default_rng(int(self.p.seed))
        ob0 = np.ones_like(self._obs.data[0])
        self.ob_g = torch.stack([
            torch.as_tensor(ob0 + 1e-3 * ((rng.random(ob0.shape) - 0.5)
                                          + 1j * (rng.random(ob0.shape) - 0.5)),
                            dtype=self.cdt, device=self.dev) for _ in range(N)])
        self.pr_g = torch.as_tensor(np.ascontiguousarray(self._prs.data),
                                    dtype=self.cdt, device=self.dev)
        self.ony, self.onx = int(self.ob_g.shape[-2]), int(self.ob_g.shape[-1])

        geom = list(self.ptycho.model.scans.values())[0].geometries[0]
        shape, lam = geom.shape, geom.lam
        V, W = u.grids(shape, lam / shape / geom.resolution, 'fft')
        a2 = V**2 + W**2

        th = self.p.slice_thickness
        z = [sum(th[:i]) for i in range(1, N)] if type(th) in (list, tuple) \
            else [i * th for i in range(1, N)]

        H = np.empty((max(N - 1, 1), *shape), dtype=np.complex128)
        for i, zi in enumerate(z):
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            Hi = np.exp(2j * np.pi * (zi / lam) * kz)
            Hi[a2 > 1.0] = 0.0
            H[i] = Hi
        self.H = torch.as_tensor(H[:N - 1], dtype=self.cdt, device=self.dev)
        self.Hb = self.H.conj()

        self._build_farfield()
        self._collect_views()

        self._phi_cache = None
        self._bg = torch.zeros((self.fny, self.fnx), dtype=self.rdt, device=self.dev)

        self._mu = float(self.p.gn_damping)
        self._gn_steps = 0
        self._t_run = 0.0
        self._history = []
        self._psihat = None

        logger.info("Gauss-Newton: cg %d, positions %.2f, damping %.2e, "
                    "probe epochs %d, warmup %d, background %s, seed %d"
                    % (int(self.p.gn_cg), float(self.p.gn_positions), self._mu,
                       int(self.p.probe_epochs), int(self.p.warmup),
                       bool(self.p.background), int(self.p.seed)))

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

    @staticmethod
    def _apply_adj(spec, w):
        k = spec['kind']
        n = float(w.shape[-2] * w.shape[-1])
        if k == 'lsi':
            return torch.fft.ifft2(torch.fft.fft2(w) * spec['H'].conj())
        if k == 'ilsi':
            return torch.fft.fft2(torch.fft.ifft2(w) * spec['H'].conj())
        if k == 'chirp':
            return spec['R'].conj() * (n * torch.fft.ifft2(spec['A'].conj() * w))
        return spec['R'].conj() * (torch.fft.fft2(spec['A'].conj() * w) / n)

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
            masks.append(np.ones_like(data[-1], dtype=bool) if m is None
                         else np.asarray(m, dtype=bool))

        self.n_pos = len(names)
        self.rows = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.I = torch.as_tensor(np.stack(data), dtype=self.rdt, device=self.dev)
        self.mask = torch.as_tensor(np.stack(masks), device=self.dev)

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
    #  Gather / scatter and the Born operator                             #
    # ------------------------------------------------------------------ #
    def _gather(self, arr, idx):
        return arr[self.rows[idx][:, :, None], self.cols[idx][:, None, :]]

    def _scatter(self, dst, idx, vals):
        B = idx.numel()
        r = self.rows[idx][:, :, None].expand(B, self.fny, self.fnx)
        c = self.cols[idx][:, None, :].expand(B, self.fny, self.fnx)
        dst.index_put_((r.reshape(-1), c.reshape(-1)), vals.reshape(-1), accumulate=True)

    def _phi(self):
        if self._phi_cache is None:
            N = self.p.number_of_slices
            ph = torch.empty((N, self.M, self.fny, self.fnx),
                             dtype=self.cdt, device=self.dev)
            ph[0] = self.pr_g
            if N > 1:
                ph[1:] = torch.fft.ifft2(self.H[:, None] * torch.fft.fft2(self.pr_g))
            self._phi_cache = (ph, (ph.abs() ** 2).sum(dim=1))
        return self._phi_cache

    def _scat(self, s, phi):
        out = s[0][:, None] * phi[0][None]
        if s.shape[0] > 1:
            Fs = torch.fft.fft2(s[1:, :, None] * phi[1:, None])
            out = out + torch.fft.ifft2((self.Hb[:, None, None] * Fs).sum(0))
        return out

    def _scat_adj(self, d, phi):
        N = phi.shape[0]
        dj = torch.empty((N, d.shape[0], self.M, self.fny, self.fnx),
                         dtype=self.cdt, device=self.dev)
        dj[0] = d
        if N > 1:
            dj[1:] = torch.fft.ifft2(self.H[:, None, None] * torch.fft.fft2(d))
        return (phi[:, None].conj() * dj).sum(dim=2), dj

    def _model(self, idx, s=None, phi=None):
        if phi is None:
            phi, _ = self._phi()
        N = self.p.number_of_slices
        if s is None:
            s = torch.stack([self._gather(self.ob_g[j], idx) for j in range(N)]) - 1.0
        Psi = self._apply(self._ff_fw, self.pr_g[None] + self._scat(s, phi))
        a = torch.sqrt((Psi.real ** 2 + Psi.imag ** 2).sum(dim=1).clamp_min(0))
        return Psi, a

    def _target(self, idx):
        return torch.sqrt((self.I[idx] - self._bg[None]).clamp_min(0))

    # ------------------------------------------------------------------ #
    #  Gauss-Newton on the object block                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rdot(a, b):
        return float((a.conj() * b).real.sum())

    def _batches(self, pos):
        B = int(self.p.batch_size)
        for st in range(0, int(pos.numel()), B):
            yield pos[st:st + B]

    def _psihat_for(self, k, idx, phi):
        if self._psihat is not None:
            return self._psihat[k:k + idx.numel()]
        Psi, a = self._model(idx, phi=phi)
        return Psi / a.clamp_min(1e-20)[:, None].to(self.cdt)

    def _build_cache(self, pos, phi):
        self._psihat = None
        if not bool(self.p.gn_cache):
            return
        n = int(pos.numel())
        want = n * self.M * self.fny * self.fnx * (16 if self.cdt is torch.complex128 else 8)
        try:
            buf = torch.empty((n, self.M, self.fny, self.fnx),
                              dtype=self.cdt, device=self.dev)
        except RuntimeError:
            logger.warning("gn_cache: %.2f GB would not allocate, recomputing "
                           "the model field each CG iteration" % (want / 2**30))
            return
        k = 0
        for idx in self._batches(pos):
            Psi, a = self._model(idx, phi=phi)
            buf[k:k + idx.numel()] = Psi / a.clamp_min(1e-20)[:, None].to(self.cdt)
            k += idx.numel()
        self._psihat = buf

    def _gn_rhs(self, pos, phi):
        g = torch.zeros_like(self.ob_g)
        f = 0.0
        k = 0
        for idx in self._batches(pos):
            Psi, a = self._model(idx, phi=phi)
            t = self._target(idx)
            w = self.mask[idx]
            r = torch.where(w, t - a, torch.zeros_like(a))
            f += float((r * r).sum())
            ph = self._psihat_for(k, idx, phi)
            v = r[:, None].to(self.cdt) * ph
            
            # Use the correct mathematical adjoint _apply_adj
            v = self._apply_adj(self._ff_fw, v)
            
            num, _ = self._scat_adj(v, phi)
            for j in range(g.shape[0]):
                self._scatter(g[j], idx, num[j])
            k += idx.numel()
        return g, f

    def _gn_normal(self, d, pos, phi, mu, Mdiag):
        out = torch.zeros_like(self.ob_g)
        N = self.p.number_of_slices
        k = 0
        for idx in self._batches(pos):
            ph = self._psihat_for(k, idx, phi)
            dp = torch.stack([self._gather(d[j], idx) for j in range(N)])
            u_ = self._apply(self._ff_fw, self._scat(dp, phi))
            jd = (ph.conj() * u_).real.sum(dim=1)
            jd = torch.where(self.mask[idx], jd, torch.zeros_like(jd))
            v = jd[:, None].to(self.cdt) * ph
            
            # Use the correct mathematical adjoint _apply_adj
            v = self._apply_adj(self._ff_fw, v)
            
            num, _ = self._scat_adj(v, phi)
            for j in range(N):
                self._scatter(out[j], idx, num[j])
            k += idx.numel()
        return out + mu * Mdiag.to(self.cdt) * d

    def _gn_linesearch(self, d, pos, phi, lams):
        N = self.p.number_of_slices
        tot = np.zeros(len(lams))
        for idx in self._batches(pos):
            Psi, _ = self._model(idx, phi=phi)
            dp = torch.stack([self._gather(d[j], idx) for j in range(N)])
            u_ = self._apply(self._ff_fw, self._scat(dp, phi))
            p0 = (Psi.real ** 2 + Psi.imag ** 2).sum(dim=1)
            p1 = 2.0 * (Psi.conj() * u_).real.sum(dim=1)
            p2 = (u_.real ** 2 + u_.imag ** 2).sum(dim=1)
            t = self._target(idx)
            w = self.mask[idx]
            for i, l in enumerate(lams):
                a = torch.sqrt((p0 + l * p1 + (l * l) * p2).clamp_min(0))
                tot[i] += float((torch.where(w, a - t, torch.zeros_like(a)) ** 2).sum())
        return tot

    def _gn_adjoint_test(self, pos, phi):
        idx = pos[:min(4, int(pos.numel()))]
        N = self.p.number_of_slices
        Psi, a = self._model(idx, phi=phi)
        ph = Psi / a.clamp_min(1e-20)[:, None].to(self.cdt)
        w = self.mask[idx]

        x = torch.randn((N, self.ony, self.onx), dtype=self.cdt, device=self.dev)
        r = torch.randn((idx.numel(), self.fny, self.fnx), dtype=self.rdt, device=self.dev)
        r = torch.where(w, r, torch.zeros_like(r))

        xp = torch.stack([self._gather(x[j], idx) for j in range(N)])
        u_ = self._apply(self._ff_fw, self._scat(xp, phi))
        jd = (ph.conj() * u_).real.sum(dim=1)
        lhs = float((torch.where(w, jd, torch.zeros_like(jd)) * r).sum())

        out = torch.zeros_like(self.ob_g)
        v = r[:, None].to(self.cdt) * ph
        
        # Use the correct mathematical adjoint _apply_adj
        v = self._apply_adj(self._ff_fw, v)
        
        num, _ = self._scat_adj(v, phi)
        for j in range(N):
            self._scatter(out[j], idx, num[j])
        rhs = self._rdot(out, x)

        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        if rel > (1e-4 if self.rdt is torch.float32 else 1e-9):
            raise RuntimeError("Gauss-Newton adjoint test failed, relative error %.2e" % rel)
        logger.info("Gauss-Newton adjoint test passed (%.2e)" % rel)

    def _object_step(self):
        N = self.p.number_of_slices
        phi, den_phi = self._phi()

        frac = float(np.clip(self.p.gn_positions, 1e-3, 1.0))
        npos = max(1, int(round(frac * self.n_pos)))
        pos = torch.randperm(self.n_pos, device=self.dev)[:npos]

        if self._gn_steps == 0:
            self._gn_adjoint_test(pos, phi)

        acc = torch.zeros((N, self.ony, self.onx), dtype=self.rdt, device=self.dev)
        for idx in self._batches(pos):
            for j in range(N):
                self._scatter(acc[j], idx, den_phi[j].expand(idx.numel(), self.fny, self.fnx))
        peak = acc.amax(dim=(-2, -1), keepdim=True)
        Mdiag = acc + float(self.p.denom_reg) * peak
        Minv = 1.0 / Mdiag.clamp_min(torch.finfo(self.rdt).tiny)

        self._build_cache(pos, phi)
        try:
            g, f0 = self._gn_rhs(pos, phi)

            d = torch.zeros_like(self.ob_g)
            r = g.clone()
            z = Minv.to(self.cdt) * r
            p = z.clone()
            rz = self._rdot(r, z)
            rz0 = rz
            for _ in range(int(self.p.gn_cg)):
                if rz <= 0.0:
                    break
                Gp = self._gn_normal(p, pos, phi, self._mu, Mdiag)
                pGp = self._rdot(p, Gp)
                if pGp <= 0.0:
                    break
                al = rz / pGp
                d = d + al * p
                r = r - al * Gp
                z = Minv.to(self.cdt) * r
                rz_new = self._rdot(r, z)
                p = z + (rz_new / max(rz, 1e-30)) * p
                rz = rz_new

            lams = np.concatenate([[0.0], np.geomspace(0.02, 2.0, 24)])
            fs = self._gn_linesearch(d, pos, phi, lams)
        finally:
            self._psihat = None

        k = int(np.argmin(fs))
        lam, f1 = float(lams[k]), float(fs[k])

        if lam > 0.0:
            self.ob_g = self.ob_g + lam * d

        fac = float(self.p.gn_damping_factor)
        if lam >= 0.8:
            self._mu = max(self._mu / fac, 1e-8)
        elif lam < 0.2:
            self._mu = min(self._mu * fac, 1e4)

        self._gn_steps += 1
        return dict(f0=f0, f1=f1, lam=lam, mu=self._mu,
                    cg=(rz0, rz), gain=f0 / max(f1, 1e-30))

    # ------------------------------------------------------------------ #
    #  Probe block and background                                         #
    # ------------------------------------------------------------------ #
    def _probe_epoch(self):
        N = self.p.number_of_slices
        eps = torch.finfo(self.rdt).eps
        reg = float(self.p.denom_reg)
        step = float(self.p.beta) / N
        order = torch.randperm(self.n_pos, device=self.dev)

        rnum = 0.0
        rden = 0.0
        for idx in self._batches(order):
            phi, _ = self._phi()
            O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(N)])
            s = O - 1.0
            psi = self.pr_g[None] + self._scat(s, phi)
            F = self._apply(self._ff_fw, psi)
            a = torch.sqrt((F.real ** 2 + F.imag ** 2).sum(dim=1).clamp_min(0))
            t = self._target(idx)
            w = self.mask[idx]

            rnum += float(torch.where(w, (a - t).abs(), torch.zeros_like(a)).sum())
            rden += float(torch.where(w, t, torch.zeros_like(t)).sum())

            scale = torch.where(w, t / a.clamp_min(1e-12), torch.ones_like(a))
            d = self._apply(self._ff_bw, F * scale[:, None].to(self.cdt)) - psi

            _, dj = self._scat_adj(d, phi)
            gp = s[0][:, None].conj() * dj[0]
            if N > 1:
                Fg = torch.fft.fft2(s[1:, :, None].conj() * dj[1:])
                gp = gp + torch.fft.ifft2((self.Hb[:, None, None] * Fg).sum(0))
            num_p = (d + gp).sum(dim=0)
            den_p = (O.abs() ** 2).sum(0).sum(0)
            dn = den_p + reg * den_p.amax()
            self.pr_g = self.pr_g + step * num_p / dn.clamp_min(eps).to(self.cdt)
            self._phi_cache = None

        # GPU-native probe support bypassing the buggy ptypy call
        if hasattr(self.p, 'probe_support') and self.p.probe_support is not None:
            sup = float(self.p.probe_support)
            if sup > 0.0:
                amax = self.pr_g.abs().amax(dim=(-2, -1), keepdim=True)
                self.pr_g = torch.where(self.pr_g.abs() >= sup * amax, self.pr_g, torch.zeros_like(self.pr_g))
                self._phi_cache = None

        return rnum / max(rden, 1e-30)

    def _background_step(self):
        acc = torch.zeros((self.fny, self.fnx), dtype=self.rdt, device=self.dev)
        cnt = torch.zeros((self.fny, self.fnx), dtype=self.rdt, device=self.dev)
        pos = torch.arange(self.n_pos, device=self.dev)
        for idx in self._batches(pos):
            _, a = self._model(idx)
            w = self.mask[idx].to(self.rdt)
            acc += (w * (self.I[idx] - a * a)).sum(dim=0)
            cnt += w.sum(dim=0)
        est = (acc / cnt.clamp_min(1.0)).clamp_min(0.0)
        eta = float(self.p.background_relax)
        self._bg = (1.0 - eta) * self._bg + eta * est
        return float(self._bg.sum())

    # ------------------------------------------------------------------ #
    #  Iteration                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        for _ in range(num):
            t0 = time.perf_counter()

            R = 0.0
            for _ in range(max(1, int(self.p.probe_epochs))):
                R = self._probe_epoch()

            info = None
            if self.curiter >= int(self.p.warmup):
                info = self._object_step()

                if (bool(self.p.background)
                        and self._gn_steps % max(1, int(self.p.background_interval)) == 0):
                    tot = self._background_step()
                    logger.info("background refreshed, total %.3e" % tot)

            if torch.cuda.is_available() and self.dev.type == 'cuda':
                torch.cuda.synchronize()
            self._t_run += time.perf_counter() - t0

            f1 = info['f1'] if info else float('nan')
            lam = info['lam'] if info else 0.0
            self._history.append((self.curiter, self._t_run, f1, R, self._mu, lam))

            if info is not None and self.curiter % 2 == 0:
                logger.info("iter %d  t %.1f s  R %.5f  f %.4e (gain %.3f)  "
                            "lambda %.3f  mu %.2e  CG %.2e -> %.2e"
                            % (self.curiter, self._t_run, R, info['f1'],
                               info['gain'], info['lam'], info['mu'],
                               info['cg'][0], info['cg'][1]))
                self._plot_iteration_progress()
                self._write_log()
            self.curiter += 1

        return {'all': np.array([R, f1 if f1 == f1 else 0.0, lam])}

    def _write_log(self):
        try:
            with open(self.p.flog, 'w') as f:
                f.write("# gn_cg=%d gn_positions=%.3f gn_damping=%.3e probe_epochs=%d "
                        "warmup=%d background=%s seed=%d slices=%d\n"
                        % (int(self.p.gn_cg), float(self.p.gn_positions),
                           float(self.p.gn_damping), int(self.p.probe_epochs),
                           int(self.p.warmup), bool(self.p.background),
                           int(self.p.seed), int(self.p.number_of_slices)))
                f.write("iter,seconds,objective,R,damping,lambda\n")
                for row in self._history:
                    f.write("%d,%.4f,%.8e,%.8e,%.6e,%.4f\n" % row)
        except Exception as e:      # noqa: BLE001
            logger.warning("could not write %s: %s" % (self.p.flog, e))

    def _plot_iteration_progress(self):
        try:
            probe = (self.pr_g[0].abs() ** 2).cpu().numpy()
            first = self.ob_g[0].angle().cpu().numpy()
            final = self.ob_g[-1].angle().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, im, title, cmap in (
                    (axes[0], probe, f"Probe Intensity (Iter {self.curiter})", 'viridis'),
                    (axes[1], first, f"Slice 0 Phase (Iter {self.curiter})", 'gray'),
                    (axes[2], final,
                     f"Slice {self.p.number_of_slices - 1} Phase (Iter {self.curiter})", 'gray')):
                h = ax.imshow(im, cmap=cmap)
                ax.set_title(title)
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f"{self.plot_progress_dir}/iter_progress_{self.curiter:04d}.png")
            plt.close(fig)
        except Exception as e:      # noqa: BLE001
            logger.warning(f"Could not plot iteration progress: {e}")

    def engine_finalize(self):
        self._write_log()

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
        info.gn_cg = self.p.gn_cg
        info.gn_positions = self.p.gn_positions
        info.gn_damping_final = self._mu
        info.gn_steps = self._gn_steps
        info.seed = self.p.seed
        info.background = np.asarray(self._bg.cpu().numpy(), dtype=np.float64)
        info.convergence = np.asarray(self._history, dtype=np.float64)
        info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                        for ob in self._object}

        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info("saving slices to %s" % self.p.fslices)
        io.h5write(self.p.fslices,
                   header={'description': 'Born Gauss-Newton reconstruction'},
                   content=info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()