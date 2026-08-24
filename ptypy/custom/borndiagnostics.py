# -*- coding: utf-8 -*-
"""
First Born vs multislice diagnostics, as a ptypy engine.

WHAT THIS IS
------------
A read-only engine. It reconstructs nothing. On its first (and only) iterate it
loads a finished reconstruction, evaluates every forward model on the real scan
and writes the numbers and figures for the paper, then stops.

Everything the comparison needs is already inside ptypy by the time an engine
initialises, and asking ptypy for it removes every convention that has to be
guessed from files:

    frame origins       pod.ob_view.dlow          (position refinement included)
    measured data       view.data, pod.mask       (oriented, cropped, binned)
    far field           pod.fw, pod.bw            (chirps, origin, fftshift)
    geometry            geom.lam, geom.resolution, geom.psize, geom.distance
    probe modes         the probe storage

The only thing it reads from disk is the reconstruction being analysed, named
by `ground_truth`.

WHAT IT MEASURES
----------------
1.  eps_M of Eq. (epsilon), the truncation error against exact multislice,

        eps_M = || psi_MS - psi^(<=M) || / || psi^(<=M) ||

    for M = 0..max_order. The orders psi^(m) come out exactly, and without
    combinatorics, from the roots-of-unity trick: substituting g_j -> t g_j
    makes the exit wave a polynomial of degree N in t (each slice contributes
    one factor 1 + t g_j -- the termination argument of Sec. "A finite series
    terminating in multislice"), so evaluating the cascade at the (N+1)-th
    roots of unity and taking the DFT recovers every order. t = 1 gives exact
    multislice for free, and M = N returns it to round-off, which is the
    end-to-end check that the whole pipeline is right.

2.  The blocked truncation of Sec. "Blocked truncation", n_blocks = 1..N.
    Identical to the order truncation at N = 2; the two differ from N = 3 up.

3.  The two perturbation choices of Sec. "The perturbation": the chord
    g_j = O_j - 1 used in this work, and the tangent g_j = i s_j = log O_j of
    Balakrishnan et al. and Yeo et al. The chord is exact at N = 1; the tangent
    is not, and the gap is the O(phi^2) real part reported here.

4.  The dose criterion of Eq. (noisecrit), against the actual recorded counts,
    plus chi^2 of each model against the measured frames under the real mask.

5.  The exact degeneracy of Eq. (null-const): the response to a constant on one
    slice minus a constant on another, which is null, against a random
    direction of the same norm.

6.  Depth separability, Eq. (depthres): the response to an antisymmetric
    modulation at transverse frequency q, swept across q_c = 1/(2 NA dz), with
    NA measured from the reconstructed probe rather than assumed.

7.  The multislice preconditioner claim of Sec. (precond): how far the
    downstream cascade is from unitary, which is what decides whether
    diag(|psi_inc|^2) is the normal operator or only an approximation to it.

USAGE
-----
Take the reconstruction script, drop every optimisation parameter, point
`ground_truth` at the rec/ folder and set numiter = 1. See the companion
run_born_diagnostics.py.

    p.engines.engine00.name = 'FirstBornDiagnostics'
    p.engines.engine00.numiter = 1
    p.engines.engine00.ground_truth = out_dir_rec
    p.engines.engine00.number_of_slices = 2
    p.engines.engine00.slice_thickness = 1580e-6

The probe and the slices must come from the same run: engine_finalize writes
the product of the slices back into the object container, so the check is
prod_j O_j == obj up to a global scale, and it is reported at startup.
"""

import os
import json
import glob
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

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

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# ===========================================================================
#  Propagator extraction (identical to the engines, so the two agree exactly)
# ===========================================================================

def _scalar(x):
    """First element as a float. ptypy geometry values are sometimes arrays
    (lam, distance, psize, resolution can each be per-axis or per-scan), and
    float() on a 1-element array is deprecated / on a 2-element one is an
    error, so everything numeric from geom goes through here."""
    return float(np.ravel(np.asarray(x))[0])


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


# ===========================================================================
#  Reading a finished reconstruction
# ===========================================================================

def _slice_iter(path):
    """Iteration count in an fslices name, rec_crop-512_slices-2_iter-0500.h5."""
    import re
    m = re.search(r'iter[-_](\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _slice_candidates(ground_truth, fslices=None):
    """Every fslices-looking file in a results folder."""
    if fslices and os.path.isfile(fslices):
        return [fslices]
    if ground_truth and os.path.isfile(ground_truth):
        return [ground_truth]
    if not ground_truth:
        return []
    hits = []
    for p in ('*slices-*.h5', '*slices*.h5', 'slices.h5'):
        hits += glob.glob(os.path.join(ground_truth, p))
        if hits:
            break
    if not hits:
        hits = glob.glob(os.path.join(ground_truth, '*.h5'))
    return sorted(set(hits))


def _pick_slices(cands, want_iter=None, verbose=False):
    """First guess: the file at the .ptyr's iteration, else the highest.

    A results folder accumulates one fslices file per run, so listing them all
    is noise when the first guess is right; they are only listed if the guess
    has to be second-guessed against the data.
    """
    if not cands:
        return None
    if verbose and len(cands) > 1:
        logger.info("%d candidate slice files:" % len(cands))
        for h in cands:
            logger.info("    iter %5d   %s" % (_slice_iter(h), os.path.basename(h)))
    if want_iter is not None:
        exact = [h for h in cands if _slice_iter(h) == want_iter]
        if exact:
            return exact[0]
    return sorted(cands, key=_slice_iter)[-1]


def _find_slices(ground_truth, fslices=None, want_iter=None):
    """Locate the engine's fslices output inside a results folder.

    p.engines.engineXX.fslices embeds the engine's numiter, and p.io.rfile
    embeds the actual iteration count, so a folder that has seen several runs
    holds several of each. Preferring the newest by modification time pairs
    files from different runs, which is exactly the failure this guards
    against: the slice file whose iteration matches the chosen .ptyr wins, and
    the highest iteration otherwise.
    """
    if fslices and os.path.isfile(fslices):
        return fslices
    if ground_truth and os.path.isfile(ground_truth):
        return ground_truth
    if not ground_truth:
        return None
    hits = []
    for p in ('*slices-*.h5', '*slices*.h5', 'slices.h5'):
        hits += glob.glob(os.path.join(ground_truth, p))
        if hits:
            break
    hits = sorted(set(hits))
    if not hits:
        hits = sorted(glob.glob(os.path.join(ground_truth, '*.h5')))
    if not hits:
        return None
    if len(hits) > 1:
        logger.info("%d candidate slice files:" % len(hits))
        for h in hits:
            logger.info("    iter %5d   %s" % (_slice_iter(h), os.path.basename(h)))
    if want_iter is not None:
        exact = [h for h in hits if _slice_iter(h) == want_iter]
        if exact:
            return exact[0]
        logger.warning("no slice file at iteration %d; the .ptyr and the "
                       "slices may come from different runs" % want_iter)
    hits.sort(key=_slice_iter)
    return hits[-1]


def _iter_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _engine_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit('_', 2)
    return parts[-2] if len(parts) >= 2 else ''


def _find_ptyr(ground_truth, ptyr=None, engine=None):
    """Newest .ptyr by iteration count, optionally restricted to one engine.

    p.io.rfile interpolates %(engine)s and %(iterations)04d, so a folder that
    has seen several engines holds several files; the one that matters is the
    one that wrote the slices.
    """
    if ptyr and os.path.isfile(ptyr):
        return ptyr
    if not ground_truth or os.path.isfile(ground_truth):
        return None
    hits = glob.glob(os.path.join(ground_truth, '*.ptyr'))
    if engine:
        sel = [h for h in hits if str(engine).lower() in _engine_of(h).lower()]
        hits = sel or hits
    if not hits:
        return None
    hits.sort(key=_iter_of)
    return hits[-1]


def _read_slices(fpath):
    """content/objects/<Cobj_o_j>/<S...>/data written by engine_finalize."""
    if h5py is None:
        raise ImportError("h5py is required")
    out = []
    with h5py.File(fpath, 'r') as f:
        grp = f['content/objects']
        names = sorted(grp.keys(),
                       key=lambda s: int(''.join(c for c in s.split('_')[-1]
                                                 if c.isdigit()) or 0))
        for n in names:
            sub = grp[n]
            sk = [k for k in sub.keys() if isinstance(sub[k], h5py.Group)]
            if not sk:
                continue
            d = np.array(sub[sk[0]]['data'])
            out.append(d[0] if d.ndim == 3 else d)
    return out


def _read_ptyr(fpath):
    """Probe and object out of a ptypy dump."""
    out = dict(probe=None, obj=None)
    if fpath is None:
        return out
    with h5py.File(fpath, 'r') as f:
        hits = []
        f.visititems(lambda n, o: hits.append((n, o))
                     if isinstance(o, h5py.Dataset) and n.endswith('/data') else None)
        for n, o in hits:
            if '/probe/' in n and out['probe'] is None:
                out['probe'] = np.array(o)
            if ('/obj/' in n or '/object/' in n) and out['obj'] is None:
                d = np.array(o)
                out['obj'] = d[0] if d.ndim == 3 else d
    return out


# ===========================================================================
#  The engine
# ===========================================================================

@register()
class FirstBornDiagnostics(stochastic.EPIE):
    """
    Read-only engine: compare the first-Born and multislice forward models on a
    finished reconstruction, write the numbers and the figures, and stop.

    Defaults:

    [name]
    default = FirstBornDiagnostics
    type = str
    help = Engine identifier

    [ground_truth]
    default = None
    type = str
    help = Reconstruction to analyse: the run's rec/ folder, or a slices .h5
    doc = If a folder, the newest *slices-*.h5 in it is taken as the slice
          stack and the .ptyr with the highest iteration count as the probe
          (restrict with `engine_name`). If a file, that file is the slice
          stack and `ptyr` supplies the probe. If None, the live P.obj and
          P.probe are used, which is only meaningful when this engine runs
          straight after a reconstruction in the same script.

    [ptyr]
    default = None
    type = str
    help = Explicit .ptyr for the probe, overriding the search

    [engine_name]
    default = None
    type = str
    help = Restrict the .ptyr search to the engine that wrote the slices
    doc = e.g. MomentumFirstBorn. The probe must come from the same run as the
          slices; the provenance check at startup reports whether it does.

    [data_check_threshold]
    default = 0.5
    type = float
    help = Minimum agreement with the recorded data before anything is measured
    doc = The amplitude correlation between the multislice prediction and the
          measured frames, under the real mask. This, not any file-naming
          convention, is what establishes that the probe, the slices, the
          positions and the geometry belong together. Set to 0 to skip.

    [check_frames]
    default = 4
    type = int
    help = Frames used for the agreement check

    [iteration]
    default = None
    type = int
    help = Restrict every reconstruction to one iteration count
    doc = A results folder holds the same reconstruction at many iterations,
          and different engines converge at different rates, so a comparison
          across engines is only like for like at a fixed iteration. Setting
          this to 500 compares the 500-iteration first-Born and 3PIE
          reconstructions and nothing else.

    [pair_mode]
    default = same-iteration
    type = str
    help = How a slice file is matched to a probe: 'same-iteration' or 'best'
    doc = 'same-iteration' pairs a slice stack only with a .ptyr written at the
          same iteration, which is the pair one run produced. 'best' searches
          every combination and takes whichever predicts the data best, which
          can mix a probe from one run with slices from another -- a state no
          engine ever reached, and not something to quote.

    [slices]
    default = None
    type = str
    help = Which reconstruction to analyse: a path, or a filename substring
    doc = A results folder can hold several reconstructions of the same scan at
          the same iteration, e.g. a first-Born one and a 3PIE one. Give a
          substring such as 'threePIE' to pick one; the default takes the
          highest iteration. The probe is paired to it by whichever .ptyr in
          the folder best predicts the recorded data, so no naming convention
          has to be trusted.

    [compare]
    default = None
    type = str, list, tuple
    help = Further reconstructions to measure in the same run
    doc = Each entry is a path or a filename substring, e.g. 'threePIE'. Every
          entry gets the full epsilon measurement with its own probe, and the
          recovered slices are compared with the primary reconstruction's. This
          is the control for the obvious objection to Sec. (xray): if eps_1 is
          the same for an object fitted with multislice as for one fitted with
          first Born, it is a property of the specimen and not an artefact of
          having fitted the object with the model under test.

    [number_of_slices]
    default = 2
    type = int
    help = Number of object slices

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Slice separation in meters
    doc = A single float, or a list of length number_of_slices - 1.

    [outdir]
    default = born_diagnostics
    type = str
    help = Folder for the figures, results.json and results.npz

    [max_order]
    default = 2
    type = int
    help = Highest truncation order M
    doc = Costs N+1 multislice evaluations per position regardless of M. At
          M = N the result must return to round-off, which is the pipeline's
          own end-to-end check.

    [blocks]
    default = None
    type = int, list, tuple
    help = Block counts for the blocked truncation; default is 1..N

    [gauge]
    default = both
    type = str
    help = Slice gauge: 'none', 'mean' or 'both'
    doc = The per-slice mean transmission is not determined by the data
          (Eq. null-const) but eps_M does depend on it, since O_j -> O_j/alpha_j
          with the product pushed onto the probe leaves multislice pointwise
          invariant and changes |g_j|. 'both' reports each, which is the honest
          thing to quote.

    [batch_size]
    default = 16
    type = int
    help = Scan positions per batch
    doc = Peak workspace is (max_order + 3) x batch x probe_modes x frame, so
          this is smaller than a reconstruction batch.

    [n_positions]
    default = None
    type = int
    help = Subsample the scan to this many positions (None uses all)

    [compare_frames]
    default = 3
    type = int
    help = Frames plotted measured beside simulated

    [counts]
    default = None
    type = float
    help = Photons per frame for Eq. (noisecrit); default is the recorded sum

    [benchmark]
    default = True
    type = bool
    help = Time and peak memory of both forward models against slice count

    [benchmark_slice_counts]
    default = None
    type = int, list, tuple
    help = Slice counts to time; default 1, 2, 4, 8, 16, 32
    doc = The stack is extended by repeating the reconstructed slices, so the
          object is thicker than the specimen. That measures the cost of the
          two evaluation orders, which is what depends on N, and says nothing
          about accuracy.

    [benchmark_batch]
    default = None
    type = int
    help = Positions per timed call; defaults to batch_size

    [benchmark_reps]
    default = 5
    type = int
    help = Repeats per timing point, after one warm-up call

    [born_mode]
    default = lowmem
    type = str
    help = How the Born sum is evaluated: 'lowmem' or 'parallel'
    doc = 'parallel' puts every slice through one batched transform, which is
          fastest but holds an (N, batch, modes, ny, nx) tensor. 'lowmem'
          accumulates the branches one at a time, so peak storage is O(batch)
          regardless of N. Both are exact and give identical fields; only the
          evaluation order differs.

    [depth_sweep]
    default = True
    type = bool
    help = Sweep the antisymmetric null direction against transverse frequency
    doc = Tests Eq. (depthres) directly, with NA measured from the probe.

    [precond_test]
    default = True
    type = bool
    help = Measure how far the downstream multislice cascade is from unitary

    [device]
    default = cuda
    type = str
    help = Torch device

    [precision]
    default = double
    type = str
    help = 'single' or 'double'
    doc = Double by default: eps_M at M = N is a round-off quantity and single
          precision puts a floor near 1e-7 on it.
    """

    def __init__(self, ptycho_parent, pars=None):
        super(FirstBornDiagnostics, self).__init__(ptycho_parent, pars)
        self._done = False
        self._results = {}

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()
        if torch is None:
            raise ImportError("FirstBornDiagnostics requires PyTorch")

        N = int(self.p.number_of_slices)
        self.N = N

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
        self.M, self.fny, self.fnx = self._prs.data.shape
        # the object grid, needed by _collect_views for its bounds check
        self.ony, self.onx = (int(self._obs.data.shape[-2]),
                              int(self._obs.data.shape[-1]))

        self.outdir = str(self.p.outdir)
        os.makedirs(self.outdir, exist_ok=True)

        # ---- geometry and views first, so the reconstruction can be judged
        #      against the recorded data rather than against a file identity
        geom = list(self.ptycho.model.scans.values())[0].geometries[0]
        self.geom = geom
        shape, lam = geom.shape, geom.lam
        V, W = u.grids(shape, lam / shape / geom.resolution, 'fft')
        self._a2 = V ** 2 + W ** 2
        self.lam = _scalar(lam)
        self._Hcache = {}
        self._build_farfield()
        self._collect_views()

        # ---- the reconstruction under test ----------------------------
        # Every candidate probe is scored against every candidate slice stack
        # by how well the pair predicts the recorded frames, so neither the
        # file naming nor the .ptyr object container has to be trusted.
        logger.info("pairing reconstructions against the data:")
        fslices, ptyr, corr = self._resolve_pair(self.p.slices,
                                                 self.p.engine_name)
        self._slices_path, self._ptyr_path = fslices, ptyr
        self._results['data_correlation'] = corr
        logger.info("using %s with %s (correlation %.3f)"
                    % (os.path.basename(fslices),
                       os.path.basename(ptyr) if ptyr else 'the live P.probe',
                       corr))
        if corr < float(self.p.data_check_threshold):
            raise RuntimeError(
                "the best (slices, probe) pair predicts the recorded data at "
                "only %.3f over %d frames.\n  slices : %s\n  probe  : %s\n"
                "Either the reconstruction is not of this scan, or the "
                "geometry does not match. Lower data_check_threshold to "
                "proceed anyway." % (corr, int(self.p.check_frames),
                                     fslices, ptyr))
        pr = _read_ptyr(ptyr) if ptyr else dict(probe=None, obj=None)

        self._provenance(pr['obj'])
        self._probe_na()

        logger.info("diagnostics: N = %d, z = %s um, %d positions, %d probe modes"
                    % (self.N, np.round(self.z * 1e6, 3).tolist(),
                       self.n_pos, self.M))

    def _set_probe(self, probe):
        """Install a probe and drop anything cached from the previous one.

        phi, the unscattered exit field and the H stacks are all functions of
        the probe, N and z, so every assignment to any of the three has to
        invalidate them together. Forgetting this is silent: the null-space and
        depth tests measure differences, in which a stale unscattered term
        cancels, so a wrong cache would not show up in their numbers.
        """
        self.pr_g = (probe if torch.is_tensor(probe)
                     else torch.as_tensor(np.ascontiguousarray(probe),
                                          dtype=self.cdt, device=self.dev))
        self._illum_cache = None

    def _load_slices(self, slices, n=None):
        """Put a slice stack on the device and set the axial positions."""
        self._illum_cache = None          # z and N feed the H stacks
        self.N = N = len(slices) if n is None else n
        self.ob_g = torch.stack([torch.as_tensor(s, dtype=self.cdt,
                                                 device=self.dev)
                                 for s in slices])
        th = self.p.slice_thickness
        if type(th) in (list, tuple):
            self.z = np.concatenate([[0.0], np.cumsum(np.asarray(th, float))])[:N]
        else:
            self.z = np.arange(N, dtype=float) * float(th)

    def _data_correlation(self, n=None):
        """Amplitude correlation between the multislice prediction and the
        recorded frames, under the real mask. Near 1 means the probe, the
        slices, the positions and the geometry are all mutually consistent;
        near 0 means at least one of them is not."""
        n = int(self.p.check_frames if n is None else n)
        n = int(min(n, self.n_pos))
        idx = torch.arange(n, dtype=torch.long, device=self.dev)
        O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(self.N)])
        I = (self._ff(self._ff_fw, self._multislice(O, self.pr_g)).abs() ** 2).sum(1)
        cs = []
        for k in range(n):
            m = self.mask[k] > 0
            a = torch.sqrt(I[k][m].clamp_min(0)).double()
            b = torch.sqrt(self.I[k][m].clamp_min(0)).double()
            a = a - a.mean()
            b = b - b.mean()
            cs.append(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30)))
        return float(np.mean(cs))

    def _provenance(self, obj):
        """Informational only: what does the .ptyr's object group hold?

        engine_finalize is supposed to leave prod_j O_j in the object
        container, but whether it does depends on when the dump was written and
        on what the engine last wrote there, so this is NOT a test of whether
        the files belong together -- _data_correlation is. Reporting which
        candidate the container matches is still useful, because 'it holds
        slice 0 only' points at a bug in engine_finalize rather than at a
        mismatched pair.
        """
        if obj is None:
            return
        ob = self.ob_g.cpu().numpy()
        if obj.shape != ob[0].shape:
            logger.warning("provenance: .ptyr object is %s, the slices are %s"
                           % (obj.shape, ob[0].shape))
            return

        def resid(a):
            a, b = a.ravel(), obj.ravel()
            al = np.vdot(b, a) / max(np.vdot(b, b).real, 1e-30)
            return float(np.linalg.norm(a - al * b) / max(np.linalg.norm(a), 1e-30))

        prod = np.ones_like(ob[0])
        for s in ob:
            prod = prod * s
        cands = {'product of slices': resid(prod)}
        for j in range(self.N):
            cands['slice %d' % j] = resid(ob[j])
        best = min(cands, key=cands.get)
        self._results['provenance'] = {k: v for k, v in cands.items()}
        detail = ', '.join('%s %.3f' % (k, v) for k, v in cands.items())
        if cands[best] < 0.1:
            logger.info("provenance: the .ptyr object is the %s (%.3e)"
                        % (best, cands[best]))
        else:
            logger.info("provenance: the .ptyr object matches none of the "
                        "slices or their product (%s). That is a remark about "
                        "what the object container held when the dump was "
                        "written, not about this pair -- the data check above "
                        "is what establishes that. Worth checking "
                        "engine_finalize if you rely on the .ptyr object."
                        % detail)

    def _build_farfield(self):
        pod = next(list(v.pods.values())[0]
                   for v in self.di.views.values() if v.active)
        shape = (self.fny, self.fnx)
        for tag, spec in (('fw', _extract_operator(pod.fw, shape, "farfield fw")),
                          ('bw', _extract_operator(pod.bw, shape, "farfield bw"))):
            g = {'kind': spec['kind']}
            for k in ('H', 'A', 'R'):
                if k in spec:
                    g[k] = torch.as_tensor(spec[k], dtype=self.cdt, device=self.dev)
            setattr(self, '_ff_' + tag, g)

    def _collect_views(self):
        """The view set stochastic.EPIE walks, with data and masks."""
        names = sorted(n for n, v in self.di.views.items() if v.active)
        rows, cols, data, masks = [], [], [], []
        for n in names:
            pods = list(self.di.views[n].pods.values())
            lo = np.asarray(pods[0].ob_view.dlow, dtype=np.int64)
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))
            data.append(np.asarray(self.di.views[n].data, dtype=np.float64))
            m = pods[0].mask
            masks.append(np.ones_like(data[-1]) if m is None
                         else np.asarray(m, dtype=np.float64))
        self.n_pos = len(names)
        self.view_names = names
        self.rows = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.I = torch.as_tensor(np.stack(data), dtype=self.rdt, device=self.dev)
        self.mask = torch.as_tensor(np.stack(masks), dtype=self.rdt, device=self.dev)
        if (int(self.rows.max()) >= self.ony or int(self.cols.max()) >= self.onx):
            raise RuntimeError("patch extends outside the object array; the "
                               "slice file does not match this scan")

    def _probe_na(self):
        """NA from the reconstructed probe, not from the sampling limit.

        The Nyquist NA of the grid is not the illumination NA and would give a
        meaningless shear in Eq. (depthres). This takes the radius containing
        90% of the probe power in the angular spectrum.
        """
        F = torch.fft.fft2(self.pr_g)
        P = (F.abs() ** 2).sum(0).cpu().numpy()
        a = np.sqrt(np.clip(self._a2, 0, None))           # = lam * |f|
        order = np.argsort(a.ravel())
        c = np.cumsum(P.ravel()[order])
        c /= max(c[-1], 1e-30)
        self.na = float(a.ravel()[order][np.searchsorted(c, 0.90)])
        dz = float(self.z[-1] - self.z[0]) if self.N > 1 else 0.0
        self.a_c = 2.0 * self.na * dz
        logger.info("probe NA (90%% power) = %.3f mrad, shear a_c = 2 NA dz "
                    "= %.3g m = %.1f object pixels"
                    % (self.na * 1e3, self.a_c,
                       self.a_c / _scalar(self.geom.resolution)))

    # ------------------------------------------------------------------ #
    #  Operators                                                          #
    # ------------------------------------------------------------------ #
    def _H(self, dz):
        key = round(float(dz), 15)
        H = self._Hcache.get(key)
        if H is None:
            kz = np.sqrt(np.maximum(1.0 - self._a2, 0.0)) - 1.0
            Hn = np.exp(2j * np.pi * (key / self.lam) * kz)
            Hn[self._a2 > 1.0] = 0.0                       # |H| = 1 exactly
            H = torch.as_tensor(Hn, dtype=self.cdt, device=self.dev)
            self._Hcache[key] = H
        return H

    def _prop(self, w, dz):
        if float(dz) == 0.0:
            return w
        return torch.fft.ifft2(self._H(dz) * torch.fft.fft2(w))

    def _ff(self, spec, w):
        k = spec['kind']
        if k == 'lsi':
            return torch.fft.ifft2(torch.fft.fft2(w) * spec['H'])
        if k == 'ilsi':
            return torch.fft.fft2(torch.fft.ifft2(w) * spec['H'])
        if k == 'chirp':
            return spec['A'] * torch.fft.fft2(spec['R'] * w)
        return spec['A'] * torch.fft.ifft2(spec['R'] * w)

    def _gather(self, arr, idx):
        return arr[self.rows[idx][:, :, None], self.cols[idx][:, None, :]]

    # -- forward models, all returning the field at the plane z_{N-1} ----
    def _multislice(self, O, P):
        psi = O[0][:, None] * P[None]
        for j in range(1, self.N):
            psi = self._prop(psi, self.z[j] - self.z[j - 1]) * O[j][:, None]
        return psi

    def _illumination(self, P):
        return torch.stack([self._prop(P, zj) for zj in self.z])

    def _born(self, g, P, phi=None):
        """Eq. (model). Identical field either way; see born_mode."""
        if phi is None:
            phi = self._illum(P)[0]
        if str(self.p.born_mode).startswith('par'):
            return self._born_parallel(g, P, phi)
        return self._born_lowmem(g, P, phi)

    def _blocked(self, g, P, n_blocks):
        N = self.N
        nb = int(max(1, min(n_blocks, N)))
        blocks = np.array_split(np.arange(N), nb)
        psi = P[None]
        z_in = self.z[0]
        for b, ids in enumerate(blocks):
            z_out = self.z[blocks[b + 1][0]] if b + 1 < len(blocks) else self.z[-1]
            out = self._prop(psi, z_out - z_in)
            for j in ids:
                inc = self._prop(psi, self.z[j] - z_in)
                out = out + self._prop(g[j][:, None] * inc, z_out - self.z[j])
            psi, z_in = out, z_out
        return psi

    def _orders(self, O, P, max_order):
        """Exact scattering-order decomposition by DFT over the roots of unity."""
        N, K = self.N, self.N + 1
        g = O - 1.0
        max_order = int(min(max_order, N))
        orders = psi_ms = None
        for k in range(K):
            t = complex(np.exp(2j * np.pi * k / K))
            tt = torch.as_tensor(t, dtype=self.cdt, device=self.dev)
            psik = self._multislice(1.0 + tt * g, P)
            if k == 0:
                psi_ms = psik.clone()                      # t = 1
                orders = torch.zeros((max_order + 1,) + tuple(psik.shape),
                                     dtype=self.cdt, device=self.dev)
            for m in range(max_order + 1):
                w = torch.as_tensor(complex(np.conj(t) ** m) / K,
                                    dtype=self.cdt, device=self.dev)
                orders[m] += w * psik
            del psik
        return psi_ms, orders

    # ------------------------------------------------------------------ #
    #  Iteration: run once, then stop                                     #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        if not self._done:
            self._run_all()
            self._done = True
        # nothing left to do; end the run whatever numiter says
        self.curiter = max(self.curiter + max(1, num), int(self.p.numiter))
        e = self._results.get('eps_order', {}).get(1, 0.0)
        return {'all': np.array([e, self._results.get('eps_linearised', 0.0), 0.0])}

    def engine_finalize(self):
        path = os.path.join(self.outdir, 'results.json')
        with open(path, 'w') as fh:
            json.dump(self._results, fh, indent=2, default=_jsonable)
        logger.info("diagnostics written to %s" % self.outdir)
        return super().engine_finalize()

    # ------------------------------------------------------------------ #
    #  The measurements                                                   #
    # ------------------------------------------------------------------ #
    def _run_all(self):
        t0 = datetime.now()
        self._results['run'] = dict(
            when=t0.isoformat(timespec='seconds'),
            n_slices=self.N, n_positions=self.n_pos, n_probe_modes=self.M,
            frame=[self.fny, self.fnx],
            z_um=(self.z * 1e6).tolist(),
            lam_A=self.lam * 1e10,
            energy_keV=1.23984193e-9 / self.lam,
            resolution_nm=_scalar(self.geom.resolution) * 1e9,
            na_mrad=self.na * 1e3, a_c_m=self.a_c,
            precision=str(self.p.precision))

        # Is the reconstruction anywhere near the natural gauge? The per-slice
        # mean transmission is not determined by the data (Eq. null-const), and
        # a reconstruction free to wander in it will. Phi_tot, |g|, the mean
        # modulus and eps_M itself are all measured from that offset, so a
        # stack carrying a constant phase of a couple of radians reports a
        # projected phase of several radians and an eps_1 above one, none of
        # which describes the specimen.
        cov = self._coverage()
        means = [complex(self.ob_g[j][cov].mean().cpu()) for j in range(self.N)]
        offs = [abs(np.angle(m)) for m in means]
        self._results['run']['slice_mean_phase_rad'] = [float(np.angle(m))
                                                        for m in means]
        self._results['run']['slice_mean_modulus'] = [float(abs(m))
                                                      for m in means]
        gauges = (['none', 'mean'] if str(self.p.gauge) == 'both'
                  else [str(self.p.gauge)])
        if max(offs) > 0.2:
            logger.warning("the slices carry a per-slice mean phase of %s rad, "
                           "so the as-reconstructed gauge is not the natural "
                           "one; quote the mean-gauge numbers"
                           % np.round(offs, 3).tolist())
            if 'mean' in gauges:
                gauges = ['mean'] + [g for g in gauges if g != 'mean']
        base_ob = self.ob_g.clone()
        base_pr = self.pr_g.clone()

        for gauge in gauges:
            self.ob_g = base_ob.clone()
            self._set_probe(base_pr.clone())
            alpha = self._apply_gauge(gauge)
            res = self._scan(gauge)
            res['alpha'] = [complex(a) for a in alpha]
            self._results['gauge_' + gauge] = res
            if gauge == gauges[0]:
                # promote the primary gauge to the top level for convenience
                self._results['primary_gauge'] = gauge
                self._results['eps_order'] = res['eps_order']
                self._results['eps_linearised'] = res['eps_linearised']

        # Structural tests. The null-space and depth tests measure ratios of
        # responses and are gauge insensitive; the preconditioner test reads
        # |O_j| directly and is not, so it runs on the gauge-fixed stack --
        # otherwise a per-slice modulus of 0.83 that is pure gauge reads as
        # 17% absorption.
        self.ob_g = base_ob.clone()
        self._set_probe(base_pr.clone())
        self._apply_gauge('mean')
        gauge_fixed_ob = self.ob_g.clone()
        self.ob_g = base_ob
        self._set_probe(base_pr)
        self._results['null_space'] = self._null_test()
        if bool(self.p.depth_sweep) and self.N > 1:
            self._results['depth_separability'] = self._depth_sweep()
        if bool(self.p.precond_test) and self.N > 1:
            keep = self.ob_g
            self.ob_g = gauge_fixed_ob
            self._results['preconditioner'] = self._precond_test()
            self.ob_g = keep

        # ---- the same measurement on other reconstructions of this scan ----
        comp = self.p.compare
        if comp:
            comp = [comp] if isinstance(comp, str) else list(comp)
            primary_ob = self.ob_g.clone()
            primary_pr = self.pr_g.clone()
            primary_N, primary_z = self.N, self.z
            for tag in comp:
                try:
                    logger.info("comparison reconstruction '%s':" % tag)
                    sl, pt, cc = self._resolve_pair(tag, None)
                    if cc < float(self.p.data_check_threshold):
                        logger.warning("  '%s' predicts the data at only %.3f;"
                                       " skipping" % (tag, cc))
                        continue
                    logger.info("  %s with %s (correlation %.3f)"
                                % (os.path.basename(sl),
                                   os.path.basename(pt) if pt else 'live probe',
                                   cc))
                    # Always the mean gauge here. Two reconstructions sit in
                    # whatever gauge their engine drifted into (Eq. null-const
                    # leaves it free), and eps, Phi_tot and |g| are all
                    # measured from that offset, so comparing them in their own
                    # gauges compares the offsets and not the specimens.
                    self._apply_gauge('mean')
                    res = self._scan('mean')
                    res['gauge_used'] = 'mean'
                    res['slices'] = sl
                    res['ptyr'] = pt
                    res['data_correlation'] = cc
                    res['object_vs_primary'] = self._object_difference(
                        primary_ob.cpu().numpy())
                    self._results['compare_' + str(tag)] = res
                except Exception as e:                      # noqa: BLE001
                    logger.warning("  comparison '%s' failed: %s" % (tag, e))
            self.ob_g = primary_ob
            self._set_probe(primary_pr)
            self.N, self.z = primary_N, primary_z

        if bool(self.p.benchmark):
            self._results['benchmark'] = self._benchmark()

        self._figures()
        self._report()
        self._results['run']['seconds'] = (datetime.now() - t0).total_seconds()

    def _resolve_pair(self, tag=None, engine=None, verbose=True):
        """Find a (slice stack, probe) pair and score it against the data.

        A results folder can hold several reconstructions of the same scan at
        the same iteration --- a first-Born one and a 3PIE one, say --- and
        several .ptyr dumps to go with them. Rather than trust a naming
        convention to pair them, every candidate probe is tried against the
        chosen slice stack and the pair that best predicts the recorded frames
        wins. That is the same test the startup check uses, and it is decisive
        where a filename is not.

        Returns (slices_path, ptyr_path, correlation), with the winning pair
        already installed.
        """
        gt = self.p.ground_truth
        cands = _slice_candidates(gt, None)
        if tag is None:
            # never let the primary silently land on a reconstruction that is
            # meant to be the control: comparing a file with itself returns a
            # ratio of exactly 1 and looks like a result
            for other in ([self.p.compare] if isinstance(self.p.compare, str)
                          else list(self.p.compare or [])):
                cands = [c for c in cands
                         if str(other).lower() not in os.path.basename(c).lower()]
        if tag:
            tag = str(tag)
            if os.path.isfile(tag):
                cands = [tag]
            else:
                cands = [c for c in cands
                         if tag.lower() in os.path.basename(c).lower()]
            if not cands:
                raise RuntimeError("no slice file matches '%s' under %s"
                                   % (tag, gt))
        if self.p.iteration:
            want = int(self.p.iteration)
            sel = [c for c in cands if _slice_iter(c) == want]
            if not sel:
                raise RuntimeError("no slice file at iteration %d under %s%s"
                                   % (want, gt,
                                      " matching '%s'" % tag if tag else ""))
            cands = sel
        cands = sorted(cands, key=_slice_iter)[::-1][:4]

        probes = []
        if self.p.ptyr and os.path.isfile(str(self.p.ptyr)):
            probes = [str(self.p.ptyr)]
        elif gt and os.path.isdir(gt):
            # every dump is a candidate, whatever engine wrote it: the probe
            # that best predicts the data is the right probe, and restricting
            # the pool by engine name would have hidden that the 200-iteration
            # probe beats the 500-iteration one
            probes = sorted(glob.glob(os.path.join(gt, '*.ptyr')),
                            key=_iter_of)[::-1]
        if not probes:
            probes = [None]

        # A probe from one run and slices from another is not a
        # reconstruction: no engine ever produced that state, so its eps and
        # chi^2 describe nothing that exists. Prefer the probe whose iteration
        # matches the slice file's, and only fall back to the whole pool when
        # there is no such probe. Agreement with the data is then a check on
        # the pair, not the thing that selects it.
        same_run = str(self.p.pair_mode).startswith('same')
        best = (None, None, -2.0)
        ranked = []
        for c in cands:
            try:
                slices = _read_slices(c)
            except Exception as e:                          # noqa: BLE001
                logger.warning("  %s unreadable (%s)" % (os.path.basename(c), e))
                continue
            if slices[0].shape != (self.ony, self.onx):
                continue
            self._load_slices(slices)
            pool = probes
            if same_run:
                matched = [p for p in probes if _iter_of(p) == _slice_iter(c)]
                if matched:
                    pool = matched
                    # within one iteration a folder may still hold dumps from
                    # several engines; prefer the one named, if any match
                    named = [p for p in pool if engine
                             and str(engine).lower() in _engine_of(p).lower()]
                    if named:
                        pool = named
                elif verbose:
                    logger.warning("    no probe at iteration %d for %s; "
                                   "falling back to the whole pool, so the "
                                   "pair may mix two runs"
                                   % (_slice_iter(c), os.path.basename(c)))
            for pp in pool:
                pr = _read_ptyr(pp)['probe'] if pp else self._prs.data
                if pr is None or pr.shape[-2:] != (self.fny, self.fnx):
                    continue
                self._set_probe(pr)
                corr = self._data_correlation()
                ranked.append((corr, c, pp))
                if corr > best[2]:
                    best = (c, pp, corr)
        if verbose and ranked:
            ranked.sort(reverse=True)
            logger.info("    best pairings by agreement with the data:")
            for corr, c, pp in ranked[:5]:
                logger.info("      %6.3f  %s + %s"
                            % (corr, os.path.basename(c),
                               os.path.basename(pp) if pp else 'live P.probe'))
        if best[0] is None:
            raise RuntimeError("no usable (slices, probe) pair under %s" % gt)
        # reinstall the winner
        self._load_slices(_read_slices(best[0]))
        self._set_probe(_read_ptyr(best[1])['probe'] if best[1]
                        else self._prs.data)
        self.M = int(self.pr_g.shape[0])
        return best

    def _object_difference(self, other):
        """How far apart are two reconstructions of the same scan?

        eps_M compares two forward models on one object. This compares two
        objects, which is the question a reader actually has: does fitting with
        the truncated model change the answer? Both stacks are put in the same
        gauge first, since the per-slice mean is not determined by the data
        (Eq. null-const), and the comparison is restricted to the illuminated
        region.
        """
        cov = self._coverage().cpu().numpy()
        a = self.ob_g.cpu().numpy()
        b = np.asarray(other)
        if b.shape != a.shape:
            return None
        out = []
        for j in range(a.shape[0]):
            x, y = a[j][cov], b[j][cov]
            x = x / (x.mean() if abs(x.mean()) > 1e-9 else 1.0)
            y = y / (y.mean() if abs(y.mean()) > 1e-9 else 1.0)
            al = np.vdot(y, x) / max(np.vdot(y, y).real, 1e-30)
            rel = float(np.linalg.norm(x - al * y) / max(np.linalg.norm(x), 1e-30))
            dphi = np.angle(x) - np.angle(al * y)
            dphi -= dphi.mean()
            out.append(dict(slice=j, relative_difference=rel,
                            phase_rms_rad=float(np.sqrt((dphi ** 2).mean())),
                            phase_ptp_rad=float(np.ptp(np.angle(x)))))
        return out

    def _coverage(self):
        """Illuminated region of the object grid."""
        cov = torch.zeros((self.ony, self.onx), dtype=torch.bool, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            r = self.rows[idx][:, :, None].expand(-1, self.fny, self.fnx)
            c = self.cols[idx][:, None, :].expand(-1, self.fny, self.fnx)
            cov[r.reshape(-1), c.reshape(-1)] = True
        return cov

    def _apply_gauge(self, gauge):
        """Divide each slice by its complex mean over the illuminated region and
        push the product onto the probe. Multislice is pointwise invariant, so
        this is a pure choice of the gauge of Eq. (null-const); the Born
        prediction is not invariant, which is why both are reported."""
        if gauge != 'mean':
            return np.ones(self.N, dtype=complex)
        cov = self._coverage()
        al = []
        for j in range(self.N):
            m = self.ob_g[j][cov].mean()
            if float(m.abs()) < 1e-9:
                m = torch.as_tensor(1.0 + 0j, dtype=self.cdt, device=self.dev)
            self.ob_g[j] = self.ob_g[j] / m
            al.append(complex(m.cpu()))
        self._set_probe(self.pr_g * torch.as_tensor(np.prod(al), dtype=self.cdt,
                                                    device=self.dev))
        return np.array(al)

    def _scan(self, gauge):
        """One pass over the scan, accumulating every error measure."""
        N = self.N
        mo = int(min(self.p.max_order, N))
        blocks = self.p.blocks
        if blocks is None:
            blocks = list(range(1, N + 1))
        elif isinstance(blocks, int):
            blocks = [blocks]
        blocks = sorted({int(b) for b in blocks if 1 <= int(b) <= N})

        B = int(self.p.batch_size)
        sel = np.arange(self.n_pos)
        if self.p.n_positions:
            sel = np.linspace(0, self.n_pos - 1, int(self.p.n_positions)).astype(int)

        onum, oden = np.zeros(mo + 1), np.zeros(mo + 1)
        bnum = {b: 0.0 for b in blocks}
        bden = {b: 0.0 for b in blocks}
        lnum = lden = lvb_n = lvb_d = 0.0
        # the same hierarchy built on the tangent perturbation
        lorder_num, lorder_den = np.zeros(mo + 1), np.zeros(mo + 1)
        lblk_num = {b: 0.0 for b in blocks}
        lblk_den = {b: 0.0 for b in blocks}
        msl_num = msl_den = 0.0             # its M = N member
        gre = gabs = gw = 0.0
        chi_ms = chi_b = chi_lin = npix = 0.0
        e1, e1l, phis = [], [], []
        rad = None
        example = None

        phi = self._illumination(self.pr_g)          # once: object free
        pw = (self.pr_g.abs() ** 2).sum(0)

        counts = (float(self.p.counts) if self.p.counts
                  else float(np.median((self.I * self.mask).sum(dim=(1, 2)).cpu().numpy())))

        for s in range(0, len(sel), B):
            idx = torch.as_tensor(sel[s:s + B], dtype=torch.long, device=self.dev)
            O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(N)])
            g = O - 1.0

            psi_ms, orders = self._orders(O, self.pr_g, mo)
            F_ms = self._ff(self._ff_fw, psi_ms)
            I_ms = (F_ms.abs() ** 2).sum(1)

            cum = torch.zeros_like(orders[0])
            F1 = None
            for m in range(mo + 1):
                cum = cum + orders[m]
                F_m = self._ff(self._ff_fw, cum)
                d = F_ms - F_m
                onum[m] += float((d.abs() ** 2).sum())
                oden[m] += float((F_m.abs() ** 2).sum())
                if m == 1:
                    F1 = F_m
                    pn = (d.abs() ** 2).sum(dim=(1, 2, 3))
                    pd = (F_m.abs() ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)
                    e1.append(np.sqrt((pn / pd).cpu().numpy()))

            del orders

            # --- the same hierarchy built on the tangent perturbation -----
            # 1 + i s_j = 1 + log O_j in place of O_j. Its M = 1 member is the
            # tangent first Born; its M = N member is the full cascade with the
            # linearised slices,
            #     (1 + i s_{N-1}) P_d ... (1 + i s_1) P_d (1 + i s_0) P,
            # propagators and all -- an ordered product, not a pointwise one.
            # It does NOT return to round-off: every propagator is exact and
            # every scattering order is retained, so what is left is purely the
            # intra-slice linearisation. The chord expansion has no such floor,
            # because O_j - 1 is the whole transmission.
            psi_ms_lin, orders_l = self._orders(1.0 + torch.log(O), self.pr_g, mo)
            cum = torch.zeros_like(orders_l[0])
            F_lin = None
            for m in range(mo + 1):
                cum = cum + orders_l[m]
                F_m = self._ff(self._ff_fw, cum)
                d = F_ms - F_m
                lorder_num[m] += float((d.abs() ** 2).sum())
                lorder_den[m] += float((F_m.abs() ** 2).sum())
                if m == 1:
                    F_lin = F_m
                    pn = (d.abs() ** 2).sum(dim=(1, 2, 3))
                    pd = (F_m.abs() ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)
                    e1l.append(np.sqrt((pn / pd).cpu().numpy()))
            del orders_l

            # tangent multislice: the last member, taken directly at t = 1 so
            # it is available even when max_order < N
            F_ms_lin = self._ff(self._ff_fw, psi_ms_lin)
            msl_num += float(((F_ms - F_ms_lin).abs() ** 2).sum())
            msl_den += float((F_ms_lin.abs() ** 2).sum())

            # blocked truncation on the tangent perturbation, for the same
            # reason: within a block it is first Born in i s rather than in
            # O - 1, so the block boundaries do not remove the residual either
            for b in blocks:
                F_lb = self._ff(self._ff_fw,
                                self._blocked(torch.log(O), self.pr_g, b))
                lblk_num[b] += float(((F_ms - F_lb).abs() ** 2).sum())
                lblk_den[b] += float((F_lb.abs() ** 2).sum())

            lnum += float(((F_ms - F_lin).abs() ** 2).sum())
            lden += float((F_lin.abs() ** 2).sum())
            lvb_n += float(((F1 - F_lin).abs() ** 2).sum())
            lvb_d += float((F1.abs() ** 2).sum())

            gd = (g - torch.log(O))
            w = pw[None, None]
            gre += float((gd.real.abs() * w).sum())
            gabs += float((g.abs() * w).sum())
            gw += float(w.sum()) * N * len(idx)



            for b in blocks:
                F_b = self._ff(self._ff_fw, self._blocked(g, self.pr_g, b))
                bnum[b] += float(((F_ms - F_b).abs() ** 2).sum())
                bden[b] += float((F_b.abs() ** 2).sum())


            ph = torch.stack([torch.angle(O[j]) for j in range(N)]).sum(0)
            phis.append(np.sqrt(((ph ** 2 * pw[None]).sum(dim=(1, 2))
                                 / pw.sum().clamp_min(1e-30)).cpu().numpy()))

            # against the data, under the real mask and at the recorded flux
            I_b = (F1.abs() ** 2).sum(1)
            I_l = (F_lin.abs() ** 2).sum(1)
            Im, mk = self.I[idx], self.mask[idx]
            for tgt, acc in ((I_ms, 'ms'), (I_b, 'b'), (I_l, 'lin')):
                sc = (mk * Im).sum(dim=(1, 2)) / (mk * tgt).sum(dim=(1, 2)).clamp_min(1e-30)
                r = mk * (tgt * sc[:, None, None] - Im) ** 2 / (Im + 1.0)
                v = float(r.sum())
                if acc == 'ms':
                    chi_ms += v
                elif acc == 'b':
                    chi_b += v
                else:
                    chi_lin += v
            npix += float(mk.sum())

            if rad is None:
                rad = self._radial(I_ms[0], I_b[0], Im[0], mk[0], counts)
                example = self._example(I_ms, I_b, Im, mk, counts)
            del O, g, psi_ms, psi_ms_lin, cum

        e1 = np.concatenate(e1)
        e1l = np.concatenate(e1l)
        phis = np.concatenate(phis)
        phi_rms = float(np.sqrt((phis ** 2).mean()))

        eps_order = {m: float(np.sqrt(onum[m] / max(oden[m], 1e-300)))
                     for m in range(mo + 1)}
        out = dict(
            gauge=gauge,
            eps_order=eps_order,
            eps_block={b: float(np.sqrt(bnum[b] / max(bden[b], 1e-300)))
                       for b in blocks},
            eps_linearised=float(np.sqrt(lnum / max(lden, 1e-300))),
            eps_order_tangent={m: float(np.sqrt(lorder_num[m]
                                                / max(lorder_den[m], 1e-300)))
                               for m in range(mo + 1)},
            eps_block_tangent={b: float(np.sqrt(lblk_num[b]
                                                / max(lblk_den[b], 1e-300)))
                               for b in blocks},
            eps_multislice_tangent=float(np.sqrt(msl_num / max(msl_den, 1e-300))),
            eps_linearised_vs_born=float(np.sqrt(lvb_n / max(lvb_d, 1e-300))),
            g_real_gap=float(gre / max(gw, 1e-300)),
            g_mean_modulus=float(gabs / max(gw, 1e-300)),
            phi_tot_rms=phi_rms,
            eps1_estimate_correlated=float(0.5 * phi_rms ** 2 * (1 - 1.0 / self.N)),
            eps1_estimate_decorrelated=float(phi_rms ** 2 / (np.sqrt(2) * self.N)),
            counts_per_frame=counts,
            counts_per_pixel=counts / float(self.fny * self.fnx),
            counts_visible_above=float(1.0 / (4 * max(eps_order.get(1, 1e-30), 1e-30) ** 2)),
            chi2_multislice=float(chi_ms / max(npix, 1)),
            chi2_born=float(chi_b / max(npix, 1)),
            chi2_linearised=float(chi_lin / max(npix, 1)),
            eps1_per_pos=e1, eps1lin_per_pos=e1l, phi_per_pos=phis,
            radial=rad, example=example)
        return out

    # -- structural tests ------------------------------------------------
    def _null_test(self, n=8):
        """Eq. (null-const): a constant on slice j contributes a_j F[P], so the
        difference of constants on two slices is exactly null while their sum is
        not. Measured against a random direction of the same norm."""
        idx = torch.arange(min(n, self.n_pos), dtype=torch.long, device=self.dev)
        phi = self._illumination(self.pr_g)
        B = idx.numel()
        shape = (self.N, B, self.fny, self.fnx)

        def resp(d):
            f = self._ff(self._ff_fw, self._born(d, self.pr_g, phi))
            f0 = self._ff(self._ff_fw, self._born(torch.zeros_like(d), self.pr_g, phi))
            return float((f - f0).abs().pow(2).sum().sqrt()) / float(
                d.abs().pow(2).sum().sqrt().clamp_min(1e-30))

        one = torch.ones(shape, dtype=self.cdt, device=self.dev)
        diff = one.clone()
        diff[1:] = -one[1:] / max(self.N - 1, 1)          # sum of coefficients 0
        rng = torch.randn(shape, dtype=torch.float64, device=self.dev)
        rand = (rng + 1j * torch.randn_like(rng)).to(self.cdt)

        r_sum, r_diff, r_rand = resp(one), resp(diff), resp(rand)
        out = dict(response_sum=r_sum, response_difference=r_diff,
                   response_random=r_rand,
                   difference_over_random=r_diff / max(r_rand, 1e-30))
        logger.info("null space: constant-difference response is %.2e of a "
                    "random direction (Eq. null-const predicts 0)"
                    % out['difference_over_random'])
        return out

    def _depth_sweep(self, n=4, nq=24):
        """Eq. (depthres): sweep an antisymmetric transverse modulation.

        delta = +exp(2 pi i q x) on slice 0 and -exp(...) on the last slice is
        invisible when the two slices see the same illumination, i.e. when the
        feature is coarser than the shear a_c = 2 NA dz, and becomes visible
        above q_c = 1/a_c. The crossing is the measurement.
        """
        idx = torch.arange(min(n, self.n_pos), dtype=torch.long, device=self.dev)
        phi = self._illumination(self.pr_g)
        B = idx.numel()
        dx = _scalar(self.geom.resolution)
        x = torch.arange(self.fnx, device=self.dev, dtype=torch.float64) * dx
        qs = np.logspace(np.log10(0.05 / max(self.a_c, 1e-9)),
                         np.log10(20.0 / max(self.a_c, 1e-9)), nq)
        qs = qs[qs < 0.5 / dx]                              # below Nyquist
        base = self._ff(self._ff_fw, self._born(
            torch.zeros((self.N, B, self.fny, self.fnx), dtype=self.cdt,
                        device=self.dev), self.pr_g, phi))
        sym, anti = [], []
        for q in qs:
            wave = torch.exp(2j * np.pi * torch.as_tensor(q, device=self.dev,
                                                          dtype=torch.float64) * x)
            wave = wave.to(self.cdt)[None, None, None, :].expand(
                1, B, self.fny, self.fnx)
            d_s = torch.cat([wave] * self.N, dim=0).clone()
            d_a = d_s.clone()
            d_a[1:] = -d_a[1:] / max(self.N - 1, 1)
            for tgt, store in ((d_s, sym), (d_a, anti)):
                f = self._ff(self._ff_fw, self._born(tgt, self.pr_g, phi))
                store.append(float((f - base).abs().pow(2).sum().sqrt()))
        ratio = np.array(anti) / np.maximum(np.array(sym), 1e-30)
        out = dict(q=qs.tolist(), response_sym=sym, response_anti=anti,
                   ratio=ratio.tolist(), q_c=float(1.0 / max(self.a_c, 1e-30)),
                   a_c_m=self.a_c, na=self.na)
        logger.info("depth separability: q_c = 1/a_c = %.3g 1/m; antisymmetric "
                    "response crosses half its plateau near %.3g 1/m"
                    % (out['q_c'], _halfway(qs, ratio)))
        out['q_half'] = float(_halfway(qs, ratio))
        return out

    def _precond_test(self, n=4, probes=4):
        """Sec. (precond): the first-Born normal operator is exactly
        diag(|phi_j|^2). The multislice one is
        diag(conj psi_inc) T_{>j}^dag T_{>j} diag(psi_inc), which reduces to the
        same thing only when the downstream cascade is unitary, i.e. only when
        no downstream slice absorbs. This measures the departure."""
        idx = torch.arange(min(n, self.n_pos), dtype=torch.long, device=self.dev)
        O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(self.N)])
        rng = torch.randn((probes, idx.numel(), self.fny, self.fnx),
                          dtype=torch.float64, device=self.dev)
        v = (rng + 1j * torch.randn_like(rng)).to(self.cdt)

        # downstream cascade from slice 0: T_{>0} = O_{N-1} P ... O_1 P
        def T(w):
            for j in range(1, self.N):
                w = self._prop(w, self.z[j] - self.z[j - 1]) * O[j]
            return w

        def Tadj(w):
            for j in range(self.N - 1, 0, -1):
                w = self._prop(w * O[j].conj(), -(self.z[j] - self.z[j - 1]))
            return w

        dev = []
        for k in range(probes):
            w = v[k]
            r = Tadj(T(w)) - w
            dev.append(float(r.abs().pow(2).sum().sqrt()
                             / w.abs().pow(2).sum().sqrt().clamp_min(1e-30)))
        mod = torch.stack([O[j].abs() for j in range(1, self.N)])
        out = dict(nonunitarity=float(np.mean(dev)),
                   downstream_modulus_mean=float(mod.mean()),
                   downstream_modulus_min=float(mod.min()),
                   downstream_absorption=float(1.0 - mod.mean()))
        logger.info("preconditioner: ||T^dag T - I|| / ||I|| = %.3e at mean "
                    "downstream |O| = %.4f (the first-Born normal operator is "
                    "exactly diagonal regardless)"
                    % (out['nonunitarity'], out['downstream_modulus_mean']))
        return out

    # -- small helpers ---------------------------------------------------
    def _radial(self, I_ms, I_b, Im, mk, counts):
        """Radial profiles: where in q the truncation error actually lives."""
        ny, nx = self.fny, self.fnx
        yy, xx = np.mgrid[:ny, :nx]
        cy, cx = ny // 2, nx // 2                      # ptypy centres the frame
        r = np.hypot(yy - cy, xx - cx).astype(int)
        nb = min(cy, cx)
        ms = I_ms.cpu().numpy()
        bn = I_b.cpu().numpy()
        me = (Im * mk).cpu().numpy()
        sc = me.sum() / max(ms.sum(), 1e-30)
        prof = {}
        for k, a in (('multislice', ms * sc), ('born', bn * sc),
                     ('measured', me), ('difference', np.abs(ms - bn) * sc)):
            prof[k] = np.bincount(r.ravel(), a.ravel(),
                                  minlength=nb + 1)[:nb] / np.maximum(
                np.bincount(r.ravel(), minlength=nb + 1)[:nb], 1)
        psz = _scalar(self.geom.psize)
        q = np.arange(nb) * psz / _scalar(self.geom.distance) / self.lam
        return dict(q=q.tolist(), **{k: v.tolist() for k, v in prof.items()})

    def _example(self, I_ms, I_b, Im, mk, counts):
        sc = float((mk[0] * Im[0]).sum() / (mk[0] * I_ms[0]).sum().clamp_min(1e-30))
        return dict(I_ms=(I_ms[0] * sc).cpu().numpy(),
                    I_born=(I_b[0] * sc).cpu().numpy(),
                    I_meas=(Im[0] * mk[0]).cpu().numpy(),
                    mask=mk[0].cpu().numpy())

    # ------------------------------------------------------------------ #
    #  Cost: time and peak memory                                         #
    # ------------------------------------------------------------------ #
    def _illum(self, P):
        """phi_j and the unscattered exit field, cached.

        Neither carries the object or the scan position, so in a reconstruction
        they are formed once per mini-batch; here the probe never changes, so
        they are formed once for the whole run. This is the amortisation of
        Sec. (cost), and it has no counterpart in multislice, whose incident
        field at slice j carries every slice above it.
        """
        if getattr(self, '_illum_cache', None) is None:
            phi = torch.stack([self._prop(P, zj) for zj in self.z])
            # H_{-z} = conj(H_z) exactly, since |H| = 1, so the backward stack
            # is not stored: it is a view-free conjugate of the forward one.
            # That is N frames of the O(N) cache saved for nothing.
            Hb = torch.stack([self._H(self.z[-1] - zj) for zj in self.z])
            self._illum_cache = (phi, self._prop(P, self.z[-1]), Hb, None)
        return self._illum_cache

    def _born_parallel(self, g, P, phi):
        """Eq. (model) with every slice in flight at once.

        The branches are independent, so all N go through one batched
        transform: N forward transforms and a single inverse, not one
        propagation per slice. Fastest, but it materialises an
        (N, batch, modes, ny, nx) tensor.
        """
        _, Pex, Hb, _ = self._illum(P)
        S = torch.fft.fft2(g[:, :, None] * phi[:, None])
        S *= Hb[:, None, None]
        return torch.fft.ifft2(S.sum(0)) + Pex[None]

    def _born_lowmem(self, g, P, phi):
        """The same sum accumulated one slice at a time, in Fourier space.

        Each branch costs one forward transform; the accumulator is summed in
        the Fourier domain and inverted once at the end, so the count is N + 1
        rather than 2N. Only the accumulator ever exists, so peak storage is
        O(batch) whatever N is, and no intermediate is kept for the gradient.
        Multislice has no equivalent: its slice j needs the field slice j-1
        produced, so the N wavefields must be held.
        """
        _, Pex, Hb, _ = self._illum(P)
        acc = None
        for j in range(self.N):
            F = torch.fft.fft2(g[j][:, None] * phi[j][None])
            F *= Hb[j]
            if acc is None:
                acc = F
            else:
                acc.add_(F)
        return torch.fft.ifft2(acc) + Pex[None]

    def _bench_multislice(self, O, P):
        """Forward and adjoint, keeping what the gradient needs.

        A multislice engine must retain every intermediate wavefield to form
        the object gradient, which is the O(NB) storage in Table 1.
        """
        psis = []
        psi = P[None].expand(O.shape[1], *P.shape)
        for j in range(self.N):
            psis.append(psi)
            e = psi * O[j][:, None]
            if j < self.N - 1:
                psi = self._prop(e, self.z[j + 1] - self.z[j])
        F = self._ff(self._ff_fw, e)
        back = self._ff(self._ff_bw, F)                  # stand-in residual
        acc = 0.0
        for j in range(self.N - 1, -1, -1):
            delta = back - psis[j] * O[j][:, None]
            acc = acc + (psis[j].conj() * delta).sum(1).abs().sum()
            if j > 0:
                back = self._prop(back, -(self.z[j] - self.z[j - 1]))
        return acc

    def _bench_born(self, g, P, parallel):
        """Forward and adjoint for Eq. (model), in either evaluation mode.

        The adjoint gets the same treatment as the forward: the detector
        residual is transformed once and the per-slice back-propagation is a
        diagonal multiply on it, so the count is N + 1 rather than 2N. What
        stops it being N exactly is ptypy's far field: Eq. (cancel) annihilates
        the inverse transform only for a plain Fraunhofer transform, and
        BasicFarfieldPropagator carries pre- and post-chirps that do not
        commute with H, so one transform survives at each end.
        """
        phi, _, Hb, _ = self._illum(P)
        psi = (self._born_parallel(g, P, phi) if parallel
               else self._born_lowmem(g, P, phi))
        F = self._ff(self._ff_fw, psi)
        back = self._ff(self._ff_bw, F)                  # stand-in residual
        Fb = torch.fft.fft2(back)                        # once, not per slice
        if parallel:
            dj = torch.fft.ifft2(Hb.conj()[:, None, None] * Fb[None])
            return (phi[:, None].conj() * dj).sum(2).abs().sum()
        acc = 0.0
        for j in range(self.N):
            dj = torch.fft.ifft2(Hb[j].conj() * Fb)
            acc = acc + (phi[j][None].conj() * dj).sum(1).abs().sum()
        return acc

    @staticmethod
    def _transform_counts(N):
        """Transforms per position, forward plus adjoint, of Table 1.

        Multislice: 2(N-1) for the propagations plus one to the detector, and
        the same again for the adjoint. First Born: N forward transforms, one
        inverse for the accumulator and one for the detector, each way.
        """
        return dict(multislice=4 * N - 2, born=2 * (N + 2))

    def _illum_bytes(self):
        """Memory held by the cached illumination, which scales with N.

        phi is (N, modes, ny, nx) and the two H stacks are (N, ny, nx) each.
        These are formed once per batch, not once per position, so they are
        amortised in time -- but they are not free in space, and because
        _time_it charges anything allocated during the warm-up call to its
        baseline they do not appear in the measured peak. Reporting them
        separately keeps the storage claim honest: the per-batch workspace is
        O(batch) and independent of N, while the persistent illumination is
        O(N) and independent of batch. Multislice needs neither, but must hold
        O(N x batch) intermediates instead.
        """
        cuda = self.dev.type == 'cuda'
        self._illum_cache = None
        if cuda:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
        self._illum(self.pr_g)
        if not cuda:
            return float('nan')
        torch.cuda.synchronize()
        return (torch.cuda.memory_allocated() - before) / 1e9

    def _time_it(self, fn, reps):
        """Wall clock per call and peak allocation above the resting level."""
        cuda = self.dev.type == 'cuda'
        fn()                                              # warm up
        if cuda:
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        mem = (torch.cuda.max_memory_allocated() - base) / 1e9 if cuda else float('nan')
        return dt, mem

    def _benchmark(self):
        """Time and peak memory against slice count.

        The slice count is varied by repeating the reconstructed slices, so the
        object gets thicker than the specimen actually is. That is deliberate:
        this measures the cost of the two evaluation orders, not their
        accuracy, and N is the axis the cost claims are made in. Everything
        else -- frame size, batch, probe modes, propagators -- is the real
        experiment.
        """
        counts = self.p.benchmark_slice_counts
        if counts is None:
            counts = [1, 2, 4, 8, 16, 32]
        elif isinstance(counts, int):
            counts = [counts]
        counts = sorted({int(c) for c in counts if int(c) >= 1})
        reps = int(self.p.benchmark_reps)
        B = int(self.p.benchmark_batch or self.p.batch_size)
        B = min(B, self.n_pos)

        idx = torch.arange(B, dtype=torch.long, device=self.dev)
        base = torch.stack([self._gather(self.ob_g[j], idx)
                            for j in range(self.N)])
        z_true, N_true = self.z, self.N
        dz = float(z_true[1] - z_true[0]) if N_true > 1 else 1e-3

        out = dict(batch=B, reps=reps, slice_counts=counts, frame=[self.fny, self.fnx],
                   probe_modes=self.M, device=str(self.dev),
                   multislice_time=[], born_parallel_time=[], born_lowmem_time=[],
                   multislice_mem=[], born_parallel_mem=[], born_lowmem_mem=[],
                   illumination_mem=[])

        logger.info("cost benchmark: batch %d, %d probe modes, %d x %d frames, "
                    "%d reps" % (B, self.M, self.fny, self.fnx, reps))
        logger.info("   N     multislice        Born parallel      Born low-mem")
        for Nb in counts:
            reps_ix = [j % N_true for j in range(Nb)]
            O = torch.stack([base[j] for j in reps_ix])
            g = O - 1.0
            self.N, self.z = Nb, np.arange(Nb, dtype=float) * dz
            self._illum_cache = None          # phi and H depend on N and z
            m_il = self._illum_bytes()        # O(N), charged separately
            try:
                t_ms, m_ms = self._time_it(lambda: self._bench_multislice(O, self.pr_g), reps)
                t_bp, m_bp = self._time_it(lambda: self._bench_born(g, self.pr_g, True), reps)
                t_bl, m_bl = self._time_it(lambda: self._bench_born(g, self.pr_g, False), reps)
            except RuntimeError as e:                     # noqa: BLE001
                logger.warning("   N = %d skipped (%s)" % (Nb, str(e)[:60]))
                if self.dev.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            for k, v in (('multislice_time', t_ms), ('born_parallel_time', t_bp),
                         ('born_lowmem_time', t_bl), ('multislice_mem', m_ms),
                         ('born_parallel_mem', m_bp), ('born_lowmem_mem', m_bl),
                         ('illumination_mem', m_il)):
                out[k].append(v)
            out.setdefault('N', []).append(Nb)
            tc = self._transform_counts(Nb)
            out.setdefault('transforms_multislice', []).append(tc['multislice'])
            out.setdefault('transforms_born', []).append(tc['born'])
            logger.info("  %3d   %7.1f ms %5.2f GB   %7.1f ms %5.2f GB   "
                        "%7.1f ms %5.2f GB (+%5.2f illum)   | transforms "
                        "%d vs %d = %.2fx"
                        % (Nb, 1e3 * t_ms, m_ms, 1e3 * t_bp, m_bp,
                           1e3 * t_bl, m_bl, m_il, tc['multislice'],
                           tc['born'], tc['born'] / tc['multislice']))
            del O, g
            if self.dev.type == 'cuda':
                torch.cuda.empty_cache()

        self.N, self.z = N_true, z_true
        self._illum_cache = None
        if out.get('N'):
            i = -1
            il = out['illumination_mem'][i]
            tot = out['born_lowmem_mem'][i] + (il if np.isfinite(il) else 0.0)
            logger.info("at N = %d: Born low-memory is %.2fx the multislice "
                        "wall clock, %.2fx its workspace, %.2fx once the O(N) "
                        "illumination cache is counted too"
                        % (out['N'][i],
                           out['born_lowmem_time'][i] / max(out['multislice_time'][i], 1e-12),
                           out['born_lowmem_mem'][i] / max(out['multislice_mem'][i], 1e-12),
                           tot / max(out['multislice_mem'][i], 1e-12)))
        return out

    # ------------------------------------------------------------------ #
    #  Output                                                             #
    # ------------------------------------------------------------------ #
    def _report(self):
        r = self._results
        gk = [k for k in ('gauge_mean', 'gauge_none') if k in r]
        # _run_all promotes the mean gauge to primary when the reconstruction
        # is not in the natural one, so follow whatever it promoted
        prim_key = ('gauge_' + r.get('primary_gauge')
                    if r.get('primary_gauge') else gk[0])
        prim = r[prim_key]
        N = self.N
        L = logger.info
        L("=" * 70)
        L("First-Born truncation error, Eq. (epsilon)   [gauge: %s]"
          % prim.get('gauge', '?'))
        L("=" * 70)
        mp = r['run'].get('slice_mean_phase_rad')
        if mp is not None and max(abs(x) for x in mp) > 0.2:
            L("  per-slice mean phase %s rad: the as-reconstructed gauge is "
              "not natural, so only the mean-gauge numbers describe the "
              "specimen" % np.round(mp, 3).tolist())
        for M, e in sorted(prim['eps_order'].items()):
            tag = ('unscattered only' if M == 0 else
                   'Eq. (model), first Born' if M == 1 else
                   'exact multislice, so round-off' if M == N else 'order <= %d' % M)
            et = prim.get('eps_order_tangent', {}).get(M)
            extra = ('   tangent %.4e' % et) if et is not None else ''
            L("  M = %d   chord %.4e%s   %s" % (M, e, extra, tag))
        if 'eps_multislice_tangent' in prim:
            L("  the tangent hierarchy does not terminate in multislice: at "
              "M = N its residual is %.4e, the intra-slice linearisation error "
              "that no truncation order removes"
              % prim['eps_multislice_tangent'])
        for b, e in sorted(prim['eps_block'].items()):
            # Counting the pairs a block structure neglects, the same way
            # Eq. (eps1-estimate) counts them, gives (1/2)Phi^2 (1/M - 1/N):
            # it reduces to Eq. (eps1-estimate) at M = 1 and to zero at M = N.
            # The draft's (1 - 1/M) does neither -- it vanishes at M = 1 and is
            # largest at M = N, where the blocked model is exact.
            pred = (0.5 * prim['phi_tot_rms'] ** 2 * (1.0 / b - 1.0 / N)
                    if N > 1 else 0.0)
            L("  %d block(s)  eps = %.4e   predicted %.4e = (1/2)Phi^2(1/M-1/N)%s"
              % (b, e, pred, '   (= exact multislice)' if b >= N else ''))
        L("")
        L("Perturbation choice:")
        L("  chord   g = O - 1   eps_1 = %.4e" % prim['eps_order'].get(1, np.nan))
        L("  tangent g = i s     eps_1 = %.4e" % prim['eps_linearised'])
        L("  <|Re(g - is)|> = %.4e against <|g|> = %.4e"
          % (prim['g_real_gap'], prim['g_mean_modulus']))
        L("")
        L("Phi_tot (probe weighted rms) = %.4f rad" % prim['phi_tot_rms'])
        L("  correlated estimate   %.4e" % prim['eps1_estimate_correlated'])
        L("  decorrelated estimate %.4e" % prim['eps1_estimate_decorrelated'])
        L("")
        L("Dose: %.3g counts/frame, %.3g per pixel; truncation stays under the "
          "shot-noise floor up to %.3g counts in the brightest pixel"
          % (prim['counts_per_frame'], prim['counts_per_pixel'],
             prim['counts_visible_above']))
        L("chi^2/px against the data: multislice %.4g, Born %.4g, tangent %.4g"
          % (prim['chi2_multislice'], prim['chi2_born'], prim['chi2_linearised']))
        comps = {k[8:]: v for k, v in r.items() if k.startswith('compare_')}
        if comps:
            L("")
            L("The same measurement on other reconstructions of this scan,")
            L("all in the mean gauge, which is the only way they compare:")
            L("  %-22s %10s %10s %10s %9s %9s"
              % ('reconstruction', 'eps_1', 'tangent', 'Phi_tot', 'chi2 MS', 'chi2 FB'))
            pm = r.get('gauge_mean', prim)
            L("  %-22s %10.4e %10.4e %10.4f %9.1f %9.1f"
              % ('primary', pm['eps_order'].get(1, np.nan),
                 pm['eps_linearised'], pm['phi_tot_rms'],
                 pm['chi2_multislice'], pm['chi2_born']))
            for k, v in comps.items():
                L("  %-22s %10.4e %10.4e %10.4f %9.1f %9.1f"
                  % (k, v['eps_order'].get(1, np.nan), v['eps_linearised'],
                     v['phi_tot_rms'], v['chi2_multislice'], v['chi2_born']))
                od = v.get('object_vs_primary')
                if od:
                    L("      recovered slices vs the primary reconstruction: "
                      + ', '.join('slice %d %.3f rel, %.3f rad rms'
                                  % (d['slice'], d['relative_difference'],
                                     d['phase_rms_rad']) for d in od))
            e_p = r.get('gauge_mean', prim)['eps_order'].get(1, np.nan)
            for k, v in comps.items():
                e_c = v['eps_order'].get(1, np.nan)
                L("  '%s' gives eps_1 = %.3fx the primary value: %s"
                  % (k, e_c / max(e_p, 1e-30),
                     'the truncation error is a property of the specimen, not '
                     'of the model that fitted the object'
                     if 0.7 < e_c / max(e_p, 1e-30) < 1.4 else
                     'the two differ, so the object does absorb part of the '
                     'truncation error -- report both'))
        if 'gauge_mean' in r and 'gauge_none' in r:
            L("gauge: eps_1 = %.4e (none) vs %.4e (mean)"
              % (r['gauge_none']['eps_order'].get(1, np.nan),
                 r['gauge_mean']['eps_order'].get(1, np.nan)))
        L("=" * 70)

    def _figures(self):
        prim_key = 'gauge_none' if 'gauge_none' in self._results else \
            [k for k in self._results if k.startswith('gauge_')][0]
        d = self._results[prim_key]
        N = self.N
        od = self.outdir

        # 1 -- truncation error
        try:
            fig, ax = plt.subplots(figsize=(5.2, 4.0))
            Ms = sorted(d['eps_order'])
            bs = sorted(d['eps_block'])
            ax.semilogy(Ms, [d['eps_order'][m] for m in Ms], 'o-', ms=4,
                        label=r'chord $g=O-1$')
            # blocked and order truncation are the same object at N = 2 (only
            # M = 1 and M = N exist), so drawing both is just a thicker line
            if N > 2:
                ax.semilogy(bs, [d['eps_block'][b] for b in bs], 's--', ms=4,
                            label=r'blocked, chord')
            if 'eps_order_tangent' in d:
                Mt = sorted(d['eps_order_tangent'])
                ax.semilogy(Mt, [d['eps_order_tangent'][m] for m in Mt], 'X-',
                            color='crimson', ms=5, label=r'tangent $g=is$')
                if N > 2:
                    bt = sorted(d['eps_block_tangent'])
                    ax.semilogy(bt, [d['eps_block_tangent'][b] for b in bt],
                                'v--', color='crimson', ms=4, alpha=0.7,
                                label=r'blocked, tangent')
                ax.axhline(d['eps_multislice_tangent'], color='crimson', ls=':',
                           lw=1, alpha=0.8)
                ax.text(0.99, d['eps_multislice_tangent'],
                        'floor: linearised cascade ',
                        transform=ax.get_yaxis_transform(), ha='right',
                        va='bottom', fontsize=7, color='crimson')
            ax.set_xticks(sorted(set(Ms) | set(bs)))
            ax.set_xlabel(r'truncation order $M$ / number of blocks')
            ax.set_ylabel(r'$\epsilon_M$')
            ax.set_title(r'truncation error, $N=%d$' % N)
            if N <= 2:
                ax.text(0.03, 0.05, 'at $N=2$ blocking and order truncation '
                        'coincide', transform=ax.transAxes, fontsize=7,
                        color='0.35')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            _save(fig, od, '01_truncation_error.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 1 failed: %s" % e)

        # 2 -- eps_1 against Phi_tot, both perturbations
        try:
            fig, ax = plt.subplots(figsize=(5.2, 4.0))
            p = d['phi_per_pos']
            ax.loglog(p, d['eps1_per_pos'], '.', ms=3, alpha=0.4,
                      label=r'$g=O-1$')
            ax.loglog(p, d['eps1lin_per_pos'], '.', ms=3, alpha=0.4,
                      color='crimson', label=r'$g=is$')
            pp = np.linspace(max(p.min(), 1e-4), p.max(), 64)
            ax.loglog(pp, 0.5 * pp ** 2 * (1 - 1.0 / N), 'k-',
                      label=r'$\frac{1}{2}\Phi^2(1-1/N)$')
            ax.loglog(pp, pp ** 2 / (np.sqrt(2) * N), 'k--',
                      label=r'$\Phi^2/(\sqrt{2}N)$')
            # Phi spans well under a decade, so there are no major ticks in
            # range: label the minor ones plainly instead of leaving the axis
            # bare or letting the default mantissa labels collide.
            from matplotlib.ticker import ScalarFormatter, LogLocator
            ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 4.0,
                                                        5.0, 6.0, 8.0)))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.tick_params(axis='x', labelsize=7)
            ax.set_xlabel(r'$\Phi_{\rm tot}$ (rad)')
            ax.set_ylabel(r'$\epsilon_1$')
            ax.set_title('per position, against the estimates')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which='both')
            _save(fig, od, '02_phi_scatter.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 2 failed: %s" % e)

        # 3 -- measured beside simulated
        try:
            self._fig_frames(od)
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 3 failed: %s" % e)

        # 4 -- radial profiles
        try:
            r = d['radial']
            q = np.array(r['q'])
            fig, ax = plt.subplots(figsize=(5.4, 4.0))
            ax.loglog(q, np.maximum(r['measured'], 1e-12), 'k-', lw=1.2,
                      label='measured')
            ax.loglog(q, np.maximum(r['multislice'], 1e-12), '-', label='multislice')
            ax.loglog(q, np.maximum(r['born'], 1e-12), '--', label='first Born')
            ax.loglog(q, np.maximum(r['difference'], 1e-12), '-',
                      color='crimson', label='|multislice - Born|')
            ax.loglog(q, np.sqrt(np.maximum(r['measured'], 1e-12)), ':',
                      color='0.5', label=r'shot noise $\sqrt{I}$')
            ax.set_xlabel(r'$q$ (1/m)')
            ax.set_ylabel('azimuthal mean intensity (counts)')
            ax.set_title('where the truncation error lives')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which='both')
            _save(fig, od, '04_radial_profiles.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 4 failed: %s" % e)

        # 5 -- depth separability
        try:
            ds = self._results.get('depth_separability')
            if ds:
                fig, ax = plt.subplots(figsize=(5.2, 4.0))
                ax.loglog(ds['q'], ds['ratio'], 'o-', ms=3)
                ax.axvline(ds['q_c'], color='r', ls='--',
                           label=r'$q_c = 1/(2\,\mathrm{NA}\,\Delta z)$')
                ax.set_xlabel(r'transverse frequency $q$ (1/m)')
                ax.set_ylabel('antisymmetric / symmetric response')
                ax.set_title('depth separability, Eq. (depthres)')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3, which='both')
                _save(fig, od, '05_depth_separability.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 5 failed: %s" % e)

        # 6 -- the reconstruction under test
        try:
            fig, axes = plt.subplots(1, N + 1, figsize=(4.0 * (N + 1), 3.8))
            im = axes[0].imshow((self.pr_g[0].abs() ** 2).cpu().numpy(),
                                cmap='viridis')
            axes[0].set_title('probe intensity, mode 0')
            fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
            for j in range(N):
                im = axes[j + 1].imshow(self.ob_g[j].angle().cpu().numpy(),
                                        cmap='Greys_r')
                axes[j + 1].set_title('slice %d phase (z = %.2f mm)'
                                      % (j, self.z[j] * 1e3))
                fig.colorbar(im, ax=axes[j + 1], fraction=0.046, pad=0.04)
            for a in axes:
                a.set_xticks([]); a.set_yticks([])
            _save(fig, od, '06_reconstruction.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 6 failed: %s" % e)

        # 8 -- the same measurement on other reconstructions
        try:
            comps = {k[8:]: v for k, v in self._results.items()
                     if k.startswith('compare_')}
            if comps:
                fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
                a = axs[0]
                Ms = sorted(d['eps_order'])
                a.semilogy(Ms, [d['eps_order'][m] for m in Ms], 'o-', ms=4,
                           label='primary (first Born fit)')
                a.semilogy(Ms, [d['eps_order_tangent'][m] for m in Ms], 'X--',
                           ms=4, color='crimson', alpha=0.6,
                           label='primary, tangent')
                for k, v in comps.items():
                    a.semilogy(sorted(v['eps_order']),
                               [v['eps_order'][m] for m in sorted(v['eps_order'])],
                               's-', ms=4, label=k)
                a.set_xticks(Ms)
                a.set_xlabel(r'truncation order $M$')
                a.set_ylabel(r'$\epsilon_M$')
                a.set_title('same measurement, different reconstructions')
                a.legend(fontsize=7)
                a.grid(alpha=0.3)

                a = axs[1]
                names = ['primary'] + list(comps)
                e1 = [d['eps_order'].get(1, np.nan)] + \
                     [v['eps_order'].get(1, np.nan) for v in comps.values()]
                chi = [d['chi2_born'] / max(d['chi2_multislice'], 1e-30) - 1] + \
                      [v['chi2_born'] / max(v['chi2_multislice'], 1e-30) - 1
                       for v in comps.values()]
                x = np.arange(len(names))
                a.bar(x - 0.2, e1, 0.4, label=r'$\epsilon_1$')
                a.bar(x + 0.2, chi, 0.4, label=r'$\chi^2$ penalty')
                a.set_xticks(x)
                a.set_xticklabels(names, fontsize=7)
                a.set_ylabel('relative')
                a.set_title('truncation cost per reconstruction')
                a.legend(fontsize=7)
                a.grid(alpha=0.3, axis='y')
                _save(fig, od, '08_reconstruction_comparison.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 8 failed: %s" % e)

        # 7 -- cost against slice count
        try:
            bm = self._results.get('benchmark')
            if bm and bm.get('N'):
                Nb = np.array(bm['N'])
                fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
                a = axs[0]
                a.loglog(Nb, 1e3 * np.array(bm['multislice_time']), 'o-', ms=4,
                         label='multislice')
                a.loglog(Nb, 1e3 * np.array(bm['born_parallel_time']), 's-', ms=4,
                         label='Born, parallel')
                a.loglog(Nb, 1e3 * np.array(bm['born_lowmem_time']), '^-', ms=4,
                         label='Born, low memory')
                a.set_xlabel('slices $N$')
                a.set_ylabel('forward + adjoint (ms per batch)')
                a.set_title('wall clock, batch %d' % bm['batch'])
                a.legend(fontsize=7)
                a.grid(alpha=0.3, which='both')

                a = axs[1]
                if np.isfinite(bm['multislice_mem']).any():
                    a.loglog(Nb, bm['multislice_mem'], 'o-', ms=4, label='multislice')
                    a.loglog(Nb, bm['born_parallel_mem'], 's-', ms=4,
                             label='Born, parallel')
                    a.loglog(Nb, bm['born_lowmem_mem'], '^-', ms=4,
                             label='Born, low memory (workspace)')
                    il = np.array(bm.get('illumination_mem', []))
                    if il.size == Nb.size and np.isfinite(il).all():
                        a.loglog(Nb, np.array(bm['born_lowmem_mem']) + il,
                                 '^--', ms=4, alpha=0.6,
                                 label='Born, low memory + illumination')
                    a.set_ylabel('peak allocation (GB)')
                else:
                    a.text(0.5, 0.5, 'memory stats need CUDA', ha='center',
                           transform=a.transAxes, color='0.4')
                a.set_xlabel('slices $N$')
                a.set_title('peak memory')
                a.legend(fontsize=7)
                a.grid(alpha=0.3, which='both')
                fig.suptitle('cost against depth (slices repeated to extend $N$)',
                             fontsize=10)
                _save(fig, od, '07_cost.png')
        except Exception as e:      # noqa: BLE001
            logger.warning("figure 7 failed: %s" % e)

        # arrays for replotting
        try:
            np.savez_compressed(
                os.path.join(od, 'results.npz'),
                **{('%s_%s' % (k, kk)): np.asarray(vv)
                   for k, v in self._results.items() if k.startswith('gauge_')
                   for kk, vv in v.items()
                   if isinstance(vv, np.ndarray)})
        except Exception as e:      # noqa: BLE001
            logger.warning("npz failed: %s" % e)

    def _fig_frames(self, od):
        n = int(self.p.compare_frames)
        if n < 1:
            return
        n = min(n, self.n_pos)
        idx = torch.arange(n, dtype=torch.long, device=self.dev)
        O = torch.stack([self._gather(self.ob_g[j], idx) for j in range(self.N)])
        phi = self._illumination(self.pr_g)
        I_ms = (self._ff(self._ff_fw, self._multislice(O, self.pr_g)).abs() ** 2).sum(1)
        I_b = (self._ff(self._ff_fw, self._born(O - 1.0, self.pr_g, phi)).abs() ** 2).sum(1)
        Im, mk = self.I[idx], self.mask[idx]

        fig, ax = plt.subplots(n, 4, figsize=(12.5, 3.1 * n), squeeze=False)
        for i in range(n):
            me = (Im[i] * mk[i]).cpu().numpy()
            ms = I_ms[i].cpu().numpy()
            bn = I_b[i].cpu().numpy()
            # one common scale factor, taken from multislice, applied to both
            sc = me.sum() / max((ms * mk[i].cpu().numpy()).sum(), 1e-30)
            ms, bn = ms * sc, bn * sc
            top = max(me.max(), ms.max())
            nrm = LogNorm(vmin=max(top * 1e-6, 1e-3), vmax=max(top, 1e-2))
            for j, (img, ttl) in enumerate((
                    (me, 'measured'), (ms, 'multislice'),
                    (bn, 'first Born'), (np.abs(ms - bn), '|multislice - Born|'))):
                h = ax[i, j].imshow(np.maximum(img, 1e-6), norm=nrm, cmap='inferno')
                if i == 0:
                    ax[i, j].set_title(ttl, fontsize=10)
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, 0].set_ylabel('frame %d' % i, fontsize=9)
            fig.colorbar(h, ax=ax[i, 3], fraction=0.046, pad=0.04)
        fig.suptitle('measured and simulated, same position, same scale')
        _save(fig, od, '03_frames.png')


# ===========================================================================
#  Utilities
# ===========================================================================

def _save(fig, od, name):
    fig.tight_layout()
    fig.savefig(os.path.join(od, name), dpi=200)
    plt.close(fig)
    logger.info("wrote %s" % os.path.join(od, name))


def _halfway(x, y):
    """x where y first reaches half its maximum."""
    y = np.asarray(y, float)
    if not len(y):
        return float('nan')
    t = 0.5 * np.nanmax(y)
    i = int(np.argmax(y >= t))
    return float(np.asarray(x, float)[i])


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return dict(median=float(np.median(o)), mean=float(np.mean(o)),
                    p05=float(np.percentile(o, 5)),
                    p95=float(np.percentile(o, 95)), n=int(o.size))
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)