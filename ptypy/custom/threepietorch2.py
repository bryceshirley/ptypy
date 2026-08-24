# -*- coding: utf-8 -*-
"""
GPU-resident, batched multislice ePIE (3PIE) for ptypy, backed by PyTorch.

This is a corrected rewrite of the earlier ThreePIETorch2. With

    batch_size   = 1
    shuffle      = False
    object_denom = 'max'

it reproduces the CPU ThreePIE engine (ptypy stochastic.EPIE + multislice)
update for update, up to floating point -- verified, see below. Above
batch_size = 1 it becomes a mini-batch method: each view still produces the
*same* normalised ePIE increment it would have produced on its own, but
increments are averaged over the batch instead of applied sequentially, so
views within a batch do not see each other's updates.


EQUIVALENCE TO THE CPU ENGINE: VERIFIED
---------------------------------------
Measured by ``firstborn_review/ptypy_experiments/verify_threepie_torch.py``,
which drives ThreePIE and ThreePIETorch2 inside one Ptycho -- one set of pods,
one copy of the data, the same initial probe and object, and the CPU engine's
unseeded shuffle replaced by a no-op so both sweep the sorted view order.
NanoMAX scan 434, 670 positions, 2 probe modes, 2 slices, 3 iterations at
batch_size 1, object_denom 'max'; compared after every iteration are all object
slices, the persistent incident probe and the per-view error triples.

    precision   worst relative deviation   per sequential update
    double            2.893e-13                0.65 eps(float64)
    single            2.580e-04                1.08 eps(float32)

A fixed tolerance is the wrong test here. At batch_size 1 the run is
670 x 3 = 2010 Gauss-Seidel updates chained end to end, so round-off is carried
forward and re-amplified rather than staying put; the meaningful quantity is
the deviation per update, against the epsilon of the working precision. Both
precisions land at ~1 epsilon per update, and moving from float32 to float64
buys nine orders of magnitude -- i.e. the separation tracks the arithmetic, not
the algorithm. An algorithmic difference (wrong step size, missed update, wrong
normaliser) would be O(1) per update and would not improve with precision.

Getting there needed one fix, in ``_op_call``: it probed ptypy's propagators
with complex64 input first, so the recovered transfer function was only
accurate to ~1e-7 however it was stored afterwards, while the CPU engine calls
the same propagators with the complex128 arrays ptypy holds at
``data_type = 'double'``. That put a spurious ~1e-6 floor under the agreement.
complex128 is now tried first, with complex64 as a fallback for dtype-strict
FFT backends. The test only bites at ``p.data_type = 'double'``: ptypy's
containers are complex64 by default, and a torch engine asked for float64 is
otherwise still being compared against a single-precision reference.


PRECONDITIONING THE SLICE UPDATES
---------------------------------
`object_denom` selects what the slice increment is divided by.

  'max'    ptypy's convention: the scalar max over the patch of
           sum_m |psi^inc_i|^2, one number per view. This is what the CPU
           engine does and it is the setting for reproducing it. At
           batch_size > 1 it divides everything by the most strongly
           illuminated pixel, so sparsely covered regions are starved.

  'local'  the per-pixel accumulated denominator, sum over the batch of
           sum_m |psi^inc_i|^2, with a floor of denom_reg times its maximum.
           Numerator and denominator are accumulated separately over the batch
           and divided once, which is the preconditioned mini-batch update.

For multislice the local denominator is the intensity of the field *incident*
on slice i, which carries every slice above it and is rebuilt at every
position. That is the quantity a multislice engine has always divided by, and
it is exactly diagonal only when nothing downstream of slice i absorbs: the
full normal operator is

    diag(conj psi^inc_i) T_{>i}^dag T_{>i} diag(psi^inc_i)

and T_{>i}^dag T_{>i} = I only for a purely phase-shifting downstream stack.
The first-Born engine's denominator is |phi_i|^2 with phi_i the *unscattered*
illumination, which is the exact diagonal of its normal operator for any
object. The two therefore differ by precisely the departure from unitarity of
the downstream cascade.

  'unscattered'  divides by sum_m |phi_i|^2 instead, phi_i being the probe
           propagated to slice i with no object in the way -- the first-Born
           denominator. This does not make multislice's preconditioner exact;
           it makes it *the same* as the other engine's, which is what a
           convergence comparison between the two forward models needs if it is
           not to be a comparison of preconditioners. Run both engines with
           'unscattered' for the controlled comparison, and each with its own
           native choice for its best showing.


CORRUPTION
----------
`corrupt_slice` damages one slice on purpose, to measure how far the damage
travels to the others. In a first-Born model a bad slice reaches another
slice's gradient only through the shared detector residual, because the
operator is built from H_i and phi_i and contains no object. Here it reaches it
through the operator as well, and asymmetrically: an error upstream corrupts
the illumination every slice below it sees, while an error downstream corrupts
the path the residual is carried back along. Corrupting the first slice and
then the last, with an identical perturbation, separates the two channels.

What makes a corruption hard to undo is its direction, not its size. A
perturbation finer than the shear a_c = 2 NA dz is depth resolvable, so the
data says which slice it belongs to and the gradient removes it; one coarser
than a_c is not, and its antisymmetric part is close to null. The engine
reports a_c in pixels at start-up so corrupt_smooth can be chosen against it
rather than guessed. Per-slice statistics are recorded every iteration and
written to the slice file, so a corrupted run can be differenced against a
clean one without extra instrumentation.

The parameter block, the corruption modes and the statistics are deliberately
identical to those of the first-Born engine: the measurement is a difference
between runs, so anything that differs between the two engines other than the
forward model is a confound.


WHAT CHANGED RELATIVE TO THE PREVIOUS VERSION
---------------------------------------------
D1  Object denominator. ptypy's `_generic_object_update` (a=0) divides by the
    SCALAR max over the patch of sum_modes|P|^2, not by the local field. That
    is now `object_denom = 'max'`, the default; 'local' and 'unscattered' are
    the preconditioned alternatives described above.

D2  Illumination denominator. `_generic_probe_update` sums abs2(pod.object)
    over pods. All pods of a view share the same object patch, so with M probe
    modes the norm is M * max|O|^2. Under 'local' the probe uses the per-pixel
    accumulated denominator instead, matching the object.

D3  Step sizes. EPIEMixin sets _pr_b = p.alpha and _ob_b = p.beta, i.e. in
    ptypy `alpha` is the PROBE step and `beta` is the OBJECT step -- the
    opposite of what the parameter help strings say. We now inherit
    self._ob_b / self._pr_b directly. object_step / probe_step are optional
    overrides, default None.

D4  probe_update_start gates every slice, because on CPU the gate lives inside
    probe_update(), which the multislice loop calls once per slice.

D5  Sequential ordering: the psi correction uses the freshly updated object
    patch, as ptypy does.

D6  Inactive slice 0: the illumination is replaced outright, not accumulated.

D7  Inactive final slice: exit = psi, and a warning. The CPU engine leaves
    pod.object set, which its own ToDo predicts is a bug.

D8  probe_support is opt-in via apply_probe_support.

D9  Per-view errors are returned, matching ptypy's expectation.

D10 shuffle=False walks views in the CPU engine's sorted order. `seed` pins
    both the shuffle and anything else random, which a paired corruption
    experiment needs.

Also: the Fourier modulus ratio uses ptypy's absolute 1e-10 floor rather than
finfo(float32).eps (~1.2e-7).


PROPAGATOR EXTRACTION (unchanged -- it was correct)
---------------------------------------------------
Each ptypy propagator is probed numerically instead of reimplementing its FFT
conventions:

  * shift-invariant (near-field):  fw(w) = IDFT(H * DFT(w))
    recovered from H = DFT(fw(delta_00)).
  * chirp-DFT (far-field):         fw(w) = A * DFT(R * w)
    recovered from A = fw(delta_00) and R = IDFT(fw(ones) / A).

Both are verified against the ptypy operator on random data, so a convention
mismatch is a startup error rather than a silently wrong reconstruction.


LIMITATIONS
-----------
* Single object storage, single object mode. Multiple probe modes supported.
* object_norm_is_global is not supported (see engine_initialize).
* No position refinement, no probe recentering.
* MPI unsupported (inherited from ptypy's stochastic engines).
"""

import os
from datetime import datetime

import numpy as np

from ptypy.engines import stochastic
from ptypy.engines import register
from ptypy.core import geometry
from ptypy.utils import Param
from ptypy.utils.verbose import logger
from ptypy import io
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase
from matplotlib.patches import Rectangle

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------#
#  Numerical extraction of a ptypy propagator                                 #
# ---------------------------------------------------------------------------#
def _op_call(op, arr):
    """
    Call a ptypy propagator, tolerating single- or double-precision input.

    complex128 is tried first, and not merely as a preference.  A propagator
    called with complex64 input carries out its FFTs in complex64, so the
    recovered transfer function is only accurate to ~1e-7 no matter what dtype
    it is stored in afterwards -- while the CPU engine calls the same
    propagator with the complex128 arrays ptypy holds at ``data_type='double'``.
    Extracting in single and comparing against a double-precision reference put
    a ~1e-6 floor under the CPU-vs-GPU agreement that looked like round-off but
    was really an artefact of the probe.  complex64 stays as a fallback for
    dtype-strict backends (pyFFTW plans are built for one dtype).
    """
    last = None
    for dt in (np.complex128, np.complex64):
        try:
            return np.asarray(op(arr.astype(dt)), dtype=np.complex128)
        except Exception as e:      # noqa: BLE001 - propagator dtype pickiness
            last = e
    raise RuntimeError("could not call propagator: %s" % last)


def _apply_numpy(kind, d, w):
    if kind == 'lsi':                       # ifft2(fft2(w) * H)
        return np.fft.ifft2(np.fft.fft2(w) * d['H'])
    if kind == 'ilsi':                      # fft2(ifft2(w) * H)
        return np.fft.fft2(np.fft.ifft2(w) * d['H'])
    if kind == 'chirp':                     # A * fft2(R * w)
        return d['A'] * np.fft.fft2(d['R'] * w)
    if kind == 'ichirp':                    # A * ifft2(R * w)
        return d['A'] * np.fft.ifft2(d['R'] * w)
    raise ValueError(kind)


def _extract_operator(op, shape, name="propagator", rtol=1e-5):
    """
    Recover a ptypy propagator exactly, without assuming its internals.

    Four candidate forms are built from two probes and the one that reproduces
    the operator on random data is kept:

        lsi     op(w) = ifft2(fft2(w) * H)       near-field
        ilsi    op(w) = fft2(ifft2(w) * H)       near-field, reverse pairing
        chirp   op(w) = A * fft2(R * w)          far-field forward
        ichirp  op(w) = A * ifft2(R * w)         far-field reverse

    numpy's convention is used throughout (fft2 unnormalised, ifft2 carrying
    1/N), which torch reproduces with the default norm='backward'.
    """
    ny, nx = shape
    Ntot = float(ny * nx)

    d00 = np.zeros(shape, dtype=np.complex128)
    d00[0, 0] = 1.0
    ones = np.ones(shape, dtype=np.complex128)

    out00 = _op_call(op, d00)
    out1 = _op_call(op, ones)

    cands = [
        ('lsi',  {'H': np.fft.fft2(out00)}),
        ('ilsi', {'H': Ntot * np.fft.ifft2(out00)}),
    ]
    if np.abs(out00).min() > 0:
        A = out00
        cands.append(('chirp',  {'A': A, 'R': np.fft.ifft2(out1 / A)}))
        cands.append(('ichirp', {'A': A, 'R': np.fft.fft2(out1 / A)}))

    rng = np.random.default_rng(0)
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    ref = _op_call(op, w)
    scale = max(np.abs(ref).max(), 1e-30)

    best_kind, best_d, best_err = None, None, np.inf
    errs = {}
    for kind, d in cands:
        err = np.abs(_apply_numpy(kind, d, w) - ref).max() / scale
        errs[kind] = err
        if err < best_err:
            best_kind, best_d, best_err = kind, d, err

    if best_err > rtol:
        raise RuntimeError(
            "%s: numerical extraction failed. Errors by form: %s. The propagator "
            "is none of ifft2(fft2(w)*H), fft2(ifft2(w)*H), A*fft2(R*w), "
            "A*ifft2(R*w). Inspect it and hard-code the transfer function."
            % (name, {k: '%.2e' % v for k, v in errs.items()}))

    spec = {'kind': best_kind}
    spec.update(best_d)
    logger.info("%s extracted as '%s', verification error %.2e" % (name, best_kind, best_err))
    return spec


@register()
class ThreePIETorch2(stochastic.EPIE):
    """
    Batched GPU multislice ePIE.

    Defaults:

    [name]
    default = ThreePIETorch2
    type = str
    help = Engine identifier

    [number_of_slices]
    default = 2
    type = int
    help = The number of slices

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = Single float, or a list of length number_of_slices - 1.

    [slice_start_iteration]
    default = 0
    type = int, list, tuple
    help = Iteration at which each slice starts being used

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the slice data

    [device]
    default = cuda
    type = str
    help = Torch device
    doc = 'cuda', 'cuda:0', or 'cpu'. Falls back to cpu with a warning if CUDA
          is unavailable.

    [batch_size]
    default = 1
    type = int
    help = Number of scan positions per update
    doc = 1 reproduces sequential 3PIE exactly. Larger values are faster but
          average the per-view increments instead of applying them in sequence.
          16-64 is usually a good trade for 512x512 frames.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

    [object_step]
    default = None
    type = float, NoneType
    help = Override for the object update step size
    doc = None inherits ptypy's value, which for EPIE is p.beta (NOT p.alpha --
          EPIEMixin sets _ob_b = beta, _pr_b = alpha, contrary to the parameter
          help strings in stochastic.py).

    [probe_step]
    default = None
    type = float, NoneType
    help = Override for the illumination update step size
    doc = None inherits ptypy's value, which for EPIE is p.alpha.

    [object_denom]
    default = max
    type = str
    help = Slice preconditioner: 'max', 'local' or 'unscattered'
    doc = 'max' is ptypy's scalar max over the patch of sum_m |psi^inc_i|^2 and
          reproduces the CPU engine. 'local' is the per-pixel accumulated
          denominator, numerator and denominator summed over the batch and
          divided once: the preconditioned mini-batch update, and the setting
          to use above batch_size = 1. 'unscattered' divides by sum_m |phi_i|^2
          with phi_i the probe propagated to slice i with no object in the way,
          which is the first-Born engine's denominator; use it in both engines
          when comparing the two forward models, so that the comparison is not
          confounded by the preconditioner.

    [denom_reg]
    default = 0.01
    type = float
    help = Floor on the 'local' and 'unscattered' denominators, as a fraction of the maximum
    doc = Ignored for 'max', which needs no regularisation. A focused probe has
          compact support and interior zeros, so the local denominator must be
          floored somewhere.

    [apply_probe_support]
    default = False
    type = bool
    help = Apply the circular probe support from p.probe_support
    doc = The CPU multislice engine never calls support_constraint, so leave
          this False when comparing the two engines. BaseEngine's
          p.probe_support defaults to 0.7.

    [sync_interval]
    default = 1
    type = int
    help = Iterations between copying GPU state back into ptypy containers
    doc = Only needed for plotting and autosave. Set higher to save a little time.

    [plot_interval]
    default = 0
    type = int
    help = Iterations between diagnostic PNGs, 0 disables

    [shuffle]
    default = False
    type = bool
    help = Reshuffle the position order every iteration
    doc = Set False to walk views in the same sorted order as the CPU engine.

    [seed]
    default = 0
    type = int
    help = Seed for the position order; 0 leaves it free
    doc = The corruption experiment is a paired comparison, so anything that
          differs between the clean and corrupted runs other than the
          corruption is a confound. Set the same non-zero seed for both runs,
          and for both engines.

    [corrupt_slice]
    default = none
    type = str
    help = Deliberately damage one slice: 'none', 'first', 'last' or an index
    doc = For measuring how far an error in one slice travels to the others.
          Here a bad slice reaches another slice's gradient through the
          operator as well as through the shared residual, and asymmetrically:
          upstream it corrupts the illumination, downstream the return path.
          Corrupting first and then last, with an identical perturbation,
          separates the two.

    [corrupt_mode]
    default = phase
    type = str
    help = phase, amplitude, antisym, duplicate, constant, vacuum, shift, swap, defocus
    doc = What makes a corruption hard to undo is its direction, not its size.
          A perturbation finer than the shear a_c = 2 NA dz is depth
          resolvable, so the data says which slice it belongs to and the
          gradient removes it; one coarser than a_c is not, and its
          antisymmetric part is close to null. Set corrupt_smooth against the
          a_c reported at start-up.
            'phase' / 'amplitude'  smooth random field on one slice
            'antisym'    the same field added to one slice and subtracted from
                         the other end slice: the depth-differential direction
            'duplicate'  another slice's content multiplied on top of this one,
                         so the corruption is a plausible object, not noise
            'constant'   a uniform phase offset, exactly null: a control that
                         should persist untouched and change no recorded value
            'vacuum' erases the slice; 'shift' translates it by
            corrupt_amplitude pixels; 'swap' exchanges first and last;
            'defocus' propagates it by corrupt_amplitude metres. Those four
            need corrupt_at_iter > 0 to mean anything.

    [corrupt_amplitude]
    default = 1.0
    type = float
    help = Strength: rad rms, or pixels, or metres, depending on corrupt_mode
    doc = Amplitude saturates: |g| = 2|sin(phi/2)| never exceeds 2, so phi = pi
          buys only 2.7 times phi = 1. Coarsening the field keeps working.

    [corrupt_smooth]
    default = 8.0
    type = float
    help = Correlation length of the random field, in pixels
    doc = White noise sits mostly outside the transfer function and would be
          filtered away in a few iterations. Compare with the shear a_c
          reported at start-up.

    [corrupt_at_iter]
    default = 0
    type = int
    help = Iteration at which to apply the corruption; 0 means at start-up

    [corrupt_seed]
    default = 1234
    type = int
    help = Seed for the corruption field
    doc = Fixed so that corrupting the first slice and corrupting the last use
          an identical perturbation, and any asymmetry belongs to the model.
    """

    def __init__(self, ptycho_parent, pars=None):
        super(ThreePIETorch2, self).__init__(ptycho_parent, pars)
        self.article = dict(
            title='Ptychographic transmission microscopy in three dimensions using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29, year=2012, page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        self.ptycho.citations.add_article(**self.article)

        self.plot_progress_dir = 'iteration_progress_%s' % datetime.now().strftime("%Y%m%d_%H%M%S")
        self._warned_inactive_last = False
        self._history = []

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()

        if torch is None:
            raise ImportError("ThreePIETorch2 requires PyTorch")

        N = int(self.p.number_of_slices)

        # ---- device and dtype ----
        want = str(self.p.device)
        if want.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            want = 'cpu'
        self.dev = torch.device(want)
        double = str(self.p.precision).lower().startswith('d')
        self.cdt = torch.complex128 if double else torch.complex64
        self.rdt = torch.float64 if double else torch.float32
        logger.info("ThreePIETorch2 on %s, %s precision" % (self.dev, 'double' if double else 'single'))

        # ---- step sizes (D3) ----
        # EPIEMixin: _ob_b = p.beta (object), _pr_b = p.alpha (probe).
        self.ob_step = float(self._ob_b if self.p.object_step is None else self.p.object_step)
        self.pr_step = float(self._pr_b if self.p.probe_step is None else self.p.probe_step)
        logger.info("object step %.4g, probe step %.4g" % (self.ob_step, self.pr_step))

        # ---- preconditioner (D1) ----
        self._denom = str(self.p.object_denom).lower()
        if self._denom not in ('max', 'local', 'unscattered'):
            raise ValueError("object_denom must be 'max', 'local' or 'unscattered'")
        self._reg = float(self.p.denom_reg)
        logger.info("slice preconditioner: %s%s" % (
            self._denom,
            "" if self._denom == 'max' else ", floor %.3g of the maximum" % self._reg))
        if self._denom == 'max' and int(self.p.batch_size) > 1:
            logger.warning("object_denom='max' at batch_size=%d divides every slice "
                           "increment by the most strongly illuminated pixel in the "
                           "batch, which starves sparsely covered regions. Use 'local'."
                           % int(self.p.batch_size))

        # ptypy's global object norm would need the product-of-slices object,
        # which is a different quantity per slice. Refuse rather than guess.
        if getattr(self, '_object_norm_is_global', False):
            raise NotImplementedError(
                "object_norm_is_global is not supported by ThreePIETorch2; set it False")

        if int(self.p.batch_size) > 1:
            logger.warning("batch_size=%d: per-view increments are averaged, not applied "
                           "sequentially. Use batch_size=1 for exact 3PIE."
                           % int(self.p.batch_size))

        # ---- slice_start_iteration ----
        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = np.ones(N) * self.p.slice_start_iteration
        self.p.slice_start_iteration = np.asarray(self.p.slice_start_iteration)
        if len(self.p.slice_start_iteration) != N:
            raise ValueError("slice_start_iteration must have length number_of_slices")

        # ---- ptypy containers, kept for saving and plotting ----
        self._object = [self.ob.copy(self.ob.ID + "_o_" + str(i)) for i in range(N)]

        if len(self.ob.storages) != 1:
            raise NotImplementedError("ThreePIETorch2 supports a single object storage")
        self._ob_storage = list(self.ob.storages.values())[0]
        if self._ob_storage.data.shape[0] != 1:
            raise NotImplementedError("ThreePIETorch2 supports a single object mode")

        if len(self.pr.storages) != 1:
            raise NotImplementedError("ThreePIETorch2 supports a single probe storage")
        self._pr_storage = list(self.pr.storages.values())[0]

        self.M, self.fny, self.fnx = self._pr_storage.data.shape

        # ---- GPU state ----
        ob0 = self._ob_storage.data[0]
        self.ob_g = torch.stack([
            torch.as_tensor(np.ascontiguousarray(ob0), dtype=self.cdt, device=self.dev)
            for _ in range(N)])                                     # (N, obny, obnx)
        self.pr_g = torch.as_tensor(np.ascontiguousarray(self._pr_storage.data),
                                    dtype=self.cdt, device=self.dev)  # (M, fny, fnx)

        self._build_propagators()
        self._collect_views()

        # ---- optional circular probe support (D8, opt-in) ----
        self.pr_support = None
        ps = getattr(self.p, 'probe_support', None)
        if bool(self.p.apply_probe_support) and ps is not None and 0 < float(ps) < 1:
            yy, xx = np.mgrid[:self.fny, :self.fnx]
            r = np.hypot(yy - self.fny / 2.0, xx - self.fnx / 2.0)
            mask = (r < float(ps) * min(self.fny, self.fnx) / 2.0).astype(np.float64)
            self.pr_support = torch.as_tensor(mask, dtype=self.rdt, device=self.dev)
            logger.info("probe support: circular, radius fraction %.3f" % float(ps))

        # sum_m |phi_i|^2 for object_denom='unscattered'; rebuilt whenever the
        # probe moves
        self._phi_cache = None

        # ---- reproducibility (D10) ----
        seed = int(self.p.seed)
        self._order_gen = None
        if seed:
            self._order_gen = torch.Generator(device=self.dev)
            self._order_gen.manual_seed(seed + 1)
            logger.info("seed = %d: the position order is reproducible" % seed)

        os.makedirs(self.plot_progress_dir, exist_ok=True)

        self._report_memory()
        self._report_shear()

        if self._corrupt_index() is not None and int(self.p.corrupt_at_iter) <= 0:
            self._apply_corruption()

    def _build_propagators(self):
        """Extract the slice-to-slice near-field and the far-field propagators."""
        N = int(self.p.number_of_slices)
        scan = list(self.ptycho.model.scans.values())[0]
        geom = scan.geometries[0]
        self._lam = float(np.ravel(geom.lam)[0])
        self._dxo = float(np.ravel(geom.resolution)[0])

        g = Param()
        g.energy = geom.energy
        g.psize = geom.resolution
        g.shape = geom.shape
        g.propagation = "nearfield"

        if type(self.p.slice_thickness) in [list, tuple]:
            if len(self.p.slice_thickness) != N - 1:
                raise ValueError("slice_thickness list must have length number_of_slices - 1")
            thicknesses = list(self.p.slice_thickness)
        else:
            thicknesses = [self.p.slice_thickness] * (N - 1)
        self._z = np.concatenate([[0.0], np.cumsum(np.asarray(thicknesses, float))]) \
            if N > 1 else np.zeros(1)

        shape = tuple(geom.shape)
        self._fw_spec, self._bw_spec = [], []
        for i, t in enumerate(thicknesses):
            g.distance = t
            G = geometry.Geo(owner=None, pars=g)
            self._fw_spec.append(_extract_operator(G.propagator.fw, shape, "nearfield fw[%d]" % i))
            self._bw_spec.append(_extract_operator(G.propagator.bw, shape, "nearfield bw[%d]" % i))

        pod = None
        for name, view in self.di.views.items():
            if view.active:
                pod = list(view.pods.values())[0]
                break
        if pod is None:
            raise RuntimeError("no active diffraction views")

        self._ff_fw = _extract_operator(pod.fw, shape, "farfield fw")
        self._ff_bw = _extract_operator(pod.bw, shape, "farfield bw")

        def to_g(spec):
            out = {'kind': spec['kind']}
            for k in ('H', 'A', 'R'):
                if k in spec:
                    out[k] = torch.as_tensor(spec[k], dtype=self.cdt, device=self.dev)
            return out

        self._fw_g = [to_g(s) for s in self._fw_spec]
        self._bw_g = [to_g(s) for s in self._bw_spec]
        self._ff_fw_g = to_g(self._ff_fw)
        self._ff_bw_g = to_g(self._ff_bw)

    @staticmethod
    def _apply(spec, w):
        """Apply an extracted operator to a stack with trailing spatial dims."""
        k = spec['kind']
        if k == 'lsi':
            return torch.fft.ifft2(torch.fft.fft2(w) * spec['H'])
        if k == 'ilsi':
            return torch.fft.fft2(torch.fft.ifft2(w) * spec['H'])
        if k == 'chirp':
            return spec['A'] * torch.fft.fft2(spec['R'] * w)
        if k == 'ichirp':
            return spec['A'] * torch.fft.ifft2(spec['R'] * w)
        raise ValueError(k)

    def _unscattered(self):
        """sum_m |phi_i|^2, the first-Born denominator: the probe propagated to
        each slice with no object in the way. Cached until the probe moves."""
        if self._phi_cache is None:
            N = int(self.p.number_of_slices)
            ph = torch.empty((N, self.M, self.fny, self.fnx),
                             dtype=self.cdt, device=self.dev)
            ph[0] = self.pr_g
            for i in range(1, N):
                ph[i] = self._apply(self._fw_g[i - 1], ph[i - 1])
            self._phi_cache = (ph.abs() ** 2).sum(dim=1)          # (N, ny, nx)
        return self._phi_cache

    def _collect_views(self):
        """Pull positions, intensities and masks onto the GPU once."""
        names = sorted([n for n, v in self.di.views.items() if v.active])
        if not names:
            raise RuntimeError("no active diffraction views")

        rows, cols, data, masks, layers = [], [], [], [], None

        for n in names:
            view = self.di.views[n]
            pods = list(view.pods.values())

            ov = pods[0].ob_view
            try:
                lo = np.asarray(ov.dlow, dtype=np.int64)
            except AttributeError:
                raise RuntimeError(
                    "This ptypy version's View has no .dlow; adapt _collect_views "
                    "to however object patches are indexed here.")
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))

            lay = [int(p.pr_view.dlayer) for p in pods]
            if layers is None:
                layers = lay
            elif lay != layers:
                raise NotImplementedError("inconsistent probe-mode layout across views")

            data.append(np.asarray(view.data, dtype=np.float64))
            m = pods[0].mask
            masks.append(np.ones_like(data[-1]) if m is None else np.asarray(m, dtype=np.float64))

        self.n_pos = len(names)
        self.view_names = names
        self.rows_g = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols_g = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.I_g = torch.as_tensor(np.stack(data), dtype=self.rdt, device=self.dev)
        self.mask_g = torch.as_tensor(np.stack(masks), dtype=self.rdt, device=self.dev)
        self.amp_g = torch.sqrt(self.I_g.clamp_min(0))

        if list(layers) != list(range(self.M)):
            logger.warning("probe layers %s do not match 0..%d; check mode ordering"
                           % (layers, self.M - 1))
        logger.info("collected %d scan positions, %d probe mode(s)" % (self.n_pos, self.M))

        # illuminated region, for the per-slice statistics
        cov = torch.zeros(self.ob_g[0].shape, dtype=self.rdt, device=self.dev)
        ones = None
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            ones = torch.ones((idx.numel(), self.fny, self.fnx),
                              dtype=self.rdt, device=self.dev)
            self._scatter_add(cov, idx, ones)
        self.cov = cov > 0
        del ones

    def _report_memory(self):
        el = 8 if self.cdt == torch.complex64 else 16
        obn = self.ob_g.numel() * el
        dat = (self.I_g.numel() + self.mask_g.numel() + self.amp_g.numel()) * \
              (4 if self.rdt == torch.float32 else 8)
        work = int(self.p.batch_size) * self.M * self.fny * self.fnx * el * \
            (2 * int(self.p.number_of_slices) + 4)
        logger.info("GPU memory estimate: objects %.2f GB, data %.2f GB, workspace %.2f GB"
                    % (obn / 1e9, dat / 1e9, work / 1e9))

    def _report_shear(self):
        """a_c = 2 NA dz, the scale above which the data cannot attribute a
        feature to one slice. The yardstick for corrupt_smooth."""
        try:
            F = torch.fft.fft2(self.pr_g)
            p = (F.abs() ** 2).sum(0).cpu().numpy()
            fy = np.fft.fftfreq(self.fny, d=self._dxo)
            fx = np.fft.fftfreq(self.fnx, d=self._dxo)
            a = self._lam * np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
            o = np.argsort(a.ravel())
            c = np.cumsum(p.ravel()[o])
            c /= max(c[-1], 1e-30)
            na = float(a.ravel()[o][np.searchsorted(c, 0.90)])
            dz = float(self._z[-1] - self._z[0]) if self._z.size > 1 else 0.0
            self._a_c = 2.0 * na * dz
            logger.info("probe NA %.3f mrad, shear a_c = 2 NA dz = %.3g m "
                        "= %.0f object pixels"
                        % (na * 1e3, self._a_c, self._a_c / self._dxo))
            if self._corrupt_index() is not None and self._a_c > 0:
                r = float(self.p.corrupt_smooth) / (self._a_c / self._dxo)
                logger.info("corrupt_smooth is %.2f of the shear: %s" % (r,
                            "depth resolvable, the fit will undo it" if r < 0.5
                            else "comparable to the shear, weakly constrained"
                            if r < 2 else
                            "far coarser than the shear, close to null"))
        except Exception as e:      # noqa: BLE001
            self._a_c = None
            logger.warning("could not estimate the shear: %s" % e)

    # ------------------------------------------------------------------ #
    #  Patch gather / scatter                                             #
    # ------------------------------------------------------------------ #
    def _gather(self, arr2d, idx):
        r = self.rows_g[idx]                    # (B, fny)
        c = self.cols_g[idx]                    # (B, fnx)
        return arr2d[r[:, :, None], c[:, None, :]]

    def _scatter_add(self, dst, idx, vals):
        B = idx.numel()
        r = self.rows_g[idx][:, :, None].expand(B, self.fny, self.fnx)
        c = self.cols_g[idx][:, None, :].expand(B, self.fny, self.fnx)
        dst.index_put_((r.reshape(-1), c.reshape(-1)), vals.reshape(-1), accumulate=True)

    # ------------------------------------------------------------------ #
    #  Deliberate corruption of one slice                                 #
    # ------------------------------------------------------------------ #
    def _corrupt_index(self):
        c = str(self.p.corrupt_slice).lower()
        N = int(self.p.number_of_slices)
        if c in ('none', '', 'off', 'false'):
            return None
        if c == 'first':
            return 0
        if c == 'last':
            return N - 1
        try:
            return int(c) % N
        except ValueError:
            logger.warning("corrupt_slice = %r not understood; ignoring" % c)
            return None

    def _smooth_field(self, gen):
        """Zero-mean random field on the object grid, correlation length
        corrupt_smooth pixels, unit rms."""
        ny, nx = int(self.ob_g.shape[-2]), int(self.ob_g.shape[-1])
        r = torch.randn((ny, nx), generator=gen, device=self.dev, dtype=torch.float32)
        sig = float(self.p.corrupt_smooth)
        if sig > 0:
            fy = torch.fft.fftfreq(ny, device=self.dev)
            fx = torch.fft.fftfreq(nx, device=self.dev)
            k2 = fy[:, None] ** 2 + fx[None, :] ** 2
            g = torch.exp(-2.0 * (np.pi * sig) ** 2 * k2)
            r = torch.fft.ifft2(torch.fft.fft2(r.to(torch.complex64)) * g).real
        r = r - r.mean()
        return (r / r.std().clamp_min(1e-12)).to(self.rdt)

    def _apply_corruption(self):
        j = self._corrupt_index()
        if j is None:
            return
        N = int(self.p.number_of_slices)
        mode = str(self.p.corrupt_mode).lower()
        amp = float(self.p.corrupt_amplitude)
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(int(self.p.corrupt_seed))
        k = (N - 1) if j == 0 else 0

        before = self._slice_stats()
        if mode == 'phase':
            self.ob_g[j] = self.ob_g[j] * torch.exp(
                1j * (amp * self._smooth_field(gen)).to(self.cdt))
        elif mode in ('amplitude', 'amp'):
            self.ob_g[j] = self.ob_g[j] * (
                1.0 + amp * self._smooth_field(gen)).to(self.cdt)
        elif mode in ('antisym', 'antisymmetric'):
            fld = (amp * self._smooth_field(gen)).to(self.cdt)
            self.ob_g[j] = self.ob_g[j] * torch.exp(1j * fld)
            self.ob_g[k] = self.ob_g[k] * torch.exp(-1j * fld)
            logger.info("corruption: antisymmetric between slices %d and %d" % (j, k))
        elif mode == 'duplicate':
            self.ob_g[j] = self.ob_g[j] * self.ob_g[k]
            logger.info("corruption: slice %d multiplied by slice %d" % (j, k))
        elif mode == 'constant':
            self.ob_g[j] = self.ob_g[j] * np.exp(1j * amp)
        elif mode == 'vacuum':
            self.ob_g[j] = torch.ones_like(self.ob_g[j])
        elif mode == 'shift':
            n = int(round(amp))
            self.ob_g[j] = torch.roll(self.ob_g[j], shifts=(n, n), dims=(0, 1))
        elif mode == 'swap':
            tmp = self.ob_g[j].clone()
            self.ob_g[j] = self.ob_g[k]
            self.ob_g[k] = tmp
            logger.info("corruption: slices %d and %d exchanged" % (j, k))
        elif mode == 'defocus':
            ny, nx = int(self.ob_g.shape[-2]), int(self.ob_g.shape[-1])
            fy = np.fft.fftfreq(ny, d=self._dxo)
            fx = np.fft.fftfreq(nx, d=self._dxo)
            a2 = (self._lam * fy)[:, None] ** 2 + (self._lam * fx)[None, :] ** 2
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            Hd = np.exp(2j * np.pi * (amp / self._lam) * kz)
            Hd[a2 > 1.0] = 0.0
            Ht = torch.as_tensor(Hd, dtype=self.cdt, device=self.dev)
            self.ob_g[j] = torch.fft.ifft2(Ht * torch.fft.fft2(self.ob_g[j]))
        else:
            logger.warning("corrupt_mode = %r not understood; nothing done" % mode)
            return

        after = self._slice_stats()
        logger.warning("CORRUPTION APPLIED at iteration %d: slice %d of %d, mode '%s', "
                       "amplitude %.4g, smooth %.4g px, seed %d"
                       % (self.curiter, j, N, mode, amp,
                          float(self.p.corrupt_smooth), int(self.p.corrupt_seed)))
        for i in range(N):
            logger.info("   slice %2d: phase rms %.4f -> %.4f rad, |g| mean %.4f -> %.4f"
                        % (i, before['phase_rms'][i], after['phase_rms'][i],
                           before['g_mean'][i], after['g_mean'][i]))
        self._history.append(dict(iter=int(self.curiter), event='corrupt',
                                  slice=int(j), mode=mode, amplitude=amp))

    def _slice_stats(self):
        """Phase rms and mean |g| per slice, over the illuminated region."""
        ph, gm = [], []
        for j in range(int(self.p.number_of_slices)):
            v = self.ob_g[j][self.cov]
            a = torch.angle(v)
            ph.append(float(torch.sqrt((a - a.mean()).pow(2).mean())))
            gm.append(float((v - 1.0).abs().mean()))
        return dict(phase_rms=ph, g_mean=gm)

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        B = int(self.p.batch_size)

        for _ in range(num):
            if (self._corrupt_index() is not None
                    and int(self.p.corrupt_at_iter) > 0
                    and int(self.curiter) == int(self.p.corrupt_at_iter)):
                self._apply_corruption()

            if self.p.shuffle:
                order = (torch.randperm(self.n_pos, device=self.dev)
                         if self._order_gen is None else
                         torch.randperm(self.n_pos, generator=self._order_gen,
                                        device=self.dev))
            else:
                # same order the CPU engine walks with a sorted, unshuffled list
                order = torch.arange(self.n_pos, device=self.dev)

            error_dct = {}
            for s in range(0, self.n_pos, B):
                idx = order[s:s + B]
                error_dct.update(self._batch_update(idx))

            self.curiter += 1
            if self.curiter % max(1, int(self.p.sync_interval)) == 0:
                self._sync_to_ptypy()

            st = self._slice_stats()
            ef = float(np.mean([v[0] for v in error_dct.values()])) if error_dct else np.nan
            self._history.append(dict(iter=int(self.curiter), err_amp=ef,
                                      phase_rms=st['phase_rms'], g_mean=st['g_mean']))

            self._plot_progress()

        return error_dct

    def _batch_update(self, idx):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        M = self.M
        eps = torch.finfo(self.rdt).eps
        use_max = (self._denom == 'max')

        active = [self.curiter >= self.p.slice_start_iteration[i] for i in range(N)]
        # D4: probe_update_start gates the psi corrections at *every* slice,
        # because on CPU the gate lives inside probe_update(), which the
        # multislice loop calls once per slice.
        do_pr = self.curiter >= self.p.probe_update_start

        if not active[-1] and not self._warned_inactive_last:
            logger.warning("final slice inactive: this engine excludes it from the Fourier "
                           "step (exit = psi); the CPU engine leaves pod.object set and "
                           "does not. Results will differ until the slice switches on.")
            self._warned_inactive_last = True

        # ---------------- forward ----------------
        ob_p = [self._gather(self.ob_g[i], idx) for i in range(N)]      # (B, ny, nx)
        psi = self.pr_g[None].expand(B, M, self.fny, self.fnx)          # incident at slice 0

        psis, exits = [], []
        for i in range(N):
            psis.append(psi)
            e = psi * ob_p[i][:, None] if active[i] else psi
            exits.append(e)
            if i < N - 1:
                psi = self._apply(self._fw_g[i], e)

        # ---------------- Fourier modulus projection ----------------
        F = self._apply(self._ff_fw_g, exits[-1])                       # (B, M, ny, nx)
        Imod = (F.real ** 2 + F.imag ** 2).sum(dim=1)                   # (B, ny, nx)
        amod = torch.sqrt(Imod.clamp_min(0))

        am = self.amp_g[idx]
        mk = self.mask_g[idx]
        I_meas = self.I_g[idx]

        # ptypy's absolute 1e-10 floor, not finfo eps
        fm = (1.0 - mk) + mk * am / (amod + 1e-10)
        e_proj = self._apply(self._ff_bw_g, F * fm[:, None].to(self.cdt))

        # ---------------- per-view errors (D9) ----------------
        npix = float(self.fny * self.fnx)
        msum = mk.sum(dim=(-2, -1)).clamp_min(1.0)
        d = amod - am
        err_fmag = (mk * d * d).sum(dim=(-2, -1)) / msum
        err_exit = ((e_proj - exits[-1]).abs() ** 2).mean(dim=(-2, -1)).sum(dim=1)
        if self.p.compute_log_likelihood:
            err_phot = (mk * (Imod - I_meas) ** 2 / (I_meas + 1.0)).sum(dim=(-2, -1)) / npix
        else:
            err_phot = torch.zeros_like(err_fmag)

        # ---------------- backward ----------------
        # 'max': the per-view increment is normalised exactly as ptypy does and
        #        the increments are averaged over the batch.
        # 'local' / 'unscattered': numerator and denominator are accumulated
        #        separately over the batch and divided once, which is the
        #        preconditioned mini-batch update.
        num_o = [torch.zeros_like(self.ob_g[0]) for _ in range(N)]
        den_o = [torch.zeros(self.ob_g[0].shape, dtype=self.rdt, device=self.dev)
                 for _ in range(N)]
        ones_patch = torch.ones((B, self.fny, self.fnx), dtype=self.rdt, device=self.dev)
        phi2 = self._unscattered() if self._denom == 'unscattered' else None

        num_pr = torch.zeros_like(self.pr_g)
        den_pr = torch.zeros((self.fny, self.fnx), dtype=self.rdt, device=self.dev)
        pr_replace = None

        target = e_proj
        for i in range(N - 1, -1, -1):
            delta = target - exits[i]                                   # (B, M, ny, nx)

            if active[i]:
                # local intensity of the field incident on this slice: the
                # multislice analogue of |phi_i|^2, carrying every slice above
                if phi2 is None:
                    w = (psis[i].abs() ** 2).sum(dim=1)                 # (B, ny, nx)
                else:
                    w = phi2[i].expand(B, self.fny, self.fnx)
                grad = (psis[i].conj() * delta).sum(dim=1)              # (B, ny, nx)

                if use_max:
                    pn = w.amax(dim=(-2, -1))[:, None, None]
                    dob = self.ob_step * grad / pn.clamp_min(eps).to(self.cdt)
                    self._scatter_add(num_o[i], idx, dob)
                    self._scatter_add(den_o[i], idx, ones_patch)
                else:
                    self._scatter_add(num_o[i], idx, grad)
                    self._scatter_add(den_o[i], idx, w)
                    # per-view estimate of the same update, for the psi
                    # correction below; identical to the batch update at B = 1
                    dob = self.ob_step * grad / (
                        w + self._reg * w.amax(dim=(-2, -1))[:, None, None]
                    ).clamp_min(eps).to(self.cdt)

                # --- illumination update (D2, D5): uses the UPDATED object
                o_new = ob_p[i] + dob                                   # (B, ny, nx)
                if do_pr:
                    if use_max:
                        on = (M * o_new.abs() ** 2).amax(dim=(-2, -1))[:, None, None, None]
                        dpsi = self.pr_step * o_new.conj()[:, None] * delta \
                            / on.clamp_min(eps).to(self.cdt)
                        if i == 0:
                            num_pr = num_pr + dpsi.sum(dim=0)
                    else:
                        ow = M * o_new.abs() ** 2                       # (B, ny, nx)
                        dpsi = self.pr_step * o_new.conj()[:, None] * delta / (
                            ow + self._reg * ow.amax(dim=(-2, -1))[:, None, None]
                        ).clamp_min(eps)[:, None].to(self.cdt)
                        if i == 0:
                            num_pr = num_pr + (o_new.conj()[:, None] * delta).sum(dim=0)
                            den_pr = den_pr + ow.sum(dim=0)
                    psi_new = psis[i] + dpsi
                else:
                    psi_new = psis[i]
            else:
                # D6: inactive slice -> the illumination is replaced outright
                psi_new = target
                if i == 0:
                    pr_replace = target.mean(dim=0)

            if i > 0:
                target = self._apply(self._bw_g[i - 1], psi_new)

        # ---------------- apply updates ----------------
        for i in range(N):
            if not active[i]:
                continue
            if use_max:
                self.ob_g[i] += num_o[i] / den_o[i].clamp_min(1.0).to(self.cdt)
            else:
                dn = den_o[i] + self._reg * den_o[i].amax()
                self.ob_g[i] += self.ob_step * num_o[i] / dn.clamp_min(eps).to(self.cdt)

        if pr_replace is not None:
            self.pr_g = pr_replace.to(self.cdt)
            self._phi_cache = None
        elif do_pr:
            if use_max:
                self.pr_g += num_pr / float(B)
            else:
                dn = den_pr + self._reg * den_pr.amax()
                self.pr_g += self.pr_step * num_pr / dn.clamp_min(eps).to(self.cdt)
            self._phi_cache = None

        if self.pr_support is not None:
            self.pr_g = self.pr_g * self.pr_support
            self._phi_cache = None

        # ---------------- pack per-view errors ----------------
        ii = idx.detach().cpu().numpy()
        ef = err_fmag.detach().cpu().numpy()
        ep = err_phot.detach().cpu().numpy()
        ee = err_exit.detach().cpu().numpy()
        return {self.view_names[int(k)]: np.array([ef[j], ep[j], ee[j]])
                for j, k in enumerate(ii)}

    # ------------------------------------------------------------------ #
    #  Sync and finalize                                                  #
    # ------------------------------------------------------------------ #
    def _sync_to_ptypy(self):
        """Copy GPU state into the ptypy containers for plotting and autosave."""
        N = int(self.p.number_of_slices)
        for i in range(N):
            arr = self.ob_g[i].detach().cpu().numpy()
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._pr_storage.data[:] = self.pr_g.detach().cpu().numpy()

        self.ob.fill(self._object[0])
        for i in range(1, N):
            self.ob *= self._object[i]

    def engine_finalize(self):
        self._sync_to_ptypy()

        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness
        slices_info.slice_start_iteration = self.p.slice_start_iteration
        slices_info.batch_size = self.p.batch_size
        slices_info.object_step = self.ob_step
        slices_info.probe_step = self.pr_step
        slices_info.object_denom = self._denom
        slices_info.denom_reg = self._reg
        slices_info.seed = int(self.p.seed)
        slices_info.corrupt_slice = str(self.p.corrupt_slice)
        slices_info.corrupt_mode = str(self.p.corrupt_mode)
        slices_info.corrupt_amplitude = float(self.p.corrupt_amplitude)
        slices_info.corrupt_smooth = float(self.p.corrupt_smooth)
        slices_info.corrupt_at_iter = int(self.p.corrupt_at_iter)
        slices_info.corrupt_seed = int(self.p.corrupt_seed)
        slices_info.shear_a_c = float(self._a_c) if self._a_c else float('nan')

        # per-iteration per-slice statistics, so a corrupted run can be
        # differenced against a clean one without extra instrumentation
        hist = [h for h in self._history if 'phase_rms' in h]
        if hist:
            slices_info.history_iter = np.array([h['iter'] for h in hist])
            slices_info.history_err = np.array([h['err_amp'] for h in hist])
            slices_info.history_phase_rms = np.array([h['phase_rms'] for h in hist])
            slices_info.history_g_mean = np.array([h['g_mean'] for h in hist])

        slices_info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                               for ob in self._object}

        header = {'description': 'multi-slice result details (torch engine).'}
        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info('Saving to %s' % self.p.fslices)
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()

    def _plot_progress(self):
        try:
            # 1. Extract Data
            probe = (self.pr_g[0].abs() ** 2).cpu().numpy()
            first = self.ob_g[0].angle().cpu().numpy()
            final = self.ob_g[-1].angle().cpu().numpy()

            # 2. Crop Object Slices ONLY (Keep Probe full-sized)
            crop = 512
            d = crop // 2
            first_cropped = first[d:-d, d:-d]
            final_cropped = final[d:-d, d:-d]

            # 3. Phase Processing on Object Slices
            rec_ph_0 = unwrap_phase(first_cropped)
            rec_ph_1 = unwrap_phase(final_cropped)

            # Offset removal
            try:
                rec_ph_0 = rec_ph_0 - np.mean(rec_ph_0[124:127, 640:680])
                rec_ph_1 = rec_ph_1 - np.mean(rec_ph_1[100:250, 350:500])
            except IndexError:
                pass

            # 4. Extents & Pixel Scaling
            psize = getattr(self.p, 'psize', [1e-6, 1e-6])
            
            # Object Extent
            Ny_obj, Nx_obj = rec_ph_0.shape
            extent_obj = [0, Nx_obj * psize[1] * 1.e6, 0, Ny_obj * psize[0] * 1.e6]
            
            # Probe Extent (Uncropped)
            Ny_pr, Nx_pr = probe.shape
            extent_pr = [0, Nx_pr * psize[1] * 1.e6, 0, Ny_pr * psize[0] * 1.e6]

            # 5. Dynamic Zoom Region (Centered 25% window of the cropped object)
            x_min, x_max = extent_obj[0], extent_obj[1]
            y_min, y_max = extent_obj[2], extent_obj[3]
            
            x_center = x_min + 0.50 * (x_max - x_min)
            y_center = y_min + 0.50 * (y_max - y_min)
            zoom_width = 0.25 * (x_max - x_min) 
            zoom_height = 0.25 * (y_max - y_min)

            zoom_xlim = (x_center - zoom_width / 2, x_center + zoom_width / 2)
            zoom_ylim = (y_center - zoom_height / 2, y_center + zoom_height / 2)

            # 6. Plotting Setup
            fig, axes = plt.subplots(2, 3, figsize=(11.4, 7), dpi=300)
            plt.suptitle(f'Iteration {self.curiter} Progress')

            columns = [
                {'data': probe, 'extent': extent_pr, 'title': 'Probe Intensity', 'cmap': 'viridis', 'label': 'Intensity'},
                {'data': rec_ph_0, 'extent': extent_obj, 'title': 'First Object Slice', 'cmap': 'Greys_r', 'label': ''},
                {'data': rec_ph_1, 'extent': extent_obj, 'title': f'Final Slice ({self.p.number_of_slices - 1})', 'cmap': 'Greys_r', 'label': 'phase shift [rad]'}
            ]

            for col_idx, col in enumerate(columns):
                data, extent = col['data'], col['extent']
                
                # --- Top Row: Full View ---
                ax_full = axes[0, col_idx]
                ax_full.set_title(col['title'])
                im_full = ax_full.imshow(data, interpolation='None', cmap=col['cmap'], extent=extent)
                ax_full.set_xticks([])
                ax_full.set_yticks([])

                if col_idx > 0:
                    rect = Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1]-zoom_xlim[0], zoom_ylim[1]-zoom_ylim[0], 
                                        linewidth=1, edgecolor='r', facecolor='none')
                    ax_full.add_patch(rect)

                sb_x = extent[1] * 0.70
                sb_y = extent[3] * 0.08
                ax_full.plot([sb_x, sb_x + 2.0], [sb_y, sb_y], lw=3, c='r')
                ax_full.text(sb_x + 1.0, sb_y + (extent[3] * 0.03), '2 μm', ha='center', c='r', fontweight='semibold', fontsize=11)

                if col['label']:
                    fig.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04, label=col['label'])

                # --- Bottom Row: Zoom View ---
                ax_zoom = axes[1, col_idx]
                ax_zoom.set_title('zoom in')
                im_zoom = ax_zoom.imshow(data, interpolation='None', cmap=col['cmap'], extent=extent)
                ax_zoom.set_xticks([])
                ax_zoom.set_yticks([])
                
                if col_idx == 0:
                    pr_cx, pr_cy = extent_pr[1] / 2, extent_pr[3] / 2
                    ax_zoom.set_xlim(pr_cx - zoom_width/2, pr_cx + zoom_width/2)
                    ax_zoom.set_ylim(pr_cy - zoom_height/2, pr_cy + zoom_height/2)
                else:
                    ax_zoom.set_xlim(zoom_xlim)
                    ax_zoom.set_ylim(zoom_ylim)

                curr_xlim = ax_zoom.get_xlim()
                curr_ylim = ax_zoom.get_ylim()
                z_sb_x = curr_xlim[0] + (curr_xlim[1] - curr_xlim[0]) * 0.60
                z_sb_y = curr_ylim[0] + (curr_ylim[1] - curr_ylim[0]) * 0.08
                
                ax_zoom.plot([z_sb_x, z_sb_x + 0.5], [z_sb_y, z_sb_y], lw=3, c='r')
                ax_zoom.text(z_sb_x + 0.25, z_sb_y + ((curr_ylim[1] - curr_ylim[0]) * 0.03), '500 nm', ha='center', c='r', fontweight='semibold', fontsize=11)

                if col['label']:
                    fig.colorbar(im_zoom, ax=ax_zoom, fraction=0.046, pad=0.04, label=col['label'])

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_progress_dir, f"iter_{self.curiter:04d}.png"))
            plt.close(fig)

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not plot iteration progress: {e}")