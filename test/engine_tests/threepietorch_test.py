"""
Parity of the torch multislice ePIE engine against the numpy one.

Reference    ptypy/custom/threepie.py       ThreePIE, numpy, subclass of EPIE
Under test   ptypy/custom/threepietorch.py  ThreePIETorch, per-slice state in torch
Shared       ptypy/custom/torch_helpers.py  propagator extraction

The suite is in five layers, cheapest first, so that a failure points at one
thing:

  1. The slice-to-slice propagator, on its own: what the two engines build for
     a given separation and energy, and its limit as the separation goes to
     zero.
  2. The per-slice forward model: the same probe, slices and separations must
     give the same incident and exit wave at every slice.
  3. One update step, object and illumination separately, from byte-identical
     state. Separately is the point -- a combined step hides which of the two
     is wrong. The object is frozen with beta = 0 and the illumination with
     alpha = 0, which are the two step sizes ptypy's EPIEMixin actually wires
     up (_ob_b = beta, _pr_b = alpha, contrary to the parameter help strings).
  4. Degenerate cases: one slice must be plain ePIE, and slices separated by
     nothing must collapse to their product. `slice_thickness` is a
     SEPARATION, so N slices have N - 1 of them; both engines are checked for
     that off-by-one.
  5. A short end-to-end run on the same MoonFlowerScan data. Bitwise agreement
     is not on offer -- ThreePIE shuffles the positions with an unseeded
     generator and ThreePIETorch walks them sorted -- so what is demanded is
     that both errors fall and that the two reconstructions agree to a stated
     overlap.

Layers 1-4 run the Ptycho at level 4 and drive the two engines by hand on the
same containers, which is the only way to get identical starting state; they
use data_type = 'double' and precision = 'double' so that a disagreement can be
read as algorithmic rather than as float32 noise. The tolerances are set at
1e-11 relative, which is far tighter than the float32 agreement the port has to
have. Layer 5 goes through test/utils.EngineTestRunner like the other engine
tests, in ptypy's default single precision.

Everything runs on the CPU, so no GPU is needed.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho, geometry
from ptypy.custom.threepie import ThreePIE
from ptypy.engines import register
from ptypy.engines.stochastic import EPIE
from test import utils as tu

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from ptypy.custom import torch_helpers
    from ptypy.custom.threepietorch import ThreePIETorch


# The near-field kernel is exp(2i pi (dz / lam) (sqrt(1 - a^2) - 1)), so at the
# MoonFlower geometry (7.2 keV, 225 nm pixels) a micron of separation moves the
# wave by 0.2%: a parity test at that thickness would pass with the propagator
# replaced by the identity. These are the separations used instead -- far
# enough apart that the propagator carries the comparison.
DZ = (5.0e-4, 9.0e-4)

# Relative tolerance for the double-precision layers. Everything below is an
# operation-for-operation comparison of two implementations of the same
# arithmetic, so this is round-off, not a fudge factor.
RTOL = 1e-11


# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------
def _level4_ptycho(outpath, num_frames=20, shape=32, data_type="double"):
    """
    A Ptycho stopped at level 4: data loaded, containers filled, nothing run.

    tu.EngineTestRunner cannot express this -- it always calls P.run() -- and
    the surgical layers need to attach two engines to one set of containers and
    step them by hand.
    """
    p = u.Param()
    p.verbose_level = "error"
    p.data_type = data_type
    p.frames_per_block = 100000
    p.io = u.Param(home=outpath, rfile=None)
    p.io.interaction = u.Param(active=False)
    p.io.autosave = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = "Full"
    p.scans.MF.propagation = "farfield"
    p.scans.MF.data = u.Param(name="MoonFlowerScan", num_frames=num_frames,
                              shape=shape, save=None, photons=1e8, psf=0.0,
                              density=0.2, add_poisson_noise=False)
    p.scans.MF.coherence = u.Param(num_probe_modes=1)
    p.engines = u.Param()
    P = Ptycho(p, level=4)
    # BaseEngine.initialize reads this and Ptycho only creates it in run()
    P.runtime.iter_info = []
    return P


def _roughen(P, seed=5):
    """
    Give the object and the probe some structure before anything is compared.

    ptypy starts the object at 1 plus a whisper of noise. Against a flat object
    every patch of the storage looks alike, so a transposed or offset gather
    would agree with the reference to the last bit; against this one it cannot.
    """
    rng = np.random.default_rng(seed)
    for c in (P.obj, P.probe):
        for s in c.storages.values():
            sh = s.data.shape
            s.data[:] = (s.data
                         * (1.0 + 0.2 * rng.standard_normal(sh))
                         * np.exp(1j * 0.3 * rng.standard_normal(sh)))


@register()
class _ThreePIESorted(ThreePIE):
    """
    ThreePIE walking the positions in sorted order, for layer 5 only.

    ThreePIE reshuffles the view order every iteration with an *unseeded*
    np.random.default_rng(), while ThreePIETorch with shuffle = False walks
    them sorted. Comparing those two head to head measures the reshuffling far
    more than it measures the port: two stock ThreePIE runs on identical data
    agree only to 0.994 on slice 1 and 0.94 on the probe. This subclass removes
    that one difference and nothing else, so the end-to-end layer can compare
    like with like; the stock engine is still run alongside it as the control
    that says how big the ordering noise is.

    Defaults:

    [name]
    default = _ThreePIESorted
    type = str
    help = ThreePIE walking the positions in sorted order
    doc =
    """

    def engine_iterate(self, num=1):
        for _ in range(num):
            error_dct = {}
            for name in sorted(self.di.views.keys()):
                view = self.di.views[name]
                if view.active:
                    error_dct[name] = self.multislice_update(view)
            self.curiter += 1
        return error_dct


def _pars(nslices, thickness, alpha=1.0, beta=1.0, **kw):
    d = dict(number_of_slices=nslices, slice_thickness=thickness, numiter=1,
             alpha=alpha, beta=beta, probe_update_start=0,
             slice_start_iteration=0)
    d.update(kw)
    return d


class _ParityFixture(unittest.TestCase):
    """Temp dir, and the CWD moved into it.

    Both engines write diagnostics next to the CWD -- ThreePIE a PNG per
    iteration, ThreePIETorch a timestamped directory made unconditionally in
    engine_initialize -- so the CWD is moved rather than left pointing at the
    source tree.
    """

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="threepietorch_test")
        self._cwd = os.getcwd()
        os.chdir(self.outpath)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.outpath, ignore_errors=True)

    # -- engine construction -------------------------------------------------
    def numpy_engine(self, P, **kw):
        p = u.Param(name="ThreePIE", fslices=self.outpath + "/np.h5", **kw)
        eng = ThreePIE(P, p)
        eng.initialize()
        eng.prepare()
        return eng

    def torch_engine(self, P, **kw):
        p = u.Param(name="ThreePIETorch", fslices=self.outpath + "/t.h5",
                    device="cpu", precision="double", batch_size=1,
                    shuffle=False, seed=0, plot_interval=0,
                    object_denom="max", **kw)
        eng = ThreePIETorch(P, p)
        eng.initialize()
        eng.prepare()
        return eng

    def both(self, P, nslices, thickness, **kw):
        """
        The numpy engine first, on purpose.

        ThreePIE names its slice containers `<Cobj>_o_<i>` unconditionally;
        torch_helpers.slice_containers steps around a name already taken. Built
        the other way round, ThreePIE walks into ptypy's duplicate-container
        UnboundLocalError.
        """
        kw = _pars(nslices, thickness, **kw)
        return self.numpy_engine(P, **kw), self.torch_engine(P, **kw)

    # -- forward models, transcribed from the two engines --------------------
    @staticmethod
    def np_forward(eng, view):
        """
        ThreePIE.multislice_update's forward pass, and nothing else.

        Returns (psi, exit), each (N, M, ny, nx): the wave incident on each
        slice and the wave leaving it.
        """
        N = int(eng.p.number_of_slices)
        for i in range(N - 1):
            for _, pod in view.pods.items():
                eng._exits[i][pod.pr_view] = (eng._probe[i][pod.pr_view]
                                              * eng._object[i][pod.ob_view])
                eng._probe[i + 1][pod.pr_view] = eng.fw[i](
                    eng._exits[i][pod.pr_view])
        for _, pod in view.pods.items():
            eng._exits[-1][pod.pr_view] = (eng._probe[-1][pod.pr_view]
                                           * eng._object[-1][pod.ob_view])
        pods = list(view.pods.values())
        psi = np.stack([np.stack([eng._probe[i][p.pr_view] for p in pods])
                        for i in range(N)])
        exi = np.stack([np.stack([eng._exits[i][p.pr_view] for p in pods])
                        for i in range(N)])
        return psi, exi

    @staticmethod
    def torch_forward(eng, k):
        """ThreePIETorch._batch_update's forward pass, for the single view k."""
        N = int(eng.p.number_of_slices)
        idx = torch.tensor([k])
        ob_p = [eng._gather(eng.ob_g[i], idx) for i in range(N)]
        psi = eng.pr_g[None].expand(1, eng.M, eng.fny, eng.fnx)
        psis, exits = [], []
        for i in range(N):
            psis.append(psi)
            e = psi * ob_p[i][:, None]
            exits.append(e)
            if i < N - 1:
                psi = eng._apply(eng._fw_g[i], e)
        return (torch.stack(psis)[:, 0].numpy(),
                torch.stack(exits)[:, 0].numpy())

    # -- comparison helpers --------------------------------------------------
    def assert_close(self, got, want, what, rtol=RTOL):
        scale = max(float(np.abs(want).max()), 1e-30)
        err = float(np.abs(np.asarray(got) - np.asarray(want)).max()) / scale
        self.assertLess(err, rtol,
                        f"{what}: relative disagreement {err:.3e} "
                        f"(peak amplitude {scale:.3e})")

    @staticmethod
    def _obj_patch(view, full):
        """
        The patch of a full-frame object array that `view` sees.

        View.slice is (layer, sy, sx); the arrays here are single-layer, so the
        layer index is dropped.
        """
        pod = next(iter(view.pods.values()))
        return full[pod.ob_view.slice[1:]]

    @staticmethod
    def slices_of(eng):
        """The N object slices of either engine as one (N, ny, nx) array."""
        if hasattr(eng, "ob_g"):
            return eng.ob_g.detach().cpu().numpy()
        return np.stack([list(c.storages.values())[0].data[0]
                         for c in eng._object])

    @staticmethod
    def probe_of(eng):
        if hasattr(eng, "pr_g"):
            return eng.pr_g.detach().cpu().numpy()
        return np.array(list(eng._probe[0].storages.values())[0].data)


# ---------------------------------------------------------------------------
#  Layer 1: the slice-to-slice propagator
# ---------------------------------------------------------------------------
@unittest.skipIf(not HAVE_TORCH, "PyTorch is not installed")
class PropagatorParityTest(unittest.TestCase):
    """
    The near-field propagator the two engines put between slices.

    ThreePIE keeps ptypy's Geo.propagator and calls it; ThreePIETorch recovers
    the same operator numerically with torch_helpers.extract_operator and
    applies it with torch FFTs. Both build the Geo with owner=None, so ptypy
    hands both of them a complex64 kernel -- the single-precision kernel is
    shared, not a difference between the engines.
    """

    ENERGY = 7.2                 # keV, as MoonFlowerScan
    PSIZE = 2.24949115e-07       # m, the object pixel at that geometry
    SHAPE = (32, 32)

    def _geo(self, dz):
        g = u.Param(energy=self.ENERGY, distance=dz, psize=self.PSIZE,
                    shape=self.SHAPE, propagation="nearfield")
        return geometry.Geo(owner=None, pars=g)

    @staticmethod
    def _rand(shape, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    def _torch_pair(self, G):
        dev = torch.device("cpu")
        fw = torch_helpers.to_torch(
            torch_helpers.extract_operator(G.propagator.fw, self.SHAPE),
            torch.complex128, dev)
        bw = torch_helpers.to_torch(
            torch_helpers.extract_operator(G.propagator.bw, self.SHAPE),
            torch.complex128, dev)
        return fw, bw

    def test_torch_matches_ptypy_for_a_given_separation(self):
        """Same separation, same energy: the two operators agree on random data."""
        w = self._rand(self.SHAPE)
        for dz in DZ:
            G = self._geo(dz)
            fw, bw = self._torch_pair(G)
            for name, ref, spec in (("fw", G.propagator.fw, fw),
                                    ("bw", G.propagator.bw, bw)):
                want = ref(w)
                got = torch_helpers.apply_torch(
                    spec, torch.as_tensor(w)).numpy()
                err = np.abs(got - want).max() / np.abs(want).max()
                self.assertLess(err, 1e-12,
                                f"dz = {dz}: {name} disagrees by {err:.3e}")

    def test_reduces_to_the_identity_at_zero_separation(self):
        """
        Both sides must tend to the identity as the separation does.

        ptypy's Geo refuses distance = 0 outright ("Distance must not be None
        or 0"), so the limit is taken rather than the value: the kernel phase
        is linear in dz, and at 1e-14 m the operator is the identity to 2e-11.
        """
        with self.assertRaises(ValueError):
            self._geo(0.0)

        w = self._rand(self.SHAPE)
        G = self._geo(1e-14)
        fw, bw = self._torch_pair(G)
        for name, got in (
                ("ptypy fw", G.propagator.fw(w)),
                ("ptypy bw", G.propagator.bw(w)),
                ("torch fw", torch_helpers.apply_torch(
                    fw, torch.as_tensor(w)).numpy()),
                ("torch bw", torch_helpers.apply_torch(
                    bw, torch.as_tensor(w)).numpy())):
            err = np.abs(got - w).max() / np.abs(w).max()
            self.assertLess(err, 1e-9,
                            f"{name} is not the identity at dz -> 0: {err:.3e}")

    def test_backward_undoes_forward(self):
        """|kernel| = 1, so bw is the exact inverse of fw, not an adjoint only."""
        w = self._rand(self.SHAPE)
        G = self._geo(DZ[0])
        fw, bw = self._torch_pair(G)
        back = torch_helpers.apply_torch(
            bw, torch_helpers.apply_torch(fw, torch.as_tensor(w))).numpy()
        self.assertLess(np.abs(back - w).max() / np.abs(w).max(), 1e-6)


# ---------------------------------------------------------------------------
#  Layers 2-4: the two engines side by side on one set of containers
# ---------------------------------------------------------------------------
@unittest.skipIf(not HAVE_TORCH, "PyTorch is not installed")
class ForwardModelParityTest(_ParityFixture):
    """Layer 2: the per-slice forward model."""

    def test_incident_and_exit_waves_match_at_every_slice(self):
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        en, et = self.both(P, 3, list(DZ))

        name = et.view_names[0]
        psi_n, exit_n = self.np_forward(en, P.diff.views[name])
        psi_t, exit_t = self.torch_forward(et, 0)

        for i in range(3):
            self.assert_close(psi_t[i], psi_n[i], f"incident wave at slice {i}")
            self.assert_close(exit_t[i], exit_n[i], f"exit wave at slice {i}")

    def test_the_engines_really_use_that_forward_model(self):
        """
        The transcriptions above are reference code; this closes the loop.

        With both step sizes at zero neither engine changes any state, so all a
        step produces is the error triple -- and every one of the three is a
        functional of the last slice's exit wave and of the far-field
        propagator. Equal errors mean the forward model each engine actually
        runs is the one compared above.
        """
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        en, et = self.both(P, 3, list(DZ), alpha=0.0, beta=0.0)

        name = et.view_names[0]
        err_n = en.multislice_update(P.diff.views[name])
        err_t = et._batch_update(torch.tensor([0]))[name]

        self.assert_close(err_t, err_n, "error triple (fmag, phot, exit)",
                          rtol=1e-9)
        # and nothing moved, so the comparison really was of the forward model
        self.assert_close(self.slices_of(et), self.slices_of(en),
                          "object slices after a zero-step update")


@unittest.skipIf(not HAVE_TORCH, "PyTorch is not installed")
class UpdateStepParityTest(_ParityFixture):
    """
    Layer 3: one update step, object and illumination separately.

    Each test takes one view from identical starting state and compares the
    whole slice stack and the probe. `moved` guards the assertion from the
    other direction: a step that does nothing on both sides would agree
    perfectly and mean nothing.
    """

    def _one_step(self, nslices=3, thickness=None, nviews=1, **kw):
        thickness = list(DZ) if thickness is None else thickness
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        en, et = self.both(P, nslices, thickness, **kw)
        ob0, pr0 = self.slices_of(et).copy(), self.probe_of(et).copy()
        for name in et.view_names[:nviews]:
            en.multislice_update(P.diff.views[name])
            et._batch_update(torch.tensor([et.view_names.index(name)]))
        return en, et, ob0, pr0

    def test_object_update_only(self):
        """
        Freeze the illumination, update the slices, compare the whole stack.

        `probe_update_start` past the end of the run is the only way to freeze
        the probe without also freezing the algorithm: alpha = 0 would freeze
        the probe too, but alpha is also the step size of the psi correction
        at every other slice, and that correction is what carries the residual
        upstream (see test_zero_probe_step_severs_the_stack).
        """
        en, et, ob0, pr0 = self._one_step(probe_update_start=99, nviews=20)
        ob_n, ob_t = self.slices_of(en), self.slices_of(et)
        self.assertEqual(float(np.abs(self.probe_of(en) - pr0).max()), 0.0,
                         "the numpy illumination moved with the probe frozen")
        self.assertEqual(float(np.abs(self.probe_of(et) - pr0).max()), 0.0,
                         "the torch illumination moved with the probe frozen")
        for i in range(ob_n.shape[0]):
            self.assertGreater(np.abs(ob_n[i] - ob0[i]).max(), 1e-3,
                               f"slice {i} did not move, so nothing was tested")
            self.assert_close(ob_t[i], ob_n[i], f"object slice {i}")

    def test_freezing_the_probe_does_not_freeze_the_upstream_slices(self):
        """
        A frozen probe must freeze the probe and nothing else.

        This is a regression test, and it failed on BOTH engines before
        ThreePIE.psi_update / the `do_pr or i > 0` gate in ThreePIETorch went
        in. `probe_update_start` used to suppress the psi correction at every
        slice, not only at slice 0. With psi_i left uncorrected,
        bw(psi_i) is exactly exits[i-1] again, so delta at slice i-1 is
        identically zero: after 20 views at three slices the upstream slices
        had moved by 1.2e-8 and 1.6e-8 -- propagator round-off -- against 1.07
        for the last slice. Zero gradient, not a small one, so no number of
        iterations recovered it. Maiden 2012 applies the same correction to
        the illumination at every slice; only the wave entering slice 0 is the
        probe.
        """
        en, et, ob0, pr0 = self._one_step(probe_update_start=99, nviews=20)
        for label, ob in (("ThreePIE", self.slices_of(en)),
                          ("ThreePIETorch", self.slices_of(et))):
            last = float(np.abs(ob[-1] - ob0[-1]).max())
            for i in range(ob.shape[0] - 1):
                moved = float(np.abs(ob[i] - ob0[i]).max())
                self.assertGreater(
                    moved, 0.01 * last,
                    f"{label}: with the probe frozen, slice {i} moved by "
                    f"{moved:.3e} against {last:.3e} for the last slice -- the "
                    f"psi correction is not reaching upstream")

    def test_zero_probe_step_severs_the_stack(self):
        """
        alpha = 0 does freeze every slice but the last, on both engines.

        Not a bug and not the same thing as the test above: in 3PIE the psi
        correction and the probe update are one formula with one step size, so
        alpha = 0 really does mean 'carry nothing back up the stack'. It is
        pinned here because it looks exactly like the defect above and is not,
        and because it is why alpha = 0 is the wrong way to write an
        object-only test for a multislice engine.
        """
        en, et, ob0, pr0 = self._one_step(alpha=0.0, nviews=20)
        for label, ob in (("ThreePIE", self.slices_of(en)),
                          ("ThreePIETorch", self.slices_of(et))):
            self.assertGreater(np.abs(ob[-1] - ob0[-1]).max(), 1e-3,
                               f"{label}: even the last slice did not move")
            for i in range(ob.shape[0] - 1):
                self.assertLess(
                    float(np.abs(ob[i] - ob0[i]).max()), 1e-6,
                    f"{label}: slice {i} moved at alpha = 0")
        # and the two engines are severed identically
        self.assert_close(self.slices_of(et), self.slices_of(en),
                          "object slices at alpha = 0")

    def test_probe_update_only(self):
        """beta = 0 freezes the slices; only the illumination may move."""
        en, et, ob0, pr0 = self._one_step(beta=0.0)
        self.assertGreater(np.abs(self.probe_of(en) - pr0).max(), 1e-3,
                           "the probe did not move, so nothing was tested")
        self.assertEqual(float(np.abs(self.slices_of(en) - ob0).max()), 0.0,
                         "beta = 0 still moved the numpy slices")
        self.assertEqual(float(np.abs(self.slices_of(et) - ob0).max()), 0.0,
                         "beta = 0 still moved the torch slices")
        self.assert_close(self.probe_of(et), self.probe_of(en), "probe")

    def test_both_updates_together(self):
        """
        The combined step, which is what the engines run.

        Worth having on top of the two above: on both sides the illumination
        update divides by the norm of the ALREADY UPDATED object, so the two
        are coupled and could agree separately and disagree together.
        """
        en, et, ob0, pr0 = self._one_step()
        ob_n, ob_t = self.slices_of(en), self.slices_of(et)
        self.assertGreater(np.abs(ob_n - ob0).max(), 1e-3)
        self.assertGreater(np.abs(self.probe_of(en) - pr0).max(), 1e-3)
        for i in range(ob_n.shape[0]):
            self.assert_close(ob_t[i], ob_n[i], f"object slice {i}")
        self.assert_close(self.probe_of(et), self.probe_of(en), "probe")

    def test_ten_sequential_views(self):
        """
        Ten views in the same order, so an error that only shows up once the
        updated state feeds the next view has somewhere to grow.
        """
        en, et, ob0, pr0 = self._one_step(nviews=10)
        ob_n, ob_t = self.slices_of(en), self.slices_of(et)
        for i in range(ob_n.shape[0]):
            self.assert_close(ob_t[i], ob_n[i], f"object slice {i}")
        self.assert_close(self.probe_of(et), self.probe_of(en), "probe")

    def test_torch_state_does_not_alias_the_ptypy_probe(self):
        """
        ThreePIETorch's probe must be its own storage.

        torch.as_tensor over a contiguous array of matching dtype shares
        memory, so pr_g was a view of the ptypy probe storage: anything that
        wrote P.probe silently rewrote the engine's working probe. In a
        single-engine run nothing does, which is why it went unnoticed, but it
        makes a side-by-side comparison impossible and it is one stray write
        away from a wrong reconstruction.
        """
        P = _level4_ptycho(self.outpath)
        et = self.torch_engine(P, **_pars(2, [DZ[0]]))
        before = et.pr_g.clone()
        for s in P.probe.storages.values():
            s.data[:] = 0.0
        self.assertGreater(float(et.pr_g.abs().max()), 0.0,
                           "pr_g aliases the ptypy probe storage")
        self.assertEqual(float((et.pr_g - before).abs().max()), 0.0)


@unittest.skipIf(not HAVE_TORCH, "PyTorch is not installed")
class DegenerateCaseTest(_ParityFixture):
    """
    Layer 4: the cases where a multislice engine has to stop being one.

    `slice_thickness` is a SEPARATION, so N slices have N - 1 of them. Every
    off-by-one in that list lands here.
    """

    def test_n_slices_have_n_minus_one_separations(self):
        for n in (1, 2, 4):
            # a fresh Ptycho each time: ThreePIE names its slice containers
            # unconditionally and a second one on the same Ptycho collides
            P = _level4_ptycho(self.outpath)
            th = [DZ[0]] * (n - 1) if n > 1 else 1e-6
            en = self.numpy_engine(P, **_pars(n, th))
            et = self.torch_engine(P, **_pars(n, th))
            self.assertEqual(len(en.fw), n - 1)
            self.assertEqual(len(en.bw), n - 1)
            self.assertEqual(len(et._fw_g), n - 1)
            self.assertEqual(len(et._bw_g), n - 1)

    def test_a_wrong_length_separation_list_is_refused_by_both(self):
        """N separations for N slices is the off-by-one; neither may accept it."""
        P = _level4_ptycho(self.outpath)
        with self.assertRaises(AssertionError):
            self.numpy_engine(P, **_pars(3, [DZ[0], DZ[1], DZ[0]]))
        with self.assertRaises(ValueError):
            self.torch_engine(P, **_pars(3, [DZ[0], DZ[1], DZ[0]]))

    def test_the_i_th_gap_uses_the_i_th_thickness(self):
        """
        With unequal separations, a shifted list is a silently wrong model.
        Each engine's gap i is checked against a Geo built for thickness i.
        """
        P = _level4_ptycho(self.outpath)
        en, et = self.both(P, 3, list(DZ))
        geom = next(iter(P.model.scans.values())).geometries[0]
        rng = np.random.default_rng(1)
        sh = tuple(geom.shape)
        w = (rng.standard_normal(sh) + 1j * rng.standard_normal(sh))

        for i, dz in enumerate(DZ):
            g = u.Param(energy=geom.energy, distance=dz,
                        psize=geom.resolution, shape=geom.shape,
                        propagation="nearfield")
            want = geometry.Geo(owner=None, pars=g).propagator.fw(w)
            self.assert_close(en.fw[i](w), want, f"numpy fw[{i}]", rtol=1e-12)
            self.assert_close(
                torch_helpers.apply_torch(et._fw_g[i], torch.as_tensor(w)).numpy(),
                want, f"torch fw[{i}]", rtol=1e-12)
            # and the other thickness must NOT reproduce it
            other = DZ[1 - i]
            g.distance = other
            wrong = geometry.Geo(owner=None, pars=g).propagator.fw(w)
            self.assertGreater(np.abs(wrong - want).max() / np.abs(want).max(),
                               1e-3,
                               "the two test separations are too close to tell "
                               "a shifted list from a correct one")

    def test_one_slice_reproduces_plain_epie(self):
        """
        At number_of_slices = 1 there is no propagation and no backward pass,
        so both engines must be exactly ptypy's stock EPIE on the same view.
        """
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        en, et = self.both(P, 1, 1e-6)
        ep = EPIE(P, u.Param(name="EPIE", numiter=1, alpha=1.0, beta=1.0,
                             probe_update_start=0))
        ep.initialize()
        ep.prepare()

        ob0 = self.slices_of(et)[0].copy()
        pr0 = self.probe_of(et).copy()

        name = et.view_names[0]
        view = P.diff.views[name]

        # ptypy's stock per-view body, from _StochasticEngine.engine_iterate
        ep.fourier_update(view)
        exit_wave = {n: pod.object * pod.probe for n, pod in view.pods.items()}
        ep.object_update(view, exit_wave)
        ep.probe_update(view, exit_wave)
        ob_epie = np.array(next(iter(P.obj.storages.values())).data[0])
        pr_epie = np.array(next(iter(P.probe.storages.values())).data)
        self.assertGreater(np.abs(ob_epie - ob0).max(), 1e-3,
                           "stock EPIE did not move the object")

        et._batch_update(torch.tensor([0]))
        self.assert_close(self.slices_of(et)[0], ob_epie,
                          "ThreePIETorch(N=1) object vs stock EPIE")
        self.assert_close(self.probe_of(et), pr_epie,
                          "ThreePIETorch(N=1) probe vs stock EPIE")

        # and the numpy multislice engine, on its own copy of the same state
        en.multislice_update(view)
        self.assert_close(self.slices_of(en)[0], ob_epie,
                          "ThreePIE(N=1) object vs stock EPIE")
        self.assert_close(self.probe_of(en), pr_epie,
                          "ThreePIE(N=1) probe vs stock EPIE")

    def test_slices_with_no_separation_collapse_to_their_product(self):
        """
        Stack N slices with nothing between them and the exit wave must be the
        one a single object equal to their product would give.

        The separation is taken to zero rather than set to zero because Geo
        refuses distance = 0; at 1e-14 m the propagator is the identity to
        2e-11, which is why the tolerance here is 1e-9 rather than RTOL.

        The single-slice reference is computed by hand rather than by building
        a second engine: two engines on one Ptycho collide on the container ID
        (ptypy answers with UnboundLocalError on nID), so the N=1 forward model
        -- illumination times object, no propagation at all -- is written out
        instead. It is one multiplication, so there is nothing to get wrong.
        """
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        N = 3
        en3, et3 = self.both(P, N, [1e-14] * (N - 1))

        # give the three slices unequal, non-trivial content
        rng = np.random.default_rng(9)
        sh = tuple(et3.ob_g.shape[-2:])
        stack = np.stack([
            (1.0 + 0.15 * rng.standard_normal(sh))
            * np.exp(1j * 0.4 * rng.standard_normal(sh)) for _ in range(N)])
        prod = np.prod(stack, axis=0)

        for i in range(N):
            et3.ob_g[i] = torch.as_tensor(stack[i], dtype=et3.cdt)
            next(iter(en3._object[i].storages.values())).data[0][:] = stack[i]

        name = et3.view_names[0]
        view = P.diff.views[name]
        psi_n, exit3_n = self.np_forward(en3, view)
        psi_t, exit3_t = self.torch_forward(et3, 0)

        # the N=1 answer: the illumination entering slice 0, times the product
        patch = self._obj_patch(view, prod)
        one_n = psi_n[0] * patch
        one_t = psi_t[0] * patch

        self.assert_close(exit3_n[-1], one_n,
                          "numpy: 3 touching slices vs their product", rtol=1e-9)
        self.assert_close(exit3_t[-1], one_t,
                          "torch: 3 touching slices vs their product", rtol=1e-9)
        self.assert_close(exit3_t[-1], exit3_n[-1],
                          "torch vs numpy, 3 touching slices", rtol=1e-11)

    def test_sync_writes_the_product_of_the_slices(self):
        """
        Both engines advertise the ptypy object as the product of the slices.
        With uniform slices that is the single-slice object raised to N.
        """
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        N = 3
        et = self.torch_engine(P, **_pars(N, [DZ[0]] * (N - 1)))
        one = self.slices_of(et)[0].copy()
        for i in range(N):
            et.ob_g[i] = torch.as_tensor(one, dtype=et.cdt)
        et._sync_to_ptypy()
        got = next(iter(P.obj.storages.values())).data[0]
        self.assert_close(got, one ** N, "product of uniform slices")


# ---------------------------------------------------------------------------
#  Layer 5: a short end-to-end run
# ---------------------------------------------------------------------------
@unittest.skipIf(not HAVE_TORCH, "PyTorch is not installed")
class EndToEndTest(unittest.TestCase):
    """
    Three runs through tu.EngineTestRunner on the same MoonFlowerScan data.

      sorted  _ThreePIESorted, the numpy engine with the reshuffle removed
      torch   ThreePIETorch, shuffle = False, single precision, CPU
      stock   ThreePIE exactly as shipped, reshuffling every iteration

    `sorted` and `torch` visit the 200 positions in the same order, so they are
    compared to each other; `stock` is the control that says how much of any
    residual disagreement is merely the order. Bitwise agreement is not on
    offer even for the matched pair -- this is float32, 1600 sequential
    updates, and the two engines reach the same arithmetic by different routes
    (ptypy view gathers against torch index_put_, ptypy's Geo propagator
    against an FFT of the extracted kernel).

    What is measured is the overlap

        |<a, b>| / (|a| |b|)

    over the well-illuminated pixels, which ignores the global phase and scale
    that neither engine can fix. Measured at these settings:

        sorted vs torch   slice0 0.999993  slice1 0.999984  probe 0.999568
        stock  vs sorted  slice0 0.997305  slice1 0.993392  probe 0.939600

    The thresholds below sit just under the first row. The second row is what
    the same thresholds would do to two runs of the *reference against itself*
    -- they would fail it -- which is the sense in which they discriminate.
    """

    NITER = 8
    NSLICES = 2

    SORTED, TORCH, STOCK = "_ThreePIESorted", "ThreePIETorch", "ThreePIE"

    @classmethod
    def _run(cls, name, extra=None):
        pars = u.Param(name=name, numiter=cls.NITER,
                       number_of_slices=cls.NSLICES, slice_thickness=DZ[0],
                       slice_start_iteration=0, probe_update_start=0,
                       alpha=1.0, beta=1.0,
                       fslices=f"{cls.outpath}/{name}.h5")
        if extra:
            pars.update(extra)
        # ThreePIE writes a matplotlib PNG per iteration and ThreePIETorch one
        # every five; neither feeds back into the reconstruction and both cost
        # more than the iterations do.
        with mock.patch.object(ThreePIE, "_plot_iteration_progress",
                               lambda self: None), \
             mock.patch.object(ThreePIETorch, "_plot_progress",
                               lambda self: None):
            return tu.EngineTestRunner(pars, output_path=cls.outpath,
                                       output_file=name, autosave=False,
                                       verbose_level="warning")

    @staticmethod
    def _errors(P):
        return np.array([v["error"][0] for v in P.runtime.iter_info])

    @staticmethod
    def _overlap(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        return float(abs(np.vdot(a, b))
                     / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def _mask(P):
        st = next(iter(P.obj.storages.values()))
        cvg = st.get_view_coverage()[0]
        return cvg > 0.5 * cvg.max()

    # The three runs happen once for the whole class: eight iterations over 200
    # frames, and the torch CPU run is an order of magnitude the slower of the
    # two implementations, so this is the expensive part of the suite.
    @classmethod
    def setUpClass(cls):
        cls.outpath = tempfile.mkdtemp(suffix="threepietorch_e2e")
        cls._cwd = os.getcwd()
        os.chdir(cls.outpath)
        try:
            cls.P = {
                cls.SORTED: cls._run(cls.SORTED),
                cls.TORCH: cls._run(cls.TORCH,
                                    dict(device="cpu", precision="single",
                                         batch_size=1, shuffle=False, seed=0,
                                         plot_interval=0, object_denom="max")),
                cls.STOCK: cls._run(cls.STOCK),
            }
        finally:
            os.chdir(cls._cwd)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.outpath, ignore_errors=True)

    # -- accessors -----------------------------------------------------------
    def _engine(self, which):
        return list(self.P[which].engines.values())[-1]

    def _slice(self, which, i):
        return next(iter(self._engine(which)._object[i]
                         .storages.values())).data[0][self._mask(self.P[which])]

    def _product(self, which):
        P = self.P[which]
        return next(iter(P.obj.storages.values())).data[0][self._mask(P)]

    def _probe(self, which):
        return next(iter(self.P[which].probe.storages.values())).data[0]

    # -- tests ---------------------------------------------------------------
    def test_all_three_errors_fall(self):
        for label in (self.SORTED, self.TORCH, self.STOCK):
            err = self._errors(self.P[label])
            self.assertEqual(len(err), self.NITER)
            self.assertLess(err[-1], 0.6 * err[0],
                            f"{label}: the Fourier error did not fall over "
                            f"{self.NITER} iterations ({err[0]:.4g} -> "
                            f"{err[-1]:.4g})")

    def test_the_two_engines_reach_the_same_error(self):
        """
        Same order, same data, same number of updates: the final Fourier error
        is a scalar summary of the whole run and the two must agree on it.
        Measured 656.9 (numpy) against 657.2 (torch), from 1343.0.
        """
        a = self._errors(self.P[self.SORTED])[-1]
        b = self._errors(self.P[self.TORCH])[-1]
        self.assertLess(abs(a - b) / a, 1e-2,
                        f"final Fourier error {a:.6g} (numpy) vs {b:.6g} "
                        f"(torch)")

    def test_the_two_reconstructions_agree(self):
        """The product object, which is what ptypy hands back to the user."""
        self.assertGreater(
            self._overlap(self._product(self.SORTED),
                          self._product(self.TORCH)), 0.9999,
            "the two reconstructions of the product object have diverged")

    def test_the_individual_slices_agree(self):
        """
        The product is not enough: the whole point of a multislice engine is
        the split between the slices, and the product is invariant under moving
        content from one slice to another.
        """
        for i in range(self.NSLICES):
            self.assertGreater(
                self._overlap(self._slice(self.SORTED, i),
                              self._slice(self.TORCH, i)), 0.9999,
                f"slice {i} of the two runs has diverged")

    def test_the_probes_agree(self):
        self.assertGreater(
            self._overlap(self._probe(self.SORTED), self._probe(self.TORCH)),
            0.999, "the two recovered probes have diverged")

    def test_the_port_is_closer_than_the_reshuffling(self):
        """
        The one comparison that calibrates all of the above.

        If the port disagrees with the reference by less than the reference
        disagrees with itself when only the walk order changes, then the
        remaining difference is not attributable to the port. This is a
        statement about the *gap* from 1, not the overlap: 0.99999 and 0.9973
        are both "close to 1" and differ by a factor of 400 in what they mean.
        """
        for label, a, b in (
                ("slice0", self._slice(self.SORTED, 0),
                 self._slice(self.TORCH, 0)),
                ("slice1", self._slice(self.SORTED, 1),
                 self._slice(self.TORCH, 1)),
                ("probe", self._probe(self.SORTED), self._probe(self.TORCH))):
            port = 1.0 - self._overlap(a, b)
            if label == "probe":
                ctl = 1.0 - self._overlap(self._probe(self.STOCK),
                                          self._probe(self.SORTED))
            else:
                i = int(label[-1])
                ctl = 1.0 - self._overlap(self._slice(self.STOCK, i),
                                          self._slice(self.SORTED, i))
            self.assertLess(port, ctl,
                            f"{label}: numpy-vs-torch is {port:.3e} from a "
                            f"perfect overlap, but numpy-vs-numpy under a "
                            f"reshuffle is only {ctl:.3e} -- the port is now "
                            f"the larger source of disagreement")


#  6. Object regularisation.
#
#     obj_smooth_std and clip_object have no numpy counterpart -- ThreePIE has
#     neither -- so this layer is not parity but behaviour: the defaults must
#     leave the engine bit-for-bit as it was, the filter must agree with the
#     one the projectional engines use, and the coverage weighting must do what
#     it claims, which is to smooth the unilluminated border harder than the
#     scanned middle.
# --------------------------------------------------------------------------- #


@unittest.skipIf(not HAVE_TORCH, "PyTorch is not installed")
class ObjectRegularisationTest(_ParityFixture):

    def _engine(self, **kw):
        P = _level4_ptycho(self.outpath)
        _roughen(P)
        return self.torch_engine(P, **_pars(2, 1e-6, **kw))

    def test_the_filter_matches_the_one_ptypy_already_uses(self):
        """torch_helpers.complex_gaussian_filter vs accelerate.base.array_utils.

        The projectional engines' obj_smooth_std is defined by that numpy
        filter, so a differently-shaped blur here would make the same parameter
        value mean two things. Includes a std wider than the array, where a
        single reflection silently truncates the kernel.
        """
        from ptypy.accelerate.base.array_utils import complex_gaussian_filter as ref
        from ptypy.custom.torch_helpers import complex_gaussian_filter as got

        rng = np.random.default_rng(0)
        for shape, std in (((1, 64, 64), 3.0), ((3, 40, 28), 7.5),
                           ((1, 20, 20), 30.0)):
            a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            want = np.stack([ref(a[i], [std, std]) for i in range(shape[0])])
            have = got(torch.as_tensor(a), std).numpy()
            err = np.abs(have - want).max() / np.abs(want).max()
            self.assertLess(err, 1e-12, f"shape {shape}, std {std}: {err:.3e}")

    def test_a_constant_object_survives_smoothing(self):
        """Nothing to smooth means nothing changes, border included.

        A filter that normalises badly, or pads with zeros, eats the edge of a
        flat field; that would show up as a dark rim on every reconstruction.
        """
        from ptypy.custom.torch_helpers import complex_gaussian_filter as got

        flat = torch.full((2, 48, 37), 0.7 - 0.3j, dtype=torch.complex128)
        self.assertTrue(torch.allclose(got(flat, 9.0), flat, atol=1e-14))

    def test_the_defaults_change_nothing(self):
        """obj_smooth_std = None and clip_object = None must be exactly inert."""
        eng = self._engine()
        before = eng.ob_g.clone()
        eng._regularise_object()
        self.assertTrue(torch.equal(eng.ob_g, before))
        self.assertIsNone(eng._smooth_w, "no coverage map should have been built")

    def test_clipping_bounds_the_amplitude_and_keeps_the_phase(self):
        eng = self._engine(clip_object=(0.0, 1.0))
        eng.ob_g = eng.ob_g * 4.0
        phase = torch.angle(eng.ob_g).clone()
        eng._regularise_object()
        self.assertLessEqual(float(eng.ob_g.abs().max()), 1.0 + 1e-12)
        self.assertTrue(torch.allclose(torch.angle(eng.ob_g), phase, atol=1e-12))

    def test_smoothing_hits_the_dark_border_hardest(self):
        """The point of the coverage weighting.

        A flat blur would damp the scanned middle as hard as the border and
        take the resolution with it. The weight must be near 1 where no
        illumination ever lands and near 0 at the median coverage, so this
        compares how much a common perturbation is damped in each region.
        """
        eng = self._engine(obj_smooth_std=3.0)
        cov = eng._coverage()
        self.assertGreater(float(cov.max()), 0.0, "no illumination anywhere")
        w = eng._smooth_weight()

        lit = cov > cov.max() * 0.5
        dark = cov == 0
        self.assertTrue(bool(lit.any()) and bool(dark.any()),
                        "fixture has no bright and dark regions to compare")
        self.assertGreater(float(w[dark].mean()), 0.9)
        self.assertLess(float(w[lit].mean()), 0.2)

        # and end to end: the same speckle is damped more in the dark region
        rng = np.random.default_rng(3)
        noise = torch.as_tensor(
            rng.standard_normal(tuple(eng.ob_g.shape[-2:])), dtype=eng.rdt)
        eng.ob_g = torch.ones_like(eng.ob_g) + 0.1 * noise
        eng._regularise_object()
        resid = (eng.ob_g[0] - 1.0).abs()
        self.assertLess(float(resid[dark].mean()), float(resid[lit].mean()))


if __name__ == "__main__":
    unittest.main()
