"""
Tests of the ptychomodels-backed engines on the CPU.

Six properties are checked on the MoonFlower synthetic scan, with the same
seeded view order for every engine so that the comparisons do not depend on
the draw:

  A. number_of_slices=1  ->  both engines match EPIE
  B. order = number_of_slices  ->  the series is the multislice sweep, so
     the two engines report the same error from the same state
  C. the background gauge is a factorisation and not an approximation, so
     it does not move the full-order model either
  D. every step rule the two engines offer decreases the Fourier error
  E. the projection steps refuse a batch of positions
  F. the line search refuses an order above 1

The engines run on torch and default to CUDA; everything here sets
device='cpu' so the suite needs no GPU.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho
from test.utils import aligned_ncorr, seeded_view_order

try:
    import torch  # noqa: F401
    import ptypy.custom.ptychomodels_born  # noqa: F401  (registers the engine)
    import ptypy.custom.ptychomodels_multislice  # noqa: F401
    HAVE_PTYCHOMODELS = True
except ImportError:
    HAVE_PTYCHOMODELS = False

# Matching an independent engine's solution needs enough data to determine it;
# these are the counts threepie_serial_test settled on for the same comparison.
NUMITER = 60
NFRAMES = 200
# The step rules only have to be shown to descend, which they do long before
# they converge.
NUMITER_SHORT = 20
SHAPE = 64
THICK = 5e-7
SEED = 5

BORN = "IntersliceBorn_ptychomodels"
MULTISLICE = "Multislice_ptychomodels"


@unittest.skipIf(not HAVE_PTYCHOMODELS, "ptychomodels is not installed")
class PtychoModelsEnginesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.outdir = tempfile.mkdtemp(prefix="ptychomodels_engines_test_")
        cls.nrun = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)

    def _params(self, engine_name, numiter, **engine_pars):
        p = u.Param()
        p.verbose_level = "error"
        p.io = u.Param()
        p.io.home = self.outdir
        p.io.autosave = u.Param(active=False)
        p.io.interaction = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = "Full"
        p.scans.MF.data = u.Param()
        p.scans.MF.data.name = "MoonFlowerScan"
        p.scans.MF.data.shape = SHAPE
        p.scans.MF.data.num_frames = NFRAMES
        p.scans.MF.data.density = 0.2
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.0
        p.scans.MF.data.save = None
        p.engines = u.Param()
        p.engines.e0 = u.Param()
        p.engines.e0.name = engine_name
        p.engines.e0.numiter = numiter
        p.engines.e0.numiter_contiguous = 1
        p.engines.e0.probe_center_tol = None
        p.engines.e0.alpha = 1
        p.engines.e0.beta = 1
        p.engines.e0.probe_update_start = 0
        if engine_name != "EPIE":
            p.engines.e0.device = "cpu"
            p.engines.e0.seed = SEED
            # the projection rules are Gauss-Seidel, and sweeping the
            # positions in scan order rather than at random stalls them
            # exactly as it stalls ePIE
            p.engines.e0.shuffle = True
            p.engines.e0.slice_thickness = THICK
            type(self).nrun += 1
            p.engines.e0.fslices = os.path.join(
                self.outdir, "slices_%d.h5" % self.nrun)
        p.engines.e0.update(engine_pars)
        return p

    def _run(self, engine_name, numiter=NUMITER, **engine_pars):
        np.random.seed(SEED)
        with seeded_view_order(SEED):
            P = Ptycho(self._params(engine_name, numiter, **engine_pars), level=5)
        ob = list(P.obj.storages.values())[0].data[0].copy()
        pr = list(P.probe.storages.values())[0].data[0].copy()
        return self._errors(P), ob, pr

    @staticmethod
    def _errors(P):
        out = []
        for it in P.runtime["iter_info"]:
            e = np.asarray(it.get("error"), dtype=float).ravel()
            if e.size and e[0] > 0:
                out.append(float(e[0]))
        return out

    def test_single_slice_matches_epie(self):
        """One slice leaves no propagation in either model, so 3PIE is ePIE
        and so is the first-order series."""
        _, ob_e, pr_e = self._run("EPIE")
        for engine, pars in ((MULTISLICE, {}), (BORN, {"order": 1})):
            with self.subTest(engine=engine):
                _, ob, pr = self._run(engine, number_of_slices=1, **pars)
                self.assertGreater(aligned_ncorr(pr_e, pr), 0.85)
                self.assertGreater(aligned_ncorr(ob_e, ob), 0.85)

    def test_full_order_matches_the_sweep(self):
        """The series is nilpotent after N orders, so at order N it is the
        sweep itself and not an approximation of it. Compared on the error
        of the first iteration, which both engines take from the same state,
        and under adam, whose gradient is the same for both; the projection
        rules differ from each other and would not agree past that point."""
        batched = dict(step="adam", batch_size=NFRAMES, number_of_slices=2)
        e_sweep, _, _ = self._run(MULTISLICE, numiter=1, **batched)
        e_series, _, _ = self._run(BORN, numiter=1, order=2, **batched)
        self.assertAlmostEqual(e_series[0] / e_sweep[0], 1.0, places=5)

    def test_the_gauge_leaves_the_full_series_alone(self):
        """O = a (1 + dO) is exact, so the background only changes where the
        series is truncated, never the model at order N."""
        common = dict(numiter=1, order=2, number_of_slices=2,
                      step="adam", batch_size=NFRAMES)
        e_mean, _, _ = self._run(BORN, background="mean", **common)
        e_vacuum, _, _ = self._run(BORN, background="vacuum", **common)
        self.assertAlmostEqual(e_vacuum[0] / e_mean[0], 1.0, places=5)

    def test_every_step_rule_decreases_the_error(self):
        for engine, pars in (
                (MULTISLICE, {"step": "3pie"}),
                (MULTISLICE, {"step": "adam"}),
                (BORN, {"step": "epie", "order": 2}),
                (BORN, {"step": "adam", "order": 2}),
                (BORN, {"step": "line_search", "order": 1}),
        ):
            with self.subTest(engine=engine, **pars):
                errs, _, _ = self._run(engine, numiter=NUMITER_SHORT,
                                       number_of_slices=2, **pars)
                self.assertGreater(len(errs), 2)
                self.assertLess(errs[-1] / errs[0], 0.8)

    def test_projection_steps_refuse_a_batch(self):
        """They are Gauss-Seidel over positions, so a batch is a different
        algorithm rather than an approximation of one."""
        for engine in (MULTISLICE, BORN):
            with self.subTest(engine=engine):
                with self.assertRaises(ValueError):
                    self._run(engine, numiter=1, number_of_slices=2, batch_size=4)

    def test_line_search_refuses_a_higher_order(self):
        """The detector field is affine in the step at order 1 and a degree-M
        polynomial above it, so the exact minimiser stops being a quartic."""
        with self.assertRaises(ValueError):
            self._run(BORN, numiter=1, number_of_slices=2, order=2,
                      step="line_search")


if __name__ == "__main__":
    unittest.main()
