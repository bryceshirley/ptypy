"""
Tests for the RAAR first Born (ISS) projection engine.

Three layers:

  1. Pure core: the branch-space operator identities the engine relies on
     -- S B = Id on the detector range (with an evanescent cutoff and a
     scaled far field), constraint satisfaction and idempotence of the
     Fourier projector, and the RAAR fixed point on model-generated data.
  2. N = 1 equivalence: on MoonFlower data the branch iteration must match
     ptypy's numpy RAAR iterate for iterate (Prop 1(v) of the paper), with
     the probe fixed and with DM-style probe updates on.
  3. A smoke run at N = 2 (the model set only; no uniqueness claim).

Everything runs on the CPU in double precision.
"""

import unittest

import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.engines import by_name

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from ptypy.custom.raarfirstborntorch import (
        branch_detector_field, branch_back, raar_fourier_chunk)
    import ptypy.custom.raarfirstborntorch  # noqa: F401


def _setup(B=3, N=3, M=2, n=32):
    torch.manual_seed(0)
    cdt = torch.complex128
    fy = torch.fft.fftfreq(n)
    a2 = fy[:, None]**2 + fy[None, :]**2
    H_fwd = torch.ones((N, 1, 1, n, n), dtype=cdt)
    for j in range(1, N):
        H_fwd[j, 0, 0] = torch.exp(
            2j * torch.pi * j * 7.0
            * (torch.sqrt((1 - 25 * a2).clamp_min(0)) - 1)) * (25 * a2 <= 1)
    H_rev = H_fwd.conj()
    Sdiag = (H_fwd[:, 0, 0].abs()**2).sum(0)
    scale = 1.7
    ff_fw = lambda x: scale * torch.fft.fft2(x, norm='ortho')   # noqa: E731
    ff_bw = lambda x: torch.fft.ifft2(x, norm='ortho') / scale  # noqa: E731
    ap = lambda spec, x: spec(x)                                # noqa: E731
    P = torch.randn(M, n, n, dtype=cdt)
    return cdt, H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap, P


@unittest.skipIf(not HAVE_TORCH, "no torch")
class PureCoreTest(unittest.TestCase):

    def test_S_rightinverse(self):
        cdt, H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap, P = _setup()
        y = torch.randn(3, 2, 32, 32, dtype=cdt)
        z = torch.zeros(2, 32, 32, dtype=cdt)
        SB = branch_detector_field(
            branch_back(y, H_fwd, Sdiag, ff_bw, ap), z, H_rev, ff_fw, ap)
        self.assertLess((SB - y).abs().max().item(), 1e-12)

    def test_fourier_projector(self):
        cdt, H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap, P = _setup()
        a_det = ff_fw(P)
        W = torch.randn(3, 3, 2, 32, 32, dtype=cdt)
        fmag = (torch.rand(3, 32, 32).double() + 0.5).sqrt()
        PF = (0., 1., 0.)
        Wf, _, _, _ = raar_fourier_chunk(
            W, torch.zeros_like(W), a_det, fmag, None, None, PF,
            H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap)
        Fv = branch_detector_field(Wf, a_det, H_rev, ff_fw, ap)
        af = (Fv.abs()**2).sum(1).sqrt()
        self.assertLess((af - fmag).abs().max().item(), 1e-9)
        Wff, _, _, _ = raar_fourier_chunk(
            Wf, torch.zeros_like(W), a_det, fmag, None, None, PF,
            H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap)
        self.assertLess((Wff - Wf).abs().max().item(), 1e-9)

    def test_raar_fixed_point(self):
        cdt, H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap, P = _setup()
        a_det = ff_fw(P)
        g = 0.3 * torch.randn(3, 3, 1, 32, 32, dtype=cdt)
        phi = torch.fft.ifft2(H_fwd[:, 0] * torch.fft.fft2(P).unsqueeze(0))
        Wt = g * phi.unsqueeze(0)
        Ft = branch_detector_field(Wt, a_det, H_rev, ff_fw, ap)
        fm = (Ft.abs()**2).sum(1).sqrt()
        Wc = Wt.clone()
        for _ in range(3):
            Wc, _, _, _ = raar_fourier_chunk(
                Wc, Wt, a_det, fm, None, None, (1 - 1.8, 0.9, 2.0),
                H_fwd, H_rev, Sdiag, ff_fw, ff_bw, ap)
        self.assertLess((Wc - Wt).abs().max().item(), 1e-9)


def _make_ptycho(tag):
    p = u.Param()
    p.frames_per_block = 100000
    p.verbose_level = "warning"
    p.data_type = "double"
    p.io = u.Param()
    p.io.home = "/tmp/raar_test_%s" % tag
    p.io.interaction = u.Param(active=False)
    p.io.autosave = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = "Full"
    p.scans.MF.propagation = "farfield"
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = "MoonFlowerScan"
    p.scans.MF.data.num_frames = 40
    p.scans.MF.data.shape = 64
    p.scans.MF.data.save = None
    p.scans.MF.data.photons = 1e8
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.density = 0.2
    p.scans.MF.data.add_poisson_noise = False
    p.scans.MF.coherence = u.Param(num_probe_modes=1)
    P = Ptycho(p, level=4)
    P.runtime.iter_info = []
    return P


_COMMON = dict(
    fourier_power_bound=0.0,
    object_inertia=1e-4,
    probe_inertia=1e-9,
    obj_smooth_std=None,
    clip_object=None,
    update_object_first=True,
    overlap_converge_factor=0.05,
    probe_center_tol=None,
    probe_support=None,
    probe_fourier_support=None,
    compute_log_likelihood=False,
)


def _torch_pars(**kw):
    pars = u.Param(name="RAARFirstBornTorch", numiter=5, beta=0.9,
                   number_of_slices=1, slice_thickness=1e-9,
                   device="cpu", precision="double", seed=1,
                   chunk_size=16, object_init="container",
                   exit_init="model", wave_device="same", plot_interval=0,
                   batch_size=16, fslices="/tmp/raar_test_fslices.h5",
                   **_COMMON)
    pars.update(kw)
    return pars


@unittest.skipIf(not HAVE_TORCH, "no torch")
class EquivalenceTest(unittest.TestCase):
    """N = 1 branch RAAR == numpy RAAR, iterate for iterate."""

    def _run(self, probe_on, iters=4, tol=1e-5):
        P1 = _make_ptycho("np_%d" % probe_on)
        P2 = _make_ptycho("tt_%d" % probe_on)
        pus = 0 if probe_on else 100000
        omi = 10 if probe_on else 1
        e1 = u.Param(name="RAAR", numiter=iters, beta=0.9,
                     probe_update_start=pus, overlap_max_iterations=omi,
                     **_COMMON)
        eng1 = by_name("RAAR")(P1, e1)
        eng2 = by_name("RAARFirstBornTorch")(
            P2, _torch_pars(numiter=iters, probe_update_start=pus,
                            overlap_max_iterations=omi))
        for eng in (eng1, eng2):
            eng.initialize()
            eng.prepare()
        names = eng2.view_names
        pods = {n: next(iter(P1.diff.views[n].pods.values())) for n in names}
        obs = next(iter(P1.obj.storages.values()))
        prs = next(iter(P1.probe.storages.values()))
        for it in range(iters):
            eng1.iterate(1)
            eng2.iterate(1)
            ex_np = np.stack([pods[n].exit for n in names])
            ex_tt = (eng2.pr_g.unsqueeze(0)
                     + eng2.W[:, 0]).squeeze(1).cpu().numpy()
            self.assertLess(
                np.abs(ex_np - ex_tt).max() / np.abs(ex_np).max(), tol,
                "exit mismatch at iteration %d" % (it + 1))
        cov = eng2.cov.cpu().numpy()
        ob_np = np.where(cov, obs.data[0], 1.0)
        ob_tt = np.where(cov, eng2.ob_g[0].cpu().numpy(), 1.0)
        self.assertLess(np.abs(ob_np - ob_tt).max()
                        / np.abs(ob_np).max(), tol)
        if probe_on:
            self.assertLess(
                np.abs(prs.data - eng2.pr_g.cpu().numpy()).max()
                / np.abs(prs.data).max(), tol)

    def test_probe_fixed(self):
        self._run(probe_on=False)

    def test_probe_updates(self):
        self._run(probe_on=True)


@unittest.skipIf(not HAVE_TORCH, "no torch")
class SmokeN2Test(unittest.TestCase):
    """The engine runs at N = 2 and the Fourier error falls."""

    def test_n2_error_falls(self):
        # RAAR is not a descent method and err_fmag is measured at the
        # reflected point, so the assertion is on the physical misfit: the
        # log-likelihood of the overlap image PO, which must fall by a
        # large factor. (The numpy RAAR shows the same err_fmag plateau on
        # this data with the probe frozen; N = 2 ends slightly lower.)
        P = _make_ptycho("n2")
        eng = by_name("RAARFirstBornTorch")(
            P, _torch_pars(numiter=30, number_of_slices=2,
                           slice_thickness=3e-4, probe_update_start=100000,
                           overlap_max_iterations=1, beta=0.75,
                           compute_log_likelihood=True))
        eng.initialize()
        eng.prepare()
        trace = []
        for it in range(30):
            err = eng.engine_iterate(1)
            trace.append(np.mean([v[1] for v in err.values()]))
        self.assertLess(min(trace[-5:]), 0.2 * trace[0])


if __name__ == "__main__":
    unittest.main()
