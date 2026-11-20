"""Smoke test of the ptychomodels-backed ptypy engine on a simulated scan."""
import ptypy, ptypy.core
from ptypy import utils as u
from ptypy.custom import ptychomodels_born  # noqa: F401  registers the engine

p = u.Param()
p.verbose_level = "info"
p.io = u.Param(home="./ptypy_ptychomodels_demo", autosave=u.Param(active=False), autoplot=u.Param(active=False),
               interaction=u.Param(active=False))
p.scans = u.Param()
p.scans.MF = u.Param(name="Full", data=u.Param(
    name="MoonFlowerScan", shape=64, num_frames=40, save=None,
    density=0.2, photons=1e8, psf=0.0, block_wait_count=1))
p.engines = u.Param()
p.engines.born = u.Param(name="IntersliceBorn_ptychomodels", numiter=3, number_of_slices=3,
                         order=1, background="mean", slice_thickness=2e-7,
                         slice_pad="1", seed=5, device="cuda",
                         fslices="./ptypy_ptychomodels_demo/slices.h5")
P = ptypy.core.Ptycho(p, level=5)
print("errors per iteration:", [list(map(float, e)) for e in P.engines["born"]._err_history])
