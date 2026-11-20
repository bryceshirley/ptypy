"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".

The multislice engine of the ptychomodels package: the Lie-Trotter sweep
with 3PIE driving it, which is the same algorithm as `moonflower_ThreePIE.py`
written against torch. At number_of_slices = 1 there is no propagation left
in the sweep and the engine is ordinary ePIE.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
from ptypy.custom import ptychomodels_multislice  # noqa: F401  registers the engine

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"

# set home path
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])

# saving intermediate results
p.io.autosave = u.Param(active=False)

# opens plotting GUI if interaction set to active)
p.io.autoplot = u.Param(active=True)
p.io.interaction = u.Param(active=True)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'Full'
p.scans.MF.data = u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 200
p.scans.MF.data.save = None

# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'Multislice_ptychomodels'
p.engines.engine00.numiter = 200
p.engines.engine00.probe_center_tol = None
p.engines.engine00.alpha = 1
p.engines.engine00.beta = 1
p.engines.engine00.probe_update_start = 0
p.engines.engine00.number_of_slices = 2
p.engines.engine00.slice_thickness = 60e-9

# 'cpu' runs the same code without a GPU
p.engines.engine00.device = 'cuda'
# 3PIE reads the object the position before it wrote, so sweeping the
# positions in scan order stalls it; reshuffling every iteration is what
# ptypy's own stochastic engines do
p.engines.engine00.shuffle = True
p.engines.engine00.seed = 5
# the per-slice objects, which the product in p.obj does not keep apart
p.engines.engine00.fslices = "/".join([tmpdir, "ptypy", "slices_multislice.h5"])

# For a gradient step instead of the projection, swap the two lines below.
# alpha and beta are then Adam learning rates rather than projection step
# sizes, so the default of 1.0 is far too large, and positions may be
# batched, which 3PIE refuses because its ordering is the algorithm.
#   p.engines.engine00.step = 'adam'
#   p.engines.engine00.alpha = 1e-2
#   p.engines.engine00.batch_size = 8

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p, level=5)
