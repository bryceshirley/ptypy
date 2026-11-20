"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".

The inter-slice Born series engine of the ptychomodels package. The series
is the order-M truncation of multislice: order 1 keeps single scattering
only, and because the series is nilpotent, order = number_of_slices is the
multislice sweep itself rather than an approximation of it. What the order
buys is depth: an update is M propagations deep instead of N, whatever N is.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
from ptypy.custom import ptychomodels_born  # noqa: F401  registers the engine

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
p.engines.engine00.name = 'IntersliceBorn_ptychomodels'
p.engines.engine00.numiter = 200
p.engines.engine00.probe_center_tol = None
p.engines.engine00.alpha = 1
p.engines.engine00.beta = 1
p.engines.engine00.probe_update_start = 0
p.engines.engine00.number_of_slices = 2
p.engines.engine00.slice_thickness = 60e-9

# orders of the series; raising it to number_of_slices makes the model
# exact, and anything above that is clamped because the series terminates
p.engines.engine00.order = 1
# the series expands in O/a - 1, so the gauge sets what is being truncated;
# centring on the window keeps the truncation small
p.engines.engine00.background = 'mean'
# 'cpu' runs the same code without a GPU
p.engines.engine00.device = 'cuda'
# the projection step reads the object the position before it wrote, so
# sweeping the positions in scan order stalls it
p.engines.engine00.shuffle = True
p.engines.engine00.seed = 5
# the per-slice objects, which the product in p.obj does not keep apart
p.engines.engine00.fslices = "/".join([tmpdir, "ptypy", "slices_born.h5"])

# Three step rules are available. 'epie' is the projection above. At order 1
# the detector field is affine in the step, so the exact minimiser along the
# projection's own direction is available and costs no extra forward:
#   p.engines.engine00.step = 'line_search'
# 'adam' descends on the analytic gradients instead, at any order. alpha and
# beta are then learning rates and the default of 1.0 is far too large; it
# has no position ordering, so positions may be batched:
#   p.engines.engine00.step = 'adam'
#   p.engines.engine00.alpha = 1e-2
#   p.engines.engine00.batch_size = 8

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p, level=5)
