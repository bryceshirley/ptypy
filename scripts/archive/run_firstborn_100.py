"""
Chord vs tangent ISS reconstruction of NanoMAX scan 434.

A copy of data/macros/ptycho/reconstruction_example.py with nothing but the
engine block changed: the engine is selected on the command line, the
iteration count is 100, and the output names carry the perturbation tag so
the two runs cannot overwrite each other. Everything else -- data block,
geometry, illumination seed, batch size, probe support -- is identical, so
the perturbation is the only variable.

    python run_firstborn_100.py chord      # ISSTorch,        g = O - 1
    python run_firstborn_100.py tangent    # ISSTangentTorch, g = i s
"""

import os
import sys
import time
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import isstorch  # registers both engines

if LooseVersion(ptypy.version) >= LooseVersion('0.5.0'):
    ptypy.load_ptyscan_module("nanomax")

tag = sys.argv[1] if len(sys.argv) >= 2 else 'chord'
ENGINE = {'chord': 'ISSTorch',
          'tangent': 'ISSTangentTorch'}[tag]

detector         = 'eiger4m'
beamtime_basedir = '/home/dnz75396/ptypy/data/'
sample           = '0002_multislice'

scannr           = 434
distance_m       = 4.18
defocus_um       = -700
cropping         = 512
binning          = 1
probe_modes      = 2
Niter            = 100
batch_size       = 32

number_of_slices = 2
slice_thickness  = 1580.e-6

probe_support    = 0.3

out_dir         = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_{slice_thickness}/'
out_dir_data    = out_dir + 'data/'
out_dir_rec     = out_dir + 'rec/'
path_data       = f'{out_dir_data}data_scan_{scannr:0>6}.ptyd'
path_rec        = f'{out_dir_rec}rec_scan_{scannr:0>6}_%(engine)s_%(iterations)04d.ptyr'

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(out_dir_rec, exist_ok=True)

p = u.Param()
p.verbose_level = 3
p.run = 'scan%d' % scannr

p.io = u.Param()
p.io.home = out_dir_rec
p.io.rfile = path_rec
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)

p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'Full'
p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = probe_modes
p.scans.scan00.coherence.num_object_modes = 1

p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'NanomaxContrast'
p.scans.scan00.data.path = beamtime_basedir + '/raw/' + sample + '/'
p.scans.scan00.data.detector = detector
p.scans.scan00.data.maskfile = None
p.scans.scan00.data.scanNumber = scannr
p.scans.scan00.data.xMotor = 'pseudo/x'
p.scans.scan00.data.yMotor = 'pseudo/y'
p.scans.scan00.data.xMotorAngle = 0
p.scans.scan00.data.yMotorAngle = 0
p.scans.scan00.data.zDetectorAngle = 0.0
p.scans.scan00.data.xyAxisSkewOffset = 1.0
p.scans.scan00.data.shape = cropping
p.scans.scan00.data.save = None                 # the .ptyd already exists
p.scans.scan00.data.dfile = path_data
p.scans.scan00.data.center = (1281, 772)
p.scans.scan00.data.cropOnLoad = True
p.scans.scan00.data.xMotorFlipped = True
p.scans.scan00.data.yMotorFlipped = False
p.scans.scan00.data.orientation = (False, True, False)
p.scans.scan00.data.distance = distance_m
p.scans.scan00.data.psize = 75e-6
p.scans.scan00.data.rebin = binning
p.scans.scan00.data.min_frames = 10
p.scans.scan00.data.load_parallel = 'all'

p.scans.scan00.illumination = u.Param()
p.scans.scan00.illumination.model = None
p.scans.scan00.illumination.aperture = u.Param()
p.scans.scan00.illumination.aperture.form = 'circ'
p.scans.scan00.illumination.aperture.size = 100e-9
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = 1. * defocus_um * 1e-6
p.scans.scan00.illumination.diversity = u.Param()
p.scans.scan00.illumination.diversity.power = (1)

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = ENGINE
p.engines.engine00.precision = 'single'
p.engines.engine00.numiter = Niter
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = probe_support
p.engines.engine00.probe_center_tol = 1
p.engines.engine00.batch_size = batch_size
p.engines.engine00.seed = 7            # same start and order for both runs
p.engines.engine00.number_of_slices = number_of_slices
p.engines.engine00.slice_thickness = slice_thickness
p.engines.engine00.fslices = (f'{out_dir_rec}rec_crop-{cropping}_slices-'
                              f'{number_of_slices}-{tag}_iter-{Niter:0>4}.h5')

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)
