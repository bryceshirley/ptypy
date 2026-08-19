"""
First Born vs multislice diagnostics on a finished NanoMAX reconstruction.

This is the reconstruction script with every optimisation parameter removed.
The data block is identical, because the whole point is that ptypy builds the
same views, pods, masks and propagators it built during the reconstruction, so
that positions, data orientation and the far field are the ones actually used
rather than reconstructions of them from files.

The engine reconstructs nothing. It loads the slice stack and probe named by
`ground_truth`, runs every comparison on the real scan, writes the figures and
results.json into `outdir`, and stops after one iterate.

    python run_born_diagnostics.py [scannr] [cropping] [slice_thickness_um]
"""

import os
import sys
import time
import numpy as np
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import borndiagnostics          # registers FirstBornDiagnostics

if LooseVersion(ptypy.version) >= LooseVersion('0.5.0'):
    ptypy.load_ptyscan_module("nanomax")

############################################################################
# hard coded user input -- must match the reconstruction being analysed
############################################################################

detector         = 'eiger4m'
beamtime_basedir = '/home/dnz75396/ptypy/data/'
sample           = '0002_multislice'

scannr           = 434
distance_m       = 4.18
defocus_um       = -700
cropping         = 512
binning          = 1
probe_modes      = 2

number_of_slices = 2
slice_thickness  = 1580.e-6

# which reconstruction to analyse
engine_name      = 'MomentumFirstBorn'   # the engine that wrote the slices
Niter_rec        = 500                   # its numiter, i.e. the fslices name
# The probe and the slices must come from the SAME run. The fslices name
# carries the engine's numiter and the .ptyr name the iteration it reached, so
# the engine picks the .ptyr for `engine_name` with the highest iteration and
# then the slice file at that same iteration. If they disagree it stops: the
# check is prod_j O_j == obj, which engine_finalize guarantees within a run.

# diagnostics
max_order        = 2       # M = N returns exact multislice, the pipeline check
blocks           = None    # None -> 1..N
gauge            = 'both'  # 'none', 'mean' or 'both'
batch_size       = 32
n_positions      = None    # None -> every position
compare_frames   = 3
precision        = 'double'
device           = 'cuda'

if len(sys.argv) >= 2:
    scannr = int(sys.argv[1])
if len(sys.argv) >= 3:
    cropping = int(sys.argv[2])
if len(sys.argv) >= 4:
    slice_thickness = int(sys.argv[3]) * 1.e-6

############################################################################
# paths, formatted exactly as the reconstruction script formats them
############################################################################

out_dir         = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_{slice_thickness}/'
out_dir_data    = out_dir + 'data/'
out_dir_rec     = out_dir + 'rec/'
out_dir_diag    = out_dir + 'diagnostics/'

path_data       = f'{out_dir_data}data_scan_{scannr:0>6}.ptyd'
path_slices     = f'{out_dir_rec}rec_crop-{cropping}_slices-{number_of_slices}_iter-{Niter_rec:0>4}.h5'

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(out_dir_diag, exist_ok=True)

############################################################################
# parameter tree
############################################################################

p = u.Param()
p.verbose_level = 3
p.run = 'diag%d' % scannr

p.io = u.Param()
p.io.home = out_dir_diag
p.io.rfile = f'{out_dir_diag}diag_scan_{scannr:0>6}_%(engine)s_%(iterations)04d.ptyr'
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)

# --- scan: identical to the reconstruction, so the views match -------------
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
p.scans.scan00.data.orientation = {'merlin': (False, False, True),
                                   'pilatus': (False, True, False),
                                   'eiger': (False, True, False),
                                   'eiger1m': (False, True, False),
                                   'eiger4m': (False, True, False)}[detector]
p.scans.scan00.data.distance = distance_m
p.scans.scan00.data.psize = {'pilatus': 172e-6,
                             'merlin': 55e-6,
                             'eiger': 75e-6,
                             'eiger1m': 75e-6,
                             'eiger4m': 75e-6}[detector]
p.scans.scan00.data.rebin = binning
p.scans.scan00.data.min_frames = 10
p.scans.scan00.data.load_parallel = 'all'

# The illumination model only seeds P.probe. The diagnostics engine replaces it
# with the reconstructed probe from the .ptyr, so this block matters only if
# ground_truth has no .ptyr in it.
p.scans.scan00.illumination = u.Param()
p.scans.scan00.illumination.model = None
p.scans.scan00.illumination.aperture = u.Param()
p.scans.scan00.illumination.aperture.form = 'circ'
p.scans.scan00.illumination.aperture.size = 100e-9
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = 1. * defocus_um * 1e-6
p.scans.scan00.illumination.diversity = u.Param()
p.scans.scan00.illumination.diversity.power = (1)

############################################################################
# the diagnostics "engine": one iterate, no optimisation
############################################################################

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'FirstBornDiagnostics'
p.engines.engine00.numiter = 1
p.engines.engine00.numiter_contiguous = 1

# what to analyse
p.engines.engine00.ground_truth = out_dir_rec      # folder, or path_slices
p.engines.engine00.engine_name = engine_name       # picks the matching .ptyr
p.engines.engine00.number_of_slices = number_of_slices
p.engines.engine00.slice_thickness = slice_thickness

# what to measure
p.engines.engine00.outdir = out_dir_diag
p.engines.engine00.max_order = max_order
p.engines.engine00.blocks = blocks
p.engines.engine00.gauge = gauge
p.engines.engine00.batch_size = batch_size
p.engines.engine00.n_positions = n_positions
p.engines.engine00.compare_frames = compare_frames
p.engines.engine00.depth_sweep = True
p.engines.engine00.precond_test = True
p.engines.engine00.precision = precision
p.engines.engine00.device = device

############################################################################

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)

# Everything is in out_dir_diag: results.json, results.npz and the figures.
# The engine object stays available for further poking:
#     eng = list(P.engines.values())[0]
#     eng._results['gauge_none']['eps_order']