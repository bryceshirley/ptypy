"""
Simulate the SPARSE, STRONGLY SCATTERING 3-slice specimen (chord-vs-tangent
demonstration sample).

Same skeleton as run_multislice_sim_100um_het.py, but the engine is
SparseStrongSimulator: three far-separated layers (1.58 mm apart), each
particle transmission rescaled O -> O**PHASE_SCALE (strongly scattering),
and placement enforcing beam-cone disjointness -- no scan position of the
divergent probe illuminates particles on two different layers. The engine
verifies that condition against every scan position and logs it.

Purpose: on this specimen the chord perturbation g = O - 1 (FirstBornTorch)
is exact despite the strong scattering, because every neglected cross term
contains the product of two slices' perturbations inside one beam cone,
which the placement makes zero. The tangent perturbation g = i s
(FirstBornTangentTorch) still carries its O(s^2) per-slice linearisation
error, which sparsity cannot remove. Reconstruct with
run_tangent_vs_chord.py afterwards.

    python run_sparse_strong_sim.py

Runs on CPU (GPU busy elsewhere).
"""

import os
import sys
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import multislicesim       # registers SparseStrongSimulator

PROBE_NA_SCALE = 1.0     # baseline probe: waist upstream, divergent cone
PROBE_DEFOCUS = 0.0      # keep the loaded probe as-is; the footprint per
                         # plane is measured numerically by the engine
PHOTON_ENERGY_KEV = 8.0

if LooseVersion(ptypy.version) >= LooseVersion('0.5.0'):
    ptypy.load_ptyscan_module("nanomax")

############################################################################
# hard coded user input -- must match the reconstruction being simulated from
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

number_of_slices = 2          # of the reconstruction being loaded
slice_thickness  = 1580.e-6   # of the reconstruction being loaded

# which reconstruction to simulate from
engine_name      = 'FirstBornTorch'
iteration        = 100
slices_tag       = None

# the synthetic specimen: 3 far-separated sparse strong layers
SIM_SLICES       = 3
SIM_THICKNESS    = (SIM_SLICES - 1) * 1580.e-6   # 1.58 mm separation
PHASE_SCALE      = 6.0        # strong; smoothing tames the speckle peaks
MAX_PHASE        = 3.0        # no wrap within a slice, but far beyond the
                              # linear regime: |exp(is)-(1+is)| ~ 2.4 at 3 rad
SMOOTH_PX        = 2.5        # sigma of the log-domain particle smoothing
PURE_PHASE       = True       # real potential: |O| = 1, no absorption
SHEAR_NA_SCALE   = 1.5        # scattered-wave NA = 1.5 x probe NA
CONE_MARGIN      = 20         # extra px on top of the cone/shear requirement
MIN_PARTICLE_PX  = 5000       # only the two real centre blobs
PARTICLE_COPIES  = 2          # 2 particles x 2 copies = 4 copies over 3 layers
TRANSFORM_COPIES = True
COPY_SCALE       = (0.35, 0.6)  # small copies pack under the big shear gaps
OVERLAP_MARGIN   = 10         # same-layer projected gap [px]
LAYER_SEED       = 7
POISSON_NOISE    = False

############################################################################
# paths, formatted exactly as the reconstruction script formats them
############################################################################

out_dir      = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_{slice_thickness}/'
out_dir_data = out_dir + 'data/'
out_dir_rec  = out_dir + 'rec/'
out_dir_sim  = out_dir + 'sim/'

path_data    = f'{out_dir_data}data_scan_{scannr:0>6}.ptyd'

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(out_dir_sim, exist_ok=True)

from datetime import datetime
SIM_TAG = comm.bcast('sim_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                     root=0)

sim_dir = f'{out_dir_sim}{SIM_TAG}_slices-{SIM_SLICES}_seed-{LAYER_SEED}/'
if comm.Get_rank() == 0:
    os.makedirs(sim_dir, exist_ok=True)
comm.barrier()


class _Tee:
    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, x):
        self._s.write(x)
        self._f.write(x)
        self._f.flush()

    def flush(self):
        self._s.flush()
        self._f.flush()

    def __getattr__(self, a):
        return getattr(self._s, a)


if comm.Get_rank() == 0:
    import shutil
    shutil.copy2(os.path.abspath(__file__),
                 sim_dir + os.path.basename(__file__))
    _logfh = open(sim_dir + 'terminal.log', 'w')
    sys.stdout = _Tee(sys.stdout, _logfh)
    sys.stderr = _Tee(sys.stderr, _logfh)
    import logging
    for _lg in (logging.getLogger(), logging.getLogger('ptypy')):
        for _h in _lg.handlers:
            if isinstance(_h, logging.StreamHandler):
                _h.stream = sys.stderr

############################################################################
# parameter tree
############################################################################

p = u.Param()
p.verbose_level = 3
p.run = 'sim%d' % scannr

p.io = u.Param()
p.io.home = out_dir_sim
p.io.rfile = f'{out_dir_sim}sim_scan_{scannr:0>6}_%(engine)s_%(iterations)04d.ptyr'
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
p.scans.scan00.data.save = None
p.scans.scan00.data.dfile = path_data
p.scans.scan00.data.center = (1281, 772)
p.scans.scan00.data.cropOnLoad = True
p.scans.scan00.data.xMotorFlipped = True
p.scans.scan00.data.yMotorFlipped = False
p.scans.scan00.data.orientation = (False, True, False)
p.scans.scan00.data.distance = distance_m
p.scans.scan00.data.psize = 75e-6
p.scans.scan00.data.energy = PHOTON_ENERGY_KEV
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

############################################################################
# the simulator "engine": one iterate, no optimisation
############################################################################

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'SparseStrongSimulator'
p.engines.engine00.numiter = 1
p.engines.engine00.numiter_contiguous = 1

# what to simulate from
p.engines.engine00.ground_truth = out_dir_rec
p.engines.engine00.engine_name = engine_name
p.engines.engine00.slices = slices_tag
p.engines.engine00.iteration = iteration
p.engines.engine00.number_of_slices = number_of_slices
p.engines.engine00.slice_thickness = slice_thickness

# the synthetic specimen
p.engines.engine00.sim_slices = SIM_SLICES
p.engines.engine00.sim_thickness = SIM_THICKNESS
p.engines.engine00.static_slice = -1
p.engines.engine00.particle_slice = -1
p.engines.engine00.segment_threshold = 0.15
p.engines.engine00.min_particle_px = MIN_PARTICLE_PX
p.engines.engine00.particle_copies = PARTICLE_COPIES
p.engines.engine00.transform_copies = TRANSFORM_COPIES
p.engines.engine00.copy_scale = COPY_SCALE
p.engines.engine00.overlap_margin_px = OVERLAP_MARGIN
p.engines.engine00.layer_seed = LAYER_SEED
p.engines.engine00.poisson_noise = POISSON_NOISE
p.engines.engine00.sim_tag = SIM_TAG
p.engines.engine00.probe_na_scale = PROBE_NA_SCALE
p.engines.engine00.probe_defocus = PROBE_DEFOCUS
p.engines.engine00.source_ptyd = path_data

# the strong/sparse knobs
p.engines.engine00.phase_scale = PHASE_SCALE
p.engines.engine00.max_phase_rad = MAX_PHASE
p.engines.engine00.smooth_px = SMOOTH_PX
p.engines.engine00.pure_phase = PURE_PHASE
p.engines.engine00.shear_na_scale = SHEAR_NA_SCALE
p.engines.engine00.cone_margin_px = CONE_MARGIN

# where and how -- CPU: the GPU is in use elsewhere
p.engines.engine00.outdir = sim_dir
p.engines.engine00.batch_size = 4
p.engines.engine00.precision = 'double'
p.engines.engine00.device = 'cpu'

############################################################################

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)

# Outputs in sim_dir:
#   <tag>_slices-3_seed-7.ptyd                the simulated data
#   <tag>_slices-3_seed-7_truth_iter-0000.h5  ground-truth layers + probe
#   *_layers.png / *_assignment.png / *_frames.png, results.json
