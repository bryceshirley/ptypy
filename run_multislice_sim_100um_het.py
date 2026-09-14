"""
Simulate a many-layer specimen from the finished NanoMAX 434 reconstruction.

Same skeleton as run_born_diagnostics.py: the data block is identical to the
reconstruction's, so ptypy rebuilds the same views, positions, masks and far
field, and the MultisliceSimulator engine inherits the diagnostics engine's
(slices, probe) pairing against the recorded frames. It then

    1. segments the particles out of the reconstructed particle slice,
    2. scatters them over SIM_SLICES layers (random layer, random lateral
       shift up to MAX_SHIFT_PX), keeping the grating slice as a static
       entrance layer,
    3. runs exact multislice through the synthetic stack, and
    4. writes a simulated .ptyd plus the ground-truth layer stack.

Reconstruct the simulated data afterwards with run_firstborn_sim.py, which
reads the same constants below so the file names line up.

    python run_multislice_sim.py [scannr] [cropping] [slice_thickness_um]
"""

import os
import sys
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import multislicesim       # registers MultisliceSimulator

# convergence-angle knob: real-space zoom of the probe (flux-preserving),
# NA scales by this factor. 1.0 = baseline; 1.5 = "slightly more convergence".
PROBE_NA_SCALE = float(sys.argv[1]) if len(sys.argv) >= 2 else 3.0
# photon energy [keV]. The real NanoMAX data is 8 keV, and the source object/
# probe/positions reused here are all 8 keV artifacts, so we simulate true-to-
# data at 8 keV. Axial resolution (dz ~ lam/NA^2) is pushed instead via NA:
# NA 3.0 at 8 keV gives dz ~= 17 um, finer than 10 keV / NA 2.5. (A genuine
# 10 keV synthetic world would need the source particles resampled onto a
# 10 keV grid + bypassing the real-data consistency check.)
PHOTON_ENERGY_KEV = 8.0
# focus-placement knob: extra Fresnel propagation of the probe [m] before it
# illuminates layer 0. Baseline focus is -700 um (upstream of the stack); +dz
# pushes the beam further downstream of focus, enlarging the entrance-layer
# footprint (effective defocus more negative). +700 um -> total ~ -1400 um so
# layer 0 is no longer the tightest, poorly-overlapped plane.
PROBE_DEFOCUS = 700.e-6

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
iteration        = 100        # rec_crop-512_slices-*_iter-0100.h5
slices_tag       = None       # substring picking one slice file, or None

# the simulated specimen -- run_firstborn_sim.py repeats these constants
SIM_SLICES       = 30         # blob layers only, no static zone-plate layer
SIM_THICKNESS    = (SIM_SLICES - 1) * 100.e-6   # Darren's stress test: 100 um
                                 # slice separation, a few DOF at 30-40 nm res
MIN_PARTICLE_PX  = 5000       # keeps only the two real centre blobs; the
                              # small segmented specks are rec noise
PARTICLE_COPIES  = 60         # non-multiple of SIM_SLICES: random 1-2 copies
                              # per particle per layer -> layers differ in
                              # density, size mix and orientation
TRANSFORM_COPIES = True       # random flip / rotation / scale per copy
COPY_SCALE       = (0.5, 1.5) # per-copy size range; small enough that
                              # 2 * SIM_SLICES copies pack without overlap
NO_OVERLAP       = True       # no two copies overlap in projection
OVERLAP_MARGIN   = 10          # min projected gap between copies [px]
MAX_SHIFT_PX     = 80         # unused when NO_OVERLAP (random placement)
LAYER_SEED       = 2
POISSON_NOISE    = False

if len(sys.argv) >= 2:
    scannr = int(sys.argv[1])
if len(sys.argv) >= 3:
    cropping = int(sys.argv[2])
if len(sys.argv) >= 4:
    slice_thickness = int(sys.argv[3]) * 1.e-6

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

# every sim run gets its own tag, so nothing is overwritten; the stamp is
# broadcast so all MPI ranks write under the same name
from datetime import datetime
SIM_TAG = comm.bcast('sim_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                     root=0)

# every sim lives in its own timestamped directory: data, truth, figures and
# a rec/ subfolder for reconstructions of THIS sim all in one place
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
p.scans.scan00.data.save = None                 # the .ptyd already exists
p.scans.scan00.data.dfile = path_data
p.scans.scan00.data.center = (1281, 772)
p.scans.scan00.data.cropOnLoad = True
p.scans.scan00.data.xMotorFlipped = True
p.scans.scan00.data.yMotorFlipped = False
p.scans.scan00.data.orientation = (False, True, False)
p.scans.scan00.data.distance = distance_m
p.scans.scan00.data.psize = 75e-6
p.scans.scan00.data.energy = PHOTON_ENERGY_KEV   # override the 8 keV in the file
p.scans.scan00.data.rebin = binning
p.scans.scan00.data.min_frames = 10
p.scans.scan00.data.load_parallel = 'all'

# The illumination model only seeds P.probe; the engine replaces it with the
# reconstructed probe from the paired .ptyr.
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
p.engines.engine00.name = 'MultisliceSimulator'
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
p.engines.engine00.static_slice = -1       # no zone-plate layer, blobs only
p.engines.engine00.particle_slice = -1     # particles live on the last slice
p.engines.engine00.segment_threshold = 0.15
p.engines.engine00.min_particle_px = MIN_PARTICLE_PX
p.engines.engine00.particle_copies = PARTICLE_COPIES
p.engines.engine00.transform_copies = TRANSFORM_COPIES
p.engines.engine00.copy_scale = COPY_SCALE
p.engines.engine00.no_overlap = NO_OVERLAP
p.engines.engine00.overlap_margin_px = OVERLAP_MARGIN
p.engines.engine00.max_shift_px = MAX_SHIFT_PX
p.engines.engine00.layer_seed = LAYER_SEED
p.engines.engine00.poisson_noise = POISSON_NOISE
p.engines.engine00.sim_tag = SIM_TAG
p.engines.engine00.probe_na_scale = PROBE_NA_SCALE
p.engines.engine00.probe_defocus = PROBE_DEFOCUS
p.engines.engine00.source_ptyd = path_data

# where and how
p.engines.engine00.outdir = sim_dir
p.engines.engine00.batch_size = 8
p.engines.engine00.precision = 'double'
p.engines.engine00.device = 'cuda'

############################################################################

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)

# _write_ptyd copies the source .ptyd metadata (8 keV) untouched, but the
# frames were simulated at PHOTON_ENERGY_KEV. Patch the copied meta so a
# PtydScan reconstruction rebuilds the geometry the data were simulated at.
if comm.Get_rank() == 0:
    import h5py, glob as _glob
    for _pt in _glob.glob(sim_dir + '*.ptyd'):
        with h5py.File(_pt, 'r+') as _f:
            if 'meta/energy' in _f:
                _f['meta/energy'][...] = PHOTON_ENERGY_KEV
                print('[energy-patch] %s meta/energy -> %.3f keV'
                      % (os.path.basename(_pt), PHOTON_ENERGY_KEV))

# Outputs in out_dir_sim:
#   sim_slices-5_seed-0.ptyd            the simulated data (PtydScan-ready)
#   sim_slices-5_seed-0_truth_iter-0000.h5   ground-truth layers + probe
#   sim_layers.png / sim_assignment.png / sim_frames.png
#   results.json
