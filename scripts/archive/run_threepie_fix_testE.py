"""
Test E: 20 iterations of ISSTorch, then ThreePIETorch warm-started
from its slices and probe, in one chained ptypy run.

engine00 (ISSTorch) writes its slice stack to fslices at finalize and
leaves its final probe in the probe container; engine01 (ThreePIETorch)
reads the slices back through init_slices and inherits the probe from the
container. The 3PIE stage runs with no stagger: the warm start replaces the
warm-up.

    python run_threepie_fix_testE.py [niter_born] [niter_3pie]
"""

import os
import sys
import glob
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import threepietorch, isstorch

label      = 'E-fbinit'
Niter_born = int(sys.argv[1]) if len(sys.argv) >= 2 else 20
Niter_3pie = int(sys.argv[2]) if len(sys.argv) >= 3 else 30

detector         = 'eiger4m'
beamtime_basedir = '/home/dnz75396/ptypy/data/'
sample           = '0002_multislice'

scannr           = 434
distance_m       = 4.18
defocus_um       = -700
cropping         = 512
binning          = 1
probe_modes      = 2
batch_size       = 1

SIM_SLICES       = 10
SIM_THICKNESS    = (SIM_SLICES - 1) * 1580.e-6
LAYER_SEED       = 0

probe_support    = 0.3

out_dir     = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_0.00158/'
out_dir_sim = out_dir + 'sim/'
out_dir_rec = out_dir_sim + 'rec/'

cands = sorted(glob.glob(f'{out_dir_sim}sim*_slices-{SIM_SLICES}_seed-'
                         f'{LAYER_SEED}.ptyd'), key=os.path.getmtime)
if not cands:
    raise FileNotFoundError('no sim data; run run_multislice_sim.py first')
path_sim = cands[-1]
sim_stem = os.path.basename(path_sim)[:-5]

fslices_born = (f'{out_dir_rec}rec_{sim_stem}_fixtest-{label}'
                f'_born_iter-{Niter_born:0>4}.h5')
fslices_3pie = (f'{out_dir_rec}rec_{sim_stem}_fixtest-{label}'
                f'_iter-{Niter_3pie:0>4}.h5')

slice_thickness = SIM_THICKNESS / (SIM_SLICES - 1)

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(out_dir_rec, exist_ok=True)

p = u.Param()
p.verbose_level = 3
p.run = 'fix%s' % label.replace('-', '')

p.io = u.Param()
p.io.home = out_dir_rec
p.io.rfile = (f'{out_dir_rec}rec_{sim_stem}_fixtest-{label}'
              '_%(engine)s_%(iterations)04d.ptyr')
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
p.scans.scan00.data.name = 'PtydScan'
p.scans.scan00.data.dfile = path_sim
p.scans.scan00.data.source = None

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

# ---- stage 1: ISSTorch, same settings as the successful sim runs ----
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'ISSTorch'
p.engines.engine00.precision = 'single'
p.engines.engine00.object_denom = 'max'
p.engines.engine00.numiter = Niter_born
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = probe_support
p.engines.engine00.probe_center_tol = 1
p.engines.engine00.batch_size = batch_size
p.engines.engine00.number_of_slices = SIM_SLICES
p.engines.engine00.slice_thickness = slice_thickness
p.engines.engine00.fslices = fslices_born

# ---- stage 2: ThreePIETorch warm-started from stage 1 ---------------------
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ThreePIETorch'
p.engines.engine01.precision = 'single'
p.engines.engine01.object_denom = 'max'
p.engines.engine01.numiter = Niter_3pie
p.engines.engine01.numiter_contiguous = 1
p.engines.engine01.probe_support = probe_support
p.engines.engine01.probe_center_tol = 1
p.engines.engine01.batch_size = batch_size
p.engines.engine01.number_of_slices = SIM_SLICES
p.engines.engine01.slice_thickness = slice_thickness
p.engines.engine01.object_step = 0.5
p.engines.engine01.probe_step = 0.5
p.engines.engine01.init_slices = fslices_born
p.engines.engine01.fslices = fslices_3pie

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)
