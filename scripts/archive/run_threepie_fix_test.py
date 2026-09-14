"""
Paired test: can ThreePIETorch be stabilised on the 10-slice sim data?

Copy of run_firstborn_sim.py with the engine block parameterised from the
command line, so several step/stagger configurations run against the SAME
simulated .ptyd that run_firstborn_sim.py uses. Run each config from its own
working directory -- the engine drops iteration_progress_* into the cwd.

    python run_threepie_fix_test.py <label> <niter> <object_step> \
        <probe_step> <stagger> [<init_slices.h5> <probe.ptyr>]

label        tag carried in every output filename
niter        iterations
object_step  ThreePIETorch object_step (default engine value is 1.0)
probe_step   ThreePIETorch probe_step  (default engine value is 1.0)
stagger      iterations between slice activations; 0 = all slices from iter 0
             (slice i becomes active at iteration i * stagger)
init_slices  optional fslices .h5 to warm-start the slice stack from
probe_ptyr   optional .ptyr whose probe seeds the run (illumination 'recon');
             required together with init_slices
"""

import os
import sys
import glob
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import threepietorch, firstborntorch

label       = sys.argv[1]
Niter       = int(sys.argv[2])
object_step = float(sys.argv[3])
probe_step  = float(sys.argv[4])
stagger     = int(sys.argv[5])
init_h5     = sys.argv[6] if len(sys.argv) >= 7 else None
probe_ptyr  = sys.argv[7] if len(sys.argv) >= 8 else None
if bool(init_h5) != bool(probe_ptyr):
    raise SystemExit('give init_slices and probe_ptyr together or not at all')

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

# must match run_multislice_sim.py
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
path_rec = (f'{out_dir_rec}rec_{sim_stem}_fixtest-{label}'
            '_%(engine)s_%(iterations)04d.ptyr')

slice_thickness = SIM_THICKNESS / (SIM_SLICES - 1)

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(out_dir_rec, exist_ok=True)

p = u.Param()
p.verbose_level = 3
p.run = 'fix%s' % label

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
p.scans.scan00.data.name = 'PtydScan'
p.scans.scan00.data.dfile = path_sim
p.scans.scan00.data.source = None

p.scans.scan00.illumination = u.Param()
if probe_ptyr:
    # warm start: the probe of a finished run, untouched by aperture or
    # diversity, so the engine begins exactly where that run left off
    p.scans.scan00.illumination.model = 'recon'
    p.scans.scan00.illumination.recon = u.Param(rfile=probe_ptyr)
    p.scans.scan00.illumination.aperture = None
    p.scans.scan00.illumination.diversity = None
else:
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
p.engines.engine00.name = 'ThreePIETorch'
p.engines.engine00.precision = 'single'
p.engines.engine00.object_denom = 'max'
p.engines.engine00.numiter = Niter
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = probe_support
p.engines.engine00.probe_center_tol = 1
p.engines.engine00.batch_size = batch_size
p.engines.engine00.number_of_slices = SIM_SLICES
p.engines.engine00.slice_thickness = slice_thickness
p.engines.engine00.object_step = object_step
p.engines.engine00.probe_step = probe_step
p.engines.engine00.slice_start_iteration = (
    [i * stagger for i in range(SIM_SLICES)] if stagger > 0 else 0)
if init_h5:
    p.engines.engine00.init_slices = init_h5
p.engines.engine00.fslices = (f'{out_dir_rec}rec_{sim_stem}'
                              f'_fixtest-{label}_iter-{Niter:0>4}.h5')

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)
