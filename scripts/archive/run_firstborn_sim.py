"""
FirstBornTorch reconstruction of the simulated many-layer data.

The data come from run_multislice_sim.py: exact multislice through a
synthetic specimen built by scattering the reconstructed particles over
SIM_SLICES layers. The constants below must match that script so the .ptyd
path lines up. Everything downstream of the data block is a copy of
run_firstborn_100.py -- same illumination guess, probe support, batch size --
so the only change against the real-data run is the data itself.

Ground truth for scoring the result:
    <out_dir_sim>/sim_slices-<N>_seed-<s>_truth_iter-0000.h5
(fslices layout, readable with borndiagnostics._read_slices).

    python run_firstborn_sim.py [niter]
"""

import os
import sys
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import threepietorch, firstborntorch, fastfirstborntorch     # registers FirstBornTorch

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
batch_size       = 1 # born_mode 'lowmem' drops the N-fold workspace, so
                       # the batch grows ~N-fold; 'parallel' OOMs a 16 GB
                       # GPU already at slices*batch ~ 1600

# must match run_multislice_sim.py
SIM_SLICES       = 10
SIM_THICKNESS    = (SIM_SLICES - 1) * 1580.e-6  # total extent the simulator used
LAYER_SEED       = 0

probe_support    = 0.3

if len(sys.argv) >= 2:
    Niter = int(sys.argv[1])

out_dir     = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_0.00158/'
out_dir_sim = out_dir + 'sim/'
out_dir_rec = out_dir_sim + 'rec/'

# sim files are timestamped (sim_<stamp>_slices-N_seed-s.ptyd); take the
# newest one matching this slice count and seed, or an explicit tag on argv
import glob
if len(sys.argv) >= 3:
    _stem = f'{sys.argv[2]}_slices-{SIM_SLICES}_seed-{LAYER_SEED}'
    path_sim = f'{out_dir_sim}{_stem}/{_stem}.ptyd'
else:
    cands = sorted(
        glob.glob(f'{out_dir_sim}sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}/'
                  f'sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}.ptyd'),
        key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError('no sim_*_slices-%d_seed-%d/ in %s -- '
                                'run run_multislice_sim.py first'
                                % (SIM_SLICES, LAYER_SEED, out_dir_sim))
    path_sim = cands[-1]
sim_stem    = os.path.basename(path_sim)[:-5]     # e.g. sim_<stamp>_slices-50_seed-0
# reconstructions live inside the sim's own folder
out_dir_rec = os.path.join(os.path.dirname(path_sim), 'rec/')
path_rec    = f'{out_dir_rec}rec_{sim_stem}_%(engine)s_%(iterations)04d.ptyr'

# the slice separation the engine gets: layers equally spaced over the extent
slice_thickness = SIM_THICKNESS / (SIM_SLICES - 1)

comm = MPI.COMM_WORLD
from datetime import datetime as _dt
RUN_TAG = comm.bcast('run_' + _dt.now().strftime('%Y%m%d_%H%M%S')
                     + '_firstborn%d' % Niter, root=0)
out_dir_rec = os.path.join(out_dir_rec, RUN_TAG) + '/'
if comm.Get_rank() == 0:
    if not os.path.isfile(path_sim):
        raise FileNotFoundError('%s not found -- run run_multislice_sim.py '
                                'first' % path_sim)
    os.makedirs(out_dir_rec, exist_ok=True)
comm.barrier()
# engines drop iteration_progress_* in the cwd; run beside the outputs
_launch_dir = os.getcwd()
os.chdir(out_dir_rec)

# tee the terminal into a log that is moved inside this run's
# iteration_progress_* directory when the run finishes
import time as _time
_run_t0 = _time.time()


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
    # archive the exact script this run executed, beside its outputs
    import shutil
    _src = (__file__ if os.path.isabs(__file__)
            else os.path.join(_launch_dir, __file__))
    shutil.copy2(_src, os.path.basename(_src))
    _logfh = open('terminal_%d.log' % os.getpid(), 'w')
    sys.stdout = _Tee(sys.stdout, _logfh)
    sys.stderr = _Tee(sys.stderr, _logfh)

    # relocate the live log into this run's iteration_progress_* directory
    # as soon as the engine creates it; the rename keeps the open handle
    # valid, so the log keeps growing in its final place during the run
    import threading

    def _relocate_log():
        for _ in range(600):
            _time.sleep(2)
            pd = [d for d in os.listdir('.')
                  if d.startswith('iteration_progress_')
                  and os.path.getmtime(d) >= _run_t0 - 1]
            if pd:
                dst = sorted(pd, key=os.path.getmtime)[0]
                try:
                    os.rename(_logfh.name, os.path.join(dst, 'terminal.log'))
                except OSError:
                    pass
                return

    threading.Thread(target=_relocate_log, daemon=True).start()
    # ptypy's logging handlers captured the original streams at import;
    # rebind them so logger output lands in the tee as well
    import logging
    for _lg in (logging.getLogger(), logging.getLogger('ptypy')):
        for _h in _lg.handlers:
            if isinstance(_h, logging.StreamHandler):
                _h.stream = sys.stderr

p = u.Param()
p.verbose_level = 3
p.run = 'sim%d' % scannr

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

# The simulated .ptyd carries the prepared frames, weights, positions and
# meta of the real scan, so PtydScan rebuilds the identical geometry.
p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'PtydScan'
p.scans.scan00.data.dfile = path_sim
p.scans.scan00.data.source = None            # read dfile as-is

# same initial probe guess as the real-data runs
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
p.engines.engine00.name = 'FastFirstBornTorch'
p.engines.engine00.precision = 'single'
p.engines.engine00.object_denom = 'local'
p.engines.engine00.denom_reg = 0.05
p.engines.engine00.numiter = Niter
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = probe_support
# C
p.engines.engine00.probe_center_tol = 1
p.engines.engine00.batch_size = batch_size
# p.engines.engine00.born_mode = 'parallel'
# p.engines.engine00.momentum = 0.0
p.engines.engine00.seed = 7
p.engines.engine00.number_of_slices = SIM_SLICES
p.engines.engine00.slice_thickness = slice_thickness
p.engines.engine00.fslices = (f'{out_dir_rec}rec_{sim_stem}'
                              f'_iter-{Niter:0>4}.h5')

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)

if comm.Get_rank() == 0:
    sys.path.insert(0, _launch_dir)
    import sim_report
    sim_report.make_report(
        rec_h5=p.engines.engine00.fslices,
        truth_h5=path_sim[:-5] + '_truth_iter-0000.h5',
        data_ptyd=path_sim)
