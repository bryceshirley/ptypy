"""
FastFirstBornTorch on the 10-slice sim data, against the same .ptyd the
other engines were tested on.

    python run_fastborn_test.py <label> <niter> <batch_size> [denom]

denom defaults to 'local' -- with an exact line search the global scale of
the direction is absorbed by the step, so the per-pixel 'local' shape is the
natural preconditioner. Pass 'max' to mirror the FirstBornTorch baseline.
"""

import os
import sys
import glob
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import fastfirstborntorch, firstborntorch, threepietorch

label      = sys.argv[1]
Niter      = int(sys.argv[2])
batch_size = int(sys.argv[3])
denom      = sys.argv[4] if len(sys.argv) >= 5 else 'local'
ls_damp    = float(sys.argv[5]) if len(sys.argv) >= 6 else 1.0
pint       = int(sys.argv[6]) if len(sys.argv) >= 7 else 1
# reconstruct with a coarser axial sampling than the sim: fewer, thicker
# slices spanning the SAME total thickness (each rec slice > DOF so they can
# separate). Defaults to SIM_SLICES (one rec slice per sim layer).
rec_slices = int(sys.argv[7]) if len(sys.argv) >= 8 else None
# bandwidth-limit the probe to this fraction of Nyquist (physical aperture
# constraint); 0 disables. Cures the high-NA central-hot-pixel divergence.
pr_ffrac   = float(sys.argv[8]) if len(sys.argv) >= 9 else 0.0
# cap on the exact line-search step magnitude (0 disables); safety rail
max_step   = float(sys.argv[9]) if len(sys.argv) >= 10 else 0.0
# iteration at which probe updates begin; set > niter to freeze the probe
pr_start   = int(sys.argv[10]) if len(sys.argv) >= 11 else 0
# probe update denominator, decoupled from the object's (arg 4). Defaults to
# the engine default ('max') when not given.
pr_denom   = sys.argv[11] if len(sys.argv) >= 12 else None

detector         = 'eiger4m'
beamtime_basedir = '/home/dnz75396/ptypy/data/'
sample           = '0002_multislice'

scannr           = 434
distance_m       = 4.18
defocus_um       = -700
cropping         = 512
probe_modes      = 2

SIM_SLICES       = 30
SIM_THICKNESS    = (SIM_SLICES - 1) * 50.e-6    # thinner sample: 50 um dz
LAYER_SEED       = 3
REC_SLICES       = rec_slices if rec_slices is not None else SIM_SLICES

out_dir     = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_0.00158/'
out_dir_sim = out_dir + 'sim/'
out_dir_rec = out_dir_sim + 'rec/'

cands = sorted(
    glob.glob(f'{out_dir_sim}sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}.ptyd')
    + glob.glob(f'{out_dir_sim}sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}/'
                f'sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}.ptyd'),
    key=os.path.getmtime)
if not cands:
    raise FileNotFoundError('no sim data; run run_multislice_sim.py first')
path_sim = cands[-1]
sim_stem = os.path.basename(path_sim)[:-5]
# reconstructions live beside the sim they belong to: sim/<tag>/rec/ for the
# per-directory layout, the shared sim/rec/ for the old flat files
out_dir_rec = os.path.join(os.path.dirname(path_sim), 'rec/')

comm = MPI.COMM_WORLD
# every reconstruction run gets its own timestamped subdirectory under the
# sim's rec/, holding its ptyr, fslices h5, report pngs, progress dir and log
from datetime import datetime as _dt
RUN_TAG = comm.bcast('run_' + _dt.now().strftime('%Y%m%d_%H%M%S')
                     + '_' + label, root=0)
out_dir_rec = os.path.join(out_dir_rec, RUN_TAG) + '/'
if comm.Get_rank() == 0:
    os.makedirs(out_dir_rec, exist_ok=True)
comm.barrier()
# engines drop their iteration_progress_* directories in the cwd; run from
# the rec dir so every run's progress plots live beside its outputs
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
p.run = 'fast%s' % label

p.io = u.Param()
p.io.home = out_dir_rec
p.io.rfile = (f'{out_dir_rec}rec_{sim_stem}_fasttest-{label}'
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
p.scans.scan00.illumination.aperture.size = 300e-9   # matches the sim's 2x-NA
# probe: base 100 nm aperture (~1148 nm spot) halved by probe_na_scale=2.0
# (~574 nm) is reproduced here by a ~300 nm aperture at -700 um defocus
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = 1. * defocus_um * 1e-6
p.scans.scan00.illumination.diversity = u.Param()
p.scans.scan00.illumination.diversity.power = (1)

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'FastFirstBornTorch'
p.engines.engine00.precision = 'single'
p.engines.engine00.object_denom = denom
p.engines.engine00.numiter = Niter
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.batch_size = batch_size
p.engines.engine00.number_of_slices = REC_SLICES
p.engines.engine00.slice_thickness = SIM_THICKNESS / (REC_SLICES - 1)
p.engines.engine00.ls_damp = ls_damp
p.engines.engine00.probe_update_interval = pint
p.engines.engine00.probe_fourier_frac = pr_ffrac
p.engines.engine00.max_step = max_step
p.engines.engine00.probe_update_start = pr_start
p.engines.engine00.fslices = (f'{out_dir_rec}rec_{sim_stem}'
                              f'_fasttest-{label}_iter-{Niter:0>4}.h5')

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

P = Ptycho(p, level=5)

# standard results package into the rec dir: slice gallery, simulated plane
# vs reconstruction, depth-correlation matrix, error history
if comm.Get_rank() == 0:
    import sys
    sys.path.insert(0, _launch_dir)
    import sim_report
    sim_report.make_report(
        rec_h5=p.engines.engine00.fslices,
        truth_h5=path_sim[:-5] + '_truth_iter-0000.h5',
        data_ptyd=path_sim)
