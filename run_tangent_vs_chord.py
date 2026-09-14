"""
Chord vs tangent Born reconstruction of the sparse strong 3-slice sim.

    python run_tangent_vs_chord.py <label> <chord|tangent> [niter] [batch]

chord   -> FirstBornTorch         g = O - 1   (exact slice transmission)
tangent -> FirstBornTangentTorch  g = i s     (linearised / weak-phase)

Everything else is identical between the two runs: same sim data, same
truth probe injected and FROZEN (object-only reconstruction, so 3
iterations isolate the object model), same seed, batch 1, CPU, a progress
plot EVERY iteration. The specimen is strongly scattering (phase ~4 rad)
but beam-cone sparse, so the chord model is exact while the tangent
model's O(s^2) per-slice error remains.
"""

import os
import sys
import glob
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import firstborntorch, fastfirstborntorch  # register engines

label = sys.argv[1] if len(sys.argv) >= 2 else 'test'
kind  = sys.argv[2] if len(sys.argv) >= 3 else 'chord'
Niter = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
batch_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
# optional: an explicit sim directory, instead of the newest matching one
sim_dir_arg = sys.argv[5] if len(sys.argv) >= 6 else None

engine_by_kind = {'chord': 'FirstBornTorch',
                  'tangent': 'FirstBornTangentTorch',
                  'fast': 'FastFirstBornTorch',
                  'fasttangent': 'FastFirstBornTangentTorch'}
if kind not in engine_by_kind:
    raise SystemExit("second argument must be 'chord', 'tangent', 'fast' "
                     "or 'fasttangent'")
engine = engine_by_kind[kind]

detector         = 'eiger4m'
beamtime_basedir = '/home/dnz75396/ptypy/data/'
sample           = '0002_multislice'

scannr           = 434
distance_m       = 4.18
cropping         = 512
probe_modes      = 2

SIM_SLICES       = 3
SIM_THICKNESS    = (SIM_SLICES - 1) * 1580.e-6
LAYER_SEED       = 7
REC_SLICES       = SIM_SLICES

out_dir     = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_0.00158/'
out_dir_sim = out_dir + 'sim/'

if sim_dir_arg:
    cands = glob.glob(os.path.join(sim_dir_arg, 'sim*.ptyd'))
else:
    cands = sorted(
        glob.glob(f'{out_dir_sim}sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}/'
                  f'sim*_slices-{SIM_SLICES}_seed-{LAYER_SEED}.ptyd'),
        key=os.path.getmtime)
if not cands:
    raise FileNotFoundError('no sparse-strong sim data; run '
                            'run_sparse_strong_sim.py first')
path_sim = cands[-1]
sim_stem = os.path.basename(path_sim)[:-5]
out_dir_rec = os.path.join(os.path.dirname(path_sim), 'rec/')

comm = MPI.COMM_WORLD
from datetime import datetime as _dt
RUN_TAG = comm.bcast('run_' + _dt.now().strftime('%Y%m%d_%H%M%S')
                     + '_' + label, root=0)
out_dir_rec = os.path.join(out_dir_rec, RUN_TAG) + '/'
if comm.Get_rank() == 0:
    os.makedirs(out_dir_rec, exist_ok=True)
comm.barrier()
_launch_dir = os.getcwd()
os.chdir(out_dir_rec)

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
    import shutil
    _src = (__file__ if os.path.isabs(__file__)
            else os.path.join(_launch_dir, __file__))
    shutil.copy2(_src, os.path.basename(_src))
    _logfh = open('terminal_%d.log' % os.getpid(), 'w')
    sys.stdout = _Tee(sys.stdout, _logfh)
    sys.stderr = _Tee(sys.stderr, _logfh)

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
    import logging
    for _lg in (logging.getLogger(), logging.getLogger('ptypy')):
        for _h in _lg.handlers:
            if isinstance(_h, logging.StreamHandler):
                _h.stream = sys.stderr

p = u.Param()
p.verbose_level = 3
p.run = 'tvc%s' % label

p.io = u.Param()
p.io.home = out_dir_rec
p.io.rfile = (f'{out_dir_rec}rec_{sim_stem}_{kind}-{label}'
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

# The illumination model only seeds P.probe; the truth probe replaces it
# below before the run, and stays frozen.
p.scans.scan00.illumination = u.Param()
p.scans.scan00.illumination.model = None
p.scans.scan00.illumination.aperture = u.Param()
p.scans.scan00.illumination.aperture.form = 'circ'
p.scans.scan00.illumination.aperture.size = 100e-9
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = -700e-6
p.scans.scan00.illumination.diversity = u.Param()
p.scans.scan00.illumination.diversity.power = (1)

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = engine
p.engines.engine00.precision = 'single'
p.engines.engine00.numiter = Niter
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.batch_size = batch_size
p.engines.engine00.number_of_slices = REC_SLICES
p.engines.engine00.slice_thickness = SIM_THICKNESS / (REC_SLICES - 1)
p.engines.engine00.object_denom = 'local'
p.engines.engine00.alpha = 1.0
p.engines.engine00.seed = 123
if kind in ('fast', 'fasttangent'):
    # exact-line-search engines, campaign best config (ls_damp load-bearing)
    p.engines.engine00.ls_damp = 0.5
    p.engines.engine00.denom_reg = 0.05
    # one exact step per slice: a single global step is dominated by the
    # best-covered (deepest) layer and under-converges the entrance layers
    p.engines.engine00.per_slice_ls = True
if kind in ('tangent', 'fasttangent'):
    # the honest weak-phase convention: s real (pure phase in the
    # potential). This is a genuinely different optimisation from the
    # chord -- the model's i puts the scattered wave in quadrature and the
    # real constraint stops the reparameterisation escape.
    p.engines.engine00.potential_real = True
p.engines.engine00.plot_interval = 1          # plot EVERY iteration
p.engines.engine00.device = 'cuda'
# truth probe, frozen for the whole run: object-only reconstruction
p.engines.engine00.probe_update_start = Niter + 1
p.engines.engine00.fslices = (f'{out_dir_rec}rec_{sim_stem}'
                              f'_{kind}-{label}_iter-{Niter:0>4}.h5')

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

# build up to engine creation, then overwrite the probe storage with the
# exact simulation probe from the truth h5, so the frozen probe IS truth
import h5py
P = Ptycho(p, level=4)
truth_h5 = path_sim[:-5] + '_truth_iter-0000.h5'
with h5py.File(truth_h5, 'r') as f:
    Ptruth = f['content/probe/S00/data'][()]
prs = next(iter(P.probe.storages.values()))
if prs.data.shape != Ptruth.shape:
    raise ValueError('truth probe shape %s != storage %s'
                     % (Ptruth.shape, prs.data.shape))
prs.data[:] = Ptruth.astype(prs.data.dtype)
if comm.Get_rank() == 0:
    print('[truth-probe] injected %s probe from %s; probe FROZEN; '
          'engine = %s (%s perturbation)'
          % (str(Ptruth.shape), os.path.basename(truth_h5), engine, kind))
P.run()
P.finalize()

# standard results package: slice gallery, truth-vs-rec, depth correlation
if comm.Get_rank() == 0:
    sys.path.insert(0, _launch_dir)
    import sim_report
    sim_report.make_report(
        rec_h5=p.engines.engine00.fslices,
        truth_h5=truth_h5,
        data_ptyd=path_sim)
