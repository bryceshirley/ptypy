"""
Detector-plane forward-model errors through the sparse strong truth sample.

    python run_borndiag_sparse.py [sim_dir]

Points ISSDiagnostics at a SparseStrongSimulator sim: the truth layer
stack is the "reconstruction under test", the truth probe is injected as the
live probe, and the recorded frames are the sim's own exact-multislice
intensities (noise off). The engine then measures, per scan position and
aggregated, how well each forward model predicts those frames:

    eps_multislice          exact multislice      -> round-off (sanity)
    eps_order[1]            chord ISS      g = O - 1
    eps_order_tangent[1]    tangent ISS    g = i s = log O
    ... higher orders and both hierarchies' floors

On a beam-cone-sparse specimen the chord eps_1 should fall to the model's
exactness (the neglected cross terms have no support), while the tangent
eps_1 keeps its O(s^2) per-slice linearisation error despite the sparsity.

sim_dir defaults to the newest sim_*_slices-3_seed-7 under the scan's sim/.
Runs on CPU.
"""

import os
import sys
import glob
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy.custom import issdiagnostics   # registers ISSDiagnostics

detector         = 'eiger4m'
beamtime_basedir = '/home/dnz75396/ptypy/data/'
sample           = '0002_multislice'
scannr           = 434
distance_m       = 4.18
cropping         = 512
probe_modes      = 2

SIM_SLICES       = 3
LAYER_SEED       = 7

out_dir     = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_slice_0.00158/'
out_dir_sim = out_dir + 'sim/'

if len(sys.argv) >= 2:
    sim_dir = sys.argv[1].rstrip('/') + '/'
else:
    dirs = sorted(glob.glob(f'{out_dir_sim}sim_*_slices-{SIM_SLICES}'
                            f'_seed-{LAYER_SEED}/'), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError('no sparse-strong sim dirs found')
    sim_dir = dirs[-1]

ptyds = glob.glob(sim_dir + 'sim_*.ptyd')
if not ptyds:
    raise FileNotFoundError('no .ptyd in %s' % sim_dir)
path_sim = ptyds[0]
truth_h5 = path_sim[:-5] + '_truth_iter-0000.h5'
if not os.path.isfile(truth_h5):
    raise FileNotFoundError(truth_h5)

# every sim already knows its slice spacing from its own results.json
import json
with open(sim_dir + 'results.json') as f:
    _res = json.load(f)
dz = float(_res['simulation']['separation_um']) * 1e-6

diag_dir = sim_dir + 'diag/'
comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(diag_dir, exist_ok=True)
comm.barrier()

print('sim   :', path_sim)
print('truth :', truth_h5)
print('dz    : %.1f um x %d slices' % (dz * 1e6, SIM_SLICES))

p = u.Param()
p.verbose_level = 3
p.run = 'diagsparse'

p.io = u.Param()
p.io.home = diag_dir
p.io.rfile = f'{diag_dir}diag_%(engine)s_%(iterations)04d.ptyr'
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

# only seeds P.probe; the truth probe replaces it below
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
p.engines.engine00.name = 'ISSDiagnostics'
p.engines.engine00.numiter = 1
p.engines.engine00.numiter_contiguous = 1

# the "reconstruction" under test: the truth stack + truth probe. The sim
# dir holds no .ptyr, so the probe pool falls back to the live P.probe,
# which is overwritten with the truth probe before the run.
p.engines.engine00.ground_truth = sim_dir
p.engines.engine00.slices = truth_h5
p.engines.engine00.iteration = None
# the probe pool is just the live P.probe (no .ptyr in a sim dir);
# 'same-iteration' pairing would try to read an iteration number off it
p.engines.engine00.pair_mode = 'best'
p.engines.engine00.number_of_slices = SIM_SLICES
p.engines.engine00.slice_thickness = dz
p.engines.engine00.max_order = SIM_SLICES

p.engines.engine00.outdir = diag_dir
p.engines.engine00.batch_size = 4
p.engines.engine00.precision = 'double'
p.engines.engine00.device = 'cuda'
# just the eps scan: the timing benchmark, depth sweep and preconditioner
# test are not part of this comparison and triple the CPU runtime
p.engines.engine00.gauge = 'mean'
p.engines.engine00.benchmark = False
p.engines.engine00.depth_sweep = False
p.engines.engine00.precond_test = False
# optional quick mode: argv2 = number of scan positions to subsample
# (fast figure regeneration; eps numbers are then subsampled — keep the
# full-scan results.json elsewhere)
if len(sys.argv) >= 3:
    p.engines.engine00.n_positions = int(sys.argv[2])
    diag_dir2 = sim_dir + 'diag_quick/'
    os.makedirs(diag_dir2, exist_ok=True)
    p.engines.engine00.outdir = diag_dir2
    p.io.home = diag_dir2

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

import h5py
import numpy as np
P = Ptycho(p, level=4)
with h5py.File(truth_h5, 'r') as f:
    Ptruth = f['content/probe/S00/data'][()]
prs = next(iter(P.probe.storages.values()))
if prs.data.shape != Ptruth.shape:
    raise ValueError('truth probe shape %s != storage %s'
                     % (Ptruth.shape, prs.data.shape))
prs.data[:] = Ptruth.astype(prs.data.dtype)
if comm.Get_rank() == 0:
    print('[truth-probe] injected %s probe from %s'
          % (str(Ptruth.shape), os.path.basename(truth_h5)))
P.run()
P.finalize()
