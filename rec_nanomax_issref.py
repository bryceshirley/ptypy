"""
Reference ISSTorch reconstruction of NanoMAX scan 434 for benchmarking the
cupy ePIE_ISB_cupy engine against it. Same prepared .ptyd, same illumination.
plot_interval=0 (this branch has no torch_helpers); the probe/object are read
from the saved fslices/ptyr and plotted separately.

    python rec_nanomax_issref.py [numiter] [batch_size]
"""
import os, sys, time
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

import ptypy.custom.torch.iss_torch   # registers ISSTorch

PTYD = ('/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/'
        'scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/'
        'data/data_scan_000434.ptyd')

numiter = int(sys.argv[1]) if len(sys.argv) > 1 else 20
batch   = int(sys.argv[2]) if len(sys.argv) > 2 else 512

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'runs', 'nanomax434_ISSTorch_it%d_b%d' % (numiter, batch))
os.makedirs(out, exist_ok=True)
import shutil as _shutil
_shutil.copy(os.path.abspath(__file__),
             os.path.join(out, os.path.basename(__file__)))

p = u.Param()
p.verbose_level = 3
p.run = 'nmx434_issref'
p.frames_per_block = 100000  # torch engine wants all frames in one block
p.io = u.Param()
p.io.home = out
p.io.rfile = os.path.join(out, 'rec_%(engine)s_%(iterations)04d.ptyr')
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)

p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'Full'
p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = 2
p.scans.scan00.coherence.num_object_modes = 1
p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'PtydScan'
p.scans.scan00.data.dfile = PTYD
p.scans.scan00.illumination = u.Param()
p.scans.scan00.illumination.model = None
p.scans.scan00.illumination.aperture = u.Param()
p.scans.scan00.illumination.aperture.form = 'circ'
p.scans.scan00.illumination.aperture.size = 100e-9
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = -700e-6
p.scans.scan00.illumination.diversity = u.Param()
p.scans.scan00.illumination.diversity.power = 1
p.scans.scan00.illumination.diversity.noise = (0.5, 1.0)

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'ISSTorch'
p.engines.engine00.numiter = numiter
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = 0.3
p.engines.engine00.number_of_slices = 2
p.engines.engine00.slice_thickness = 1580.e-6
p.engines.engine00.alpha = 1.0
p.engines.engine00.beta = 1.0
p.engines.engine00.object_denom = 'local'
p.engines.engine00.probe_update_interval = 1
p.engines.engine00.batch_size = batch
p.engines.engine00.shuffle = False
p.engines.engine00.plot_interval = 0
p.engines.engine00.seed = 0
p.engines.engine00.device = 'cuda'
p.engines.engine00.fslices = os.path.join(out, 'slices_issref.h5')

t0 = time.time()
P = Ptycho(p, level=5)
print('== ISSTorch done: %d iters in %.1f s' % (numiter, time.time() - t0))
