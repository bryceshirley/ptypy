"""Phase-depth comparison: truth vs chord vs tangent reconstructions.

For each truth particle: the phase at the particle's deepest pixel and the
mean phase over its footprint, in truth and in both reconstructions, plus
the modulus range (the tangent's spurious amplitude). One figure: phase
line profiles through the largest particle of each layer.
"""
import sys, glob, os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sim_dir = sys.argv[1].rstrip('/') + '/'
truth_h5 = glob.glob(sim_dir + '*_truth_iter-0000.h5')[0]

def read_slices(path, key_guess=('S00', 'Sscan00G00')):
    out = []
    with h5py.File(path, 'r') as f:
        objs = f['content/objects']
        for k in sorted(objs.keys(), key=lambda s: int(s.split('_')[-1])):
            g = objs[k]
            kk = [x for x in key_guess if x in g][0]
            out.append(g[kk]['data'][()][0])
    return np.stack(out)

truth = read_slices(truth_h5)
recs = {}
for kind in ('chord', 'tangent'):
    runs = sorted(glob.glob(sim_dir + f'rec/run_*{kind[:3] if kind=="tangent" else kind}*/rec_*_iter-*.h5'),
                  key=os.path.getmtime)
    runs = [r for r in runs if f'_{kind}-' in os.path.basename(r)]
    if runs:
        recs[kind] = read_slices(runs[-1])
        print(f'{kind}: {os.path.basename(runs[-1])}')

print('\nper-layer particle phase (rad) and modulus:')
print('%-6s %-22s %-22s %-22s' % ('layer', 'truth', 'chord', 'tangent'))
prof = {}
for L in range(truth.shape[0]):
    ph_t = np.angle(truth[L])
    foot = np.abs(truth[L] - 1.0) > 0.3
    if not foot.any():
        continue
    iy, ix = np.unravel_index(np.argmax(np.abs(ph_t) * foot), ph_t.shape)
    def stats(arr):
        try:
            from skimage.restoration import unwrap_phase
            ph = np.asarray(unwrap_phase(np.angle(arr[L])))
            # reference the unwrapped surface to zero background
            ph = ph - np.median(ph[~foot])
        except ImportError:
            ph = np.angle(arr[L])
        m = np.abs(arr[L])
        return ('min %+0.2f mean %+0.2f |O| %0.2f-%0.2f'
                % (ph[foot].min(), ph[foot].mean(), m[foot].min(), m[foot].max()))
    row = ['%-6d' % L, stats(truth)]
    for kind in ('chord', 'tangent'):
        row.append(stats(recs[kind]) if kind in recs else 'n/a')
    print('  '.join(row))
    prof[L] = (iy, ix)

fig, axes = plt.subplots(2, len(prof), figsize=(5.2 * len(prof), 7), squeeze=False)
for c, (L, (iy, ix)) in enumerate(sorted(prof.items())):
    sl = slice(max(ix - 90, 0), ix + 90)
    ax = axes[0][c]
    ax.plot(np.unwrap(np.angle(truth[L])[iy, sl]), 'k-', lw=2, label='truth')
    for kind, st in (('chord', 'C0-'), ('tangent', 'C3--')):
        if kind in recs:
            ax.plot(np.unwrap(np.angle(recs[kind][L])[iy, sl]), st, label=kind)
    ax.set_title('layer %d: phase through particle (unwrapped)' % L)
    ax.set_ylabel('phase [rad]'); ax.legend()
    ax = axes[1][c]
    ax.plot(np.abs(truth[L])[iy, sl], 'k-', lw=2, label='truth')
    for kind, st in (('chord', 'C0-'), ('tangent', 'C3--')):
        if kind in recs:
            ax.plot(np.abs(recs[kind][L])[iy, sl], st, label=kind)
    ax.set_title('layer %d: modulus' % L)
    ax.set_ylabel('|O|'); ax.set_xlabel('pixel'); ax.legend()
fig.suptitle('sparse strong 3-slice sample: chord vs tangent readout, 3 iterations')
fig.tight_layout()
out = sim_dir + 'chord_vs_tangent_phase_profiles.png'
fig.savefig(out, dpi=150)
print('\nfigure:', out)
