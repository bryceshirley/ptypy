"""
Composite result figure: chord vs tangent line-search reconstructions.

    python plot_chord_vs_tangent_final.py <sim_dir> <chord_label> <tan_label> \
        <chord_log> <tan_log>

Top: phase of the three layers, rows = ground truth / first Born (chord) /
first Born (tangent), one shared grayscale (sequential: lightness carries
the phase magnitude). Bottom: both engines' Fourier-error histories on one
log axis (categorical: blue = chord, orange = tangent, the validated
reference-palette pair).
"""
import sys
import re
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# reference-palette slots 1 and 2 (documented as passing the adjacent-pair
# checks in light mode); text wears text tokens, never the series color
C_CHORD, C_TAN = '#2a78d6', '#eb6834'
INK, INK2 = '#0b0b0b', '#52514e'
SURFACE = '#fcfcfb'

sim_dir = sys.argv[1].rstrip('/') + '/'
lab_c, lab_t = sys.argv[2], sys.argv[3]
log_c, log_t = sys.argv[4], sys.argv[5]


def read_slices(path):
    with h5py.File(path, 'r') as f:
        objs = f['content/objects']
        keys = sorted(objs.keys(), key=lambda s: int(s.split('_')[-1]))
        return np.stack([objs[k][list(objs[k].keys())[0]]['data'][()][0]
                         for k in keys])


def read_errors(logpath):
    """Per-iteration detector (Fourier) error from the run log."""
    it, err = [], []
    for line in open(logpath):
        m = re.search(r'Errors :: Fourier ([0-9.e+-]+),', line)
        if m:
            err.append(float(m.group(1)))
            it.append(len(err))
    return np.array(it), np.array(err)


truth = read_slices(glob.glob(sim_dir + '*_truth_iter-0000.h5')[0])
rec_c = read_slices(sorted(glob.glob(
    sim_dir + f'rec/run_*_{lab_c}/rec_*_iter-*.h5'))[-1])
rec_t = read_slices(sorted(glob.glob(
    sim_dir + f'rec/run_*_{lab_t}/rec_*_iter-*.h5'))[-1])
it_c, e_c = read_errors(log_c)
it_t, e_t = read_errors(log_t)

# display window: bounding box of the structure, padded
devmax = np.abs(truth - 1.0).max(axis=0) > 1e-6
ys, xs = np.nonzero(devmax)
pad = 60
win = (slice(max(ys.min() - pad, 0), ys.max() + pad),
       slice(max(xs.min() - pad, 0), xs.max() + pad))

rows = [('ground truth', truth), ('first Born, chord $g=O-1$', rec_c),
        ('first Born, tangent $g=is$, $s$ real', rec_t)]


def unwrapped(arr, L):
    """2D-unwrapped phase, referenced to zero on the empty background."""
    from skimage.restoration import unwrap_phase
    u = np.asarray(unwrap_phase(np.angle(arr[L][win])))
    bg = np.abs(truth[L][win] - 1.0) < 1e-6
    if bg.any():
        u -= np.median(u[bg])
    return u


ph = [[unwrapped(arr, L) for L in range(3)] for _, arr in rows]
# scale limits from the TANGENT row's percentiles, so its low-contrast
# artifacts are visible; the deep truth/chord particles saturate (their
# minima are direct-labeled per panel)
tan_all = np.concatenate([p.ravel() for p in ph[2]])
vmin = float(np.percentile(tan_all, 0.2))
vmax = float(np.percentile(tan_all, 99.8))

fig = plt.figure(figsize=(10.5, 13.2), facecolor=SURFACE)
gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.85],
              hspace=0.12, wspace=0.06,
              left=0.09, right=0.9, top=0.94, bottom=0.06)

for r, (name, _) in enumerate(rows):
    for c in range(3):
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(ph[r][c], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#d8d7d2')
        if r == 0:
            ax.set_title('layer %d' % c, fontsize=11, color=INK)
        if c == 0:
            ax.set_ylabel(name, fontsize=10, color=INK)
        # selective direct label: the deepest phase in this panel
        ax.text(0.03, 0.04, 'min %.2f rad' % ph[r][c].min(),
                transform=ax.transAxes, fontsize=8, color=INK2)
cax = fig.add_axes([0.915, 0.40, 0.018, 0.5])
cb = fig.colorbar(im, cax=cax)
cb.set_label('unwrapped phase [rad]', color=INK)
cb.ax.tick_params(colors=INK2, labelsize=8)
cb.outline.set_edgecolor('#d8d7d2')

axl = fig.add_subplot(gs[3, :])
axl.set_facecolor(SURFACE)
axl.semilogy(it_c, e_c, '-', color=C_CHORD, lw=2, label='chord, exact line search')
axl.semilogy(it_t, e_t, '-', color=C_TAN, lw=2,
             label='tangent ($s$ real), exact line search')
# selective direct labels at the endpoints, in ink
axl.annotate('%.2f' % e_c[-1], (it_c[-1], e_c[-1]), textcoords='offset points',
             xytext=(6, -3), fontsize=9, color=INK)
axl.annotate('%.2f' % e_t[-1], (it_t[-1], e_t[-1]), textcoords='offset points',
             xytext=(6, -3), fontsize=9, color=INK)
axl.set_xlabel('iteration', color=INK)
axl.set_ylabel('detector (Fourier) error', color=INK)
axl.set_xlim(0, max(it_c[-1], it_t[-1]) + 4)
axl.grid(alpha=0.25, which='both')
axl.tick_params(colors=INK2)
for s in ('top', 'right'):
    axl.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    axl.spines[s].set_color('#d8d7d2')
axl.legend(fontsize=10, frameon=False, labelcolor=INK,
           loc='center right')

fig.suptitle('Sparse strong pure-phase specimen (3 layers): '
             'chord vs tangent first Born', fontsize=13, color=INK)
out = sim_dir + 'chord_vs_tangent_final.png'
fig.savefig(out, dpi=170, facecolor=SURFACE)
print('figure:', out)
print('final errors: chord %.3g, tangent %.3g (ratio %.1fx)'
      % (e_c[-1], e_t[-1], e_t[-1] / e_c[-1]))
