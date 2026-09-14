"""
Standard results package for a simulated-data reconstruction, written into
the sim's rec/ directory so every run carries its own analysis:

    <tag>_layers.png         reconstructed slice gallery (phase)
    <tag>_truth_vs_rec.png   each simulated plane above its reconstruction
    <tag>_error_history.png  per-iteration error from the newest .ptyr

Importable (sim_report.make_report(...) at the end of every runner) or
callable:  python sim_report.py <rec_fslices.h5> [<truth.h5>]
"""

import glob
import os
import re

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def read_slices(path):
    """Slice stack from an fslices file, either naming scheme (engines write
    objects/Cobj_o_<j>/..., the truth dump writes objects/obj_<j>/S00)."""
    with h5py.File(path, 'r') as f:
        objs = f['content/objects']
        keys = sorted(objs.keys(),
                      key=lambda k: int(re.search(r'(\d+)$', k).group(1)))
        out = []
        for k in keys:
            g = objs[k]
            sk = sorted(g.keys())[0]
            out.append(np.asarray(g[sk]['data'])[0])
    return np.stack(out)


def gauge(s):
    m = np.mean(s)
    return s / (m / abs(m)) if abs(m) > 0 else s


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.ravel() @ b.ravel() / d) if d > 0 else 0.0


def _gallery(stacks, titles, path, suptitle):
    """One row per stack, one column per layer. Each row gets its own phase
    scale and colorbar: truth phases are often 5-10x deeper than the
    reconstruction's, and a common scale washes the weaker row out."""
    rows = len(stacks)
    N = max(s.shape[0] for s in stacks)  # stacks may differ in slice count
    ph = [np.angle(s) for s in stacks]
    fig, ax = plt.subplots(rows, N, figsize=(2.2 * N + 1.2, 2.4 * rows),
                           squeeze=False)
    for r in range(rows):
        lo, hi = np.percentile(ph[r], 0.5), np.percentile(ph[r], 99.5)
        for j in range(N):
            if j >= ph[r].shape[0]:
                ax[r][j].axis('off')   # this stack has fewer slices
                continue
            im = ax[r][j].imshow(ph[r][j], cmap='gray', vmin=lo, vmax=hi)
            ax[r][j].set_xticks([]), ax[r][j].set_yticks([])
            if r == 0:
                ax[r][j].set_title('layer %d' % j, fontsize=9)
        ax[r][0].set_ylabel(titles[r], fontsize=10)
        fig.colorbar(im, ax=list(ax[r]), fraction=0.008,
                     label='phase [rad]')
    fig.suptitle(suptitle)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_report(rec_h5, truth_h5=None, ptyr=None, outdir=None,
                data_ptyd=None):
    outdir = outdir or os.path.dirname(os.path.abspath(rec_h5))
    tag = os.path.basename(rec_h5)[:-3]
    rec = np.stack([gauge(s) for s in read_slices(rec_h5)])
    N = rec.shape[0]
    written = []

    _gallery([rec], ['reconstruction'],
             os.path.join(outdir, tag + '_layers.png'), tag)
    written.append(tag + '_layers.png')

    if truth_h5 and os.path.isfile(truth_h5):
        from scipy import ndimage
        truth = np.stack([gauge(s) for s in read_slices(truth_h5)])
        _gallery([truth, rec], ['simulated', 'reconstructed'],
                 os.path.join(outdir, tag + '_truth_vs_rec.png'),
                 tag + ' -- simulated plane vs reconstruction')
        written.append(tag + '_truth_vs_rec.png')

        Nt, Nr = truth.shape[0], rec.shape[0]  # may differ (coarse-axial rec)
        masks = [ndimage.binary_dilation(np.abs(truth[j] - 1.0) > 0.1,
                                         iterations=3) for j in range(Nt)]
        C = np.zeros((Nr, Nt))
        for j in range(Nt):
            tp = np.angle(truth[j])[masks[j]]
            for i in range(Nr):
                C[i, j] = corr(np.angle(rec[i])[masks[j]], tp)
        u = np.any(masks, axis=0)
        proj = corr(np.sum(np.angle(rec), axis=0)[u],
                    np.sum(np.angle(truth), axis=0)[u])
        diag = float(np.diag(C).mean()) if Nr == Nt else float('nan')

        # depth-correlation heatmap: rows = reconstructed slices, cols = truth
        # layers. A tight bright diagonal is clean axial separation; bright
        # off-diagonal bands are depth crosstalk (features smeared across
        # slices). When Nr != Nt the diagonal is only a guide.
        figc, axc = plt.subplots(figsize=(0.28 * Nt + 2.5, 0.28 * Nr + 2.0))
        imc = axc.imshow(C, cmap='inferno', vmin=-0.1, vmax=1.0,
                         aspect='auto', origin='upper')
        axc.set_xlabel('truth layer'), axc.set_ylabel('reconstructed slice')
        axc.set_title('%s\ndepth correlation (diag mean %.3f, proj %.3f)'
                      % (tag, diag, proj), fontsize=8)
        figc.colorbar(imc, ax=axc, fraction=0.046, pad=0.04, label='phase corr')
        figc.tight_layout()
        figc.savefig(os.path.join(outdir, tag + '_depth_corr.png'), dpi=200)
        plt.close(figc)
        written.append(tag + '_depth_corr.png')

        print('report: diag mean %.3f | proj corr %.3f' % (diag, proj))

    if ptyr is None:
        cands = sorted(glob.glob(os.path.join(outdir, '*.ptyr')),
                       key=os.path.getmtime)
        ptyr = cands[-1] if cands else None
    if ptyr and os.path.isfile(ptyr):
        with h5py.File(ptyr, 'r') as f:
            g = f['content/runtime/iter_info']
            err = np.stack([np.asarray(g[k]['error'])
                            for k in sorted(g.keys())])
        it = np.arange(1, err.shape[0] + 1)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.semilogy(it, np.abs(err[:, 0]) + 1e-30, lw=1.8)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Fourier error (engine units)')
        ax.set_title('%s -- error %.3g after %d iterations'
                     % (tag, err[-1, 0], len(it)))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, tag + '_error_history.png'), dpi=300)
        plt.close(fig)
        written.append(tag + '_error_history.png')

    # measured vs predicted diffraction frames (multislice and Born),
    # through born_error's standalone forward machinery
    if (data_ptyd and os.path.isfile(data_ptyd)
            and ptyr and os.path.isfile(ptyr)):
        try:
            with h5py.File(rec_h5, 'r') as f:
                nsl = int(np.array(f['content/number_of_slices']))
                th = float(np.ravel(np.array(f['content/slice_thickness']))[0])
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import born_error
            born_error.main([
                '--slices', rec_h5, '--ptyr', ptyr, '--ptyd', data_ptyd,
                '--thickness', str(th), '--n-slices', str(nsl),
                '--compare-frames', '3', '--max-order', '1', '--blocks', '1',
                '--n-positions', '32', '--batch-size', '8',
                '--out', os.path.join(outdir, tag + '_born')])
            if os.path.isfile(os.path.join(outdir, tag + '_born_frames.png')):
                written.append(tag + '_born_frames.png')
        except SystemExit:
            pass
        except Exception as e:
            print('frames comparison skipped: %s' % e)

    print('report written to %s: %s' % (outdir, ', '.join(written)))
    return written


if __name__ == '__main__':
    import sys
    make_report(sys.argv[1],
                truth_h5=sys.argv[2] if len(sys.argv) > 2 else None,
                ptyr=sys.argv[3] if len(sys.argv) > 3 else None,
                data_ptyd=sys.argv[4] if len(sys.argv) > 4 else None)
