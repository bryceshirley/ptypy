"""
Plot every reconstructed slice of an fslices .h5 in the exact style of the
simulator's sim_layers.png, so reconstruction and ground truth are compared
panel for panel.

    python plot_rec_slices.py <fslices.h5> [out.png]

Each slice is gauged by its complex mean (the same 'mean' gauge the
diagnostics engines use), so the empty background sits at phase 0 like the
simulated layers do.
"""

import re
import sys
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt


def read_slices(path):
    """Slices and axial positions from an fslices file, either naming
    scheme: the engines write objects/Cobj_o_<j>/<S>/data, the simulator's
    truth writes objects/obj_<j>/S00/data."""
    with h5py.File(path, 'r') as f:
        objs = f['content/objects']
        keys = sorted(objs.keys(),
                      key=lambda k: int(re.search(r'(\d+)$', k).group(1)))
        slices = []
        for k in keys:
            g = objs[k]
            sk = list(g.keys())[0]
            slices.append(np.asarray(g[sk]['data'])[0])
        if 'z' in f['content']:
            z = np.asarray(f['content/z'])
        else:
            dz = float(f['content/slice_thickness'][()])
            z = np.arange(len(slices)) * dz
    return np.stack(slices), z


def main():
    path = sys.argv[1]
    out = (sys.argv[2] if len(sys.argv) > 2
           else os.path.splitext(path)[0] + '_layers.png')

    layers, z = read_slices(path)
    nl = layers.shape[0]

    # mean gauge per slice: background at 1, like the simulated layers
    for j in range(nl):
        m = np.mean(layers[j])
        if np.abs(m) > 0:
            layers[j] = layers[j] / (m / np.abs(m))

    ncol = min(nl, 4)
    nrow = int(np.ceil(nl / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow),
                             squeeze=False)
    for j in range(nrow * ncol):
        ax = axes[j // ncol][j % ncol]
        if j >= nl:
            ax.axis('off')
            continue
        im = ax.imshow(np.angle(layers[j]), cmap='gray',
                       vmin=-1.5, vmax=0.3)
        ax.set_title('layer %d, z = %.0f um' % (j, z[j] * 1e6))
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label='phase [rad]')
    fig.suptitle('reconstructed slices (%d layers) -- %s'
                 % (nl, os.path.basename(path)))
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('written', out)


if __name__ == '__main__':
    main()
