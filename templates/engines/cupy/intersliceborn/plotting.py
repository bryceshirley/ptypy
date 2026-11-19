#!/usr/bin/env python
"""Slice figures for the inter-slice Born demos, in the style of the
published NanoMAX figure.

Ports data/nanomax/notebooks/Figure_reconstruction_result.ipynb: crop
512 off every border, unwrap the phase, subtract the offset measured in
a vacuum window, Greys_r on a fixed phase range, micrometre extent, a
red zoom box on the full views and scale bars on both rows. Keeping the
notebook's constants means a demo's output can be laid next to the
published figure and compared directly.

Used as a library by the demos, and runnable on a finished run
directory to render every reconstruction in it:

    python plotting.py results/demo_01 [more_dirs ...]
"""

import argparse
import glob
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase

HERE = os.path.dirname(os.path.abspath(__file__))

# notebook settings
CROP = 512
VMIN, VMAX = -1.2, 0.1
ZOOM_X, ZOOM_Y = (3.5, 5.5), (5.5, 7.5)
# vacuum windows the notebook averages to zero each slice's background
OFFSET_WIN = ((slice(124, 127), slice(640, 680)),
              (slice(100, 250), slice(350, 500)))


def load_slices(path):
    """Cropped, unwrapped, offset-removed phase of every object slice in
    a ptypy slice file, plus the object pixel size.

    Slices beyond the notebook's two vacuum windows fall back to the
    median, which is the same convention the comparison demo uses.
    """
    with h5py.File(path) as f:
        keys = sorted(f["content/objects"])
        obj = [f[f"content/objects/{k}/Sscan00G00/data"][0] for k in keys]
        psize = f[f"content/objects/{keys[0]}/Sscan00G00/_psize"][...]
    d = CROP // 2
    ph = []
    for j, o in enumerate(obj):
        p = unwrap_phase(np.angle(o[d:-d, d:-d]))
        offset = (np.mean(p[OFFSET_WIN[j]]) if j < len(OFFSET_WIN)
                  else np.median(p))
        ph.append(p - offset)
    return ph, psize


def _slice_name(j, n):
    if n == 2:
        return ("upstream object slice", "downstream object slice")[j]
    return f"object slice {j} (of {n})"


def slice_figure(ph, psize, title, png):
    """Two rows, full view and zoom, by one column per object slice."""
    n = len(ph)
    Ny, Nx = ph[0].shape
    extent = [0, Nx * psize[1] * 1e6, 0, Ny * psize[0] * 1e6]
    rectx = [ZOOM_X[0], ZOOM_X[0], ZOOM_X[1], ZOOM_X[1], ZOOM_X[0]]
    recty = [ZOOM_Y[0], ZOOM_Y[1], ZOOM_Y[1], ZOOM_Y[0], ZOOM_Y[0]]

    fig = plt.figure(figsize=(3.8 * n, 7))
    plt.suptitle(title)
    for col in range(n):
        for row in (0, 1):                  # row 0 full view, row 1 zoom
            ax = plt.subplot(2, n, row * n + col + 1)
            ax.set_title(_slice_name(col, n) if row == 0 else "zoom in")
            im = ax.imshow(ph[col], interpolation="None", cmap="Greys_r",
                           extent=extent, vmin=VMIN, vmax=VMAX)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == n - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="phase shift in [rad]")
            if row == 0:
                ax.plot(rectx, recty, lw=1, c="r")
                ax.text(10.3, 0.8, "2 μm", ha="center", c="r",
                        fontweight="semibold", fontsize=12)
                ax.plot([9.3, 11.3], [0.5, 0.5], lw=3, c="r")
            else:
                ax.text(5.15, 5.65, "500 nm", ha="center", c="r",
                        fontweight="semibold", fontsize=12)
                ax.plot([4.90, 5.40], [5.6, 5.6], lw=3, c="r")
                ax.set_xlim(ZOOM_X)
                ax.set_ylim(ZOOM_Y)
    plt.tight_layout()
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("wrote", png)


def plot_run_dir(out, pattern="slices_*.h5"):
    """Render every reconstruction in a run directory. Returns the PNGs
    written."""
    written = []
    for path in sorted(glob.glob(os.path.join(out, pattern))):
        tag = os.path.splitext(os.path.basename(path))[0]
        try:
            ph, psize = load_slices(path)
        except (OSError, KeyError) as e:
            print(f"unreadable, skipped: {path} ({e})")
            continue
        for j, p in enumerate(ph):
            print(f"{tag} slice {j}: phase min {p.min():+.3f}  "
                  f"0.5th pct {np.percentile(p, 0.5):+.3f} rad")
        png = os.path.join(out, f"{tag}.png")
        slice_figure(ph, psize, f"{tag}, scan #434", png)
        written.append(png)
    if not written:
        print("no readable", pattern, "in", out)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="run directories to render")
    ap.add_argument("--pattern", default="slices_*.h5")
    args = ap.parse_args()
    for d in args.dirs:
        d = d if os.path.isabs(d) else os.path.join(HERE, d)
        print(f"== {d}")
        plot_run_dir(d, args.pattern)


if __name__ == "__main__":
    main()
