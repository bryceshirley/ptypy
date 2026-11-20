"""
Demo 01: the real NanoMAX scan, inter-slice first Born against multislice.

Reconstructs NanoMAX scan 434 (the prepared two-slice .ptyd of the paper
campaign) with three GPU engines on identical data, illumination and
seeds:

  * IntersliceBorn at order 1: the inter-slice first Born model, fixed
    step size;
  * IntersliceFirstBornLineSearch: the same model with the step size set
    by a line search on the amplitude-l2 fidelity each iteration, so it
    is given fewer iterations;
  * ThreePIE_cupy: upstream ptypy's sequential multislice engine.

Prints the wall time per iteration and the final error of each, and
writes one figure, demo_01_born_vs_multislice.png, putting the two
reconstructions side by side in the style of the published NanoMAX
figure.

    python demo_01_nanomax_engines.py [--numiter 20] [--ls-numiter 10]
"""

import os
import sys

# Run in place against the repo these templates ship with: walk up until
# the ptypy package is found, rather than counting directory levels, which
# breaks whenever the templates move. This directory goes first so the
# demos can import each other.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while (_REPO != os.path.dirname(_REPO)
       and not os.path.isdir(os.path.join(_REPO, "ptypy", "core"))):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import argparse
import time

import h5py
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase

from ptypy import utils as u
from ptypy.core import Ptycho

# Results live beside the template, not in whatever directory the demo
# was launched from, so a relative --out always resolves here.
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_out(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)

PTYD = ('/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/'
        'scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/'
        'data/data_scan_000434.ptyd')


# Figure settings taken from the published NanoMAX figure,
# data/nanomax/notebooks/Figure_reconstruction_result.ipynb: crop 512 off
# every border, unwrap, subtract the offset measured in a vacuum window,
# Greys_r on a fixed phase range, micrometre extent, a red zoom box on the
# full views and scale bars on both rows.
CROP = 512
VMIN, VMAX = -1.2, 0.1
ZOOM_X, ZOOM_Y = (3.5, 5.5), (5.5, 7.5)
OFFSET_WIN = ((slice(124, 127), slice(640, 680)),
              (slice(100, 250), slice(350, 500)))


def load_slices(path):
    """Cropped, unwrapped, offset-removed phase of every object slice in a
    ptypy slice file, plus the object pixel size. Slices beyond the two
    vacuum windows fall back to the median."""
    with h5py.File(path) as f:
        keys = sorted(f["content/objects"])
        obj = [f[f"content/objects/{k}/Sscan00G00/data"][0] for k in keys]
        psize = f[f"content/objects/{keys[0]}/Sscan00G00/_psize"][...]
    d = CROP // 2
    ph = []
    for j, o in enumerate(obj):
        q = unwrap_phase(np.angle(o[d:-d, d:-d]))
        offset = (np.mean(q[OFFSET_WIN[j]]) if j < len(OFFSET_WIN)
                  else np.median(q))
        ph.append(q - offset)
    return ph, psize


def _slice_name(j, n):
    if n == 2:
        return ("upstream object slice", "downstream object slice")[j]
    return f"object slice {j} (of {n})"


def draw_slices(fig, ph, psize, margins=None):
    """Two rows, full view and zoom, by one column per object slice, drawn
    into any figure or subfigure so several can sit side by side.

    margins overrides the horizontal subplot margins, which is what sets
    the gap when two of these sit next to each other. It has to be applied
    before the colour bars are made, because those are positioned against
    the axes as they stand."""
    n = len(ph)
    Ny, Nx = ph[0].shape
    extent = [0, Nx * psize[1] * 1e6, 0, Ny * psize[0] * 1e6]
    rectx = [ZOOM_X[0], ZOOM_X[0], ZOOM_X[1], ZOOM_X[1], ZOOM_X[0]]
    recty = [ZOOM_Y[0], ZOOM_Y[1], ZOOM_Y[1], ZOOM_Y[0], ZOOM_Y[0]]

    axes = fig.subplots(2, n, squeeze=False)
    if margins:
        fig.subplots_adjust(**margins)
    for col in range(n):
        for row in (0, 1):
            ax = axes[row][col]
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
                ax.text(10.3, 0.8, "2 \u03bcm", ha="center", c="r",
                        fontweight="semibold", fontsize=12)
                ax.plot([9.3, 11.3], [0.5, 0.5], lw=3, c="r")
            else:
                ax.text(5.15, 5.65, "500 nm", ha="center", c="r",
                        fontweight="semibold", fontsize=12)
                ax.plot([4.90, 5.40], [5.6, 5.6], lw=3, c="r")
                ax.set_xlim(ZOOM_X)
                ax.set_ylim(ZOOM_Y)


def slice_figure(ph, psize, title, png):
    """One reconstruction on its own."""
    fig = plt.figure(figsize=(3.8 * len(ph), 7))
    fig.suptitle(title)
    draw_slices(fig, ph, psize)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("wrote", png)


def comparison_figure(panels, png):
    """Several reconstructions side by side, one titled block each, every
    block carrying its own colour bar.

    The images are square and limited by the row height, so any slack in
    an axes box turns into padding inside the block. To close the gap
    between blocks without opening gaps within them, only the facing
    margins are tightened and the figure is narrowed by what they free.
    """
    n = len(panels)
    fig = plt.figure(figsize=(7.0 * n, 7.4))
    subs = fig.subfigures(1, n, wspace=0.02)
    subs = [subs] if n == 1 else list(subs)
    for i, (sub, (title, (ph, psize))) in enumerate(zip(subs,
                                                        panels.items())):
        sub.suptitle(title, fontsize=14)
        margins = dict(left=0.125 if i == 0 else 0.0625,
                       right=0.9 if i == n - 1 else 0.95,
                       wspace=0.12)
        draw_slices(sub, ph, psize, margins=margins)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", png)


def base_pars(out, numiter):
    p = u.Param()
    p.verbose_level = 1
    p.frames_per_block = 100000
    p.io = u.Param(home=out, rfile=None,
                   autosave=u.Param(active=False),
                   autoplot=u.Param(active=False),
                   interaction=u.Param(active=False))
    p.scans = u.Param()
    p.scans.scan00 = u.Param()
    p.scans.scan00.name = 'Full'
    p.scans.scan00.coherence = u.Param(num_probe_modes=2,
                                       num_object_modes=1)
    p.scans.scan00.data = u.Param(name='PtydScan', dfile=PTYD)
    p.scans.scan00.illumination = u.Param(
        model=None,
        aperture=u.Param(form='circ', size=100e-9),
        propagation=u.Param(parallel=-700e-6),
        diversity=u.Param(power=1, noise=(0.5, 1.0)))
    p.engines = u.Param()
    p.engines.e00 = u.Param(numiter=numiter, numiter_contiguous=1)
    return p


def error_history(P, eng):
    """Per-iteration error curve: the Born engines record their own
    history; ptypy engines leave per-iteration error in runtime.iter_info."""
    hist = getattr(eng, '_err_history', None)
    if hist is not None and len(hist):
        return np.asarray(hist)[:, 0]
    info = P.runtime.iter_info
    return np.array([float(np.mean([e[0] for e in it['error'].values()]))
                     if isinstance(it['error'], dict)
                     else float(np.asarray(it['error']).ravel()[0])
                     for it in info])


def run(tag, engine_pars, numiter, out):
    p = base_pars(out, numiter)
    p.engines.e00.update(engine_pars)
    p.engines.e00.fslices = os.path.join(out, f'slices_{tag}.h5')
    t0 = time.time()
    P = Ptycho(p, level=5)
    dt = time.time() - t0
    eng = list(P.engines.values())[-1]
    err = error_history(P, eng)
    del P, eng
    # subtract data-import/init time by timing iterations only would need
    # engine hooks; report total minus a data-load estimate from a probe
    # run would overfit, so state the whole-run time and the slope
    return err, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numiter", type=int, default=20)
    ap.add_argument("--ls-numiter", type=int, default=20,
                    dest="ls_numiter")
    ap.add_argument("--out", default="results/demo_01")
    args = ap.parse_args()
    out = resolve_out(args.out)
    os.makedirs(out, exist_ok=True)

    import ptypy.custom.intersliceborn_cupy    # registers the ISB engines
    import ptypy.custom.threepie_cupy          # registers ThreePIE_cupy

    common = dict(number_of_slices=2, slice_thickness=1580.e-6,
                  probe_support=0.3)
    # label -> (slice-file tag, iterations, engine parameters). The line
    # search sets its own step every iteration, so it is given fewer.
    runs = {
        "inter-slice first Born reconstruction": (
            "IntersliceBorn_order_1", args.numiter,
            dict(name="IntersliceBorn", order=1, seed=123, shuffle=False,
                 **common)),
        "inter-slice first Born, line search": (
            "IntersliceFirstBornLineSearch", args.ls_numiter,
            dict(name="IntersliceFirstBornLineSearch", fidelity="amplitude",
                 wavefield_delta_object=1.0, wavefield_delta_probe=1.0,
                 seed=123, shuffle=False, **common)),
        "multislice reconstruction": (
            "ThreePIE_cupy", args.numiter,
            dict(name="ThreePIE_cupy", **common)),
    }

    slice_files = {}
    print(f"NanoMAX scan 434, {args.numiter} iterations per engine "
          f"({args.ls_numiter} for the line search)")
    for label, (tag, numiter, pars) in runs.items():
        print(f"== {label}")
        try:
            err, dt = run(tag, u.Param(pars), numiter, out)
            slice_files[label] = os.path.join(out, f"slices_{tag}.h5")
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED: {e}")
            continue
        per_it = dt / max(numiter, 1)
        print(f"   total {dt:.1f} s (~{per_it:.2f} s/iteration incl. "
              f"init+data), final error {err[-1]:.5g}")

    panels = {label: load_slices(fp) for label, fp in slice_files.items()
              if os.path.exists(fp)}
    if len(panels) == len(runs):
        comparison_figure(
            panels, os.path.join(out, "demo_01_born_vs_multislice.png"))
    else:
        print("not both engines finished; no comparison figure written")


if __name__ == "__main__":
    main()
