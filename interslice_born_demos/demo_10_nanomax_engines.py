"""
Demo 10: the real NanoMAX scan -- ThreePIE_cupy vs ISBCupy.

Reconstructs NanoMAX scan 434 (the prepared two-slice .ptyd of the paper
campaign) with three GPU engines on identical data, illumination and
seeds:

  * ISBCupy at order 1  -- the inter-slice first Born model;
  * ISBCupy at order 2  -- exact multislice through the parallel series
    (2 slices), with CUDA-graph replay;
  * ThreePIE_cupy       -- upstream ptypy's sequential multislice engine.

Reports the wall time per iteration of each engine and overlays their
error histories. The engines log different error functionals (the Born
engines an amplitude-l2 per pixel, ptypy's kernels their own Fourier
metric), so the curves are normalised to each engine's first iteration:
the comparison is decay and speed, not absolute level.

    python demo_10_nanomax_engines.py [--numiter 20]
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np
from matplotlib import pyplot as plt

from ptypy import utils as u
from ptypy.core import Ptycho

PTYD = ('/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/'
        'scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/'
        'data/data_scan_000434.ptyd')


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
    # run would overfit -- state the whole-run time and per-iteration slope
    return err, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numiter", type=int, default=20)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    import ptypy.custom.interslice_born.isb_cupy    # registers ISBCupy
    import ptypy.custom.threepie_cupy               # registers ThreePIE_cupy

    common = dict(number_of_slices=2, slice_thickness=1580.e-6,
                  probe_support=0.3)
    runs = {
        'ISBCupy order 1 (first Born)': dict(
            name='ISBCupy', order=1, device='cuda', precision='single',
            seed=123, shuffle=False, plot_interval=0, cuda_graphs=True,
            **common),
        'ISBCupy order 2 (exact multislice, graphs)': dict(
            name='ISBCupy', order=2, device='cuda', precision='single',
            seed=123, shuffle=False, plot_interval=0, cuda_graphs=True,
            **common),
        'ThreePIE_cupy (sequential multislice)': dict(
            name='ThreePIE_cupy', **common),
    }

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    colors = ['#2a78d6', '#7C3AED', '#eb6834']
    slice_files = {}
    print(f"NanoMAX scan 434, {args.numiter} iterations per engine")
    for col, (label, pars) in zip(colors, runs.items()):
        print(f"== {label}")
        try:
            tag = label.split()[0] + label.split()[1]
            err, dt = run(tag, u.Param(pars), args.numiter, out)
            slice_files[label] = os.path.join(out, f'slices_{tag}.h5')
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"   FAILED: {e}")
            continue
        per_it = dt / max(args.numiter, 1)
        print(f"   total {dt:.1f} s (~{per_it:.2f} s/iteration incl. "
              "init+data)")
        ax.semilogy(np.arange(1, err.size + 1),
                    np.maximum(err / max(err[0], 1e-30), 1e-30),
                    color=col, label=f"{label} ({dt:.0f} s)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error / error at iteration 1")
    ax.set_title("NanoMAX scan 434: inter-slice Born (cupy) vs "
                 "sequential multislice (cupy)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "demo_10_nanomax_engines.png")
    fig.savefig(path)
    print("wrote", path)

    # ground truth is unknown for real data; show the reconstructions:
    # slice phases per engine, shared grey scale, one panel grid
    import h5py
    stacks = {}
    for label, fp in slice_files.items():
        if not os.path.exists(fp):
            continue
        with h5py.File(fp, 'r') as f:
            objs = f['content/objects']
            sl = []
            for k in sorted(objs.keys()):
                sk = list(objs[k].keys())[0]
                d = np.array(objs[k][sk]['data'])
                sl.append(d[0] if d.ndim == 3 else d)
        stacks[label] = np.stack(sl)
    if stacks:
        nsl = max(st.shape[0] for st in stacks.values())
        figr, axr = plt.subplots(len(stacks), nsl,
                                 figsize=(4.2 * nsl, 4.0 * len(stacks)),
                                 dpi=130, constrained_layout=True,
                                 squeeze=False)
        for i, (label, st) in enumerate(stacks.items()):
            for j in range(nsl):
                ax2 = axr[i, j]
                ph = np.angle(st[j]) if j < st.shape[0] else None
                if ph is None:
                    ax2.axis('off')
                    continue
                lo, hi = np.percentile(ph, 1), np.percentile(ph, 99)
                ax2.imshow(ph, cmap='Greys_r', vmin=lo, vmax=hi)
                if j == 0:
                    ax2.set_ylabel(label, fontsize=8)
                if i == 0:
                    ax2.set_title(f'slice {j}', fontsize=9)
                ax2.set_xticks([]), ax2.set_yticks([])
        figr.suptitle('NanoMAX scan 434: reconstructed slice phases')
        rp = os.path.join(out, 'demo_10_nanomax_recon.png')
        figr.savefig(rp)
        print('wrote', rp)


if __name__ == "__main__":
    main()
