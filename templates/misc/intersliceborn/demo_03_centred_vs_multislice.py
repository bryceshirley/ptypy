#!/usr/bin/env python
"""Demo 03: centred first Born vs sequential multislice, 200 iterations.

One arm per invocation so the two can share the box on separate GPUs:

    python demo_03_centred_vs_multislice.py --arm isb       # centred ISB
    python demo_03_centred_vs_multislice.py --arm threepie  # ThreePIE_cupy
    python demo_03_centred_vs_multislice.py --arm plot      # compare outputs

The sample is two slices, so ThreePIE is exact multislice and IntersliceBorn at
order 1 is the only truncation in play. At 30 iterations the two
backgrounds fit the data equally well (error 35.047 vacuum vs 35.052
centred) but give different objects, the centred one carrying ~10% more
phase in both slices, which is the direction of the multislice result.
This runs both to 200 to see whether centred first Born actually lands on
the multislice reconstruction.
"""

import os
import sys

# These templates sit at <repo>/templates/engines/cupy/intersliceborn,
# so the repo they ship with is four levels up; run in place against it
# and put this directory first so the demos can import each other.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while (_REPO != os.path.dirname(_REPO)
       and not os.path.isdir(os.path.join(_REPO, "ptypy", "core"))):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import argparse
import time

import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho

from demo_01_nanomax_engines import base_pars, error_history

# Results live beside the template, not in whatever directory the demo
# was launched from, so a relative --out always resolves here.
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_out(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)

COMMON = dict(number_of_slices=2, slice_thickness=1580.e-6, probe_support=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["isb", "threepie", "plot"],
                    required=True)
    ap.add_argument("--numiter", type=int, default=200)
    ap.add_argument("--out", default="results/demo_03")
    ap.add_argument("--background", default="mean",
                    choices=["vacuum", "mean"],
                    help="ISB arm only: the a_j the series expands about")
    args = ap.parse_args()
    out = resolve_out(args.out)
    os.makedirs(out, exist_ok=True)

    if args.arm == "plot":
        return compare(out)

    if args.arm == "isb":
        import ptypy.custom.intersliceborn_cupy  # noqa: F401
        pars = dict(name='IntersliceBorn', order=1,
                    seed=123, shuffle=False,
                    background=args.background,
                    **COMMON)
    else:
        import ptypy.custom.threepie_cupy  # noqa: F401
        pars = dict(name='ThreePIE_cupy', **COMMON)

    p = base_pars(out, args.numiter)
    p.engines.e00.update(u.Param(pars))
    p.engines.e00.fslices = os.path.join(out, f'slices_{args.arm}.h5')
    t0 = time.time()
    P = Ptycho(p, level=5)
    dt = time.time() - t0
    eng = list(P.engines.values())[-1]
    err = error_history(P, eng)
    np.save(os.path.join(out, f'err_{args.arm}.npy'), err)
    print(f"[{args.arm}] total {dt:.1f} s over {args.numiter} iterations, "
          f"error {err[0]:.5g} -> {err[-1]:.5g}", flush=True)


REF = ("/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/"
       "scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/"
       "rec/rec_crop-512_slices-2_iter-0500.h5")


def compare(out, ref=REF):
    """Object-space comparison, unwrapped and on one shared colour bar.

    The engines' error functionals differ, so only the objects compare.
    Phase is unwrapped (the raw arrays wrap at +-pi), the background is
    shifted to zero, and every panel shares one scale with vmax = 0,
    the sample only retards the wave, so all the signal is negative and a
    common upper bound of zero makes the panels directly readable against
    each other.

    The multislice reference defaults to the PUBLISHED 500-iteration
    ThreePIE reconstruction shipped in the Zenodo archive, not the local
    ThreePIE_cupy run: that engine currently exits without converging on
    this data (phase std ~2.6 rad, 60-92% of pixels pinned at +-pi, error
    history flat at zero).
    """
    import h5py
    import numpy as np
    from matplotlib import pyplot as plt
    from skimage.restoration import unwrap_phase

    def slices(path):
        with h5py.File(path) as f:
            ks = sorted(f['content/objects'])
            return [f[f'content/objects/{k}/Sscan00G00/data'][0] for k in ks]

    def prep(z):
        ph = unwrap_phase(np.angle(z).astype(np.float64))
        return ph - np.median(ph)          # background to 0

    a = [prep(z) for z in slices(os.path.join(out, 'slices_isb.h5'))]
    b = [prep(z) for z in slices(ref)]
    vmin = min(float(np.percentile(x, 0.5)) for x in a + b)
    vmax = 0.0
    print(f"shared colour scale: [{vmin:.3f}, {vmax:.3f}] rad (unwrapped, "
          "background at 0)")

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.4), dpi=140)
    for j, (sa, sb) in enumerate(zip(a, b)):
        d = np.linalg.norm(sa - sb) / np.linalg.norm(sb)
        print(f"slice {j}: centred ISB vs published multislice, unwrapped "
              f"phase rel diff {d:.4f};  min phase ISB {sa.min():.3f}  "
              f"MS {sb.min():.3f}")
        panels = ((sa, 'centred ISB order 1 (200 it)'),
                  (sb, 'multislice reference (3PIE, 500 it)'),
                  (sa - sb, 'difference'))
        for k, (img, name) in enumerate(panels):
            ax = axs[j, k]
            if k < 2:
                m = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
            else:
                lim = float(np.percentile(np.abs(img), 99.5))
                m = ax.imshow(img, cmap='coolwarm', vmin=-lim, vmax=lim)
            ax.set_title(f"slice {j}: {name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, fraction=0.046, label='phase (rad)')
    fig.suptitle("NanoMAX scan 434, two slices: unwrapped phase, shared "
                 f"scale [{vmin:.2f}, 0] rad", fontsize=11)
    fig.tight_layout()
    png = os.path.join(out, 'demo_03_centred_vs_multislice.png')
    fig.savefig(png)
    print("wrote", png)


if __name__ == "__main__":
    main()
