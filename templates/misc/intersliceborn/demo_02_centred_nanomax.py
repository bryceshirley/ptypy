#!/usr/bin/env python
"""Demo 02: the a_j backgrounds on the real NanoMAX scan.

Two IntersliceBorn reconstructions of NanoMAX scan 434, identical in every
respect except the background the Born series is expanded about
(the engine's [background] parameter):

  * 'vacuum': a_j = 1, the plain series g_j = O_j - 1;
  * 'mean': a_j = <O_j>, the patch mean, with the gauge
    A = prod_j a_j restored on the detector field.

The sample is two slices, so order 1 (first Born) is the only truncation
that exists: order 2 is already exact. On the published 500-iteration
object the centred first-Born detector field is 160x closer to exact
two-slice multislice than the vacuum one (4.49 -> 0.028 relative), so
this asks what that buys in an actual reconstruction.

    python demo_02_centred_nanomax.py [--numiter 30]
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
from matplotlib import pyplot as plt

from ptypy import utils as u
from ptypy.core import Ptycho

from demo_01_nanomax_engines import base_pars, error_history

# Results live beside the template, not in whatever directory the demo
# was launched from, so a relative --out always resolves here.
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_out(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


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
    return err, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numiter", type=int, default=30)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--out", default="results/demo_02")
    args = ap.parse_args()
    out = resolve_out(args.out)
    os.makedirs(out, exist_ok=True)

    import ptypy.custom.intersliceborn_cupy    # registers IntersliceBorn

    common = dict(name='IntersliceBorn', order=args.order,
                  seed=123, shuffle=False,
                  number_of_slices=2, slice_thickness=1580.e-6,
                  probe_support=0.3)
    runs = {
        'vacuum background': dict(background='vacuum', **common),
        'centred, patch mean': dict(background='mean', **common),
    }

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    print(f"NanoMAX scan 434, order {args.order}, "
          f"{args.numiter} iterations per arm")
    results = {}
    for col, (label, pars) in zip(['#eb6834', '#5aa02c', '#2a78d6'],
                                  runs.items()):
        print(f"== {label}", flush=True)
        tag = pars['background']
        err, dt = run(tag, u.Param(pars), args.numiter, out)
        results[label] = (err, dt)
        print(f"   total {dt:.1f} s (~{dt/max(args.numiter,1):.2f} s/it), "
              f"error {err[0]:.4g} -> {err[-1]:.4g}", flush=True)
        ax.semilogy(np.arange(1, len(err) + 1), err, color=col, lw=1.8,
                    label=f"{label}: {err[-1]:.4g}")

    ax.set_xlabel("iteration")
    ax.set_ylabel("engine error (amplitude $\\ell_2$ per pixel)")
    ax.set_title(f"NanoMAX scan 434, two slices, IntersliceBorn "
                 f"order {args.order}\n"
                 "vacuum vs centred background")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(out, f"demo_02_centred_order{args.order}.png")
    fig.savefig(png)
    print("wrote", png)
    errs = [e for e, _ in results.values()]
    n = min(len(e) for e in errs)
    base = errs[0][n-1]
    print("\nfinal error")
    for label, (e, _) in results.items():
        print(f"  {label:<26} {e[n-1]:.5g}   (x{e[n-1]/base:.3f} vs vacuum)")


if __name__ == "__main__":
    main()
