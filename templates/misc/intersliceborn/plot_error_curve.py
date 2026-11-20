#!/usr/bin/env python
"""Convergence curve for an ISB run directory.

Reads err_isb.npy (the engine's per-iteration amplitude-l2 error, one
entry per iteration) and writes error_isb.png next to it.

    python plot_error_curve.py results/demo_03 [more_dirs ...]
"""

import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

# Run directories resolve beside the template unless given absolutely.
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--name", default="err_isb.npy")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    for d in args.dirs:
        d = d if os.path.isabs(d) else os.path.join(HERE, d)
        path = os.path.join(d, args.name)
        if not os.path.exists(path):
            print("missing, skipped:", path)
            continue
        err = np.load(path)
        it = np.arange(1, len(err) + 1)
        ax.semilogy(it, err, lw=1.8, marker="o", ms=3,
                    label=f"{os.path.basename(d.rstrip('/'))}: "
                          f"{err[0]:.4g} -> {err[-1]:.4g} in {len(err)} it")
        drop = err[0] / err[-1]
        print(f"{d}: {len(err)} iterations, {err[0]:.4g} -> {err[-1]:.4g} "
              f"({drop:.2f}x), monotone {bool(np.all(np.diff(err) < 0))}")

    ax.set_xlabel("iteration")
    ax.set_ylabel(r"engine error (amplitude $\ell_2$ per pixel)")
    ax.set_title("IntersliceBorn convergence, NanoMAX scan 434")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(os.path.abspath(args.dirs[0]), "error_isb.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
