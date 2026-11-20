#!/usr/bin/env python
"""Render object slices in the style of the NanoMAX paper figure.

Uses the figure code in demo_01_nanomax_engines.py and applies it to the
centred-ISB reconstruction
as well as to the published 500-iteration multislice reference, so the
two are directly comparable with the published figure.

    python demo_03_plot_slices.py [--out results/demo_03]
"""

import argparse
import os

import numpy as np

from demo_01_nanomax_engines import load_slices, slice_figure

# Results live beside the template, not in whatever directory the demo
# was launched from, so a relative --out always resolves here.
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_out(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


REF = ("/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/"
       "scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/"
       "rec/rec_crop-512_slices-2_iter-0500.h5")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/demo_03")
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    out = resolve_out(args.out)

    # the vacuum arm writes the same filename in its own folder
    kind = "vacuum" if "vacuum" in os.path.basename(out) else "centred"
    runs = {
        f"{kind}_isb": (os.path.join(out, "slices_isb.h5"),
                        f"{kind} first-Born reconstruction of scan #434 "
                        f"({args.iters} iterations)"),
        "multislice_ref": (REF,
                           "multi-slice reconstruction of scan #434 "
                           "(500 iterations)"),
    }
    phases = {}
    for tag, (path, title) in runs.items():
        if not os.path.exists(path):
            print("missing, skipped:", path)
            continue
        ph, psize = load_slices(path)
        phases[tag] = ph
        for j, p in enumerate(ph):
            print(f"{tag} slice {j}: phase min {p.min():+.3f}  "
                  f"0.5th pct {np.percentile(p, 0.5):+.3f} rad")
        slice_figure(ph, psize, title, os.path.join(out, f"slices_{tag}.png"))

    if len(phases) == 2:
        a, b = phases[f"{kind}_isb"], phases["multislice_ref"]
        for j, (x, y) in enumerate(zip(a, b)):
            d = np.linalg.norm(x - y) / np.linalg.norm(y)
            print(f"slice {j}: {kind} ISB vs multislice reference, "
                  f"relative phase difference {d:.4f}")


if __name__ == "__main__":
    main()
