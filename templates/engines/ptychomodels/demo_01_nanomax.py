"""Demo 01: the NanoMAX scan, inter-slice Born against sequential multislice.

Reconstructs the prepared two-slice NanoMAX scan with three engines on
identical data, illumination and seeds:

  * IntersliceBorn_ptychomodels at order 1 — the inter-slice first Born model;
  * IntersliceBorn_ptychomodels at order 2 — exact multislice through the
    parallel series, since the series terminates at the slice count;
  * ThreePIE_cupy — ptypy's own sequential multislice engine.

Reports the steady-state time per iteration of each engine, excluding
the first, and each engine's error history. The engines log different
functionals (the Born engine an amplitude-l2 per pixel, ptypy's kernels
their own Fourier metric), so the curves are comparable by decay and by
cost, not by absolute level.

    python demo_01_nanomax.py [--numiter 20] [--ptyd PATH]
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[os.pardir] * 3)))

PTYD = (
    "/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/scan_000434/"
    "ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/data/data_scan_000434.ptyd"
)
# ptypy's illumination is not normalised against this scan's data; both
# arms are put on the same photon scale so their errors are comparable
COMMON = dict(number_of_slices=2, slice_thickness=1580.0e-6, probe_support=0.3)


def parameters(out, numiter, ptyd):
    p = u.Param()
    p.verbose_level = 1
    p.frames_per_block = 100000
    p.io = u.Param(
        home=out,
        rfile=None,
        autosave=u.Param(active=False),
        autoplot=u.Param(active=False),
        interaction=u.Param(active=False),
    )
    p.scans = u.Param()
    p.scans.scan00 = u.Param(
        name="Full",
        coherence=u.Param(num_probe_modes=2, num_object_modes=1),
        data=u.Param(name="PtydScan", dfile=ptyd),
        illumination=u.Param(
            model=None,
            aperture=u.Param(form="circ", size=100e-9),
            propagation=u.Param(parallel=-700e-6),
            diversity=u.Param(power=1, noise=(0.5, 1.0)),
        ),
    )
    p.engines = u.Param()
    p.engines.e00 = u.Param(numiter=numiter, numiter_contiguous=1)
    return p


def timed_iterations(engine_class):
    """Wall time of every engine_iterate call, so the first iteration —
    which carries setup — can be dropped from the steady state."""
    marks = []
    original = engine_class.engine_iterate

    def timing(self, num=1):
        start = time.time()
        out = original(self, num)
        marks.append(time.time() - start)
        return out

    engine_class.engine_iterate = timing
    return marks, lambda: setattr(engine_class, "engine_iterate", original)


def errors(ptycho, engine):
    history = getattr(engine, "_err_history", None)
    if history is not None and len(history):
        return np.asarray(history)[:, 0]
    return np.array(
        [
            float(np.mean([e[0] for e in it["error"].values()]))
            if isinstance(it["error"], dict)
            else float(np.asarray(it["error"]).ravel()[0])
            for it in ptycho.runtime.iter_info
        ]
    )


def run(label, engine_class, engine_pars, numiter, out, ptyd):
    p = parameters(out, numiter, ptyd)
    p.engines.e00.update(engine_pars)
    marks, restore = timed_iterations(engine_class)
    try:
        ptycho = Ptycho(p, level=5)
    finally:
        restore()
    engine = list(ptycho.engines.values())[-1]
    curve = errors(ptycho, engine)
    steady = statistics.mean(marks[1:]) if len(marks) > 1 else marks[0]
    print(
        f"{label:46s} first {marks[0]:7.2f} s   steady {steady:7.2f} s/iter   "
        f"error {curve[0]:.4g} -> {curve[-1]:.4g}"
    )
    return steady, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numiter", type=int, default=20)
    ap.add_argument("--ptyd", default=PTYD)
    ap.add_argument("--out", default=os.path.join(HERE, "results_nanomax"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ptypy.custom import threepie_cupy  # noqa: F401  registers ThreePIE_cupy
    from ptypy.custom.ptychomodels_born import IntersliceBorn_ptychomodels
    from ptypy.engines import ENGINES

    runs = [
        (
            "IntersliceBorn_ptychomodels order 1 (first Born)",
            IntersliceBorn_ptychomodels,
            dict(name="IntersliceBorn_ptychomodels", order=1, step="epie", seed=123, shuffle=False),
        ),
        (
            "IntersliceBorn_ptychomodels order 2 (exact multislice)",
            IntersliceBorn_ptychomodels,
            dict(name="IntersliceBorn_ptychomodels", order=2, step="epie", seed=123, shuffle=False),
        ),
        (
            "ThreePIE_cupy (sequential multislice)",
            ENGINES["ThreePIE_cupy"],
            dict(name="ThreePIE_cupy"),
        ),
    ]
    print(f"NanoMAX scan 434, {args.numiter} iterations per engine")
    for label, cls, pars in runs:
        pars.update(COMMON)
        pars["fslices"] = os.path.join(args.out, label.split("(")[0].strip().replace(" ", "_"))
        try:
            run(label, cls, u.Param(pars), args.numiter, args.out, args.ptyd)
        except Exception as failure:  # one engine failing should not hide the others
            print(f"{label:46s} FAILED: {type(failure).__name__}: {failure}")


if __name__ == "__main__":
    main()
