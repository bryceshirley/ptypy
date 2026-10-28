"""
Demo 8: does the Born order M* depend on the slicing N?

Reproduces the study of ~/papers/figs-p1/paper1_M_vs_N.py with the
production CuPy primitives of ptypy.custom.interslice_born instead of
the ptyrad torch backend: for one fixed physical specimen, re-sliced
into N in {16, 32, 64, 128}, the detector orders D_1..D_16 come from
isb_cupy.born_detector_orders, the same-N exact multislice detector
field from isb_cupy.born_multislice_target, and for every truncation M
the readout coefficients are the TSVD/QR least-squares fit of the
multislice tail onto the retained orders (base.born_qr_coeffs, the
engines' 'tail' fit). M* is the smallest M whose relative detector-
INTENSITY error delta_I = ||I_fit - I_ms|| / ||I_ms|| is below each
tolerance in {1e-2, 1e-3, 1e-4}, with M capped at 16. The claim under
test: M* is set by the specimen's total phase contrast, not by N.

SUBSTITUTION vs the reference: abtem and ase are not installed in this
python environment, so the SrTiO3 [001] slice potentials are replaced by
ONE fixed synthetic 3D phase volume -- a smooth Gaussian random field
(Fourier low-passed in all three axes) sampled on 128 fine sub-slices of
a 128 x 128 grid. For each N the fine sub-slices are summed in
consecutive groups of 128/N, so the physical object (its total projected
phase, feature for feature) is IDENTICAL for every N, exactly as the
reference's fixed atomic stack is. Contrast scales {0.25, 0.5, 1.0}
multiply the slice phases, as in the reference. The native (x1.0) peak
total projected phase is set to 4.0 rad rather than the reference's
measured 19.5 rad: the SrTiO3 peak lives in sharp sub-Angstrom atomic
columns, which a smooth volume cannot mimic -- a smooth 19.5 rad object
would need M far beyond the 16-order cap -- so the native contrast here
is chosen to land M* inside the range the reference observed (2..9).
Propagators are the reference's Fresnel stack H_j = exp(-i pi lam z_j f^2)
(entrance plane H_0 = 1), 200 keV, 40 A box, 100 A thickness; plane-wave
probe; everything in complex128 on the GPU (no timing here, accuracy
matters at tol 1e-4).

Figure demo_out/demo_08_M_vs_N.png follows the reference layout: (a) specimen
beam view (total projected phase), (b) side view, (c) M* against N per
contrast -- extended to all three tolerances via line styles (the
reference's panel (c) drew tol 1e-2 only). Raw numbers go to
demo_out/demo_08_M_vs_N.json.

Run:  python demo_08_M_vs_N.py [--recompute] [--gpts 128]
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_out")
JSON = os.path.join(OUT, "demo_08_M_vs_N.json")
PNG = os.path.join(OUT, "demo_08_M_vs_N.png")

N_LIST = [16, 32, 64, 128]
SCALES = [0.25, 0.5, 1.0]
TOLS = (1e-2, 1e-3, 1e-4)
M_CAP = 16
N_FINE = 128           # fine sub-slicing of the fixed volume
BOX_A = 40.0           # box side (Angstrom)
THICK_A = 100.0        # total thickness (Angstrom)
LAM_A = 0.025079       # 200 keV electron wavelength (Angstrom)
PEAK_PHASE = 4.0       # native peak total projected phase (rad), see docstring


def build_volume(n):
    """The ONE fixed specimen: smooth random phase per fine sub-slice,
    (N_FINE, n, n), peak |total projected phase| = PEAK_PHASE rad."""
    rng = np.random.default_rng(7)
    vol = rng.standard_normal((N_FINE, n, n))
    fz = np.fft.fftfreq(N_FINE)[:, None, None]
    fy = np.fft.fftfreq(n)[None, :, None]
    fx = np.fft.fftfreq(n)[None, None, :]
    filt = np.exp(-0.5 * ((fz / 0.08) ** 2 + (fy / 0.05) ** 2
                          + (fx / 0.05) ** 2))
    vol = np.fft.ifftn(np.fft.fftn(vol) * filt).real
    vol *= PEAK_PHASE / np.abs(vol.sum(axis=0)).max()
    return vol


def reslice(vol, N):
    """Sum consecutive fine sub-slices: the same physical object at N
    slices, (N, n, n) phase per slice in rad."""
    assert N_FINE % N == 0
    return vol.reshape(N, N_FINE // N, *vol.shape[1:]).sum(axis=1)


def recompute(gpts):
    import cupy as cp

    from ptypy.custom.interslice_born.base import born_qr_coeffs
    from ptypy.custom.interslice_born.isb_cupy import (
        born_detector_orders, born_multislice_target)

    vol = build_volume(gpts)
    dx = BOX_A / gpts
    f = np.fft.fftfreq(gpts, d=dx)
    F2 = f[:, None] ** 2 + f[None, :] ** 2

    def build_H(N):
        """Fresnel propagator powers H^j, j = 0..N-1, (N, 1, 1, ny, nx)
        complex128, the reference's build_H in the isb convention
        (H_fwd[0] = 1 at the entrance plane)."""
        dz = THICK_A / N
        base = np.exp(-1j * math.pi * LAM_A * dz * F2)
        Hs = np.stack([base ** j for j in range(N)])
        return cp.asarray(Hs[:, None, None], dtype=cp.complex128)

    probe = cp.ones((1, gpts, gpts), dtype=cp.complex128)  # plane wave, Mp=1
    results = []
    struct = None
    for N in N_LIST:
        ph = reslice(vol, N)
        if N == max(N_LIST):
            struct = {"beam": ph.sum(axis=0), "side": ph.sum(axis=1)}
        H_fwd = build_H(N)
        H_rev = cp.conj(H_fwd)
        for s in SCALES:
            O = cp.asarray(np.exp(1j * s * ph))[:, None]  # [N, B=1, ny, nx]
            D0, D = born_detector_orders(O, probe, H_fwd, H_rev,
                                         min(M_CAP, N))
            T_full = born_multislice_target(O, probe, H_fwd)
            D0h = cp.asnumpy(D0)          # [B, Mp, ny, nx]
            Dh = cp.asnumpy(D)            # [M_CAP, B, Mp, ny, nx]
            Th_full = cp.asnumpy(T_full)
            T = Th_full - D0h
            d0n2 = float((np.abs(D0h) ** 2).sum())
            I_ms = (np.abs(Th_full) ** 2).sum(axis=1)      # modes
            need = {f"{t:g}": None for t in TOLS}
            dIs = []
            for M in range(1, min(M_CAP, N) + 1):
                c, _ = born_qr_coeffs(Dh[:M], T, d0n2)
                field = D0h + np.tensordot(c, Dh[:M], axes=1)
                I_fit = (np.abs(field) ** 2).sum(axis=1)
                dI = float(np.linalg.norm(I_fit - I_ms)
                           / np.linalg.norm(I_ms))
                dIs.append(dI)
                for t in TOLS:
                    k = f"{t:g}"
                    if need[k] is None and dI < t:
                        need[k] = M
                if all(v is not None for v in need.values()):
                    break
            phimax = float(np.abs(ph.sum(axis=0)).max()) * s
            results.append({"N": N, "scale": s, "total_phase_max": phimax,
                            "M_star": need, "delta_I_per_M": dIs})
            print(f"N={N:5d} s={s:g}: phase_max {phimax:.2f} rad, "
                  f"M* {need}, dI tail {dIs[-1]:.2e}", flush=True)
            del O, D0, D, T_full
            cp.get_default_memory_pool().free_all_blocks()
        del H_fwd, H_rev
        cp.get_default_memory_pool().free_all_blocks()

    os.makedirs(OUT, exist_ok=True)
    out = {"box_A": BOX_A, "thick_A": THICK_A, "gpts": gpts,
           "lam_A": LAM_A, "tols": list(TOLS), "M_cap": M_CAP,
           "N_fine": N_FINE, "peak_phase_native_rad": PEAK_PHASE,
           "specimen": "synthetic smooth random 3D phase volume "
                       "(abtem/ase unavailable; see module docstring)",
           "backend": "ptypy.custom.interslice_born isb_cupy "
                      "(born_detector_orders + born_multislice_target, "
                      "complex128) + base.born_qr_coeffs",
           "struct_beam": struct["beam"].tolist(),
           "struct_side": struct["side"].tolist(),
           "results": results}
    with open(JSON, "w") as fjs:
        json.dump(out, fjs, indent=1)
    print("wrote", JSON)
    return out


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    COLORS = {0.25: "#0072B2", 0.5: "#D55E00", 1.0: "#009E73"}
    MARKERS = {0.25: "o", 0.5: "s", 1.0: "^"}
    STYLES = {"0.01": "-", "0.001": "--", "0.0001": ":"}

    results = r["results"]
    beam = np.asarray(r["struct_beam"])
    side = np.asarray(r["struct_side"])
    box, thick = r["box_A"], r["thick_A"]

    fig = plt.figure(figsize=(10.6, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35],
                          height_ratios=[2.6, 1.0],
                          left=0.06, right=0.985, top=0.86, bottom=0.175,
                          wspace=0.24, hspace=0.55)

    # (a) specimen, beam view: total projected phase (native contrast)
    axA = fig.add_subplot(gs[0, 0])
    imA = axA.imshow(beam, origin="lower", cmap="magma",
                     extent=[0, box, 0, box])
    axA.set_xlabel(r"$x$ (Å)", fontsize=8)
    axA.set_ylabel(r"$y$ (Å)", fontsize=8)
    axA.tick_params(labelsize=7)
    axA.set_title("(a) specimen, beam view: smooth random volume\n"
                  f"{box:g} x {box:g} x {thick:g} Å, plane wave, "
                  f"peak {r['peak_phase_native_rad']:g} rad", fontsize=8)
    cb = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label("projected phase (rad)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # (b) specimen, side view along the beam
    axB = fig.add_subplot(gs[1, 0])
    axB.imshow(side.T, origin="lower", cmap="magma", aspect="auto",
               extent=[0, thick, 0, box])
    axB.set_xlabel(r"$z$, beam direction (Å)", fontsize=8)
    axB.set_ylabel(r"$x$ (Å)", fontsize=8)
    axB.tick_params(labelsize=7)
    axB.set_title("(b) side view (phase projected along $y$)", fontsize=8)

    # (c) the result: M* against slicing N, per contrast, per tolerance
    axC = fig.add_subplot(gs[:, 1])
    for s in SCALES:
        sel = [x for x in results if x["scale"] == s]
        for t in TOLS:
            k = f"{t:g}"
            xs = [x["N"] for x in sel if x["M_star"][k] is not None]
            ys = [x["M_star"][k] for x in sel if x["M_star"][k] is not None]
            axC.semilogx(xs, ys, STYLES[k], marker=MARKERS[s],
                         color=COLORS[s], markersize=6, linewidth=1.7,
                         label=(rf"peak phase {sel[0]['total_phase_max']:.1f}"
                                " rad" + (" (native)" if s == 1.0
                                          else rf" ($\times${s:g})"))
                         if t == TOLS[0] else None)
    axC.set_xticks(N_LIST)
    axC.set_xticklabels([str(n) for n in N_LIST])
    axC.xaxis.set_minor_formatter(plt.NullFormatter())
    axC.yaxis.set_major_locator(MaxNLocator(integer=True))
    axC.set_xlabel("number of slices $N$ (same physical stack)")
    axC.set_ylabel(r"Born order $M^*$ for $\delta_I <$ tol"
                   "\nvs same-$N$ multislice")
    axC.grid(True, which="both", alpha=0.25, linewidth=0.5)
    leg1 = axC.legend(frameon=False, fontsize=8.5,
                      title="specimen contrast", title_fontsize=8.5,
                      loc="upper left")
    axC.add_artist(leg1)
    axC.legend(handles=[Line2D([], [], color="0.3", ls=STYLES[f"{t:g}"],
                               label=f"tol {t:g}") for t in TOLS],
               frameon=False, fontsize=8.5, loc="lower right")
    axC.set_title("(c) order needed is set by contrast, not slicing",
                  fontsize=9)
    fig.suptitle("Inter-slice Born order $M^*$ against slicing $N$ "
                 "(CuPy primitives)", fontsize=11)
    fig.savefig(PNG, dpi=300)
    print("wrote", PNG)

    # headline: is M* flat in N?
    for s in SCALES:
        sel = [x for x in results if x["scale"] == s]
        for t in TOLS:
            k = f"{t:g}"
            ms = [x["M_star"][k] for x in sel]
            flat = ("FLAT" if len(set(ms)) == 1 else
                    f"spread {min(m for m in ms if m) }.."
                    f"{max(m for m in ms if m)}"
                    if all(m is not None for m in ms) else "not reached")
            print(f"scale {s:4g} tol {t:g}: M* over N {ms} -> {flat}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--gpts", type=int, default=128)
    a = ap.parse_args()
    if a.recompute or not os.path.exists(JSON):
        r = recompute(a.gpts)
    else:
        r = json.load(open(JSON))
    plot(r)
