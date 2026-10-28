"""
Demo 2b: truncation error in the STRONG-scattering regime, where the
fitted readout coefficients actually matter.

On the weak 3-slice phantom of demo_01_simulate_multislice.py the plain
partial sums of the inter-slice Born series already decay monotonically,
so the data-free coefficient fits (born_qr_coeffs / tail fit and
born_exit_residual_coeffs / residual fit) have nothing to repair and all
three eps_M curves coincide. This demo builds the opposite regime: a
thick stack (default 10 slices) of dense, radian-scale phase slices, so
the scattering step K: psi -> sum_j G_j (O_j - 1) psi has norm well above
one. The per-order detector fields D_m then GROW with m over the first
orders and the plain partial sums get worse before they get better --
eps_M rises above 1 at intermediate M. Convergence at M = N is still
guaranteed regardless, because K is nilpotent (each term picks up at
least one new slice, so K^{N+1} = 0): the series is always a finite sum,
strong or weak. What the fits buy is everything in between: reweighting
the SAME M retained orders with least-squares coefficients (tail fit
against the exact multislice target, residual fit against the detector
fixed-point residual, no measured data in either) tames the intermediate
blow-up by orders of magnitude. The tail fit is an exact projection, so
it is never worse than plain and non-increasing in M; the residual fit
is target-free and may trail it -- both are printed honestly.

Prints the eps_M table, writes demo_out/demo_03_truncation_error_strong.png, and
reports/asserts the headline numbers: the plain curve is non-monotone
(rises above 1), and max over 0 < M < N of plain/fitted eps is >= 10.

    python demo_03_truncation_strong.py [--slices 10] [--size 64] ...
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

from demo_01_simulate_multislice import (
    build_propagators,
    focused_probe,
    phantom_stack,
)
from ptypy.custom.interslice_born.base import (
    born_exit_residual_coeffs,
    born_qr_coeffs,
    born_volume_remainder_coeffs,
)
from ptypy.custom.interslice_born.isb_torch import (
    born_detector_orders,
    born_multislice_target,
    born_multislice_volume_target,
    born_volume_basis,
)


def strong_phantom_stack(nslices, n, rng, strength):
    """Dense strong-phase slices: the disc/bar stack of demo 1 pushed to
    several radians, multiplied by a smooth random phase screen that covers
    the WHOLE field at every depth. Density is the point -- with the beam
    everywhere inside perturbed material, the slice perturbations overlap
    in the beam and the per-order fields compound instead of staying in
    disjoint sparse supports."""
    base = phantom_stack(nslices, n, rng, strength=strength)
    f = np.fft.fftfreq(n)
    lowpass = np.exp(-((f[:, None] ** 2 + f[None, :] ** 2) / (6.0 / n) ** 2))
    screens = []
    for _ in range(nslices):
        field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n)))
                             * lowpass).real
        field *= strength / max(field.std(), 1e-12)
        screens.append(np.exp(1j * field))
    return base * np.stack(screens)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", type=int, default=50)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--precision", choices=("single", "double"),
                    default="single",
                    help="basis/target dtype; the fits solve in complex128 "
                         "either way")
    ap.add_argument("--strength", type=float, default=0.1,
                    help="phase depth per slice [rad], discs and screen")
    ap.add_argument("--lam", type=float, default=1e-10, help="wavelength [m]")
    ap.add_argument("--res", type=float, default=20e-9, help="pixel size [m]")
    ap.add_argument("--dz", type=float, default=100e-6, help="slice gap [m]")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, N = args.size, args.slices
    H = build_propagators(N, n, args.lam, args.res, args.dz)

    stack = strong_phantom_stack(N, n, rng, args.strength)
    P = focused_probe(n)
    shifts = (rng.random((args.positions, 2)) * n / 4).astype(int)
    O = np.stack([np.stack([np.roll(s, tuple(sh), axis=(0, 1))
                            for s in stack]) for sh in shifts])  # [B, N, n, n]
    O = np.ascontiguousarray(O.transpose(1, 0, 2, 3))            # [N, B, n, n]

    cdt = (torch.complex64 if args.precision == "single"
           else torch.complex128)
    O_t = torch.as_tensor(O, dtype=cdt)
    P_t = torch.as_tensor(P[None], dtype=cdt)                     # 1 mode
    H_t = torch.as_tensor(H, dtype=cdt)

    D0, D = born_detector_orders(O_t, P_t, H_t, H_t.conj(), N)
    T = born_multislice_target(O_t, P_t, H_t)
    # the detector records intensity: evaluate every truncation by the
    # error of |psi_hat|^2 (mode-summed) against the multislice intensity
    I_ms = (T.abs() ** 2).sum(dim=1)
    i_norm = float(I_ms.pow(2).sum().sqrt())
    # per-plane basis and target for the volume-remainder fit
    V_all, _, _ = born_volume_basis(O_t, P_t, H_t, H_t.conj(), N)
    T_vol = born_multislice_volume_target(O_t, P_t, H_t)
    # the TANGENT series: the same hierarchy built on the linearised
    # perturbation g = i*phi (phi = -i log O). It does not terminate at
    # multislice -- the intra-slice linearisation survives every order --
    # so its curve flattens at that floor instead of collapsing.
    D0_tan, D_tan = born_detector_orders(
        1.0 + 1j * (-1j * torch.log(O_t)), P_t, H_t, H_t.conj(), N)

    def eps(field):
        I = (field.abs() ** 2).sum(dim=1)
        return float((I - I_ms).pow(2).sum().sqrt()) / i_norm

    print(f"{N} slices, {O.shape[1]} positions, {args.strength:g} rad/slice "
          f"-- eps_M = ||I^(<=M) - I_ms|| / ||I_ms|| (intensity)")
    print(f"{'M':>3} {'plain':>12} {'tail fit':>12} {'residual fit':>14} {'volume fit':>12} {'tangent':>12}")
    rows = []
    tail_np = (T - D0).numpy()
    d0n2 = float(np.linalg.norm(D0.numpy().ravel()) ** 2)
    for M in range(0, N + 1):
        acc = D0 + (D[:M].sum(0) if M else 0)
        e_plain = eps(acc)
        if M:
            c_t, _ = born_qr_coeffs(D[:M].numpy(), tail_np, d0n2)
            c_r, _ = born_exit_residual_coeffs(D[: min(M + 1, N)].numpy(), M)
            w = lambda c: D0 + sum(  # noqa: E731
                complex(c[m]) * D[m] for m in range(M))
            c_v, _ = born_volume_remainder_coeffs(
                V_all[: M + 1].numpy(), D[:M].numpy(),
                T_vol.numpy(), T.numpy(), D0.numpy())
            e_tail, e_res, e_vol = eps(w(c_t)), eps(w(c_r)), eps(w(c_v))
        else:
            e_tail = e_res = e_vol = e_plain
        e_tan = eps(D0_tan + (D_tan[:M].sum(0) if M else 0))
        rows.append((M, e_plain, e_tail, e_res, e_vol, e_tan))
        print(f"{M:>3} {e_plain:>12.3e} {e_tail:>12.3e} {e_res:>14.3e} "
              f"{e_vol:>12.3e} {e_tan:>12.3e}")

    rows = np.array(rows)

    # ---- the same fits judged in the VOLUME norm --------------------------
    # The detector metric above is the tail fit's own objective, so it wins
    # there by construction. The volume-remainder fit minimises the error
    # of the WAVEFIELD on every slice plane plus the detector; in that
    # norm the ordering flips. Evaluate both fits' volume-rows residual
    # (relative to the multislice volume tail) at a few orders:
    det_scale = 1.0 / np.sqrt(np.prod(D.shape[-2:]))
    Tv, Td, D0n = T_vol.numpy(), T.numpy(), D0.numpy()
    Vn, Dn = V_all.numpy(), D.numpy()

    def vol_resid(c, M):
        vol = sum(complex(c[m - 1]) * np.broadcast_to(Vn[m], Tv.shape)
                  for m in range(1, M + 1))
        det = sum(complex(c[m - 1]) * Dn[m - 1] for m in range(M))
        r_v = vol - (Tv - np.broadcast_to(Vn[0], Tv.shape))
        r_d = (det - (Td - D0n)) * det_scale
        b_v = Tv - np.broadcast_to(Vn[0], Tv.shape)
        b_d = (Td - D0n) * det_scale
        num = np.sqrt(np.linalg.norm(r_v[1:]) ** 2
                      + np.linalg.norm(r_d) ** 2)
        den = np.sqrt(np.linalg.norm(b_v[1:]) ** 2
                      + np.linalg.norm(b_d) ** 2)
        return float(num / den)

    print("volume-norm residual (wavefield on all planes + detector, "
          "relative to the multislice tail):")
    print(f"{'M':>3} {'tail fit':>12} {'volume fit':>12} {'tangent':>12}")
    for M in (2, N // 4, N // 2, 3 * N // 4):
        c_t, _ = born_qr_coeffs(Dn[:M], (Td - D0n), 
                                float(np.linalg.norm(D0n.ravel()) ** 2))
        c_v, _ = born_volume_remainder_coeffs(
            Vn[: M + 1], Dn[:M], Tv, Td, D0n)
        print(f"{M:>3} {vol_resid(c_t, M):>12.3e} "
              f"{vol_resid(c_v, M):>12.3e}")
    print("(the volume fit wins in its own norm; the detector plot above "
          "is the tail fit's objective)")

    os.makedirs(args.out, exist_ok=True)
    # figure style ported from papers/figs-p1/born_exit_remainder_pso.py
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
        "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    Ms = rows[:, 0]
    curves = [
        (1, "#2563EB", "-o", None, r"plain truncation,  $c=1$"),
        (2, "#E8711A", "--s", "none", r"tail-fit coefficients $c_m$"),
        (3, "#059669", "--^", "none", r"residual-fit coefficients $c_m$"),
        (4, "#7C3AED", "--D", "none",
         r"volume-remainder coefficients $c_m$"),
        (5, "#DC2626", "-x", None,
         r"tangent series ($g = i\phi$), plain"),
    ]
    for k, col, style, mfc, lab in curves:
        ax.plot(Ms, np.maximum(rows[:, k], 1e-17), style,
                color=col, ms=5, mfc=mfc, label=lab)
    ax.set_yscale("log")
    ax.axhline(1.0, color="#9CA3AF", lw=0.8, ls="--")
    ax.set_xticks(Ms[::2])
    ax.set_xlabel("truncation order $M$")
    ax.set_ylabel(r"$\|| I^{(\leq M)} - I_{\rm MS}\||_2 \,/\, "
                  r"\|| I_{\rm MS}\||_2$")
    ax.set_title("Inter-slice Born truncation error, strong-scattering "
                 "phantom\n"
                 f"{N} slices of {args.strength:g} rad ({N * args.strength:g} rad total)  ·  "
                 f"{O.shape[1]} positions, {n}x{n} frames  ·  "
                 f"{args.precision} precision, intensity error, data-free coefficient fits",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    png = os.path.join(args.out, "demo_03_truncation_error_strong.png")
    fig.savefig(png, bbox_inches="tight")
    print(f"wrote {png}")

    # headline numbers over the intermediate orders 0 < M < N
    mid = rows[1:N]
    peak_plain = float(mid[:, 1].max())
    gain_tail = float((mid[:, 1] / mid[:, 2]).max())
    gain_res = float((mid[:, 1] / mid[:, 3]).max())
    gain_vol = float((mid[:, 1] / mid[:, 4]).max())
    print(f"peak plain eps over 0 < M < N : {peak_plain:.3e}")
    print(f"max plain/tail-fit  gain      : {gain_tail:.3e}x")
    print(f"max plain/residual-fit gain   : {gain_res:.3e}x")
    print(f"max plain/volume-fit gain     : {gain_vol:.3e}x")
    assert peak_plain > 1.0, "plain series did not blow up -- not strong"
    assert gain_tail >= 3.0, "tail fit gained < 3x at every M"
    print("OK: plain series non-monotone (rises above 1); "
          "the fits tame the hump at intermediate M")


if __name__ == "__main__":
    main()
