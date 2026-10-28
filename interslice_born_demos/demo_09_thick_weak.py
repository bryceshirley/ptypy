"""
Demo 9: a thick, weakly scattering sample -- the first/second Born regime.

Many thin, weak slices (default 32 slices of 0.05 rad) of dense phantom
structure, exact multislice data from the production
born_multislice_target. In this regime the inter-slice Born truncation
error eps_M is tiny already at M = 1 and negligible at M = 2, so an
order-1 or order-2 reconstruction is as good as full multislice at a
fraction of the sequential depth. The demo shows both facts:

  * the model side: the eps_M table of the ground truth (printed);
  * the inversion side: fixed-step reconstructions of the SAME multislice
    data with the order-M core at M = 1, 2 and 4 -- their error curves
    coincide down to each order's model floor, which for M >= 2 is below
    anything the iteration reaches.

    python demo_09_thick_weak.py [--slices 32] [--strength 0.05]
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt

from ptypy.custom.interslice_born.isb_torch import (
    born_detector_orders,
    born_multislice_target,
    born_series_step,
)
from demo_01_simulate_multislice import build_propagators


def broad_probe(n, radius=0.33, edge=6.0):
    """A broad soft-edged disc of illumination: lights most of every
    window, so adjacent scan positions overlap strongly and every blob
    is seen from many angles -- a focused spot leaves coverage holes at
    this scan density."""
    yy, xx = np.mgrid[:n, :n] - n / 2
    r = np.hypot(yy, xx)
    amp = 0.5 * (1 - np.tanh((r - radius * n) / edge))
    return amp * np.exp(1j * 0.5 * (r / (radius * n)) ** 2)


def blob_stack(N, nc, rng, strength, n_blobs=9, sigma_px=5.0):
    """Weak, well-separated smooth blobs distributed over the slices:
    centres drawn from a shuffled coarse grid so every pair is at least
    one grid cell apart in projection, blob b living on slice b mod N.
    Structured and sparse, so the projected phase is carried by
    recoverable spatial frequencies rather than a low-frequency carpet.
    """
    # keep every blob inside the region the focused probe actually
    # lights: window centres span roughly [0.3, 0.7] of the canvas
    cells = 3
    lo, span = 0.30 * nc, 0.40 * nc
    pitch = span / cells
    centres = [(lo + (i + 0.5) * pitch, lo + (j + 0.5) * pitch)
               for i in range(cells) for j in range(cells)]
    rng.shuffle(centres)
    centres = centres[:n_blobs]
    yy, xx = np.mgrid[:nc, :nc]
    stack = np.zeros((N, nc, nc))
    for b, (cy, cx) in enumerate(centres):
        cy += (rng.random() - 0.5) * 0.3 * pitch
        cx += (rng.random() - 0.5) * 0.3 * pitch
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                        / (2 * sigma_px ** 2)))
        stack[b % N] -= strength * blob
    return np.exp(1j * stack), stack

CDT = torch.complex128


def ff_fw(x):
    return torch.fft.fft2(x, norm="ortho")


def ff_bw(x):
    return torch.fft.ifft2(x, norm="ortho")


def reconstruct(M, I, P0, positions, nc, n, H, iters, dev, alpha=1.9,
                update_probe=False):
    """Fixed-step Jacobi reconstruction with the order-M Born core."""
    N = H.shape[0]
    ob = torch.ones((N, nc, nc), dtype=CDT, device=dev)
    P = P0.clone()
    eps = torch.finfo(torch.float64).eps
    errs = []
    for it in range(iters):
        e0 = 0.0
        for (r, c), Ii in zip(positions, I):
            patch = ob[:, r:r + n, c:c + n].unsqueeze(1)
            do_probe = update_probe and it >= 2
            g, K, gP, KP, _, Im = born_series_step(
                patch, P, Ii.unsqueeze(0), H, H.conj(), ff_fw, ff_bw,
                M=M, calc_probe_grad=do_probe)
            e0 += float(((Im.clamp_min(0).sqrt()
                          - Ii.sqrt()) ** 2).sum()) / Ii.numel()
            upd = g / K.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
            ob[:, r:r + n, c:c + n] += (alpha / N) * upd[:, 0]
            if do_probe:
                P = P + (1.0 / N) * gP / KP.amax().clamp_min(eps)
        errs.append(e0 / len(positions))
    return ob.cpu(), np.asarray(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", type=int, default=32)
    ap.add_argument("--strength", type=float, default=0.05,
                    help="phase scale per slice [rad]")
    ap.add_argument("--canvas", type=int, default=160)
    ap.add_argument("--frame", type=int, default=96)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--update-probe", action="store_true",
                    help="also reconstruct the probe; off by default -- the "
                         "illumination is exactly known for simulated data, "
                         "and at this weak contrast a free probe absorbs "
                         "object structure (probe-object ambiguity)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    rng = np.random.default_rng(5)
    nc, n, N = args.canvas, args.frame, args.slices

    # weak, well-separated blobs distributed over the slices
    big, phases = blob_stack(N, nc, rng, strength=args.strength)
    truth = torch.as_tensor(big, dtype=CDT)
    H = torch.as_tensor(
        build_propagators(N, n, lam=1e-10, res=20e-9, dz=50e-6),
        dtype=CDT).to(dev)
    P = torch.as_tensor(broad_probe(n)[None], dtype=CDT).to(dev)

    stops = np.linspace(0, nc - n, 6).astype(int)
    positions = [(r, c) for r in stops for c in stops]

    # ---- the model side: eps_M of the ground truth ----------------------
    num = np.zeros(N + 1)
    den = 0.0
    I = []
    for r, c in positions:
        Op = truth[:, r:r + n, c:c + n].unsqueeze(1).to(dev)
        T = born_multislice_target(Op, P, H)
        D0, D = born_detector_orders(Op, P, H, H.conj(),
                                     min(N, max(args.orders) + 2))
        acc = D0.clone()
        num[0] += float(((acc - T).abs() ** 2).sum())
        for m in range(D.shape[0]):
            acc = acc + D[m]
            num[m + 1] += float(((acc - T).abs() ** 2).sum())
        den += float((T.abs() ** 2).sum())
        I.append((T.abs() ** 2).sum(1)[0] / (n * n))
    eps = np.sqrt(num / den)
    tot = args.strength * N
    print(f"{N} slices x {args.strength:g} rad ({tot:g} rad total): "
          "truncation error of the ground truth")
    for m in range(min(N, max(args.orders) + 2) + 1):
        print(f"  eps_{m} = {eps[m]:.3e}")

    # ---- the inversion side: order-M reconstructions of the same data ---
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(args.orders)))
    finals = {}
    recons = {}
    for col, M in zip(colors, args.orders):
        print(f"reconstructing at order M = {M} ...")
        ob, errs = reconstruct(M, I, P, positions, nc, n, H,
                               args.iters, dev,
                               update_probe=args.update_probe)
        # object fidelity against the truth: correlation of the PROJECTED
        # phase (summed over layers). At 0.3 rad total the per-slice split
        # is barely determined by the data (the free per-slice constants of
        # the null-space property), but the projection is -- and it is the
        # model-validity check the data-fit error alone cannot give.
        a = np.angle(ob.numpy()).sum(0).ravel()
        b = np.angle(big).sum(0).ravel()
        a = a - a.mean(); b = b - b.mean()
        fid = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b),
                                1e-30))
        finals[M] = (errs[-1], fid)
        recons[M] = np.angle(ob.numpy()).sum(0)
        ax.semilogy(np.arange(1, errs.size + 1), errs, color=col,
                    label=f"order M = {M} (final {errs[-1]:.3g}, "
                          f"phase corr {finals[M][1]:.3f})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("detector (Fourier) error")
    ax.set_title(f"thick weak sample ({N} slices, {tot:g} rad total): "
                 "order-M reconstructions of multislice data")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "demo_09_thick_weak_orders.png"))
    print("wrote demo_out/demo_09_thick_weak_orders.png")
    # ground truth and reconstructions: projected phase, shared scale
    panels = [("ground truth", np.angle(big).sum(0))] + [
        (f"order M = {M}", recons[M]) for M in args.orders]
    vals = np.stack([q for _, q in panels])
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    figr, axr = plt.subplots(1, len(panels),
                             figsize=(3.2 * len(panels), 3.4), dpi=150,
                             constrained_layout=True)
    for ax2, (lab, q) in zip(axr, panels):
        im = ax2.imshow(q, cmap="Greys_r", vmin=vmin, vmax=vmax)
        ax2.set_title(lab, fontsize=9)
        ax2.set_xticks([]), ax2.set_yticks([])
    figr.colorbar(im, ax=list(axr), fraction=0.025, pad=0.01,
                  label="projected phase [rad]")
    figr.suptitle("thick weak sample: truth and order-M reconstructions")
    figr.savefig(os.path.join(args.out, "demo_09_thick_weak_recon.png"))
    plt.close(figr)
    print("wrote demo_out/demo_09_thick_weak_recon.png")
    Ms = sorted(recons)
    ref = recons[Ms[-1]].ravel() - recons[Ms[-1]].mean()
    agree = {M: float((recons[M].ravel() - recons[M].mean()) @ ref
                      / max(np.linalg.norm(recons[M].ravel()
                                           - recons[M].mean())
                            * np.linalg.norm(ref), 1e-30))
             for M in Ms[:-1]}
    print("final (error, projected-phase corr vs truth):",
          {k: (float(f"{v[0]:.3g}"), float(f"{v[1]:.3f}"))
           for k, v in finals.items()})
    print(f"order-vs-order agreement (corr with M = {Ms[-1]}):",
          {k: float(f"{v:.6f}") for k, v in agree.items()},
          "-- every order fits the data to round-off and lands on the",
          "SAME object; the residual difference from the truth is the",
          "weak-contrast null space of the data (low spatial frequencies",
          "of the projected phase), identical at every order -- the",
          "discarded orders are invisible here, which is the point")


if __name__ == "__main__":
    main()
