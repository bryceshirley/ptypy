"""
Demo 1: simulate exact multislice diffraction from a synthetic slice stack.

Builds an N-slice phantom (discs and bars at different depths), a focused
probe, and the band-limited free-space propagators, then evaluates the
EXACT multislice detector field with the production function
born_multislice_target from ptypy.custom.interslice_born.isb_torch -- the
same code path the ISB engines' 'tail' coefficient fit uses. Far-field
intensities, the stack, the probe and the propagators go to
demo_out/demo_01_simulated_multislice.npz for demo_truncation_error.py.

    python demo_01_simulate_multislice.py [--slices 3] [--size 128]
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import torch

from ptypy.custom.interslice_born.isb_torch import born_multislice_target


def build_propagators(nslices, n, lam, res, dz):
    """H_{z_j} stacks, [N, 1, 1, n, n] complex128, H_0 = 1 (entrance plane),
    exactly as the engines build them."""
    f = np.fft.fftfreq(n)
    a2 = ((f[:, None] * lam / res) ** 2 + (f[None, :] * lam / res) ** 2)
    kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
    H = np.ones((nslices, 1, 1, n, n), dtype=np.complex128)
    for j in range(1, nslices):
        H[j, 0, 0] = np.exp(2j * np.pi * (j * dz / lam) * kz) * (a2 <= 1.0)
    return H


def phantom_stack(nslices, n, rng, strength=0.5):
    """Transmission slices: each depth carries its own pattern of discs and
    bars so the depth separation is visible in a reconstruction."""
    yy, xx = np.mgrid[:n, :n] / n
    slices = []
    for j in range(nslices):
        phase = np.zeros((n, n))
        for _ in range(6 + 4 * j):
            cy, cx, r = rng.random(3) * [0.8, 0.8, 0.06] + [0.1, 0.1, 0.02]
            phase -= strength * ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)
        if j % 2:  # bars on the odd slices
            phase -= strength * 0.5 * (np.sin((8 + 4 * j) * np.pi * xx) > 0.6)
        slices.append(np.exp(1j * phase))
    return np.stack(slices)


def focused_probe(n, frac=0.2):
    """A circular aperture brought near focus: aperture, quadratic phase."""
    yy, xx = np.mgrid[:n, :n] - n / 2
    r2 = (yy ** 2 + xx ** 2) / (frac * n / 2) ** 2
    ap = (r2 < 1.0).astype(complex)
    return np.fft.ifft2(np.fft.fftshift(ap * np.exp(-1j * 3 * r2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", type=int, default=3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1e-10, help="wavelength [m]")
    ap.add_argument("--res", type=float, default=20e-9, help="pixel size [m]")
    ap.add_argument("--dz", type=float, default=50e-6, help="slice gap [m]")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    n, N = args.size, args.slices
    H = build_propagators(N, n, args.lam, args.res, args.dz)

    # a small scan: the same stack seen through shifted probe windows
    stack = phantom_stack(N, n, rng)
    P = focused_probe(n)
    shifts = (rng.random((args.positions, 2)) * n / 4).astype(int)
    O = np.stack([np.stack([np.roll(s, tuple(sh), axis=(0, 1))
                            for s in stack]) for sh in shifts])  # [B, N, n, n]
    O = np.ascontiguousarray(O.transpose(1, 0, 2, 3))            # [N, B, n, n]

    O_t = torch.as_tensor(O, dtype=torch.complex128)
    P_t = torch.as_tensor(P[None], dtype=torch.complex128)        # 1 mode
    H_t = torch.as_tensor(H, dtype=torch.complex128)

    # exact multislice detector field, entrance-plane Fourier gauge; the
    # far-field intensity is its squared modulus (unimodular common factor)
    T = born_multislice_target(O_t, P_t, H_t)
    I = (T.abs() ** 2).sum(dim=1).numpy()                         # [B, n, n]

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "demo_01_demo_01_simulated_multislice.npz")
    np.savez_compressed(out, slices=O, probe=P[None], H=H, I=I,
                        lam=args.lam, res=args.res, dz=args.dz)
    print(f"wrote {out}: {args.positions} positions, {N} slices, "
          f"{n}x{n} frames, mean counts {I.mean():.3g}")


if __name__ == "__main__":
    main()
