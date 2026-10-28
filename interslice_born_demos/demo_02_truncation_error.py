"""
Demo 2: the truncation error of the inter-slice Born series, order by
order, with and without fitted readout coefficients.

Loads the stack simulated by demo_01_simulate_multislice.py and uses only
production functions: born_detector_orders for the per-order detector
fields, born_multislice_target for the exact reference, and the two
data-free coefficient fits from ptypy.custom.interslice_born.base
(born_qr_coeffs, born_exit_residual_coeffs). Prints the eps_M table --
eps_M = ||psi_ms - psi^(<=M)|| / ||psi_ms|| -- and writes
demo_out/demo_02_truncation_error.png. At M = N every number returns to round-off
(nilpotent termination), which is the end-to-end check of the pipeline.

    python demo_02_truncation_error.py [demo_out/demo_01_simulated_multislice.npz]
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from matplotlib import pyplot as plt

from ptypy.custom.interslice_born.base import (
    born_exit_residual_coeffs,
    born_qr_coeffs,
)
from ptypy.custom.interslice_born.isb_torch import (
    born_detector_orders,
    born_multislice_target,
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 \
        else "demo_out/demo_01_simulated_multislice.npz"
    d = np.load(path)
    O = torch.as_tensor(d["slices"], dtype=torch.complex128)
    P = torch.as_tensor(d["probe"], dtype=torch.complex128)
    H = torch.as_tensor(d["H"], dtype=torch.complex128)
    N = O.shape[0]

    D0, D = born_detector_orders(O, P, H, H.conj(), N)
    T = born_multislice_target(O, P, H)
    t_norm = float(T.abs().pow(2).sum().sqrt())

    def eps(field):
        return float((field - T).abs().pow(2).sum().sqrt()) / t_norm

    print(f"{N} slices, {O.shape[1]} positions -- "
          "eps_M = ||psi_ms - psi^(<=M)|| / ||psi_ms||")
    print(f"{'M':>3} {'plain':>12} {'tail fit':>12} {'residual fit':>14}")
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
            e_tail, e_res = eps(w(c_t)), eps(w(c_r))
        else:
            e_tail = e_res = e_plain
        rows.append((M, e_plain, e_tail, e_res))
        print(f"{M:>3} {e_plain:>12.3e} {e_tail:>12.3e} {e_res:>14.3e}")

    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)
    for k, (label, mk) in enumerate(
            [("plain truncation", "o"), ("tail-fit coefficients", "s"),
             ("residual-fit coefficients", "^")]):
        ax.semilogy(rows[:, 0], np.maximum(rows[:, k + 1], 1e-17),
                    marker=mk, label=label)
    ax.set_xlabel("truncation order M")
    ax.set_ylabel(r"$\epsilon_M$")
    ax.set_title("Inter-slice Born truncation error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("demo_out/demo_02_truncation_error.png")
    print("wrote demo_out/demo_02_truncation_error.png")


if __name__ == "__main__":
    main()
