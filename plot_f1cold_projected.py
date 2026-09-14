"""10-slice sim, probe recovered from scratch (F1cold): combined figure.

Top block: per-slice phase, reconstruction row on top of the ground-truth
row. Bottom block: projected (depth-summed) phase of truth, reconstruction,
and their difference. Built from existing run outputs only.
"""

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

S = ("/home/dnz75396/ptypy/data/process/0002_multislice/scan_000434/"
     "ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/"
     "sim_20260911_212155_slices-10_seed-2")
REC = (S + "/rec/run_20260912_083606_F1cold/"
       "rec_sim_20260911_212155_slices-10_seed-2_fasttest-F1cold_iter-0100.h5")
TRU = S + "/sim_20260911_212155_slices-10_seed-2_truth_iter-0000.h5"
OUT = "/home/dnz75396/ptypy/draft_paper/sim_F1cold_projected.png"
PX_NM = 16.87
CROP = 250


def load_stack(path, fmt):
    with h5py.File(path, "r") as f:
        n = 0
        while fmt.format(n) in f:
            n += 1
        return np.stack([f[fmt.format(j) + "/data"][0] for j in range(n)])


def slice_phase(stack):
    out = []
    for s in stack:
        m = s.mean()
        out.append(np.angle(s / (m if abs(m) > 1e-9 else 1.0)))
    return np.stack(out)


def main():
    tru = slice_phase(load_stack(TRU, "content/objects/obj_{:02d}/S00"))
    rec = slice_phase(load_stack(REC, "content/objects/Cobj_o_{}/Sscan00G00"))
    N = tru.shape[0]
    tru = tru[:, CROP:-CROP, CROP:-CROP]
    rec = rec[:, CROP:-CROP, CROP:-CROP]
    pt, pr = tru.sum(0), rec.sum(0)
    print("projected-phase correlation (cropped): %.3f"
          % np.corrcoef(pt.ravel(), pr.ravel())[0, 1])

    slo, shi = np.percentile(tru, 0.2), np.percentile(tru, 99.8)
    lo, hi = np.percentile(pt, 0.5), np.percentile(pt, 99.5)
    # symmetric difference scale set by the residual itself, not the signal,
    # so the colour differences are visible rather than washed out
    dmax = np.percentile(np.abs(pr - pt), 99.0)

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = GridSpec(3, 3 * N, figure=fig, height_ratios=[1.0, 1.0, 3.2],
                  hspace=0.15, wspace=0.06,
                  left=0.045, right=0.955, top=0.95, bottom=0.03)

    # top block: reconstruction row above the ground-truth row, per slice
    for r, (stack, rname) in enumerate(((rec, "reconstructed"),
                                        (tru, "truth"))):
        for j in range(N):
            ax = fig.add_subplot(gs[r, 3 * j:3 * j + 3])
            ax.imshow(stack[j], cmap="gray", vmin=slo, vmax=shi)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title("slice %d" % j, fontsize=8, pad=2)
            if j == 0:
                ax.set_ylabel(rname, fontsize=9)

    # bottom block: projected trio; one labelled colourbar (difference),
    # the two grey panels share the slice rows' convention
    for k, (im, t, cm, v0, v1, cbar) in enumerate((
            (pt, "truth, projected phase", "gray", lo, hi, False),
            (pr, "reconstruction, projected phase", "gray", lo, hi, False),
            (pr - pt, "difference", "RdBu_r", -dmax, dmax, True))):
        ax = fig.add_subplot(gs[2, 10 * k:10 * k + 10])
        h = ax.imshow(im, cmap=cm, vmin=v0, vmax=v1)
        ax.set_axis_off()
        ax.set_title(t, fontsize=10)
        if cbar:
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03, label="rad")
        if k == 0:
            px = 2000.0 / PX_NM
            ny, nx = im.shape
            ax.plot([nx * 0.06, nx * 0.06 + px], [ny * 0.93] * 2, "w-", lw=3)
            ax.text(nx * 0.06 + px / 2, ny * 0.89, r"2 $\mu$m", color="w",
                    ha="center", fontsize=9)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved", OUT, "| grey scale %.2f..%.2f rad, diff +/- %.2f rad"
          % (lo, hi, dmax))


if __name__ == "__main__":
    main()
