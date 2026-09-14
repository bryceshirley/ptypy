"""30-slice (2.9 mm) sim: truth vs FastFirstBorn reconstruction figure.

Built from the existing run outputs only (no reconstruction is run):
projected (depth-summed) phase of the truth stack and of the recovered stack,
plus their difference, for the paper's Sec on simulations.
"""

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = ("/home/dnz75396/ptypy/data/process/0002_multislice/scan_000434/"
     "ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/"
     "sim_20260912_104339_slices-30_seed-2")
REC = (S + "/rec/run_20260912_104616_F1_3mm5/"
       "rec_sim_20260912_104339_slices-30_seed-2_fasttest-F1_3mm5_iter-0005.h5")
TRU = S + "/sim_20260912_104339_slices-30_seed-2_truth_iter-0000.h5"
OUT = "/home/dnz75396/ptypy/draft_paper/sim_3mm_truth_vs_rec.png"
PX_NM = 16.87


def load_stack(path, fmt):
    with h5py.File(path, "r") as f:
        n = 0
        while fmt.format(n) in f:
            n += 1
        return np.stack([f[fmt.format(j) + "/data"][0] for j in range(n)])


def proj_phase(stack):
    # per-slice mean-gauge, then depth-summed phase (the gauge-invariant view)
    ph = np.zeros(stack.shape[1:], dtype=float)
    for s in stack:
        m = s.mean()
        ph += np.angle(s / (m if abs(m) > 1e-9 else 1.0))
    return ph


def main():
    tru = load_stack(TRU, "content/objects/obj_{:02d}/S00")
    rec = load_stack(REC, "content/objects/Cobj_o_{}/Sscan00G00")
    print("truth", tru.shape, "| rec", rec.shape)
    pt, pr = proj_phase(tru), proj_phase(rec)
    c = 250                       # trim the un-illuminated border
    pt, pr = pt[c:-c, c:-c], pr[c:-c, c:-c]
    corr = np.corrcoef(pt.ravel(), pr.ravel())[0, 1]
    print("projected-phase correlation (cropped): %.3f" % corr)
    lo, hi = np.percentile(pt, 0.5), np.percentile(pt, 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    for ax, im, t in ((axes[0], pt, "truth, projected phase"),
                      (axes[1], pr, "reconstruction, projected phase"),
                      (axes[2], pr - pt, "difference")):
        v = ax.imshow(im, cmap="gray" if t != "difference" else "RdBu_r",
                      vmin=lo if t != "difference" else -0.5 * (hi - lo),
                      vmax=hi if t != "difference" else 0.5 * (hi - lo))
        ax.set_axis_off()
        ax.set_title(t, fontsize=10)
        fig.colorbar(v, ax=ax, fraction=0.046, label="rad")
    # scale bar: 2 um
    px = 2000.0 / PX_NM
    ny, nx = pt.shape
    axes[0].plot([nx * 0.06, nx * 0.06 + px], [ny * 0.93] * 2, "w-", lw=3)
    axes[0].text(nx * 0.06 + px / 2, ny * 0.89, r"2 $\mu$m", color="w",
                 ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
