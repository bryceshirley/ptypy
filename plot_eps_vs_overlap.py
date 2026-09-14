"""eps_1 per scan position against the measured inter-slice overlap.

Replaces the Phi_tot scatter (reviewer part 1c): the overlap integral is what
the support argument of Sec. 2.1 predicts. Reads the diagnostics results.npz
(no engine run).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = ("/home/dnz75396/ptypy/data/process/0002_multislice/scan_000434/"
     "ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/diagnostics_v4/results.npz")
z = np.load(D)
ov = z["gauge_none_overlap_per_pos"]
e1 = z["gauge_none_eps1_per_pos"]
e1t = z["gauge_none_eps1lin_per_pos"]
r = np.corrcoef(np.log(ov), np.log(e1))[0, 1]
print("positions:", ov.shape, "| corr(log ov, log eps1) = %.3f" % r,
      "| corr(lin) = %.3f" % np.corrcoef(ov, e1)[0, 1])
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.loglog(ov, e1, ".", ms=3.5, alpha=0.45, color="tab:blue",
          label=r"chord $\Delta O = O - 1$")
ax.loglog(ov, e1t, ".", ms=3.5, alpha=0.45, color="crimson",
          label=r"tangent $\Delta O = is$")
ax.set_xlabel("probe-weighted inter-slice overlap")
ax.set_ylabel(r"$\epsilon_1$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("/home/dnz75396/ptypy/draft_paper/eps1_vs_overlap.png", dpi=200)
print("saved eps1_vs_overlap.png")
