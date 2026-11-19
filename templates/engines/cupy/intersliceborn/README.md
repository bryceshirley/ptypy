# Inter-slice Born templates

Runnable scripts for the inter-slice Born engine, `IntersliceBorn`, which
lives in `ptypy.custom.intersliceborn_cupy` with its fused CUDA core in
`ptypy.custom.intersliceborn_utils`.

All three demos reconstruct NanoMAX scan 434 — the prepared two-slice
`.ptyd` of the paper campaign — so demo 01 owns the shared setup
(`base_pars`, `error_history`) and the other two import it. Run them
from this directory.

Each demo owns one folder under `results/` — `results/demo_01`,
`results/demo_02`, `results/demo_03` — holding its reconstructions, its
figures and its logs. A relative `--out` resolves against this template
directory rather than the shell's working directory, so the demos can be
launched from anywhere.

1. `demo_01_nanomax_engines.py` — the real NanoMAX scan with three GPU
   engines on identical data, illumination and seeds: `IntersliceBorn` at
   order 1 (inter-slice first Born), `IntersliceBorn` at order 2 (exact
   multislice through the parallel series, two slices), and upstream
   `ThreePIE_cupy` (sequential multislice). Reports wall time per
   iteration and overlays the error histories, normalised to each
   engine's first iteration because the engines log different
   functionals.
2. `demo_02_centred_nanomax.py` — the `[background]` parameter: two
   `IntersliceBorn` reconstructions identical except for the background a_j
   the series is expanded about, `vacuum` (a_j = 1) against `mean`
   (a_j = <O_j>, with the gauge A = prod_j a_j restored at the
   detector). With two slices, order 1 is the only truncation that
   exists.
3. `demo_03_centred_vs_multislice.py` — centred first Born against
   sequential multislice, one arm per invocation so the two can share
   the box on separate GPUs (`--arm isb` / `--arm threepie` /
   `--arm plot`). Asks whether centred first Born actually lands on the
   multislice reconstruction.

`run_all_demos.sh` runs the lot on two free GPUs, writing each step's
log into that demo's own folder. Edit the two GPU UUIDs at the top first — pin by UUID, never by
index, because the index ordering is not stable.

## Plotting

`plotting.py` is the shared figure code: a port of
`data/nanomax/notebooks/Figure_reconstruction_result.ipynb` — crop 512
off every border, unwrap, subtract the offset measured in a vacuum
window, Greys_r on [-1.2, 0.1] rad, micrometre extent, red zoom box and
scale bars. Keeping the notebook's constants means any demo's output
can be laid next to the published figure and compared directly.

It is also runnable, and renders every reconstruction in a run
directory:

    python plotting.py results/demo_01

- `demo_03_plot_slices.py` — the same figure for demo 03's output, next
  to the published 500-iteration multislice reference, with the
  relative phase difference between them printed.
- `plot_error_curve.py` — convergence curve for a run directory; reads
  `err_isb.npy` and writes `error_isb.png` next to it.