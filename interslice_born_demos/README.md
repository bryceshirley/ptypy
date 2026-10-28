# Inter-slice Born demos

Runnable scripts built on the production package
`ptypy.custom.interslice_born`. Run from this directory with the
`ptypy_full` environment, in order:

1. `demo_01_simulate_multislice.py` — synthesise an N-slice phantom and
   its exact multislice diffraction (`born_multislice_target`); writes
   `demo_out/simulated_multislice.npz` for demo 02.
2. `demo_02_truncation_error.py` — eps_M per truncation order on that
   stack, plain and with the data-free coefficient fits; round-off at
   M = N (nilpotent termination).
3. `demo_03_truncation_strong.py` — the strong/thick regime where the
   plain series error grows before terminating and the coefficient fits
   (tail, exit-residual, volume-remainder) pay off; includes the tangent
   series' linearisation floor; paper figure style, single precision,
   intensity error.
4. `demo_04_sparse_first_born.py` — the sparse strong specimen on which
   the chord first Born model is EXACT (overlap property): the native
   10-rad campaign simulation (truth, probe and the 670-position scan
   taken as-is), chord-vs-tangent fixed-step reconstructions matching
   the original torch-engine campaign figure; the scan is a uniform
   raster at the campaign scan's pitch over a vacuum-padded canvas, so
   coverage is complete (corner particles included).
6. `demo_06_backend_benchmark.py` — forward + adjoint wall time of one
   per-position update vs the slice count N at 32x32 frames (the
   launch-latency regime), eager and CUDA-graph compiled: the CuPy Born
   core at M = 1..5 against ThreePIE_cupy's sequential multislice
   kernel chain, both arms on ptypy's accelerate FFT machinery, after
   figs-p1/cost_ptyrad_1x1_mutliple_scattering.py.
8. `demo_08_M_vs_N.py` — the paper's M*-vs-N study (Born order needed
   for a tolerance is set by contrast, not slicing), with the
   interslice_born cupy primitives, after figs-p1/paper1_M_vs_N.py.
9. `demo_09_thick_weak.py` — thick, weakly scattering sample with data
   from exact multislice: eps_1/eps_2 already negligible, and order-1/2
   reconstructions match the high-order one in data fit AND projected
   phase.
10. `demo_10_nanomax_engines.py` — the real NanoMAX scan: ISBCupy at
    order 1 and 2 (CUDA graphs) vs upstream ThreePIE_cupy, error decay
    and wall time per iteration.

## Reference engines

`issdiagnostics_torch.py` and `theepiesimscan_torch.py` are the original
read-only diagnostic engine (`ISSDiagnostics`) and simulators
(`MultisliceSimulator`, `SparseStrongSimulator`), kept here purely as
validation references — no demo imports them, and the demos reproduce
their results from production code alone (demo 04 prints its eps table
next to the stored `sim/.../diag/results.json` of the sparse campaign:
same structure, chord eps_1 at or below the campaign's 7.9e-4 with exact
round-off on fully cone-disjoint scans). The original drivers live in
`~/ptypy/ISS-work/`.
