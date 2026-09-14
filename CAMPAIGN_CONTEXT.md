# Thick-sample ptychography campaign — handoff context (2026-09-11)

Read this to continue the campaign in a fresh session. Deeper background:
`fastfirstborn_maths.md` (derivations, failure analysis, results gallery
with figures) and the memory files under
`~/.claude/projects/-home-dnz75396-ptypy/memory/`.

## Environment
- Repo `/home/dnz75396/ptypy`, branch `butterfly-firstborn`, all work
  uncommitted (user reviews; do not commit/push unasked).
- Python: `/home/dnz75396/miniconda3/envs/ptypy_full/bin/python`
  (torch + CUDA, 16 GB GPU). CPU engine runs also work in this env.
- WebSearch tool is blocked by Vertex org policy; `curl` and WebFetch
  work (arXiv API, Crossref, Semantic Scholar all reachable).

## Data & directory conventions (all enforced by the runners)
- Base: `data/process/0002_multislice/scan_000434/
  ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/`
- `sim/` contains ONLY per-sim folders `sim_<stamp>_slices-N_seed-s/`
  (ptyd, truth h5, figures, terminal.log, sim-script copy) each with
  `rec/` containing ONLY per-run folders `run_<stamp>_<label>/`
  (script copy, ptyr, fslices h5, report pngs, iteration_progress dir
  with live terminal.log).
- Sims: seed-0 = main 10-slice, 1.58 mm spacing (`sim_20260910_111021...`);
  seed-1 = K100 100 um spacing; seed-2 = L100het heterogeneous 100 um.
  8 keV, 512 px, 75 um det pixels, 4.18 m, obj pixel 16.9 nm, 670 views.
- Every runner auto-generates a report (`sim_report.py`): slice gallery,
  truth-vs-rec panel, error history, measured-vs-predicted frames via
  `born_error.py`. Scoring = `threepie_fix_tests/score_layers.py`
  ("diag/proj corr", phase correlation on truth footprints).

## Engines (ptypy/custom/)
- `firstborntorch.py` FirstBornTorch — affine Born model, fixed step.
- `fastfirstborntorch.py` FastFirstBornTorch — exact quartic line search
  (cubic root, `solve_step_cubic`), ls_damp=0.5, cross-term exact via
  F <- F + aD. Best config: batch 1, probe every view, denom_reg 0.05.
  object_denom: `local` for well-matched multislice (seed-0 10-slice), but
  `max` is REQUIRED for tight high-NA probes or coarse-axial recs — `local`
  there collapses the probe to a central hot pixel with runaway peak/mean
  (2026-09-11 battery: local 1268@it2->3719@it8 vs truth 595; max 623->232
  bounded). New params: `probe_denom` (default max, decoupled), `probe_
  fourier_frac` (aperture bandwidth limit, default 0 — NOT the spike fix),
  per-iter probe peak/mean watchdog in the log. `firstborntorch.py`:
  plot_interval default 3 and now honored (was hardcoded %5); probe
  freezing off (probe_converged_rtol default 0). Runners
  run_fastborn_hna.py & run_fastborn_100um_het.py take a rec_slices argv
  (coarser axial sampling, same total thickness); het runner also takes a
  plot_interval argv.
- `fasterfirstborntorch.py` — k=2 plane search; validated NEGATIVE
  result (momentum direction useless at batch 1; a2 clipped [0,0.9]).
- `fastthreepietorch.py` FastThreePIETorch — full multislice cascade,
  exact per-slice line searches + exact field maintenance down the
  sweep; REQUIRES `object_denom='max'` (local poisons it); float64
  quartic coefficients (float32 overflows ~1e40 on first sweeps).
- `run_hybrid_test.py` — FastFirstBorn 15 it -> FastThreePIE via
  `init_slices` + shared probe storage. `run_hybrid2_test.py` — same
  but standard ThreePIETorch stage B.

## Results matrix (seed-0 sim, batch 1, probe every view, 50 it unless noted)
| run | config | diag/proj | final err | s/it |
|---|---|---|---|---|
| N50b | FastFirstBorn local reg0.05 | 0.986/0.973 | 1.23 (plateau = Born model-mismatch floor) | 16.9 |
| M50 | FastThreePIE max | 0.995/0.988 | 0.0435 | 26.9 |
| H1 | Born15 -> Fast3PIE35 | **0.996/0.996** | **0.0265** | — best; no swap penalty |
| H2 | Born15 -> standard 3PIE35 | 0.982/0.973 | 0.914 | fixed step spikes at swap, poor corrector |
| ref100 | ThreePIETorch max h=1, 100 it | 0.268/0.005 | stuck (patience fails) | 13 |
| step0p3 / step2p0 | h=0.3 / h=2.0, 15 it | 0.318/0.005 / **0.950/0.898** | under-stepped, not un-steppable | |
| Jmax | FastThreePIE max, 15 it | 0.981/0.963 | | 27.7 |
| gpu3pie-ref3 vs CPU | parity verified (1-2% every iter) | | | 12.9 vs 272 |
Figures: `run_20260911_124605_comparison-4way-v2/convergence_4way.png`
(in the md gallery). 100 um tests: K100 0.599/0.721, L100het 0.575/0.704
=> effective axial resolution ~150-200 um; truth layers verified
statistically independent (off-diag exactly 0).

## Standing recommendation
Hybrid: FastFirstBorn ~15 it, then FastThreePIE(`max`) warm-started via
`init_slices`. On real data expect Born's plateau and multislice's edge
to be less clean (sims favour the exact-multislice model).

## Literature (checked 2026-09-11 via arXiv/Crossref/S2 APIs)
- Thibault & Guizar-Sicairos, NJP 14, 063004 (2012): EXACT polynomial
  line search for ptychography ML (degree-8 for Gaussian, closed-form
  roots) — SINGLE-SLICE only. Closest single-slice prior art; verified
  from full text.
- Odstrcil et al., Opt. Express 26, 3108 (2018) LSQML: computed optimal
  steps via LINEARIZED least-squares subproblems (not exact polynomial);
  multislice ML in Tsai et al., Opt. Express 24, 29089 (2016) and
  PtychoShelves (Wakonig 2020). Full text NOT yet verified (Optica pages
  JS-blocked) — before claiming novelty, check Tsai 2016 Sec. 2 and the
  PtychoShelves/fold_slice MS-LSQML code for whether their per-slice
  steps are exact or linearized, and whether they maintain fields
  exactly through the sweep. Also check Van den Broek & Koch (PRL 2012,
  inverse dynamical electron scattering) for multislice CG line-search
  details, and AD-based multislice (Adorym, Du et al. 2020).
- Apparent novelty of FastThreePIE: exact quartic per-slice steps on the
  true intensity cost + exact elementwise field maintenance through the
  sequential sweep + the max-vs-local direction finding + the Born->
  multislice hybrid. The Born engine with fully closed-form steps also
  appears new, but the Born model itself for ptychography predates us.

## High-NA / coarse-slice thread (Darren feedback: 1.5mm too thick, need
## higher convergence angle for sub-100nm res / ~100um probing)
- New sim seed-3: `sim_20260911_181003_slices-30_seed-3` = 30 layers x
  50 um dz = 1.45 mm, probe_na_scale=2.0 (~384 nm rms focal spot, 2-mode
  probe peak/mean ~595). Its truth probe PNG (`..._truth_probe.png`) is in
  the sim dir. run_fastborn_hna.py / run_multislice_sim_hna.py drive it
  (aperture 300nm to match the 2x-NA probe).
- Reconstructing seed-3 with 15 coarse slices works ONLY with
  object_denom=max (probe bounded 623->232, depth-corr shows a clear
  diagonal band = axial localization working; upstream slices still weak
  at 15 it).
- Coarse-slice test on seed-2 (10-slice, 900 um, NA=1) with 5 slices
  (180 um each), object_denom=max, 40 it: probe stable (peak/mean ~220),
  depth-corr diagonal (rec slice i <-> truth ~2i), proj corr 0.619,
  middle slice weakest + some crosstalk. 100 it -> proj 0.670.
- "slightly more convergence" test: regenerated seed-2 sim with
  probe_na_scale=1.5 (`sim_20260911_201429_slices-10_seed-2`, same layers/
  seed for clean A/B), reconstructed 5 slices object_denom=max, aperture
  150 nm, 100 it. NEGATIVE result: proj corr 0.565 (WORSE than NA=1's
  0.670), diag=nan. Front rec slice 0 became a diffuse low-frequency SINK
  (no particles) while slices 1-2,4 held structure. Cause: focus at -700um
  sits near the sample ENTRANCE, so at higher NA the front plane lands on
  the tight focal waist = worst position-overlap = collapses to a junk sink.
  Higher NA alone is not the fix; the focus must be CENTERED in the volume
  (deeper defocus ~-1150um) so no plane sits at the poorly-overlapped waist.
  run_multislice_sim_100um_het.py now has PROBE_NA_SCALE (argv1, default
  1.5; NOTE it collides with legacy scannr argv1 -- run with no args) and
  run_fastborn_100um_het.py now takes aperture_nm as argv9.

## Sparse strong chord-vs-tangent demo (2026-09-13)
- New engine-side sample: `SparseStrongSimulator` (multislicesim.py) — 3
  layers, dz 1.58 mm, particles rescaled O -> O**s (bg-referenced to exact
  phase 0), placement enforces BEAM-CONE disjointness from measured probe
  footprints per plane + verifies it (0/670 positions hit two layers).
  New params: phase_scale, max_phase_rad (no-wrap cap), cone_margin_px,
  footprint_quantile. Layers gallery plots UNWRAPPED phase (skimage).
- Sims: `sim_20260913_134409_slices-3_seed-7` (scale 6, phase to -5.5 rad,
  wraps) and `sim_20260913_143227_...` (capped 2.93 rad, no wrap).
- Runners: run_sparse_strong_sim.py, run_tangent_vs_chord.py
  (label kind niter batch [sim_dir]; truth probe injected+frozen, seed 123),
  run_borndiag_sparse.py (FirstBornDiagnostics on a sim truth; pair_mode
  'best' REQUIRED — 'same-iteration' crashes on the probe-less pool).
- Detector-plane eps (diag/ in each sim dir): chord eps1 1.5e-3 (wrapped
  sample) / 1.1e-3 (capped), ->4e-16 at M=N; tangent eps1 0.34 / 0.19 with
  a floor no order removes — WORSE than eps0 (no model). Sparsity makes the
  chord exact; the tangent keeps its O(s^2) per-slice error.
- KEY INSIGHT: with an unconstrained complex potential the tangent ENGINE
  is the chord engine in g-space (the -i chain rule and +i model cancel),
  so error histories are IDENTICAL by construction; only the readout
  differs (chord 1+g vs tangent exp(g)). Failure shows in object space:
  capped sample, 3 it: chord proj corr 0.961 vs tangent 0.774; particle
  phase depth truth -2.9 -> chord -2.5, tangent -1.0 (sin-compressed).
- Scoring pitfall: score_layers phase corr uses WRAPPED angle — useless on
  >pi samples (both engines score ~0 at 30 it). Use complex DeltaO corr
  (/tmp lost; recreate: vdot(g_t, g_r)/norms on truth footprints).
- Fixed-step churn: chord at 3 it |DeltaO corr| 0.999/layer; at 30 it
  degrades to 0.86 mean, data err rises 0.50->0.66. Early stopping or
  FastFirstBorn line search for long runs.
- FastFirstBorn retest (fast30s, ls_damp 0.5, denom_reg 0.05, GPU): CONVERGES
  where fixed step churned — Fourier err monotone to 0.174 @ it 9 (3x below
  fixed-step's 0.50 floor), then mild +0.0045/it creep to 0.262 @ 30 while
  the OBJECT stays intact: |DeltaO corr| 0.996 mean at it 30 (vs 0.856
  fixed-step), per-layer magnitude 0.85-1.08x truth. run_tangent_vs_chord.py
  now takes kind 'fast'.
- Figures: <sim>/chord_vs_tangent_phase_profiles.png (unwrapped),
  <sim>/diag/01_truncation_error.png, per-run truth_vs_rec/report pngs.
- CORRECTED sample `sim_20260913_164331` (2026-09-13 pm): per-position chord
  eps floor of the first samples (1e-4..1e-2) traced to (a) speckle from the
  power rescale scattering beyond the probe NA and (b) footprint-only gap
  criterion. Fixes in SparseStrongSimulator: smooth_px (log-domain gaussian
  + soft edge taper, default 2.5), shear_na_scale (cross-layer gap now
  r+r+max(R_j+R_k, 2*NA_eff*dz)+margin), hardest-layer-first placement (a
  skipped late copy can no longer empty a layer). Result: 1 particle/layer,
  5.10 rad smooth, min slack 6 px; chord eps1 4.1e-4 (median 7.5e-5/pos,
  empty-pos floor ~1e-5 = probe tail coupling, the finite-field limit) vs
  tangent 0.194 — 470x. 03_frames.png now has a 'first Born (tangent)'
  column (borndiagnostics._fig_frames); on a particle-hitting frame the
  tangent floods the detector with spurious scatter while chord matches
  multislice. run_borndiag_sparse.py argv2 = n_positions quick mode writes
  to <sim>/diag_quick/. 04_radial_profiles gained chord+tangent curves,
  then (user request, 2026-09-13 evening) radial profiles, the
  01_truncation_error figure and ALL blocked-truncation computations/keys
  (eps_block*) were REMOVED from FirstBornDiagnostics; [blocks] param is
  now accepted-but-ignored, _blocked/_radial methods remain unused.
  Old diag/ dirs still hold the retired figures.

## Real-potential (constrained tangent) experiment (2026-09-13 late)
- Sample `sim_20260913_203229`: pure-phase variant of the corrected sparse
  sample (SparseStrongSimulator new param pure_phase=True -> |O|=1.000000
  exactly, phase to 5.10 rad, same verified sparse layout).
- run_tangent_vs_chord.py now sets potential_real=True for kind 'tangent'
  (real s = honest weak-phase model; NO LONGER the chord-in-disguise).
- 5 it, batch 1, GPU, truth probe frozen: error histories now DIVERGE as
  predicted. Chord Fourier err 5.6->1.9 (falling); tangent stalls flat at
  ~16.2 from iteration 1 (8.5x worse) — the real-s constraint locks the
  scattered wave in quadrature and the model cannot fit the data at all.
  Objects: chord |DeltaO corr| 0.90/1.00/1.00 (phase depth reaches truth
  -5.0 rad on layers 1-2, layer 0 still filling in); tangent 0.55 mean,
  phase compressed to -0.4 mean / -2.1 min vs truth -2.7 / -5.1.
- Runs: rec/run_*_chord5pp, run_*_tan5pp; profiles figure in the sim dir.
- LINE-SEARCH pair (30 it, GPU): new engine FastFirstBornTangentTorch in
  fastfirstborntorch.py (tangent + exact quartic line search; new
  _dir_to_model hook maps the parameter direction to the model direction
  i*d — field stays affine along real-s directions so the closed-form step
  is still exact, now on the constraint manifold). run_tangent_vs_chord.py
  kinds 'fast'/'fasttangent'. Results on the pure-phase sample:
  fast chord 6.54 -> 0.28 MONOTONE over 30 it (line search cures the
  fixed-step churn; |DeltaO corr| 0.995 mean, magnitude 0.89-1.02x truth);
  fasttangent (s real) converges to ITS OWN optimum by it ~4 and pins flat
  at 16.0 for 26 more iterations = model floor, 57x the chord, optimizer
  exonerated. Object corr 0.60 mean, magnitude 0.38-0.48x.

## Per-slice line search + truth tracking (2026-09-13 night)
- FastFirstBornTorch new param per_slice_ls (default False; runner sets True
  for fast/fasttangent): sequential exact quartic step PER SLICE with exact
  field maintenance (F <- F + a_j D_j), because a single global step is
  dominated by the best-covered (deepest) layer and under-converges the
  entrance layers. On the wrapped pure-phase sample layer 2 is essentially
  converged after ONE iteration (rel dO 0.032).
- FirstBornTorch new param truth_h5 (runner sets it): per-iteration
  diagnostic 'truth error' = rel ||dO_rec - dO_truth||/||dO_truth|| per
  layer + rms phase error angle(O_rec conj(O_t)) (equivalent to comparing
  against the UNWRAPPED truth wherever the mismatch < pi). Motivation: the
  aggregate detector error is NOT monotone at batch 1 even with exact line
  searches (each step exactly fits ONE view and can unfit the other 669) —
  user question answered; convergence must be judged against truth (sims)
  or full-batch steps implemented (the campaign 'endgame' idea).
- Truth tracking exposed the real optimum ~ iteration 10-12 (mean rms phase
  err 0.39 rad), then batch-1 ping-pong degrades mostly as per-layer complex
  scale/phase drift (corr stays 0.99+ while offsets grow). Composite uses
  images from 12-it runs + full 30-it truth-error curves.
- Best chord result (fast12tt, per-slice LS, 12 it): min phase
  -4.44/-5.01/-5.03 vs truth -5.08/-5.06/-4.94; dO corr 0.993/1.000/0.998.
  Tangent (real s) pinned at rms phase err ~2.0 rad, corr ~0.57.
- Final deliverable figure: <sim 203229>/chord_vs_tangent_final.png
  (truth/chord/tangent unwrapped-phase grid, scale set to the tangent
  artifacts, + truth-error convergence curves). Detector-side figures:
  diag/02_phi_scatter.png (x = probe-weighted inter-slice overlap now, the
  paper's axis) and diag_quick/03_frames.png (multislice|chord|tangent,
  colorbar outside the grid).
- NOTE fairness: real-s tangent vs UNCONSTRAINED chord; an |O|=1 chord
  constraint for full symmetry was discussed and deferred (user: ok).
- FINAL (user direction, 2026-09-13 latest): truth-tracking code REMOVED
  from firstborntorch.py again (user wants detector error only; truth_h5
  param + _track_truth are gone); deliverable runs redone at 10 iterations
  (user had deleted rec/): fast10f / ftan10f on sim 203229 (per_slice_ls
  on). Chord detector err 5.9->0.55 monotone in 10 it, phase depth
  -4.57/-5.02/-5.02 vs truth -5.08/-5.06/-4.94, dO corr 0.982/1.000/0.998;
  tangent flat 18.2 (33x), depth capped ~-2. Final figure
  <sim 203229>/chord_vs_tangent_final.png = truth/chord/tangent unwrapped
  grid + detector-error curves. plot script parses 'Errors :: Fourier'.
- Physics note (user Q): forcing s real does NOT force |1+g|=1 — the
  tangent model transmission |1+is| = sqrt(1+s^2) leaves the unit circle
  at O(s^2); demanding |1+g|=1 for a real potential IS the chord
  (O = exp(is)). The constraint sets coincide only to first order.
- Review fixes applied to the write-up (external review, ledger 44e04ab):
  |exp(is)-(1+is)| at s=3 corrected 2.4 -> 3.5; footprint mechanism restated
  as ~(247/150)^2 = 2.7x more scan positions hitting the deep particle (not
  "3x probe power"); NA_eff 1.8 mrad vs paper's 1.20 mrad flagged as
  different definitions; frames caption now "far-field (detector)
  intensities"; placement explicitly closes the old no-overlap-in-projection
  approximation; integration notes list the main.tex collisions (line ~1681
  "joint all-slice step", ~2614 "not implemented" — the layer-wise search
  now IS implemented) + naming ruling (line-search first Born, never Fast*).
- JOINT PROBE+OBJECT confirmation (2026-09-14): run_tangent_vs_chord.py
  argv7... argv6 probe mode 'truth'|'free'; 'free' reconstructs the probe
  from the aperture+defocus init (probe_update_start=0). 50 it, per-slice
  LS, batch 1, GPU, sim 203229: CHORD err 32 -> 0.023 MONOTONE (free probe
  absorbs init mismatch — better than every frozen-probe control), object
  dO corr 0.998/0.996/0.997, phases -4.9/-5.1/-5.0, probe intensity corr
  0.9973 / subspace 0.968 / power gauge 0.78 (object 1.1x compensates).
  TANGENT (s real): err hangs 12.1 (528x), object 0.708, probe corrupted
  to 0.20x power / subspace 0.65 — model error leaks into the probe.
  Headline figure + write-up updated to the free-probe 50-it runs
  (fast50fp/ftan50fp).
- Write-up: ~/draft_paper/SPARSE_STRONG_RESULTS.md with the three figures
  copied as sparse_chord_vs_tangent_recon.png, sparse_eps1_vs_overlap.png
  (full-scan diag 02), sparse_forward_frames.png (diag_quick 03, fixed
  layout). Quoted numbers from sim 203229 full diag: chord eps1 7.9e-4
  (median 1.4e-4/pos), eps2 1.1e-6, eps3 4e-16; tangent 0.217 all orders
  (> eps0 0.148); recon: chord 5.9->0.55 in 10 it, tangent flat 18.2.
  Includes the layer-wise line-search explanation.

## Open threads
- User launched `run_lsqml_test.py L15` (their own new runner, LSQML
  baseline, 15 it) — results not yet reviewed; compare to matrix above.
- Draft reply to Darren Batey (8 keV, 16.9 nm px, axial res 150-200 um,
  in-situ window cleaning use case) — drafted in chat, not sent.
- fastfirstborn_maths.md is the paper skeleton: standard Born, Fast
  Born, where standard multislice fails (step-scan table), FastThreePIE,
  gallery. Section numbering: 1,2,3(failure),4(Fast3PIE),5(gallery).
- Possible next steps discussed: full-batch exact-step "endgame" with
  Krylov directions; adaptive ls_damp via TR ratio test; real-data
  (butterfly I08-1) application per data_butterfly/context.md.
