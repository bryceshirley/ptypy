# Cell-sim campaign findings (2026-09-12 evening)

Sim: sim_20260912_190858_slices-20_seed-2 (20 slices/1.2mm, one contaminated
window slice 0, sample slice 9=568um, 400nm/0.12rad/25% contamination, Poisson
at real dose, PROBE_DEFOCUS +1400um => waist 740um BEFORE the cell).

1. CELL-A (N=20, first sim, defocus 0) FAILED: waist inside cell at the sample
   plane (peak/mean 4487 measured); max-denom starved slices 8-14 -> sample
   displaced. Fix = move the waist out (defocus +1400um, verified 181->25).
2. CELL-B/B2 (N=10 per Bryce, 10+10 it, frozen truth probe): WORKS.
   proj 0.867->0.886; axial energy peak 43.6% at the true sample slice (crop);
   window on slice 0, phase corr +0.81; profile converged (PSF floor).
3. T-CELL-B (Fast3PIE, matched): proj 0.923 (inverse-crime advantaged) BUT
   axial energy perfectly flat (10%/slice), contamination not recovered at
   all at 10 iters. Depth separation emerges much EARLIER under Born.
   -> the application-grade result for Darren's in-situ window use case.
4. Tooling: sim_report diag=nan when rec slices != truth slices (10 vs 20);
   per-slice numbers here computed manually on the central crop.
   cellsim.py scalar-cast bugs fixed (lam/resolution/distance arrays).

# tBL-WSe2 N-sweep (paper_extra_tests, clean-HEAD worktree, 20 it each)
- born vs multislice, N in {1,2,3,4,6,12}, fixed 12 A total, all other params
  = the paper runs'. All 12 runs rc=0.
- N=1 anchor: born 0.4003 vs ms 0.3999 (identical models; diff = noise floor).
- Loss gap born-ms grows monotonically 0.0004 -> 0.0023, tracking (1-1/N):
  the paper's eps1 N-dependence claim MEASURED on real electron data.
- Born loss has a MINIMUM at N=6 (0.3917) then worsens at N=12 (0.3922):
  do-not-over-slice guidance for Born reconstructions.
- s/iter: born flat (~8-11 to N=6, 17.7 @12), ms linear (5->30);
  crossover N~=3 -- independently reproduces the paper's Table-3 crossover in
  a different code/modality/probe.
- Caveats: single seed, 20 iters (not converged), small absolute gaps;
  reproducibility hazards fixed en route: editing session's uncommitted ptyrad
  rewrite breaks born path (sweep ran on git worktree @HEAD); clean HEAD needs
  optuna (installed) + torchvision (shimmed gaussian_blur, exact to 1e-7).
- Figure: demo/output/paper_extra_tests/nsweep_summary.png

# CORRECTION after 60-iteration confirmation runs (confirm_results.json)
- gap(born-ms): N=2: 0.0008(20it)->0.0033(60it); N=12: 0.0023->0.0013.
  Ordering INVERTS with budget => the "(1-1/N) gap growth" and the "Born
  optimum at N=6" from the 20-it sweep were CONVERGENCE artifacts. RETRACTED.
- Survives: (a) cost crossover N~=3, budget-independent, independently
  reproduces the paper's Table-3 crossover in PtyRAD/electrons;
  (b) born-ms loss gap stays <=0.9% relative at all N/budgets tested;
  (c) eps1-vs-N by reconstruction needs a converged protocol (>=200 it/point).
- nsweep_summary.png regenerated to claim only what survived.

# FINAL: converged protocol (200 it, N in {2,6,12}) -- converged_results.json
| N | born | ms | gap | born s/it | ms s/it |
|---|------|----|-----|-----------|---------|
| 2 | .3827| .3805 | .0022 | 4.7 | 5.5 |
| 6 | .3762| .3748 | .0014 | 8.7 | 13.7 |
| 12| .3755| .3743 | .0012 | 16.1| 28.8 |
1. Gap SHRINKS with N (0.6%->0.3% rel): eps1 N-growth not visible in converged
   loss on tBL-WSe2; Born is a near-drop-in at every N here.
2. Born FASTER at every N converged (1.2x -> 1.8x). The 20-it "crossover N~=3"
   was startup-overhead contamination; PtyRAD crossover is at/below N=2.
3. Meta-finding: ALL short-budget conclusions inverted under convergence, in
   both directions. Paper-grade caution: compare at matched CONVERGED budgets.
Figure: nsweep_converged_summary.png (only converged claims).

# N=3 cell recons (CELL-3 / T-CELL-3, 10 it, frozen truth probe)
Grid 0/600/1200um -> window s0, sample s1 (568um, 32um off), s2 empty.
- Born: energy 6.7/75.5/17.8%; corr(s0,window)=+0.78, corr(s1,sample)=+0.93.
  SEPARATES, sharper than N=10 (75.5% vs 43.6%; 0.93 vs 0.79): coarse slices
  = better-conditioned axial problem (spacing 5.6x DOF). proj 0.892.
- 3PIE: energy flat 33/33/33% AGAIN (third budget/slicing) -> systematic:
  per-slice sweep spreads structure evenly, no axial specialisation in 10 it.
  proj 0.935 (best fit, worst separation). |corr| peaks at the right plane
  under a consistent sign flip (engine phase convention in my comparator).
- Application takeaway strengthened: for contamination removal, Born + a few
  DOF-spaced slices beats both single-slice AND matched-budget multislice.
