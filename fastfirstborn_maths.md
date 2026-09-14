# Line searches for thick-sample ptychography: First Born and multislice

Notation follows the code (`ptypy/custom/firstborntorch.py`,
`ptypy/custom/fastfirstborntorch.py`, `ptypy/custom/fastthreepietorch.py`). All fields live on the $n_y \times n_x$
frame grid; $j = 1..N$ indexes slices, $m$ probe modes, $b$ views in a batch.
$\mathcal{F}$ is the 2-D FFT, $\mathsf{F}$ the calibrated far-field propagator
(exit plane $\to$ detector), and

$$
\mathcal{P}_{z}[x] \;=\; \mathcal{F}^{-1}\!\big[ H_{z}\, \mathcal{F}[x] \big],
\qquad
H_{z}(\mathbf{f}) \;=\; e^{\,2\pi i \frac{z}{\lambda}\left(\sqrt{1-(\lambda
\mathbf{f})^2}-1\right)}
$$

is angular-spectrum free-space propagation over distance $z$ ($H_0 = 1$,
$H_{-z} = H_z^{*}$, unitary).

---

## 1. Standard First Born (`FirstBornTorch`)

### 1.1 From multislice to the Born sum

Exact multislice propagates the probe through the slice stack as an ordered
product:

$$
\psi_1 = P, \qquad
\psi_{j+1} = \mathcal{P}_{\Delta z_j}\!\big[\, O_j\, \psi_j \,\big],
\qquad \psi_{\text{exit}} = O_N \psi_N .
$$

Write each slice transmission as identity plus a scattering perturbation,

$$
O_j = 1 + g_j ,
$$

and expand the cascade in powers of $g$. The zeroth-order term is the
unscattered probe propagated to the exit plane; the first-order terms scatter
exactly once, at slice $j$, and propagate freely before and after. Referring
everything back to the entrance plane (the code's convention) gives the
**first Born model**:

$$
\boxed{\;
\psi \;=\; P \;+\; \sum_{j=1}^{N} \mathcal{P}_{-z_j}\!\big[\, g_j\, \phi_j
\,\big],
\qquad
\phi_j \;=\; \mathcal{P}_{z_j}[P]
\;}
$$

where $z_j$ is the depth of slice $j$ and $\phi_j$ — the *unscattered
illumination* — depends only on the probe, so it is computed once per batch
(and cached outright once the probe freezes).

The neglected terms are all of the form $g_k \,\mathcal{P}[\,g_j \phi_j\,]$
(scatter at $j$, then again at $k$). Since $g = 0$ in vacuum, they vanish
identically wherever the illuminated structures on different slices do not
overlap in the beam path: for sparse specimens the model is not an
approximation to multislice but **equal to it**, at $O(1)$ depth cost instead
of $O(N)$.

The key structural fact, used everywhere below: **$\psi$ is affine in the
slices** ($\partial^2 \psi / \partial O^2 \equiv 0$) **and linear in the
probe**.

### 1.2 Measurement and cost

Per view, with data intensity $I_{\text{dat}}$ and mask $m$:

$$
F_m = \mathsf{F}\,\psi_m, \qquad
u = \sum_m |F_m|^2 ,
$$

and the engine tracks two errors — the amplitude cost
$\sum m\,(\sqrt{u}-\sqrt{I_{\text{dat}}})^2$ and the **Gaussian intensity
cost**

$$
Q \;=\; \sum_{\text{pix}} \omega\, \big(u - I_{\text{dat}}\big)^2 ,
\qquad
\omega = \frac{m}{I_{\text{dat}} + 1},
$$

whose variance weighting $1/(I+1)$ is the Gaussian approximation to Poisson
noise (+1 regularises dark pixels).

### 1.3 Residual and gradients

The update direction comes from the amplitude projection. With

$$
f \;=\; \sqrt{ I_{\text{dat}} / u } \;-\; 1
\quad (\text{masked}), \qquad
r_m \;=\; \mathsf{F}^{\dagger}\big[ F_m\, f \big],
$$

$r$ is the back-propagated modulus mismatch (for unitary $\mathsf{F}$ it is
exactly $\psi^{\text{proj}} - \psi$). Because the forward map is affine, its
adjoint is elementwise:

$$
\frac{\partial \psi}{\partial g_j} = \phi_j\,\mathcal{P}_{-z_j}[\,\cdot\,]
\quad\Longrightarrow\quad
\nabla_{g_j} \;=\; \sum_m \phi_{j,m}^{*}\;
\mathcal{P}_{z_j}\!\big[\, r_m \,\big],
$$

(the code's `phi[j].conj() * ifft2(H_fwd[j] * fft2(r))`). The probe enters
twice — directly through the identity and through its scattered echo — so

$$
\nabla_{P} \;=\; r \;+\; \sum_j \mathcal{P}_{-z_j}\!\big[\, g_j^{*}\,
\mathcal{P}_{z_j}[r] \,\big].
$$

### 1.4 Preconditioning and the fixed step

The affine model separates per mode into a constant term and a sum of
per-slice **linear forward operators** $A_{j,m}$,

$$
F_m \;=\; \mathsf{F}\,P_m \;+\; \sum_{j=1}^{N} A_{j,m}\, g_j ,
\qquad
A_{j,m}\, x \;\equiv\; \mathsf{F}\, \mathcal{P}_{-z_j}\!\big[\, x\,
\phi_{j,m} \,\big],
$$

i.e. $A_{j,m}$ multiplies a slice perturbation by the unscattered
illumination, propagates it back to the entrance plane and out to the
detector. Its adjoint is read off by moving each factor across the inner
product ($\mathsf{F}$ and $\mathcal{P}_z$ are unitary, multiplication by
$\phi$ conjugates):

$$
A_{j,m}^{\dagger}\, y \;=\; \phi_{j,m}^{*}\;
\mathcal{P}_{z_j}\!\big[\, \mathsf{F}^{\dagger} y \,\big],
$$

which is exactly the gradient formula of §1.3 applied to the residual.
Because $\mathcal{P}_{z_j}$ and $\mathsf{F}$ are unitary they cancel in the
normal operator, which is therefore **exactly diagonal** (summing modes,
$A_j^{\dagger}A_j \equiv \sum_m A_{j,m}^{\dagger} A_{j,m}$):

$$
A_j^{\dagger} A_j \;=\; \operatorname{diag}\!\Big( K_j \Big),
\qquad
K_j \;=\; \sum_m |\phi_{j,m}|^2 .
$$

Two preconditioners are offered:

- **`local`** — divide by $K_j + \varepsilon\, \max K_j$ per pixel: the exact
  diagonal of $A^{\dagger}A$ (diagonal Gauss–Newton), floored at a fraction
  $\varepsilon$ (`denom_reg`) of its own peak so dark regions don't blow up.
- **`max`** — divide by the scalar $\max_{\text{pix}} K_j$ (the ePIE
  convention). Same direction *up to per-pixel shape*: weakly illuminated
  pixels are not boosted.

The update is a damped Jacobi step, simultaneous over slices:

$$
O_j \;\leftarrow\; O_j \;+\; \frac{\alpha}{N}\;
\frac{\nabla_{g_j}}{K_j + \varepsilon \max K_j} .
$$

With the exact diagonal preconditioner each per-slice operator is the
identity, so the classical Jacobi stability bound is $\alpha < 2$; the $1/N$
accounts for $N$ slices moving at once. **$\alpha$ is a hand-tuned constant**
— the same number for every view, every pixel-pattern, every stage of
convergence. That is the inefficiency the fast solver removes.

---

## 2. Fast First Born (`FastFirstBornTorch`)

### 2.1 The far field moves on a straight line

Let $d = (d_1,\dots,d_N)$ be any object direction (the preconditioned
gradient in practice). Because $\psi$ is affine in the slices,

$$
\psi(g + \alpha d) \;=\; \psi(g) \;+\; \alpha\, \delta\psi,
\qquad
\delta\psi = \sum_j \mathcal{P}_{-z_j}\big[\, d_j\, \phi_j \,\big],
$$

with **no higher-order terms** — the second derivative is identically zero.
Hence at the detector, exactly and for every $\alpha$:

$$
F_m(\alpha) \;=\; F_m + \alpha\, D_m,
\qquad
D_m = \mathsf{F}\,\delta\psi_m ,
$$

one extra forward half-pass, total. Per-pixel intensity is then *exactly
quadratic* in $\alpha$:

$$
I(\alpha) \;=\; \sum_m |F_m + \alpha D_m|^2
\;=\; u \;+\; 2\alpha\, v \;+\; \alpha^2 w,
$$

$$
u = \sum_m |F_m|^2, \qquad
v = \sum_m \operatorname{Re}\!\big( F_m^{*} D_m \big), \qquad
w = \sum_m |D_m|^2 .
$$

### 2.2 The cost along the line is an exact quartic

Insert into the Gaussian cost, with $e = u - I_{\text{dat}}$:

$$
Q(\alpha)
= \sum_{\text{pix}} \omega\,\big( e + 2\alpha v + \alpha^2 w \big)^2 ,
$$

a degree-4 polynomial whose stationarity condition is a **cubic** with plain
masked-sum coefficients:

$$
\frac{dQ}{d\alpha} = 4\big( c_0 + c_1 \alpha + c_2 \alpha^2 + c_3 \alpha^3
\big) = 0,
$$

$$
c_0 = \sum \omega\, e\, v, \qquad
c_1 = \sum \omega\, \big( e\, w + 2 v^2 \big), \qquad
c_2 = 3 \sum \omega\, v\, w, \qquad
c_3 = \sum \omega\, w^2 .
$$

The optimal step is the real root with the lowest $Q$ — a **global, exact,
closed-form line search**: no trial evaluations, no tuning. The parent's
$\alpha/N$ survives only as a fallback for degenerate cubics (zero direction,
fully masked frame). In practice the exact step is multiplied by
`ls_damp` $=\tfrac12$: each *batch's* exact minimiser is too greedy across
overlapping views, and half the exact step is the classical under-relaxation
for cyclic sweeps (measured: undamped plateaus an order of magnitude high;
damped descends monotonically).

### 2.3 The probe, and the bilinear cross term

$\psi$ is linear in $P$ (the direct path enters through the identity), so for
a probe direction $q$:

$$
\delta\psi_P \;=\; q \;+\; \sum_j \mathcal{P}_{-z_j}\!\big[\, g_j\,
\mathcal{P}_{z_j}[q] \,\big],
\qquad
D_P = \mathsf{F}\,\delta\psi_P ,
$$

and the same cubic gives the exact probe step $\beta$.

The object–probe cross term costs nothing: after the object step the far
field is **exactly** $F + \alpha D$, an elementwise update with no forward
re-evaluation, and $u, e$ update by the same quadratic. The probe line search
inside the same batch therefore runs against the *already-updated* field —
the bilinear coupling is handled exactly, which no fixed-step simultaneous
update can do. (The scattered echo also uses the post-step perturbation
$g + \alpha d$.)

Cost: $\sim 1.5\times$ the parent's transforms per batch for the object step,
$2\times$ when the probe updates too — repaid many times over per epoch.

### 2.4 Does the exact search need the `local` preconditioner?

**No.** Write the direction as $d = M^{-1} \nabla$ for any positive
preconditioner $M$. The exact line search minimises over the *ray*
$\{ g + \alpha d \}$, so rescaling $M \to cM$ rescales $d \to d/c$ and the
optimal $\alpha \to c\,\alpha$: **the applied update is invariant to the
overall scale of the preconditioner.** This is precisely the scale the parent
engine has to get right by hand through $\alpha/N$ — and it is why the fast
engine has nothing to tune.

What the search does *not* absorb is the per-pixel **shape** of $M$:

- `max` ($M = \max K_j$, a scalar per slice): the direction is the raw
  gradient. Perfectly valid — the cubic finds its exact step — but the
  direction under-weights weakly illuminated pixels, so convergence along
  those pixels is slower, and at batch size $> 1$ sparsely covered regions
  are starved (divided by the most-illuminated pixel's $K$).
- `local` ($M = K_j + \varepsilon \max K_j$): the direction is the diagonal
  Gauss–Newton step, i.e. the gradient measured in the metric of the exact
  normal-operator diagonal. Same cost per batch, better-shaped direction.

So: `max` is *allowed* and the run stays stable (the step is still exactly
optimal along whatever direction it is given); `local` simply provides a
better direction for the same price and is the sensible default. The
run scripts expose both (`run_fastborn_test.py <label> <niter> <B> max` for
a direct comparison).

### 2.5 Summary of what changes

| | `FirstBornTorch` | `FastFirstBornTorch` |
|---|---|---|
| direction | preconditioned gradient | same |
| object step | fixed $\alpha/N$ | exact cubic root, $\times$ `ls_damp` |
| probe step | fixed $\beta/N$ | exact cubic root, $\times$ `ls_damp` |
| cross term | ignored within batch | exact ($F \leftarrow F + \alpha D$) |
| tuning | $\alpha, \beta$ | none ($\alpha,\beta$ = fallbacks only) |
| cost per batch | $1\times$ | $\sim1.5\times$ ($2\times$ with probe) |

---

## 3. Where standard multislice (3PIE) fails, and why

### 3.1 The observed failure

On the 10-slice sim, ThreePIETorch with the conventional settings
(`max` preconditioner, batch 1, sequential sweep — verified to reproduce
the CPU `threepie.py` update for update, errors agreeing to 1–2 % at
every iteration) does this in 15 iterations:

- error descends ~3 % per iteration and **decelerates**
  (1270 → 532, visibly flattening);
- depth correlation **0.289**, projection correlation **0.002** — no
  reconstruction, versus 0.981 / 0.963 for the exact-step engine at the
  same 15 iterations on the same data.

Run on to 100 iterations (`ref100`, 2026-09-11) the same configuration
flatlines at error **224**, depth 0.268 / projection 0.005: the crawl is
a hard floor, not slow convergence.

Each simulated plane holds two sharp particles; the fixed-step
reconstruction recovers almost none of that structure:

![standard 3PIE failure: truth vs reconstruction](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_102824_gpu3pie-ref15/rec_sim_20260910_111021_slices-10_seed-0_gpu3pie-ref15_iter-0015_truth_vs_rec.png)

![standard 3PIE failure: decelerating crawl](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_102824_gpu3pie-ref15/rec_sim_20260910_111021_slices-10_seed-0_gpu3pie-ref15_iter-0015_error_history.png)

The earlier campaign saw the complementary failure mode on the same
data: the cascade **overshoots at low spatial frequencies** and the
degenerate modes drift unboundedly (±100 rad smooth phase blobs) even
while the data error stalls. Slow crawl in the data-constrained modes
and unbounded drift in the unconstrained low-f ones are two faces of
the same defect: the step is wrong mode by mode, not merely too small
or too large overall.

### 3.2 Why it fails: the nominal step stalls, and the working step is not knowable a priori

3PIE takes the same nominal step for every slice, every view, every
stage of convergence. The exact quartic tells us what the step *should*
have been, because the working engine logs it: across a converged sweep
the optimal per-slice step has **median ≈ 1.0 but ranges from 0.28 to
1.8** (5th–95th percentile) between slices and views. A caveat before
reading those numbers across engines: they are measured along the fast
engine's own directions (fixed-adjoint, preconditioned), not along
3PIE's sequentially re-derived residual, so they do **not** translate
into a 3PIE step multiplier — the scan below confirms it (the good
fixed value is 2.0, not 1.0). What the spread does show is that no one
scalar is right per slice, per view and per stage at once, and in an
ordered product those errors do not average out — they compound down
the cascade:

The full scalar-step scan (same data, 15 iterations unless noted):

| fixed step $h$ | depth corr / proj corr | reading |
|---|---|---|
| 0.3 | 0.318 / 0.005 | crawl |
| 1.0 (nominal) | 0.289 / 0.002 | crawl |
| 1.0, **100 iterations** | 0.268 / 0.005 | **stuck, not slow** — patience buys nothing |
| 2.0 | 0.950 / 0.898 | mostly works |
| exact quartic (Fast) | **0.981 / 0.963** | works, no tuning |

Two honest conclusions:

- The nominal step does not merely converge slowly — it **stalls**: 100
  iterations end where 15 did. The soft low-frequency modes (the
  directions that carry depth, where the data constrains only the
  product of the slices) make no net progress per sweep at $h \le 1$.
- A **retuned scalar at the classical ePIE stability edge** ($h = 2$)
  recovers most of the reconstruction. So standard 3PIE on this data is
  *under-stepped, not un-steppable* — but the working scalar had to be
  found by scanning, sits at the edge of the stability bound, and still
  trails the exact search, which picks its steps (measured 0.28–1.8,
  median 1.0) per slice, per view and per iteration with nothing to
  tune. The exact quartic is the step-size scan, done analytically at
  every update.

This diagnosis was cross-checked: the CPU engine fails identically at
the nominal step (not a port bug); the exact step with a bad direction
also fails (`local`'s dark-pixel boost, §4.3); the exact step with the
raw-gradient direction works untouched.

### 3.3 Standard 3PIE as a corrector

Warm-starting standard 3PIE (nominal step) from a converged Born
reconstruction — Hybrid B in the comparison figure of §5 — isolates its
*refinement* ability from its *building* ability. The fixed step first
**damages** the warm start (a clear error spike at the swap), then
recovers and crawls just below the Born plateau (final error 0.914 vs
Born's 1.23; scores stuck at the Born level, 0.982 / 0.973). The exact
step engine given the identical warm start (Hybrid A) plunges instead —
final error 0.0265, scores 0.996 / 0.996. The fixed step is a poor
corrector for the same reason it is a poor builder.

---

## 4. Fast ThreePIE (`FastThreePIETorch`): the same idea on the full cascade

### 4.1 What survives when the model is no longer affine

Exact multislice is the ordered product

$$
\psi_1 = P, \qquad
\psi_{j+1} = \mathcal{P}_{\Delta z_j}\!\big[\, O_j\, \psi_j \,\big],
\qquad
F_m = \mathsf{F}\, \big( O_N\, \psi_N \big)_m .
$$

A joint perturbation of all $N$ slices moves the far field on a degree-$N$
curve — the affinity of §2 is gone. But the cascade is **multilinear**:
linear in each slice $O_j$ separately (hold the others fixed), and linear
in the probe. So a step confined to ONE slice still moves the far field
exactly on a straight line,

$$
F(a) \;=\; F \;+\; a\, D_j,
\qquad
D_j \;=\; \mathsf{F}\,\Big[\, \text{push } (d_j\, \psi_j^{\text{in}})
\text{ through slices } j{+}1 .. N \,\Big],
$$

and the Gaussian cost along $a$ is the **same exact quartic** as §2.2 —
same coefficients, same cubic root, reused verbatim.

### 4.2 The sweep, with exact field maintenance

Slices are stepped sequentially, $j = 1..N$. After slice $j$'s exact step:

- the far field is exactly $F + a_j D_j$ (elementwise update, no forward
  re-evaluation), and
- the incident field at every downstream slice changes by exactly the
  intermediates already computed while building $D_j$
  ($\psi_k^{\text{in}} \mathrel{+}= a_j\,\iota_k$ for $k > j$),

so each slice's line search minimises the true cost *given every update
already applied in this batch*. Only the directions (preconditioned
gradients from one initial adjoint pass) are held fixed across the sweep.
The probe line search runs last, against the fully updated cascade
(linear in $P$, one extra cascade for $D_P$).

Directions: the residual $r$ back-propagated up the cascade,
$\chi_j^{\text{in}} = O_j^{*}\,\chi_j^{\text{out}}$,
$\chi_{j-1}^{\text{out}} = \mathcal{P}^{\dagger}[\chi_j^{\text{in}}]$
(the adjoint-state / Lagrange-multiplier recursion), with
$\nabla_j = \sum_m \psi_{j,m}^{\text{in}\,*}\, \chi_{j,m}^{\text{out}}$.

Cost: $\sum_j (N-j) \approx N^2/2$ extra propagations per batch. Measured
at $N = 10$: **the same wall time as plain ThreePIETorch** (~28 s per
iteration at batch 1) — the responses are elementwise-FFT cheap next to
the engine's fixed overhead.

### 4.3 Two measured facts that are now baked in

**The preconditioner must be `max` (raw-gradient direction).** In the
Born engines, `local` divides by $K_j = \sum_m |\phi_{j,m}|^2$, the exact
diagonal of the normal operator — a clean metric. In the cascade the
incident intensity $\sum_m |\psi_{j,m}^{\text{in}}|^2$ carries every
upstream slice, and its per-pixel inverse (floored at
`denom_reg` $\times$ peak) boosts dark-pixel noise that the exact search
then defends against by freezing (slice steps collapse to $\sim 10^{-5}$).
Since the exact step absorbs any scalar, `max` under a line search IS the
raw gradient direction — and it works.

**The quartic coefficients need float64.** The first sweeps see
projection residuals of $\sim 10^{10}$, so $\sum \omega\, w^2 \sim
10^{40}$ — past float32's $3.4\times10^{38}$. The sums are taken in
double; `solve_step_cubic` treats non-finite coefficients as degenerate.

### 4.4 The ablation (10-slice NanoMAX sim, 15 iterations, batch 1)

| engine | direction | step | depth corr / projection corr |
|---|---|---|---|
| ThreePIETorch, `max` | raw gradient | fixed 1.0 (nominal) | 0.289 / 0.002 |
| ThreePIETorch, `max` | raw gradient | fixed 2.0 (hand-retuned, §3.2) | 0.950 / 0.898 |
| FastThreePIETorch, `local` | per-pixel boosted | exact quartic | 0.267 / 0.040 |
| **FastThreePIETorch, `max`** | raw gradient | exact quartic | **0.981 / 0.963** |
| (reference: FastFirstBornTorch, `local`) | diag. Gauss–Newton | exact quartic | 0.982 / 0.965 |

Neither ingredient fixes multislice alone at the defaults: the nominal
fixed step descends at ~3 % per iteration and stalls (the low-frequency
cascade modes it cannot step far enough along); the exact step with the
`local` direction freezes. A hand-retuned scalar closes most of the gap
(row 2 — see §3.2 for why that value could not have been predicted),
but the exact step still scores higher at the same budget, keeps
descending where the scalar plateaus (§3.3), and has nothing to tune.
With the raw-gradient direction it matches the Born engine's quality on
data where Born is exact — and it remains applicable when Born is not.

---

## 5. Results gallery

**FastFirstBornTorch, 100 iterations, batch 1** — every simulated plane
above its reconstruction (depth corr 0.986, projection 0.975):

![FastFirstBorn truth vs reconstruction](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_080137_firstborn100/rec_sim_20260910_111021_slices-10_seed-0_iter-0100_truth_vs_rec.png)

![FastFirstBorn convergence](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_080137_firstborn100/rec_sim_20260910_111021_slices-10_seed-0_iter-0100_error_history.png)

**FastThreePIETorch (`max`), 15 iterations, batch 1** — full multislice,
same data (depth corr 0.981, projection 0.963):

![FastThreePIE truth vs reconstruction](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_095446_fast3pie-Jmax/rec_sim_20260910_111021_slices-10_seed-0_fast3pie-Jmax_iter-0015_truth_vs_rec.png)

![FastThreePIE convergence](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_095446_fast3pie-Jmax/rec_sim_20260910_111021_slices-10_seed-0_fast3pie-Jmax_iter-0015_error_history.png)

**Measured vs predicted diffraction frames** (multislice and Born forward
models through the FastFirstBorn reconstruction):

![measured vs predicted frames](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_080137_firstborn100/rec_sim_20260910_111021_slices-10_seed-0_iter-0100_born_frames.png)

**Head to head, 50 iterations, batch 1, probe every view, denom_reg
0.05 throughout.** All Born-stage curves overlap exactly for the first
15 iterations (same seed, same settings — the built-in consistency
check). FastFirstBorn descends fastest early and plateaus — the Born
model-mismatch floor on exact-multislice data (0.986 / 0.973).
FastThreePIE (`max`) crosses it at ~8 min wall despite 27 vs
17 s/iteration and keeps descending (0.995 / 0.988). **Hybrid A** —
FastFirstBorn for 15 iterations, then FastThreePIE warm-started from
its slice stack (`init_slices`) and probe — is the best of all four:
no penalty at the swap (dotted line), an immediate plunge, final error
0.0265, and the campaign's best scores, **0.996 / 0.996**, reaching
pure FastThreePIE's final error ~1.6x sooner in wall time. **Hybrid B**
— the same warm start handed to *standard* fixed-step 3PIE — spikes at
the swap, recovers, and crawls to just below the Born floor (0.914;
scores stay at the Born level): the fixed step is as poor a corrector
as it is a builder (§3.3).

The 5-way version adds the hand-retuned standard 3PIE (steps 2.0, §3.2)
as a fifth curve — it tracks the pack to error ~2 over its 15
iterations, then stops where its budget ends:

![5-way convergence comparison](data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/sim_20260910_111021_slices-10_seed-0/rec/run_20260911_132443_comparison-5way/convergence_5way.png)

---

## 6. The published adaptive-step baseline: multislice LSQ-ML

Computed-rather-than-tuned step lengths are not new: Odstrčil, Menzel &
Guizar-Sicairos, *Iterative least-squares solver for generalized
maximum-likelihood ptychography*, Opt. Express **26**, 3108 (2018)
(LSQ-ML) derives per-view optimal steps from a *linearised*
least-squares solve, and the multilayer engines built on it
(PtychoShelves `MLs`, fold_slice) apply them at every slice of the 3PIE
cascade. To position the exact line search against that prior art, the
method was implemented here as `LSQMLTorch`
(`ptypy/custom/lsqmltorch.py`, driver `run_lsqml_test.py`): identical
cascade, residual and error conventions as `ThreePIETorch`, with the
fixed ePIE steps replaced at every slice by the joint real solution
$(\alpha, \beta)$ of the 2x2 normal equations of

$$
\chi \;\approx\; \alpha\, (\Delta O\, \psi) \;+\; \beta\, (\Delta\psi\, O),
\qquad
\Delta O = \frac{\psi^{*}\chi}{\sum_m |\psi|^2 + \delta \max},
\qquad
\Delta\psi = \frac{O^{*}\chi}{M |O|^2 + \delta \max},
$$

Tikhonov-regularised, coefficients summed in float64 (same overflow
trap as §4.3). Not implemented, deliberately: the momentum-accelerated
aggregation ('MLc'), variable-intensity and position refinement — at
batch 1 this is the sequential LSQ-ML step rule in isolation, which is
the thing being compared.

On the 10-slice sim, 15 iterations, batch 1:

| step rule | depth / proj corr |
|---|---|
| fixed 1.0 (3PIE default) | 0.289 / 0.002 |
| fixed 2.0 (hand-scanned, §3.2) | 0.950 / 0.898 |
| LSQ-ML, `local` directions (faithful) | 0.946 / 0.851 |
| LSQ-ML, `max` directions | 0.949 / 0.883 |
| **exact quartic (FastThreePIE, `max`)** | **0.981 / 0.963** |

Three facts worth keeping:

- **LSQ-ML lands at the level of the perfectly hand-tuned fixed step,
  untuned.** Its solved incident-field steps settle at median ~1.9
  (`local`) / ~2.1 (`max`) — it *computes* the step value the scalar
  scan found by search, which independently validates both results.
- **LSQ-ML is insensitive to direction shape** (`local` vs `max`
  nearly equal), unlike the exact search, where `local` is fatal
  (§4.3–4.4). The per-view LSQ $\beta$ on the incident field absorbs
  what the exact search has to freeze against.
- **The exact step still wins at equal budget.** So the correct
  novelty claim is not "adaptive steps" — that is LSQ-ML's — but the
  *exact* per-slice step (global minimiser of the true quartic, not a
  linearisation) with exact field maintenance through the sweep.

---

## 7. Depth scaling: N = 30 and N = 50

All runs 15 iterations, batch 1, probe every view, on the 30- and
50-slice NanoMAX sims (same specimen recipe, 1580 µm separation).

**FastFirstBorn scales gracefully** — the affine simultaneous update
has no depth pathology, and the cost grows linearly:

| slices | depth / proj corr | s/iter | error at 15 it |
|---|---|---|---|
| 10 | 0.982 / 0.965 | ~17 | plateaued (Born floor) |
| 30 | 0.975 / 0.922 | ~48 | 4.2, still falling |
| 50 | 0.913 / 0.774 | ~77 | 5.4, still falling |

(The N = 50 run is not converged at 15 iterations — a 50-iteration run
is in flight at the time of writing.)

**Every sequential multislice step rule fails from a cold start at
N = 30** — and the failure ranking *inverts* the N = 10 story:

| method (N = 30, cold start) | depth / proj corr | failure mode |
|---|---|---|
| FastFirstBorn (Born model) | 0.975 / 0.922 | — (converging) |
| exact quartic (FastThreePIE `max`) | 0.015 / −0.002 | steps freeze at ~1e-5 |
| LSQ-ML (`local`) | 0.072 / 0.001 | stalls at error ~890 |
| fixed 2.0 (hand-retuned) | 0.853 / 0.628 | partial — error 14, falling |
| hybrid, stage-B `ls_damp` 0.5 | 0.919 / 0.733 | stage B *damages* the warm start (4.3 → 39) |
| hybrid, stage-B `ls_damp` 0.25 | in flight | error 4.3 → 1.8 by +8 it, falling |

The interpretation that fits all six rows: **at depth, per-view
optimality is the failure mechanism, not the cure.** The degenerate
low-frequency subspace grows with N, and the depth information lives in
directions that barely reduce any *single* view's cost. A per-view
optimal step — exact or linearised — therefore correctly computes
"don't move" (the exact search freezes; LSQ-ML crawls), while the
over-aggressive fixed step keeps pushing through the degenerate
subspace and lets structure accumulate across views. What works at
N = 30 is the Born warm start plus *gentler* exact refinement
(`ls_damp` 0.25 rather than the 0.5 that is right at N = 10): the
warm start supplies the depth structure the per-view steps cannot
build, and the heavier under-relaxation controls the cross-view greed
that at 0.5 actively un-builds it.

**Implementation note (slice-wise parallelism).** `FastThreePIETorch`
gained `response_mode = 'wavefront'`: all N per-slice responses march
through the cascade as one stacked tensor (N batched stages, one
batched far field) instead of N²/2 sequential launches, with the line
searches still applied sequentially against the exactly updated far
field. Quality is at parity (0.979 / 0.961 vs 0.981 / 0.963 at
N = 10) but wall time at N = 30 is unchanged (117 vs 119 s/it): on
this GPU the cost is FFT flops, not kernel launches. Reducing the
depth cost needs flop reduction (grouped slices, a joint-response
accumulator), not batching; the mode is kept for wider hardware.
