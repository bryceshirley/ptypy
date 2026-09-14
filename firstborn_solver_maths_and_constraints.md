# First Born & Fast First Born solvers — maths, constraints, and findings

*Thick-sample high-convergence X-ray ptychography (8 keV; samples: 10 slices @ 100 µm
over 900 µm, and 30 slices @ 100 µm over 3 mm).*
Engines: `ptypy/custom/firstborntorch.py` (`FirstBornTorch`) and
`ptypy/custom/fastfirstborntorch.py` (`FastFirstBornTorch`, subclass).

This document is the up-to-date reference for the physics, the exact
optimisation, every probe/object constraint, **how the test data are generated
(and why that is an inverse crime for multislice but not for Born)**, and the
empirical findings from the probe-recovery and thick-sample campaigns. It
supersedes the older `fastfirstborn_maths.md`.

All mathematics is written in LaTeX. Notation: $\mathcal{F}$ is the transverse
Fourier transform and $\mathcal{F}^{-1}$ its inverse; $\mathcal{F}_{\mathrm{ff}}$
is the far-field (detector) transform; $P$ is the probe, $O_j$ the object
transmission of slice $j$, $g_j = O_j-1$ its scattering perturbation; $\psi$ the
exit wave; $\Psi=\mathcal{F}_{\mathrm{ff}}\{\psi\}$ the detector field; and
$I=\sum_m|\Psi|^2$ the model intensity (sum over probe modes $m$).

---

## 1. Forward model (both Born engines)

Referred to the entrance plane, with the sample split into $N$ transverse slices
at depths $z_j$:

$$
g_j = O_j - 1
$$

$$
\phi_j = \mathcal{F}^{-1}\!\big[\,H_{z_j}\,\mathcal{F}[P]\,\big]
\qquad\text{(unscattered illumination at slice } j)
$$

$$
\psi = P \;+\; \sum_{j=0}^{N-1}
\mathcal{F}^{-1}\!\big[\,H_{-z_j}\,\mathcal{F}[\,g_j\,\phi_j\,]\,\big]
\qquad\text{(exit wave at the entrance plane)}
$$

with the angular-spectrum (Fresnel/Helmholtz) propagator

$$
H_{z}(f) = \exp\!\Big[\,2\pi i\,\tfrac{z}{\lambda}
\big(\sqrt{1-(\lambda f)^2}-1\big)\Big],
\qquad H_0 = 1 ,
$$

where $f$ is the transverse spatial frequency and $\lambda$ the wavelength. The
detector field is $\Psi=\mathcal{F}_{\mathrm{ff}}\{\psi\}$ and the model intensity
is $I=\sum_m|\Psi|^2$.

This is a **single-scattering (first Born)** model: each slice scatters the
*unscattered* illumination $\phi_j$ once, and the contributions are summed. It is
**not** the exact multislice cascade used to make the data (§2) — that distinction
is the whole point of the method and the source of the model-error floor (§1.2).

### 1.1 The structural facts we exploit

1. **The slices enter as an affine SUM, not an ordered product.** Exact
   multislice composes $N$ propagate-multiply stages (depth $O(N)$); the Born
   model sums $N$ independent single-scattering contributions (depth $O(1)$).
   Because $\phi_j$ depends only on the probe, it is computed **once per
   mini-batch**.

2. **The map is bilinear:** linear in the probe $P$ for fixed object, and affine
   in the object slices $g_j$ for fixed probe:

$$
\psi(g + a\,d,\;P) = \psi(g,P) + a\,\delta\psi_o ,
\qquad
\psi(g,\;P + b\,q) = \psi(g,P) + b\,\delta\psi_p ,
$$

   with $\delta\psi_o$ independent of $a$ and $\delta\psi_p$ independent of $b$.
   The second derivative of the field w.r.t. the object is **identically zero**.

3. **The model-curvature term vanishes exactly; the diagonal preconditioner is an
   approximation.** Because $\psi$ is affine in $g$, the Hessian of the cost
   $C(g)=f(\psi(g))$ loses its model-curvature piece and equals $A^\dagger \mathsf{H} A$
   exactly, where $\mathsf{H}=\nabla^2 f$ at $\psi$. But $\mathsf{H}$ is **not** the
   identity — the modulus/intensity data term is nonlinear (this is where the
   phase-retrieval nonconvexity lives), so replacing $A^\dagger \mathsf{H} A$ by
   $A^\dagger A$ is the modulus linearisation. And even $A^\dagger A$ is **not**
   block-diagonal:

$$
(A^\dagger A\,g)_j = |\phi_j|^2\,g_j
\;+\; \sum_{k\neq j}\phi_j^{*}\,
\mathcal{P}_{z_j - z_k}\!\big[\,g_k\,\phi_k\,\big] .
$$

   The off-diagonal blocks are small when slice separation exceeds the depth of
   field (§1.3) but are not zero. The engine uses the **diagonal**
   $K_j=|\phi_j|^2$ as a preconditioner — an approximation, not an identity —
   which is why the Jacobi bound stays $\alpha < 2/N$ (§3.2), not $\alpha<2$.

### 1.2 The overlap property (why the Born model is not merely an approximation)

Every higher-order multiple-scattering term the Born model drops has the form

$$
\mathcal{P}_{-z_k}\!\Big[\,g_k \,\cdot\,
\mathcal{P}_{z_k - z_j}\!\big[\,g_j\,\phi_j\,\big]\Big] .
$$

The inner operator is **diffractive**, not geometric: $g_j\phi_j$ spreads by
$\sim\!\sqrt{\lambda\,\Delta z}$ before it reaches slice $k$, and the
angular-spectrum kernel has no compact support. So the neglected term is small
when the transverse separation exceeds the **Fresnel spread** — not, as a
ray-optics argument would have it, only when the material supports are
geometrically disjoint. For sparse or transversely-shifted layers the term is
small and the Born model is a close, $O(1)$-depth surrogate for $O(N)$
multislice; it is **not exact**.

**How close, measured (§9.1).** With the *same* frozen truth probe, FastBorn
reaches proj corr $0.968$ and FastThreePIE $0.998$, a $0.030$ gap. Read this as an
*upper bound* on the Born model error, not a clean measurement of it, for two
reasons that both favour multislice:

- **Inverse crime (§2.3).** FastThreePIE reconstructs with the *same* exact
  multislice operator that generated the data, so its model error is zero **by
  construction**; FastBorn uses a *different* (single-scattering) operator, so it
  carries an irreducible model-error floor the data can actually penalise. A large
  part of the $0.030$ is this asymmetry, not physics that would appear on real data.
- **Update cadence.** Both runs are batch 1 with the probe frozen, but
  FastThreePIE-`exact` takes $N$ per-slice object line searches per view while
  FastBorn takes **one** combined step per view (§4). At equal epochs multislice
  does $\sim\!N\times$ more object updates, so some of the gap is convergence rate.

A matched comparison (same operator only where physically warranted, one step per
view each, same `response_mode`) would isolate the true model error. Use $0.030$ as
a ceiling, not a claim of exactness. See also §2.3 and §9.1.

### 1.3 Depth of field (why the diagonal preconditioner is safe here)

The off-diagonal coupling in §1.1(3) is controlled by the depth of field versus
the 100 µm slice separation. At 8 keV ($\lambda = 1.5498\ \text{Å}$) the supported
convergence semi-angle from `sim_report` is $\mathrm{NA}\approx 6.5\ \text{mrad}$,
giving

$$
\mathrm{DoF} = \frac{\lambda}{\mathrm{NA}^2}\approx 3.7\ \mu\text{m}
\qquad\Big(\text{or }\tfrac{2\lambda}{\mathrm{NA}^2}\approx 7.3\ \mu\text{m}
\text{ by the looser convention}\Big) .
$$

So the 100 µm spacing is $\sim\!14\text{–}27\times$ the DoF. That is the
quantitative reason the diagonal $|\phi_j|^2$ preconditioner is adequate on this
geometry; it would start to degrade only once the separation fell to single-digit
µm. (Note: "NA 3.0" in the run labels is shorthand for the high-convergence
setting; a true numerical aperture cannot exceed 1 in vacuum, and X-ray refractive
indices sit just below 1 — read it as the convergence-angle *regime*, not a
literal aperture of 3.0.)

### 1.4 Depth resolution (why the slices are separately recoverable)

The axial (depth) resolution is set by the same angular aperture as the DoF:

$$
\delta_z \approx \frac{\lambda}{\mathrm{NA}^2}\approx 3.7\ \mu\text{m}
\qquad(\mathrm{NA}=6.5\ \text{mrad},\ \S1.3).
$$

The slice separation (100 µm) is $\sim\!27\times\,\delta_z$, so the slices sit far
apart on the axial-resolution scale and are, in principle, independently
resolvable — the recon is not being asked to split sub-resolution layers. The
transverse probe scale from `sim_report` is $a_c\approx 11.7\ \mu\text{m}$; the
axial extent of a single caustic, $\sim a_c/\mathrm{NA}\approx 1.8\ \text{mm}$,
spans the whole 900 µm sample, which is why the probe waist lands *inside* the
stack (§9.3) and why depth information is carried by defocus diversity across
slices rather than by any slice sitting at its own focus. Net: depth *resolution*
is comfortable; the difficulty is the probe caustic, not axial under-sampling.

---

## 2. Data generation and the inverse crime

**This section is essential to reading every ceiling in this document.** The test
data are synthetic, produced by `MultisliceSimulator`
(`ptypy/custom/multislicesim.py`, a subclass of `FirstBornDiagnostics`), driven by
`run_multislice_sim_100um_het.py`. Understanding the generating model is what makes
the FastBorn↔FastThreePIE comparison interpretable.

### 2.1 The synthetic specimen

1. A finished NanoMAX scan-434 reconstruction is loaded and mean-gauged so the
   empty background sits at $O=1$ and the physical flux is carried by the probe.
2. The particle-bearing slice is **segmented** into connected components of
   $|O-1|$ above `segment_threshold`, keeping only components larger than
   `min_particle_px` (5000 px — only the two real centre blobs survive; the small
   specks are reconstruction noise).
3. Each particle is **copied** `particle_copies` times, each copy given a random
   layer, a random flip/rotation/scale (`transform_copies`), and a random
   in-plane position under a **no-overlap-in-projection** constraint
   (`no_overlap`, `overlap_margin_px`). Copies are spread over the $N$ layers as
   uniformly as the count allows. Overlapping transmissions **multiply**
   (`layers[L][dst] *= 1 + patch`), as physical transmissions do.

Sample settings used here: `SIM_SLICES` $\in\{10,30\}$, 100 µm spacing
(`SIM_THICKNESS = (SIM_SLICES-1)\times100\ \mu\text{m}$; 900 µm or 2.9 mm span, i.e.
a 3 mm sample as $30\times100\ \mu\text{m}$ slabs), `LAYER_SEED = 2`,
`PARTICLE_COPIES` = 15 (10-slice) or 60 (30-slice, denser $\sim$4 blobs/slice).

### 2.2 The simulated frames — exact multislice, noiseless

`_simulate` (line 570) pushes the reconstructed probe through the synthetic stack
with the **exact multislice cascade** (the same `_multislice` operator the
diagnostics engine uses), then takes the far-field intensity:

$$
\psi_0 = P,\qquad
\psi_{j+1} = \mathcal{P}_{\Delta z}\!\big[\,O_j\,\psi_j\,\big],\qquad
I^{\text{dat}} = \sum_m \big|\,\mathcal{F}_{\mathrm{ff}}\{O_{N-1}\,\psi_{N-1}\}\,\big|^2 ,
$$

where $\mathcal{P}_{\Delta z}$ is the angular-spectrum propagator over one
inter-slice gap $\Delta z$. Denote this operator $\mathcal{M}_{\text{exact}}$:

$$
I^{\text{dat}} = \big|\,\mathcal{F}_{\mathrm{ff}}\,
\mathcal{M}_{\text{exact}}\!\big(O^{\text{truth}},\,P^{\text{truth}}\big)\big|^2 .
$$

The frames are written into a copy of the scan's `.ptyd` (identical geometry,
positions, mask, frame order), so any ptypy engine reconstructs them via a plain
`PtydScan`. **`POISSON_NOISE = False` in these runs — the data are noiseless.** The
probe used (optionally NA-scaled and Fresnel-defocused by `PROBE_DEFOCUS`
$=700\ \mu\text{m}$) and the full layer stack are written to
`..._truth_iter-0000.h5`, which is exactly what `truth_probe=1` injects and
freezes (§7). So a frozen-probe run is handed the **exact generating probe**.

### 2.3 The inverse crime: 3PIE self-consistent, Born handicapped

An **inverse crime** is reconstructing data with the same forward model that
generated it: the model can represent the data exactly, so residual model error is
zero by construction and the achievable ceiling is set only by noise, sampling and
optimisation — not physics. Here the data-generating operator is
$\mathcal{M}_{\text{exact}}$ (multislice), so:

| engine | reconstruction operator | vs generator | inverse crime? |
|---|---|---|---|
| (Fast)ThreePIE | $\mathcal{M}_{\text{exact}}$ (multislice) | **identical** | **yes** |
| (Fast)FirstBorn | $\mathcal{M}_{\text{Born}}$ (single scattering) | **different** | **no** |

Consequences that must colour every ceiling in §9:

- **ThreePIE is marking its own homework.** With noiseless data and the frozen
  truth probe, its only obstacles are optimisation and sampling; the $0.998$
  ceiling is an inverse-crime ceiling, not evidence of superior physics.
- **Born is penalised for a real model mismatch.** Its irreducible floor is

$$
\varepsilon \;=\; \big\|\,\mathcal{M}_{\text{Born}}(O,P) -
\mathcal{M}_{\text{exact}}(O,P)\,\big\|\ \text{on the data},
$$

  nonzero because single scattering cannot reproduce a multiply-scattered field.
  That FastBorn still reaches $0.968$ / proj corr $0.90\text{–}0.97$ is achieved
  **despite** this handicap.
- **The $0.030$ ceiling gap (§1.2, §9.1) is therefore an upper bound**, inflated by
  the inverse crime plus the cadence asymmetry — not a clean model-error figure.
- **On real data neither engine commits the crime** (the true physics is neither
  exact-multislice-in-this-basis nor Born), so the fair comparison lives there;
  the simulator systematically flatters multislice.

The honest framing for the method: Born is proposed as a fast $O(1)$-depth
**preconditioner / initialiser**, and its early-iteration reconstruction quality
is what matters (a preconditioner need not converge to the exact minimiser). Judge
it on speed-to-usable-object and on real data, not on the inverse-crime ceiling of
its multislice competitor.

---

## 3. Inversion & preconditioning (`FirstBornTorch`)

Notation used in `_batch_update`:

- $\text{fw},\ \text{bw}$ : forward / backward far-field propagators
  (`self._apply(self._ff_fw/_ff_bw, .)`).
- $u$ : model intensity $\sum_m|F|^2$ per detector pixel.
- $\text{fac}$ : amplitude projection factor $\sqrt{I_{\text{dat}}/u}-1$.
- $\text{acc\_grad}$ : accumulated object gradient (adjoint of the $\text{fac}$-weighted residual).
- $K = \text{acc\_K}$ : accumulated preconditioner weight $=|\phi_j|^2$ summed over the batch.
- $\text{peak\_K}$ : the spatial max of $K$ (a scalar per slice).

### 3.1 Amplitude projection

The Gaussian-noise data term drives the field toward the measured modulus while
keeping the model phase:

$$
\text{fac} = \frac{\sqrt{I_{\text{dat}}}}{\sqrt{u}} - 1
\qquad (u=\text{model intensity}).
$$

### 3.2 Object update (Jacobi)

Using the **approximate** diagonal $K=|\phi_j|^2$ (not the true normal operator
$A^\dagger A$, which has small off-diagonal blocks — §1.1(3)) as a Jacobi
preconditioner, the step is

$$
O_j \;\leftarrow\; O_j \;+\; \frac{\alpha}{N}\,\frac{\text{acc\_grad}}{\text{dn}_o},
$$

with $\text{dn}_o$ the denominator (see §5). Because the preconditioner is only the
diagonal, updating all $N$ slices simultaneously is a **Jacobi** iteration whose
worst-case stable step is $\alpha < 2/N$ (the generic bound for $N$ simultaneous
blocks under a diagonal preconditioner, *not* off-diagonal-specific); the code
divides by $N$, so $\alpha=1.0$ is a safe aggressive default. Were $K$ the exact
inverse, the bound would be $\alpha<2$; the factor-$N$ gap is the price of the
diagonal approximation.

### 3.3 Probe update

Analogous preconditioned step with denominator $\text{dn}_p$ built from the
probe-side weight $K_P$ and its peak $\text{peak\_P}$ (see §5).

---

## 4. Exact line searches (`FastFirstBornTorch`)

**What is *not* Born-unique (corrected).** The exact far-field update after a step
(§4.3, $F\leftarrow F + a\,D$) is **not** a Born privilege. The multislice exit
field is multilinear, so a step confined to one slice also moves the far field
exactly linearly, and `fastthreepietorch._batch_update` maintains the downstream
incident fields exactly through the sweep (`incs`, `psis[k] += a*inc`). The
exact-field trick transfers verbatim.

**What *is* categorically Born-only.** Born's cost is an exact quartic along
**any** direction — including a **joint step over all $N$ slices at once**.
Multislice is quartic only along a **single-slice** direction ($D=D_{\text{all}}[j]$
is per-slice); its sweep recovers exactness *sequentially*, one slice at a time,
but can never take one optimal step in all slices simultaneously. This is what
makes the $N$-dimensional vector line search of §9.3 (one $a_j$ per slice from a
single $N$-D quartic) reachable by the Born model and by no multislice
reformulation.

### 4.1 Why a closed form exists

Along any update direction the far field moves on a straight line (2nd derivative
zero):

$$
F(a) = F + a\,D
\quad\Longrightarrow\quad
I(a) = |F(a)|^2 = u + 2av + a^2 w \quad\text{(per pixel)},
$$

with $u=|F|^2$, $v=\mathrm{Re}(F^{*}D)$, $w=|D|^2$. The Gaussian intensity cost is
then an **exact quartic** in the scalar step $a$:

$$
Q(a) = \sum \text{mask}\,\frac{\big(I(a)-I_{\text{dat}}\big)^2}{I_{\text{dat}}+1} .
$$

Its minimiser is a root of the cubic
$\;\dfrac{dQ}{da} = 4\big(c_0 + c_1 a + c_2 a^2 + c_3 a^3\big) = 0$.
This is global, closed-form, and needs **zero trial evaluations** — one extra
forward half-pass to form $D$. Exact multislice/3PIE composes $N$ nonlinear
stages, so along a line its exit field is a degree-$N$ polynomial in $a$ (a product
of $N$ affine factors), the intensity degree $2N$, and the cost degree $4N$ — a
polynomial, but an impractically high-degree one (degree $40$ at $N=10$), so in
practice it uses fixed steps ($\alpha/N,\ \beta/N$) or bracketed searches. The Born
advantage in *polynomial degree* is therefore **quantitative and grows with $N$**
(the cost stays quartic regardless of depth). This concerns the cost's degree along
a line, **not** FFT-pass count — on raw arithmetic Born is *dearer* below
$N\approx 6$ and only $\sim\!1.25\times$ cheaper at the batch-1 winning recipe
(§4.4), and in wall-clock the line search is nearly free (§4.5). The categorical
Born wins are the closed-form step and the joint-all-slice quartic.

### 4.2 Line-search coefficients (`_line_search`, per pixel then summed)

With $e=u-I_{\text{dat}}$ (model **minus** data — the sign the code uses; the
residual is $r(a)=e+2av+a^2w=I(a)-I_{\text{dat}}$), $v=\mathrm{Re}(F^{*}D)$,
$w=|D|^2$, and weight $\omega=\text{mask}/(I_{\text{dat}}+1)$:

$$
c_0 = \sum \omega\,e\,v,\quad
c_1 = \sum \omega\,(e\,w + 2v^2),\quad
c_2 = 3\sum \omega\,v\,w,\quad
c_3 = \sum \omega\,w^2 ,
$$

$$
a = \text{solve\_step\_cubic}(c_0,c_1,c_2,c_3,Q,\text{fallback})\times \texttt{ls\_damp}
\quad\text{(capped by \texttt{ls\_cap})}.
$$

The sign of $e$ does **not** cancel: flipping it flips $c_0$ and the $e\,w$ part of
$c_1$ while leaving $2v^2$, $c_2$, $c_3$ untouched — so $e=I_{\text{dat}}-u$ would
minimise nothing. Use $e=u-I_{\text{dat}}$.

`solve_step_cubic` (line 59) drops leading $\approx$0 coefficients, takes
`np.roots`, keeps real roots, and returns the one with lowest $Q$; if the cubic
degenerates (zero direction, fully masked batch) it returns the parent's
fixed-step fallback ($\alpha/N$ or $\beta/N$).

- `ls_damp` (empirically **0.5** is load-bearing). At large batch the
  under-relaxation absorbs the bilinear object–probe cross-term averaged across the
  batch; **but at batch 1 (the optimal setting, §9.4) there is no intra-batch
  overlap**, so that cannot be the reason there. The leading suspect at batch 1 is
  an **objective mismatch**: the update *direction* is the gradient of the
  *amplitude* cost ($\sqrt{I_{\text{dat}}/u}-1$), while the *step* minimises the
  Gaussian *intensity* quartic $Q(a)$. Different objectives, different curvature —
  the exact intensity-optimal step overshoots the amplitude-descent direction, and
  $0.5$ reconciles them. (Sequential over-relaxation across views also
  contributes.)
- `ls_cap` bounds a single step.

### 4.3 Exact bilinear cross-term handling

After the object step the far field is **exactly** $F+a\,D$ elementwise (no
re-evaluation), so the probe line search in the same batch runs against the
already-updated field. The probe direction's response is $q$ plus its scattered
echo, and its step is the root of another cubic. This handles the object–probe
cross term exactly — something no fixed-step *simultaneous* update does.

Directions themselves are identical to the parent's preconditioned gradients; only
the **step sizes** change ($\alpha/N,\beta/N \to$ exact minimisers). `momentum`
mixes the previous displacement into the direction *before* the line search (heavy
ball with an exact step $\approx$ conjugate gradient). Cost $\approx 1.5\times$ the
parent per object step, $2\times$ when the probe also updates — **in FFT-pass count**
(§4.4); in wall-clock the overhead is only $\sim\!7\%$ (§4.5).

### 4.4 Cost vs FastThreePIE — honest accounting (the $O(1)$-vs-$O(N)$ framing oversells)

Counting $(B,M,n_y,n_x)$-sized FFT passes per batch:

| | object only | with probe | at $N=10$, +probe |
|---|---|---|---|
| FastFirstBorn      | $3N + 6$      | $5N + 8$      | 58 |
| FastThreePIE (`exact`) | $N^2/2 + 2.5N$ | $N^2/2 + 3.5N$ | 85 |

The $N^2/2$ in multislice is unavoidable: slice $j$ is pushed through $N-1-j$
remaining stages, $\sum(N-1-j)=N(N-1)/2$. But two facts undercut §1.1(1)'s
depth-$O(1)$-vs-$O(N)$ framing:

1. **The curves cross at $N\approx 6$.** $5N+8 = N^2/2+3.5N \Rightarrow N^2-3N-16=0$,
   root $5.77$. Below six slices FastFirstBorn is the *more expensive* engine per
   batch ($N=2$: 18 vs 9): Born pays three separate `_scattered` sums (forward,
   object LS, probe LS) at $N$ transforms each, while the cascade is $N-1$
   propagations reusing stored `psis`. The $O(1)$-depth claim describes the affine
   *structure*, not the arithmetic at small $N$.

2. **At the winning recipe the margin is $\sim\!1.25\times$, not "3–4×".**
   `self._phi_f` is invalidated on every probe update (`self._phi_f = None` at the
   end of `_batch_update`); at batch 1 with `probe_update_interval = 1` that is
   every batch, and recomputing $\phi$ costs $N\!\cdot\!M$-sized transforms —
   another $\sim\!N$ units. Born's real cost becomes $6N+8=68$ vs multislice's 85.
   The docstring's "3–4× a ThreePIETorch iteration" is against *plain* (non-fast)
   ThreePIE and must not be carried across to FastThreePIE.

Memory tells the same story: Born holds $\phi$ at $N\!\cdot\!M$ + a few
$B\!\cdot\!M$; multislice holds `psis` at $N\!\cdot\!B\!\cdot\!M$ + `incs`. Born's
advantage scales with $B$ — and $B=1$ in the winning recipe, where
$N\!\cdot\!B\!\cdot\!M = N\!\cdot\!M$ and the advantage largely evaporates.

**Honest statement:** comparable per-batch cost at $N=10$, an advantage that grows
beyond it, a penalty below $N\approx 6$. The categorical wins are the closed-form
step and the joint-all-slice quartic (§4), not raw FLOP count.

### 4.5 Wall-clock is bandwidth-bound, not FFT-bound (measured — Item 0 profile)

A profiler run on one steady-state `_batch_update` at the winning recipe
($B=1$, $N=10$, $M=2$, $512^2$, probe on; `profile_batch_update.py`) settles the
op-count-vs-wall-clock question:

| metric | value |
|---|---|
| wall time (sync-bracketed) | 23.89 ms |
| CUDA kernel time (self) | 23.77 ms |
| **ratio CUDA/wall** | **0.995** |
| kernel launches | 1053 |
| host↔device syncs | 62 |
| **FFT share of kernel time** | **12.9 %** |

The batch is **compute/bandwidth-bound**, not launch- or sync-bound: the GPU is
busy 99.5 % of wall time, only $\sim\!0.12$ ms ($0.5\%$) is non-overlapped host
stall, and the 1053 launches / 62 syncs hide under async kernels. The cost lives in
**elementwise arithmetic** over the $\sim\!42$ MB $(N,B,M,n_y,n_x)$ tensors
($\texttt{aten::mul}$ alone $\approx 27\%$; elementwise kernels $\gtrsim 60\%$
combined), while **FFTs are only $12.9\%$**.

Two direct consequences:

- **The exact line search is nearly free in wall time.** It adds roughly one extra
  forward propagation per line search (object: $D=\mathcal{F}_{\mathrm{ff}}\{
  \text{scattered}(d_p\!\cdot\!\phi)\}$; probe: $q$, `qz`, $D_P$) plus cheap
  reductions for $c_0..c_3$ and a host-side 4-coefficient solve. In FFT-pass count
  that is the $\sim\!1.5\times$ of §4.3, but since FFTs are only $12.9\%$ of the
  batch it costs only $\sim\!7\%$ wall: **measured 46 s/iter (fast) vs 43 s/iter
  (standard, fixed step)** on the 3 mm/30-slice sample (§9.6), for a large accuracy
  gain (proj corr $0.904$ vs $0.807$).
- **The perf levers that would help are fusion of the elementwise passes** (cutting
  memory traffic over the big tensors), *not* collapsing syncs or launch counts —
  those are already hidden. This reverses the naive "reduce launches" instinct.

### 4.6 Known numerical issue — float32 overflow in the coefficients (unfixed)

`FastFirstBornTorch._line_search` forms $c_0..c_3$ in native float32.
`fastthreepietorch._batch_update` deliberately promotes $e,v,w,\omega$ to float64
first, with the comment: early sweeps see residuals $\sim\!10^{10}$, so $w^2$ in the
coefficients reaches $\sim\!10^{40}$, past float32's $3.4\times10^{38}$. That is
**overflow to $\infty$, not precision loss** — $c_3=\infty$,
`solve_step_cubic`'s `np.isfinite` guard fires, and the step silently falls back to
$\alpha/N$ (a line search that looks like it is working but has quietly stopped
searching, visible only as `_steps_o` clustering at the fallback). **Measured on
F1cold:** it did *not* bite — object median step $0.262$ (fallback would be exactly
$0.1$), probe median $3.46\to3.76$ with live spread, no clustering. So F1cold's
residual scale stayed inside float32. Still worth the one-line `.double()`
promotion in FastFirstBorn for parity and safety on hotter inits. (When promoting,
cast *before* the products — casting after the `.sum()` does not prevent the
float32 $w^2$ overflow.)

---

## 5. Preconditioners (object & probe denominators)

Set by `object_denom` (argv[4] in the runner). The denominator is what makes or
breaks the thick, out-of-focus recon.

| mode | object denominator $\text{dn}_o$ | behaviour |
|---|---|---|
| `max` | $\text{peak\_K}$ (scalar per slice) | Uniform, well-bounded step. **Starves off-focus pixels** — where $|\phi_j|^2\ll\text{peak}$, updates are tiny; concentrates residual error at the focal caustic. Stable "safe" choice. |
| `local` | $\text{acc\_K} + \text{peak\_K}\cdot\text{reg}$ | Per-pixel Tikhonov-floored inverse. Spreads updates to off-focus pixels but **amplifies noise** where $K$ is small (the multislice "poison" if `reg` too small). |

- $\text{reg}=\texttt{denom\_reg}$ is the Tikhonov floor as a fraction of $\text{peak\_K}$.
- Probe side mirrors this: $\text{dn}_p=\text{peak\_P}$ (`max`) or
  $K_P+\text{peak\_P}\cdot\text{reg}$ (`local`).

**Caveat — `local` degenerates to `max` on the caustic slice.** $\text{peak\_K}$ is
taken *per slice*. On the focal slice (§9.3) $K=|\phi_j|^2$ spans $\sim\!4$ orders
of magnitude, so $\text{reg}=0.01$ floors the denominator at 1 % of an enormous
peak — i.e. `local` behaves like `max` exactly where the object is most
under-updated, while behaving as intended on the defocused slices. Flooring
relative to the slice's *median* (or mean) $K$ rather than its peak would restore
the intended per-pixel behaviour on the caustic slice.

Prior multislice/3PIE campaign: `local` with too small `reg` blew up the 10-slice
recon; `max` + exact per-slice steps was the fix (memory
`fastthreepie-max-fixes-multislice`). For Born probe recovery the same caution
applies — see §9.

**Aside — FastFirstBorn's probe $K_P$ is correct; FastThreePIE's is cosmetic.**
`fastfirstborntorch.py:441` builds $K_P=\sum_{N,B}|O|^2$ over all slices and the
batch (`dim=(0,1)`, $O$ is $(N,B,n_y,n_x)$) — the right probe-side diagonal weight.
`fastthreepietorch.py:305` instead uses $M\cdot|\text{ob\_p}[0]|^2$ — only slice
0's transmission, with a uniform scalar $M$. Both are **inert there**: the $M$
cancels in $K_P.\text{amax}()$ and in $\text{reg}\cdot K_P.\text{amax}()$, and under
the exact per-slice step the preconditioner's *magnitude* is absorbed by the
line-search step anyway. So the multislice ceiling is unaffected, but the slice-0
form should not be read as a correct cascade preconditioner.

---

## 6. Probe constraints (the probe-recovery machinery)

The probe is unknown on the target sim (unless `truth_probe=1` injects and freezes
the exact generating probe, §2.2/§7). Recovering it from scratch previously failed
(collapse to a hot central spike, $\text{peak/mean}$ 300–2500 vs truth $\approx24.6$,
object proj corr $\approx0$). These constraints are applied at the end of
`_batch_update` (after the probe line search), each gated by its own flag:

```python
self._apply_probe_fourier()      # bandlimit / aperture in the far field
self._apply_probe_modulus()      # replace far-field modulus with measured envelope
self._apply_probe_rsamp()        # real-space amplitude lock (Fienup / two-plane)
self._normalize_probe_power()    # fix the probe/object scaling gauge
self._phi_f = None               # invalidate cached phi_j (probe changed)
```

### 6.1 Fourier aperture / bandlimit — `_apply_probe_fourier` (`probe_fourier_frac`)

Re-projects the probe onto a circular Fourier mask of radius
$\texttt{pffrac}\cdot f_{\text{Nyq}}$ each update. Removes the broadband central
spike that neither the denominator nor a real-space support can suppress.

### 6.2 Far-field modulus constraint — `_apply_probe_modulus`

**The physical leverage.** The Born detector is an **in-line hologram**:

$$
F_j = U_j + S_j,\qquad
U_j = \mathcal{F}_{\mathrm{ff}}\{P_j\} = \mathcal{F}_{\mathrm{ff}}\{P\}\,e^{i k\cdot r_j},\qquad
S_j = \mathcal{F}_{\mathrm{ff}}\{P\,(T-1)\},
$$

$$
I_j = |U|^2 + |S|^2 + 2\,\mathrm{Re}(U^{*}S) .
$$

The direct-beam **modulus is position-independent** ($|U_j|=|\mathcal{F}_{\mathrm{ff}}\{P\}|$).
The position average of the field is
$\langle \mathcal{F}_{\mathrm{ff}}\{P g_j\}\rangle = \mathcal{F}_{\mathrm{ff}}\{P\bar g\}$
with $\bar g$ the position-averaged object patch. For any solid sample
$\bar g\neq 0$, so the cross-term does **not** simply vanish:

$$
\langle I\rangle \approx |\mathcal{F}_{\mathrm{ff}}\{P\}|^2
\;+\; 2\,\mathrm{Re}\!\big[\mathcal{F}_{\mathrm{ff}}\{P\}^{*}\,
\mathcal{F}_{\mathrm{ff}}\{P\bar g\}\big]
\;+\; \langle|S|^2\rangle .
$$

The middle (first-order) term is generally *larger* than the $\langle|S|^2\rangle$
term that §6.3 debiases. **Why the modulus constraint still works:** if $\bar g$ is
roughly a uniform complex constant over the probe footprint,
$\mathcal{F}_{\mathrm{ff}}\{P\bar g\}\approx \bar g\,\mathcal{F}_{\mathrm{ff}}\{P\}$
and the middle term collapses to
$2\,\mathrm{Re}(\bar g)\,|\mathcal{F}_{\mathrm{ff}}\{P\}|^2$ — a pure
multiplicative *scale* (a gauge), absorbed by `_normalize_probe_power` (§6.5). So
pinning $|\mathcal{F}_{\mathrm{ff}}\{P\}|$ is right, but because the residual is a
gauge, not because cross-terms average to zero. **Where it breaks:** at sample
edges and for heterogeneous samples $\bar g$ is not uniform,
$\mathcal{F}_{\mathrm{ff}}\{P\bar g\}$ is not proportional to
$\mathcal{F}_{\mathrm{ff}}\{P\}$, and this first-order term distorts the envelope
*shape*. Treating

$$
\langle I\rangle \approx |\mathcal{F}_{\mathrm{ff}}\{P\}|^2 + \langle|S|^2\rangle
$$

as the working model is therefore an approximation valid under near-uniform
$\bar g$, not an identity.

`_apply_probe_modulus` forward-FFs the probe, replaces the mode-summed far-field
amplitude with the precomputed envelope `self._pr_env`, keeps the phase, and
back-FFs. The default envelope is

$$
\texttt{self.\_pr\_env} = \sqrt{\langle I\rangle}
= \sqrt{\text{amp\_dat}^2.\text{mean}(0)} .
$$

This pins $|\mathcal{F}_{\mathrm{ff}}\{P\}|$ (kills the spike, a wrong envelope) and
leaves only the far-field **phase** free — exactly the gauge the update should move.

**Bias:** $\sqrt{\langle I\rangle}$ OVERESTIMATES $|\mathcal{F}_{\mathrm{ff}}\{P\}|$
by the scattered power $\langle|S|^2\rangle$.

### 6.3 Self-consistent (debiased) modulus — `probe_modulus_selfcons`

Removes the $\langle|S|^2\rangle$ bias by measuring the scattered power from the
current model and subtracting it from the envelope target:

$$
\text{target } |\mathcal{F}_{\mathrm{ff}}\{P\}|
= \sqrt{\big(\langle I\rangle - \langle|S|^2\rangle\big)_+ } .
$$

Implementation:
- `engine_initialize`: cache `self._pr_Idat_mean` $=\langle I\rangle$, set
  `self._pr_env` $=\sqrt{\langle I\rangle}$, allocate `self._pr_S2_acc`.
- `_batch_update` (after $F,u$): accumulate scattered power $S=F-U$,
  `self._pr_S2_acc += `$\sum_m|S|^2$, `self._pr_S2_n += B`, with
  $U=\mathcal{F}_{\mathrm{ff}}\{\text{pr\_g}\}$ the unscattered field.
- `engine_iterate` (end of epoch): update
  `self._pr_env` $=\sqrt{(\text{Idat\_mean}-\text{s2\_mean})_+}$, log the
  scattered-power fraction, reset the accumulator.

Targets the plateau where the probe $\text{peak/mean}$ sticks $\approx24$ (truth
$\approx29$) under the fixed $\sqrt{\langle I\rangle}$ envelope. **Code written;
test F2 pending.**

**Stability requirement (not yet implemented).** The target is now model-dependent:
envelope $\leftarrow$ object $\leftarrow$ probe $\leftarrow$ envelope. If the model
under-explains scattering, $\langle|S|^2\rangle$ comes out too small, the envelope
too large, the probe strengthens, and the loop self-reinforces. The current
per-epoch outright *replacement* has no damping. It should use an **EMA** of
$\langle|S|^2\rangle$ (suggest $\alpha\approx0.3$) and a **clamp on the per-epoch
fractional change of `_pr_env`** (suggest $\pm5\%$): the clamp is the hard
non-divergence rail, the EMA the smoothing.

**Validity diagnostic.** $\langle|S|^2\rangle/\langle I\rangle$ is a direct proxy
for the Born small parameter (scattered ÷ total power): above $\sim\!20\%$ the
single-scattering premise is in question and the model-error gap (§1.2) widens.
**Measured on the 10-slice sample we are far below that:**
$\varepsilon_{\text{born,direct}} = 0.022$, $g_{\text{mean modulus}} = 0.033$ (mean
$|O-1|$), ceiling gap $0.030$ — all at the few-% level, a genuinely weak-scattering
regime. (The 30-slice/dense sample scatters more per unit thickness; see §9.6.)

### 6.4 Real-space amplitude lock — `_apply_probe_rsamp` (`probe_rsamp_frac`)

A Fienup-style two-plane constraint: locks $|P|$ in real space to a precomputed
envelope `self._pr_rs_env`, keeping the phase. Combined with §6.2 this is a
dual-space (modulus + real-space amplitude) lock. **Caveat:** rsamp CAPS the probe
amplitude to the (imperfect) init envelope, so it cannot grow toward truth —
modulus-only outperformed dual-lock in this campaign (§9).

### 6.5 Power normalisation — `_normalize_probe_power` (gauge fix)

Rescales `self.pr_g` so its total power equals the measured per-frame flux. Removes
the runaway probe/object scaling degree of freedom (the bilinear model is invariant
under $P\to cP,\ O\to O/c$).

---

## 7. Probe initialisation (config-only, via ptypy `illumination`)

The probe init is seeded straight from ptypy's illumination model into
`self.pr_g` (`firstborntorch.py:445`). Two levers matter:

- **`illumination.model = 'stxm'`**: back-propagates $\sqrt{\langle I\rangle}$ —
  pins the far-field envelope from the averaged diffraction, exactly the quantity
  the failed runs got wrong.
- **`illumination.propagation.parallel` (defocus)**: converts the flat-phase
  far-field spike into the correct broad, defocused real-space probe. **The sign is
  critical.**

When `truth_probe=1`, this init is overridden entirely: the exact generating probe
(§2.2) is injected and frozen (`probe_update_start = niter+1`), which is how the
frozen-probe ceilings of §9.1 are measured.

### 7.1 The wrong-sign defocus finding (root cause of the whole failure)

The prior "probe collapse" traces almost entirely to the init defocus being the
**wrong sign**. The truth probe focuses at $-625\ \mu\text{m}$; the correct ptypy
`illumination.propagation.parallel = +625\ \mu\text{m}$ (NOT the $-570$ tried
earlier).

- Wrong-sign init correlates **0.002** with truth.
- **peak/mean calibration is sign-blind** — a Fresnel focus-scan (correlation vs
  propagation distance) is required.
- Correct-sign stxm-family init ceiling $\approx$ **0.635** correlation with truth.

**Why no constraint can fix it (twin-image ambiguity).** Defocus is a pure
quadratic phase in the far field. The modulus constraint (§6.2) pins
$|\mathcal{F}_{\mathrm{ff}}\{P\}|$ and frees far-field *phase* — so it cannot
distinguish $+z$ from $-z$ by construction. The real-space amplitude lock (§6.4)
pins $|P|$, also sign-blind for a near-symmetric probe. This is the twin-image
ambiguity of in-line holography that §6.2's $F=U+S$ framing predicts. **The init
must supply the defocus sign; no probe constraint can recover it.**

---

## 8. Warm-start (`warm_start_h5`, `warm_start_ptyr`)

Continue a run from a prior recon instead of restarting:
- `warm_start_h5` : an `fslices` h5 whose `content/objects/Cobj_o_{0..N-1}` slices
  seed `self.ob_g` (slice count must match $N$).
- `warm_start_ptyr` : a `.ptyr` whose `content/probe` seeds `self.pr_g` (shape must
  match).

Runner (`run_fastborn_100um_het.py`) takes an absolute `warm_start_dir` (argv[18])
and globs the latest `*iter-*.h5` and `*.ptyr`. **Must be absolute** — the runner
`os.chdir`s to the output dir before the glob runs. Validated: E1 resumed exactly
at D4's Fourier $0.285$ / peak-mean $24.1$ with no restart transient.

---

## 9. Empirical findings

### 9.1 Frozen-truth-probe ceilings (upper bounds — and inverse-crime-flattered)

Both measured at **batch 1**, `max` denom, 100 iters, the **exact generating
probe** (§2.2) injected and **frozen** (`probe_update_start = 101 > niter`), so only
the object is solved; `response_mode = exact` for 3PIE. Data noiseless.
Runs: FastBorn `run_20260911_212258`, FastThreePIE `run_20260911_214617`.

| engine | batch | frozen | resp_mode | diag | proj corr |
|---|---|---|---|---|---|
| FastBorn | 1 | yes | n/a | 0.911 | 0.968 |
| FastThreePIE | 1 | yes | exact | 0.963 | 0.998 |

**Vs F1cold: NOT batch-confounded.** Both are batch 1, so F1cold's proj $0.967$ vs
the FastBorn ceiling $0.968$ is a matched comparison — "essentially at ceiling" is
real.

**Between the two ceilings: confounded — read §1.2 and §2.3.** The $0.030$ gap
mixes (i) the **inverse crime** (3PIE reconstructs with the generating operator, so
its ceiling is set by optimisation not physics; Born pays a real model-mismatch
floor), and (ii) an **update-cadence** handicap ($N$ per-slice 3PIE steps/view vs
one Born step). It is an upper bound on the Born model error, not a measurement.

### 9.2 Probe-from-scratch progression (10-slice sim, correct +625 µm init)

| run | recipe | batch | diag | proj | probe / pk-mean |
|---|---|---|---|---|---|
| C2 | frozen +625 init | 32 | 0.432 | 0.119 | (init) |
| D2 | dual-lock (modulus + rsamp) | 32 | 0.375 | 0.111 | 0.629→0.661 |
| D4 | modulus-only, 100 it | 32 | 0.607 | 0.199 | 0.751 / 23.4 |
| E1 | modulus-only, warm-start D4, +100 it | 32 | 0.716 | 0.286 | pk-mean 23.4 |
| **F1cold** | **modulus-only, cold stxm, 100 it** | **1** | **0.909** | **0.967** | pk-mean **23.6** |
| T1cold | FastThreePIE, cold stxm, 100 it | 1 | 0.958 | 0.963 | — |

**BATCH SIZE IS THE DECISIVE FACTOR for probe recovery — given the correct recipe.**
At batch 1 the cold modulus-only run reaches proj corr **0.967** in 100 iters,
matching the batch-1 frozen ceiling ($0.968$). The batch-32 runs (D4/E1), *same
recipe*, plateaued at $0.199$–$0.286$ with $\text{peak/mean}$ stuck at $23.4$: batch
32 averages the probe update over $\sim\!32$ views ($\sim\!21$ updates/epoch) into a
spiky-probe local minimum. Batch 1 does $\sim\!670$ updates/epoch, escapes it, and
the focal caustic (§9.3) clears by $\sim$iter 25. **Caveat:** batch 1 is necessary
but not sufficient — the earliest batch-1 attempts failed for lack of the +625 µm
defocus and/or the modulus constraint.

**Cold-probe Born $\approx$ cold-probe 3PIE.** From a cold stxm init at batch 1,
FastBorn (0.967 proj / 0.909 diag) slightly *beats* FastThreePIE (0.963 / 0.958) on
projection correlation while trailing on per-slice diag — despite Born's
inverse-crime handicap (§2.3). 3PIE's diag edge is depth localisation; Born's proj
edge is faster projected convergence at batch 1.

**proj > diag near convergence is normal.** Both frozen references also have
proj > diag; the failed rows have diag > proj because per-slice patches carry local
correlation while the projected (depth-summed) structure — needing the inter-slice
phases right — is still wrong. The reversal reflects convergence.

**Winning recipe:** cold stxm init at the **correct +625 µm defocus**,
`illumination.model='stxm'`, far-field **modulus-only** constraint, `max` denom,
`ls_damp 0.5`, **batch 1**, 100 iters. No warm-start needed.

> **Metric provenance & cross-batch caveat.** `diag mean`, `proj corr`, `probe corr`
> are **external** metrics (correlations vs the truth h5 by `sim_report`), hence
> batch-size-independent and safe to compare across runs. The engine-printed
> `Errors :: Fourier`/`Photons` are **per-batch-normalised** ($\sim\!1/B$) and must
> **not** be compared across batch sizes; the `Exit` term is a bare sum and is
> batch-independent. **Compare cross-batch runs only via the external metrics.**

### 9.3 The focal-point artifact (diagnosed, not a bug)

The probe waist falls **inside the sample at slice $\sim\!6$** ($\sim\!600\ \mu\text{m}$
in), $\text{peak/mean}\approx 12716$ at the caustic. The central caustic dumps
amplified *probe error* into the object as a dark vortex + Fresnel rings on the mid
slices. User confirmed **no real structure** there. The artifact **shrinks as the
probe refines** — a probe-error symptom, not an object feature and not a
preconditioner bug. `max` denom concentrates it at the centre; `local` (with
adequate `reg`) spreads updates but risks noise blow-up (and here degenerates to
`max`, §5).

**A scalar step cannot serve a stack this unequal.** $\text{peak/mean}\approx 12716$
means $K_j=|\phi_j|^2$ varies by $\sim\!4$ orders of magnitude *within* the focal
slice, and slices differ enormously in their $K$ distributions. One scalar
line-search step per batch is a compromise. Restricting $Q$ to
$g + \sum_j a_j d_j$ keeps it an **exact multivariate quartic** in
$a\in\mathbb{R}^N$, still computable from a handful of masked sums.

**This is the one improvement multislice provably cannot copy** (§4): the joint
$N$-slice quartic exists only because Born is quartic along *any* direction. Its
gradient/Hessian need the **cross-slice sums** $\mathrm{Re}(D_j^{*}\!\cdot D_k)$
(and $\mathrm{Re}(F^{*}D_j)$, $|D_j|^2$) — the code currently forms only the
**combined** direction $D=\sum_j D_j$ and its scalar $v=\mathrm{Re}(F^{*}D)$,
$w=|D|^2$, so the per-slice/cross-slice sums are not yet computed. It is therefore a
coupled cubic system (the gradient of a quartic), not one Newton solve. Flagged as
future work.

*Batch 1:* the focal artifact clears by $\sim$iter 25 (F1cold), markedly faster than
batch 32 — consistent with the many more probe updates per epoch.

### 9.4 Batch size

**Batch 1 is optimal** (re-confirmed by F1cold). Batch 1 = probe updated every view
($\sim\!670$/epoch) vs $\sim\!21$ at batch 32. Larger batches average the probe
update and slow recovery.

### 9.5 Never trust Fourier-below-floor

A "breakthrough" with Fourier error below the frozen-probe floor is **overfitting**,
not success (the earlier B4 false positive: Fourier below floor but proj corr
$0.070$). Always judge by **projection correlation**, not Fourier error alone.

### 9.6 Thick-sample stress test — 3 mm / 30 slices, dense specimen

Sample: `PARTICLE_COPIES = 60` (∼4 blobs/slice vs ∼1 in the 10-slice sim),
100 µm spacing, 30 slices (3 mm). 5 iterations, **frozen truth probe** (§2.2), plot
every iteration, `max` denom, `ls_damp 0.5`, batch 1.

| engine | proj corr | diag | time/iter | note |
|---|---|---|---|---|
| **FastFirstBorn** | **0.904** | 0.683 | ~46 s | Fourier err $16.5\to1.84$ |
| **FastThreePIE** (`exact`) | 0.853 | **0.700** | ~128 s | Fourier err $41\to8.8$; **inverse-crime advantaged** |
| **std FirstBorn** (fixed step) | 0.807 | 0.647 | ~43 s | — |
| **std ThreePIE** (`object_step 1`) | 0.238 | 0.428 | ~49 s | **stalled**, Fourier $51\to50$ |

Findings:

1. **The exact line search earns its keep for Born and is nearly free (§4.5):**
   FastBorn 0.904 vs std Born 0.807 proj corr, for $\sim\!7\%$ extra wall time
   (46 vs 43 s/iter).

2. **FastBorn leads on projected fidelity and speed** (0.904, $\sim\!2.8\times$
   faster/iter than FastThreePIE) even though FastThreePIE enjoys the inverse crime
   (§2.3). FastThreePIE keeps a slight per-slice **diag** edge (0.700) at high cost.

3. **std ThreePIE stalls — an under-stepping tuning fault, not a bug.** Same `max`
   denom, same cascade as FastThreePIE; only the step rule differs. Std uses a fixed
   $\text{ob\_step}=1.0$ applied *uniformly to all 30 slices*
   (`threepietorch.py:781`, $\text{dob}=\text{ob\_step}\cdot\text{grad}/\text{pn}$),
   which under-steps the deep-slice cascade so the Fourier error barely moves. Step
   scan (`OBJSTEP` env, `run_stdthreepie_100um_het.py`): step 2.0 unsticks it
   (Fourier $51\to24.4$ at iter 1 and descending). FastThreePIE's exact per-slice
   line search removes the hand-tuning by solving for the optimal step per slice per
   view — which a 30-slice cascade (each slice wanting a different step magnitude)
   genuinely needs. See memory `threepie-n10-lowfreq-instability`.

These are **5-iteration** numbers and still descending; for Born-as-preconditioner
that is the relevant regime (early-iteration quality), so plateau values are not the
figure of merit.

### 9.7 Note on the plot-interval fix

`threepietorch.py` hardcoded `if self.curiter % 5 == 0: self._plot_progress()`,
ignoring `plot_interval`; FastThreePIE inherits this via `super().engine_iterate`.
Fixed to `if self._plot_int > 0 and self.curiter % self._plot_int == 0`
(matching `firstborntorch.py`), so both engines honour `plot_interval`.

---

## 10. Queued / completed tests

- **F1cold** — *done.* Batch 1, modulus-only, cold stxm init. proj 0.967 / diag
  0.909, at the ceiling (§9.2).
- **T1cold** — *done.* FastThreePIE (`exact`), cold batch 1, +625 µm, 100 it. proj
  0.963 / diag 0.958 (§9.2).
- **3 mm / 30-slice four-way** — *done* (§9.6).
- **F2** — self-consistent modulus (`pmsc=1`), batch 1 (§6.3 debiasing; watch
  $\text{peak/mean}$ climb from 23.6 toward $\approx29$). Still motivated.
- **F3** — `local` object denom + tuned `denom_reg` (focal-slice artifact).
- **Vector line search** — the joint $N$-slice quartic (§4, §9.3): the one
  structural win multislice cannot copy. Needs the cross-slice
  $\mathrm{Re}(D_j^{*}\!\cdot D_k)$ sums the code does not yet form.

---

## 11. Key file map

| item | location |
|---|---|
| Forward model, inversion, overlap property | `firstborntorch.py` (header) |
| Object/probe Jacobi update, warm-start | `firstborntorch.py` (`engine_initialize`, `_batch_update`) |
| Exact line-search rationale | `fastfirstborntorch.py` (header) |
| `solve_step_cubic` | `fastfirstborntorch.py:59` |
| `_line_search` (coefficients) | `fastfirstborntorch.py` |
| `_batch_update` (forward, precond, line searches, constraints) | `fastfirstborntorch.py` |
| Probe modulus / rsamp / power / self-consistent modulus | `fastfirstborntorch.py` |
| **Data generator (exact multislice, inverse crime)** | `multislicesim.py` (`MultisliceSimulator._simulate`) |
| Multislice reconstruction engines | `threepietorch.py`, `fastthreepietorch.py` |
| Profiling harness (Item 0) | `profile_batch_update.py` |
| Runners (argv, engine wiring) | `run_fastborn_100um_het.py`, `run_fastthreepie_100um_het.py`, `run_stdborn_100um_het.py`, `run_stdthreepie_100um_het.py`, `run_multislice_sim_100um_het.py` |
