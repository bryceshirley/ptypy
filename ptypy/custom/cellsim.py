# -*- coding: utf-8 -*-
"""
In-situ cell simulator: contaminated windows around a sample, on a dense
slice grid.

WHAT THIS IS
------------
A subclass of MultisliceSimulator for the geometry Darren described: a
specimen inside an in-situ cell (liquid, gas, electrochemical, heating),
with beam-induced contamination -- carbon deposition, salt crust,
condensation -- on the windows, sitting at a different depth from the thing
you actually want to image.

The question it answers is whether a depth-resolving forward model keeps the
contamination OFF the sample plane. Not whether it reconstructs the
contamination well: nobody cares about the crud. That is a far weaker
requirement than the cross-model comparison in the paper, and it is the one
that matters at a beamline.

WHY THE GRID IS DENSE AND THE TRUTH IS SPARSE
---------------------------------------------
A first draft of this engine used three slices -- window, sample, window --
and that was wrong for four reasons.

1. COST. Table 3 of the paper puts the ISS/multislice crossover at
   N ~ 3: below it the Born model is SLOWER (48.5 vs 42.2 ms at N=2). A
   three-slice demonstration would win on physics and lose on wall clock,
   which is the trap the N=2 X-ray section already fell into. At N = 10-20
   the cost claim and the accuracy claim point the same way for the first
   time in the paper.

2. SAMPLING. The slice spacing is set by the depth of field, not by where
   the structure happens to be. At 8 keV and NA 1.20 mrad the DOF is 108 um,
   so a 1 mm cell wants ~10 slices whatever the truth looks like. Three
   slices is a coarse discretisation of a thick volume, not a model of one.

3. YOU DO NOT KNOW WHERE THE WINDOWS ARE. Not to the ~50 um a three-slice
   reconstruction would require you to specify. On a dense grid you do not
   have to: the reconstruction localises the structure and the depths are
   read off afterwards. That is both a stronger demonstration and a more
   honest one.

4. THE EMPTY SLICES ARE THE POINT. With structure on 3 of 13 slices, most
   of the C(N,2) neglected cross terms of Eq. (6) contain a g_k that is
   identically zero, so they vanish exactly rather than approximately --
   the sparse-support argument of the paper's Sec. 2.1, arriving as a
   measurement instead of an assertion.

A corollary worth stating: a uniform liquid fill contributes a CONSTANT per
slice, which by Eq. (22) is exactly null and absorbed into the probe scale.
The electrolyte is free. Only its structure costs anything.

THE THREE SEPARABILITY SCALES
-----------------------------
Depth separability requires a <~ a_c = 2 * NA * dz (paper Eq. 21), and with
a dense grid there are three relevant dz, not one. For NA = 1.20 mrad,
13 slices over 1.2 mm (100 um spacing), windows at the faces and the sample
in the middle:

    adjacent slices,  dz = 100 um  ->  a_c = 240 nm   (14 object px)
    window to sample, dz = 600 um  ->  a_c = 1.44 um  (85 px)
    window to window, dz = 1.2 mm  ->  a_c = 2.88 um  (171 px)

So contamination finer than ~240 nm can in principle be pinned to a single
slice; between 240 nm and 1.44 um it smears axially over several slices but
stays clear of the sample plane, which is all the application needs; above
1.44 um it leaks into the sample plane and no forward model will prevent
that. Three regimes, one sweep -- see the runner notes.

USAGE
-----
As MultisliceSimulator with the engine name swapped:

    p.engines.engine00.name       = 'CellSimulator'
    p.engines.engine00.n_slices   = 13
    p.engines.engine00.cell_span  = 1.2e-3
    p.engines.engine00.window_z   = (0.0, 1.2e-3)
    p.engines.engine00.sample_z   = (0.6e-3,)
    p.engines.engine00.contam_feature_nm = 400.
"""

import numpy as np

from ptypy.engines import register
from ptypy.utils.verbose import logger

from . import multislicesim

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@register()
class CellSimulator(multislicesim.MultisliceSimulator):
    """
    Simulate an in-situ cell on a dense slice grid: contaminated windows and
    a sample at specified depths, empty slices everywhere else.

    Defaults:

    [name]
    default = CellSimulator
    type = str
    help = Engine identifier

    [n_slices]
    default = 13
    type = int
    help = Number of slices spanning the cell
    doc = Set by depth-of-field sampling, not by the number of structured
          planes. The engine logs spacing/DOF at startup and warns outside
          roughly 0.5-2: much above 2 under-samples the volume, much below
          0.5 makes adjacent slices near-degenerate (their difference
          approaches the exactly-null constant direction of Eq. 22) and
          wastes unknowns. 10-20 is the useful range on this geometry, and
          it is also where Table 3 puts ISS ahead of multislice on
          cost.

    [cell_span]
    default = 1.2e-3
    type = float
    help = Total axial extent of the slice grid in meters
    doc = Slices are equally spaced over this span, so the separation is
          cell_span / (n_slices - 1). Overrides sim_thickness. Should cover
          the whole cell including both windows, since anything outside the
          grid is folded onto the end slices.

    [window_z]
    default = (0.0, 1.2e-3)
    type = tuple
    help = Axial depths carrying window contamination, in meters
    doc = Snapped to the nearest slice of the grid; the snapping error is
          reported. Use a one-element tuple for a single contaminated
          window, which is the simpler test and the one to run first -- two
          contaminated planes add an ambiguity about which window a given
          deposit belongs to, on top of the question being asked.

    [sample_z]
    default = None
    type = tuple
    help = Axial depths carrying the sample, in meters
    doc = None places the sample at the centre of the span. A multi-element
          tuple distributes the segmented particles over several planes,
          which is the realistic case for anything suspended in a liquid
          rather than sitting on a membrane.

    [contam_feature_nm]
    default = 400.
    type = float
    help = Characteristic contamination feature size in nanometres
    doc = FWHM of the smoothing applied to the random field: roughly the
          diameter of one deposit. THIS IS THE EXPERIMENT'S CONTROLLED
          VARIABLE. The engine logs it against all three a_c scales at
          startup. Sweep it from well below the adjacent-slice a_c to well
          above the window-to-sample a_c; the prediction from Eq. (21) and
          the measured linear-in-q suppression is that leakage into the
          sample plane rises smoothly across the latter rather than
          switching on at it.

    [contam_phase]
    default = 0.12
    type = float
    help = Peak phase shift of the contamination, radians (positive; applied
           as a negative shift to match the sample convention)
    doc = Beam-induced carbon on a nitride window is a weak phase object.
          0.05-0.3 rad spans light deposition to a visibly crusted window.
          Keep it well below the sample's ~1.4 rad so the specimen is sparse
          in contrast as well as in z.

    [contam_beta_delta]
    default = 0.005
    type = float
    help = Absorption-to-phase ratio of the contamination
    doc = beta/delta for low-Z material at 8 keV is of order 1e-2 to 1e-3.
          Nonzero matters for one specific reason: the paper's
          preconditioner argument (Sec. 5.2) turns on a downstream slice
          absorbing, so a purely-phase contamination layer would quietly
          delete the case being compared. Set 0 only deliberately.

    [contam_coverage]
    default = 0.25
    type = float
    help = Fraction of the window area covered by contamination
    doc = Area fraction above the threshold applied to the smoothed random
          field. 0.15-0.4 is plausible for a used window.

    [contam_seed]
    default = 7
    type = int
    help = Seed for the contamination field; offset per window plane

    [contam_within_coverage]
    default = True
    type = bool
    help = Confine contamination to the scanned region
    doc = Structure outside the probe coverage is unrecoverable and only
          drags truth-vs-reconstruction correlations down for reasons that
          have nothing to do with the forward model. On by default so the
          scores mean what they say.

    [liquid_phase]
    default = 0.0
    type = float
    help = Uniform phase shift per empty slice, radians
    doc = A uniform fill contributes a CONSTANT g_j, which by Eq. (22) is
          exactly null -- only the sum over slices is observable and it is
          absorbed by the probe scale. Nonzero here therefore tests that
          degeneracy end to end rather than adding physics: the
          reconstruction should be indifferent to it. Leave 0 for the
          headline runs.

    [sample_copies]
    default = 8
    type = int
    help = Copies of each segmented particle pasted into the sample plane(s)
    doc = Copies may overlap each other and MUST be free to sit under the
          contamination in projection -- contamination that never obscures
          the sample cannot corrupt it, and the test would be trivial. The
          parent's no_overlap placement is therefore disabled.

    [sample_spread_px]
    default = 400
    type = int
    help = Lateral spread of the pasted sample copies, in object pixels
    """

    # ------------------------------------------------------------------ #
    #  Geometry                                                           #
    # ------------------------------------------------------------------ #
    def _cell_geometry(self):
        """Slice grid, structured-plane indices, and the separability scales."""
        nl = int(self.p.n_slices)
        if nl < 2:
            raise ValueError("n_slices must be at least 2")
        span = float(self.p.cell_span)
        z = np.linspace(0.0, span, nl)
        dz = z[1] - z[0]

        na = self._probe_na()
        lam = self._wavelength()
        psize = self._object_pixel()
        dof = lam / max(na ** 2, 1e-30)

        def snap(zz):
            i = int(np.argmin(np.abs(z - float(zz))))
            return i, float(z[i] - float(zz))

        w_idx, w_err = [], []
        for zz in np.atleast_1d(self.p.window_z):
            i, e = snap(zz)
            w_idx.append(i)
            w_err.append(e)

        s_spec = self.p.sample_z
        if s_spec is None:
            s_spec = (span / 2.0,)
        s_idx, s_err = [], []
        for zz in np.atleast_1d(s_spec):
            i, e = snap(zz)
            s_idx.append(i)
            s_err.append(e)

        if set(w_idx) & set(s_idx):
            raise ValueError(
                "a window plane and a sample plane snap to the same slice "
                "(indices %s vs %s) -- increase n_slices or move them apart"
                % (w_idx, s_idx))

        logger.info("cell: %d slices over %.0f um, spacing %.0f um"
                    % (nl, span * 1e6, dz * 1e6))
        logger.info("      NA %.2f mrad, lambda %.4f A, DOF %.0f um"
                    % (na * 1e3, lam * 1e10, dof * 1e6))
        ratio = dz / dof
        logger.info("      spacing / DOF = %.2f%s"
                    % (ratio,
                       "" if 0.5 <= ratio <= 2.0 else
                       ("   <-- UNDER-SAMPLED, raise n_slices"
                        if ratio > 2.0 else
                        "   <-- adjacent slices near-degenerate, "
                        "lower n_slices")))
        logger.info("      windows on slices %s (snap error %s um)"
                    % (w_idx, ["%.0f" % (e * 1e6) for e in w_err]))
        logger.info("      sample  on slices %s (snap error %s um)"
                    % (s_idx, ["%.0f" % (e * 1e6) for e in s_err]))
        n_struct = len(set(w_idx) | set(s_idx))
        logger.info("      z-sparsity: %d of %d slices carry structure "
                    "(%.0f%% empty, so %.0f%% of cross terms vanish exactly)"
                    % (n_struct, nl, 100.0 * (1 - n_struct / nl),
                       100.0 * (1 - (n_struct * (n_struct - 1))
                                / max(nl * (nl - 1), 1))))

        # the three scales that matter, and the feature size against each
        gap_ws = min(abs(z[i] - z[j]) for i in w_idx for j in s_idx)
        scales = {'adjacent slices': dz, 'window to sample': gap_ws}
        if len(set(w_idx)) > 1:
            scales['window to window'] = abs(z[max(w_idx)] - z[min(w_idx)])
        feat = float(self.p.contam_feature_nm) * 1e-9
        ac = {}
        for name, d in scales.items():
            a_c = 2.0 * na * d
            ac[name] = a_c
            logger.info("      a_c(%-17s) = %6.0f nm = %4.0f px   "
                        "feature = %.2f a_c"
                        % (name, a_c * 1e9, a_c / psize, feat / a_c))
        if feat > ac['window to sample']:
            logger.warning("      contamination is COARSER than the "
                           "window-to-sample a_c: expect it in the sample "
                           "slice regardless of forward model")

        return nl, z, w_idx, s_idx, ac

    def _probe_na(self):
        """Power-weighted NA of the probe in use: the radius containing 90%
        of the far-field power, as an angle. Matches the convention the
        paper quotes for the NanoMAX 434 probe (1.20 mrad)."""
        with torch.no_grad():
            F = self._ff(self._ff_fw, self.pr_g.unsqueeze(0))
            I = (F.abs() ** 2).sum(dim=(0, 1)).cpu().numpy()
        ny, nx = I.shape
        rr = np.hypot(np.arange(ny)[:, None] - ny / 2.0,
                      np.arange(nx)[None, :] - nx / 2.0)
        order = np.argsort(rr.ravel())
        cum = np.cumsum(I.ravel()[order])
        cum /= max(cum[-1], 1e-30)
        r90 = rr.ravel()[order][int(np.searchsorted(cum, 0.90))]
        geo = next(iter(self.ptycho.model.scans.values())).geometries[0]
        return float(np.atleast_1d(r90 * geo.psize[0]
                                   / geo.distance).ravel()[0])

    def _object_pixel(self):
        return float(np.atleast_1d(
            next(iter(self.ptycho.model.scans.values()))
            .geometries[0].resolution).ravel()[0])

    def _wavelength(self):
        return float(np.atleast_1d(
            next(iter(self.ptycho.model.scans.values()))
            .geometries[0].lam).ravel()[0])

    # ------------------------------------------------------------------ #
    #  Contamination field                                                #
    # ------------------------------------------------------------------ #
    def _contamination(self, shape, seed):
        """Band-limited random field, thresholded to a target coverage and
        scaled to a target peak phase.

        White noise smoothed to the requested feature size, so the power
        spectrum is compact and the characteristic scale is a parameter
        rather than an emergent property -- which is what makes the sweep
        against a_c interpretable. Returns a complex transmission, exactly
        1 on clean window.
        """
        from scipy import ndimage

        rng = np.random.default_rng(int(seed))
        feat_px = max(float(self.p.contam_feature_nm) * 1e-9
                      / self._object_pixel(), 1.0)
        sigma = feat_px / 2.355                       # FWHM -> sigma

        sm = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
        frac = float(np.clip(self.p.contam_coverage, 1e-3, 0.95))
        thr = np.quantile(sm, 1.0 - frac)

        # hard threshold then a light smooth: deposits get finite edge
        # gradients rather than a binary step
        depth = ndimage.gaussian_filter((sm > thr).astype(float), sigma / 3.0)
        peak = depth.max()
        if peak > 0:
            depth /= peak
        if bool(self.p.contam_within_coverage):
            depth = depth * self._coverage().cpu().numpy()

        phi = float(self.p.contam_phase)
        bd = float(self.p.contam_beta_delta)
        return np.exp((-1j - bd) * phi * depth).astype(np.complex128)

    # ------------------------------------------------------------------ #
    #  Layer assembly                                                     #
    # ------------------------------------------------------------------ #
    def _build_layers(self, src, particles):
        """Dense slice grid: contamination on the window planes, particles
        on the sample plane(s), exactly empty everywhere else."""
        nl, z, w_idx, s_idx, ac = self._cell_geometry()
        self._cell_z = z
        self._cell_ac = ac
        self._cell_w_idx = w_idx
        self._cell_s_idx = s_idx

        ny, nx = src.shape
        layers = np.ones((nl, ny, nx), dtype=np.complex128)

        # --- sample plane(s) ---------------------------------------------
        rng = np.random.default_rng(int(self.p.layer_seed))
        cov = self._coverage().cpu().numpy()
        spread = int(self.p.sample_spread_px)
        copies = max(int(self.p.sample_copies), 1)
        g = src - 1.0
        table = []

        for k, (box, foot) in enumerate(particles):
            for c in range(copies):
                L = int(s_idx[(k * copies + c) % len(s_idx)])
                patch, ft, tf = (
                    self._transform_copy(g[box] * foot, foot, rng)
                    if bool(self.p.transform_copies)
                    else (g[box] * foot, foot, dict(scale=1.0, angle=0.0)))
                if not ft.any():
                    continue
                h, w = ft.shape
                oyc = box[0].start + (box[0].stop - box[0].start) / 2.0
                oxc = box[1].start + (box[1].stop - box[1].start) / 2.0
                dyv = int(rng.integers(-spread, spread + 1)) if spread else 0
                dxv = int(rng.integers(-spread, spread + 1)) if spread else 0
                y0 = int(round(oyc)) - h // 2 + dyv
                x0 = int(round(oxc)) - w // 2 + dxv
                cy0, cx0 = max(-y0, 0), max(-x0, 0)
                cy1 = h - max(y0 + h - ny, 0)
                cx1 = w - max(x0 + w - nx, 0)
                if cy1 <= cy0 or cx1 <= cx0:
                    continue
                dst = (slice(y0 + cy0, y0 + cy1), slice(x0 + cx0, x0 + cx1))
                if not cov[dst].any():
                    continue
                # transmissions multiply: copies may overlap each other and
                # WILL sit under the contamination in projection, which is
                # the whole point of the test
                layers[L][dst] *= 1.0 + patch[cy0:cy1, cx0:cx1]
                yy, xx = np.nonzero(ft)
                table.append(dict(label=k, layer=L, dy=int(dyv), dx=int(dxv),
                                  area=int(ft.sum()),
                                  cy=float(y0 + yy.mean()),
                                  cx=float(x0 + xx.mean()),
                                  scale=float(tf['scale']),
                                  angle=float(tf['angle'])))

        # --- window plane(s) ----------------------------------------------
        for n, L in enumerate(w_idx):
            layers[L] = self._contamination((ny, nx),
                                            int(self.p.contam_seed) + 1000 * n)

        # --- optional uniform fill on the empty slices ---------------------
        lq = float(self.p.liquid_phase)
        if lq != 0.0:
            empty = [L for L in range(nl)
                     if L not in set(w_idx) | set(s_idx)]
            for L in empty:
                layers[L] *= np.exp(-1j * lq)
            logger.info("  uniform fill %.3f rad on %d empty slices "
                        "(exactly null by Eq. 22 -- the reconstruction "
                        "should be indifferent to it)" % (lq, len(empty)))

        # --- report --------------------------------------------------------
        m = cov > 0
        for L in range(nl):
            kind = ('window' if L in w_idx else
                    'sample' if L in s_idx else 'empty')
            if kind == 'empty' and lq == 0.0:
                continue
            ph = np.angle(layers[L])
            logger.info("  slice %2d (z = %5.0f um, %-6s): phase min %.3f "
                        "mean %.4f rad, |O| min %.4f"
                        % (L, z[L] * 1e6, kind, float(ph[m].min()),
                           float(ph[m].mean()),
                           float(np.abs(layers[L])[m].min())))

        # how much of the sample actually sits behind contamination: if this
        # is small the test is close to trivial
        occ_s = np.zeros((ny, nx), dtype=bool)
        for L in s_idx:
            occ_s |= (np.abs(layers[L] - 1.0) > 0.05) & m
        occ_w = np.zeros((ny, nx), dtype=bool)
        for L in w_idx:
            occ_w |= (np.abs(layers[L] - 1.0)
                      > 0.05 * float(self.p.contam_phase)) & m
        frac = (float((occ_s & occ_w).sum()) / float(occ_s.sum())
                if occ_s.sum() else 0.0)
        logger.info("  %.1f%% of the sample's projected area sits behind "
                    "contamination%s"
                    % (100.0 * frac, "" if frac > 0.1 else
                       "   <-- TOO LOW, raise contam_coverage"))

        self._results.setdefault('cell', {}).update(
            n_slices=nl, cell_span_um=float(self.p.cell_span) * 1e6,
            slice_spacing_um=float(z[1] - z[0]) * 1e6,
            window_slices=[int(i) for i in w_idx],
            sample_slices=[int(i) for i in s_idx],
            z_um=(z * 1e6).tolist(),
            n_structured=len(set(w_idx) | set(s_idx)),
            probe_na_mrad=float(self._probe_na() * 1e3),
            dof_um=float(self._wavelength() / self._probe_na() ** 2 * 1e6),
            a_c_nm={k: float(v * 1e9) for k, v in ac.items()},
            feature_over_ac={k: float(self.p.contam_feature_nm * 1e-9 / v)
                             for k, v in ac.items()},
            contam_feature_nm=float(self.p.contam_feature_nm),
            contam_phase_rad=float(self.p.contam_phase),
            contam_coverage=float(self.p.contam_coverage),
            contam_beta_delta=float(self.p.contam_beta_delta),
            liquid_phase_rad=lq,
            sample_behind_contamination=frac,
            n_sample_copies=len(table))

        return layers, table

    def _run_all(self):
        # derive the parent's grid parameters from the cell description, so
        # the runner keeps one description of the geometry rather than two
        self.p.sim_slices = int(self.p.n_slices)
        self.p.sim_thickness = float(self.p.cell_span)
        self.p.static_slice = -1
        self.p.particle_slice = -1
        self.p.no_overlap = False
        return super()._run_all()


# ===========================================================================
#  RUNNER NOTES
# ===========================================================================
#
# Copy run_multislice_sim.py and change:
#
#     from ptypy.custom import cellsim          # registers CellSimulator
#
#     PROBE_NA_SCALE = 1.0   # <-- NOT 3.0. The real 1.20 mrad probe gives
#                            #     DOF 108 um and 65 nm resolution, which is
#                            #     the regime Darren asked about. NA 3.6 mrad
#                            #     is a different experiment.
#     PROBE_DEFOCUS  = 0.0   # but see FOCAL WAIST below
#
#     p.engines.engine00.name              = 'CellSimulator'
#     p.engines.engine00.n_slices          = 13
#     p.engines.engine00.cell_span         = 1.2e-3
#     p.engines.engine00.window_z          = (0.0,)     # one window first
#     p.engines.engine00.sample_z          = (0.6e-3,)
#     p.engines.engine00.contam_feature_nm = 400.
#     p.engines.engine00.contam_phase      = 0.12
#     p.engines.engine00.contam_coverage   = 0.25
#     p.engines.engine00.sample_copies     = 8
#     p.engines.engine00.poisson_noise     = True       # see NOISE
#     p.engines.engine00.photon_scale      = 1.0
#
# FOCAL WAIST. The reconstructed probe carries a -700 um focus. Over a
# 1.2 mm span the waist may land inside the stack, which in the 30-layer
# campaign dumped amplified probe error into the mid slices as a dark vortex
# and Fresnel rings. Check the reported per-slice |phi_j| peak/mean and use
# PROBE_DEFOCUS to push the waist outside the span if it bites. This is a
# simulation artefact of reusing a focused probe, not physics -- a real cell
# experiment would place the focus deliberately.
#
# NOISE. The parent defaults poisson_noise False and so does the existing
# runner. Turn it ON here. The contamination is a ~0.1 rad perturbation, and
# whether it is recoverable is a dose question before it is a model
# question; a noiseless demonstration answers neither. The recorded flux is
# 1.04e9 photons/frame, 3.9e3 per pixel, so photon_scale = 1.0 reproduces the
# real dose and values below 1 test the shorter exposures an in-situ time
# series actually uses.
#
# THE SWEEP. One run is a demonstration; the sweep is the result. Hold
# everything else and vary contam_feature_nm across the three a_c scales
# (13 slices over 1.2 mm, NA 1.20 mrad):
#
#     a_c(adjacent)          =  240 nm
#     a_c(window to sample)  = 1440 nm
#     features: 120, 240, 480, 960, 1920, 3840 nm
#
# Predicted behaviour, from Eq. (21) and the measured linear-in-q
# suppression of Fig. 8: below 240 nm the contamination pins to its own
# slice; between 240 nm and 1.44 um it spreads axially over several slices
# but stays off the sample plane, which is all the application requires;
# above 1.44 um it leaks into the sample plane smoothly rather than
# abruptly. If that curve comes out, the depth-separability figure and the
# contamination application are the same physics measured twice -- a much
# stronger paper than either alone.
#
# THE CONTROL. Reconstruct each dataset three ways: SINGLE SLICE (the
# projection approximation, i.e. what people do now), ISS at
# n_slices, and multislice at n_slices. The single-slice run is the baseline
# the method has to beat, and beating it is a far weaker requirement than
# beating multislice -- which conveniently sidesteps the chi^2 result in
# Sec. 7.2.
#
# WHAT TO SCORE. Not chi^2, and not the contamination's own fidelity, which
# nobody cares about. Two numbers:
#   1. sample-slice correlation against the known truth layer, versus the
#      single-slice baseline;
#   2. the AXIAL PROFILE of reconstructed contamination energy -- sum
#      |O_j - 1|^2 over the illuminated region, per slice. On a dense grid
#      the contamination will occupy a band of slices rather than one, with
#      a width set by the axial PSF (~DOF / spacing slices). The quantity
#      that matters is how much of that band's energy lands on the sample
#      plane. That profile is also the most legible figure this experiment
#      can produce: truth is two spikes, a good reconstruction is two
#      slightly broadened spikes, a failure is a smear.
#
# SPARSITY CHECK. With structure on 3 of 13 slices, ~94% of the C(N,2) cross
# terms in Eq. (6) contain a g_k that is identically zero. Run the
# issdiagnostics engine on the truth stack and confirm epsilon_1 comes out
# far below the dense-stack estimate of Eq. (18) at the same Phi_tot. That
# is the Sec. 5.4 sparse-support claim measured rather than asserted, and it
# is the cleanest demonstration in the paper of when the truncation is
# nearly free.