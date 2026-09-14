# -*- coding: utf-8 -*-
"""
Multislice data simulator, as a ptypy engine.

WHAT THIS IS
------------
A companion to ISSDiagnostics, and a subclass of it: everything about
loading a finished reconstruction -- the (slices, probe) pairing scored
against the recorded data, the geometry, the views, the far-field operators,
the multislice cascade -- is inherited unchanged. What this engine does with
the reconstruction is different: instead of measuring truncation errors, it

    1. takes the particle-bearing slice of the reconstruction and segments it
       into individual particles (connected components of |O - 1| above a
       threshold),
    2. redistributes those particles over `sim_slices` axial layers -- each
       particle is assigned a random layer, and optionally a random lateral
       shift -- so the synthetic specimen is genuinely many slices thick,
    3. pushes the reconstructed probe through the synthetic stack with EXACT
       multislice (the same `_multislice` the diagnostics engine uses), and
    4. writes the simulated far-field intensities into a copy of the scan's
       .ptyd, frame for frame, so any ptypy reconstruction can be pointed at
       the simulated data with nothing but a PtydScan data block:

           p.scans.scan00.data.name  = 'PtydScan'
           p.scans.scan00.data.dfile = <the simulated .ptyd>

The ground truth -- the layer stack, the axial positions, the probe the data
were simulated with, and the per-particle assignment table -- is written to an
.h5 in the same content/objects layout the reconstruction engines use for
fslices, so issdiagnostics._read_slices and every scoring tool read it
directly.

Because the geometry, positions, mask and far field all come from the same
live ptypy instance that loaded the real scan, the simulated frames are
drop-in replacements for the measured ones: same shape, same orientation,
same frame indices. The only things that change between the real data and the
simulated data are the specimen and (optionally) the photon noise.

USAGE
-----
Take run_born_diagnostics.py, swap the engine name, and give it the sim
parameters. See the companion run_multislice_sim.py.

    p.engines.engine00.name = 'MultisliceSimulator'
    p.engines.engine00.numiter = 1
    p.engines.engine00.ground_truth = out_dir_rec
    p.engines.engine00.number_of_slices = 2       # of the loaded rec
    p.engines.engine00.slice_thickness = 1580e-6  # of the loaded rec
    p.engines.engine00.sim_slices = 5
    p.engines.engine00.outdir = out_dir_sim
"""

import os
import json
import shutil
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from ptypy.engines import register
from ptypy.utils.verbose import logger

from . import issdiagnostics
from .issdiagnostics import _jsonable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


@register()
class MultisliceSimulator(issdiagnostics.ISSDiagnostics):
    """
    Load a finished reconstruction, redistribute its particles over many
    layers, simulate detector data with exact multislice, and write a .ptyd
    plus the ground truth. Runs once and stops, like the diagnostics engine.

    Defaults:

    [name]
    default = MultisliceSimulator
    type = str
    help = Engine identifier

    [sim_slices]
    default = 5
    type = int
    help = Number of layers in the simulated specimen
    doc = Includes the static layer if static_slice is enabled, so with the
          default static_slice = 0 there are sim_slices - 1 particle layers.

    [static_slice]
    default = 0
    type = int
    help = Reconstruction slice kept whole as the first simulated layer
    doc = The NanoMAX scan 434 reconstruction has the grating structure on
          slice 0 and the particles on slice 1; keeping the grating as a fixed
          entrance layer makes the simulated frames resemble the measured
          ones. Set to a negative value to build the specimen from particles
          alone.

    [particle_slice]
    default = -1
    type = int
    help = Reconstruction slice the particles are segmented from
    doc = Negative counts from the end, so the default is the last slice.

    [sim_thickness]
    default = None
    type = float
    help = Total axial extent of the simulated specimen in meters
    doc = Layers are equally spaced over this extent, so the separation is
          sim_thickness / (sim_slices - 1). None keeps the extent of the
          loaded reconstruction.

    [segment_threshold]
    default = 0.15
    type = float
    help = Particle segmentation threshold on |O - 1|
    doc = Applied to the mean-gauged particle slice, where the empty
          background sits at O = 1. Particles in this scan reach phase shifts
          of -1.4 rad, i.e. |O - 1| near 1, while the background residual is
          a few percent.

    [min_particle_px]
    default = 30
    type = int
    help = Components smaller than this many pixels are dropped as noise

    [dilate_px]
    default = 2
    type = int
    help = Dilation of each particle footprint, to catch the soft edges

    [particle_copies]
    default = 1
    type = int
    help = Number of copies of each particle pasted into the specimen
    doc = Each copy gets its own layer and lateral shift, so a handful of
          real particles can populate many layers. Layers are assigned by
          shuffling an evenly-tiled layer list, so the copies spread over
          the layers as uniformly as the count allows.

    [transform_copies]
    default = False
    type = bool
    help = Randomly flip, rotate and rescale each pasted copy
    doc = Gives every copy its own shape, so a specimen built from a few
          real particles does not repeat the identical footprint on every
          layer. Rotation is by an arbitrary angle (bilinear on the complex
          transmission deviation, nearest-neighbour on the footprint).

    [copy_scale]
    default = (0.7, 1.3)
    type = tuple
    help = Range of the random per-copy scale factor
    doc = Only used when transform_copies is on.

    [no_overlap]
    default = False
    type = bool
    help = Place copies so no two footprints overlap in projection
    doc = Copies are placed at random positions whose frame centre lies in
          the scanned region, rejecting any position where the footprint
          leaves the illuminated area or comes within overlap_margin_px of
          an already-placed copy on ANY layer. Replaces the max_shift_px
          placement around the particle's reconstructed position.

    [overlap_margin_px]
    default = 10
    type = int
    help = Minimum projected gap between placed copies, in object pixels

    [max_shift_px]
    default = 0
    type = int
    help = Maximum random lateral shift of each particle, in object pixels
    doc = 0 keeps every particle at its reconstructed transverse position, so
          the projection of the simulated stack equals the reconstructed
          particle slice exactly. Nonzero shifts break that degeneracy and
          spread the particles out.

    [layer_seed]
    default = 0
    type = int
    help = Seed for the layer assignment and the lateral shifts

    [poisson_noise]
    default = False
    type = bool
    help = Apply Poisson noise to the simulated intensities

    [photon_scale]
    default = 1.0
    type = float
    help = Flux multiplier for the Poisson noise
    doc = The expected counts are scaled by this factor before sampling and
          the sample is divided by it afterwards, so the data scale is
          unchanged and only the noise level moves: 10 means 10x the recorded
          flux, i.e. sqrt(10) less relative noise.

    [match_counts]
    default = False
    type = bool
    help = Rescale each simulated frame to the measured masked photon count
    doc = Off by default: the reconstructed probe already carries the
          physical flux, and the per-frame ratio to the measured counts is
          reported either way in results.json.

    [source_ptyd]
    default = None
    type = str
    help = The scan's prepared .ptyd, used as the template for the output
    doc = None takes dfile from the scan's own data block, which is the right
          file whenever this engine runs inside the reconstruction script it
          was copied from.

    [probe_na_scale]
    default = 1.0
    type = float
    help = Scale factor on the probe's convergence angle (NA)
    doc = Applied before simulating: pr(x) -> s * pr(s x), a flux-preserving
          real-space zoom that multiplies every angle in the probe's
          spectrum by s and shrinks the focal spot by s. 1.0 leaves the
          reconstructed probe untouched. The footprint at the sample plane
          also shrinks by ~s (the baked-in defocus curvature scales as
          s^2), so check the scan overlap stays adequate. The scaled probe
          is what lands in the truth h5.

    [probe_defocus]
    default = 0.0
    type = float
    help = Extra Fresnel propagation of the probe before simulating [m]
    doc = Angular-spectrum propagation of the loaded probe by this distance
          before it illuminates layer 0, applied after probe_na_scale. The
          reconstructed probe carries a baked-in focus (e.g. -700 um upstream
          of the sample); +dz here pushes the beam further downstream of that
          focus, enlarging its footprint on the entrance layers (effective
          defocus becomes more negative). Use to move the focal waist relative
          to the synthetic stack so no plane sits at the poorly-overlapped
          waist. The propagated probe is what lands in the truth h5.

    [sim_tag]
    default = sim
    type = str
    help = Tag carried in every output filename
    """

    # ------------------------------------------------------------------ #
    #  The one-shot run: simulate instead of measure                      #
    # ------------------------------------------------------------------ #
    def _run_all(self):
        t0 = datetime.now()
        if h5py is None:
            raise ImportError("MultisliceSimulator requires h5py")

        # Background at O = 1 and the flux pushed onto the probe, so pasted
        # particles sit on an exactly-empty background and the probe carries
        # the physical scale the data were fitted at.
        alpha = self._apply_gauge('mean')

        s_na = float(self.p.probe_na_scale)
        if s_na != 1.0:
            self._set_probe(self._zoom_probe(self.pr_g, s_na))
            logger.info("probe NA scaled by %.3g (focal spot / footprint "
                        "shrink by the same factor, flux preserved)" % s_na)

        dz_pr = float(self.p.probe_defocus)
        if dz_pr != 0.0:
            self._set_probe(self._prop(self.pr_g, dz_pr))
            logger.info("probe Fresnel-propagated by %.1f um before simulating "
                        "(moves the focal waist relative to the stack)"
                        % (dz_pr * 1e6))

        src_idx = int(self.p.particle_slice) % self.N
        src = self.ob_g[src_idx].cpu().numpy()
        particles = self._segment(src)
        if not particles:
            raise RuntimeError(
                "no particles found on slice %d at threshold %.3f -- check "
                "segment_threshold against the slice statistics" %
                (src_idx, float(self.p.segment_threshold)))

        layers, table = self._build_layers(src, particles)
        nl = layers.shape[0]

        # axial grid: equally spaced over the requested extent
        T = (float(self.p.sim_thickness) if self.p.sim_thickness
             else float(self.z[-1] - self.z[0]))
        z_sim = np.arange(nl, dtype=float) * (T / max(nl - 1, 1))

        logger.info("simulating: %d layers over %.1f um (dz = %.1f um), "
                    "%d particles, %d positions"
                    % (nl, T * 1e6, (z_sim[1] - z_sim[0]) * 1e6 if nl > 1
                       else 0.0, len(table), self.n_pos))

        # install the synthetic specimen; every cache keyed on (probe, N, z)
        # is invalidated by _set_probe via _load_slices semantics
        self.N = nl
        self.z = z_sim
        self.ob_g = torch.stack([torch.as_tensor(l, dtype=self.cdt,
                                                 device=self.dev)
                                 for l in layers])
        self._illum_cache = None

        I_sim, stats = self._simulate()

        # ---- outputs ----------------------------------------------------
        tag = '%s_slices-%d_seed-%d' % (str(self.p.sim_tag), nl,
                                        int(self.p.layer_seed))
        ptyd_path = os.path.join(self.outdir, '%s.ptyd' % tag)
        truth_path = os.path.join(self.outdir,
                                  '%s_truth_iter-0000.h5' % tag)
        self._write_ptyd(ptyd_path, I_sim)
        self._write_truth(truth_path, layers, z_sim, table)
        self._figures_sim(src, layers, z_sim, table, I_sim, tag)

        self._results['simulation'] = dict(
            when=t0.isoformat(timespec='seconds'),
            sim_slices=nl,
            z_um=(z_sim * 1e6).tolist(),
            separation_um=float((z_sim[1] - z_sim[0]) * 1e6) if nl > 1 else 0.0,
            n_particles=len(table),
            particles_per_layer={int(l): int(sum(1 for t in table
                                                 if t['layer'] == l))
                                 for l in sorted({t['layer'] for t in table})},
            static_slice=(int(self.p.static_slice)
                          if int(self.p.static_slice) >= 0 else None),
            particle_slice=src_idx,
            segment_threshold=float(self.p.segment_threshold),
            max_shift_px=int(self.p.max_shift_px),
            layer_seed=int(self.p.layer_seed),
            poisson_noise=bool(self.p.poisson_noise),
            photon_scale=float(self.p.photon_scale),
            match_counts=bool(self.p.match_counts),
            gauge_alpha=[complex(a) for a in alpha],
            ptyd=ptyd_path,
            truth=truth_path,
            n_positions=self.n_pos,
            **stats)
        self._results['run'] = dict(seconds=(datetime.now() - t0).total_seconds())
        logger.info("simulated data written to %s" % ptyd_path)
        logger.info("ground truth written to %s" % truth_path)

    # ------------------------------------------------------------------ #
    #  Particle segmentation and layer assembly                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _zoom_probe(pr, s):
        """pr(x) -> s * pr(s x): every angle in the spectrum multiplied by s,
        focal spot shrunk by s, total flux preserved (in the continuum;
        edge truncation is negligible for a centred probe)."""
        import torch.nn.functional as Fnl
        M, ny, nx = pr.shape
        # grid must share the sampled tensor's real dtype: grid_sample rejects
        # a float32 grid against a float64 (double-precision probe) input
        rdt = pr.real.dtype
        yy = torch.linspace(-1.0, 1.0, ny, device=pr.device, dtype=rdt) * s
        xx = torch.linspace(-1.0, 1.0, nx, device=pr.device, dtype=rdt) * s
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)[None].expand(
            M, ny, nx, 2).contiguous()
        ri = torch.stack([pr.real, pr.imag], dim=1)      # (M, 2, ny, nx)
        out = Fnl.grid_sample(ri, grid, mode="bilinear",
                              padding_mode="zeros", align_corners=True)
        return (torch.complex(out[:, 0], out[:, 1]) * s).to(pr.dtype)

    def _segment(self, src):
        """Connected components of |O - 1| above the threshold.

        Returns a list of (slice_pair, footprint) with the footprint already
        dilated, both living on a bounding box padded by dilate_px so the
        dilation has room.
        """
        from scipy import ndimage

        dev = np.abs(src - 1.0)
        m = ndimage.binary_opening(dev > float(self.p.segment_threshold))
        lab, n = ndimage.label(m)
        objs = ndimage.find_objects(lab)
        pad = int(self.p.dilate_px)
        min_px = int(self.p.min_particle_px)
        ny, nx = src.shape
        out = []
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            if int((lab[sl] == i).sum()) < min_px:
                continue
            y0 = max(sl[0].start - pad, 0)
            y1 = min(sl[0].stop + pad, ny)
            x0 = max(sl[1].start - pad, 0)
            x1 = min(sl[1].stop + pad, nx)
            box = (slice(y0, y1), slice(x0, x1))
            foot = lab[box] == i
            if pad:
                foot = ndimage.binary_dilation(foot, iterations=pad)
            out.append((box, foot))
        logger.info("segmentation: %d components, %d kept above %d px"
                    % (n, len(out), min_px))
        return out

    def _build_layers(self, src, particles):
        """Ones everywhere, the static slice on layer 0 if kept, and each
        particle's transmission pasted into its assigned layer at its
        (optionally shifted) position. Overlaps multiply, as transmissions
        do."""
        nl = int(self.p.sim_slices)
        static = int(self.p.static_slice)
        lo = 1 if static >= 0 else 0
        if nl - lo < 1:
            raise RuntimeError("sim_slices = %d leaves no particle layer" % nl)

        ny, nx = src.shape
        layers = np.ones((nl, ny, nx), dtype=np.complex128)
        if static >= 0:
            layers[0] = self.ob_g[static % len(self.ob_g)].cpu().numpy()

        rng = np.random.default_rng(int(self.p.layer_seed))
        shift = int(self.p.max_shift_px)
        g = src - 1.0
        table = []
        # each particle's copies land on distinct layers (until the layer
        # count is exhausted), so copies never pile up within one layer
        copies = max(int(self.p.particle_copies), 1)
        reps = int(np.ceil(copies / max(nl - lo, 1)))
        instances = [(k, box, foot, int(L))
                     for k, (box, foot) in enumerate(particles)
                     for L in rng.permutation(
                         np.tile(np.arange(lo, nl), reps))[:copies]]

        no_overlap = bool(self.p.no_overlap)
        margin = int(self.p.overlap_margin_px)
        if no_overlap:
            from scipy import ndimage
            cov = self._coverage().cpu().numpy()
            occ = np.zeros_like(cov)
            # sample copy centres from the bbox of the scan's frame centres,
            # so every copy sits in well-covered territory
            fr = self.rows[:, self.fny // 2].cpu().numpy()
            fc = self.cols[:, self.fnx // 2].cpu().numpy()
            sub = cov[fr.min():fr.max() + 1, fc.min():fc.max() + 1]
            allowed = np.argwhere(sub) + [fr.min(), fc.min()]
            if not len(allowed):
                allowed = np.argwhere(cov)
            # big blobs first: they are the hardest to place
            instances.sort(key=lambda t: -int(t[2].sum()))

        for k, box, foot, L in instances:
            patch = g[box] * foot
            tf = dict(scale=1.0, angle=0.0)
            if bool(self.p.transform_copies):
                patch, foot, tf = self._transform_copy(patch, foot, rng)
                if not foot.any():
                    continue
            h, w = foot.shape
            oyc = box[0].start + (box[0].stop - box[0].start) / 2.
            oxc = box[1].start + (box[1].stop - box[1].start) / 2.

            if no_overlap:
                # margin of zeros around the patch, so the dilated footprint
                # stays inside it; the zero rim pastes as *= 1 + 0
                pat = np.pad(patch, margin)
                ft = np.pad(foot, margin)
                ftm = (ndimage.binary_dilation(ft, iterations=margin)
                       if margin else ft)
                H, W = ft.shape
                placed = False
                for attempt in range(4):
                    if attempt:
                        # shrink and retry: a smaller copy beats a lost one
                        sc = 0.8
                        pat = ndimage.zoom(pat.real, sc, order=1) + \
                            1j * ndimage.zoom(pat.imag, sc, order=1)
                        ft = ndimage.zoom(ft.astype(float), sc,
                                          order=0) > 0.5
                        pat = pat * ft
                        ftm = (ndimage.binary_dilation(ft, iterations=margin)
                               if margin else ft)
                        tf['scale'] *= sc
                        H, W = ft.shape
                    for _ in range(2000):
                        yc, xc = allowed[int(rng.integers(len(allowed)))]
                        y0, x0 = int(yc) - H // 2, int(xc) - W // 2
                        if y0 < 0 or x0 < 0 or y0 + H > ny or x0 + W > nx:
                            continue
                        dst = (slice(y0, y0 + H), slice(x0, x0 + W))
                        if not cov[dst][ft].all() or (occ[dst] & ftm).any():
                            continue
                        layers[L][dst] *= 1.0 + pat
                        occ[dst] |= ftm
                        placed = True
                        break
                    if placed:
                        break
                if not placed:
                    logger.warning("could not place copy of particle %d on "
                                   "layer %d without overlap; skipping" %
                                   (k, L))
                    continue
                foot = ft
                yy, xx = np.nonzero(ft)
                cyv, cxv = y0 + yy.mean(), x0 + xx.mean()
                dyv, dxv = int(round(cyv - oyc)), int(round(cxv - oxc))
            else:
                dyv = int(rng.integers(-shift, shift + 1)) if shift else 0
                dxv = int(rng.integers(-shift, shift + 1)) if shift else 0
                # destination window around the reconstructed position,
                # clipped to the array; crop the patch to match
                y0 = int(round(oyc)) - h // 2 + dyv
                x0 = int(round(oxc)) - w // 2 + dxv
                y1, x1 = y0 + h, x0 + w
                cy0, cx0 = max(-y0, 0), max(-x0, 0)
                cy1 = h - max(y1 - ny, 0)
                cx1 = w - max(x1 - nx, 0)
                if cy1 <= cy0 or cx1 <= cx0:
                    continue
                dst = (slice(y0 + cy0, y0 + cy1), slice(x0 + cx0, x0 + cx1))
                layers[L][dst] *= 1.0 + patch[cy0:cy1, cx0:cx1]
                yy, xx = np.nonzero(foot)
                cyv, cxv = y0 + yy.mean(), x0 + xx.mean()

            table.append(dict(label=k, layer=L, dy=int(dyv), dx=int(dxv),
                              area=int(foot.sum()),
                              cy=float(cyv), cx=float(cxv),
                              scale=float(tf['scale']),
                              angle=float(tf['angle'])))
        return layers, table

    def _transform_copy(self, patch, foot, rng):
        """Random flip / rotation / scale of one copy's transmission
        deviation and footprint. The deviation is interpolated bilinearly
        (real and imaginary parts), the footprint nearest-neighbour."""
        from scipy import ndimage

        if rng.random() < 0.5:
            patch, foot = patch[::-1], foot[::-1]
        if rng.random() < 0.5:
            patch, foot = patch[:, ::-1], foot[:, ::-1]
        ang = float(rng.uniform(0., 360.))
        s_lo, s_hi = (float(s) for s in self.p.copy_scale)
        sc = float(rng.uniform(s_lo, s_hi))

        re = ndimage.rotate(np.ascontiguousarray(patch.real), ang,
                            reshape=True, order=1, cval=0.)
        im = ndimage.rotate(np.ascontiguousarray(patch.imag), ang,
                            reshape=True, order=1, cval=0.)
        ft = ndimage.rotate(foot.astype(float), ang,
                            reshape=True, order=0, cval=0.)
        if abs(sc - 1.) > 1e-3:
            re = ndimage.zoom(re, sc, order=1)
            im = ndimage.zoom(im, sc, order=1)
            ft = ndimage.zoom(ft, sc, order=0)
        foot = ft > 0.5
        patch = (re + 1j * im) * foot
        return patch, foot, dict(scale=sc, angle=ang)

    # ------------------------------------------------------------------ #
    #  Forward simulation                                                 #
    # ------------------------------------------------------------------ #
    def _simulate(self):
        """Exact multislice through the synthetic stack at every position.

        Returns the intensity frames as float64 numpy (n_pos, ny, nx) plus
        the summary statistics against the measured data.
        """
        B = int(self.p.batch_size)
        I_sim = np.empty((self.n_pos, self.fny, self.fnx), dtype=np.float64)
        corr, ratio = [], []
        with torch.no_grad():
            for s in range(0, self.n_pos, B):
                idx = torch.arange(s, min(s + B, self.n_pos),
                                   dtype=torch.long, device=self.dev)
                O = torch.stack([self._gather(self.ob_g[j], idx)
                                 for j in range(self.N)])
                psi = self._multislice(O, self.pr_g)
                I = (self._ff(self._ff_fw, psi).abs() ** 2).sum(1)
                Im, mk = self.I[idx], self.mask[idx]
                ms = (mk * I).sum(dim=(1, 2)).clamp_min(1e-30)
                mm = (mk * Im).sum(dim=(1, 2))
                if bool(self.p.match_counts):
                    I = I * (mm / ms)[:, None, None]
                    ratio += (mm / ms).cpu().numpy().tolist()
                else:
                    ratio += (ms / mm.clamp_min(1e-30)).cpu().numpy().tolist()
                for k in range(idx.numel()):
                    m = mk[k] > 0
                    a = torch.sqrt(I[k][m].clamp_min(0)).double()
                    b = torch.sqrt(Im[k][m].clamp_min(0)).double()
                    a, b = a - a.mean(), b - b.mean()
                    corr.append(float((a @ b) /
                                      (a.norm() * b.norm()).clamp_min(1e-30)))
                I_sim[s:s + idx.numel()] = I.cpu().numpy()
                del O, psi, I

        if bool(self.p.poisson_noise):
            sc = float(self.p.photon_scale)
            rng = np.random.default_rng(int(self.p.layer_seed) + 1)
            for k in range(self.n_pos):
                I_sim[k] = rng.poisson(np.clip(I_sim[k], 0, None) * sc) / sc

        key = ('measured_over_sim_counts' if bool(self.p.match_counts)
               else 'sim_over_measured_counts')
        stats = {
            'sim_vs_measured_correlation': float(np.mean(corr)),
            key: dict(median=float(np.median(ratio)),
                      p05=float(np.percentile(ratio, 5)),
                      p95=float(np.percentile(ratio, 95))),
            'counts_per_frame_median': float(np.median(
                I_sim.sum(axis=(1, 2)))),
        }
        logger.info("simulated frames correlate with the measured ones at "
                    "%.3f; %s median %.3f"
                    % (stats['sim_vs_measured_correlation'], key,
                       stats[key]['median']))
        return I_sim, stats

    # ------------------------------------------------------------------ #
    #  Outputs                                                            #
    # ------------------------------------------------------------------ #
    def _source_ptyd(self):
        if self.p.source_ptyd and os.path.isfile(str(self.p.source_ptyd)):
            return str(self.p.source_ptyd)
        for scan in self.ptycho.p.scans.values():
            df = scan.get('data', {}).get('dfile')
            if df and os.path.isfile(str(df)):
                return str(df)
        raise RuntimeError("no source .ptyd found; set source_ptyd")

    def _write_ptyd(self, path, I_sim):
        """A copy of the scan's .ptyd with the data frames replaced.

        The frame order in a chunk is given by its `indices` dataset, and a
        diffraction view's `.layer` is exactly that global index (manager.py
        sets dv.layer = index), so the mapping needs no convention about how
        either side happens to be sorted. Weights, positions, info and meta
        are untouched, which is the point: a PtydScan on this file rebuilds
        the identical geometry.
        """
        src = self._source_ptyd()
        shutil.copyfile(src, path)
        view_layers = [int(np.ravel(self.di.views[n].layer)[0])
                       for n in self.view_names]
        written = 0
        with h5py.File(path, 'r+') as f:
            loc = {}
            for cname, grp in f['chunks'].items():
                for row, gi in enumerate(np.asarray(grp['indices']).ravel()):
                    loc[int(gi)] = (cname, row)
            missing = [l for l in view_layers if l not in loc]
            if missing:
                raise RuntimeError("%d view layers have no frame in %s "
                                   "(first: %s)" % (len(missing), src,
                                                    missing[:5]))
            newdata = {}
            for cname, grp in f['chunks'].items():
                newdata[cname] = np.asarray(grp['data']).astype(np.float32)
            for k, l in enumerate(view_layers):
                cname, row = loc[l]
                newdata[cname][row] = I_sim[k].astype(np.float32)
                written += 1
            for cname, arr in newdata.items():
                del f['chunks'][cname]['data']
                f['chunks'][cname].create_dataset('data', data=arr)
        logger.info("replaced %d of %d frames in %s"
                    % (written, sum(a.shape[0] for a in newdata.values()),
                       os.path.basename(path)))

    def _write_truth(self, path, layers, z_sim, table):
        """Ground truth in the fslices layout, so _read_slices and every
        scoring tool read it like any reconstruction."""
        with h5py.File(path, 'w') as f:
            for j in range(layers.shape[0]):
                f.create_dataset('content/objects/obj_%02d/S00/data' % j,
                                 data=layers[j][None].astype(np.complex64))
            f.create_dataset('content/probe/S00/data',
                             data=self.pr_g.cpu().numpy().astype(np.complex64))
            f.create_dataset('content/z', data=z_sim)
            g = f.create_group('content/particles')
            for key in ('label', 'layer', 'dy', 'dx', 'area'):
                g.create_dataset(key, data=np.array([t[key] for t in table],
                                                    dtype=np.int64))
            for key in ('cy', 'cx', 'scale', 'angle'):
                g.create_dataset(key, data=np.array([t.get(key, 0.0)
                                                     for t in table]))

    # ------------------------------------------------------------------ #
    #  Figures                                                            #
    # ------------------------------------------------------------------ #
    def _figures_sim(self, src, layers, z_sim, table, I_sim, tag):
        cov = self._coverage().cpu().numpy()
        ys, xs = np.nonzero(cov)
        win = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

        nl = layers.shape[0]
        ncol = min(nl, 4)
        nrow = int(np.ceil(nl / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow),
                                 squeeze=False)
        for j in range(nrow * ncol):
            ax = axes[j // ncol][j % ncol]
            if j >= nl:
                ax.axis('off')
                continue
            im = ax.imshow(np.angle(layers[j][win]), cmap='gray',
                           vmin=-1.5, vmax=0.3)
            npart = sum(1 for t in table if t['layer'] == j)
            ax.set_title('layer %d, z = %.0f um, %s' %
                         (j, z_sim[j] * 1e6,
                          'static' if j == 0 and int(self.p.static_slice) >= 0
                          else '%d particles' % npart))
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, label='phase [rad]')
        fig.suptitle('simulated specimen (%d layers)' % nl)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, '%s_layers.png' % tag), dpi=150)
        plt.close(fig)

        # where every particle went, on the source slice
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.angle(src[win]), cmap='gray', vmin=-1.5, vmax=0.3)
        lo = 1 if int(self.p.static_slice) >= 0 else 0
        cmap = plt.get_cmap('turbo', max(nl - lo, 1))
        for t in table:
            ax.plot(t['cx'] - win[1].start, t['cy'] - win[0].start, 'o',
                    ms=4, color=cmap(t['layer'] - lo))
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(lo - 0.5, nl - 0.5))
        fig.colorbar(sm, ax=ax, ticks=range(lo, nl), label='assigned layer')
        ax.set_title('particle layer assignment (on the source slice)')
        ax.set_xticks([]), ax.set_yticks([])
        fig.savefig(os.path.join(self.outdir, '%s_assignment.png' % tag), dpi=150)
        plt.close(fig)

        # a few frames, measured beside simulated
        nshow = min(int(self.p.compare_frames), self.n_pos)
        pick = np.linspace(0, self.n_pos - 1, nshow).astype(int)
        fig, axes = plt.subplots(2, nshow, figsize=(4 * nshow, 8),
                                 squeeze=False)
        for c, k in enumerate(pick):
            meas = (self.I[k] * self.mask[k]).cpu().numpy()
            vmax = max(meas.max(), 1.0)
            for r, (img, name) in enumerate(((meas, 'measured'),
                                             (I_sim[k], 'simulated'))):
                ax = axes[r][c]
                ax.imshow(np.clip(img, 1e-1, None),
                          norm=LogNorm(vmin=1e-1, vmax=vmax), cmap='viridis')
                ax.set_title('%s, frame %d' % (name, k))
                ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, '%s_frames.png' % tag), dpi=150)
        plt.close(fig)


@register()
class SparseStrongSimulator(MultisliceSimulator):
    """
    Strongly scattering, beam-cone-sparse specimen simulator.

    Same machinery as MultisliceSimulator, but the specimen is built for one
    demonstration: the chord perturbation g = O - 1 stays exact for a THICK,
    STRONGLY scattering sample as long as the sample is sparse in the beam
    cone -- no scan position illuminates structure on two different layers --
    while the tangent (linearised, g = i s) perturbation fails there, because
    its per-slice error is O(s^2) regardless of sparsity.

    Two things change relative to the parent:

    1. Each pasted particle transmission is rescaled O -> O**phase_scale
       (phase multiplied by phase_scale, |O| <= 1 preserved), making the
       particles strongly scattering.
    2. Placement enforces beam-cone disjointness, not just projected
       non-overlap: the probe is propagated to every simulated plane, its
       footprint radius R(z) measured (encircled-energy quantile), and two
       copies on layers j != k must satisfy

           |c_a - c_b| > R(z_j) + R(z_k) + r_a + r_b + cone_margin_px,

       so no position of the divergent beam can hit a particle on one layer
       while shining onto a particle on another. Copies on the SAME layer
       only need the ordinary projected gap. After placement the condition
       is verified directly against every scan position and the result is
       logged and written to results.json.

    Defaults:

    [name]
    default = SparseStrongSimulator
    type = str
    help = Engine identifier

    [phase_scale]
    default = 3.0
    type = float
    help = Power-law rescale O -> O**phase_scale of each pasted particle
    doc = Multiplies the particle phase by this factor (and deepens the
          absorption by the same power), so the source particles' ~1.4 rad
          become strongly scattering at 3.0 (~4 rad, |g| up to 2) while the
          background stays exactly 1.

    [cone_margin_px]
    default = 20
    type = int
    help = Extra projected gap on top of the computed beam-cone requirement

    [footprint_quantile]
    default = 0.99
    type = float
    help = Encircled-energy fraction defining the probe footprint radius R(z)

    [max_phase_rad]
    default = 3.0
    type = float
    help = Cap on the pasted particle phase, so the true object never wraps
    doc = The effective power is min(phase_scale, max_phase_rad / max source
          phase), computed from the segmented particles after background
          referencing. Keeps |phase| < pi so np.angle of the truth layers is
          single-valued; set >= pi * phase_scale to disable the cap.

    [smooth_px]
    default = 2.5
    type = float
    help = Gaussian sigma smoothing each particle's complex log, in pixels
    doc = The power rescale amplifies the source reconstruction's pixel
          speckle, which scatters far beyond the probe NA and reaches the
          other layers however far apart the particles sit. Smoothing the
          log-transmission (phase and absorption together) and tapering the
          edge with the smoothed footprint gives a compact scattering
          spectrum, so the shear criterion below actually bounds the
          cross terms. 0 disables.

    [pure_phase]
    default = False
    type = bool
    help = Zero the particles' absorption: real potential, |O| = 1 exactly
    doc = Keeps only the imaginary part of the smoothed log-transmission,
          so s = -i log O is real everywhere. The matched truth for a
          weak-phase-object (real-potential) reconstruction: a tangent
          solver constrained to real s then fails by its linearisation
          alone, not by missing absorption.

    [shear_na_scale]
    default = 1.5
    type = float
    help = Scattered-wave NA, as a multiple of the measured probe NA
    doc = The neglected Born cross terms vanish when particles on layers j
          and k are transversely disjoint within the scattered-wave shear
          2 NA_eff |z_j - z_k| (draft Eq. depthres). The scattered wave
          spreads by the probe NA plus the particle's own spectrum, so
          NA_eff = shear_na_scale * NA(probe). The cross-layer gap enforced
          is r_a + r_b + max(R_j + R_k, shear) + cone_margin_px.

    [overlap_gap_px]
    default = -1
    type = int
    help = Deterministic collinear placement for the dose experiment [px]
    doc = Negative (default) uses the random beam-cone-disjoint placement
          above. When >= 0 the single strongest segmented particle is pasted
          on EVERY layer, collinear along x, with adjacent centres this many
          pixels apart (symmetric about the coverage centre). Sliding this one
          gap from beyond the scattered-wave shear (disjoint, eps1 ~ 1e-4)
          down towards zero (fully overlapped, eps1 ~ 1e-2) makes lateral
          overlap the only variable, which is exactly what the dose-criterion
          sweep of Sec. sec:sparse needs: eps1 tuned by geometry alone.
    """

    # ------------------------------------------------------------------ #
    #  Probe footprint at a simulated plane                               #
    # ------------------------------------------------------------------ #
    def _probe_footprint_px(self, z):
        """Radius (in object pixels) containing footprint_quantile of the
        probe power at plane z, measured about the intensity centroid."""
        w = self._prop(self.pr_g, float(z))
        I = (w.abs() ** 2).sum(0).cpu().numpy()
        ny, nx = I.shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        tot = max(I.sum(), 1e-30)
        cy, cx = (I * yy).sum() / tot, (I * xx).sum() / tot
        r = np.hypot(yy - cy, xx - cx).ravel()
        order = np.argsort(r)
        c = np.cumsum(I.ravel()[order])
        c /= max(c[-1], 1e-30)
        q = min(max(float(self.p.footprint_quantile), 0.5), 0.9999)
        return float(r[order][np.searchsorted(c, q)])

    @staticmethod
    def _copy_radius(foot):
        """Largest distance of a footprint pixel from its centroid [px]."""
        yy, xx = np.nonzero(foot)
        return float(np.hypot(yy - yy.mean(), xx - xx.mean()).max())

    # ------------------------------------------------------------------ #
    #  Layer assembly with the beam-cone constraint                       #
    # ------------------------------------------------------------------ #
    def _build_layers(self, src, particles):
        from scipy import ndimage

        nl = int(self.p.sim_slices)
        if int(self.p.static_slice) >= 0:
            raise RuntimeError(
                "SparseStrongSimulator builds from particles alone; "
                "set static_slice = -1")
        if nl < 2:
            raise RuntimeError("sim_slices must be >= 2, got %d" % nl)

        # The axial grid _run_all will install afterwards, recomputed here
        # because the footprint radii depend on it.
        T = (float(self.p.sim_thickness) if self.p.sim_thickness
             else float(self.z[-1] - self.z[0]))
        z_sim = np.arange(nl, dtype=float) * (T / max(nl - 1, 1))
        R = [self._probe_footprint_px(z) for z in z_sim]
        logger.info(
            "beam-cone footprints (%.0f%% energy): %s px at z = %s um"
            % (100 * float(self.p.footprint_quantile),
               ["%.0f" % r for r in R], ["%.0f" % (z * 1e6) for z in z_sim]))

        rng = np.random.default_rng(int(self.p.layer_seed))
        ny, nx = src.shape
        layers = np.ones((nl, ny, nx), dtype=np.complex128)
        g = src - 1.0

        # Scattered-wave shear between layer pairs, in pixels: the Born
        # cross terms vanish only if the wave scattered by one layer's
        # particle misses the other layer's particle after shearing by
        # 2 NA_eff dz in transit -- the probe FOOTPRINT alone is not the
        # criterion (a per-position eps floor of 1e-4..1e-2 on the first
        # sample traced back to exactly this).
        res = float(np.asarray(self.geom.resolution).ravel()[0])
        na_eff = float(self.na) * float(self.p.shear_na_scale)
        shear_px = [[2.0 * na_eff * abs(z_sim[j] - z_sim[k]) / res
                     for k in range(nl)] for j in range(nl)]
        logger.info("scattered-wave shear (NA_eff %.2f mrad): adjacent %.0f "
                    "px, extreme %.0f px"
                    % (na_eff * 1e3, shear_px[0][1] if nl > 1 else 0,
                       shear_px[0][nl - 1]))

        # Per-particle prep: background-referenced complex log, smoothed and
        # edge-tapered so the particle is a compact-spectrum strong scatterer
        # instead of amplified reconstruction speckle.
        from scipy import ndimage
        sig = max(float(self.p.smooth_px), 0.0)
        pad = int(4 * sig) + 4
        prepped = []
        max_src = 0.0
        for k, (box, foot) in enumerate(particles):
            y0 = max(box[0].start - pad, 0)
            y1 = min(box[0].stop + pad, ny)
            x0 = max(box[1].start - pad, 0)
            x1 = min(box[1].stop + pad, nx)
            ebox = (slice(y0, y1), slice(x0, x1))
            efoot = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            efoot[box[0].start - y0:box[0].stop - y0,
                  box[1].start - x0:box[1].stop - x0] = foot
            box_o = 1.0 + g[ebox]
            bg_px = box_o[~efoot]
            bg = complex(bg_px.mean()) if bg_px.size else 1.0
            L = np.log(box_o / bg)          # |source phase| < 1 rad: safe
            if bool(self.p.pure_phase):
                L = 1j * L.imag             # real potential: |O| = 1 exactly
            if sig > 0:
                L = (ndimage.gaussian_filter(L.real, sig)
                     + 1j * ndimage.gaussian_filter(L.imag, sig))
                taper = ndimage.gaussian_filter(efoot.astype(float), sig)
            else:
                taper = efoot.astype(float)
            Lm = L * taper                  # -> 0 smoothly at the edge
            fb = taper > 0.01
            if not fb.any():
                continue
            prepped.append((k, ebox, Lm, fb))
            max_src = max(max_src, float(np.abs(Lm.imag).max()))

        # Effective power: strong, but never past the max_phase_rad cap, so
        # the true object phase does not wrap (disable with a large cap).
        s_pow = float(self.p.phase_scale)
        cap = float(self.p.max_phase_rad)
        if max_src > 0 and s_pow * max_src > cap:
            s_eff = cap / max_src
            logger.info(
                "phase_scale %.2f would reach %.2f rad and wrap; capped to "
                "%.2f (max source phase %.3f rad -> max pasted %.2f rad)"
                % (s_pow, s_pow * max_src, s_eff, max_src, cap))
            s_pow = s_eff
        logger.info("pasted particle phase reaches %.2f rad (smooth_px %.1f)"
                    % (s_pow * max_src, sig))

        if int(self.p.overlap_gap_px) >= 0:
            return self._build_layers_collinear(
                layers, prepped, s_pow, z_sim, R, shear_px)

        cov = self._coverage().cpu().numpy()
        fr = self.rows[:, self.fny // 2].cpu().numpy()
        fc = self.cols[:, self.fnx // 2].cpu().numpy()
        sub = cov[fr.min():fr.max() + 1, fc.min():fc.max() + 1]
        allowed = np.argwhere(sub) + [fr.min(), fc.min()]
        if not len(allowed):
            allowed = np.argwhere(cov)

        # copies cycle through the layers, so every layer is populated as
        # evenly as the copy count allows
        copies = max(int(self.p.particle_copies), 1)
        insts, L = [], 0
        for k, ebox, Lm, fb in prepped:
            for _ in range(copies):
                insts.append((k, ebox, Lm, fb, L % nl))
                L += 1
        # hardest first: far layers need the largest cross-layer gaps, so
        # they claim the empty field before the easy copies box them out
        # (a skipped late copy must never be a layer's only occupant)
        insts.sort(key=lambda t: (max(max(R[t[4]] + R[k],
                                          shear_px[t[4]][k])
                                      for k in range(nl) if k != t[4])
                                  if nl > 1 else 0, R[t[4]]),
                   reverse=True)

        margin = int(self.p.overlap_margin_px)
        cone_m = int(self.p.cone_margin_px)
        placed = []          # (cy, cx, r_copy, layer)
        table = []
        for k, box, Lm, foot, Lz in insts:
            # the patch travels in LOG space (smoothed, bg-referenced,
            # edge-tapered) and is exponentiated only at paste time
            patch = Lm
            tf = dict(scale=1.0, angle=0.0)
            if bool(self.p.transform_copies):
                patch, foot, tf = self._transform_copy(patch, foot, rng)
                if not foot.any():
                    continue
            oyc = box[0].start + (box[0].stop - box[0].start) / 2.
            oxc = box[1].start + (box[1].stop - box[1].start) / 2.

            pat, ft = patch, foot
            done = False
            for attempt in range(4):
                if attempt:
                    # shrink and retry: a smaller copy beats a lost one
                    sc = 0.8
                    pat = ndimage.zoom(pat.real, sc, order=1) + \
                        1j * ndimage.zoom(pat.imag, sc, order=1)
                    ft = ndimage.zoom(ft.astype(float), sc, order=0) > 0.5
                    pat = pat * ft
                    tf['scale'] *= sc
                    if not ft.any():
                        break
                r_copy = self._copy_radius(ft)
                h, w = ft.shape
                for _ in range(4000):
                    yc, xc = allowed[int(rng.integers(len(allowed)))]
                    y0, x0 = int(yc) - h // 2, int(xc) - w // 2
                    if y0 < 0 or x0 < 0 or y0 + h > ny or x0 + w > nx:
                        continue
                    dst = (slice(y0, y0 + h), slice(x0, x0 + w))
                    if not cov[dst][ft].all():
                        continue
                    yy, xx = np.nonzero(ft)
                    cyv, cxv = y0 + yy.mean(), x0 + xx.mean()
                    ok = True
                    for (pcy, pcx, pr_, pL) in placed:
                        if pL == Lz:
                            need = r_copy + pr_ + margin
                        else:
                            # both conditions: no beam cone reaches two
                            # layers' particles (footprints), and no
                            # scattered wave shears from one particle onto
                            # another (2 NA_eff dz)
                            need = (r_copy + pr_ + cone_m
                                    + max(R[Lz] + R[pL], shear_px[Lz][pL]))
                        if np.hypot(cyv - pcy, cxv - pcx) <= need:
                            ok = False
                            break
                    if not ok:
                        continue
                    # strong-scattering paste: exp(s * log O) multiplies the
                    # particle phase by s; the zero log rim pastes as 1
                    layers[Lz][dst] *= np.exp(s_pow * pat)
                    placed.append((float(cyv), float(cxv), r_copy, Lz))
                    table.append(dict(
                        label=k, layer=Lz,
                        dy=int(round(cyv - oyc)), dx=int(round(cxv - oxc)),
                        area=int(ft.sum()), cy=float(cyv), cx=float(cxv),
                        scale=float(tf['scale']), angle=float(tf['angle'])))
                    done = True
                    break
                if done:
                    break
            if not done:
                logger.warning(
                    "could not place copy of particle %d on layer %d within "
                    "the beam-cone constraint; skipping" % (k, Lz))

        if not table:
            raise RuntimeError(
                "no copies could be placed under the beam-cone constraint; "
                "shrink copy_scale, lower particle_copies or cone_margin_px")

        self._verify_sparsity(placed, R, z_sim, fr, fc)

        dev = np.abs(layers - 1.0)
        logger.info(
            "strong-scattering stats after O**%.2g: max |O-1| = %.3f, "
            "max |phase| = %.2f rad (unwrapped, from the log)"
            % (s_pow, dev.max(), s_pow * max_src))
        gaps = []
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                a, b = placed[i], placed[j]
                if a[3] == b[3]:
                    continue
                need = (a[2] + b[2] + cone_m
                        + max(R[a[3]] + R[b[3]], shear_px[a[3]][b[3]]))
                gaps.append(np.hypot(a[0] - b[0], a[1] - b[1]) - need)
        if gaps:
            logger.info("cross-layer slack beyond the shear+cone "
                        "requirement: min %.0f px" % min(gaps))
        return layers, table

    # ------------------------------------------------------------------ #
    #  Deterministic overlap-controlled placement (dose experiment)       #
    # ------------------------------------------------------------------ #
    def _build_layers_collinear(self, layers, prepped, s_pow, z_sim, R,
                                shear_px):
        """The single strongest segmented particle pasted on EVERY layer,
        collinear along x with adjacent centres overlap_gap_px apart, symmetric
        about the coverage centre. The gap is the ONLY variable, so eps1 is
        tuned by lateral overlap alone -- the abscissa the dose-criterion
        sweep needs."""
        nl = layers.shape[0]
        ny, nx = layers.shape[1], layers.shape[2]
        gap = float(self.p.overlap_gap_px)
        # strongest prepped particle is the shared template
        k0, ebox0, Lm0, fb0 = max(
            prepped, key=lambda t: float(np.abs(t[2].imag).max()))
        patch, foot = Lm0, fb0
        h, w = foot.shape
        r_copy = self._copy_radius(foot)
        yy, xx = np.nonzero(foot)
        fcy, fcx = yy.mean(), xx.mean()

        cov = self._coverage().cpu().numpy()
        ys, xs = np.nonzero(cov)
        ccy = int(round(0.5 * (ys.min() + ys.max())))
        ccx = int(round(0.5 * (xs.min() + xs.max())))

        placed, table = [], []
        for j in range(nl):
            cyv = ccy
            cxv = ccx + int(round((j - (nl - 1) / 2.0) * gap))
            y0 = min(max(cyv - h // 2, 0), ny - h)
            x0 = min(max(cxv - w // 2, 0), nx - w)
            dst = (slice(y0, y0 + h), slice(x0, x0 + w))
            layers[j][dst] *= np.exp(s_pow * patch)
            cy_a, cx_a = y0 + fcy, x0 + fcx
            placed.append((float(cy_a), float(cx_a), r_copy, j))
            table.append(dict(label=int(k0), layer=j, dy=0, dx=0,
                              area=int(foot.sum()), cy=float(cy_a),
                              cx=float(cx_a), scale=1.0, angle=0.0))

        fr = self.rows[:, self.fny // 2].cpu().numpy()
        fc = self.cols[:, self.fnx // 2].cpu().numpy()
        self._verify_sparsity(placed, R, z_sim, fr, fc)
        dev = np.abs(layers - 1.0)
        logger.info(
            "collinear overlap placement: gap %.0f px on %d layers, "
            "adjacent shear %.0f px, max |O-1| = %.3f"
            % (gap, nl, shear_px[0][1] if nl > 1 else 0.0, dev.max()))
        return layers, table

    # ------------------------------------------------------------------ #
    #  Figures: the layers gallery shows UNWRAPPED phase                  #
    # ------------------------------------------------------------------ #
    def _figures_sim(self, src, layers, z_sim, table, I_sim, tag):
        """Parent figures first, then the layers gallery is replaced with an
        unwrapped-phase version: at phase_scale strengths the particles pass
        +-pi and the wrapped np.angle display speckles."""
        super()._figures_sim(src, layers, z_sim, table, I_sim, tag)
        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            logger.warning("skimage missing: layers gallery stays wrapped")
            return

        cov = self._coverage().cpu().numpy()
        ys, xs = np.nonzero(cov)
        win = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

        nl = layers.shape[0]
        ph = []
        for j in range(nl):
            u_ph = np.asarray(unwrap_phase(np.angle(layers[j][win])))
            bg = np.abs(layers[j][win] - 1.0) < 1e-6
            if bg.any():
                u_ph -= np.median(u_ph[bg])
            ph.append(u_ph)
        vmin = min(p.min() for p in ph)
        vmax = max(p.max() for p in ph)

        ncol = min(nl, 4)
        nrow = int(np.ceil(nl / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow),
                                 squeeze=False)
        for j in range(nrow * ncol):
            ax = axes[j // ncol][j % ncol]
            if j >= nl:
                ax.axis('off')
                continue
            im = ax.imshow(ph[j], cmap='gray', vmin=vmin, vmax=vmax)
            npart = sum(1 for t in table if t['layer'] == j)
            ax.set_title('layer %d, z = %.0f um, %d particles, min %.2f rad'
                         % (j, z_sim[j] * 1e6, npart, ph[j].min()))
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046,
                         label='unwrapped phase [rad]')
        fig.suptitle('simulated specimen (%d layers, unwrapped phase)' % nl)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, '%s_layers.png' % tag), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Direct verification of the stated condition                        #
    # ------------------------------------------------------------------ #
    def _verify_sparsity(self, placed, R, z_sim, fr, fc):
        """For every scan position, list which layers the beam cone touches;
        the specimen is sparse iff no position touches two layers."""
        hits_per_pos = []
        viol = 0
        for py, px in zip(fr, fc):
            touched = set()
            for (cy, cx, r_copy, Lz) in placed:
                if np.hypot(py - cy, px - cx) < R[Lz] + r_copy:
                    touched.add(Lz)
            hits_per_pos.append(touched)
            if len(touched) > 1:
                viol += 1
        n_hit = sum(1 for t in hits_per_pos if t)
        msg = ("beam-cone sparsity check: %d/%d positions hit a particle, "
               "%d positions hit particles on MORE THAN ONE layer"
               % (n_hit, len(hits_per_pos), viol))
        (logger.warning if viol else logger.info)(msg)
        self._results['sparsity'] = dict(
            footprint_px=[float(r) for r in R],
            z_um=(z_sim * 1e6).tolist(),
            positions_hitting_any=int(n_hit),
            positions_hitting_multi=int(viol),
            n_positions=int(len(hits_per_pos)),
            n_copies=len(placed))
