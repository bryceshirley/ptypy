# -*- coding: utf-8 -*-
"""
First Born multislice engine for ptychography (CPU).

FORWARD MODEL
-------------
The sample is a stack of N thin slices. Referred to the entrance plane, the
first Born exit wave is

    phi_j = F^-1[ H_j F[P] ]                        illumination at slice j
    psi   = c * ( P + sum_j F^-1[ Hbar_j F[ s_j phi_j ] ] )

with H_j = exp(2i pi (z_j/lam) (sqrt(1 - (lam f)^2) - 1)) and H_0 = 1.

Slices enter as a SUM, not a product, and phi_j contains no object -- it is the
unscattered probe propagated to slice j. Both properties follow from dropping
multiple scattering, and together they make the model linear in {s_j} and free
of any sequential dependency in z.

Optionally each slice is expanded about its complex mean,

    O_j = Obar_j (1 + s_j),    c = prod_j Obar_j     (mean_field = True)

which resums the constant part of the transmission exactly into c and pins
<s_j> = 0, removing the DC gauge freedom between slices. With mean_field = False
this reduces to s_j = O_j - 1 and c = 1.
"""
from ptypy.engines import stochastic
from ptypy.engines import register
from ptypy.utils import Param
from ptypy.utils.verbose import logger
from ptypy import utils as u
from ptypy import io
import numpy as np
import scipy.fft as sfft

import os
from datetime import datetime
from matplotlib import pyplot as plt


@register()
class FirstBorn(stochastic.EPIE):
    """
    First Born multislice ptychography.

    Defaults:

    [name]
    default = FirstBorn
    type = str
    help = Engine identifier

    [number_of_slices]
    default = 2
    type = int
    help = Number of object slices

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Slice separation in meters
    doc = A single float, or a list of length number_of_slices - 1.

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the reconstructed slices

    [alpha]
    default = 0.2
    type = float
    help = Object update step size
    doc = Divided by number_of_slices internally. Per slice the normal operator
          A_j^dag A_j = |phi_j|^2 is exactly diagonal, so a full step is correct
          for one slice alone; updating N slices simultaneously is a Jacobi
          iteration and needs step < 2/N.

    [beta]
    default = 0.2
    type = float
    help = Probe update step size
    doc = Divided by number_of_slices internally, as for object_step.

    [object_denom]
    default = max
    type = str
    help = Object preconditioner: 'max' or 'local'
    doc = 'max' uses the scalar max sum_m |phi_j,m|^2 (ePIE convention).
          'local' uses the per-pixel sum_m |phi_j,m|^2 + denom_reg * max, which
          is the exact diagonal of A_j^dag A_j since the propagator is unitary.

    [denom_reg]
    default = 0.01
    type = float
    help = Floor on the 'local' denominator, as a fraction of its maximum

    [mean_field]
    default = False
    type = bool
    help = Expand each slice about its complex mean instead of about vacuum

    [mean_field_relax]
    default = 0.5
    type = float
    help = Relaxation on the Obar update; Obar <- (1-r) Obar + r <O>

    [fft_workers]
    default = -1
    type = int
    help = Threads for scipy.fft; -1 uses all cores
    """

    def __init__(self, ptycho_parent, pars=None):
        super(FirstBorn, self).__init__(ptycho_parent, pars)
        self.article = dict(
            title='Ptychographic transmission microscopy in three dimensions using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29, year=2012, page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        self.ptycho.citations.add_article(**self.article)

        self.plot_progress_dir = (f'iteration_progress{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.plot_progress_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()

        N = self.p.number_of_slices
        self._w = int(self.p.fft_workers)

        # object slices, vacuum plus a little noise to break symmetry
        self._object = []
        for i in range(N):
            ob = self.ob.copy(self.ob.ID + "_o_%d" % i)
            for st in ob.storages.values():
                n = (np.random.rand(*st.data.shape) - 0.5) \
                    + 1j * (np.random.rand(*st.data.shape) - 0.5)
                st.data[:] = 1.0 + 1e-3 * n
            self._object.append(ob)

        # slice propagators for j = 1 .. N-1; H_0 = 1 identically and is never
        # stored or applied, which removes 1/N of the transforms
        geom = list(self.ptycho.model.scans.values())[0].geometries[0]
        shape, lam = geom.shape, geom.lam
        V, W = u.grids(shape, lam / shape / geom.resolution, 'fft')
        a2 = V**2 + W**2

        th = self.p.slice_thickness
        z = [sum(th[:i]) for i in range(1, N)] if type(th) in (list, tuple) \
            else [i * th for i in range(1, N)]

        self.H = np.empty((N - 1, *shape), dtype=np.complex128)
        for i, zi in enumerate(z):
            # clamp the evanescent band so |H| == 1 exactly; referring the model
            # to the entrance plane by factoring out H^N depends on it
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            Hi = np.exp(2j * np.pi * (zi / lam) * kz)
            Hi[a2 > 1.0] = 0.0
            self.H[i] = Hi
        self.Hb = np.conj(self.H)

        # mean-field background
        self._Obar = np.ones(N, dtype=np.complex128)
        self._mf_c = 1.0 + 0.0j

        # illuminated region, so <O_j> is not diluted by the untouched border
        cov = self.ob.copy(self.ob.ID + "_cov")
        for st in cov.storages.values():
            st.data[:] = 0.0
        for view in self.di.views.values():
            if view.active:
                for pod in view.pods.values():
                    cov[pod.ob_view] += 1.0
        self._cov = {sid: np.abs(st.data) > 0 for sid, st in cov.storages.items()}
        frac = float(np.mean([m.mean() for m in self._cov.values()]))
        logger.info("illuminated fraction of object array: %.4f" % frac)
        if frac < 1e-3:
            raise RuntimeError("coverage mask empty; check pod.ob_view indexing")

        # phi depends only on the probe, so it can be reused until the probe moves
        self._phi_cache = {}

    # ------------------------------------------------------------------ #
    #  Mean-field background                                              #
    # ------------------------------------------------------------------ #
    def _slice_mean(self, j):
        num, den = 0.0 + 0.0j, 0
        for sid, st in self._object[j].storages.items():
            m = self._cov[sid]
            num += complex(st.data[m].sum())
            den += int(m.sum())
        return num / den if den else 1.0 + 0.0j

    def _update_mean_field(self):
        """Refresh Obar and c once per iteration, before the sweep, so the
        gradients below are exact for the whole sweep."""
        N = self.p.number_of_slices
        if not self.p.mean_field:
            self._Obar[:] = 1.0
            self._mf_c = 1.0 + 0.0j
            return

        r = float(self.p.mean_field_relax)
        new = np.array([self._slice_mean(j) for j in range(N)])
        new[np.abs(new) < 1e-6] = self._Obar[np.abs(new) < 1e-6]
        self._Obar = (1.0 - r) * self._Obar + r * new
        self._mf_c = complex(np.prod(self._Obar))

    # ------------------------------------------------------------------ #
    #  Forward, projection, adjoint                                       #
    # ------------------------------------------------------------------ #
    def _phi(self, pod):
        """Unscattered illumination at every slice, (N, ny, nx).

        phi_0 = P exactly, since H_0 = 1. Contains no object, so it depends only
        on the probe and is cached until the probe is updated.
        """
        key = int(pod.pr_view.dlayer)
        ph = self._phi_cache.get(key)
        if ph is None:
            P = self.pr[pod.pr_view]
            ph = np.empty((self.p.number_of_slices, *P.shape), dtype=np.complex128)
            ph[0] = P
            if self.p.number_of_slices > 1:
                ph[1:] = sfft.ifft2(self.H * sfft.fft2(P, workers=self._w),
                                    axes=(-2, -1), workers=self._w)
            self._phi_cache[key] = ph
        return ph

    def _fourier_update(self, view):
        """Mixed-state modulus projection: intensities sum over probe modes, one
        shared real scale is applied to every mode, masked pixels keep the
        model value."""
        pods = list(view.pods.values())
        I = np.asarray(view.data, dtype=np.float64)

        F = [pod.fw(pod.exit) for pod in pods]
        Imod = np.zeros_like(I)
        for f in F:
            Imod += np.abs(f)**2

        amod = np.sqrt(Imod)
        ameas = np.sqrt(np.maximum(I, 0.0))
        m = pods[0].mask
        m = np.ones_like(I) if m is None else np.asarray(m, dtype=np.float64)

        scale = np.where(m > 0, ameas / np.maximum(amod, 1e-12), 1.0)

        msum = max(float(m.sum()), 1.0)
        err = np.array([float((m * (amod - ameas)**2).sum() / msum),
                        float((m * (Imod - I)**2 / (I + 1.0)).sum() / msum),
                        0.0])
        for pod, f in zip(pods, F):
            new = pod.bw(f * scale)
            err[2] += float(np.abs(new - pod.exit).sum())
            pod.exit = new
        return err

    def firstBorn_update(self, view):
        N = self.p.number_of_slices
        pods = list(view.pods.items())

        # every pod in a view shares one object patch and differs only by probe
        # mode, so the slice stack is read once
        ov = pods[0][1].ob_view
        O = np.stack([self._object[j][ov] for j in range(N)])
        s = O / self._Obar[:, None, None] - 1.0

        # --- forward ---
        phi, model = {}, {}
        for name, pod in pods:
            ph = self._phi(pod)
            phi[name] = ph

            scat = s[0] * ph[0]                                  # H_0 = 1
            if N > 1:
                Fs = sfft.fft2(s[1:] * ph[1:], axes=(-2, -1), workers=self._w)
                scat = scat + sfft.ifft2((self.Hb * Fs).sum(0), workers=self._w)

            model[name] = self._mf_c * (ph[0] + scat)
            pod.exit = model[name].copy()

        err = self._fourier_update(view)

        # --- adjoint ---
        num = np.zeros_like(O)
        den = np.zeros(O.shape, dtype=np.float64)
        chain = np.conj(self._mf_c / self._Obar)[:, None, None]

        step_o = float(self.p.alpha) / N
        step_p = float(self.p.beta) / N

        for name, pod in pods:
            ph = phi[name]
            d = pod.exit - model[name]

            dj = np.empty_like(ph)
            dj[0] = d                                            # H_0 = 1
            if N > 1:
                dj[1:] = sfft.ifft2(self.H * sfft.fft2(d, workers=self._w),
                                    axes=(-2, -1), workers=self._w)

            num += chain * np.conj(ph) * dj
            den += np.abs(ph)**2

            if self.curiter >= self.p.probe_update_start:
                g = np.conj(s[0]) * dj[0]
                if N > 1:
                    Fg = sfft.fft2(np.conj(s[1:]) * dj[1:], axes=(-2, -1), workers=self._w)
                    g = g + sfft.ifft2((self.Hb * Fg).sum(0), workers=self._w)
                dP = np.conj(self._mf_c) * (d + g)
                dpn = np.abs(self._mf_c)**2 * np.max(np.sum(np.abs(O)**2, axis=0))
                self.pr[pod.pr_view] += step_p * dP / max(dpn, 1e-8)
                self._phi_cache.pop(int(pod.pr_view.dlayer), None)

        # one object update per view, summed over probe modes
        den *= np.abs(self._mf_c / self._Obar)[:, None, None]**2
        if str(self.p.object_denom).lower().startswith('l'):
            den = den + float(self.p.denom_reg) * den.max(axis=(-2, -1), keepdims=True)
        else:
            den = np.broadcast_to(den.max(axis=(-2, -1), keepdims=True), den.shape)

        upd = step_o * num / np.maximum(den, 1e-8)
        for j in range(N):
            self._object[j][ov] += upd[j]

        return err

    def _plot_iteration_progress(self):
        try:
            probe_base = self.pr.copy(self.pr.ID + "_p_base")
            probe_raw = next(iter(probe_base.storages.values())).data
            probe_data = probe_raw[0] if probe_raw.ndim == 3 else probe_raw

            slice_first_raw = next(iter(self._object[0].storages.values())).data
            slice_first = slice_first_raw[0] if slice_first_raw.ndim == 3 else slice_first_raw

            slice_final_raw = next(iter(self._object[-1].storages.values())).data
            slice_final = slice_final_raw[0] if slice_final_raw.ndim == 3 else slice_final_raw

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im0 = axes[0].imshow(np.abs(probe_data)**2, cmap='viridis')
            axes[0].set_title(f"Probe Intensity (Iter {self.curiter})")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(np.angle(slice_first), cmap='gray')
            axes[1].set_title(f"Slice 0 (First) Phase (Iter {self.curiter})")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(np.angle(slice_final), cmap='gray')
            axes[2].set_title(f"Slice {self.p.number_of_slices-1} (Final) Phase (Iter {self.curiter})")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            ob = self._Obar.ravel()
            plt.suptitle(f"|c| = {abs(self._mf_c):.4f}   "
                            f"|Obar| = {np.array2string(np.abs(ob), precision=3)}   "
                            f"arg(Obar) = {np.array2string(np.angle(ob), precision=3)}")
            plt.tight_layout()
            plt.savefig(f"{self.plot_progress_dir}/iter_progress_{self.curiter:04d}.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not plot iteration progress: {e}")

    # ------------------------------------------------------------------ #
    #  Iteration                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        order = sorted(self.di.views.keys())
        rng = np.random.default_rng()

        for _ in range(num):
            self._update_mean_field()
            rng.shuffle(order)

            err = {}
            for name in order:
                view = self.di.views[name]
                if view.active:
                    err[name] = self.firstBorn_update(view)

            logger.info("iter %d: |Obar| = %s, arg(Obar) = %s"
                        % (self.curiter,
                           np.array2string(np.abs(self._Obar), precision=4),
                           np.array2string(np.angle(self._Obar), precision=4)))
            self.curiter += 1

        return err

    def engine_finalize(self):
        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        info = Param()
        info.number_of_slices = self.p.number_of_slices
        info.slice_thickness = self.p.slice_thickness
        info.mean_field = self.p.mean_field
        info.object_denom = self.p.object_denom
        info.Obar = self._Obar
        info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                        for ob in self._object}

        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info("saving slices to %s" % self.p.fslices)
        io.h5write(self.p.fslices,
                   header={'description': 'First Born multislice reconstruction'},
                   content=info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()