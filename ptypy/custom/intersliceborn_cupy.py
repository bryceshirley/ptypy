"""
Inter-slice Born series engine for ptychography, CuPy backend: the
order-M truncation of the exact Born expansion of multislice. order = 1
is the inter-slice first Born (ISS) model; order = N is exact
multislice, because the series terminates after N orders for N slices.
"""

import os
import sys

import numpy as np

from ptypy.custom.multislice_utils import (
    crop_pad_last2, normalize_slice_pad)

from .intersliceborn_utils import (
    BornCudaStep, cumulative_propagators, farfield_chirp,
    pull_storages_to_host, write_slice_file)
from ptypy.accelerate.cuda_cupy.mem_utils import \
    make_pagelocked_paired_arrays as mppa
from ptypy.engines import register, stochastic
from ptypy.utils.verbose import logger

import cupy as cp

# The core is the fused single-precision CUDA kernel set, so complex64
# is the engine's dtype throughout. There is no precision switch.
_EPS = float(np.finfo(np.float32).eps)


def slice_containers(ob, n):
    """
    n copies of the object container, under IDs no other engine has taken.

    A second multislice engine in the same run asks for the same names as the
    first, and ptypy's pool answers a duplicate container ID with an
    UnboundLocalError from _new_ptypy_object rather than with a message. That
    is what a probe warm-up stage followed by a multislice stage looks like, so
    the names are made unique here instead.
    """
    pool = getattr(ob.owner, "containers", {})
    out = []
    for i in range(n):
        ID, k = f"{ob.ID}_o_{i}", 1
        while ID in pool:
            ID = f"{ob.ID}_o{k}_{i}"
            k += 1
        out.append(ob.copy(ID))
    return out

@register()
class IntersliceBorn(stochastic.EPIE):
    """
    Inter-slice Born series multislice ptychography on the GPU, CuPy
    backend. order = 1 is the inter-slice first Born model;
    order = sequential depth.


    Defaults:

    [name]
    default = IntersliceBorn
    type = str
    help = Engine identifier

    [order]
    default = 1
    type = int
    help = Truncation order M of the Born series
    doc = Clamped to [1, number_of_slices]. 1 is the inter-slice first Born
          model; number_of_slices terminates the series and is
          exact multislice. Each order adds one scatter-propagate step, so
          the cost grows linearly and the sequential depth is M, not N.

    [number_of_slices]
    default = 2
    type = int
    help = Number of object slices

    [background]
    default = mean
    type = str
    help = Background a_j the series is expanded about: 'vacuum' or 'mean'
    doc = Centred series (paper eq. 3.3): the perturbation becomes
          eps_j = O_j/a_j - 1 and the gauge A = prod_j a_j is restored on
          the detector field. 'vacuum' is a_j = 1, the plain series
          g_j = O_j - 1; 'mean' is a_j = <O_j>, the mean transmission of
          slice j over the patch.

          a_j is recomputed from the current object at every update and
          then held fixed, so it never enters the adjoint.


    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Slice separation in meters
    doc = A single float, or a list of length number_of_slices - 1.

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the reconstructed slices

    [shuffle]
    default = False
    type = bool
    help = Reshuffle the position order every iteration
    doc = Set False to walk views in sorted order, as the CPU engine does with
          its shuffle disabled.

    [seed]
    default = 123
    type = int
    help = Seed for the position order and the object start; 0 leaves them free
    doc = Set the same non-zero seed for both runs of a paired comparison, and
          for both engines, so that nothing but the thing under test differs.

    [slice_pad]
    default = auto
    type = str
    help = Zero-padding factor for the inter-slice near-field propagation
    doc = Positive integer, or 'auto' for the smallest factor that meets
          the angular-spectrum sampling limit at the largest slice gap
          (capped at 4), the same rule the ThreePIE engines use. Padding
          keeps the pixel size and enlarges the propagation grid, so the
          field that leaves the patch has somewhere to go instead of
          wrapping onto the other side. Every transform in the core then
          runs on pad^2 times the pixels, so the cost scales with it. 1
          disables it and leaves [slice_bandlimit] as the only guard.

    [slice_bandlimit]
    default = True
    type = bool
    help = Apply angular-spectrum anti-aliasing support for slice propagation
    doc = The same mask the ThreePIE engines apply
          (multislice_utils.slice_bandlimit), here on each cumulative
          branch propagator H_z.
    """

    BACKGROUNDS = {
        "vacuum": "vacuum (a_j = 1, g = O - 1)",
        "mean": "centred on the patch mean a_j = <O_j>, frozen per update",
    }

    @property
    def _centred(self):
        return self._background != "vacuum"

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.article = dict(
            title='Parallel-in-depth ptychography of thick samples',
            author='B. Shirley et al.',
            journal='in preparation',
            volume=0,
            year=2026,
            page=0,
            doi='',
            comment='The inter-slice Born series reconstruction algorithm',
        )
        ptycho_parent.citations.add_article(**self.article)

    def _save_run_script(self):
        """Copy the entry script next to the slice file, so an output
        directory records exactly what produced it."""
        import shutil
        src = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        if not src.endswith(".py") or not os.path.isfile(src):
            return
        dst_dir = os.path.dirname(os.path.abspath(str(self.p.fslices))) or "."
        try:
            shutil.copy2(src, os.path.join(
                dst_dir, "run_script_" + os.path.basename(src)))
            logger.info("run script saved: %s" % os.path.basename(src))
        except OSError as e:
            logger.warning("could not save the run script: %s" % e)

    def _log_git_commit(self):
        """Record the ptypy git commit (and dirty state) in the run log."""
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            head = subprocess.run(
                ["git", "-C", here, "rev-parse", "--short=12", "HEAD"],
                capture_output=True, text=True, timeout=5)
            if head.returncode != 0:
                return
            dirty = subprocess.run(
                ["git", "-C", here, "status", "--porcelain", "-uno"],
                capture_output=True, text=True, timeout=5)
            flag = " (dirty)" if dirty.stdout.strip() else ""
            logger.info("ptypy git commit: %s%s"
                        % (head.stdout.strip(), flag))
        except (OSError, subprocess.SubprocessError):
            pass

    def _setup_infra(self):
        N = self.p.number_of_slices
        self._obs = next(iter(self.ob.storages.values()))
        self._prs = next(iter(self.pr.storages.values()))
        self.M, self.fny, self.fnx = self._prs.data.shape

        self._object = slice_containers(self.ob, N)
        self._probe = [self.pr]
        seed = int(self.p.seed)
        ob0 = np.ones_like(self._obs.data[0])
        rng = np.random.default_rng(seed if seed else None)
        for i in range(N):
            noise = ((rng.random(ob0.shape) - 0.5)
                     + 1j * (rng.random(ob0.shape) - 0.5))
            for st in self._object[i].storages.values():
                st.data[0][:] = ob0 + 1e-3 * noise
        logger.info(
            "object start: 1 + 1e-3 noise, seed %s"
            % (seed if seed else "free (not reproducible)")
        )
        oID = next(iter(self._object[0].S))
        self.ob_g = cp.asarray(
            np.stack([self._object[i].S[oID].data[0] for i in range(N)]),
            dtype=cp.complex64)
        self.pr_g = cp.asarray(np.ascontiguousarray(self._prs.data),
                               dtype=cp.complex64)

        self._order_rng = None
        if seed:
            self._order_rng = np.random.default_rng(seed + 1)
            logger.info(f"seed = {seed}: the position order is reproducible")
        self.ony, self.onx = self._obs.data.shape[-2:]

        geom = next(iter(self.ptycho.model.scans.values())).geometries[0]
        spad = self.p.get("slice_pad", 1)
        if isinstance(spad, str) and spad.strip().lower() != "auto":
            spad = int(spad)
        self._pad = normalize_slice_pad(
            spad, geom.shape, geom.resolution,
            float(np.ravel(geom.energy)[0]), self.p.slice_thickness)
        self.det_shape = tuple(int(v) for v in geom.shape)
        H_fwd, pshape = cumulative_propagators(
            geom, self.p.slice_thickness, N, pad=self._pad,
            bandlimit=bool(self.p.get("slice_bandlimit", True)))
        logger.info("inter-slice propagation grid: %dx%d (pad %d of the "
                    "%dx%d frame)" % (pshape + (self._pad,) + self.det_shape))
        self.H_fwd = cp.asarray(H_fwd, dtype=cp.complex64)
        self.H_rev = cp.asarray(np.conj(H_fwd), dtype=cp.complex64)

        self._collect_views()

        self._err_history = []
        self._save_run_script()
        self._log_git_commit()

    def _collect_views(self):
        names = sorted(n for n, v in self.di.views.items() if v.active)
        rows, cols, amps, masks = [], [], [], []
        trivial = True
        for n in names:
            pod = self.di.views[n].pod
            lo = np.asarray(pod.ob_view.dlow, dtype=np.int64)
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))
            I = np.asarray(self.di.views[n].data, dtype=np.float64)
            amps.append(np.sqrt(np.maximum(I, 0.0)))
            m = (
                np.ones_like(I, dtype=bool)
                if pod.mask is None
                else np.asarray(pod.mask) > 0
            )
            trivial = trivial and bool(m.all())
            masks.append(m)

        self.n_pos = len(names)
        self.rows = cp.asarray(np.stack(rows), dtype=cp.int64)
        self.cols = cp.asarray(np.stack(cols), dtype=cp.int64)
        self.amp_dat = cp.asarray(np.stack(amps), dtype=cp.float32)
        self.maskb = (
            None if trivial else cp.asarray(np.stack(masks), dtype=cp.bool_)
        )

    def engine_prepare(self):
        """Move the store to the device, as ThreePIE_cupy does: every
        slice storage gets pinned host memory and a device array. The
        device side is a VIEW into the engine's stacked array, so the
        container the views address and the memory the core reads are
        one allocation and cannot drift apart."""
        super().engine_prepare()
        for container in self._object + self._probe:
            for storage in container.S.values():
                if not hasattr(storage, "gpu"):
                    storage.gpu, storage.data = mppa(storage.data)
        self._bind_store()
        pID = next(iter(self._probe[0].S))
        supp = self._probe_support.get(pID)
        self.pr_support = (None if supp is None
                           else cp.asarray(supp, dtype=cp.float32))
        if supp is not None:
            logger.info("probe support: ptypy circular, area fraction %.3f"
                        % float(self.p.probe_support))

    def _bind_store(self):
        """Point every slice storage's device array at its plane of the
        engine's array. Called again whenever a subclass replaces ob_g."""
        oID = next(iter(self._object[0].S))
        for i in range(self.p.number_of_slices):
            self._object[i].S[oID].gpu = self.ob_g[i:i + 1]
        self._probe[0].S[next(iter(self._probe[0].S))].gpu = self.pr_g

    def _sync_primary_gpu_arrays(self):
        """Host-side copy of the store, then the product object in
        self.ob, as ThreePIE_cupy does. The device arrays are views on
        ob_g already, so only the host mirrors need writing."""
        oID = next(iter(self._object[0].S))
        pID = next(iter(self._probe[0].S))
        for i in range(self.p.number_of_slices):
            self._object[i].S[oID].data[0][:] = cp.asnumpy(self.ob_g[i])
        self._probe[0].S[pID].data[:] = cp.asnumpy(self.pr_g)
        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

    def _flat(self, pos):
        """Flat indices into one object slice of the patch at position
        pos, in the patch's own row-major order."""
        return (self.rows[pos][:, None] * self.onx
                + self.cols[pos][None, :]).reshape(-1)

    def _grad_to_param(self, grad):
        """Chain rule from the model perturbation g to the stored parameter.

        The chord engine's parameter is O with g = O - 1, so dg/dO = 1 and
        this is the identity. The tangent subclass stores s with g = i s and
        overrides this with the conjugate factor, -i.
        """
        return grad

    def _write_slices(self):
        self._stream.synchronize()
        self._sync_primary_gpu_arrays()
        pull_storages_to_host(self._object + self._probe)
        write_slice_file(
            self.p.fslices, self._object, probes=self._probe,
            extras={"number_of_slices": self.p.number_of_slices,
                    "slice_thickness": self.p.slice_thickness,
                    "order": self._M,
                    "background": self._background,
                    "slice_pad": self._pad})


    def engine_initialize(self):
        super().engine_initialize()
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._setup_infra()
        N = int(self.p.number_of_slices)

        self._acc_grad = cp.zeros((N, self.ony, self.onx),
                                  dtype=cp.complex64)
        self._acc_K = cp.zeros((N, self.ony, self.onx), dtype=cp.float32)
        self._err = cp.zeros(3, dtype=cp.float64)
        # the frozen background and gauge of the current update, set by
        # _update_gauge before every call to _model_object
        self._bg = None
        self._gauge = None

        self._M = max(1, min(int(self.p.order), N))
        if int(self.p.order) != self._M:
            logger.warning("order %d clamped to %d (number_of_slices)"
                           % (int(self.p.order), self._M))
        logger.info("inter-slice Born series: order M = %d of N = %d%s"
                    % (self._M, N,
                       " (exact multislice)" if self._M == N else ""))
        self._background = str(self.p.background or "mean").lower()
        if self._background not in self.BACKGROUNDS:
            raise ValueError("background must be one of %s, got %r"
                             % (sorted(self.BACKGROUNDS), self.p.background))
        logger.info("background: %s" % self.BACKGROUNDS[self._background])
        self._setup_raw_step()

    def _setup_raw_step(self):
        """Preallocate the fused-kernel core and alias the probe view.
        These kernels are the only core, so a geometry they do not
        support is an error rather than a fallback."""
        pod = next(iter(self.di.views.values())).pod
        chirp = farfield_chirp(pod.geometry.propagator,
                               (self.fny, self.fnx))
        N = int(self.p.number_of_slices)
        with self._stream:
            self._raw_step = BornCudaStep(
                cp.zeros((N, self.fny, self.fnx), dtype=cp.complex64),
                self.pr_g,
                cp.zeros((self.fny, self.fnx), dtype=cp.float32),
                self.H_fwd[:, 0, 0], self.H_rev[:, 0, 0], self._M,
                pre=chirp,
                det_shape=self.det_shape if self._pad != 1 else None)

    def _update_view(self, pos, do_probe):
        """
        One scan position: gather the slice patches, run the order-M core,
        track the errors, and apply the preconditioned Jacobi update.
        The same update at every order, except for the core call; device
        work only.
        """
        N = int(self.p.number_of_slices)
        flat = self._flat(pos)

        O_patches = cp.stack([
            self.ob_g[j].reshape(-1)[flat].reshape(self.fny, self.fnx)
            for j in range(N)
        ])
        self._update_gauge(O_patches)
        amp_dat = self.amp_dat[pos]
        mask = (None if self.maskb is None
                else self.maskb[pos].astype(cp.float32))

        st = self._raw_step
        st.O[...] = self._model_object(O_patches)
        st.Idat[...] = amp_dat ** 2
        if mask is not None:
            st.mask[...] = mask
        st.g2[0] = (1.0 if self._gauge is None
                    else cp.abs(self._gauge) ** 2)
        st.run()

        self._accumulate_error(amp_dat, mask)

        grad_g = (st.grad if self._bg is None
                  else st.grad / cp.conj(self._bg))
        self._acc_grad.fill(0)
        self._acc_K.fill(0)
        for j in range(N):
            self._acc_grad[j].reshape(-1)[flat] = grad_g[j].ravel()
            self._acc_K[j].reshape(-1)[flat] = st.K[j].ravel()

        if do_probe:
            self.pr_g += (float(self.p.beta) / N) * (
                st.gP / cp.maximum(st.KP.max(), _EPS))
            if self.pr_support is not None:
                self.pr_g *= self.pr_support

        dn_o = self._acc_K.max(axis=(-2, -1), keepdims=True)
        upd = self._grad_to_param(self._acc_grad / cp.maximum(dn_o, _EPS))
        self.ob_g += (float(self.p.alpha) / N) * upd

    def _accumulate_error(self, amp_dat, mask):
        """Add this position's three error metrics to the iteration sum:
        masked-mean amplitude l2, masked-mean Poisson-weighted intensity
        l2, and the l1 norm of the residual field."""
        st = self._raw_step
        I_dat = amp_dat ** 2
        amp_mod = cp.sqrt(cp.maximum(st.Imod, 0))
        weight = 1.0 if mask is None else mask
        norm = (float(st.Imod.size) if mask is None
                else cp.maximum(mask.sum(), 1))
        e0 = (weight * (amp_mod - amp_dat) ** 2).sum() / norm
        e1 = (weight * (st.Imod - I_dat) ** 2 / (I_dat + 1.0)).sum() / norm
        self._err += cp.stack(
            (e0, e1, cp.abs(st.rI).sum())).astype(cp.float64)

    def _update_gauge(self, O_patches):
        """Freeze this update's background a_j and gauge A = prod_j a_j.

        p.background selects a_j (paper eq. 3.3): 'vacuum' is a_j = 1,
        handled by returning None so the core keeps g = O - 1; 'mean' is
        a_j = <O_j>, the mean transmission of slice j over the patch.

        a_j is frozen: recomputed here from the current object at the
        start of every update, then treated as a constant by the adjoint.
        That keeps the hand-written gradient untouched; only the 1/a_j
        of the reparametrisation and the conj(A) on the residual enter it.
        """
        if not self._centred:
            self._bg = None
            self._gauge = None
            return
        self._bg = O_patches.mean(axis=(-2, -1), keepdims=True)
        self._gauge = cp.prod(self._bg)

    def _model_object(self, O_patches):
        """Centred: the core forms g = O/a - 1 = eps, the perturbation
        about the frozen background. Uncentred: identity, g = O - 1."""
        if self._bg is None:
            return O_patches
        return O_patches / self._bg

    def _position_order(self):
        if not self.p.shuffle:
            return np.arange(self.n_pos)
        rng = self._order_rng if self._order_rng is not None else np.random
        return rng.permutation(self.n_pos)

    def engine_iterate(self, num=1):
        cp.cuda.Device().synchronize()
        err = None
        for _ in range(num):
            do_probe = self.curiter >= self.p.probe_update_start
            with self._stream:
                self._err.fill(0)
                for pos in self._position_order():
                    self._update_view(int(pos), do_probe)

            self._stream.synchronize()
            err = cp.asnumpy(self._err / float(self.n_pos))
            self._err_history.append(err.copy())

            self.curiter += 1
        return {"all": err}

    def engine_finalize(self):
        self._write_slices()
        return super().engine_finalize()


@register()
class IntersliceBornLinearised(IntersliceBorn):
    """
    Order-M inter-slice Born series with the tangent (linearised-
    transmission) perturbation, CuPy backend: the reconstructed unknown is
    the scattering potential s_j with g_j = i s_j, exactly as
    the tangent is to the chord engine. The truncation order, batching,
    propagators, probe update and seeding are all inherited unchanged.

    Defaults:

    [name]
    default = IntersliceBornLinearised
    type = str
    help = Engine identifier

    [potential_real]
    default = False
    type = bool
    help = Constrain the potential s to be real, i.e. a pure phase object
    doc = The strict weak-phase-object convention: g = i s with s real
          forbids absorption. False reconstructs a complex s, whose imaginary
          part carries the absorption.
    """

    def engine_initialize(self):
        super().engine_initialize()
        self.ob_g = -1j * cp.log(self.ob_g)
        if bool(self.p.potential_real):
            self.ob_g = self.ob_g.real.astype(cp.complex64)
            logger.info("tangent potential constrained to be real "
                        "(pure phase)")
        self._bind_store()

    def _model_object(self, O_patches):
        return 1.0 + 1j * O_patches

    def _grad_to_param(self, grad):
        g = -1j * grad
        return (g.real.astype(cp.complex64) if bool(self.p.potential_real)
                else g)

    def engine_finalize(self):
        self.ob_g = cp.exp(1j * self.ob_g)
        self._bind_store()
        return super().engine_finalize()


@register()
class IntersliceFirstBornLineSearch(IntersliceBorn):
    """
    Inter-slice first Born with a line search on the step size.

    The order-1 model is affine in each unknown, so substituting an update
    along a search direction makes the detector field affine in the step
    size, psi(g) = psi + g*dpsi. The model intensity is then exactly
    quadratic in g and the intensity Gaussian likelihood is a quartic,
    whose minimiser is a real root of the derivative cubic. That root is
    the exact minimiser along the direction, found with one extra forward
    evaluation and no tuned step size.

    Object and probe are searched separately, as a block coordinate
    descent. Keeping them separate is what holds the likelihood at degree
    four: a common step would make the field quadratic in the step size
    and the likelihood degree eight. The object step is taken first, after
    which the updated detector field is known in closed form, so the probe
    step is evaluated against the updated object. The probe direction is
    the one computed before the object update, which avoids a second
    backward pass; it is no longer exactly the gradient there, but the
    search still finds the best step along it, which may be negative.

    Because the search rescales the whole direction, any common factor is
    absorbed into the step. The slice preconditioners therefore only set
    the relative weighting between slices, and the probe preconditioner
    has no effect at all. The inherited alpha and beta step sizes play no
    part: the step is what the line search returns.

    The series is affine in the step size only at order 1; at order M the
    field is a polynomial of degree M. The order is therefore not a
    parameter of this engine, it is set to 1.

    Defaults:

    [name]
    default = IntersliceFirstBornLineSearch
    type = str
    help = Engine identifier

    [fidelity]
    default = amplitude
    type = str
    help = Functional the step size minimises: 'gaussian' or 'amplitude'
    doc = 'gaussian' is the intensity Gaussian likelihood, weighted by
          w/(I + 1) as in ptypy's Gaussian model, for which the step is
          the exact minimiser along the direction. 'amplitude' is the
          amplitude-l2 fidelity the search directions themselves come
          from; it has no closed-form minimiser, so a single Newton step
          is taken, exact in its second-order expansion.

    [wavefield_delta_object]
    default = 0.1
    type = float
    help = Wavefield preconditioner damping constant for the object.

    [wavefield_delta_probe]
    default = 0.1
    type = float
    help = Wavefield preconditioner damping constant for the probe.
    """

    FIDELITIES = ("gaussian", "amplitude")

    def engine_initialize(self):
        # Not a parameter of this engine: only order 1 makes the model
        # affine in the object step, which is what the line search needs.
        self.p.order = 1
        super().engine_initialize()
        self._fidelity = str(self.p.fidelity or "gaussian").lower()
        if self._fidelity not in self.FIDELITIES:
            raise ValueError("fidelity must be one of %s, got %r"
                             % (sorted(self.FIDELITIES), self.p.fidelity))
        logger.info("line search: exact quartic on the intensity Gaussian "
                    "likelihood" if self._fidelity == "gaussian" else
                    "line search: Newton step on the amplitude-l2 fidelity")

    def _step(self, weight, mask):
        """Step size along the direction whose detector field is in Dd."""
        st = self._raw_step
        if self._fidelity == "gaussian":
            return st.line_search_gaussian(weight)
        return st.line_search_newton(mask)

    def _update_view(self, pos, do_probe):
        """
        One scan position: the same gradients as the parent engine, but
        the scalar step along each is found by a line search instead of
        being fixed. Device work only.
        """
        N = int(self.p.number_of_slices)
        flat = self._flat(pos)

        O_patches = cp.stack([
            self.ob_g[j].reshape(-1)[flat].reshape(self.fny, self.fnx)
            for j in range(N)
        ])
        self._update_gauge(O_patches)
        amp_dat = self.amp_dat[pos]
        mask = (None if self.maskb is None
                else self.maskb[pos].astype(cp.float32))

        st = self._raw_step
        st.O[...] = self._model_object(O_patches)
        st.Idat[...] = amp_dat ** 2
        if mask is not None:
            st.mask[...] = mask
        st.g2[0] = (1.0 if self._gauge is None
                    else cp.abs(self._gauge) ** 2)
        st.run()

        self._accumulate_error(amp_dat, mask)

        # ptypy's Gaussian weighting: the Poisson variance offset by one
        # count, so empty pixels keep a finite weight
        w = 1.0 if mask is None else mask
        weight = w / (st.Idat + 1.0)

        # object: the preconditioned gradient sets the direction, the
        # line search sets how far along it to go
        dn_o = st.K.max(axis=(-2, -1), keepdims=True)
        d = st.grad / (dn_o + float(self.p.wavefield_delta_object))
        st.direction_object(d)
        gamma = self._step(weight, w)
        st.advance(gamma, d)
        # d is a direction in the model perturbation; the stored
        # parameter is the object itself, so undo the centring
        upd = self._grad_to_param(d if self._bg is None else d * self._bg)
        for j in range(N):
            self.ob_g[j].reshape(-1)[flat] += (gamma * upd[j]).ravel()

        if do_probe:
            d_P = st.gP / (st.KP.max() + float(self.p.wavefield_delta_probe))
            st.direction_probe(d_P)
            beta = self._step(weight, w)
            self.pr_g += beta * d_P
            if self.pr_support is not None:
                self.pr_g *= self.pr_support
