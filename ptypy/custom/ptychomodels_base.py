"""
Shared ptypy plugin interface for the ptychomodels engines.

The host side: views, containers, geometry, propagator kernels, the position
sweep, the error history and the output file. The propagation, its adjoint and
the step rules come from ptychomodels.

A subclass supplies STEPS, _build_propagators, _update_view and _core_fn, and
inherits the rest including the parameters below, which ptypy's parse_doc
picks up from this base class.
"""

import numpy as np
import torch
from ptychomodels import intensity

from ptypy.custom.multislice_utils import normalize_slice_pad, slice_bandlimit
from ptypy.engines import stochastic
from ptypy.utils.verbose import logger

__all__ = [
    "PtychoModelsEngine",
    "farfield_chirp",
    "slice_containers",
    "slice_step_kernel",
]


def slice_containers(ob, n):
    """n copies of the object container, under IDs no other engine has taken:
    ptypy answers a duplicate container ID with an UnboundLocalError rather
    than a message."""
    pool = getattr(ob.owner, "containers", {})
    out = []
    for i in range(n):
        ID, k = f"{ob.ID}_o_{i}", 1
        while ID in pool:
            ID = f"{ob.ID}_o{k}_{i}"
            k += 1
        out.append(ob.copy(ID))
    return out


def slice_step_kernel(geom, slice_thickness, n_slices, pad=1, bandlimit=True):
    """The single-gap angular-spectrum kernel on the (optionally padded)
    propagation grid, as [*pshape] complex128.

    ptychomodels raises it to whatever powers a model needs — the
    cumulative powers for the Born series, one step at a time for the
    multislice sweep — which is what keeps the two the same operator and
    the exactness at order = N exact. Evaluating ptypy's geometry
    separately at each depth z_j would be slightly different kernels, and
    the two models would stop agreeing where they should.

    The band limit is therefore taken at the FULL depth rather than at
    the gap: the mask is a support, so the conservative one is the one
    that survives being raised to a power.
    """
    from ptypy.core import geometry
    from ptypy.utils import Param

    energy = float(np.ravel(geom.energy)[0])
    pshape = tuple(int(v) * int(pad) for v in geom.shape)
    gaps = (
        slice_thickness
        if isinstance(slice_thickness, (list, tuple))
        else [slice_thickness] * (n_slices - 1)
    )
    if len(set(gaps)) > 1:
        raise NotImplementedError("this engine takes a uniform slice separation")
    if not gaps:
        # ptypy's geometry raises on a zero distance rather than returning identity
        return np.ones(pshape, dtype=np.complex128), pshape

    gpar = Param()
    gpar.energy = energy
    gpar.psize = geom.resolution
    gpar.shape = pshape
    gpar.propagation = "nearfield"
    gpar.distance = gaps[0] if gaps else 0.0
    # an ownerless Geo pins its propagator to complex64; rebuild at full
    # precision through the public helper. The kernel is NaN in the evanescent region.
    G = geometry.Geo(owner=None, pars=gpar)
    kernel = np.nan_to_num(
        np.asarray(geometry.get_propagator(G.p, dtype=np.complex128).kernel, dtype=np.complex128)
    )
    if bandlimit:
        kernel = kernel * slice_bandlimit(
            kernel.shape, geom.resolution, energy, sum(gaps) or 1.0
        )
    return kernel, pshape


def farfield_chirp(propagator, tol=1e-5):
    """The entrance-plane chirp of a ptypy far-field propagator, or None when
    the geometry needs none. Only pre_fft is kept: post_fft is unimodular and
    cancels between the forward intensity and the adjoint residual. A geometry
    breaking those assumptions raises rather than reconstructing the wrong
    thing."""
    pre = np.asarray(propagator.pre_fft)
    post = np.asarray(propagator.post_fft)
    pre_i = np.asarray(propagator.pre_ifft)
    post_i = np.asarray(propagator.post_ifft)

    if (np.asarray(propagator.crop_pad) != 0).any():
        raise NotImplementedError("this engine does not support crop_pad geometries")
    if np.abs(np.abs(post) - 1.0).max() > tol:
        raise NotImplementedError(
            "a unimodular far-field post-filter is needed; |post_fft| departs from 1 by %.2e"
            % np.abs(np.abs(post) - 1.0).max()
        )
    if np.abs(pre_i - np.conj(post)).max() > tol or np.abs(post_i - np.conj(pre)).max() > tol:
        raise NotImplementedError("a conjugate-inverse far-field filter pair is needed")
    if np.abs(pre - 1.0).max() < tol:
        return None
    return pre


class PtychoModelsEngine(stochastic.EPIE):
    """
    Host plumbing shared by the ptychomodels engines. Not registered: it
    solves nothing on its own.

    Defaults:


    [number_of_slices]
    default = 2
    type = int
    help = The number of slices
    doc = Defines how many slices are used for the multi-slice object. 1 is the
          single-slice limit: no inter-slice propagation, and the models collapse to
          ordinary ptychography.

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = A uniform gap; a list is not yet supported here. The propagator is built once
          and raised to the powers each model needs, so the two engines share an
          operator exactly.

    [fix_probe_int]
    default = False
    type = bool
    help = Hold the probe's total intensity at its initial value
    doc = Off by default: the probe's total power looks like a gauge the data cannot see,
          but the overlap does constrain it. Available for a host that wants it.

    [compile]
    default = False
    type = bool
    help = Hand the per-view model region to torch.compile
    doc = The package's entry points survive compilation in one graph, fullgraph=True, and
          match eager to within 3e-7. Off by default because the gain depends on how much of
          the sweep the region is; worth turning on for more slices, more probe modes or
          bigger grids.

    [batch_size]
    default = 1
    type = int
    lowlim = 1
    help = Scan positions per update; only with step 'adam'
    doc = The projection steps are Gauss-Seidel over positions, so a batch is a different
          algorithm and they refuse it. Adam has no such ordering. A larger batch costs
          memory in proportion, the adjoint holding one sweep per position.

    [compile_mode]
    default = reduce-overhead
    type = str
    help = torch.compile mode, when compile is on
    doc = 'reduce-overhead' uses CUDA graphs, which is what a launch-bound region needs and
          what a batch-1 sweep usually is; 'max-autotune' measures the same. 'default' has no
          graphs and can be slower than eager, its guards costing without the launches being
          collapsed.

    [fslices]
    default = slices.h5
    type = str
    help = File path for the slice data
    doc =

    [shuffle]
    default = False
    type = bool
    help = Reshuffle the position order every iteration

    [seed]
    default = 123
    type = int
    help = Seed for the position order and the object start; 0 leaves them free
    doc = Set the same non-zero seed for both runs of a paired comparison.

    [slice_pad]
    default = 1
    type = int, str
    help = Zero-padding factor for inter-slice near-field propagation
    doc = Positive integer or ``"auto"``. Padding keeps the real-space pixel
          size and enlarges the propagation grid, which raises the
          angular-spectrum sampling limit for a fixed slice spacing, at pad^2
          times the pixels per transform. 1 leaves [slice_bandlimit] as the
          only guard.

    [slice_bandlimit]
    default = True
    type = bool
    help = Apply angular-spectrum anti-aliasing support for slice propagation

    [device]
    default = cuda
    type = str
    help = Torch device the object, probe and propagators live on

    [cpu_threads]
    default = 1
    type = int
    lowlim = 0
    help = Torch thread count to hold while this engine runs on the CPU
    doc = One update is a few transforms of a single frame, which is small enough that
          the thread pool spends more on barriers than the arithmetic saves; the default
          thread count is one per core and costs an order of magnitude on a large host.
          0 leaves the host's setting alone. Ignored on cuda.
    """

    #: step rules the subclass accepts; the first is its default
    STEPS = ()
    #: what the subclass calls itself in the slice file
    MODEL = "ptychomodels"

    def engine_initialize(self):
        super().engine_initialize()
        self._device = torch.device(self.p.device)
        self._threads = None
        cpu_threads = int(self.p.cpu_threads)
        if self._device.type == "cpu" and 0 < cpu_threads < torch.get_num_threads():
            self._threads = torch.get_num_threads()
            torch.set_num_threads(cpu_threads)
            logger.info("torch threads %d -> %d for the CPU run"
                        % (self._threads, cpu_threads))
        self._step = str(self.p.step or self.STEPS[0]).lower()
        if self._step not in self.STEPS:
            raise ValueError(
                "step must be one of %s, got %r" % (sorted(self.STEPS), self.p.step)
            )
        self._batch = max(1, int(self.p.batch_size))
        if self._batch > 1 and self._step != "adam":
            raise ValueError(
                "batch_size %d needs step 'adam': %r is Gauss-Seidel over "
                "positions and a batch is a different algorithm, not an "
                "approximation of it" % (self._batch, self._step)
            )
        self._configure()
        self._setup_infra()
        self._core = self._core_fn()
        if bool(self.p.compile):
            # fullgraph: a break anywhere gives the kernel launches back
            mode = str(self.p.compile_mode)
            self._core = torch.compile(self._core, mode=mode, fullgraph=True)
            logger.info("per-view model region compiled (%s, fullgraph)" % mode)
        if self._step == "adam":
            # alpha and beta are the learning rates here, not a second pair;
            # their inherited default of 1.0 is a projection step size
            self.ob_g.requires_grad_(True)
            self.pr_g.requires_grad_(True)
            self._optimizer = torch.optim.Adam(
                [
                    {"params": [self.ob_g], "lr": float(self.p.alpha)},
                    {"params": [self.pr_g], "lr": float(self.p.beta)},
                ]
            )
            logger.info(
                "adam: object lr %g (alpha), probe lr %g (beta), batch %d"
                % (float(self.p.alpha), float(self.p.beta), self._batch)
            )
            if float(self.p.alpha) >= 1.0 or float(self.p.beta) >= 1.0:
                # Adam's step is in parameter units, so the probe's rate has to
                # carry the ratio of the two moduli
                logger.warning(
                    "adam with alpha %g, beta %g: these are the inherited "
                    "PROJECTION step sizes, and as learning rates they are "
                    "large enough to diverge. On NanoMAX the object was "
                    "stable to alpha 8e-3 with beta 60x that."
                    % (float(self.p.alpha), float(self.p.beta))
                )

    # ---- hooks a subclass fills in -------------------------------------

    def _configure(self):
        """Read the model-specific parameters and log what will be run.
        Called before the device arrays exist."""

    def _build_propagators(self, H1, n_slices):
        """Turn the single-gap kernel into whatever the model steps with.
        H1 is already on the device, shaped (1, 1, 1, *grid)."""
        self.H1 = H1

    def _core_fn(self):
        """The per-view region handed to torch.compile when p.compile."""
        raise NotImplementedError

    def _update_view(self, pos, do_probe):
        raise NotImplementedError

    def _result_info(self, info):
        """Add model-specific fields to the slice file."""

    # ---- device setup ---------------------------------------------------

    def _setup_infra(self):
        N = int(self.p.number_of_slices)
        self._obs = next(iter(self.ob.storages.values()))
        self._prs = next(iter(self.pr.storages.values()))
        self.Mp, self.fny, self.fnx = self._prs.data.shape
        self.ony, self.onx = self._obs.data.shape[-2:]

        self._object = slice_containers(self.ob, N)
        self._probe = [self.pr]
        seed = int(self.p.seed)
        rng = np.random.default_rng(seed if seed else None)
        ob0 = np.ones_like(self._obs.data[0])
        for i in range(N):
            noise = (rng.random(ob0.shape) - 0.5) + 1j * (rng.random(ob0.shape) - 0.5)
            for st in self._object[i].storages.values():
                st.data[0][:] = ob0 + 1e-3 * noise
        logger.info(
            "object start: 1 + 1e-3 noise, seed %s" % (seed if seed else "free (not reproducible)")
        )

        oID = next(iter(self._object[0].S))
        self.ob_g = torch.as_tensor(
            np.stack([self._object[i].S[oID].data[0] for i in range(N)]),
            dtype=torch.complex64,
            device=self._device,
        )
        self.pr_g = torch.as_tensor(
            np.ascontiguousarray(self._prs.data), dtype=torch.complex64, device=self._device
        )
        self._order_rng = np.random.default_rng(seed + 1) if seed else None

        geom = next(iter(self.ptycho.model.scans.values())).geometries[0]
        spad = self.p.get("slice_pad", 1)
        if isinstance(spad, str) and spad.strip().lower() != "auto":
            spad = int(spad)
        # one slice: nothing for the pad to protect, at pad^2 the pixels
        self._pad = 1 if N == 1 else normalize_slice_pad(
            spad, geom.shape, geom.resolution, float(np.ravel(geom.energy)[0]), self.p.slice_thickness
        )
        self.det_shape = tuple(int(v) for v in geom.shape)
        step, pshape = slice_step_kernel(
            geom, self.p.slice_thickness, N, pad=self._pad,
            bandlimit=bool(self.p.get("slice_bandlimit", True)),
        )
        logger.info(
            "inter-slice propagation grid: %dx%d (pad %d of the %dx%d frame)"
            % (pshape + (self._pad,) + self.det_shape)
        )
        H1 = torch.as_tensor(step, dtype=torch.complex64, device=self._device).view(
            1, 1, 1, *pshape
        )
        self._build_propagators(H1, N)
        self._grid = pshape
        self._off = (pshape[0] - self.det_shape[0]) // 2
        self._centre = (
            Ellipsis,
            slice(self._off, self._off + self.det_shape[0]),
            slice(self._off, self._off + self.det_shape[1]),
        )

        chirp = farfield_chirp(next(iter(self.di.views.values())).pod.geometry.propagator)
        self._chirp = (
            None if chirp is None
            else torch.as_tensor(chirp, dtype=torch.complex64, device=self._device)
        )
        self._collect_views()
        self._err_history = []
        self._support = None  # filled in engine_prepare, see there
        self._report_illumination_scale()
        self._probe_power = float(self.pr_g.abs().square().sum())

    def _report_illumination_scale(self):
        """Warn when the probe is far off the data's amplitude scale.

        Left alone ptypy normalises the illumination against the data and this
        never fires; it catches a hand-set illumination.photons that is wrong,
        which the first modulus projection would otherwise have to make up in
        one step. Reported rather than corrected."""
        free = intensity(torch.fft.fft2(self.pr_g[None, :, None], norm="ortho"))
        mismatch = float((self.amp_dat.square().mean() / free.mean()).sqrt())
        if not 0.2 < mismatch < 5.0:
            logger.warning(
                "the illumination is %.3g x off the data's amplitude scale; set "
                "illumination.photons, or the first modulus projection has to make "
                "it up in one step" % mismatch
            )

    def _collect_views(self):
        names = sorted(n for n, v in self.di.views.items() if v.active)
        rows, cols, amps, masks = [], [], [], []
        trivial = True
        for n in names:
            pod = self.di.views[n].pod
            lo = np.asarray(pod.ob_view.dlow, dtype=np.int64)
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))
            # into the package's frame: intensity() fftshifts the model, so the
            # data is shifted with it. Without this every model pixel is paired
            # with the datum half a period away, which still "converges".
            I = np.fft.fftshift(np.asarray(self.di.views[n].data, dtype=np.float64))
            amps.append(np.sqrt(np.maximum(I, 0.0)))
            m = (
                np.ones_like(I, dtype=bool)
                if pod.mask is None
                else np.fft.fftshift(np.asarray(pod.mask) > 0)
            )
            trivial = trivial and bool(m.all())
            masks.append(m)

        self.n_pos = len(names)
        dev = self._device
        # patches are contiguous rectangles, so the origins are all that is
        # needed; kept on the host because basic slicing wants Python ints
        self.origins = np.stack([np.array([r[0], c[0]]) for r, c in zip(rows, cols)])
        self.rows = torch.as_tensor(np.stack(rows), device=dev)
        self.cols = torch.as_tensor(np.stack(cols), device=dev)
        self.amp_dat = torch.as_tensor(np.stack(amps), dtype=torch.float32, device=dev)
        self.maskb = (
            None if trivial
            else torch.as_tensor(np.stack(masks), dtype=torch.float32, device=dev)
        )
        # maskb is shifted, with amp_dat, where the error is evaluated; the
        # correction lives in the model's unshifted frame and needs its own copy
        self._mask_model = (
            None if self.maskb is None
            else torch.fft.ifftshift(self.maskb, dim=(-2, -1))
        )

    def engine_prepare(self):
        """Pick up the probe support, which only exists by this point.

        BaseEngine.prepare builds _probe_support and then calls this;
        engine_initialize runs earlier and would silently get an empty dict,
        leaving the probe unconstrained without any error. Fetched once, not
        per position, which would be a host transfer per view."""
        super().engine_prepare()
        support = self._probe_support.get(next(iter(self._probe[0].S)))
        self._support = (
            None
            if support is None
            else torch.as_tensor(support, dtype=torch.float32, device=self._device)
        )
        if support is not None:
            logger.info(
                "probe support: ptypy circular, area fraction %.3f"
                % float(self.p.probe_support)
            )
        elif self.p.probe_support is not None:
            # the exact shape the bug above took, so it cannot be silent twice
            logger.warning(
                "probe_support = %s was asked for but ptypy built no support for "
                "this storage; the probe will be left unconstrained"
                % (self.p.probe_support,)
            )

    # ---- the canvas, the grid and the detector --------------------------

    def _window(self, pos):
        """The canvas slice of the patch at pos, all slices at once. Basic
        slicing, so the patch is a view: nothing is gathered and the
        write-back is an in-place add."""
        r, c = self.origins[pos]
        return (slice(None), slice(r, r + self.fny), slice(c, c + self.fnx))

    def _patches(self, pos):
        """The object patches of one position, in the detector frame."""
        # [None, None] and not .view: a window into the canvas is strided
        return self.ob_g[self._window(pos)][None, None]

    def _patches_batch(self, positions):
        """Object patches for a batch, as (B, 1, Nz, fny, fnx): B windows
        stacked, the patches being contiguous rectangles."""
        return torch.stack([self.ob_g[self._window(p)] for p in positions]).unsqueeze(1)

    def _scatter_grad_batch(self, canvas, positions, grad):
        """Accumulate a batch of patch gradients onto the canvas, one add per
        member: positions overlap wherever the scan does, and separate adds
        accumulate there where a single indexed assignment would not."""
        if canvas.grad is None:
            canvas.grad = torch.zeros_like(canvas)
        for k, pos in enumerate(positions):
            canvas.grad[self._window(pos)] += grad[k]

    def _probe_on_grid(self):
        if self._pad == 1:
            return self.pr_g[None]
        grid = self.pr_g.new_zeros((self.Mp, *self._grid))
        grid[self._centre] = self.pr_g
        return grid[None]

    def _to_detector(self, psi_k):
        """Model far field on the padded grid to the detector frame. A crop does
        not commute with propagation, so the field is carried to the exit plane
        and cropped there."""
        if self._pad == 1 and self._chirp is None:
            return psi_k
        exit_wave = torch.fft.ifft2(psi_k, norm="ortho")[self._centre]
        if self._chirp is not None:
            exit_wave = self._chirp * exit_wave
        return torch.fft.fft2(exit_wave, norm="ortho")

    def _from_detector(self, correction):
        """The adjoint of _to_detector: the detector-frame residual back onto
        the padded grid the recursion runs on."""
        if self._pad == 1 and self._chirp is None:
            return correction
        frame = torch.fft.ifft2(correction, norm="ortho")
        if self._chirp is not None:
            frame = self._chirp.conj() * frame
        padded = frame.new_zeros(frame.shape[:-2] + self._grid)
        padded[self._centre] = frame
        return torch.fft.fft2(padded, norm="ortho")

    def _exit_to_detector(self, exit_wave):
        """Exit wave on the propagation grid to the detector-frame spectrum in
        one transform: the same result as _to_detector(fft2(w)), which would
        undo the transform first."""
        w = exit_wave[self._centre] if self._pad != 1 else exit_wave
        if self._chirp is not None:
            w = self._chirp * w
        return torch.fft.fft2(w, norm="ortho")

    def _exit_from_detector(self, correction):
        """The adjoint of _exit_to_detector: a detector-frame correction back to
        a real-space one on the propagation grid."""
        frame = torch.fft.ifft2(correction, norm="ortho")
        if self._chirp is not None:
            frame = self._chirp.conj() * frame
        if self._pad == 1:
            return frame
        padded = frame.new_zeros(frame.shape[:-2] + self._grid)
        padded[self._centre] = frame
        return padded

    def _crop(self, grid):
        """Back from the padded propagation grid to the detector frame."""
        return grid[self._centre] if self._pad != 1 else grid

    def _write_patch(self, pos, delta):
        self.ob_g[self._window(pos)] += delta

    def _write_probe(self, delta):
        self.pr_g += delta
        if self._support is not None:
            self.pr_g *= self._support
        if bool(self.p.fix_probe_int):
            self.pr_g *= (self._probe_power / self.pr_g.abs().square().sum()).sqrt()


    # ---- the sweep ------------------------------------------------------

    def _accumulate_error(self, pos, pattern):
        """Masked-mean amplitude l2 and Poisson-weighted intensity l2."""
        amp_dat = self.amp_dat[pos]
        I_dat = amp_dat**2
        weight = 1.0 if self.maskb is None else self.maskb[pos]
        norm = float(I_dat.numel()) if self.maskb is None else self.maskb[pos].sum().clamp_min(1)
        model = pattern[0]
        self._err += torch.stack(
            [
                (weight * (model.clamp_min(0).sqrt() - amp_dat) ** 2).sum() / norm,
                (weight * (model - I_dat) ** 2 / (I_dat + 1.0)).sum() / norm,
                model.new_zeros(()),
            ]
        ).double()

    def _position_order(self):
        if not self.p.shuffle:
            return np.arange(self.n_pos)
        rng = self._order_rng if self._order_rng is not None else np.random
        return rng.permutation(self.n_pos)

    def engine_iterate(self, num=1):
        err = None
        for _ in range(num):
            do_probe = self.curiter >= self.p.probe_update_start
            self._err = torch.zeros(3, dtype=torch.float64, device=self._device)
            order = self._position_order()
            if self._step == "adam":
                # one path at every batch size, so batch 1 cannot drift from it
                for start in range(0, len(order), self._batch):
                    self._update_batch(order[start:start + self._batch], do_probe)
            else:
                for pos in order:
                    self._update_view(int(pos), do_probe)
            err = (self._err / float(self.n_pos)).cpu().numpy()
            self._err_history.append(err.copy())
            self.curiter += 1
        return {"all": err}

    @torch.no_grad()
    def _update_batch(self, positions, do_probe):
        """One Adam step over several positions at once. Under no_grad: the
        gradients are the analytic adjoint written into .grad, so a tape over
        the sweep would hold memory for nothing. Subclasses implement
        _batch_gradients."""
        positions = np.asarray(positions)
        grad_O, grad_probe = self._batch_gradients(positions)
        self._scatter_grad_batch(self.ob_g, positions, grad_O)
        if do_probe:
            if self.pr_g.grad is None:
                self.pr_g.grad = torch.zeros_like(self.pr_g)
            self.pr_g.grad += grad_probe
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

    def _batch_gradients(self, positions):
        """(grad_O, grad_probe) for a batch, grad_O as (B, Nz, fny, fnx)
        in the detector frame and grad_probe summed over the batch."""
        raise NotImplementedError

    def engine_finalize(self):
        self._write_slices()
        if self._threads is not None:
            torch.set_num_threads(self._threads)
        return super().engine_finalize()

    def _write_slices(self):
        """Host-side mirror of the device store, then the product object in
        self.ob, and the per-slice output file."""
        from ptypy import io
        from ptypy.utils import Param

        oID = next(iter(self._object[0].S))
        pID = next(iter(self._probe[0].S))
        for i in range(int(self.p.number_of_slices)):
            self._object[i].S[oID].data[0][:] = self.ob_g[i].detach().cpu().numpy()
        self._probe[0].S[pID].data[:] = self.pr_g.detach().cpu().numpy()
        self.ob.fill(self._object[0])
        for i in range(1, int(self.p.number_of_slices)):
            self.ob *= self._object[i]

        info = Param()
        info.number_of_slices = self.p.number_of_slices
        info.slice_thickness = self.p.slice_thickness
        info.step = self._step
        info.slice_pad = self._pad
        self._result_info(info)
        info.objects = {
            ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()} for ob in self._object
        }
        info.probes = {
            pr.ID: {ID: S._to_dict() for ID, S in pr.storages.items()} for pr in self._probe
        }
        # h5write cannot serialise ptypy's own class types
        h5opt = io.h5options["UNSUPPORTED"]
        io.h5options["UNSUPPORTED"] = "ignore"
        logger.info("Saving to %s" % self.p.fslices)
        io.h5write(
            str(self.p.fslices),
            header={"description": "%s result details." % self.MODEL},
            content=info,
        )
        io.h5options["UNSUPPORTED"] = h5opt
