"""
GPU-resident, batched multislice ePIE (3PIE) for ptypy, backed by PyTorch.

    batch_size   = 1
    shuffle      = False
    object_denom = 'max'

it reproduces the CPU ThreePIE engine (ptypy stochastic.EPIE + multislice)
update for update, up to floating point -- verified, see below.


PRECONDITIONING THE SLICE UPDATES
---------------------------------
`object_denom` selects what the slice increment is divided by.

  'max'    ptypy's convention: the scalar max over the patch of
           sum_m |psi^inc_i|^2, one number per view. This is what the CPU
           engine does and it is the setting for reproducing it. At
           batch_size > 1 it divides everything by the most strongly
           illuminated pixel, so sparsely covered regions are starved.

  'local'  the per-pixel accumulated denominator, sum over the batch of
           sum_m |psi^inc_i|^2, with a floor of denom_reg times its maximum.
           Numerator and denominator are accumulated separately over the batch
           and divided once, which is the preconditioned mini-batch update.
"""

import os
from datetime import datetime

import numpy as np

from ptypy import io
from ptypy.core import geometry
from ptypy.custom import torch_helpers
from ptypy.custom.engine_hooks import EngineHooks
from ptypy.engines import register, stochastic
from ptypy.utils import Param
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@register()
class ThreePIETorch(stochastic.EPIE, EngineHooks):
    """
    Batched GPU multislice ePIE.

    Defaults:

    [name]
    default = ThreePIETorch
    type = str
    help = Engine identifier

    [number_of_slices]
    default = 2
    type = int
    help = The number of slices

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = Single float, or a list of length number_of_slices - 1.

    [slice_start_iteration]
    default = 0
    type = int, list, tuple
    help = Iteration at which each slice starts being used

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the slice data

    [init_slices]
    default = None
    type = str, NoneType
    help = Optional fslices .h5 to initialise the slice stack from
    doc = Path to a file in the fslices layout (content/objects/.../data),
          as written by this engine, FirstBornTorch or the simulator's truth
          dump. Each stored slice is loaded into the corresponding working
          slice; the file must hold number_of_slices slices on the same
          object canvas. Written for warm-starting multislice ePIE from a
          cheaper engine's result. The probe is NOT read from this file --
          seed it through the scan's illumination block (model = 'recon').

    [device]
    default = cuda
    type = str
    help = Torch device
    doc = 'cuda', 'cuda:0', or 'cpu'. Falls back to cpu with a warning if CUDA
          is unavailable.

    [batch_size]
    default = 1
    type = int
    help = Number of scan positions per update
    doc = 1 reproduces sequential 3PIE exactly. Larger values are faster but
          average the per-view increments instead of applying them in sequence.
          16-64 is usually a good trade for 512x512 frames.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

    [object_step]
    default = None
    type = float, NoneType
    help = Override for the object update step size
    doc = None inherits ptypy's value, which for EPIE is p.beta (NOT p.alpha --
          EPIEMixin sets _ob_b = beta, _pr_b = alpha, contrary to the parameter
          help strings in stochastic.py).

    [probe_step]
    default = None
    type = float, NoneType
    help = Override for the illumination update step size
    doc = None inherits ptypy's value, which for EPIE is p.alpha.

    [object_denom]
    default = max
    type = str
    help = Slice preconditioner: 'max', 'local' or 'unscattered'
    doc = 'max' is ptypy's scalar max over the patch of sum_m |psi^inc_i|^2 and
          reproduces the CPU engine. 'local' is the per-pixel accumulated
          denominator, numerator and denominator summed over the batch and
          divided once: the preconditioned mini-batch update, and the setting
          to use above batch_size = 1. 'unscattered' divides by sum_m |phi_i|^2
          with phi_i the probe propagated to slice i with no object in the way,
          which is the first-Born engine's denominator; use it in both engines
          when comparing the two forward models, so that the comparison is not
          confounded by the preconditioner.

    [denom_reg]
    default = 0.1
    type = float
    help = Floor on the 'local' and 'unscattered' denominators, as a fraction of the maximum
    doc = Ignored for 'max', which needs no regularisation. A focused probe has
          compact support and interior zeros, so the local denominator must be
          floored somewhere.

    [obj_smooth_std]
    default = None
    type = float, NoneType
    help = Gaussian smoothing (pixels) applied to each slice once per iteration
    doc = None disables it. Same name and intent as the projectional engines'
          obj_smooth_std: damp the spurious structure that accumulates where the
          illumination barely reaches. It is applied as a blend weighted by
          illumination coverage, not as a flat blur -- see obj_smooth_cover.

    [obj_smooth_cover]
    default = 0.1
    type = float
    help = Coverage at which the smoothing blend is half strength, as a fraction of the peak
    doc = DM and RAAR smooth only the inertia term, so the blur competes against
          sum|P|^2 and loses wherever the data has anything to say; that is why
          obj_smooth_std = 20 does not destroy their resolution. A flat blur on
          an ePIE object has no such competition and would. This reproduces the
          projectional behaviour: with cov the accumulated sum|P|^2 and
          c = obj_smooth_cover * cov.max(), the blend weight is c/(c + cov), so
          a pixel at the peak of the coverage is left nearly alone, one at c is
          half smoothed and the untouched border is smoothed fully.

          The reference is the PEAK and not the median over cov > 0: a focused
          probe has wings that reach numerical nonzero everywhere, so on real
          data every pixel of the canvas is lit to some degree and a median over
          them is a property of the wings rather than of the scan.

    [clip_object]
    default = None
    type = tuple, list, NoneType
    help = Clip slice amplitude into this interval, e.g. (0, 1)
    doc = ptypy's clip_object, applied per slice once per iteration. A
          non-amplifying sample has |O| <= 1, so (0, 1) is a physical statement
          rather than a tuning knob -- though note that N slices whose product
          must equal the true object can each legitimately exceed 1 when the
          gauge drifts, so clip the product's worth, not less.

    [apply_probe_support]
    default = False
    type = bool
    help = Apply the circular probe support from p.probe_support
    doc = The CPU multislice engine never calls support_constraint, so leave
          this False when comparing the two engines. BaseEngine's
          p.probe_support defaults to 0.7.

    [sync_interval]
    default = 1
    type = int
    help = Iterations between copying GPU state back into ptypy containers
    doc = Only needed for plotting and autosave. Set higher to save a little time.

    [plot_interval]
    default = 1
    type = int
    help = Iterations between diagnostic PNGs, 0 disables

    [shuffle]
    default = False
    type = bool
    help = Reshuffle the position order every iteration
    doc = Set False to walk views in the same sorted order as the CPU engine.

    [seed]
    default = 0
    type = int
    help = Seed for the position order; 0 leaves it free
    doc = Set the same non-zero seed for both runs of a paired comparison, and
          for both engines, so that nothing but the thing under test differs.
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.article = {
            "title": "Ptychographic transmission microscopy in three dimensions using a multi-slice approach",
            "author": "A. M. Maiden et al.",
            "journal": "J. Opt. Soc. Am. A",
            "volume": 29,
            "year": 2012,
            "page": 1606,
            "doi": "10.1364/JOSAA.29.001606",
            "comment": "The 3PIE reconstruction algorithm",
        }
        self.ptycho.citations.add_article(**self.article)

        # Local time on purpose: this names a run directory the user reads.
        # The class name is in it so a folder of runs identifies its solver.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        self.plot_progress_dir = f"iteration_progress_{type(self).__name__}_{stamp}"
        self._warned_inactive_last = False

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def engine_initialize(self):
        super().engine_initialize()

        if torch is None:
            raise ImportError("ThreePIETorch requires PyTorch")

        N = int(self.p.number_of_slices)

        # ---- device and dtype ----
        want = str(self.p.device)
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            want = "cpu"
        self.dev = torch.device(want)
        double = str(self.p.precision).lower().startswith("d")
        self.cdt = torch.complex128 if double else torch.complex64
        self.rdt = torch.float64 if double else torch.float32
        logger.info(
            "ThreePIETorch on {}, {} precision".format(
                self.dev, "double" if double else "single"
            )
        )

        # ---- step sizes (D3) ----
        # EPIEMixin: _ob_b = p.beta (object), _pr_b = p.alpha (probe).
        self.ob_step = float(
            self._ob_b if self.p.object_step is None else self.p.object_step
        )
        self.pr_step = float(
            self._pr_b if self.p.probe_step is None else self.p.probe_step
        )
        logger.info(f"object step {self.ob_step:.4g}, probe step {self.pr_step:.4g}")

        # ---- preconditioner (D1) ----
        self._denom = str(self.p.object_denom).lower()
        if self._denom not in ("max", "local", "unscattered"):
            raise ValueError("object_denom must be 'max', 'local' or 'unscattered'")
        self._reg = float(self.p.denom_reg)
        self._plot_int = max(0, int(self.p.plot_interval))
        logger.info(
            "slice preconditioner: {}{}".format(
                self._denom,
                ""
                if self._denom == "max"
                else f", floor {self._reg:.3g} of the maximum",
            )
        )
        if self._denom == "max" and int(self.p.batch_size) > 1:
            logger.warning(
                f"object_denom='max' at batch_size={int(self.p.batch_size)} divides "
                "every slice increment by the most strongly illuminated pixel in the "
                "batch, which starves sparsely covered regions. Use 'local'."
            )

        # ptypy's global object norm would need the product-of-slices object,
        # which is a different quantity per slice. Refuse rather than guess.
        if getattr(self, "_object_norm_is_global", False):
            raise NotImplementedError(
                "object_norm_is_global is not supported by ThreePIETorch; set it False"
            )

        if int(self.p.batch_size) > 1:
            logger.warning(
                f"batch_size={int(self.p.batch_size)}: per-view increments are averaged, "
                "not applied sequentially. Use batch_size=1 for exact 3PIE."
            )

        # ---- slice_start_iteration ----
        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = np.ones(N) * self.p.slice_start_iteration
        self.p.slice_start_iteration = np.asarray(self.p.slice_start_iteration)
        if len(self.p.slice_start_iteration) != N:
            raise ValueError("slice_start_iteration must have length number_of_slices")

        # ---- ptypy containers, kept for saving and plotting ----
        self._object = torch_helpers.slice_containers(self.ob, N)

        if len(self.ob.storages) != 1:
            raise NotImplementedError("ThreePIETorch supports a single object storage")
        self._ob_storage = next(iter(self.ob.storages.values()))
        if self._ob_storage.data.shape[0] != 1:
            raise NotImplementedError("ThreePIETorch supports a single object mode")

        if len(self.pr.storages) != 1:
            raise NotImplementedError("ThreePIETorch supports a single probe storage")
        self._pr_storage = next(iter(self.pr.storages.values()))

        self.M, self.fny, self.fnx = self._pr_storage.data.shape

        # ---- GPU state ----
        ob0 = self._ob_storage.data[0]
        self.ob_g = torch.stack(
            [
                torch.as_tensor(
                    np.ascontiguousarray(ob0), dtype=self.cdt, device=self.dev
                )
                for _ in range(N)
            ]
        )  # (N, obny, obnx)
        # .clone() is load-bearing: torch.as_tensor over a contiguous array of
        # the matching dtype shares its memory, so without it pr_g is a *view*
        # of the ptypy probe storage. Anything that writes P.probe -- another
        # engine in the same run, a reload, a plotting helper -- then silently
        # rewrites this engine's working probe, and _sync_to_ptypy becomes a
        # self-assignment that hides it. ob_g above is safe because
        # torch.stack always copies.
        self.pr_g = torch.as_tensor(
            np.ascontiguousarray(self._pr_storage.data), dtype=self.cdt, device=self.dev
        ).clone()  # (M, fny, fnx)

        if self.p.init_slices:
            self._load_init_slices(str(self.p.init_slices))

        self._build_propagators()
        self._collect_views()

        # ---- optional circular probe support (D8, opt-in) ----
        self.pr_support = None
        if bool(self.p.apply_probe_support):
            ps = getattr(self.p, "probe_support", None)
            self.pr_support = torch_helpers.circular_support(
                ps, (self.fny, self.fnx), self.rdt, self.dev
            )
            if self.pr_support is not None:
                logger.info(f"probe support: circular, radius fraction {float(ps):.3f}")

        # sum_m |phi_i|^2 for object_denom='unscattered'; rebuilt whenever the
        # probe moves
        self._phi_cache = None

        # ---- object regularisation ----
        self._smooth_std = (
            None if self.p.obj_smooth_std is None else float(self.p.obj_smooth_std)
        )
        self._smooth_w = None  # built lazily, once the probe has settled
        self._clip = None if self.p.clip_object is None else tuple(self.p.clip_object)
        if self._smooth_std:
            logger.info(
                f"object smoothing: std {self._smooth_std:.3g} px, half strength at "
                f"{float(self.p.obj_smooth_cover):.3g} of the peak coverage"
            )
        if self._clip is not None:
            logger.info(f"object amplitude clipped to {self._clip}")

        # ---- reproducibility (D10) ----
        seed = int(self.p.seed)
        self._order_gen = None
        if seed:
            self._order_gen = torch.Generator(device=self.dev)
            self._order_gen.manual_seed(seed + 1)
            logger.info(f"seed = {seed}: the position order is reproducible")

        self._err_history = []
        os.makedirs(self.plot_progress_dir, exist_ok=True)

        self.post_initialize()

    def _load_init_slices(self, path):
        """Replace the flat slice start with a stack read from an fslices .h5.

        Reads either naming scheme (the engines' Cobj_o_<j>, the truth dump's
        obj_<j>), ordered by the trailing slice index, exactly as the plotting
        and scoring tools read these files.
        """
        import re

        import h5py

        N = int(self.p.number_of_slices)
        with h5py.File(path, "r") as f:
            objs = f["content/objects"]
            keys = sorted(
                objs.keys(), key=lambda k: int(re.search(r"(\d+)$", k).group(1))
            )
            slices = []
            for k in keys:
                g = objs[k]
                sk = sorted(g.keys())[0]
                slices.append(np.asarray(g[sk]["data"])[0])
        if len(slices) != N:
            raise ValueError(
                "init_slices %s holds %d slices, number_of_slices is %d"
                % (path, len(slices), N)
            )
        if slices[0].shape != tuple(self.ob_g.shape[-2:]):
            raise ValueError(
                "init_slices canvas %s does not match the object canvas %s"
                % (slices[0].shape, tuple(self.ob_g.shape[-2:]))
            )
        self.ob_g = torch.stack(
            [
                torch.as_tensor(
                    np.ascontiguousarray(s), dtype=self.cdt, device=self.dev
                )
                for s in slices
            ]
        )
        logger.info("slice stack initialised from %s" % path)

    def _build_propagators(self):
        """Extract the slice-to-slice near-field and the far-field propagators."""
        N = int(self.p.number_of_slices)
        scan = next(iter(self.ptycho.model.scans.values()))
        geom = scan.geometries[0]

        g = Param()
        g.energy = geom.energy
        g.psize = geom.resolution
        g.shape = geom.shape
        g.propagation = "nearfield"

        if type(self.p.slice_thickness) in [list, tuple]:
            if len(self.p.slice_thickness) != N - 1:
                raise ValueError(
                    "slice_thickness list must have length number_of_slices - 1"
                )
            thicknesses = list(self.p.slice_thickness)
        else:
            thicknesses = [self.p.slice_thickness] * (N - 1)
        shape = tuple(geom.shape)
        self._fw_spec, self._bw_spec = [], []
        for i, t in enumerate(thicknesses):
            g.distance = t
            G = geometry.Geo(owner=None, pars=g)
            self._fw_spec.append(
                torch_helpers.extract_operator(
                    G.propagator.fw, shape, f"nearfield fw[{i}]"
                )
            )
            self._bw_spec.append(
                torch_helpers.extract_operator(
                    G.propagator.bw, shape, f"nearfield bw[{i}]"
                )
            )

        pod = None
        for view in self.di.views.values():
            if view.active:
                pod = next(iter(view.pods.values()))
                break
        if pod is None:
            raise RuntimeError("no active diffraction views")

        self._ff_fw = torch_helpers.extract_operator(pod.fw, shape, "farfield fw")
        self._ff_bw = torch_helpers.extract_operator(pod.bw, shape, "farfield bw")

        def to_g(spec):
            return torch_helpers.to_torch(spec, self.cdt, self.dev)

        self._fw_g = [to_g(s) for s in self._fw_spec]
        self._bw_g = [to_g(s) for s in self._bw_spec]
        self._ff_fw_g = to_g(self._ff_fw)
        self._ff_bw_g = to_g(self._ff_bw)

    @staticmethod
    def _apply(spec, w):
        return torch_helpers.apply_torch(spec, w)

    def _unscattered(self):
        """sum_m |phi_i|^2, the first-Born denominator: the probe propagated to
        each slice with no object in the way. Cached until the probe moves."""
        if self._phi_cache is None:
            N = int(self.p.number_of_slices)
            ph = torch.empty(
                (N, self.M, self.fny, self.fnx), dtype=self.cdt, device=self.dev
            )
            ph[0] = self.pr_g
            for i in range(1, N):
                ph[i] = self._apply(self._fw_g[i - 1], ph[i - 1])
            self._phi_cache = (ph.abs() ** 2).sum(dim=1)  # (N, ny, nx)
        return self._phi_cache

    def _collect_views(self):
        """Pull positions, intensities and masks onto the GPU once."""
        names = sorted([n for n, v in self.di.views.items() if v.active])
        if not names:
            raise RuntimeError("no active diffraction views")

        rows, cols, data, masks, layers = [], [], [], [], None

        for n in names:
            view = self.di.views[n]
            pods = list(view.pods.values())

            ov = pods[0].ob_view
            try:
                lo = np.asarray(ov.dlow, dtype=np.int64)
            except AttributeError:
                raise RuntimeError(
                    "This ptypy version's View has no .dlow; adapt _collect_views "
                    "to however object patches are indexed here."
                )
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))

            lay = [int(p.pr_view.dlayer) for p in pods]
            if layers is None:
                layers = lay
            elif lay != layers:
                raise NotImplementedError("inconsistent probe-mode layout across views")

            data.append(np.asarray(view.data, dtype=np.float64))
            m = pods[0].mask
            masks.append(
                np.ones_like(data[-1]) if m is None else np.asarray(m, dtype=np.float64)
            )

        self.n_pos = len(names)
        self.view_names = names
        self.rows_g = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols_g = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.I_g = torch.as_tensor(np.stack(data), dtype=self.rdt, device=self.dev)
        self.mask_g = torch.as_tensor(np.stack(masks), dtype=self.rdt, device=self.dev)
        self.amp_g = torch.sqrt(self.I_g.clamp_min(0))

        if list(layers) != list(range(self.M)):
            logger.warning(
                f"probe layers {layers} do not match 0..{self.M - 1}; check mode ordering"
            )
        logger.info(f"collected {self.n_pos} scan positions, {self.M} probe mode(s)")

        # Illuminated region.  Any statistic taken over a slice is meaningless
        # outside it: most of the object array is never touched and stays at
        # exactly 1.  Read from outside the engine, by diagnostics.
        cov = torch.zeros(self.ob_g[0].shape, dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            ones = torch.ones(
                (idx.numel(), self.fny, self.fnx), dtype=self.rdt, device=self.dev
            )
            self._scatter_add(cov, idx, ones)
        self.cov = cov > 0

    # ------------------------------------------------------------------ #
    #  Patch gather / scatter                                             #
    # ------------------------------------------------------------------ #
    def _gather(self, arr2d, idx):
        r = self.rows_g[idx]  # (B, fny)
        c = self.cols_g[idx]  # (B, fnx)
        return arr2d[r[:, :, None], c[:, None, :]]

    def _coverage(self):
        """Accumulated sum_m |P|^2 over every scan position, on the object grid.

        This is the denominator the projectional engines weigh their smoothed
        inertia term against. Built once: the probe changes during a run, but
        only its overall scale matters here and the weight is normalised by its
        own median, so a rebuild would not move the blend.
        """
        cov = torch.zeros(self.ob_g.shape[-2:], dtype=self.rdt, device=self.dev)
        w = (self.pr_g.abs() ** 2).sum(0)  # over probe modes -> (fny, fnx)
        # _scatter_add materialises B*fny*fnx int64 indices twice, so this walks
        # in update-sized chunks rather than one pass: at 512x512 frames a batch
        # of 256 would ask for a gigabyte of indices to build a 6 MB map.
        step = max(1, int(self.p.batch_size))
        for s in range(0, self.n_pos, step):
            idx = torch.arange(
                s, min(s + step, self.n_pos), device=self.dev, dtype=torch.long
            )
            self._scatter_add(cov, idx, w.expand(idx.numel(), -1, -1))
        return cov

    def _smooth_weight(self):
        """Per-pixel blend weight for the smoothing regulariser, in [0, 1]."""
        cov = self._coverage()
        peak = cov.max()
        if peak <= 0:  # no illumination anywhere: smooth uniformly
            return torch.ones_like(cov)
        c = float(self.p.obj_smooth_cover) * peak
        w = c / (c + cov)
        core = cov > 0.5 * peak
        faint = cov < 0.01 * peak
        logger.info(
            "smoothing blend: weight {:.3f} in the illuminated core ({:.1f} % of "
            "the canvas), {:.3f} out in the faint surround ({:.1f} %)".format(
                float(w[core].mean()),
                100.0 * float(core.sum()) / cov.numel(),
                float(w[faint].mean()) if bool(faint.any()) else float("nan"),
                100.0 * float(faint.sum()) / cov.numel(),
            )
        )
        return w

    def _regularise_object(self):
        """Smooth and clip the slices. Called once per iteration, after the sweep.

        Order matters: smooth first, then clip, so the clip is the last word and
        the object handed to the next iteration honours it exactly.
        """
        if self._smooth_std:
            if self._smooth_w is None:
                self._smooth_w = self._smooth_weight()
            blurred = torch_helpers.complex_gaussian_filter(self.ob_g, self._smooth_std)
            self.ob_g = self.ob_g + self._smooth_w * (blurred - self.ob_g)
        if self._clip is not None:
            self.ob_g = torch_helpers.clip_amplitude(self.ob_g, self._clip)

    def _scatter_add(self, dst, idx, vals):
        B = idx.numel()
        r = self.rows_g[idx][:, :, None].expand(B, self.fny, self.fnx)
        c = self.cols_g[idx][:, None, :].expand(B, self.fny, self.fnx)
        dst.index_put_(
            (r.reshape(-1), c.reshape(-1)), vals.reshape(-1), accumulate=True
        )

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #
    def engine_iterate(self, num=1):
        B = int(self.p.batch_size)

        for _ in range(num):
            self.pre_iteration()

            if self.p.shuffle:
                order = (
                    torch.randperm(self.n_pos, device=self.dev)
                    if self._order_gen is None
                    else torch.randperm(
                        self.n_pos, generator=self._order_gen, device=self.dev
                    )
                )
            else:
                # same order the CPU engine walks with a sorted, unshuffled list
                order = torch.arange(self.n_pos, device=self.dev)

            error_dct = {}
            for s in range(0, self.n_pos, B):
                idx = order[s : s + B]
                error_dct.update(self._batch_update(idx))
            self._err_history.append(
                np.mean(np.stack(list(error_dct.values())), axis=0))

            self._regularise_object()

            self.curiter += 1
            if self.curiter % max(1, int(self.p.sync_interval)) == 0:
                self._sync_to_ptypy()

            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()

        return error_dct

    def _batch_update(self, idx):
        N = int(self.p.number_of_slices)
        B = idx.numel()
        M = self.M
        eps = torch.finfo(self.rdt).eps
        use_max = self._denom == "max"

        active = [self.curiter >= self.p.slice_start_iteration[i] for i in range(N)]
        # D4: probe_update_start freezes the probe, i.e. psi_0. It does NOT
        # reach the psi corrections at the slices above -- see the long note
        # at the `do_pr or i > 0` in the backward loop for why gating those
        # too freezes every slice but the last. ThreePIE.psi_update in
        # threepie.py draws the same line, so the two engines still agree.
        do_pr = self.curiter >= self.p.probe_update_start

        if not active[-1] and not self._warned_inactive_last:
            logger.warning(
                "final slice inactive: this engine excludes it from the Fourier "
                "step (exit = psi); the CPU engine leaves pod.object set and "
                "does not. Results will differ until the slice switches on."
            )
            self._warned_inactive_last = True

        # ---------------- forward ----------------
        ob_p = [self._gather(self.ob_g[i], idx) for i in range(N)]  # (B, ny, nx)
        psi = self.pr_g[None].expand(B, M, self.fny, self.fnx)  # incident at slice 0

        psis, exits = [], []
        for i in range(N):
            psis.append(psi)
            e = psi * ob_p[i][:, None] if active[i] else psi
            exits.append(e)
            if i < N - 1:
                psi = self._apply(self._fw_g[i], e)

        # ---------------- Fourier modulus projection ----------------
        F = self._apply(self._ff_fw_g, exits[-1])  # (B, M, ny, nx)
        Imod = (F.real**2 + F.imag**2).sum(dim=1)  # (B, ny, nx)
        amod = torch.sqrt(Imod.clamp_min(0))

        am = self.amp_g[idx]
        mk = self.mask_g[idx]
        I_meas = self.I_g[idx]

        # ptypy's absolute 1e-10 floor, not finfo eps
        fm = (1.0 - mk) + mk * am / (amod + 1e-10)
        e_proj = self._apply(self._ff_bw_g, F * fm[:, None].to(self.cdt))

        # ---------------- per-view errors (D9) ----------------
        npix = float(self.fny * self.fnx)
        msum = mk.sum(dim=(-2, -1)).clamp_min(1.0)
        d = amod - am
        err_fmag = (mk * d * d).sum(dim=(-2, -1)) / msum
        err_exit = ((e_proj - exits[-1]).abs() ** 2).mean(dim=(-2, -1)).sum(dim=1)
        if self.p.compute_log_likelihood:
            err_phot = (mk * (Imod - I_meas) ** 2 / (I_meas + 1.0)).sum(
                dim=(-2, -1)
            ) / npix
        else:
            err_phot = torch.zeros_like(err_fmag)

        # ---------------- backward ----------------
        # 'max': the per-view increment is normalised exactly as ptypy does and
        #        the increments are averaged over the batch.
        # 'local' / 'unscattered': numerator and denominator are accumulated
        #        separately over the batch and divided once, which is the
        #        preconditioned mini-batch update.
        num_o = [torch.zeros_like(self.ob_g[0]) for _ in range(N)]
        den_o = [
            torch.zeros(self.ob_g[0].shape, dtype=self.rdt, device=self.dev)
            for _ in range(N)
        ]
        ones_patch = torch.ones(
            (B, self.fny, self.fnx), dtype=self.rdt, device=self.dev
        )
        phi2 = self._unscattered() if self._denom == "unscattered" else None

        num_pr = torch.zeros_like(self.pr_g)
        den_pr = torch.zeros((self.fny, self.fnx), dtype=self.rdt, device=self.dev)
        pr_replace = None

        target = e_proj
        for i in range(N - 1, -1, -1):
            delta = target - exits[i]  # (B, M, ny, nx)

            if active[i]:
                # local intensity of the field incident on this slice: the
                # multislice analogue of |phi_i|^2, carrying every slice above
                if phi2 is None:
                    w = (psis[i].abs() ** 2).sum(dim=1)  # (B, ny, nx)
                else:
                    w = phi2[i].expand(B, self.fny, self.fnx)
                grad = (psis[i].conj() * delta).sum(dim=1)  # (B, ny, nx)

                if use_max:
                    pn = w.amax(dim=(-2, -1))[:, None, None]
                    dob = self.ob_step * grad / pn.clamp_min(eps).to(self.cdt)
                    self._scatter_add(num_o[i], idx, dob)
                    self._scatter_add(den_o[i], idx, ones_patch)
                else:
                    self._scatter_add(num_o[i], idx, grad)
                    self._scatter_add(den_o[i], idx, w)
                    # per-view estimate of the same update, for the psi
                    # correction below; identical to the batch update at B = 1
                    dob = (
                        self.ob_step
                        * grad
                        / (w + self._reg * w.amax(dim=(-2, -1))[:, None, None])
                        .clamp_min(eps)
                        .to(self.cdt)
                    )

                # --- illumination update (D2, D5): uses the UPDATED object
                o_new = ob_p[i] + dob  # (B, ny, nx)
                #
                # `do_pr or i > 0`: probe_update_start freezes the PROBE,
                # which is psi_0 and nothing else. Maiden 2012 applies the same
                # correction to the wave incident on every slice, and at i > 0
                # that correction is the only thing that carries the Fourier
                # residual back up the stack: leave psi_i alone and
                # bw(psi_i) is exactly exits[i-1] again, so delta at slice
                # i-1 is identically zero and every slice above the last
                # freezes with it -- not slowly, but to propagator round-off
                # (measured: 1.2e-8 after 20 views against 1.07 for the last
                # slice). More iterations do not help; the gradient is zero,
                # not small. Gating only at i == 0 still freezes the probe,
                # because psi_0 is where the probe increment comes from.
                if do_pr or i > 0:
                    if use_max:
                        on = (M * o_new.abs() ** 2).amax(dim=(-2, -1))[
                            :, None, None, None
                        ]
                        dpsi = (
                            self.pr_step
                            * o_new.conj()[:, None]
                            * delta
                            / on.clamp_min(eps).to(self.cdt)
                        )
                        if i == 0:
                            num_pr = num_pr + dpsi.sum(dim=0)
                    else:
                        ow = M * o_new.abs() ** 2  # (B, ny, nx)
                        dpsi = (
                            self.pr_step
                            * o_new.conj()[:, None]
                            * delta
                            / (ow + self._reg * ow.amax(dim=(-2, -1))[:, None, None])
                            .clamp_min(eps)[:, None]
                            .to(self.cdt)
                        )
                        if i == 0:
                            num_pr = num_pr + (o_new.conj()[:, None] * delta).sum(dim=0)
                            den_pr = den_pr + ow.sum(dim=0)
                    psi_new = psis[i] + dpsi
                else:
                    psi_new = psis[i]
            else:
                # D6: inactive slice -> the illumination is replaced outright
                psi_new = target
                if i == 0:
                    pr_replace = target.mean(dim=0)

            if i > 0:
                target = self._apply(self._bw_g[i - 1], psi_new)

        # ---------------- apply updates ----------------
        for i in range(N):
            if not active[i]:
                continue
            if use_max:
                self.ob_g[i] += num_o[i] / den_o[i].clamp_min(1.0).to(self.cdt)
            else:
                dn = den_o[i] + self._reg * den_o[i].amax()
                self.ob_g[i] += self.ob_step * num_o[i] / dn.clamp_min(eps).to(self.cdt)

        if pr_replace is not None:
            self.pr_g = pr_replace.to(self.cdt)
            self._phi_cache = None
        elif do_pr:
            if use_max:
                self.pr_g += num_pr / float(B)
            else:
                dn = den_pr + self._reg * den_pr.amax()
                self.pr_g += self.pr_step * num_pr / dn.clamp_min(eps).to(self.cdt)
            self._phi_cache = None

        if self.pr_support is not None:
            self.pr_g = self.pr_g * self.pr_support
            self._phi_cache = None

        # ---------------- pack per-view errors ----------------
        ii = idx.detach().cpu().numpy()
        ef = err_fmag.detach().cpu().numpy()
        ep = err_phot.detach().cpu().numpy()
        ee = err_exit.detach().cpu().numpy()
        return {
            self.view_names[int(k)]: np.array([ef[j], ep[j], ee[j]])
            for j, k in enumerate(ii)
        }

    # ------------------------------------------------------------------ #
    #  Sync and finalize                                                  #
    # ------------------------------------------------------------------ #
    def _sync_to_ptypy(self):
        """Copy GPU state into the ptypy containers for plotting and autosave."""
        N = int(self.p.number_of_slices)
        for i in range(N):
            arr = self.ob_g[i].detach().cpu().numpy()
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._pr_storage.data[:] = self.pr_g.detach().cpu().numpy()

        self.ob.fill(self._object[0])
        for i in range(1, N):
            self.ob *= self._object[i]

    def engine_finalize(self):
        self._sync_to_ptypy()

        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness
        slices_info.slice_start_iteration = self.p.slice_start_iteration
        slices_info.batch_size = self.p.batch_size
        slices_info.object_step = self.ob_step
        slices_info.probe_step = self.pr_step
        slices_info.object_denom = self._denom
        slices_info.denom_reg = self._reg
        slices_info.seed = int(self.p.seed)
        slices_info.update(self.extra_slices_info())

        slices_info.objects = {
            ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
            for ob in self._object
        }

        header = {"description": "multi-slice result details (torch engine)."}
        h5opt = io.h5options["UNSUPPORTED"]
        io.h5options["UNSUPPORTED"] = "ignore"
        logger.info(f"Saving to {self.p.fslices}")
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options["UNSUPPORTED"] = h5opt

        return super().engine_finalize()

    def _plot_progress(self):
        torch_helpers.plot_progress(
            self.pr_g,
            self.ob_g,
            self.curiter,
            self.plot_progress_dir,
            psize=getattr(self.p, "psize", (1e-6, 1e-6)),
            engine_name=type(self).__name__,
        )
        torch_helpers.plot_error_history(
            self._err_history,
            self.plot_progress_dir,
            engine_name=type(self).__name__,
        )
