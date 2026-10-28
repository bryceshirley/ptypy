"""
Shared engine infrastructure for the inter-slice Born family.

Two layers live here. The slice containers and ISBEngineInfra --
progress-plot state, run recording (entry script copy, git commit) and the
terminal log, with the Defaults for the shared plotting and logging
options -- are backend-neutral. CupyISBEngineInfra puts that mixin on
cupy: device/dtype setup, the BaseEngine probe-support convention,
far-field operators on the device, the figure rendering and
the ISS/ISB family plumbing -- object/probe arrays
with warm starts, the branch propagators, view collection with flat-index
gather/scatter, and the slice write-out. The figures live in
plotting.py; the far-field pair comes from ptypy's accelerate
PropagationKernel, allocated on the engine stream.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime

import numpy as np

from ptypy import io
from ptypy.core import geometry
from ptypy.custom.multislice_utils import slice_bandlimit
from ptypy.utils import Param
from ptypy.utils.verbose import logger

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from ptypy.utils import verbose

from .plotting import _HostView, plot_error_history, plot_progress

# Buffer ptypy's log records from import time, so a run's log file can
# start with the Ptycho init output that precedes engine_initialize.
# flushLevel above CRITICAL: nothing flushes until an engine attaches a
# file; at capacity the stdlib handler drops the buffer, which caps it.
_log_memory_handler = logging.handlers.MemoryHandler(
    200000, flushLevel=logging.CRITICAL + 1)
logger.addHandler(_log_memory_handler)
_log_file_handler = None


def slice_containers(ob, n, tag="o"):
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
        ID, k = f"{ob.ID}_{tag}_{i}", 1
        while ID in pool:
            ID = f"{ob.ID}_{tag}{k}_{i}"
            k += 1
        out.append(ob.copy(ID))
    return out

class ISBEngineInfra:
    """Backend-neutral engine infrastructure for the inter-slice Born
    custom engines: the progress-plot state, run recording (entry script
    copy, git commit), the terminal log and its Defaults below, plus the
    whole ISS/ISB family plumbing -- object/probe arrays with warm
    starts, the branch propagators, view collection with flat-index
    gather/scatter, probe support, far-field extraction, plotting and the
    slice write-out. Everything stateful is prepared on the host in numpy
    and moved once, so a backend reduces to the hooks: _setup_device
    (sets dev/cdt/rdt and the index/bool dtypes idt/bdt), _dev (host
    array -> device array), _to_host (device array -> numpy), _scatter
    (accumulating flat-index scatter), _build_farfield (the device
    far-field pair) and
    _setup_order_source (seeded position-order source); _sync is a no-op
    unless the backend runs on its own stream. CupyISBEngineInfra below
    is exactly that hook set.

    Defaults:

    [plot_interval]
    default = 0
    type = int
    help = Iterations between diagnostic PNGs, 0 disables

    [plot_dir]
    default = None
    type = str
    help = Directory for the progress figures and terminal log; None
      auto-creates a timestamped iteration_progress_<engine>_<stamp>/ in
      the CURRENT WORKING directory (not the run's output directory)

    [plot_error_channels]
    default = fourier,photon,exit
    type = str
    help = Which error channels the convergence PNG draws
    doc = Comma-separated subset of fourier, photon, exit. The error
          HISTORY always records all three; this only selects what
          error_history.png shows.

    [plot_zoom_center]
    default = None
    type = tuple
    help = Fractional (x, y) centre of the object zoom box in the progress
      figures, origin lower-left; None means the panel centre

    [plot_phase_zero_top]
    default = True
    type = bool
    help = Shift each slice's displayed phase so the background sits at 0 and
      the colour bar tops out there (phase purely negative, the x-ray paper
      figure convention)

    [plot_phase_clim]
    default = None
    type = tuple
    help = Explicit (vmin, vmax) for the phase panels; None uses robust
      percentiles

    [plot_scalebar]
    default = None
    type = tuple
    help = Red scale bars (full_um, zoom_um) on the object panels, e.g.
      (2.0, 0.5); None draws none

    [slice_bandlimit]
    default = True
    type = bool
    help = Apply angular-spectrum anti-aliasing support for slice propagation
    doc = The same mask the ThreePIE engines apply
          (multislice_utils.slice_bandlimit), here on each cumulative
          branch propagator H_z.

    [terminal_log]
    default = auto
    type = str
    help = Write this run's ptypy log into a file
    doc = 'auto' writes terminal.log inside the progress directory when
          plotting is on, else <fslices stem>_terminal.log; an explicit
          path is used as given; '' disables. Standard-library logging on
          the "ptypy" logger: the file starts with the buffered Ptycho
          init output and carries every logger record at the active
          verbosity; a run script's own print() output is not captured.
    """

    # {"probe": ..., "phase": ...} colormaps for the progress figures;
    # subclasses may override to restyle without touching the helper.
    PLOT_CMAPS = {"probe": "inferno", "phase": "Greys_r"}

    def _stamp_plot_dir(self):
        explicit = self.p.get("plot_dir", None)
        if explicit:
            self.plot_progress_dir = str(explicit)
            return
        # Local time on purpose: this names a run directory the user reads.
        # The class name is in it so a folder of runs identifies its solver
        # without the log (type(self) so subclasses label themselves).
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        self.plot_progress_dir = \
            f"iteration_progress_{type(self).__name__}_{stamp}"

    def _setup_plot_state(self):
        """Error history, plot cadence, progress dir, terminal log, and
        the reproducibility record (run script copy, git commit)."""
        self._err_history = []
        self._plot_int = max(0, int(self.p.plot_interval))
        # Only when plotting is on: with plot_interval = 0 nothing is ever
        # written there and the empty timestamped directory is pure litter.
        if self._plot_int > 0:
            os.makedirs(self.plot_progress_dir, exist_ok=True)
            self._save_run_script()
        self._start_terminal_log()
        self._log_git_commit()

    def _save_run_script(self):
        """Copy the entry script into the progress directory, so a run
        folder records exactly what produced it."""
        import shutil
        src = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        if not src.endswith(".py") or not os.path.isfile(src):
            return
        try:
            shutil.copy2(src, os.path.join(
                self.plot_progress_dir,
                "run_script_" + os.path.basename(src)))
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
            # -uno: the tracked files decide dirtiness, untracked scans of
            # a large data tree would stall the engine start
            dirty = subprocess.run(
                ["git", "-C", here, "status", "--porcelain", "-uno"],
                capture_output=True, text=True, timeout=5)
            flag = " (dirty)" if dirty.stdout.strip() else ""
            logger.info("ptypy git commit: %s%s"
                        % (head.stdout.strip(), flag))
        except (OSError, subprocess.SubprocessError):
            pass

    def _start_terminal_log(self):
        """Attach the run's log file to ptypy's own logger.

        Standard-library logging on the existing "ptypy" logger: a
        buffering MemoryHandler has been collecting records since the
        engine module was imported, so the file starts with the full
        Ptycho init output (data import, POD creation, probe/object
        init) even though the path is only known now. The buffer is
        flushed into a FileHandler with ptypy's FILE_FORMAT and replaced
        by it. p.terminal_log gives the path; 'auto' resolves into the
        progress directory when plotting is on, else next to fslices;
        '' disables. Idempotent: one log file per process, a second
        engine returns immediately. Only logger records are captured --
        a run script's own print() output is not (scripts that need a
        verbatim terminal mirror keep their own tee, which is unaffected
        because the console handler still writes to sys.stdout).
        """
        global _log_file_handler
        spec = str(self.p.get("terminal_log", "auto") or "")
        if not spec:
            return
        if _log_file_handler is not None:
            return              # an earlier engine already logs this process
        if spec.lower() == "auto":
            if self._plot_int > 0:
                path = os.path.join(self.plot_progress_dir, "terminal.log")
            else:
                path = os.path.splitext(str(self.p.fslices))[0] \
                    + "_terminal.log"
        else:
            path = spec
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = logging.FileHandler(path, mode="a")
        fh.setFormatter(verbose.CustomFormatter(verbose.FILE_FORMAT))
        if _log_memory_handler is not None:
            _log_memory_handler.setTarget(fh)
            _log_memory_handler.flush()
            logger.removeHandler(_log_memory_handler)
        logger.addHandler(fh)
        _log_file_handler = fh
        logger.info("run log: %s" % path)

    def _display_object(self):
        """Slice transmissions for plotting; the tangent subclasses, which
        store log O, override this with the exponential."""
        return self.ob_g


    def _setup_infra(self):
        self._setup_device()

        N = self.p.number_of_slices
        self._obs = next(iter(self.ob.storages.values()))
        self._prs = next(iter(self.pr.storages.values()))
        self.M, self.fny, self.fnx = self._prs.data.shape

        # Initialize object slices and probe.
        #
        # The start is 1 + 1e-3 * uniform, symmetry-broken so the slices are
        # not degenerate.  It is drawn from self.p.seed rather than from OS
        # entropy: a paired comparison whose two runs begin from different
        # objects has a twin-run difference that is not zero at iteration 0,
        # which is usually the quantity being measured.  seed = 0 leaves the
        # draw free, which is what the original engine did.
        self._object = slice_containers(self.ob, N)
        seed = int(self.p.seed)
        ob0 = np.ones_like(self._obs.data[0])
        rng = np.random.default_rng(seed if seed else None)
        self.ob_g = self._dev(
            np.stack([
                ob0
                + 1e-3
                * (
                    (rng.random(ob0.shape) - 0.5)
                    + 1j * (rng.random(ob0.shape) - 0.5)
                )
                for _ in range(N)
            ]),
            self.cdt,
        )
        logger.info(
            "object start: 1 + 1e-3 noise, seed %s"
            % (seed if seed else "free (not reproducible)")
        )
        self.pr_g = self._dev(
            np.ascontiguousarray(self._prs.data), self.cdt
        )

        # Warm start: overwrite the fresh object/probe init with a previous
        # reconstruction so a run can continue where another stopped.
        ws_ob = getattr(self.p, "warm_start_h5", None)
        if ws_ob:
            import h5py as _h5
            with _h5.File(ws_ob, "r") as _f:
                objs = _f["content/objects"]
                keys = sorted(objs.keys(), key=lambda s: int(s.split("_")[-1]))
                if len(keys) != N:
                    raise ValueError(
                        "warm_start_h5 has %d slices, engine wants %d"
                        % (len(keys), N))
                for i, k in enumerate(keys):
                    arr = objs[k]["Sscan00G00"]["data"][()][0]
                    self.ob_g[i] = self._dev(arr, self.cdt)
            logger.info("warm-start: loaded %d object slices from %s"
                        % (N, ws_ob))
        ws_pr = getattr(self.p, "warm_start_ptyr", None)
        if ws_pr:
            import h5py as _h5
            with _h5.File(ws_pr, "r") as _f:
                g = _f["content/probe"]
                pk = list(g.keys())[0]
                Pw = g[pk]["data"][()]
            if Pw.shape != tuple(self.pr_g.shape):
                raise ValueError("warm_start_ptyr probe shape %s != %s"
                                 % (Pw.shape, tuple(self.pr_g.shape)))
            self.pr_g = self._dev(Pw, self.cdt)
            logger.info("warm-start: loaded probe %s from %s"
                        % (str(Pw.shape), ws_pr))

        self._order_gen = None
        self._order_rng = None
        if seed:
            self._setup_order_source(seed)
            logger.info(f"seed = {seed}: the position order is reproducible")
        self.ony, self.onx = self.ob_g.shape[-2:]

        # Precompute Free-Space Propagators (H_z and H_{-z}) on the host
        # through ptypy's own nearfield geometry (the identical
        # angular-spectrum kernel exp(2j pi (z/lam) (sqrt(1-a2)-1)), built
        # here per CUMULATIVE depth: the series refers every branch to the
        # entrance plane), with the ThreePIE engines' alias-free band
        # limit, then move the broadcast-ready stacks once.
        geom = next(iter(self.ptycho.model.scans.values())).geometries[0]
        z_depths = [
            sum(self.p.slice_thickness[:i])
            if isinstance(self.p.slice_thickness, (list, tuple))
            else i * self.p.slice_thickness
            for i in range(1, N)
        ]

        gpar = Param()
        # ptyd metadata can hand the energy over as a 1-element array
        energy = float(np.ravel(geom.energy)[0])
        gpar.energy = energy
        gpar.psize = geom.resolution
        gpar.shape = tuple(int(v) for v in geom.shape)
        gpar.propagation = "nearfield"
        H = np.empty((max(N - 1, 1), *geom.shape), dtype=np.complex128)
        for i, zi in enumerate(z_depths):
            gpar.distance = zi
            G = geometry.Geo(owner=None, pars=gpar)
            # ownerless Geo pins its propagator to complex64; rebuild the
            # same kernel at full precision through the public helper
            prop = geometry.get_propagator(G.p, dtype=np.complex128)
            # evanescent guard: the geometry kernel is NaN where a2 > 1
            k = np.nan_to_num(np.asarray(prop.kernel, dtype=np.complex128))
            if bool(self.p.get("slice_bandlimit", True)):
                k *= slice_bandlimit(k.shape, geom.resolution, energy, zi)
            H[i] = k

        self.H = self._dev(H[: N - 1], self.cdt)
        self.Hb = self._dev(np.conj(H[: N - 1]), self.cdt)

        H_fwd = np.ones((N, 1, 1, *geom.shape), dtype=np.complex128)
        if N > 1:
            H_fwd[1:, 0, 0] = H[: N - 1]
        self.H_fwd = self._dev(H_fwd, self.cdt)
        self.H_rev = self._dev(np.conj(H_fwd), self.cdt)

        self._build_farfield()
        self._collect_views()

        self._setup_probe_support()
        self._setup_plot_state()

    def _build_farfield(self):
        """Backend hook: provide the device far-field pair as the
        callables _ff_fw / _ff_bw."""
        raise NotImplementedError

    def _setup_probe_support(self):
        """Circular probe support whenever p.probe_support is set."""
        # BaseEngine's [probe_support] convention: a circular support is
        # applied whenever the fraction is set (None disables, the
        # PositionCorrectionEngine default). The mask is built on the host
        # and moved once, the same for both backends.
        frac = getattr(self.p, "probe_support", None)
        if frac is None or not 0 < float(frac) < 1:
            self.pr_support = None
            return
        ny, nx = self.fny, self.fnx
        yy, xx = np.mgrid[:ny, :nx]
        r = np.hypot(yy - ny / 2.0, xx - nx / 2.0)
        mask = r < float(frac) * min(ny, nx) / 2.0
        self.pr_support = self._dev(mask.astype(np.float64), self.rdt)
        logger.info(
            f"probe support: circular, radius fraction {float(frac):.3f}"
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
        rows_np, cols_np = np.stack(rows), np.stack(cols)
        self.rows = self._dev(rows_np, self.idt)
        self.cols = self._dev(cols_np, self.idt)
        self.amp_dat = self._dev(np.stack(amps), self.rdt)
        self.maskb = (
            None if trivial else self._dev(np.stack(masks), self.bdt)
        )
        self.view_names = names

        # Illuminated region.  Any statistic taken over a slice is
        # meaningless outside it: most of the object array is never touched
        # and stays at exactly 1, which would drag every rms towards zero by
        # an arbitrary factor.  Read from outside the engine, by
        # diagnostics. Built on the host: overlapping patches need an
        # accumulate-scatter, which numpy's add.at gives without a device
        # kernel, chunked to bound the index arrays it materialises.
        cov = np.zeros((self.ony, self.onx))
        for s in range(0, self.n_pos, 128):
            np.add.at(
                cov,
                (rows_np[s:s + 128, :, None], cols_np[s:s + 128, None, :]),
                1.0,
            )
        self.cov = self._dev(cov > 0, self.bdt)

    def _flat(self, idx):
        return (
            self.rows[idx][:, :, None] * self.onx + self.cols[idx][:, None, :]
        ).reshape(-1)

    def _gather(self, arr, flat, B):
        return arr.reshape(-1)[flat].reshape(B, self.fny, self.fnx)

    def _model_object(self, O_patches):
        """What the forward core sees as the slice transmissions.

        The chord engine stores the transmissions O_j themselves, so this is
        the identity and the core's g = O - 1 is the chord perturbation. The
        tangent subclass stores the scattering potential s_j instead and
        returns 1 + i s_j, so the same core computes g = i s_j: the
        linearised transmission.
        """
        return O_patches

    def _grad_to_param(self, grad):
        """Chain rule from the model perturbation g to the stored parameter.

        The chord engine's parameter is O with g = O - 1, so dg/dO = 1 and
        this is the identity. The tangent subclass stores s with g = i s and
        overrides this with the conjugate factor, -i.
        """
        return grad

    def _refit_coefficients(self):
        """Refit the readout coefficients on a few scan positions.

        Data-free (paper Sec. coefficients): 'residual' minimises the
        multislice fixed-point residual at the detector, needing one Born
        order beyond those retained and no multislice solve; 'tail'
        projects the exact multislice tail onto the retained orders,
        needing one sequential multislice sweep over the sample positions.
        The basis runs on the device through the engine's _born_basis /
        _born_target; the small least-squares lands on the host. The
        result replaces self._coeffs for every following update.
        """
        mode = getattr(self, "_coeff_fit", "none")
        if mode == "none":
            return
        M, N = self._M, int(self.p.number_of_slices)
        npos = max(1, min(int(self.p.coeff_fit_positions), self.n_pos))
        sel = np.unique(np.linspace(0, self.n_pos - 1, npos).astype(np.int64))
        idx = self._dev(sel, self.idt)
        flat = self._flat(idx)
        O = self._stack([self._gather(self.ob_g[j], flat, sel.size)
                         for j in range(N)])
        O = self._model_object(O)
        if mode == "volume":
            V, D0, D = self._born_volume_basis(
                O, self.pr_g, self.H_fwd, self.H_rev, M)
            T_vol = self._born_volume_target(O, self.pr_g, self.H_fwd)
            T_det = self._born_target(O, self.pr_g, self.H_fwd)
            c, score = born_volume_remainder_coeffs(
                self._to_host(V), self._to_host(D),
                self._to_host(T_vol), self._to_host(T_det),
                self._to_host(D0))
        else:
            n_orders = min(M + 1, N) if mode == "residual" else M
            D0, D = self._born_basis(O, self.pr_g, self.H_fwd, self.H_rev,
                                     n_orders)
            D0 = self._to_host(D0)
            D = self._to_host(D)
            if mode == "residual":
                c, score = born_exit_residual_coeffs(D, M)
            else:
                T = self._to_host(self._born_target(O, self.pr_g,
                                                    self.H_fwd))
                c, score = born_qr_coeffs(
                    D[:M], T - D0, float(np.linalg.norm(D0.ravel()) ** 2))
        self._coeffs = [complex(v) for v in c]
        logger.info("coefficient fit (%s, %d positions): c = %s, score %.3e"
                    % (mode, sel.size,
                       np.array2string(np.asarray(c), precision=4), score))

    def _sync(self):
        """Order device work before a host read; backends on their own
        stream override this."""

    def _plot_progress(self):
        self._sync()
        plot_progress(
            _HostView(self._to_host(self.pr_g)),
            _HostView(self._to_host(self._display_object())),
            self.curiter,
            self.plot_progress_dir,
            psize=getattr(self.p, "psize", (1e-6, 1e-6)),
            engine_name=type(self).__name__,
            cmaps=self.PLOT_CMAPS,
            zoom_center=self.p.get("plot_zoom_center", None),
            phase_clim=self.p.get("plot_phase_clim", None),
            scalebar=self.p.get("plot_scalebar", None),
            phase_zero_top=bool(self.p.get("plot_phase_zero_top", False)),
        )
        plot_error_history(
            self._err_history,
            self.plot_progress_dir,
            engine_name=type(self).__name__,
            channels=self.p.get("plot_error_channels", None),
        )

    def _write_slices(self):
        self._sync()
        for i in range(self.p.number_of_slices):
            arr = self._to_host(self.ob_g[i])
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._prs.data[:] = self._to_host(self.pr_g)

        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        info = Param(
            number_of_slices=self.p.number_of_slices,
            slice_thickness=self.p.slice_thickness,
            alpha=self.p.get("alpha", None),
            objects={
                ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                for ob in self._object
            },
        )

        # Tell ptypy to ignore its own custom class types when saving to HDF5
        h5opt = io.h5options["UNSUPPORTED"]
        io.h5options["UNSUPPORTED"] = "ignore"

        logger.info(f"saving slices to {self.p.fslices}")
        io.h5write(
            self.p.fslices,
            header={"description": "ISS multislice"},
            content=info,
        )

        io.h5options["UNSUPPORTED"] = h5opt


class CupyISBEngineInfra(ISBEngineInfra):
    """The backend hooks of ISBEngineInfra, on CuPy."""

    def _setup_device(self):
        """Resolve device and dtypes from p.device / p.precision."""
        if cp is None:
            raise ImportError("the cupy engines require CuPy")
        want = str(self.p.device)
        if not want.startswith("cuda"):
            raise ValueError(
                "the cupy engines run on CUDA only (device=%r)" % want)
        devno = int(want.split(":")[1]) if ":" in want else 0
        cp.cuda.Device(devno).use()
        self.dev = devno
        dbl = str(self.p.precision).lower().startswith("d")
        self.cdt = cp.complex128 if dbl else cp.complex64
        self.rdt = cp.float64 if dbl else cp.float32
        self.idt = cp.int64
        self.bdt = cp.bool_
        self._eps = float(np.finfo(np.float64 if dbl else np.float32).eps)

    def _dev(self, a, dtype):
        return cp.asarray(a, dtype=dtype)

    @staticmethod
    def _to_host(a):
        return cp.asnumpy(a)

    def _scatter(self, dst, flat, vals):
        # Accumulating scatter. The engine only ever scatters one view at a
        # time (batch 1), whose patch indices are unique, so the plain
        # advanced-indexed add is exact -- and, unlike cupyx.scatter_add, it
        # works for complex values and inside a CUDA graph capture.
        dst.reshape(-1)[flat] += vals.ravel()

    def _build_farfield(self):
        """The far-field pair through ptypy's accelerate PropagationKernel
        (planned cuFFT with the geometry's pre/post filters), allocated
        once on the engine stream for the fixed batch of probe modes.
        Both directions are verified against the host pod propagator on
        random data at startup, so a convention mismatch is an error here
        rather than a silently wrong reconstruction."""
        from ptypy.accelerate.cuda_cupy.kernels import PropagationKernel

        pod = next(iter(self.di.views.values())).pod
        aux = cp.zeros((self.M, self.fny, self.fnx), dtype=self.cdt)
        # the filtered-cufft extension is single precision; the planned
        # cupy FFT handles both
        fft_type = "cuda" if self.cdt == cp.complex64 else "cupy"
        self._ff_prop = PropagationKernel(
            aux, pod.geometry.propagator,
            queue_thread=getattr(self, "_stream", None), fft_type=fft_type)
        self._ff_prop.allocate()
        self._ff_fw_buf = aux
        self._ff_bw_buf = cp.zeros_like(aux)

        rng = np.random.default_rng(0)
        w = (rng.standard_normal((self.fny, self.fnx))
             + 1j * rng.standard_normal((self.fny, self.fnx)))
        w_dev = self._dev(np.broadcast_to(w, aux.shape), self.cdt)
        for tag, mine, ref_op in (("fw", self._ff_fw, pod.fw),
                                  ("bw", self._ff_bw, pod.bw)):
            got = self._to_host(mine(w_dev))[0]
            ref = np.asarray(ref_op(w.astype(np.complex128)))
            err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
            if err > 1e-4:
                raise RuntimeError(
                    "farfield %s: PropagationKernel disagrees with the pod "
                    "propagator (rel err %.2e)" % (tag, err))
            logger.info("farfield %s: PropagationKernel verified, "
                        "rel err %.2e" % (tag, err))
        self._sync()

    def _ff_fw(self, x):
        v = x.reshape(-1, self.fny, self.fnx)
        self._ff_prop.fw(v, self._ff_fw_buf)
        return self._ff_fw_buf.reshape(x.shape)

    def _ff_bw(self, x):
        v = x.reshape(-1, self.fny, self.fnx)
        self._ff_prop.bw(v, self._ff_bw_buf)
        return self._ff_bw_buf.reshape(x.shape)

    def _setup_order_source(self, seed):
        self._order_rng = np.random.default_rng(seed + 1)

    @staticmethod
    def _stack(seq):
        return cp.stack(seq)

    def _sync(self):
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.synchronize()


# ---------------------------------------------------------------------------#
#  Coefficient fitting (paper Sec. coefficients). Pure numpy on host stacks  #
# of detector orders; data-free:                                             #
#  neither fit ever sees measured intensities, so the coefficients are a     #
#  deterministic function of the current object and probe and cannot absorb  #
#  object error.                                                             #
# ---------------------------------------------------------------------------#
def born_qr_coeffs(D, T, d0_norm2):
    """THE Born-coefficient solve: truncated-SVD least squares of the
    truncation tail onto the retained orders, in x = c - 1 coordinates:

        min_x || sum_m x_m D_m - (T - sum_m D_m) ||^2,    c = 1 + x.

    D : (M, ...) stack of the order-1..M detector readouts
    T : (...) the exact scattered target, multislice minus the unscattered
        field, in the same gauge (born_multislice_target - D0)
    d0_norm2 : ||D0||^2, the degenerate-input guard -- a dead reference
        field returns c = 1 immediately.

    numpy's lstsq with rcond=None is the same rank-aware TSVD solve as
    LAPACK gelsd: a well-conditioned basis gets the exact minimum-norm
    projection, a degenerate one cannot blow up the coefficients, and the
    solution shrinks toward c = 1 under rank truncation. Limits:
    T_tail -> 0 (weak object, or M = N) => c -> 1. Returns (c, delta) with
    delta = ||sum_m c_m D_m - T|| / ||T||, the calibration-free detector
    error estimate of the weighted truncation.
    """
    M = D.shape[0]
    A = D.reshape(M, -1).T.astype(np.complex128)
    t = T.reshape(-1).astype(np.complex128)
    t_norm = np.linalg.norm(t)
    if not np.isfinite(t_norm) or t_norm <= 0 or d0_norm2 <= 0:
        return np.ones(M, dtype=complex), 0.0
    b = t - A.sum(axis=1)  # the tail: what the plain series misses
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    delta = float(np.linalg.norm(A @ x - b) / t_norm)
    return 1.0 + x, delta


def born_exit_residual_coeffs(D, M, offset_from_one=False):
    """Exit-residual fit, target-free: least squares on the detector
    entries of the multislice fixed-point residual -- columns
    D_m - D_{m+1} (m = 1..M), right-hand side D_1. Needs the detector
    orders run to M+1 (D from the basis with n = M+1); at M = N the basis
    tops out at order N and D_{N+1} = 0 is implied (nilpotency). No
    multislice target is ever formed and no measured data enters.

    Solves for c DIRECTLY (zero prior), so under rank truncation the
    solution shrinks toward c = 0; offset_from_one=True solves the same
    problem in x = c - 1 coordinates (rhs D_{M+1}), shrinking toward
    c = 1. Returns (c, rel_res) with
    rel_res = ||A c - D_1|| / ||D_1||.
    """
    n_det = D.shape[0]
    if n_det < M:
        raise ValueError(f"need detector orders 1..{M}, got {n_det}")
    Dm = D[:M]
    if n_det > M:
        Dnext = D[1 : M + 1]
    else:  # M == N: D_{N+1} = 0 by nilpotency
        Dnext = np.concatenate([D[1:M], np.zeros_like(D[:1])])
    A = (Dm - Dnext).reshape(M, -1).T.astype(np.complex128)
    b = D[0].reshape(-1).astype(np.complex128)
    rhs = b - A.sum(axis=1) if offset_from_one else b
    x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    c = 1.0 + x if offset_from_one else x
    b_norm = np.linalg.norm(b)
    rel = float(np.linalg.norm(A @ c - b) / b_norm) if b_norm > 0 else 0.0
    return c, rel


def born_volume_remainder_coeffs(V, D, T_vol, T_det, D0):
    """VOLUME-remainder fit: least squares over every slice plane PLUS the
    detector entry, target the stored multislice wavefield itself -- the
    tail fit of born_qr_coeffs extended from the detector rows to the full
    volume. Columns are the order fields Psi^(m) (m = 1..M) on planes
    1..N-1 and D_m at the detector; the right-hand side is the multislice
    tail (T_vol - V[0] on the planes, T_det - D0 at the detector). Solved
    for offsets x = c - 1, so rank truncation shrinks toward c = 1; needs
    no order beyond those retained, unlike the residual fits.

    Normalisation: unitary. The per-plane fields are real-space rows kept
    as-is; the detector entries come from an unnormalised fft2, which
    inflates their norm by sqrt(Ny*Nx), so the detector block is divided
    by sqrt(Ny*Nx). Plane-0 rows are dropped (every order >= 1 and the
    tail are zero there -- psi_0 is the probe for any object).

    V     : (M+1, N, ...) per-order per-plane fields (order 0 included)
    D     : (M, ...) detector readouts of orders 1..M
    T_vol : (N, ...) multislice field incident on every slice plane
    T_det : (...) multislice detector field, same gauge as D
    D0    : (...) unscattered detector field

    Returns (c, delta) with delta the relative residual of the fitted
    combination against the full (volume + detector) tail.
    """
    M = D.shape[0]
    ny_nx = D.shape[-2] * D.shape[-1]
    det_scale = 1.0 / np.sqrt(ny_nx)

    def rows(vol_block, det_block):
        return np.concatenate([
            np.asarray(vol_block)[1:].ravel(),
            (np.asarray(det_block) * det_scale).ravel(),
        ])

    A = np.stack([rows(np.broadcast_to(V[m], T_vol.shape), D[m - 1])
                  for m in range(1, M + 1)], axis=1).astype(np.complex128)
    b = rows(T_vol - np.broadcast_to(V[0], T_vol.shape),
             T_det - D0).astype(np.complex128)
    b_norm = np.linalg.norm(b)
    if not np.isfinite(b_norm) or b_norm <= 0:
        return np.ones(M, dtype=complex), 0.0
    rhs = b - A.sum(axis=1)  # offsets from c = 1
    x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    delta = float(np.linalg.norm(A @ (1.0 + x) - b) / b_norm)
    return 1.0 + x, delta
