"""
ISB (Inter-Slice Born series) engine for ptychography, CuPy backend: the
order-M truncation of the exact Born expansion of multislice. order = 1 is
the inter-slice first Born (ISS) model; order = N is exact multislice,
because the series terminates after N orders for N slices.

The forward recursion is the factorised scattering step GV applied as
scatter, refer to the entrance plane, one exclusive cumulative sum in
depth, carry forward; the inclusive sum's last entry is each completed
order's detector readout for free. The adjoint orders come through the
reversed cumulative sum, paired into the gradients. See paper-1-parallel
for the maths and the equation references.

The optional CUDA-graph replay captures the whole per-position update; its
scratch lives in a private memory pool kept alive for the run, because the
captured kernels keep writing to the recorded addresses on every replay.
The truncation order and the readout coefficients are baked into a captured
graph: change neither mid-run.
"""

import numpy as np

from ptypy.engines import register, stochastic
from ptypy.utils.verbose import logger

from .base import CupyISBEngineInfra

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


# ===========================================================================
#  Pure Mathematical Core (Native CuPy)
# ===========================================================================
def born_series_step(
    O, P, I_dat, H_fwd, H_rev, ff_fw, ff_bw, M=1, coeffs=None, mask=None,
    calc_probe_grad=True,
):
    """
    Pure mathematical core for the order-M inter-slice Born series.

    Reduces to the inter-slice first Born model at M = 1 and to
    multislice (field and exact gradients) at M = N. Returns grad_g [N, B, Ny, Nx], K_O [N, B, Ny, Nx],
    grad_P, K_P, r_field, I_mod, with grad_* the DESCENT directions
    -dJ/d(conj .) for the amplitude l2 fidelity.
    """
    has_batch = O.ndim == 4
    if not has_batch:
        O = O[:, None]
        I_dat = I_dat[None]
        if mask is not None:
            mask = mask[None]

    N, B = O.shape[0], O.shape[1]
    M = max(1, min(int(M), N))  # nilpotent termination: orders > N vanish

    def c(m):  # readout coefficient of order m (1-based)
        return 1.0 if coeffs is None else coeffs[m - 1]

    def cbar(m):
        v = c(m)
        return v.conjugate() if hasattr(v, "conjugate") else v

    # ---------------------------------------------------------
    # 0. PERTURBATION & UNSCATTERED FIELD
    # ---------------------------------------------------------
    g = (O - 1.0)[:, :, None]  # [N, B, 1, Ny, Nx]
    P_f = cp.fft.fft2(P)  # [Mp, Ny, Nx]
    # psi^(0)_j = P_{z_j} P, the order-0 field at every slice plane
    orders = [cp.fft.ifft2(H_fwd * P_f[None])]  # [N, 1, Mp, Ny, Nx]

    # ---------------------------------------------------------
    # 1. FORWARD PASS: the series, order by order
    # ---------------------------------------------------------
    # Entrance-plane Fourier accumulator of the detector field,
    # psi_hat = F[P] + sum_m c_m D_m with D_m the order-m readout.
    total_f = P_f[None]  # [1, Mp, Ny, Nx], broadcasts to the batch
    for m in range(1, M + 1):
        # scatter at every slice and refer to the entrance plane
        t = H_rev * cp.fft.fft2(g * orders[-1])  # [N, B, Mp, Ny, Nx]
        cs = cp.cumsum(t, axis=0)
        # the inclusive sum's last entry is the completed order's readout
        total_f = total_f + c(m) * cs[-1]
        if m < M:
            # exclusive prefix: plane k receives the fields scattered
            # strictly upstream, carried to its own depth -- GV applied
            u = cp.concatenate([cp.zeros_like(cs[:1]), cs[:-1]], axis=0)
            orders.append(cp.fft.ifft2(H_fwd * u))
        del t, cs

    # ---------------------------------------------------------
    # 2. DETECTOR PROJECTION (identical to the ISS core)
    # ---------------------------------------------------------
    psi = cp.fft.ifft2(total_f)  # [B, Mp, Ny, Nx], entrance-plane gauge
    F_psi = ff_fw(psi)
    I_mod = (F_psi.real**2 + F_psi.imag**2).sum(axis=1)
    fac = cp.sqrt(I_dat / cp.maximum(I_mod, 1e-12)) - 1.0
    if mask is not None:
        fac = cp.where(mask > 0, fac, cp.zeros_like(fac))
    r_field = ff_bw(F_psi * fac[:, None])  # [B, Mp, Ny, Nx]

    # ---------------------------------------------------------
    # 3. ADJOINT PASS: the adjoint orders Phi^(a)
    # ---------------------------------------------------------
    # Phi^(0)_j = P_{z_j} [detector correction at the entrance plane]
    phis = [cp.fft.ifft2(H_fwd * cp.fft.fft2(r_field)[None])]
    acc_P_f = None
    for a in range(M):
        # t^(a)_j = H_{-z_j} F[conj(g_j) Phi^(a)_j]: needed to advance the
        # adjoint recursion (a < M-1) and for the probe readout
        if a < M - 1 or calc_probe_grad:
            tA = H_rev * cp.fft.fft2(cp.conj(g) * phis[a])
            if calc_probe_grad:
                term = tA.sum(axis=0) * cbar(a + 1)
                acc_P_f = term if acc_P_f is None else acc_P_f + term
            if a < M - 1:
                # G^dag: the same factorisation with the cumulative sum
                # reversed -- plane j receives everything strictly
                # downstream of it
                rcs = cp.flip(cp.cumsum(cp.flip(tA, 0), axis=0), 0)
                u = cp.concatenate([rcs[1:], cp.zeros_like(rcs[:1])], axis=0)
                phis.append(cp.fft.ifft2(H_fwd * u))
            del tA

    # ---------------------------------------------------------
    # 4. GRADIENTS & PRECONDITIONERS
    # ---------------------------------------------------------
    # -dJ/d(conj g_j) = sum_m conj(c_m) sum_{l<m} conj(psi^(l)_j)
    #                                            Phi^(m-1-l)_j
    # regrouped by forward order l, with the weighted adjoint sum inside
    grad_g = None
    for l in range(M):
        w = None
        for a in range(M - l):
            w = phis[a] * cbar(l + a + 1) if w is None \
                else w + phis[a] * cbar(l + a + 1)
        term = (cp.conj(orders[l]) * w).sum(axis=2)  # modes
        grad_g = term if grad_g is None else grad_g + term  # [N, B, Ny, Nx]

    # K_O from the field that multiplies g_j in the readout,
    # sum_m c_m psi^(m-1)_j; at M = 1 this is the unscattered illumination
    w_inc = None
    for m in range(1, M + 1):
        w_inc = orders[m - 1] * c(m) if w_inc is None \
            else w_inc + orders[m - 1] * c(m)
    K_O = (w_inc.real**2 + w_inc.imag**2).sum(axis=2)
    K_O = cp.broadcast_to(K_O, (N, B) + K_O.shape[-2:])  # [N, B, Ny, Nx]

    if calc_probe_grad:
        grad_P = (r_field + cp.fft.ifft2(acc_P_f)).sum(axis=0)  # [Mp, Ny, Nx]
        K_P = (O.real**2 + O.imag**2).sum(axis=(0, 1))
    else:
        grad_P, K_P = None, None

    if not has_batch:
        grad_g = grad_g[:, 0]
        K_O = K_O[:, 0]
        r_field = r_field[0]
        I_mod = I_mod[0]

    return grad_g, K_O, grad_P, K_P, r_field, I_mod


def born_detector_orders(O, P, H_fwd, H_rev, n):
    """D0 = F[P] and the order-1..n detector readouts in the entrance-plane
    Fourier gauge, so the plain-series detector field is D0 + sum_m D_m.
    Returns (D0, D) with D of shape (n, B, Mp, Ny, Nx); the coefficient
    fits in base.py consume the host copies. No gradients.
    """
    g = (O - 1.0)[:, :, None]
    P_f = cp.fft.fft2(P)
    cur = cp.fft.ifft2(H_fwd * P_f[None])
    Ds = []
    for m in range(1, int(n) + 1):
        t = H_rev * cp.fft.fft2(g * cur)
        cs = cp.cumsum(t, axis=0)
        Ds.append(cs[-1])
        if m < n:
            u = cp.concatenate([cp.zeros_like(cs[:1]), cs[:-1]], axis=0)
            cur = cp.fft.ifft2(H_fwd * u)
    D = cp.stack(Ds)
    return cp.broadcast_to(P_f[None], D[0].shape).copy(), D


def born_multislice_target(O, P, H_fwd):
    """Exact multislice detector field of the current object, in the same
    entrance-plane Fourier gauge as the detector orders (the unimodular
    common propagator divided out), by one sequential sweep. The 'tail'
    coefficient fit's target is this minus D0. No gradients.
    """
    N = O.shape[0]
    psi = O[0][:, None] * P[None]  # [B, Mp, Ny, Nx]
    for j in range(1, N):
        Hstep = H_fwd[j, 0, 0] * cp.conj(H_fwd[j - 1, 0, 0])
        psi = O[j][:, None] * cp.fft.ifft2(
            Hstep * cp.fft.fft2(psi))
    return cp.conj(H_fwd[N - 1, 0, 0]) * cp.fft.fft2(psi)


def born_volume_basis(O, P, H_fwd, H_rev, n):
    """Per-order PER-PLANE fields plus the detector entries, for the
    volume-remainder fit: V[m] is the order-m real-space field at every
    slice plane (zero on planes < m), m = 0..n, together with D0 and D as
    from born_detector_orders. One more advance than the detector basis,
    so V[n] is populated.
    """
    g = (O - 1.0)[:, :, None]
    B = O.shape[1]
    P_f = cp.fft.fft2(P)
    cur = cp.fft.ifft2(H_fwd * P_f[None])
    V, Ds = [cur], []
    for m in range(1, int(n) + 1):
        t = H_rev * cp.fft.fft2(g * cur)
        cs = cp.cumsum(t, axis=0)
        Ds.append(cs[-1])
        u = cp.concatenate([cp.zeros_like(cs[:1]), cs[:-1]], axis=0)
        cur = cp.fft.ifft2(H_fwd * u)
        V.append(cur)
    D = cp.stack(Ds)
    V = cp.stack([cp.broadcast_to(v, (v.shape[0], B) + v.shape[2:])
                  for v in V])
    return V, cp.broadcast_to(P_f[None], D[0].shape).copy(), D


def born_multislice_volume_target(O, P, H_fwd):
    """Multislice field INCIDENT on every slice plane, [N, B, Mp, Ny, Nx],
    by one sequential sweep that stores every plane -- same plane
    convention as the volume basis, whose orders sum to it exactly
    (nilpotent termination).
    """
    N, B = O.shape[0], O.shape[1]
    psi = cp.broadcast_to(P[None], (B,) + P.shape).copy()
    planes = [psi]
    for j in range(1, N):
        Hstep = H_fwd[j, 0, 0] * cp.conj(H_fwd[j - 1, 0, 0])
        psi = cp.fft.ifft2(
            Hstep * cp.fft.fft2(planes[-1] * O[j - 1][:, None]))
        planes.append(psi)
    return cp.stack(planes)


# ---------------------------------------------------------------------------#
#  ISB Engine wrapper                                                        #
# ---------------------------------------------------------------------------#
@register()
class ISBCupy(CupyISBEngineInfra, stochastic.EPIE):
    """
    Inter-slice Born series multislice ptychography on the GPU, CuPy
    backend. order = 1 is the inter-slice first Born model;
    order = number_of_slices is exact multislice with its exact gradients,
    evaluated with sequential depth M instead of N.

    Defaults:

    [name]
    default = ISBCupy
    type = str
    help = Engine identifier

    [order]
    default = 2
    type = int
    help = Truncation order M of the Born series
    doc = Clamped to [1, number_of_slices]. 1 is the inter-slice first Born
          model; number_of_slices terminates the series and is
          exact multislice. Each order adds one scatter-propagate step, so
          the cost grows linearly and the sequential depth is M, not N.
          Baked into a captured CUDA graph; do not change mid-run.

    [coefficients]
    default = None
    type = list, tuple
    help = Readout coefficients c_1..c_M; None is the plain truncation
    doc = Real per-order weights on the detector sum (paper Sec.
          coefficient tuning). They reweight only the readout -- the orders
          themselves stay the plain series -- and all ones (None) at
          order = number_of_slices is exact multislice. Fewer entries than
          the order are padded with ones. Baked into a captured CUDA graph.

    [number_of_slices]
    default = 2
    type = int
    help = Number of object slices

    [coeff_fit]
    default = none
    type = str
    help = Data-free readout-coefficient fit: 'none', 'residual', 'tail' or 'volume'
    doc = 'residual' minimises the multislice fixed-point residual at the
          detector (one Born order beyond those retained, no multislice
          solve); 'tail' least-squares the exact multislice tail onto the
          retained orders (one sequential multislice sweep over the fit
          positions); 'volume' extends the tail fit from the detector rows
          to the wavefield on every slice plane (one per-plane multislice
          sweep, the costliest and the most constrained). None sees
          measured data, so the coefficients are a function of the current
          object and probe only. Overrides the static [coefficients]
          whenever it runs.

    [coeff_fit_interval]
    default = 1
    type = int
    help = Iterations between coefficient refits

    [coeff_fit_positions]
    default = 8
    type = int
    help = Scan positions the coefficient fit is evaluated on

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Slice separation in meters
    doc = A single float, or a list of length number_of_slices - 1.

    [fslices]
    default = slices.h5
    type = str
    help = Output path for the reconstructed slices

    [warm_start_h5]
    default = None
    type = str
    help = Path to a previous fslices h5 whose object slices seed this run
    doc = When set, the object slices (Cobj_o_i) replace the 1+noise init after
          engine_initialize. Must have number_of_slices slices. Used to continue
          a prior reconstruction without re-running its iterations.

    [warm_start_ptyr]
    default = None
    type = str
    help = Path to a previous .ptyr whose probe seeds this run
    doc = When set, the probe replaces the illumination init after
          engine_initialize. Shape must match the probe storage.

    [alpha]
    default = 1.0
    type = float
    help = Object update step size
    doc = Divided by number_of_slices internally. Values near 1 are safe and
          aggressive; 0.2 is roughly five times slower than it needs to be.

    [beta]
    default = 1.0
    type = float
    help = Probe update step size
    doc = Divided by number_of_slices internally, as for alpha.

    [probe_update_interval]
    default = 1
    type = int
    help = Batches between probe updates
    doc = The probe gradient is a quarter of the transforms in a batch and the
          probe converges long before the object, so 4 costs almost nothing in
          quality and saves close to 20 percent of the runtime.

    [cuda_graphs]
    default = False
    type = bool
    help = Capture the per-position update as a CUDA graph and replay it
    doc = The first visit of each update variant (with / without the probe
          gradient) runs eagerly to build the cuFFT plans and warm the memory
          pool; the variant is captured on its second visit and replayed for
          every following position, removing the per-batch kernel-launch
          overhead. The capture bakes in alpha, beta, the probe schedule, the
          mask and every array shape, so change nothing mid-run. If a capture
          fails the engine logs a warning and continues eagerly.

    [raw_cuda]
    default = False
    type = bool
    help = Hand-written fused CUDA kernels for the order-M core
    doc = Replaces the cupy core call with born_cuda.BornCudaStep (one
          fused kernel per pointwise stage, planned cuFFT), everything
          around it unchanged, so cuda_graphs capture covers it too.
          Requires single precision, a single probe mode, no detector
          mask and the plain series (no coefficients, coeff_fit none);
          anything else falls back to the cupy core with a warning. The
          step is verified against the cupy core on the first scan
          position's real data at startup.

    [device]
    default = cuda
    type = str
    help = CUDA device: 'cuda' or 'cuda:0'
    doc = CuPy is CUDA-only.

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

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

    [plot_interval]
    default = 10
    type = int
    help = Iterations between diagnostic PNGs, 0 disables
    doc = On by
          default: with no explicit progress directory the figures and the
          terminal log land in iteration_progress_<engine>_<stamp>/ in the
          working directory, so every run leaves a browsable record.

    """

    _born_basis = staticmethod(born_detector_orders)
    _born_target = staticmethod(born_multislice_target)
    _born_volume_basis = staticmethod(born_volume_basis)
    _born_volume_target = staticmethod(born_multislice_volume_target)

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.article = dict(
            title='An inter-slice Born series for parallel-in-depth '
                  'ptychography of thick samples',
            author='B. Shirley et al.',
            journal='in preparation',
            volume=0,
            year=2026,
            page=0,
            doi='',
            comment='The inter-slice Born series reconstruction algorithm',
        )
        ptycho_parent.citations.add_article(**self.article)

    def engine_initialize(self):
        super().engine_initialize()
        self._stamp_plot_dir()
        # All engine work runs on one non-default stream: CUDA graph capture
        # is illegal on the legacy default stream, and keeping the eager path
        # on the same stream means the two modes are ordered identically.
        # Created before the infrastructure so the far-field
        # PropagationKernel allocates its FFT plans on it.
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._setup_infra()
        N = self.p.number_of_slices

        # Persistent Gradient / Preconditioner Accumulators
        self._acc_grad = cp.zeros((N, self.ony, self.onx), dtype=self.cdt)
        self._acc_K = cp.zeros((N, self.ony, self.onx), dtype=self.rdt)

        self._err = cp.zeros(3, dtype=cp.float64)
        self._nbatch = 0
        self._pint = max(1, int(self.p.probe_update_interval))

        self._use_graphs = bool(self.p.cuda_graphs)
        self._graphs = {}        # do_probe -> captured cupy.cuda.Graph
        self._graph_warm = set() # variants that have run eagerly once
        self._idx_buf = cp.zeros((1,), dtype=cp.int64)
        if self._use_graphs:
            # The graph's intermediates must live in their own pool, kept
            # alive for the run. Captured kernels keep writing to those
            # scratch addresses on every replay; if the blocks returned to
            # the shared pool, any later allocation (the per-iteration
            # position order, the error read-out) could land inside them
            # and be silently overwritten -- measured as a reconstruction
            # that fits a corrupted position order with a tiny error.
            self._graph_pool = cp.cuda.MemoryPool()
            logger.info("cuda_graphs on: each update variant runs eagerly "
                        "once, is captured on its second visit, and replays "
                        "thereafter")

        N = int(self.p.number_of_slices)

        self._M = max(1, min(int(self.p.order), N))
        if int(self.p.order) != self._M:
            logger.warning("order %d clamped to %d (number_of_slices)"
                           % (int(self.p.order), self._M))
        self._coeffs = None
        if self.p.coefficients is not None:
            cs = [float(v) for v in self.p.coefficients][: self._M]
            cs += [1.0] * (self._M - len(cs))
            if any(v != 1.0 for v in cs):
                self._coeffs = cs
        logger.info(
            "inter-slice Born series: order M = %d of N = %d%s%s"
            % (self._M, N,
               " (exact multislice)" if self._M == N else "",
               ", coefficients %s" % self._coeffs if self._coeffs else ""))
        self._coeff_fit = str(self.p.coeff_fit or "none").lower()
        self._cf_int = max(1, int(self.p.coeff_fit_interval))
        if self._coeff_fit != "none" and self._use_graphs:
            logger.warning("coeff_fit changes the readout between captures; "
                           "cuda_graphs is disabled for this run")
            self._use_graphs = False
        self._setup_raw_step()

    def _setup_raw_step(self):
        """Optional raw-CUDA core (p.raw_cuda): preallocate the fused-
        kernel step, alias the probe view, and verify it against the
        cupy core on the first scan position before trusting it."""
        self._raw_step = None
        if not bool(self.p.get("raw_cuda", False)):
            return
        why = None
        if self.cdt != cp.complex64:
            why = "double precision"
        elif self.M != 1:
            why = "%d probe modes" % self.M
        elif self.maskb is not None:
            why = "a detector mask"
        elif self._coeffs is not None or self._coeff_fit != "none":
            why = "readout coefficients"
        if why is not None:
            logger.warning("raw_cuda: the fused kernels do not support %s; "
                           "falling back to the cupy core" % why)
            return
        # the geometry's far-field filters: post cancels through the
        # modulus projection when unimodular and bw is the conjugate
        # inverse pair; only the pre-FFT chirp enters the kernels
        fp = self._ff_prop._p
        pre, post = np.asarray(fp.pre_fft), np.asarray(fp.post_fft)
        pre_i, post_i = np.asarray(fp.pre_ifft), np.asarray(fp.post_ifft)
        if (fp.crop_pad != 0).any():
            logger.warning("raw_cuda: crop_pad geometries are not "
                           "supported; falling back to the cupy core")
            return
        if (np.abs(np.abs(post) - 1.0).max() > 1e-5
                or np.abs(pre_i - np.conj(post)).max() > 1e-5
                or np.abs(post_i - np.conj(pre)).max() > 1e-5):
            logger.warning("raw_cuda: the far-field filters are not a "
                           "unimodular conjugate-inverse pair; falling "
                           "back to the cupy core")
            return
        chirp = None if np.abs(pre - 1.0).max() < 1e-6 \
            else cp.asarray(np.broadcast_to(pre, (self.fny, self.fnx)),
                            dtype=self.cdt)
        from .born_cuda import BornCudaStep, verify_step
        N = int(self.p.number_of_slices)
        with self._stream:
            step = BornCudaStep(
                cp.zeros((N, self.fny, self.fnx), dtype=self.cdt),
                self.pr_g[0],
                cp.zeros((self.fny, self.fnx), dtype=self.rdt),
                self.H_fwd[:, 0, 0], self.H_rev[:, 0, 0], self._M,
                pre=chirp)
            # first position's real data as the verification input,
            # against the ENGINE's far-field pair
            idx = self._dev(np.zeros(1, dtype=np.int64), self.idt)
            flat = self._flat(idx)
            O_p = cp.stack([self._gather(self.ob_g[j], flat, 1)[0]
                            for j in range(N)])
            step.O[...] = self._model_object(O_p)
            step.Idat[...] = self.amp_dat[0] ** 2
            ok, rep = verify_step(step, ff_fw=self._ff_fw,
                                  ff_bw=self._ff_bw)
            self._stream.synchronize()
        if not ok:
            raise RuntimeError("raw_cuda: fused kernels disagree with the "
                               "cupy core: %s" % rep)
        logger.info("raw_cuda on: fused kernels verified against the cupy "
                    "core, max rel err %.2e"
                    % max(rep.values()))
        self._raw_step = step

    def _step(self, idx):
        """One scan position: probe scheduling on the host, then the update
        eagerly or as a CUDA-graph replay."""
        self._nbatch += 1
        do_probe = (
            self.curiter >= self.p.probe_update_start
            and self._nbatch % self._pint == 0
        )

        if not self._use_graphs:
            self._batch_update(idx, do_probe)
            return

        # The graph reads the position index from this persistent buffer;
        # refilling it is the only per-position host work after capture.
        cp.copyto(self._idx_buf, idx)
        graph = self._graphs.get(do_probe)
        if graph is None:
            # Warm-up and capture both allocate from the engine's private
            # pool: the warm-up populates it (so the capture itself never
            # calls cudaMalloc, which is illegal mid-capture) and the
            # captured scratch addresses stay out of the shared pool for
            # good -- see the note at _graph_pool.
            prev_alloc = cp.cuda.get_allocator()
            cp.cuda.set_allocator(self._graph_pool.malloc)
            try:
                if do_probe not in self._graph_warm:
                    # eager warm-up: builds the cuFFT plans and pools the
                    # workspaces this variant will ask for during capture
                    self._graph_warm.add(do_probe)
                    self._batch_update(self._idx_buf, do_probe)
                    return
                try:
                    self._stream.begin_capture()
                    self._batch_update(self._idx_buf, do_probe)
                    graph = self._stream.end_capture()
                except Exception as e:  # noqa: BLE001 - capture errors
                    try:
                        self._stream.end_capture()
                    except Exception:  # noqa: BLE001
                        pass
                    logger.warning(
                        "CUDA graph capture failed (%s); continuing eagerly"
                        % e)
                    self._use_graphs = False
                    self._batch_update(self._idx_buf, do_probe)
                    return
            finally:
                cp.cuda.set_allocator(prev_alloc)
            self._graphs[do_probe] = graph
            logger.info("captured CUDA graph for the %s variant"
                        % ("probe-update" if do_probe else "object-only"))
        graph.launch(self._stream)

    def _batch_update(self, idx, do_probe):
        """
        One scan position: gather the slice patches, run the order-M core,
        track the errors, and apply the preconditioned Jacobi update.
        The same update at every order, except for the core call and
        the batch-resolved K_O; device work only, so the inherited CUDA
        graph capture records it unchanged.
        """
        B = int(idx.size)
        flat = self._flat(idx)

        O_patches = cp.stack(
            [
                self._gather(self.ob_g[j], flat, B)
                for j in range(self.p.number_of_slices)
            ]
        )
        O_patches = self._model_object(O_patches)
        amp_dat = self.amp_dat[idx]
        I_dat = amp_dat**2
        mask = self.maskb[idx].astype(self.rdt) if self.maskb is not None \
            else None

        if self._raw_step is not None:
            st = self._raw_step
            st.O[...] = O_patches[:, 0]
            st.Idat[...] = I_dat[0]
            st.run()
            grad_g, K_O = st.grad[:, None], st.K[:, None]
            grad_P, K_P = st.gP[None], st.KP
            r_field, I_mod = st.rI[None, None], st.Imod[None]
        else:
            grad_g, K_O, grad_P, K_P, r_field, I_mod = born_series_step(
                O=O_patches,
                P=self.pr_g,
                I_dat=I_dat,
                H_fwd=self.H_fwd,
                H_rev=self.H_rev,
                ff_fw=self._ff_fw,
                ff_bw=self._ff_bw,
                M=self._M,
                coeffs=self._coeffs,
                mask=mask,
                calc_probe_grad=do_probe,
            )

        # ---- convergence metrics ----
        amp_mod = cp.sqrt(cp.maximum(I_mod, 0))
        m_weight = 1.0 if mask is None else mask
        m_sum = (
            float(B * self.fny * self.fnx)
            if mask is None
            else cp.maximum(mask.sum(), 1)
        )
        e0 = (m_weight * (amp_mod - amp_dat) ** 2).sum() / m_sum
        e1 = (m_weight * (I_mod - I_dat) ** 2 / (I_dat + 1.0)).sum() / m_sum
        self._err += cp.stack(
            (e0, e1, cp.abs(r_field).sum())
        ).astype(cp.float64)

        # ---- scatter and preconditioned update ----
        self._acc_grad.fill(0)
        self._acc_K.fill(0)
        for j in range(self.p.number_of_slices):
            self._scatter(self._acc_grad[j], flat, grad_g[j])
            self._scatter(self._acc_K[j], flat, K_O[j])

        N = self.p.number_of_slices
        eps = self._eps

        if do_probe:
            dn_p = K_P.max()
            self.pr_g += (float(self.p.beta) / N) * (
                grad_P / cp.maximum(dn_p, eps)
            )
            if self.pr_support is not None:
                self.pr_g *= self.pr_support

        dn_o = self._acc_K.max(axis=(-2, -1), keepdims=True)
        upd = self._grad_to_param(self._acc_grad / cp.maximum(dn_o, eps))
        self.ob_g += (float(self.p.alpha) / N) * upd

    def engine_iterate(self, num=1):
        B = 1
        # Order anything still pending on other streams (initialisation
        # transfers run on the default stream) before the engine stream
        # touches the same arrays.
        cp.cuda.Device().synchronize()
        for _ in range(num):

            if self._coeff_fit != "none" and self.curiter % self._cf_int == 0:
                with self._stream:
                    self._refit_coefficients()
                self._stream.synchronize()

            with self._stream:
                if self.p.shuffle:
                    perm = (
                        self._order_rng.permutation(self.n_pos)
                        if self._order_rng is not None
                        else np.random.permutation(self.n_pos)
                    )
                    order = cp.asarray(perm, dtype=cp.int64)
                else:
                    order = cp.arange(self.n_pos, dtype=cp.int64)

                self._err.fill(0)
                for s in range(0, int(order.size), B):
                    self._step(order[s : s + B])

            self._stream.synchronize()
            err = cp.asnumpy(self._err / float(self.n_pos))
            self._err_history.append(err.copy())

            # Increment first, then plot: the frame is named for the number
            # of iterations behind it, and the last one is saved.
            self.curiter += 1
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        return {"all": err}

    def engine_finalize(self):
        self._write_slices()
        return super().engine_finalize()


@register()
class ISBTangentCupy(ISBCupy):
    """
    Order-M inter-slice Born series with the tangent (linearised-
    transmission) perturbation, CuPy backend: the reconstructed unknown is
    the scattering potential s_j with g_j = i s_j, exactly as
    the tangent is to the chord engine. Everything else -- the truncation order,
    the readout coefficients, cuda_graphs, batching, propagators, probe
    update, seeding -- is inherited unchanged.

    Defaults:

    [name]
    default = ISBTangentCupy
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
        # the working object becomes the potential s = -i log O; the start
        # 1 + 1e-3 noise maps to s ~ 1e-3 noise, the same symmetry breaking
        self.ob_g = -1j * cp.log(self.ob_g)
        if bool(self.p.potential_real):
            self.ob_g = self.ob_g.real.astype(self.cdt)
            logger.info("tangent potential constrained to be real (pure phase)")

    def _model_object(self, O_patches):
        # the core subtracts 1, so 1 + i s makes the perturbation g = i s
        return 1.0 + 1j * O_patches

    def _grad_to_param(self, grad):
        # dL/ds* = conj(dg/ds) dL/dg* = -i dL/dg*; for real s keep the
        # real part, which is the Wirtinger descent direction on the reals
        g = -1j * grad
        return g.real.astype(self.cdt) if bool(self.p.potential_real) else g

    def _display_object(self):
        return cp.exp(1j * self.ob_g)

    def engine_finalize(self):
        # hand transmissions, not potentials, to the containers and file
        self.ob_g = cp.exp(1j * self.ob_g)
        return super().engine_finalize()


@register()
class ISBLineSearchCupy(ISBCupy):
    """
    Inter-slice first Born with an exact-model line search
    (paper-1a-parallel-opt, Sec. "Line search").

    At order 1 the detector field is AFFINE in the step size: the object
    update O_j -> O_j + gamma d_j gives psi(gamma) = psi + gamma dhat,
    with dhat the first-Born forward model of the search direction
    (no unscattered term), one extra forward evaluation per position.
    The model intensity along the line is the exact quadratic
    I(gamma) = I0 + gamma c1 + gamma^2 c2, c1 = 2 Re(conj(psi) dhat),
    c2 = |dhat|^2, so for the amplitude l2 fidelity (J' = 1 -
    sqrt(Idat/I), J'' = sqrt(Idat)/2 I^-3/2) the step follows from a few
    Newton iterations gamma <- gamma - L'/L'', refined pixelwise with no
    further transforms. The probe step beta is found the same way
    against the ALREADY UPDATED object (its detector field is known in
    closed form), with dhat_P including the unscattered term; block
    coordinate descent as in ePIE. alpha and beta are not used -- the
    line search sets both scales; the preconditioners only weight the
    slices relative to each other. Everything stays on the device
    (fixed-count Newton loop, branchless safeguard), so cuda_graphs
    capture still applies. Chord model, order 1 only; raw_cuda is
    ignored.

    Defaults:

    [name]
    default = ISBLineSearchCupy
    type = str
    help =
    doc =

    [newton_iters]
    default = 2
    type = int
    lowlim = 1
    help = Newton refinements of each step size on the amplitude loss
    doc = The first iterate is the Taylor-expansion minimiser at the
          current state; further iterations cost pixelwise work only.

    [ls_relax]
    default = 1.0
    type = float
    lowlim = 0.0
    help = Relaxation of the found step sizes, gamma <- ls_relax * gamma
    doc = The search is exact for ONE scan position's loss, so late in a
          reconstruction the exact steps zig-zag around the summed
          objective (stochastic overshoot); 0.7-0.9 keeps the early
          speed and recovers the small-step averaging of the tail.
    """

    def engine_initialize(self):
        super().engine_initialize()
        if self._M != 1:
            raise ValueError("ISBLineSearchCupy implements the affine "
                             "(order 1) line search; got order %d"
                             % self._M)
        if self._raw_step is not None:
            logger.warning("raw_cuda is ignored by the line-search engine")
            self._raw_step = None
        self._newton_iters = max(1, int(self.p.newton_iters))
        self._ls_relax = float(self.p.ls_relax)

    def _newton_step_size(self, I0, c1, c2, I_dat, amp_dat, w):
        """Newton iterations on the amplitude-l2 loss along the line,
        on the exact quadratic I(gamma); returns a 0-d device scalar.
        Branchless safeguard: a non-convex iterate leaves gamma as it
        was (gamma stays 0 if the first expansion has no minimiser)."""
        tiny = cp.asarray(np.finfo(np.float32).tiny, dtype=self.rdt)
        gam = cp.zeros((), dtype=self.rdt)
        for _ in range(self._newton_iters):
            Ig = cp.maximum(I0 + gam * c1 + gam * gam * c2, 1e-12)
            dI = c1 + 2.0 * gam * c2
            rIg = cp.sqrt(Ig)
            Jp = 1.0 - amp_dat / rIg
            Jpp = 0.5 * amp_dat / (Ig * rIg)
            L1 = (w * Jp * dI).sum()
            L2 = (w * (Jpp * dI * dI + 2.0 * Jp * c2)).sum()
            gam = cp.where(L2 > 0, gam - L1 / cp.maximum(L2, tiny), gam)
        return self._ls_relax * gam

    def _batch_update(self, idx, do_probe):
        """One scan position: direction from the order-1 core, exact
        affine line search for the object step, then the probe step
        against the updated detector field. Device work only."""
        B = int(idx.size)
        flat = self._flat(idx)
        N = self.p.number_of_slices
        eps = self._eps

        O_p = cp.stack([self._gather(self.ob_g[j], flat, B)
                        for j in range(N)])
        O_model = self._model_object(O_p)
        amp_dat = self.amp_dat[idx]
        I_dat = amp_dat ** 2
        mask = self.maskb[idx].astype(self.rdt) if self.maskb is not None \
            else None
        w = mask if mask is not None else 1.0
        cf = complex(self._coeffs[0]) if self._coeffs else 1.0

        grad_g, K_O, grad_P, K_P, r_field, I_mod = born_series_step(
            O=O_model, P=self.pr_g, I_dat=I_dat,
            H_fwd=self.H_fwd, H_rev=self.H_rev,
            ff_fw=self._ff_fw, ff_bw=self._ff_bw,
            M=1, coeffs=self._coeffs, mask=mask,
            calc_probe_grad=do_probe)

        # ---- convergence metrics (pre-update, as in the base engine) ----
        amp_mod = cp.sqrt(cp.maximum(I_mod, 0))
        m_weight = 1.0 if mask is None else mask
        m_sum = (float(B * self.fny * self.fnx) if mask is None
                 else cp.maximum(mask.sum(), 1))
        e0 = (m_weight * (amp_mod - amp_dat) ** 2).sum() / m_sum
        e1 = (m_weight * (I_mod - I_dat) ** 2 / (I_dat + 1.0)).sum() / m_sum
        self._err += cp.stack(
            (e0, e1, cp.abs(r_field).sum())).astype(cp.float64)

        # ---- the current detector field and the object direction's ----
        # entrance-Fourier gauge through born_detector_orders, detector
        # plane through the engine's far-field pair (buffers are reused
        # by _ff_fw, so each result is copied out)
        D0, D = born_detector_orders(O_model, self.pr_g,
                                     self.H_fwd, self.H_rev, 1)
        psi_det = self._ff_fw(cp.fft.ifft2(D0 + cf * D[0])).copy()

        d_param = self._grad_to_param(
            grad_g / cp.maximum(K_O.max(axis=(-2, -1), keepdims=True),
                                eps))
        # model-space direction; the difference form covers any affine
        # parametrisation of the model object
        dg = self._model_object(O_p + d_param) - O_model
        d_det = self._ff_fw(cp.fft.ifft2(
            cf * born_detector_orders(1.0 + dg, self.pr_g,
                                      self.H_fwd, self.H_rev, 1)[1][0]
        )).copy()

        I0 = (psi_det.real ** 2 + psi_det.imag ** 2).sum(axis=1)
        c1 = 2.0 * (psi_det.conj() * d_det).real.sum(axis=1)
        c2 = (d_det.real ** 2 + d_det.imag ** 2).sum(axis=1)
        gam = self._newton_step_size(I0, c1, c2, I_dat, amp_dat, w)

        self._acc_grad.fill(0)
        upd = gam * d_param
        for j in range(N):
            self._scatter(self._acc_grad[j], flat, upd[j])
        self.ob_g += self._acc_grad

        # ---- probe step against the UPDATED detector field ----
        if do_probe:
            d_P = grad_P / cp.maximum(K_P.max(), eps)
            Dp0, Dp = born_detector_orders(O_model + gam * dg, d_P,
                                           self.H_fwd, self.H_rev, 1)
            dP_det = self._ff_fw(cp.fft.ifft2(Dp0 + cf * Dp[0])).copy()
            psi_upd = psi_det + gam * d_det
            I0p = (psi_upd.real ** 2 + psi_upd.imag ** 2).sum(axis=1)
            c1p = 2.0 * (psi_upd.conj() * dP_det).real.sum(axis=1)
            c2p = (dP_det.real ** 2 + dP_det.imag ** 2).sum(axis=1)
            beta = self._newton_step_size(I0p, c1p, c2p, I_dat, amp_dat, w)
            self.pr_g += beta * d_P
            if self.pr_support is not None:
                self.pr_g *= self.pr_support
