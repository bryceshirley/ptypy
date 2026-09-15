"""
Projection engines for the ISS (Inter-Slice Single Scattering) model:
a branch-space core (_ProjectionEngineISS) with the RAARISSTorch and DMISSTorch
engines built through ptypy's projection mixins, exactly as DM and RAAR
are built on _ProjectionEngine in engines/projectional.py. Model, maths
and parameter tree in the _ProjectionEngineISS docstring.
"""

import time

import numpy as np

from ptypy.custom.torch import base_torch
from ptypy.custom.torch.base_torch import ISSEngineInfra
from ptypy.engines import projectional
from ptypy.engines import register
from ptypy.engines.base import PositionCorrectionEngine
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:
    torch = None


# ===========================================================================
#  Pure mathematical core
# ===========================================================================
def branch_fw(Psi, P_hat, H_rev, ff_fw, apply_fn):
    """Detector field for a chunk of state vectors,

    .. math::
        \\hat\\psi = \\mathcal{F}[P + \\mathcal{A}\\Psi], \\quad
        \\mathcal{A}\\Psi = \\sum_j \\mathcal{P}_{-z_j}(\\psi_j)

    Parameters
    ----------
    Psi : (B, N, M, ny, nx) complex tensor
        State vectors.
    P_hat : (M, ny, nx) complex tensor
        Affine offset :math:`\\mathcal{F}[P]`, view independent.
    H_rev : (N, 1, 1, ny, nx) complex tensor
        Reverse propagators (parent layout; ``H_rev[j, 0, 0]`` is used).

    Returns
    -------
    psi_hat : (B, M, ny, nx) complex tensor
        Detector fields.
    """
    acc = (H_rev[:, 0].unsqueeze(0) * torch.fft.fft2(Psi)).sum(dim=1)
    return P_hat.unsqueeze(0) + apply_fn(ff_fw, torch.fft.ifft2(acc))


def branch_bw(u, H_fwd, Sigma, ff_bw, apply_fn):
    """Pseudoinverse of the assembly map, distributed per branch.

    With :math:`v = \\mathrm{ff\\_bw}(u)` the correction back in the
    sample plane,

    .. math::
        (\\mathcal{A}^{+}v)_j = \\mathcal{F}^{-1}\\left[
            \\frac{H_{z_j}}{\\Sigma}\\mathcal{F}[v]\\right]

    Parameters
    ----------
    u : (B, M, ny, nx) complex tensor
        Detector-domain correction.

    Returns
    -------
    (B, N, M, ny, nx) complex tensor
        The correction distributed over the branches.
    """
    back = torch.fft.fft2(apply_fn(ff_bw, u)) / Sigma
    return torch.fft.ifft2(H_fwd[:, 0].unsqueeze(0) * back.unsqueeze(1))


def projection_update_chunk(Psi, PsiO, P_hat, fmag, fmask, pbound, coeffs,
                       H_fwd, H_rev, Sigma, ff_fw, ff_bw, apply_fn,
                       compute_ll=False):
    """One generalized projection update on a chunk of state vectors.

    ptypy's ``projection_update_generalized`` in branch space. With the
    intermediate state

    .. math::
        V = (1 - c)\\Psi_n + c\\,\\Pi_O(\\Psi_n)

    the update is

    .. math::
        \\Psi_{n+1} = \\Psi_n + (a + bc)(\\Pi_O(\\Psi_n) - \\Psi_n)
                      + b(\\Pi_F(V) - V)

    where the Fourier projection distributes the modulus correction
    through the pseudoinverse of the assembly map,

    .. math::
        \\Pi_F(V)_j = v_j + \\mathcal{F}^{-1}\\left[
            \\frac{H_{z_j}}{\\Sigma}(\\Upsilon - 1)\\hat\\psi_V\\right]

    with :math:`\\Upsilon = \\sqrt{I_{\\mathrm{data}}}/|\\hat\\psi|` the
    ptypy modulus correction, including the mask and pbound three-way
    logic (plain / relaxed / bypass) per view.

    Parameters
    ----------
    Psi : (B, N, M, ny, nx) complex tensor
        State vectors.
    PsiO : (B, N, M, ny, nx) complex tensor
        Overlap projection :math:`\\Pi_O(\\Psi)`.
    fmag : (B, ny, nx) real tensor
        Measured moduli :math:`\\sqrt{I_{\\mathrm{data}}}`.
    fmask : (B, ny, nx) real tensor, optional
        Weights in [0, 1]; ``None`` means no mask.
    pbound : (B,) real tensor, optional
        Power bound per view; ``None`` disables the relaxation.
    coeffs : tuple of float
        The projection coefficients (a, b, c).

    Returns
    -------
    Psi_new, err_fmag, err_phot, err_exit
        The updated states and the three per-view error channels.
    """
    a_c, b, c = coeffs
    B = Psi.shape[0]

    V = Psi if c == 1 else (1.0 - c) * Psi + c * PsiO
    psihat_v = branch_fw(V, P_hat, H_rev, ff_fw, apply_fn)
    af2 = psihat_v.abs().square().sum(dim=1)
    af = af2.clamp_min(0).sqrt()
    fdev = af - fmag
    if fmask is None:
        npix = float(fdev.shape[-2] * fdev.shape[-1])
        err_fmag = fdev.square().sum(dim=(-2, -1)) / npix
    else:
        npix = fmask.sum(dim=(-2, -1)).clamp_min(1.0)
        err_fmag = (fmask * fdev.square()).sum(dim=(-2, -1)) / npix

    if pbound is None:
        # plain modulus replacement: Upsilon = fmag / af, i.e. renorm = 0
        renorm = torch.zeros(B, dtype=af.dtype, device=af.device)
        bypass = torch.zeros(B, dtype=torch.bool, device=af.device)
    else:
        bypass = err_fmag <= pbound
        renorm = torch.sqrt(pbound / err_fmag.clamp_min(1e-30)).clamp(max=1.0)

    scaled = (fmag + fdev * renorm[:, None, None]) / (af + 1e-10)
    Upsilon = scaled if fmask is None else (1.0 - fmask) + fmask * scaled

    corr = (Upsilon - 1.0).unsqueeze(1) * psihat_v
    dPsi = (a_c + b * c) * (PsiO - Psi) \
        + b * branch_bw(corr, H_fwd, Sigma, ff_bw, apply_fn)
    if bypass.any():
        dPsi_bypass = (a_c + b * c) * (PsiO - Psi)
        dPsi = torch.where(bypass[:, None, None, None, None], dPsi_bypass, dPsi)

    err_exit = dPsi.abs().square().mean(dim=(-4, -3, -2, -1))

    if compute_ll:
        psihat_o = branch_fw(PsiO, P_hat, H_rev, ff_fw, apply_fn)
        I_po = psihat_o.abs().square().sum(dim=1)
        I_dat = fmag.square()
        LL = (I_po - I_dat).square() / (I_dat + 1.0)
        if fmask is not None:
            LL = fmask * LL
        err_phot = LL.sum(dim=(-2, -1)) / npix
    else:
        err_phot = torch.zeros_like(err_fmag)

    return Psi + dPsi, err_fmag, err_phot, err_exit


# ===========================================================================
#  Engine
# ===========================================================================
class _ProjectionEngineISS(ISSEngineInfra, projectional._ProjectionEngine):
    """
    Branch-space projection engine core for the ISS (Inter-Slice Single
    Scattering) model; a coefficient mixin (RAARMixin, DMMixin) makes it
    a concrete engine.

    Notation follows T09 (Thibault et al., 2009) plus :math:`\\mathcal{F}`
    (far field) and :math:`\\mathcal{P}_z = \\mathcal{F}^{-1}H_z\\mathcal{F}`
    (free-space propagation, :math:`|H_z| \\in \\{0, 1\\}`, :math:`H_0 = 1`).

    The ISS model is a multi-probe thin-sample problem: slice j pairs
    its own defocused probe :math:`P_j = \\mathcal{P}_{z_j}(P)` with its
    own perturbation :math:`\\Delta O_j = O_j - 1`, coupled only through
    the coherent detector sum and the shared probe. The engine iterates
    the state vector :math:`\\Psi = (\\psi_0, \\dots, \\psi_{N-1})` of
    branch waves :math:`\\psi_j \\sim \\Delta O_j P_j` (sample plane) per
    view; the detector field is

    .. math::
        \\hat\\psi = \\mathcal{F}[P + \\mathcal{A}\\Psi], \\quad
        \\mathcal{A}\\Psi = \\sum_j \\mathcal{P}_{-z_j}(\\psi_j)

    with the unscattered probe P an affine offset outside the assembly
    map :math:`\\mathcal{A}`. Two properties of the assembly map make the
    projections cheap. First, it is built from fixed partial isometries
    with a diagonal row sum,

    .. math::
        \\mathcal{A}\\mathcal{A}^{\\dagger} =
            \\mathcal{F}^{-1}\\Sigma\\,\\mathcal{F}, \\quad
        \\Sigma = \\sum_j |H_{z_j}|^2 \\geq 1

    (= N at every propagating frequency; :math:`\\Sigma \\geq 1` always
    since :math:`H_0 = 1`), so the minimum-norm right inverse
    :math:`(\\mathcal{A}^{+}u)_j = \\mathcal{F}^{-1}[(H_{z_j}/\\Sigma)
    \\mathcal{F}[u]]` is closed form and the Fourier projector needs no
    inner solve:

    .. math::
        \\Pi_F(\\Psi)_j = \\psi_j + \\mathcal{F}^{-1}\\left[
            \\frac{H_{z_j}}{\\Sigma}(\\Upsilon - 1)\\hat\\psi\\right]

    with :math:`\\Upsilon = \\sqrt{I_{\\mathrm{data}}}/|\\hat\\psi|`, the
    standard ptypy modulus correction (mask, pbound relaxation). Second,
    :math:`\\Pi_O` decouples across slices: the object solve is the
    thin-sample DM object update with :math:`P_j` as the probe.

    The generalized (a, b, c) combination of ptypy's projection engines
    (engines/utils.py: projection_update_generalized) carries over
    verbatim; RAAR is :math:`(1 - 2\\beta, \\beta, 2)`, DM with
    relaxation alpha is :math:`(-\\alpha, 1, 1 + \\alpha)`. At N = 1
    (:math:`z_0 = 0`, :math:`\\mathcal{A} = \\mathcal{I}`,
    :math:`\\Sigma = 1`) the branch iteration is algebraically identical
    to the standard thin-sample iteration under :math:`\\psi = P +
    \\psi_0`, which is the acceptance test for this engine.

    The iterate is :math:`\\Psi`; the overlap solve never overwrites it
    (exactly as pod.exit is the iterate in ptypy's projection engines),
    and the solution is read from :math:`\\Pi_O(\\Psi)`, i.e. the stored
    object slices.

    Parameter tree: ptypy's projection family (RAAR relaxation [beta] from
    RAARMixin; pbound, overlap and inertia parameters from
    _ProjectionEngine) plus the engine-owned entries below -- no gradient/
    stochastic parameters exist on this engine. Single MPI rank only.

    Defaults:

    [number_of_slices]
    default = 2
    type = int
    help = Number of object slices

    [slice_thickness]
    default = 1e-6
    type = float
    help = Inter-slice spacing [m]; a list gives per-gap spacings

    [fslices]
    default = slices.h5
    type = str
    help = File to which the reconstructed slices are written

    [device]
    default = cuda
    type = str
    help = Torch device: 'cuda', 'cuda:0' or 'cpu'

    [precision]
    default = single
    type = str
    help = 'single' (complex64) or 'double' (complex128)

    [seed]
    default = 123
    type = int
    help = Seed for the object start; 0 leaves it free

    [warm_start_h5]
    default = None
    type = str
    help = Optional fslices-format h5 with per-slice object warm start

    [warm_start_ptyr]
    default = None
    type = str
    help = Optional .ptyr whose probe overwrites the initial probe

    [chunk_size]
    default = 32
    type = int
    help = Views per chunk (memory only; the result is independent of it)

    [object_init]
    default = noise
    type = str
    help = Object start; 'noise' keeps the parent's 1 + 1e-3 seeded noise on
      every slice (identical to the ISSTorch/multislice pipelines),
      'container' copies the ptypy object into slice 0 (for warm starts held
      in the container)

    [wave_device]
    default = same
    type = str
    help = Where the branch waves live; 'same' keeps them on p.device, 'cpu'
      keeps them host-side and streams chunks

    [probe_denom]
    default = slices
    type = str
    help = Diagonal surrogate for the probe normal operator; 'slices' uses
      sum_j |O_j|^2 (legacy, exact only at N = 1: overestimates the diagonal
      by ~N for weak objects and shrinks the probe each sweep on a cold
      start), 'aggregate' uses |1 + sum_j DeltaO_j|^2, exact at dO = 0 and at N = 1

    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self._stamp_plot_dir()

    def engine_initialize(self):
        """Set up the torch state: object/probe tensors, branch waves Psi,
        propagators, caches and overlap accumulators.

        Bypasses _ProjectionEngine.engine_initialize, whose five helper
        containers only its own numpy updates read; position-refinement
        setup still runs.
        """
        PositionCorrectionEngine.engine_initialize(self)
        self.error = []
        self._setup_torch_infra()
        N = self.p.number_of_slices

        if self.p.probe_center_tol is not None:
            logger.warning("probe_center_tol is ignored: center_probe "
                           "recentres the ptypy containers, which this "
                           "engine does not read back")

        # a warm_start_h5 already loaded by the parent beats the container
        if (str(self.p.object_init).lower().startswith("c")
                and not getattr(self.p, "warm_start_h5", None)):
            self.ob_g[0] = torch.as_tensor(
                np.ascontiguousarray(self._obs.data[0]),
                dtype=self.cdt, device=self.dev)
            logger.info("object start: slice 0 from container")

        self._set_sigma()

        self._wave_dev = (torch.device("cpu")
                          if str(self.p.wave_device).lower() == "cpu"
                          else self.dev)
        self.chunk = max(1, int(self.p.chunk_size))

        # probe-derived caches (position independent: positions live in
        # the object gather, ptypy never shifts the probe)
        self._Pj_c = None
        self._P_hat_c = None

        # branch waves: the projection iterate, started at the model
        # image (the exit = probe*object convention of the parent)
        self.Psi = torch.zeros(
            (self.n_pos, N, self.M, self.fny, self.fnx),
            dtype=self.cdt, device=self._wave_dev)
        self._reinit_waves()

        # overlap accumulator; filled in the Fourier pass and valid
        # until the probe (and with it Pj and Psi) changes
        self._num_acc = torch.zeros((N, self.ony, self.onx),
                                    dtype=self.cdt, device=self.dev)
        self._acc_valid = False
        self._den_cache = None

        # zero start makes the first probe round report change = 1,
        # as in projectional
        self._pr_buf = torch.zeros_like(self.pr_g)

        gib = self.Psi.numel() * self.Psi.element_size() / 2**30
        logger.info("branch waves: %s on %s (%.2f GiB)"
                    % (tuple(self.Psi.shape), self._wave_dev, gib))

    # -- state helpers ---------------------------------------------------------
    def _set_sigma(self):
        """Diagonal Sigma of A A^dag: Sigma = sum_j |H_{z_j}|^2 in the
        fft2 domain. H_0 = 1 exactly, so Sigma >= 1 and dividing by it
        is safe."""
        H = self.H_fwd[:, 0, 0]
        self._Sigma = H.abs().square().sum(dim=0)

    def _reinit_waves(self):
        """Set the branch waves to the model image psi_j = DeltaO_j P_j."""
        for s in range(0, self.n_pos, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                               device=self.dev)
            self.Psi[idx.to(self._wave_dev)] = \
                self._model_waves(idx).to(self._wave_dev)

    # -- probe-derived caches ------------------------------------------------
    @property
    def _Pj(self):
        """Per-slice probes P_j = Prop_{z_j} P, cached until P moves."""
        if self._Pj_c is None:
            P_f = torch.fft.fft2(self.pr_g)
            self._Pj_c = torch.fft.ifft2(self.H_fwd[:, 0] * P_f.unsqueeze(0))
        return self._Pj_c

    @property
    def _P_hat(self):
        """Affine offset in the detector plane, P_hat = F[P], cached
        until P moves."""
        if self._P_hat_c is None:
            self._P_hat_c = self._apply(self._ff_fw, self.pr_g)
        return self._P_hat_c

    def _drop_probe_caches(self):
        """Invalidate everything derived from the probe: Pj, a, the
        overlap denominator and the fused numerator."""
        self._Pj_c = None
        self._P_hat_c = None
        self._den_cache = None
        self._acc_valid = False

    def _den_total(self):
        """cfact-free overlap denominator: sum_p |P_j|^2 on the canvas.

        Independent of Psi, so it is built once per probe state -- the
        scatter pattern only re-runs when Pj changes.
        """
        if self._den_cache is None:
            Pj = self._Pj
            Kj = Pj.abs().square().sum(dim=1)
            den = torch.zeros((len(Pj), self.ony, self.onx),
                              dtype=self.rdt, device=self.dev)
            for s in range(0, self.n_pos, self.chunk):
                idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                                   device=self.dev)
                B = idx.numel()
                flat = self._flat(idx)
                for j in range(len(Pj)):
                    self._scatter(den[j], flat,
                                  Kj[j].expand(B, self.fny, self.fnx))
            self._den_cache = den
        return self._den_cache

    # -- helpers ---------------------------------------------------------------
    def _gather_dO(self, idx):
        """DeltaO_j patches for a chunk: (B, N, 1, ny, nx)."""
        B = idx.numel()
        flat = self._flat(idx)
        O = torch.stack([self._gather(self.ob_g[j], flat, B)
                         for j in range(self.ob_g.shape[0])])
        dO = self._model_object(O) - 1.0                     # (N, B, ny, nx)
        return dO.transpose(0, 1).unsqueeze(2)

    def _model_waves(self, idx):
        """Pi_O image for a chunk: psi_j = DeltaO_j P_j, (B, N, M, ny, nx)."""
        return self._gather_dO(idx) * self._Pj.unsqueeze(0)

    # -- prepare ---------------------------------------------------------------
    def engine_prepare(self):
        """pbound_scan and mean_power exactly as _ProjectionEngine
        .engine_prepare (minus its ob_viewcover fill; the helper containers
        don't exist here), then the per-view pbound tensor and the two
        inertia factors."""
        if self.ptycho.new_data:
            mean_power = 0.
            self.pbound_scan = {}
            for s in self.di.storages.values():
                if self.p.fourier_power_bound is None:
                    pb = .25 * self.p.fourier_relax_factor**2 * s.pbound_stub
                else:
                    pb = self.p.fourier_power_bound
                self.pbound_scan[s.label] = max(
                    pb, self.pbound_scan.get(s.label, pb))
                mean_power += s.mean_power
            self.mean_power = mean_power / len(self.di.storages)
        pb = np.empty(self.n_pos)
        for i, n in enumerate(self.view_names):
            pb[i] = self.pbound_scan[self.di.views[n].storage.label]
        self.pbound = torch.as_tensor(pb, dtype=self.rdt, device=self.dev)
        self._cfact_ob = float(self.p.object_inertia) * float(self.mean_power)
        self._cfact_pr = (float(self.p.probe_inertia)
                          * self.n_pos / max(1, self.M))

    # -- the iteration -----------------------------------------------------------
    def engine_iterate(self, num=1):
        """Fourier -> overlap -> position update per iteration, plus
        progress plotting; returns the per-view error dict."""
        to = 0.0
        tf = 0.0
        tp = 0.0
        for _ in range(num):
            t1 = time.time()
            # Fourier Update
            error_dct = self.fourier_update()
            t2 = time.time()
            tf += t2 - t1

            # Overlap update
            self.overlap_update()

            t3 = time.time()
            to += t3 - t2

            # Position Update
            self.position_update()

            t4 = time.time()
            tp += t4 - t3

            self.curiter += 1
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in Position update: %.2f' % tp)
        return error_dct

    def fourier_update(self):
        """Chunked generalized projection update of Psi on the measured
        moduli, returning the parent's per-view error dict. The
        just-updated chunk also feeds the overlap numerator, so Psi is
        read once per iteration instead of twice."""
        err = np.zeros((self.n_pos, 3))
        Pj = self._Pj
        self._num_acc.zero_()
        for s in range(0, self.n_pos, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                               device=self.dev)
            flat = self._flat(idx)
            Psi_c = self.Psi[idx.to(self._wave_dev)].to(self.dev)
            PsiO = self._model_waves(idx)
            fmag = self.amp_dat[idx]
            fmask = (self.maskb[idx].to(self.rdt)
                     if self.maskb is not None else None)
            Psi_n, e0, e1, e2 = projection_update_chunk(
                Psi_c, PsiO, self._P_hat, fmag, fmask, self.pbound[idx],
                (self._a, self._b, self._c),
                self.H_fwd, self.H_rev, self._Sigma,
                self._ff_fw, self._ff_bw, base_torch.apply_torch,
                compute_ll=bool(self.p.compute_log_likelihood))
            self.Psi[idx.to(self._wave_dev)] = Psi_n.to(self._wave_dev)
            for j in range(len(Pj)):
                self._scatter(self._num_acc[j], flat,
                              (Pj[j].conj().unsqueeze(0) * Psi_n[:, j]).sum(1))
            err[s:s + idx.numel(), 0] = e0.double().cpu().numpy()
            err[s:s + idx.numel(), 1] = e1.double().cpu().numpy()
            err[s:s + idx.numel(), 2] = e2.double().cpu().numpy()
        self._acc_valid = True
        self._err_history.append(err.mean(axis=0))
        return {name: err[i] for i, name in enumerate(self.view_names)}

    def object_update(self):
        """Overlap object solve, decoupled per slice (T09 Eq. 7 with
        :math:`P_j` as the probe):

        .. math::
            \\Delta O_j = \\frac{\\sum_p \\overline{P_j}\\,\\psi_{j,p}
                + c_{\\mathrm{ob}}\\,\\Delta O_j^{\\mathrm{old}}}
                {\\sum_p |P_j|^2 + c_{\\mathrm{ob}}}

        Chord parametrisation only -- the stored slices are transmissions
        :math:`O_j = 1 + \\Delta O_j`; a tangent subclass would override
        this solve."""
        cfact = self._cfact_ob
        dO_old = self.ob_g - 1.0
        std = self.p.obj_smooth_std
        if std:
            dO_old = base_torch.complex_gaussian_filter(dO_old, float(std))
        if not self._acc_valid:
            # Psi or Pj moved since the Fourier pass: re-accumulate
            Pj = self._Pj
            self._num_acc.zero_()
            for s in range(0, self.n_pos, self.chunk):
                idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                                   device=self.dev)
                flat = self._flat(idx)
                Psi_c = self.Psi[idx.to(self._wave_dev)].to(self.dev)
                for j in range(len(Pj)):
                    self._scatter(
                        self._num_acc[j], flat,
                        (Pj[j].conj().unsqueeze(0) * Psi_c[:, j]).sum(1))
            self._acc_valid = True
        num = cfact * dO_old + self._num_acc
        dO_new = num / (cfact + self._den_total()).clamp_min(
            torch.finfo(self.rdt).eps)
        self.ob_g = 1.0 + dO_new
        if self.p.clip_object is not None:
            self.ob_g = base_torch.clip_amplitude(
                self.ob_g, self.p.clip_object)

    def probe_update(self):
        """Entrance-plane DM probe solve
        :math:`\\min_P \\lVert \\mathcal{C}P - \\psi \\rVert^2`, with
        :math:`\\mathcal{C}` the entrance-plane transmission map; returns
        the relative probe change for the overlap loop.

        Deliberately NOT the branch-space metric projection, whose
        denominator ~ sum |DeltaO_j|^2 is tiny for weak objects and badly
        conditioned. This solve (denominator sum |O_j|^2 ~ N) keeps the
        identity branch, where most probe information lives, and reduces
        exactly to ptypy's DM probe update at N = 1; the overlap step is
        then not a projection in the iterate's space -- standard
        ptychography DM makes the same trade. The Fourier set moves with P
        (a = ff_fw(P)), so the branches are compensated afterwards to keep
        the physical iterate psi fixed, consistent with the parent engines
        rather than strict Douglas-Rachford theory."""
        numP = self._cfact_pr * self.pr_g.clone()
        KP = torch.full((self.fny, self.fnx), self._cfact_pr,
                        dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                               device=self.dev)
            Psi_c = self.Psi[idx.to(self._wave_dev)].to(self.dev)
            dO = self._gather_dO(idx)                       # (B, N, 1, ny, nx)
            # entrance-plane exit wave: psi = P + A Psi
            psi = self.pr_g.unsqueeze(0) + torch.fft.ifft2(
                (self.H_rev[:, 0].unsqueeze(0)
                 * torch.fft.fft2(Psi_c)).sum(1))
            # C^dag psi = psi + sum_j Prop_{-z_j}[ conj(DeltaO_j) Prop_{z_j} psi ]
            psi_f = torch.fft.fft2(psi)
            adj = psi.clone()
            for j in range(dO.shape[1]):
                r_j = torch.fft.ifft2(self.H_fwd[j, 0] * psi_f)
                adj = adj + torch.fft.ifft2(
                    self.H_rev[j, 0]
                    * torch.fft.fft2(dO[:, j].conj() * r_j))
            numP = numP + adj.sum(dim=0)
            if str(self.p.get("probe_denom", "slices")).startswith("agg"):
                # straight-through transmission |1 + sum_j DeltaO_j|^2: the
                # correct normal-operator diagonal at dO = 0
                agg = 1.0 + dO.sum(dim=1)                   # (B, 1, ny, nx)
                KP = KP + agg.abs().square().sum(dim=(0, 1))
            else:
                O_abs2 = (1.0 + dO).abs().square()          # (B, N, 1, ny, nx)
                KP = KP + O_abs2.sum(dim=(0, 1, 2))
        new_P = numP / KP.clamp_min(torch.finfo(self.rdt).eps)
        if self.pr_support is not None:
            new_P = new_P * self.pr_support
        change = float(
            torch.linalg.vector_norm(new_P - self._pr_buf)
            / torch.linalg.vector_norm(new_P).clamp_min(
                torch.finfo(self.rdt).eps))
        self._pr_buf = new_P.clone()

        # moving offset: keep psi = P + A Psi invariant under the probe
        # change with the minimum-norm redistribution A^+ dP per branch,
        # exact even inside the evanescent cutoff
        dP = self.pr_g - new_P                              # (M, ny, nx)
        dPsi = torch.fft.ifft2(
            self.H_fwd[:, 0] * (torch.fft.fft2(dP) / self._Sigma
                                ).unsqueeze(0))            # (N, M, ny, nx)
        self.Psi += dPsi.unsqueeze(0).to(self._wave_dev)

        self.pr_g = new_P
        self._drop_probe_caches()
        return change

    def engine_finalize(self):
        """Write the slices, then sync refined positions; skips
        _ProjectionEngine's cleanup of containers never allocated here."""
        del self.Psi
        self._write_slices()
        return PositionCorrectionEngine.engine_finalize(self)


# ===========================================================================
#  Engines: the ISS projection core plus one coefficient mixin each,
#  the DM/RAAR-on-_ProjectionEngine pattern of engines/projectional.py.
# ===========================================================================
@register()
class RAARISSTorch(_ProjectionEngineISS, projectional.RAARMixin):
    """
    RAAR flavour of the ISS projection engine,

    .. math::
        (a, b, c) = (1 - 2\\beta, \\beta, 2)

    Defaults:

    [name]
    default = RAARISSTorch
    type = str
    help = Engine identifier

    [beta_anneal]
    default = None
    type = tuple
    help = Optional (beta_start, beta_final, n_iterations) linear ramp of the
      RAAR relaxation (Luke's annealing); None keeps beta fixed
    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngineISS.__init__(self, ptycho_parent, pars)
        projectional.RAARMixin.__init__(self, self.p.beta)

    def fourier_update(self):
        """Anneal beta through the mixin setter before the projection;
        with no beta_anneal the coefficients are the constructor's, as in
        projectional's RAAR."""
        ann = self.p.get("beta_anneal", None)
        if ann is not None:
            b0, b1, n_it = float(ann[0]), float(ann[1]), max(1, int(ann[2]))
            t = min(1.0, self.curiter / n_it)
            self.beta = b0 + t * (b1 - b0)  # mixin setter -> (_a, _b, _c)
        return super().fourier_update()


@register()
class DMISSTorch(_ProjectionEngineISS, projectional.DMMixin):
    """
    Difference-map flavour of the ISS projection engine,

    .. math::
        (a, b, c) = (-\\alpha, 1, 1 + \\alpha)

    Defaults:

    [name]
    default = DMISSTorch
    type = str
    help = Engine identifier
    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngineISS.__init__(self, ptycho_parent, pars)
        projectional.DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)
