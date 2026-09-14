"""
RAAR projection engine for the first Born (ISS) model.

The first Born model is a multi-probe thin-sample problem: slice j pairs
its own defocused probe phi_j = Prop_{z_j} P with its own perturbation
g_j = O_j - 1, coupled only through the coherent detector sum and the
shared probe. This engine runs RAAR in *branch space*: per view the
iterate is the stack of branch waves W = (w_0 .. w_{N-1}), w_j ~ g_j phi_j
(sample plane), and the detector field is

    psi_hat = a + S W,   a = ff_fw(P),   S W = ff_fw( sum_j Prop_{-z_j} w_j ).

Two structural facts make the projections cheap:

  * S B = diag(Sdiag) with the back map
    B(y)_j = Prop_{z_j}[ ff_bw(y) ] and Sdiag = sum_j |H_j|^2 (= N at every
    propagating frequency, >= 1 always since H_0 = 1). The Fourier
    projector is therefore CLOSED FORM -- no inner solve:
        P_F(W) = W + B[ (fm - 1) (a + S W) ] / Sdiag
    with fm the standard ptypy modulus map (mask, pbound relaxation).
  * The overlap projector's object step is diagonal per slice: the
    thin-sample DM object update with phi_j as the probe.

The generalized (a,b,c) combination of ptypy's projection engines
(engines/utils.py: projection_update_generalized) carries over verbatim;
RAAR is (a, b, c) = (1 - 2 beta, beta, 2). At N = 1 (z_0 = 0) the branch
iteration is algebraically identical to the standard RAAR iteration under
psi = P + w, which is the acceptance test for this engine.

The RAAR iterate is W; the overlap solve never overwrites W (exactly as
pod.exit is the iterate in ptypy's projection engines).

Inherited but IGNORED parent parameters: alpha, batch_size, shuffle,
momentum, born_mode, probe_update_interval, object_denom, denom_reg,
probe_converged_rtol. The parent's [beta] (a probe step size) is
repurposed here with the RAAR meaning. Single MPI rank only.

Defaults:

[name]
default = RAARFirstBornTorch
type = str
help = Engine identifier

[beta]
default = 0.9
type = float
lowlim = 0.0
help = RAAR relaxation parameter
doc = Maps to the generalized projection coefficients
  (a, b, c) = (1 - 2 beta, beta, 2). Replaces the parent's probe step
  size, which this engine does not use.

[update_object_first]
default = True
type = bool
help = Update object before probe in the overlap loop

[overlap_max_iterations]
default = 10
type = int
help = Maximum iterations of the overlap (object/probe) inner loop

[overlap_converge_factor]
default = 0.05
type = float
help = Exit the overlap loop when the relative probe change falls below this

[object_inertia]
default = 1e-4
type = float
help = Object inertia (cfact fraction of the mean diffraction power)

[probe_inertia]
default = 1e-9
type = float
help = Probe inertia

[fourier_power_bound]
default = None
type = float
help = Fourier power bound (pbound); None derives it from fourier_relax_factor

[fourier_relax_factor]
default = 0.05
type = float
help = Sets the pbound as in the ptypy projection engines when
  fourier_power_bound is None

[obj_smooth_std]
default = None
type = float
help = Gaussian smoothing (pixels) of the object prior to its update

[clip_object]
default = None
type = tuple
help = Clip the object amplitude into this interval

[chunk_size]
default = 32
type = int
help = Views per chunk (memory only; the result is independent of it)

[object_init]
default = container
type = str
help = Object start; 'container' copies the ptypy object into slice 0
  (g_j = 0 elsewhere, plus the parent's seeded noise), 'noise' keeps the
  parent's 1 + 1e-3 noise start on every slice

[exit_init]
default = model
type = str
help = Branch-wave start; 'model' sets w_j = g_j phi_j (the analogue of
  ptypy's exit = probe*object), 'zero' starts from W = 0

[wave_device]
default = same
type = str
help = Where the branch waves live; 'same' keeps them on p.device, 'cpu'
  keeps them host-side and streams chunks

"""

import numpy as np

from ptypy.custom import torch_helpers
from ptypy.custom.firstborntorch import FirstBornTorch
from ptypy.engines import register
from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:
    torch = None


# ===========================================================================
#  Pure mathematical core
# ===========================================================================
def branch_detector_field(W, a_det, H_rev, ff_fw, apply_fn):
    """a + S W for a chunk of branch waves.

    W     : (B, N, M, ny, nx) complex
    a_det : (M, ny, nx) complex  -- ff_fw(P), view independent
    H_rev : (N, 1, 1, ny, nx) complex (parent layout; H_rev[j, 0, 0] used)
    """
    acc = (H_rev[:, 0].unsqueeze(0) * torch.fft.fft2(W)).sum(dim=1)
    return a_det.unsqueeze(0) + apply_fn(ff_fw, torch.fft.ifft2(acc))


def branch_back(y, H_fwd, Sdiag, ff_bw, apply_fn):
    """B(y) / Sdiag: the closed-form right inverse of S applied per branch.

    y : (B, M, ny, nx) detector-domain correction.
    Returns (B, N, M, ny, nx).
    """
    back = torch.fft.fft2(apply_fn(ff_bw, y)) / Sdiag
    return torch.fft.ifft2(H_fwd[:, 0].unsqueeze(0) * back.unsqueeze(1))


def raar_fourier_chunk(W, PO, a_det, fmag, fmask, pbound, coeffs,
                       H_fwd, H_rev, Sdiag, ff_fw, ff_bw, apply_fn,
                       compute_ll=False):
    """One generalized projection update on a chunk of branch waves.

    Implements ptypy's projection_update_generalized in branch space:
        V   = (1 - c) W + c PO
        F_V = a + S V
        df  = (a_c + b c)(PO - W) + b B[(fm - 1) F_V] / Sdiag
    with the pbound three-way logic (plain / relaxed / bypass) per view.

    W, PO  : (B, N, M, ny, nx)
    fmag   : (B, ny, nx) measured moduli
    fmask  : (B, ny, nx) real weights in [0, 1] or None
    pbound : (B,) real tensor or None
    coeffs : (a_c, b, c)
    Returns (W_new, err_fmag, err_phot, err_exit) with per-view errors.
    """
    a_c, b, c = coeffs
    B = W.shape[0]

    V = W if c == 1 else (1.0 - c) * W + c * PO
    F_v = branch_detector_field(V, a_det, H_rev, ff_fw, apply_fn)
    af2 = (F_v.real.square() + F_v.imag.square()).sum(dim=1)
    af = af2.clamp_min(0).sqrt()
    fdev = af - fmag
    if fmask is None:
        npix = float(fdev.shape[-2] * fdev.shape[-1])
        err_fmag = fdev.square().sum(dim=(-2, -1)) / npix
    else:
        npix = fmask.sum(dim=(-2, -1)).clamp_min(1.0)
        err_fmag = (fmask * fdev.square()).sum(dim=(-2, -1)) / npix

    if pbound is None:
        # plain modulus replacement: fm = fmag / af, i.e. renorm = 0
        renorm = torch.zeros(B, dtype=af.dtype, device=af.device)
        bypass = torch.zeros(B, dtype=torch.bool, device=af.device)
    else:
        bypass = err_fmag <= pbound
        renorm = torch.sqrt(pbound / err_fmag.clamp_min(1e-30)).clamp(max=1.0)

    scaled = (fmag + fdev * renorm[:, None, None]) / (af + 1e-10)
    fm = scaled if fmask is None else (1.0 - fmask) + fmask * scaled

    corr = (fm - 1.0).unsqueeze(1) * F_v
    dW = (a_c + b * c) * (PO - W) \
        + b * branch_back(corr, H_fwd, Sdiag, ff_bw, apply_fn)
    if bypass.any():
        dW_bypass = (a_c + b * c) * (PO - W)
        dW = torch.where(bypass[:, None, None, None, None], dW_bypass, dW)

    err_exit = dW.abs().square().mean(dim=(-4, -3, -2, -1))

    if compute_ll:
        F_po = branch_detector_field(PO, a_det, H_rev, ff_fw, apply_fn)
        I_po = (F_po.real.square() + F_po.imag.square()).sum(dim=1)
        I_dat = fmag.square()
        LL = (I_po - I_dat).square() / (I_dat + 1.0)
        if fmask is not None:
            LL = fmask * LL
        err_phot = LL.sum(dim=(-2, -1)) / npix
    else:
        err_phot = torch.zeros_like(err_fmag)

    return W + dW, err_fmag, err_phot, err_exit


# ===========================================================================
#  Engine
# ===========================================================================
@register()
class RAARFirstBornTorch(FirstBornTorch):
    __doc__ = __doc__

    SUPPORTED_MODELS = FirstBornTorch.SUPPORTED_MODELS

    def engine_initialize(self):
        super().engine_initialize()
        N = self.p.number_of_slices

        beta = float(self.p.beta)
        self._coeffs = (1.0 - 2.0 * beta, beta, 2.0)

        # Optional container start: slice 0 from the ptypy object (a seed
        # reconstruction or the model's init), later slices stay at the
        # parent's symmetry-broken 1 + noise.
        if str(self.p.object_init).lower().startswith("c"):
            self.ob_g[0] = torch.as_tensor(
                np.ascontiguousarray(self._obs.data[0]),
                dtype=self.cdt, device=self.dev)
            logger.info("object start: slice 0 from container")

        # S B diagonal: sum_j |H_j|^2 in the fft2 domain. H_0 = 1 exactly,
        # so Sdiag >= 1 everywhere and the division is safe.
        self._Sdiag = (self.H_fwd[:, 0, 0].real.square()
                       + self.H_fwd[:, 0, 0].imag.square()).sum(dim=0)

        self._wave_dev = (torch.device("cpu")
                          if str(self.p.wave_device).lower() == "cpu"
                          else self.dev)
        self.chunk = max(1, int(self.p.chunk_size))

        # Lazy probe-derived caches (position independent: ptypy never
        # shifts the probe, positions live in the object gather).
        self._phi_c = None
        self._a_det_c = None

        # Branch waves: the RAAR iterate.
        self.W = torch.zeros(
            (self.n_pos, N, self.M, self.fny, self.fnx),
            dtype=self.cdt, device=self._wave_dev)
        if str(self.p.exit_init).lower().startswith("m"):
            for s in range(0, self.n_pos, self.chunk):
                idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                                   device=self.dev)
                self.W[idx] = self._model_waves(idx).to(self._wave_dev)

        # Overlap accumulators (object canvas).
        self._num_g = torch.zeros((N, self.ony, self.onx),
                                  dtype=self.cdt, device=self.dev)
        self._den_g = torch.zeros((N, self.ony, self.onx),
                                  dtype=self.rdt, device=self.dev)

        # Previous-probe buffer for the ptypy change metric (starts at zero,
        # so the first probe round reports change = 1 as in projectional).
        self._pr_buf = torch.zeros_like(self.pr_g)

        gib = self.W.numel() * self.W.element_size() / 2**30
        logger.info("RAAR branch waves: %s on %s (%.2f GiB)"
                    % (tuple(self.W.shape), self._wave_dev, gib))

    # -- probe-derived caches ------------------------------------------------
    @property
    def _phi(self):
        if self._phi_c is None:
            P_f = torch.fft.fft2(self.pr_g)
            self._phi_c = torch.fft.ifft2(
                self.H_fwd[:, 0] * P_f.unsqueeze(0))       # (N, M, ny, nx)
        return self._phi_c

    @property
    def _a_det(self):
        if self._a_det_c is None:
            self._a_det_c = self._apply(self._ff_fw, self.pr_g)
        return self._a_det_c

    def _drop_probe_caches(self):
        self._phi_c = None
        self._a_det_c = None

    # -- helpers ---------------------------------------------------------------
    def _gather_g(self, idx):
        """g_j patches for a chunk: (B, N, 1, ny, nx)."""
        B = idx.numel()
        flat = self._flat(idx)
        O = torch.stack([self._gather(self.ob_g[j], flat, B)
                         for j in range(self.p.number_of_slices)])
        g = self._model_object(O) - 1.0                     # (N, B, ny, nx)
        return g.transpose(0, 1).unsqueeze(2)

    def _model_waves(self, idx):
        """P_O image for a chunk: w_j = g_j phi_j, (B, N, M, ny, nx)."""
        return self._gather_g(idx) * self._phi.unsqueeze(0)

    # -- prepare ---------------------------------------------------------------
    def engine_prepare(self):
        super().engine_prepare()
        pb = np.empty(self.n_pos)
        explicit = self.p.fourier_power_bound
        for i, n in enumerate(self.view_names):
            s = self.di.views[n].storage
            pb[i] = (.25 * self.p.fourier_relax_factor**2 * s.pbound_stub
                     if explicit is None else explicit)
        self.pbound = torch.as_tensor(pb, dtype=self.rdt, device=self.dev)
        mean_power = np.mean([s.mean_power
                              for s in self.di.storages.values()])
        self._cfact_ob = float(self.p.object_inertia) * float(mean_power)
        self._cfact_pr = (float(self.p.probe_inertia)
                          * self.n_pos / max(1, self.M))

    # -- the iteration -----------------------------------------------------------
    def engine_iterate(self, num=1):
        for _ in range(num):
            self.pre_iteration()
            err = self._fourier_update()
            self._overlap_update()
            self.curiter += 1
            if self._plot_int > 0 and self.curiter % self._plot_int == 0:
                self._plot_progress()
        error_dct = {name: err[i] for i, name in enumerate(self.view_names)}
        self._err_history.append(err.mean(axis=0))
        return error_dct

    def _fourier_update(self):
        err = np.zeros((self.n_pos, 3))
        for s in range(0, self.n_pos, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                               device=self.dev)
            Wc = self.W[idx.to(self._wave_dev)].to(self.dev)
            PO = self._model_waves(idx)
            fmag = self.amp_dat[idx]
            fmask = (self.maskb[idx].to(self.rdt)
                     if self.maskb is not None else None)
            Wn, e0, e1, e2 = raar_fourier_chunk(
                Wc, PO, self._a_det, fmag, fmask, self.pbound[idx],
                self._coeffs, self.H_fwd, self.H_rev, self._Sdiag,
                self._ff_fw, self._ff_bw, torch_helpers.apply_torch,
                compute_ll=bool(self.p.compute_log_likelihood))
            self.W[idx.to(self._wave_dev)] = Wn.to(self._wave_dev)
            err[s:s + idx.numel(), 0] = e0.double().cpu().numpy()
            err[s:s + idx.numel(), 1] = e1.double().cpu().numpy()
            err[s:s + idx.numel(), 2] = e2.double().cpu().numpy()
        return err

    def _overlap_update(self):
        do_probe = self.curiter >= self.p.probe_update_start
        for inner in range(max(1, int(self.p.overlap_max_iterations))):
            if bool(self.p.update_object_first) or inner > 0 or not do_probe:
                self._object_update()
            if not do_probe:
                break
            change = self._probe_update()
            logger.debug("overlap %d: probe change %.4f" % (inner, change))
            if change < float(self.p.overlap_converge_factor):
                break

    def _object_update(self):
        N = self.p.number_of_slices
        cfact = self._cfact_ob
        g_old = self.ob_g - 1.0
        std = self.p.obj_smooth_std
        if std:
            g_old = torch_helpers.complex_gaussian_filter(g_old, float(std))
        self._num_g[:] = cfact * g_old
        self._den_g.fill_(cfact)
        phi = self._phi                                     # (N, M, ny, nx)
        Kj = (phi.real.square() + phi.imag.square()).sum(dim=1)  # (N, ny, nx)
        for s in range(0, self.n_pos, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                               device=self.dev)
            B = idx.numel()
            flat = self._flat(idx)
            Wc = self.W[idx.to(self._wave_dev)].to(self.dev)
            for j in range(N):
                self._scatter(self._num_g[j], flat,
                              (phi[j].conj().unsqueeze(0) * Wc[:, j]).sum(1))
                self._scatter(self._den_g[j], flat,
                              Kj[j].expand(B, self.fny, self.fnx))
        g_new = self._num_g / self._den_g.clamp_min(
            torch.finfo(self.rdt).eps)
        # Chord parametrisation only: the stored slices are transmissions
        # O = 1 + g. A tangent subclass would need to override this solve.
        self.ob_g = 1.0 + g_new
        if self.p.clip_object is not None:
            self.ob_g = torch_helpers.clip_amplitude(
                self.ob_g, self.p.clip_object)

    def _probe_update(self):
        N = self.p.number_of_slices
        numP = self._cfact_pr * self.pr_g.clone()
        KP = torch.full((self.fny, self.fnx), self._cfact_pr,
                        dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.n_pos),
                               device=self.dev)
            Wc = self.W[idx.to(self._wave_dev)].to(self.dev)
            g = self._gather_g(idx)                        # (B, N, 1, ny, nx)
            # entrance-plane exit wave: psi = P + sum_j Prop_{-z_j} w_j
            psi = self.pr_g.unsqueeze(0) + torch.fft.ifft2(
                (self.H_rev[:, 0].unsqueeze(0) * torch.fft.fft2(Wc)).sum(1))
            # A_P^dag psi = psi + sum_j Prop_{-z_j}[ conj(g_j) Prop_{z_j} psi ]
            psi_f = torch.fft.fft2(psi)
            adj = psi.clone()
            for j in range(N):
                r_j = torch.fft.ifft2(self.H_fwd[j, 0] * psi_f)
                adj = adj + torch.fft.ifft2(
                    self.H_rev[j, 0]
                    * torch.fft.fft2(g[:, j].conj() * r_j))
            numP = numP + adj.sum(dim=0)
            O_abs2 = ((1.0 + g).real.square()
                      + (1.0 + g).imag.square())           # (B, N, 1, ny, nx)
            KP = KP + O_abs2.sum(dim=(0, 1, 2))
        new_P = numP / KP.clamp_min(torch.finfo(self.rdt).eps)
        if self.pr_support is not None:
            new_P = new_P * self.pr_support
        change = float(
            torch.linalg.vector_norm(new_P - self._pr_buf)
            / torch.linalg.vector_norm(new_P).clamp_min(
                torch.finfo(self.rdt).eps))
        self._pr_buf = new_P.clone()

        # The RAAR iterate is the physical field psi = P + sum_j Prop_{-z_j}
        # w_j, but the state W stores the probe-offset part only. Keep psi
        # invariant under the probe change (as pod.exit is in ptypy) by
        # distributing -dP over the branches; the Sdiag division makes the
        # compensation exact even inside the evanescent cutoff.
        dP = self.pr_g - new_P                              # (M, ny, nx)
        dW = torch.fft.ifft2(
            self.H_fwd[:, 0] * (torch.fft.fft2(dP) / self._Sdiag
                                ).unsqueeze(0))             # (N, M, ny, nx)
        self.W += dW.unsqueeze(0).to(self._wave_dev)

        self.pr_g = new_P
        self._drop_probe_caches()
        return change

    def engine_finalize(self):
        del self.W
        return super().engine_finalize()
