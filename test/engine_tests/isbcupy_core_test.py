"""
Test of the fused CUDA Born-series core,
intersliceborn_utils.BornCudaStep,
against torch-free anchors.

The core is single precision by construction, so the anchors are layered
rather than compared head-on at double precision:

  * step_numpy, a plain-numpy transcription of the same maths written
    against the paper's recursion rather than the CUDA code path, is
    held to central finite differences of the amplitude-l2 cost in
    float64. This is what pins the gradients, and it needs no GPU;
  * BornCudaStep is then held to step_numpy in float32, which is what
    pins the kernels: the index arithmetic, the cumulative sums, the
    gauge, the mask, the probe modes and the padded propagation grid;
  * an independent sequential multislice simulation, which the series
    must reproduce at M = N (nilpotent termination), pins the model.

Together: gradients are right, the kernels implement them, and the
forward model is multislice.

The float32 tolerances are empirical headroom over the observed error,
not a statement about the method's accuracy.

The CUDA half is skipped when CuPy or a GPU is unavailable; the numpy
half always runs.
"""

import unittest

import numpy as np

try:
    import cupy as cp
    from ptypy.custom.intersliceborn_utils import BornCudaStep
    cp.cuda.runtime.getDeviceCount()
    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False
except Exception:  # pragma: no cover - cupy present but no usable device
    HAVE_GPU = False

# Relative tolerances for the float32 core against the float64
# reference. Measured worst case over the cases below is 1.7e-6 on the
# fields and 2.2e-6 on the gradients (both on the padded grid), so these
# leave roughly a decade of headroom for cuFFT and hardware variation.
TOL_FIELD = 1e-5      # I_mod, r_field: one forward pass
TOL_GRAD = 2e-5       # gradients: forward + adjoint + the pairing sums


def make_problem(N=3, Mp=2, n=16, strength=0.4, seed=1, nd=None):
    """Random slices, probe, data and unimodular propagators.

    The propagators are built on an n x n grid; when nd is given the
    object and probe live on the smaller nd x nd detector frame, which
    is the engine's padded-propagation case.
    """
    rng = np.random.default_rng(seed)
    nd = n if nd is None else nd
    g = strength * (rng.standard_normal((N, nd, nd))
                    + 1j * rng.standard_normal((N, nd, nd)))
    O = (1.0 + g).astype(np.complex128)
    P = (rng.standard_normal((Mp, nd, nd))
         + 1j * rng.standard_normal((Mp, nd, nd))).astype(np.complex128)
    f = np.fft.fftfreq(n)
    a2 = f[:, None] ** 2 + f[None, :] ** 2
    kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
    H = np.ones((N, n, n), dtype=np.complex128)
    for j in range(1, N):
        H[j] = np.exp(2j * np.pi * (7.0 * j) * kz) * (a2 <= 1.0)
    I_dat = rng.random((nd, nd)) * 4.0 + 0.1
    return O, P, H, np.conj(H), I_dat


def _embed(O, P, n):
    """Put the object and probe on the n x n propagation grid: the
    object padded with ones (vacuum, so O - 1 vanishes outside the
    patch), the probe with zeros. Returns the arrays and the centre
    slice that crops back to the detector frame."""
    nd = O.shape[-1]
    if nd == n:
        return O, P, (Ellipsis, slice(None), slice(None))
    off = (n - nd) // 2
    ctr = (Ellipsis, slice(off, off + nd), slice(off, off + nd))
    Opad = np.ones(O.shape[:-2] + (n, n), dtype=O.dtype)
    Ppad = np.zeros(P.shape[:-2] + (n, n), dtype=P.dtype)
    Opad[ctr] = O
    Ppad[ctr] = P
    return Opad, Ppad, ctr


def step_numpy(O, P, I_dat, H_fwd, H_rev, M=1, gauge=1.0, mask=None):
    """Plain-numpy order-M Born update, the reference the CUDA core is
    held to. Same contract: grad_* are the descent directions
    -dJ/d(conj .) of the amplitude-l2 fidelity.

    O and P may be smaller than the propagator grid, in which case they
    are embedded as the core embeds them and the detector sees the
    cropped exit-plane frame. gauge is the centred background's
    A = prod_j a_j; only |A|^2 enters.
    """
    n = H_fwd.shape[-1]
    N = O.shape[0]
    M = max(1, min(int(M), N))
    g2 = float(abs(gauge) ** 2)

    Og, Pg, ctr = _embed(O, P, n)
    g = (Og - 1.0)[:, None]                       # [N, 1, n, n]
    Hf, Hr = H_fwd[:, None], H_rev[:, None]       # [N, 1, n, n]

    P_f = np.fft.fft2(Pg)
    orders = [np.fft.ifft2(Hf * P_f[None])]
    total_f = P_f
    for m in range(1, M + 1):
        t = Hr * np.fft.fft2(g * orders[-1])
        cs = np.cumsum(t, axis=0)
        total_f = total_f + cs[-1]
        if m < M:
            u = np.concatenate([np.zeros_like(cs[:1]), cs[:-1]], axis=0)
            orders.append(np.fft.ifft2(Hf * u))

    # The readout is entrance-referred. With a crop it must be carried
    # to the exit plane first, because a crop does not commute with
    # propagation. With no crop the carry is a unimodular Fourier
    # multiplier that the intensity and the adjoint both undo, so the
    # core skips it, and r_field is then the entrance-plane residual.
    cropping = Og.shape[-1] != O.shape[-1]
    carry = H_fwd[N - 1] if cropping else 1.0
    psi = np.fft.ifft2(carry * total_f)[ctr]
    F_psi = np.fft.fft2(psi, norm="ortho")
    I_mod = g2 * (F_psi.real ** 2 + F_psi.imag ** 2).sum(axis=0)
    fac = (np.sqrt(I_dat / np.maximum(I_mod, 1e-12)) - 1.0) * g2
    if mask is not None:
        fac = np.where(mask > 0, fac, 0.0)
    r_field = np.fft.ifft2(F_psi * fac, norm="ortho")

    # back to the entrance plane on the propagation grid
    rpad = np.zeros_like(Pg)
    rpad[ctr] = r_field
    R_f = np.conj(carry) * np.fft.fft2(rpad)

    phis = [np.fft.ifft2(Hf * R_f[None])]
    acc_P_f = np.zeros_like(P_f)
    for a in range(M):
        tA = Hr * np.fft.fft2(np.conj(g) * phis[a])
        acc_P_f = acc_P_f + tA.sum(axis=0)
        if a < M - 1:
            rcs = np.flip(np.cumsum(np.flip(tA, 0), axis=0), 0)
            u = np.concatenate([rcs[1:], np.zeros_like(rcs[:1])], axis=0)
            phis.append(np.fft.ifft2(Hf * u))

    grad_g = np.zeros(Og.shape, dtype=Og.dtype)
    for l in range(M):
        w = sum(phis[a] for a in range(M - l))
        grad_g = grad_g + (np.conj(orders[l]) * w).sum(axis=1)

    w_inc = sum(orders[m] for m in range(M))
    K_O = (w_inc.real ** 2 + w_inc.imag ** 2).sum(axis=1)

    grad_P = np.fft.ifft2(R_f)[ctr] + np.fft.ifft2(acc_P_f)[ctr]
    K_P = (O.real ** 2 + O.imag ** 2).sum(axis=0)
    return grad_g[ctr], K_O[ctr], grad_P, K_P, r_field, I_mod


def multislice_intensity(O, P, H_fwd, gauge=1.0):
    """Independent sequential multislice: no series, no cumsum, just the
    classic sweep, as the identity check's second opinion. Entrance-plane
    gauge (each inter-slice step divides out the previous depth), the
    same convention the series uses."""
    n = H_fwd.shape[-1]
    N = O.shape[0]
    Og, Pg, ctr = _embed(O, P, n)
    psi = Og[0][None] * Pg
    for j in range(1, N):
        Hstep = H_fwd[j] * np.conj(H_fwd[j - 1])
        psi = Og[j][None] * np.fft.ifft2(Hstep * np.fft.fft2(psi))
    # the sweep leaves psi at the exit plane already; just crop
    F_psi = np.fft.fft2(psi[ctr], norm="ortho")
    return abs(gauge) ** 2 * (F_psi.real ** 2
                              + F_psi.imag ** 2).sum(axis=0)


# The numpy reference against finite differences, no GPU needed
class BornNumpyGradientTest(unittest.TestCase):
    """Pins step_numpy's gradients, in float64, so that the float32
    comparison below inherits a trustworthy reference."""

    @staticmethod
    def _cost(O, P, I_dat, H_fwd, H_rev, M):
        I_mod = step_numpy(O, P, I_dat, H_fwd, H_rev, M=M)[-1]
        a = np.sqrt(np.maximum(I_mod, 1e-12))
        return float(((a - np.sqrt(I_dat)) ** 2).sum())

    def _fd_check(self, which, M, N=3, n=8, Mp=1):
        """grad = -dJ/d(conj z), so for J real the central differences
        along the real and imaginary axes must return -2 Re(grad) and
        -2 Im(grad) (Wirtinger real-pair packing)."""
        O, P, H_fwd, H_rev, I_dat = make_problem(N=N, Mp=Mp, n=n,
                                                 strength=0.3)
        grad_g, _, grad_P, _, _, _ = step_numpy(O, P, I_dat, H_fwd, H_rev,
                                                M=M)
        grad = grad_g if which == "object" else grad_P
        arr = O if which == "object" else P
        rng = np.random.default_rng(7)
        h = 1e-6
        for _ in range(6):
            idx = tuple(int(rng.integers(0, s)) for s in arr.shape)
            for delta, part in ((h, np.real), (1j * h, np.imag)):
                ap, am = arr.copy(), arr.copy()
                ap[idx] += delta
                am[idx] -= delta
                Jp = self._cost(ap if which == "object" else O,
                                ap if which == "probe" else P,
                                I_dat, H_fwd, H_rev, M)
                Jm = self._cost(am if which == "object" else O,
                                am if which == "probe" else P,
                                I_dat, H_fwd, H_rev, M)
                fd = (Jp - Jm) / (2 * h)
                an = -2.0 * part(grad[idx])
                self.assertLess(abs(fd - an), 1e-4 * max(abs(an), 1.0),
                                f"{which} d{part.__name__} at {idx}")

    def test_object_gradient_finite_difference(self):
        self._fd_check("object", M=2)

    def test_object_gradient_finite_difference_full_order(self):
        self._fd_check("object", M=3)

    def test_probe_gradient_finite_difference(self):
        self._fd_check("probe", M=3)

    def test_probe_gradient_finite_difference_multimode(self):
        self._fd_check("probe", M=2, Mp=2)

    def test_full_order_is_multislice(self):
        """At M = N the series' forward intensity must equal a plain
        sequential multislice sweep, a different code path entirely."""
        O, P, H_fwd, H_rev, I_dat = make_problem(N=4, Mp=1, n=16)
        I_mod = step_numpy(O, P, I_dat, H_fwd, H_rev, M=4)[-1]
        I_ref = multislice_intensity(O, P, H_fwd)
        rel = np.abs(I_mod - I_ref).max() / I_ref.max()
        self.assertLess(float(rel), 1e-12)


# The CUDA kernels against the numpy reference
@unittest.skipUnless(HAVE_GPU, "CuPy with a CUDA device is required")
class BornCudaStepTest(unittest.TestCase):

    @staticmethod
    def _run_cuda(O, P, I_dat, H_fwd, H_rev, M, gauge=1.0, mask=None):
        N, nd = O.shape[0], O.shape[-1]
        n = H_fwd.shape[-1]
        step = BornCudaStep(
            cp.zeros((N, nd, nd), dtype=cp.complex64),
            cp.asarray(P, dtype=cp.complex64),
            cp.zeros((nd, nd), dtype=cp.float32),
            cp.asarray(H_fwd, dtype=cp.complex64),
            cp.asarray(H_rev, dtype=cp.complex64),
            M, det_shape=(nd, nd) if nd != n else None)
        step.O[...] = cp.asarray(O, dtype=cp.complex64)
        step.Idat[...] = cp.asarray(I_dat, dtype=cp.float32)
        step.g2[0] = float(abs(gauge) ** 2)
        if mask is not None:
            step.mask[...] = cp.asarray(mask, dtype=cp.float32)
        step.run()
        cp.cuda.Stream.null.synchronize()
        return [cp.asnumpy(a) for a in (step.grad, step.K, step.gP,
                                        step.KP, step.rI, step.Imod)]

    def _compare(self, M, N=3, Mp=2, n=16, nd=None, gauge=1.0,
                 masked=False):
        O, P, H_fwd, H_rev, I_dat = make_problem(N=N, Mp=Mp, n=n, nd=nd)
        mask = None
        if masked:
            rng = np.random.default_rng(3)
            mask = (rng.random(I_dat.shape) > 0.2).astype(np.float64)
            # a masked pixel may carry a non-finite intensity; the core
            # must drop it rather than multiply it by zero
            I_dat = np.where(mask > 0, I_dat, np.inf)

        got = self._run_cuda(O, P, I_dat, H_fwd, H_rev, M,
                             gauge=gauge, mask=mask)
        ref = step_numpy(O, P, I_dat, H_fwd, H_rev, M=M,
                         gauge=gauge, mask=mask)

        names = ["grad_g", "K_O", "grad_P", "K_P", "r_field", "I_mod"]
        tols = [TOL_GRAD, TOL_GRAD, TOL_GRAD, TOL_FIELD,
                TOL_FIELD, TOL_FIELD]
        for name, c, t, tol in zip(names, got, ref, tols):
            scale = max(np.abs(t).max(), 1e-30)
            rel = np.abs(c - t).max() / scale
            self.assertLess(float(rel), tol,
                            f"{name} M={M} rel={rel:.2e}")

    def test_first_order(self):
        self._compare(M=1)

    def test_second_order(self):
        self._compare(M=2)

    def test_full_order(self):
        self._compare(M=3, N=3)

    def test_single_probe_mode(self):
        self._compare(M=2, Mp=1)

    def test_masked_detector(self):
        self._compare(M=2, masked=True)

    def test_centred_gauge(self):
        self._compare(M=2, gauge=0.7 - 0.4j)

    def test_padded_propagation_grid(self):
        self._compare(M=2, n=32, nd=16)

    def test_padded_grid_full_order(self):
        self._compare(M=3, N=3, n=32, nd=16)

    def test_multislice_identity_independent(self):
        """At M = N the core's forward intensity must equal a plain
        sequential multislice sweep, a different code path entirely."""
        O, P, H_fwd, H_rev, I_dat = make_problem(N=4, Mp=1, n=16)
        I_mod = self._run_cuda(O, P, I_dat, H_fwd, H_rev, M=4)[-1]
        I_ref = multislice_intensity(O, P, H_fwd)
        rel = np.abs(I_mod - I_ref).max() / I_ref.max()
        self.assertLess(float(rel), TOL_FIELD)


if __name__ == "__main__":
    unittest.main()
