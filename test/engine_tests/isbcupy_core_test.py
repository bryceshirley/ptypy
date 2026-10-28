"""
Test of the CuPy Born-series core isb_cupy.born_series_step against
torch-free anchors, double precision, ortho FFTs as the far-field pair:

  * a plain-numpy transcription of the same maths, written against the
    paper's recursion rather than the cupy code path, at M = 1, at a
    truncated order with coefficients and at M = N;
  * an INDEPENDENT sequential multislice simulation, which the series
    must reproduce exactly at M = N (nilpotent termination);
  * central finite differences of the amplitude-l2 cost through the
    forward model, which the returned object and probe gradients must
    match (the autograd anchor of the retired torch core, re-derived
    numerically).

Skipped when CuPy or a GPU is unavailable.
"""

import unittest

import numpy as np

try:
    import cupy as cp
    from ptypy.custom.interslice_born.isb_cupy import born_series_step
    cp.cuda.runtime.getDeviceCount()
    HAVE_GPU = True
except Exception:  # noqa: BLE001 - no cupy or no device
    HAVE_GPU = False


def make_problem(N=3, B=2, Mp=2, n=16, strength=0.4, seed=1):
    """Random slices, probe, data and unimodular propagators (float64
    grids, so |H| = 1 to double precision)."""
    rng = np.random.default_rng(seed)
    g = strength * (rng.standard_normal((N, B, n, n))
                    + 1j * rng.standard_normal((N, B, n, n)))
    O = (1.0 + g).astype(np.complex128)
    P = (rng.standard_normal((Mp, n, n))
         + 1j * rng.standard_normal((Mp, n, n))).astype(np.complex128)
    f = np.fft.fftfreq(n)
    a2 = f[:, None] ** 2 + f[None, :] ** 2
    kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
    H = np.ones((N, 1, 1, n, n), dtype=np.complex128)
    for j in range(1, N):
        H[j, 0, 0] = np.exp(2j * np.pi * (7.0 * j) * kz) * (a2 <= 1.0)
    I_dat = rng.random((B, n, n)) * 4.0 + 0.1
    return O, P, H, np.conj(H), I_dat


def step_numpy(O, P, I_dat, H_fwd, H_rev, M=1, coeffs=None,
               calc_probe_grad=True):
    """Plain-numpy order-M Born update, the reference the cupy core is
    held to. Same contract: grad_* are the descent directions
    -dJ/d(conj .) of the amplitude-l2 fidelity."""
    N, B = O.shape[0], O.shape[1]
    M = max(1, min(int(M), N))

    def c(m):
        return 1.0 if coeffs is None else coeffs[m - 1]

    g = (O - 1.0)[:, :, None]
    P_f = np.fft.fft2(P)
    orders = [np.fft.ifft2(H_fwd * P_f[None])]
    total_f = P_f[None]
    for m in range(1, M + 1):
        t = H_rev * np.fft.fft2(g * orders[-1])
        cs = np.cumsum(t, axis=0)
        total_f = total_f + c(m) * cs[-1]
        if m < M:
            u = np.concatenate([np.zeros_like(cs[:1]), cs[:-1]], axis=0)
            orders.append(np.fft.ifft2(H_fwd * u))

    psi = np.fft.ifft2(total_f)
    F_psi = np.fft.fft2(psi, norm="ortho")
    I_mod = (F_psi.real ** 2 + F_psi.imag ** 2).sum(axis=1)
    fac = np.sqrt(I_dat / np.maximum(I_mod, 1e-12)) - 1.0
    r_field = np.fft.ifft2(F_psi * fac[:, None], norm="ortho")

    phis = [np.fft.ifft2(H_fwd * np.fft.fft2(r_field)[None])]
    acc_P_f = None
    for a in range(M):
        if a < M - 1 or calc_probe_grad:
            tA = H_rev * np.fft.fft2(np.conj(g) * phis[a])
            if calc_probe_grad:
                term = tA.sum(axis=0) * np.conj(c(a + 1))
                acc_P_f = term if acc_P_f is None else acc_P_f + term
            if a < M - 1:
                rcs = np.flip(np.cumsum(np.flip(tA, 0), axis=0), 0)
                u = np.concatenate([rcs[1:], np.zeros_like(rcs[:1])],
                                   axis=0)
                phis.append(np.fft.ifft2(H_fwd * u))

    grad_g = None
    for l in range(M):
        w = None
        for a in range(M - l):
            w = phis[a] * np.conj(c(l + a + 1)) if w is None \
                else w + phis[a] * np.conj(c(l + a + 1))
        term = (np.conj(orders[l]) * w).sum(axis=2)
        grad_g = term if grad_g is None else grad_g + term

    w_inc = None
    for m in range(1, M + 1):
        w_inc = orders[m - 1] * c(m) if w_inc is None \
            else w_inc + orders[m - 1] * c(m)
    K_O = (w_inc.real ** 2 + w_inc.imag ** 2).sum(axis=2)
    K_O = np.broadcast_to(K_O, (N, B) + K_O.shape[-2:])

    if calc_probe_grad:
        grad_P = (r_field + np.fft.ifft2(acc_P_f)).sum(axis=0)
        K_P = (O.real ** 2 + O.imag ** 2).sum(axis=(0, 1))
    else:
        grad_P, K_P = None, None
    return grad_g, K_O, grad_P, K_P, r_field, I_mod


def multislice_intensity(O, P, H_fwd):
    """Independent sequential multislice: no series, no cumsum -- the
    classic sweep, as the identity check's second opinion. Entrance-plane
    gauge (each inter-slice step divides out the previous depth), the
    same convention the series uses."""
    N = O.shape[0]
    psi = O[0][:, None] * P[None]
    for j in range(1, N):
        Hstep = H_fwd[j, 0, 0] * np.conj(H_fwd[j - 1, 0, 0])
        psi = O[j][:, None] * np.fft.ifft2(Hstep * np.fft.fft2(psi))
    F_psi = np.fft.fft2(psi, norm="ortho")
    return (F_psi.real ** 2 + F_psi.imag ** 2).sum(axis=1)


@unittest.skipUnless(HAVE_GPU, "CuPy with a CUDA device is required")
class BornCupyCoreTest(unittest.TestCase):

    @staticmethod
    def _run_cupy(O, P, I_dat, H_fwd, H_rev, **kw):
        outs = born_series_step(
            cp.asarray(O), cp.asarray(P), cp.asarray(I_dat),
            cp.asarray(H_fwd), cp.asarray(H_rev),
            lambda x: cp.fft.fft2(x, norm="ortho"),
            lambda x: cp.fft.ifft2(x, norm="ortho"), **kw)
        return [None if o is None else cp.asnumpy(o) for o in outs]

    def _compare(self, M, coeffs=None, calc_probe_grad=True, N=3):
        O, P, H_fwd, H_rev, I_dat = make_problem(N=N)
        outs_c = self._run_cupy(O, P, I_dat, H_fwd, H_rev, M=M,
                                coeffs=coeffs,
                                calc_probe_grad=calc_probe_grad)
        outs_n = step_numpy(O, P, I_dat, H_fwd, H_rev, M=M, coeffs=coeffs,
                            calc_probe_grad=calc_probe_grad)
        names = ["grad_g", "K_O", "grad_P", "K_P", "r_field", "I_mod"]
        for name, c, t in zip(names, outs_c, outs_n):
            if t is None:
                self.assertIsNone(c, name)
                continue
            rel = np.abs(c - t).max() / max(np.abs(t).max(), 1e-30)
            self.assertLess(float(rel), 1e-12, f"{name} M={M}")

    def test_first_order(self):
        self._compare(M=1)

    def test_truncated_with_coefficients(self):
        self._compare(M=2, coeffs=[1.0, 0.7])

    def test_full_order_is_multislice(self):
        self._compare(M=3, N=3)

    def test_no_probe_grad(self):
        self._compare(M=2, calc_probe_grad=False)

    def test_multislice_identity_independent(self):
        """At M = N the series' forward intensity must equal a plain
        sequential multislice sweep -- a different code path entirely."""
        O, P, H_fwd, H_rev, I_dat = make_problem(N=4, B=1)
        I_mod = self._run_cupy(O, P, I_dat, H_fwd, H_rev, M=4,
                               calc_probe_grad=False)[-1]
        I_ref = multislice_intensity(O, P, H_fwd)
        rel = np.abs(I_mod - I_ref).max() / I_ref.max()
        self.assertLess(float(rel), 1e-12)

    # ---- finite-difference anchor for the gradients -------------------
    def _cost(self, O, P, I_dat, H_fwd, H_rev, M, coeffs):
        I_mod = self._run_cupy(O, P, I_dat, H_fwd, H_rev, M=M,
                               coeffs=coeffs, calc_probe_grad=False)[-1]
        a = np.sqrt(np.maximum(I_mod, 1e-12))
        return float(((a - np.sqrt(I_dat)) ** 2).sum())

    def _fd_check(self, which, M, coeffs=None):
        """grad = -dJ/d(conj z), so for J real the central differences
        along the real and imaginary axes must return -2 Re(grad) and
        -2 Im(grad) (Wirtinger real-pair packing)."""
        O, P, H_fwd, H_rev, I_dat = make_problem(N=3, B=1, Mp=1, n=8,
                                                 strength=0.3)
        grad_g, _, grad_P, _, _, _ = self._run_cupy(
            O, P, I_dat, H_fwd, H_rev, M=M, coeffs=coeffs)
        grad = grad_g[:, 0] if which == "object" else grad_P
        arr = O if which == "object" else P
        rng = np.random.default_rng(7)
        h = 1e-6
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            gidx = (idx[0],) + idx[2:] if which == "object" else idx
            for delta, part in ((h, np.real), (1j * h, np.imag)):
                ap = arr.copy()
                am = arr.copy()
                ap[idx] += delta
                am[idx] -= delta
                Jp = self._cost(ap if which == "object" else O,
                                ap if which == "probe" else P,
                                I_dat, H_fwd, H_rev, M, coeffs)
                Jm = self._cost(am if which == "object" else O,
                                am if which == "probe" else P,
                                I_dat, H_fwd, H_rev, M, coeffs)
                fd = (Jp - Jm) / (2 * h)
                an = -2.0 * part(grad[gidx])
                self.assertLess(abs(fd - an), 1e-4 * max(abs(an), 1.0),
                                f"{which} d{part.__name__} at {idx}")

    def test_object_gradient_finite_difference(self):
        self._fd_check("object", M=2, coeffs=[1.0, 0.7])

    def test_probe_gradient_finite_difference(self):
        self._fd_check("probe", M=3)


if __name__ == "__main__":
    unittest.main()
