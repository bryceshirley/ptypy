"""
Hand-written CUDA implementation of the order-M inter-slice Born update,
for the backend benchmark (demo 06).

Every non-FFT stage is a custom fused CUDA kernel (RawModule): the
scatter, the H-weighted EXCLUSIVE cumulative sum over slices (one kernel
produces the next order's Fourier field AND the order's detector readout
in a single pass), the detector projection, the adjoint mirror with the
reversed cumulative sum, and the gradient pairings. The FFTs are batched
cuFFT through precomputed plans -- the same library torch and cupy call;
nobody hand-rolls an FFT. All work is enqueued on one stream with zero
Python between kernels beyond the order loop, so CUDA-graph capture of
run() removes what little launch overhead remains.

Scope: the benchmark's operating point -- batch 1, a single probe mode,
complex64, no mask -- and the plain series (c_m = 1). Numerics follow
isb_cupy.born_series_step with the ortho far-field pair; outputs are
verified against it by verify() to complex64 tolerance.
"""

import cupy as cp
import numpy as np
from cupyx.scipy.fft import get_fft_plan

_SRC = r"""
#include <cupy/complex.cuh>
typedef complex<float> cf;

extern "C" {

// s[j,p] = (O[j,p] - 1) * cur[j,p]
__global__ void scatter(const cf* O, const cf* cur, cf* s, int NP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NP) s[i] = (O[i] - cf(1.f, 0.f)) * cur[i];
}

// s[j,p] = conj(O[j,p] - 1) * phi[j,p]
__global__ void scatter_conj(const cf* O, const cf* phi, cf* s, int NP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NP) s[i] = conj(O[i] - cf(1.f, 0.f)) * phi[i];
}

// forward H-weighted exclusive cumsum over slices, one thread per pixel:
//   curf[j,p] = Hf[j,p] * sum_{k<j} Hr[k,p]*Sf[k,p]
//   total[p] += sum_j Hr[j,p]*Sf[j,p]           (the order's readout)
__global__ void cumsum_fwd(const cf* Sf, const cf* Hf, const cf* Hr,
                           cf* curf, cf* total, int N, int P) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    cf acc(0.f, 0.f);
    for (int j = 0; j < N; ++j) {
        curf[j * P + p] = Hf[j * P + p] * acc;
        acc += Hr[j * P + p] * Sf[j * P + p];
    }
    total[p] += acc;
}

// reversed exclusive cumsum (the adjoint G^dag), plus the probe readout:
//   uf[j,p] = Hf[j,p] * sum_{k>j} Hr[k,p]*Af[k,p];  tP[p] += sum_j ...
__global__ void cumsum_rev(const cf* Af, const cf* Hf, const cf* Hr,
                           cf* uf, cf* tP, int N, int P, int do_probe) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    cf acc(0.f, 0.f);
    for (int j = N - 1; j >= 0; --j) {
        uf[j * P + p] = Hf[j * P + p] * acc;
        acc += Hr[j * P + p] * Af[j * P + p];
    }
    if (do_probe) tP[p] += acc;
}

// detector: I = |total|^2/P ; fac = sqrt(Idat/max(I,eps)) - 1 ;
// Rf = total*fac  (raw entrance-Fourier gauge)
__global__ void detector(const cf* total, const float* Idat, float* Imod,
                         cf* Rf, int P, float invP) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    cf t = total[p];
    float I = (t.real() * t.real() + t.imag() * t.imag()) * invP;
    Imod[p] = I;
    float f = sqrtf(Idat[p] / fmaxf(I, 1e-12f)) - 1.f;
    Rf[p] = t * f;
}

// grad[j,p] += conj(a[j,p]) * b[j,p]
__global__ void pair_accum(const cf* a, const cf* b, cf* grad, int NP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NP) grad[i] += conj(a[i]) * b[i];
}

// w[j,p] += a[j,p] ; and on the last order K[j,p] = |w[j,p]|^2
__global__ void winc_accum(const cf* a, cf* w, float* K, int NP, int last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;
    cf v = w[i] + a[i];
    w[i] = v;
    if (last) K[i] = v.real() * v.real() + v.imag() * v.imag();
}

// grad_P[p] = scale_r * Rf_ifft[p] + scale_t * tP_ifft[p]
__global__ void probe_grad(const cf* rI, const cf* tI, cf* gP, int P,
                           float sr, float st) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) gP[p] = rI[p] * sr + tI[p] * st;
}

// K_P[p] = sum_j |O[j,p]|^2
__global__ void kp_kernel(const cf* O, float* KP, int N, int P) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float a = 0.f;
    for (int j = 0; j < N; ++j) {
        cf v = O[j * P + p];
        a += v.real() * v.real() + v.imag() * v.imag();
    }
    KP[p] = a;
}

// c[i] = a[i] * b[i]
__global__ void mult(const cf* a, const cf* b, cf* c, int Np) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Np) c[i] = a[i] * b[i];
}

// c[i] = conj(a[i]) * b[i];  optionally also K[i] = |a[i]|^2
__global__ void mult_conj(const cf* a, const cf* b, cf* c, float* K,
                          int Np, int withK) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Np) return;
    cf av = a[i];
    c[i] = conj(av) * b[i];
    if (withK) K[i] = av.real() * av.real() + av.imag() * av.imag();
}

}  // extern C
"""

_mod = cp.RawModule(code=_SRC, options=("-std=c++17",))
_k = {n: _mod.get_function(n) for n in
      ("scatter", "scatter_conj", "cumsum_fwd", "cumsum_rev", "detector",
       "pair_accum", "winc_accum", "probe_grad", "kp_kernel", "mult",
       "mult_conj")}

_TPB = 256


def _launch(name, size, *args):
    _k[name](((size + _TPB - 1) // _TPB,), (_TPB,), args)


class BornCudaStep:
    """One order-M Born update in raw CUDA kernels + planned cuFFT.

    Batch 1, single probe mode, complex64, no mask, plain series. All
    buffers are preallocated; run() enqueues only kernels and planned
    FFTs, so it is CUDA-graph capturable as-is.
    """

    def __init__(self, O, P, I_dat, H_fwd, H_rev, M):
        N, n = O.shape[0], O.shape[-1]
        self.N, self.n, self.M = N, n, int(M)
        self.Pix = n * n
        cdt = cp.complex64
        self.O = cp.ascontiguousarray(O.reshape(N, n, n), dtype=cdt)
        self.Pr = cp.ascontiguousarray(P.reshape(n, n), dtype=cdt)
        self.Idat = cp.ascontiguousarray(I_dat.reshape(n, n),
                                         dtype=cp.float32)
        self.Hf = cp.ascontiguousarray(H_fwd.reshape(N, n, n), dtype=cdt)
        self.Hr = cp.ascontiguousarray(H_rev.reshape(N, n, n), dtype=cdt)

        z = lambda *s: cp.zeros(s, dtype=cdt)  # noqa: E731
        self.P_f = z(n, n)
        self.cur = [z(N, n, n) for _ in range(self.M)]   # forward orders
        self.phi = [z(N, n, n) for _ in range(self.M)]   # adjoint orders
        self.s = z(N, n, n)
        self.buf = z(N, n, n)
        self.total = z(n, n)
        self.Rf = z(n, n)
        self.rI = z(n, n)
        self.tP = z(n, n)
        self.tPI = z(n, n)
        self.w = z(N, n, n)
        self.grad = z(N, n, n)
        self.gP = z(n, n)
        self.Imod = cp.zeros((n, n), dtype=cp.float32)
        self.K = cp.zeros((N, n, n), dtype=cp.float32)
        self.KP = cp.zeros((n, n), dtype=cp.float32)

        self.plan1 = get_fft_plan(self.P_f, axes=(-2, -1))
        self.planN = get_fft_plan(self.s, axes=(-2, -1))

    def _fftN(self, x, out, inverse=False):
        from cupyx.scipy.fft import fft2, ifft2
        with self.planN:
            out[...] = (ifft2(x, axes=(-2, -1), norm="backward") if inverse
                        else fft2(x, axes=(-2, -1), norm="backward"))

    def _fft1(self, x, out, inverse=False):
        from cupyx.scipy.fft import fft2, ifft2
        with self.plan1:
            out[...] = (ifft2(x, axes=(-2, -1), norm="backward") if inverse
                        else fft2(x, axes=(-2, -1), norm="backward"))

    def run(self):
        N, M, Pix = self.N, self.M, self.Pix
        NP = N * Pix
        # ---- unscattered field and readout accumulator ----
        self._fft1(self.Pr, self.P_f)
        self.buf[...] = self.Hf * self.P_f[None]     # broadcast mult
        self._fftN(self.buf, self.cur[0], inverse=True)
        self.total[...] = self.P_f

        # ---- forward orders ----
        for m in range(1, M + 1):
            _launch("scatter", NP, self.O, self.cur[m - 1], self.s, NP)
            self._fftN(self.s, self.buf)
            _launch("cumsum_fwd", Pix, self.buf, self.Hf, self.Hr,
                    self.s, self.total, N, Pix)
            if m < M:
                self._fftN(self.s, self.cur[m], inverse=True)

        # ---- detector (ortho gauge folded into scales) ----
        _launch("detector", Pix, self.total, self.Idat, self.Imod,
                self.Rf, Pix, np.float32(1.0 / Pix))

        # ---- adjoint orders ----
        self.buf[...] = self.Hf * self.Rf[None]
        self._fftN(self.buf, self.phi[0], inverse=True)
        self.tP.fill(0)
        for a in range(M):
            last = a == M - 1
            _launch("scatter_conj", NP, self.O, self.phi[a], self.s, NP)
            self._fftN(self.s, self.buf)
            _launch("cumsum_rev", Pix, self.buf, self.Hf, self.Hr,
                    self.s, self.tP, N, Pix, np.int32(1))
            if not last:
                self._fftN(self.s, self.phi[a + 1], inverse=True)

        # ---- gradients ----
        self.grad.fill(0)
        self.w.fill(0)
        for l in range(M):
            # sum_{a <= M-1-l} phi[a], reusing buf as the running sum
            self.buf.fill(0)
            for a in range(M - l):
                self.buf += self.phi[a]
            _launch("pair_accum", NP, self.cur[l], self.buf,
                    self.grad, NP)
            _launch("winc_accum", NP, self.cur[l], self.w, self.K, NP,
                    np.int32(l == M - 1))

        # probe gradient: r_field + ifft(tP), in the reference's ortho
        # scaling: r_field = ifft_raw(Rf) * (Pix/ sqrt(Pix)) / Pix etc.
        self._fft1(self.Rf, self.rI, inverse=True)
        self._fft1(self.tP, self.tPI, inverse=True)
        _launch("probe_grad", Pix, self.rI, self.tPI, self.gP, Pix,
                np.float32(1.0), np.float32(1.0))
        _launch("kp_kernel", Pix, self.O, self.KP, N, Pix)


def verify(step, atol=2e-3):
    """Compare against isb_cupy.born_series_step on the same inputs."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from ptypy.custom.interslice_born.isb_cupy import born_series_step

    def ff(x):
        return cp.fft.fft2(x, norm="ortho")

    def fb(x):
        return cp.fft.ifft2(x, norm="ortho")

    O = step.O[:, None]
    g, K, gP, KP, r, Im = born_series_step(
        O, step.Pr[None], step.Idat[None], step.Hf[:, None, None],
        step.Hr[:, None, None], ff, fb, M=step.M)
    step.run()
    rep = {}
    rep["I_mod"] = float(cp.abs(step.Imod - Im[0]).max()
                         / cp.abs(Im).max())
    rep["grad_g"] = float(cp.abs(step.grad - g[:, 0]).max()
                          / cp.abs(g).max())
    rep["K_O"] = float(cp.abs(step.K - K[:, 0]).max() / cp.abs(K).max())
    rep["grad_P"] = float(cp.abs(step.gP - gP[0]).max()
                          / cp.abs(gP).max())
    rep["K_P"] = float(cp.abs(step.KP - KP).max() / cp.abs(KP).max())
    ok = all(v < atol for v in rep.values())
    return ok, rep


class MultisliceCudaStep:
    """One sequential multislice update (3PIE-style) in raw CUDA kernels
    + planned cuFFT: forward sweep storing the incident field on every
    slice, amplitude-l2 modulus projection, adjoint sweep producing the
    per-slice object gradients (paired with |psi_j|^2 preconditioners)
    and the probe gradient. Batch 1, single probe mode, complex64,
    uniform slice spacing. CUDA-graph capturable as-is.
    """

    def __init__(self, O, P, I_dat, Hstep, inverse_gauge=True):
        N, n = O.shape[0], O.shape[-1]
        self.N, self.n, self.Pix = N, n, n * n
        cdt = cp.complex64
        self.O = cp.ascontiguousarray(O.reshape(N, n, n), dtype=cdt)
        self.Pr = cp.ascontiguousarray(P.reshape(n, n), dtype=cdt)
        self.Idat = cp.ascontiguousarray(I_dat.reshape(n, n),
                                         dtype=cp.float32)
        self.Hs = cp.ascontiguousarray(Hstep.reshape(n, n), dtype=cdt)
        self.Hsc = cp.conj(self.Hs).copy()
        z = lambda *sh: cp.zeros(sh, dtype=cdt)  # noqa: E731
        self.psi = z(N, n, n)       # incident field per slice
        self.ex = z(n, n)
        self.b1 = z(n, n)
        self.b2 = z(n, n)
        self.Rf = z(n, n)
        self.t = z(n, n)
        self.grad = z(N, n, n)
        self.gP = z(n, n)
        self.K = cp.zeros((N, n, n), dtype=cp.float32)
        self.KP = cp.zeros((n, n), dtype=cp.float32)
        self.Imod = cp.zeros((n, n), dtype=cp.float32)
        self.plan1 = get_fft_plan(self.ex, axes=(-2, -1))

    def _fft(self, x, out, inverse=False):
        from cupyx.scipy.fft import fft2, ifft2
        with self.plan1:
            out[...] = (ifft2(x, axes=(-2, -1), norm="backward") if inverse
                        else fft2(x, axes=(-2, -1), norm="backward"))

    def run(self):
        N, Pix = self.N, self.Pix
        # ---- forward sweep ----
        self.psi[0] = self.Pr
        for j in range(N):
            _launch("mult", Pix, self.O[j], self.psi[j], self.ex, Pix)
            if j < N - 1:
                self._fft(self.ex, self.b1)
                self.b2[...] = self.Hs * self.b1
                self._fft(self.b2, self.psi[j + 1], inverse=True)
        # ---- detector (far field of the exit wave, ortho scaling) ----
        self._fft(self.ex, self.b1)
        _launch("detector", Pix, self.b1, self.Idat, self.Imod,
                self.Rf, Pix, np.float32(1.0 / Pix))
        # ---- adjoint sweep ----
        self._fft(self.Rf, self.t, inverse=True)   # dL/d(exit), raw scale
        for j in range(N - 1, -1, -1):
            _launch("mult_conj", Pix, self.psi[j], self.t, self.b2,
                    self.K[j], Pix, np.int32(1))
            self.grad[j] = self.b2
            _launch("mult_conj", Pix, self.O[j], self.t, self.b2,
                    self.KP, Pix, np.int32(0))
            if j > 0:
                self._fft(self.b2, self.b1)
                self.b2[...] = self.Hsc * self.b1
                self._fft(self.b2, self.t, inverse=True)
            else:
                self.gP[...] = self.b2
        _launch("kp_kernel", Pix, self.O, self.KP, N, Pix)


def verify_multislice(step, atol=2e-3):
    """Check against a plain cupy transcription of the same update."""
    N, n = step.N, step.n
    O, P, Hs = step.O, step.Pr, step.Hs
    psi = [P]
    for j in range(N - 1):
        psi.append(cp.fft.ifft2(Hs * cp.fft.fft2(O[j] * psi[j])))
    exit_w = O[N - 1] * psi[N - 1]
    F = cp.fft.fft2(exit_w, norm="ortho")
    Im = F.real ** 2 + F.imag ** 2
    fac = cp.sqrt(step.Idat / cp.maximum(Im, 1e-12)) - 1.0
    t = cp.fft.ifft2(F * fac, norm="ortho")
    grads, Ks = [], []
    gP = None
    for j in range(N - 1, -1, -1):
        grads.append(cp.conj(psi[j]) * t)
        Ks.append(psi[j].real ** 2 + psi[j].imag ** 2)
        back = cp.conj(O[j]) * t
        if j > 0:
            t = cp.fft.ifft2(cp.conj(Hs) * cp.fft.fft2(back))
        else:
            gP = back
    grads, Ks = grads[::-1], Ks[::-1]
    step.run()
    rel = lambda a, b: float(cp.abs(a - b).max()  # noqa: E731
                             / cp.abs(b).max().clip(1e-30))
    rep = {"I_mod": rel(step.Imod, Im),
           "grad_g": max(rel(step.grad[j], grads[j]) for j in range(N)),
           "K_O": max(rel(step.K[j], Ks[j]) for j in range(N)),
           "grad_P": rel(step.gP, gP)}
    return all(v < atol for v in rep.values()), rep
