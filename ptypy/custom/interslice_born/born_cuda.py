"""
Hand-written CUDA implementation of the order-M inter-slice Born update.

Every non-FFT stage is a custom fused CUDA kernel (RawModule, compiled on
first use): the scatter, the H-weighted EXCLUSIVE cumulative sum over
slices (one kernel produces the next order's Fourier field AND the
order's detector readout in a single pass), the detector projection, the
adjoint mirror with the reversed cumulative sum, and the gradient
pairings. The FFTs are batched cuFFT through precomputed plans -- the
same library the cupy core calls; nobody hand-rolls an FFT. All work is
enqueued with zero Python between kernels beyond the order loop, so
run() is CUDA-graph capturable as-is and ISBCupy's whole-update capture
covers it.

Scope: the engines' operating point -- batch 1, a single probe mode,
complex64, no mask -- and the plain series (c_m = 1). ISBCupy
(raw_cuda=True) writes the gathered slice patches and the position's
intensities into the step's input buffers, keeps the probe as an
aliased view so probe updates are visible without a copy, and verifies
the step against the production cupy core on real data at startup
(verify_step), so a kernel regression is a startup error rather than a
silently wrong reconstruction.
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

// c[i] = a[i] * b[i]
__global__ void mult(const cf* a, const cf* b, cf* c, int Np) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Np) c[i] = a[i] * b[i];
}

// c[i] = conj(a[i]) * b[i]
__global__ void mult_conj(const cf* a, const cf* b, cf* c, int Np) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Np) c[i] = conj(a[i]) * b[i];
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

}  // extern C
"""

_NAMES = ("scatter", "scatter_conj", "cumsum_fwd", "cumsum_rev", "detector",
          "pair_accum", "winc_accum", "probe_grad", "kp_kernel", "mult",
          "mult_conj")
_TPB = 256
_kernels = None


def _get_kernels():
    """Compile on first use, so importing the module costs nothing and a
    GPU-less process can still import the package."""
    global _kernels
    if _kernels is None:
        mod = cp.RawModule(code=_SRC, options=("-std=c++17",))
        _kernels = {n: mod.get_function(n) for n in _NAMES}
    return _kernels


class BornCudaStep:
    """One order-M Born update in raw CUDA kernels + planned cuFFT.

    Batch 1, single probe mode, complex64, no mask, plain series. All
    buffers are preallocated; run() enqueues only kernels and planned
    FFTs, so it is CUDA-graph capturable as-is.

    The inputs stay aliased: O and Idat are buffers the engine overwrites
    per scan position; Pr, Hf, Hr are kept as the views passed in, so an
    in-place probe update is seen by the next run() without a copy
    (contiguous complex64/float32 views are required and asserted).
    """

    def __init__(self, O, P, I_dat, H_fwd, H_rev, M, pre=None):
        N, n = O.shape[0], O.shape[-1]
        self.N, self.n, self.M = N, n, int(M)
        self.Pix = n * n
        cdt = cp.complex64

        def _view(a, dt, shape):
            a = a.reshape(shape)
            if a.dtype != dt or not a.flags.c_contiguous:
                a = cp.ascontiguousarray(a, dtype=dt)
            return a

        self.O = _view(O, cdt, (N, n, n))
        self.Pr = _view(P, cdt, (n, n))
        self.Idat = _view(I_dat, cp.float32, (n, n))
        self.Hf = _view(H_fwd, cdt, (N, n, n))
        self.Hr = _view(H_rev, cdt, (N, n, n))
        # the geometry's real-space pre-FFT chirp; its post-FFT partner
        # is unimodular and cancels through the modulus projection, so
        # only pre and conj(pre) appear (None = identity fast path)
        self.pre = None if pre is None else _view(pre, cdt, (n, n))

        self._k = _get_kernels()

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
        self.e1 = z(n, n)
        self.e2 = z(n, n)
        self.Imod = cp.zeros((n, n), dtype=cp.float32)
        self.K = cp.zeros((N, n, n), dtype=cp.float32)
        self.KP = cp.zeros((n, n), dtype=cp.float32)

        self.plan1 = get_fft_plan(self.P_f, axes=(-2, -1))
        self.planN = get_fft_plan(self.s, axes=(-2, -1))

    def _launch(self, name, size, *args):
        self._k[name](((size + _TPB - 1) // _TPB,), (_TPB,), args)

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
            self._launch("scatter", NP, self.O, self.cur[m - 1], self.s, NP)
            self._fftN(self.s, self.buf)
            self._launch("cumsum_fwd", Pix, self.buf, self.Hf, self.Hr,
                         self.s, self.total, N, Pix)
            if m < M:
                self._fftN(self.s, self.cur[m], inverse=True)

        # ---- detector (ortho gauge folded into the 1/Pix scale) ----
        if self.pre is None:
            self._launch("detector", Pix, self.total, self.Idat,
                         self.Imod, self.Rf, Pix, np.float32(1.0 / Pix))
            self._fft1(self.Rf, self.rI, inverse=True)
        else:
            # the engine's far field is post * F_ortho(pre * psi): fac is
            # computed on the chirped spectrum, the correction comes back
            # through conj(pre), and Rf re-enters the entrance-Fourier
            # gauge for the adjoint sweep
            self._fft1(self.total, self.e1, inverse=True)      # psi
            self._launch("mult", Pix, self.pre, self.e1, self.e2, Pix)
            self._fft1(self.e2, self.e1)                        # chirped D
            self._launch("detector", Pix, self.e1, self.Idat,
                         self.Imod, self.e2, Pix, np.float32(1.0 / Pix))
            self._fft1(self.e2, self.e1, inverse=True)
            self._launch("mult_conj", Pix, self.pre, self.e1, self.rI,
                         Pix)                                   # r_field
            self._fft1(self.rI, self.Rf)

        # ---- adjoint orders ----
        self.buf[...] = self.Hf * self.Rf[None]
        self._fftN(self.buf, self.phi[0], inverse=True)
        self.tP.fill(0)
        for a in range(M):
            last = a == M - 1
            self._launch("scatter_conj", NP, self.O, self.phi[a], self.s,
                         NP)
            self._fftN(self.s, self.buf)
            self._launch("cumsum_rev", Pix, self.buf, self.Hf, self.Hr,
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
            self._launch("pair_accum", NP, self.cur[l], self.buf,
                         self.grad, NP)
            self._launch("winc_accum", NP, self.cur[l], self.w, self.K,
                         NP, np.int32(l == M - 1))

        # probe gradient: r_field + ifft(tP); the backward-normalised
        # ifft of the entrance-Fourier fields IS the core's ortho-gauge
        # real-space result, so both scales are exactly 1 (rI, computed
        # in the detector stage, doubles as the core's r_field)
        self._fft1(self.tP, self.tPI, inverse=True)
        self._launch("probe_grad", Pix, self.rI, self.tPI, self.gP, Pix,
                     np.float32(1.0), np.float32(1.0))
        self._launch("kp_kernel", Pix, self.O, self.KP, N, Pix)


def verify_step(step, atol=2e-3, ff_fw=None, ff_bw=None):
    """Run the step on its CURRENT buffer contents and compare every
    output against the production cupy core on the same inputs, through
    the ENGINE's far-field pair when given (bare ortho FFTs otherwise).
    Returns (ok, {name: rel_err}); complex64 tolerance."""
    from .isb_cupy import born_series_step

    ff = ff_fw if ff_fw is not None else \
        (lambda x: cp.fft.fft2(x, norm="ortho"))
    fb = ff_bw if ff_bw is not None else \
        (lambda x: cp.fft.ifft2(x, norm="ortho"))

    O = step.O[:, None]
    g, K, gP, KP, r, Im = born_series_step(
        O, step.Pr[None], step.Idat[None], step.Hf[:, None, None],
        step.Hr[:, None, None], ff, fb, M=step.M)
    step.run()
    rel = lambda a, b: float(cp.abs(a - b).max()  # noqa: E731
                             / cp.abs(b).max().clip(1e-30))
    rep = {
        "I_mod": rel(step.Imod, Im[0]),
        "grad_g": rel(step.grad, g[:, 0]),
        "K_O": rel(step.K, K[:, 0]),
        "grad_P": rel(step.gP, gP[0]),
        "K_P": rel(step.KP, KP),
        "r_field": rel(step.rI, r[0, 0]),
    }
    return all(v < atol for v in rep.values()), rep
