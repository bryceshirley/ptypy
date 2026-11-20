"""
CUDA implementation of the order-M inter-slice Born update.
"""

import cupy as cp
import numpy as np
from cupyx.scipy.fft import fft2, get_fft_plan, ifft2

_SRC = r"""
#include <cupy/complex.cuh>
typedef complex<float> cf;

extern "C" {

// Index convention: fields carry a probe-mode axis, [N, Mp, P], flat
// i = (j*Mp + m)*P + p. The object and the propagators do not: [N, P],
// flat j*P + p. P = n*n.

// s[j,m,p] = (O[j,p] - 1) * cur[j,m,p]
__global__ void scatter(const cf* O, const cf* cur, cf* s,
                        int NMP, int MP, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NMP) return;
    int j = i / MP, p = i % P;
    s[i] = (O[j * P + p] - cf(1.f, 0.f)) * cur[i];
}

// s[j,m,p] = conj(O[j,p] - 1) * phi[j,m,p]
__global__ void scatter_conj(const cf* O, const cf* phi, cf* s,
                             int NMP, int MP, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NMP) return;
    int j = i / MP, p = i % P;
    s[i] = conj(O[j * P + p] - cf(1.f, 0.f)) * phi[i];
}

// forward H-weighted exclusive cumsum over slices, one thread per
// (mode, pixel):
//   curf[j,m,p] = Hf[j,p] * sum_{k<j} Hr[k,p]*Sf[k,m,p]
//   total[m,p] += sum_j Hr[j,p]*Sf[j,m,p]      (the order's readout)
__global__ void cumsum_fwd(const cf* Sf, const cf* Hf, const cf* Hr,
                           cf* curf, cf* total, int N, int Mp, int P) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Mp * P) return;
    int m = t / P, p = t % P, MP = Mp * P;
    cf acc(0.f, 0.f);
    for (int j = 0; j < N; ++j) {
        curf[j * MP + m * P + p] = Hf[j * P + p] * acc;
        acc += Hr[j * P + p] * Sf[j * MP + m * P + p];
    }
    total[t] += acc;
}

// reversed exclusive cumsum (the adjoint G^dag), plus the probe readout
__global__ void cumsum_rev(const cf* Af, const cf* Hf, const cf* Hr,
                           cf* uf, cf* tP, int N, int Mp, int P) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Mp * P) return;
    int m = t / P, p = t % P, MP = Mp * P;
    cf acc(0.f, 0.f);
    for (int j = N - 1; j >= 0; --j) {
        uf[j * MP + m * P + p] = Hf[j * P + p] * acc;
        acc += Hr[j * P + p] * Af[j * MP + m * P + p];
    }
    tP[t] += acc;
}

// detector, incoherent over probe modes:
//   I = g2 * sum_m |total[m,p]|^2 / P ;  fac = sqrt(Idat/max(I,eps)) - 1
//   Rf[m,p] = g2 * total[m,p] * fac
// g2 = |A|^2 is the centred background gauge. Only its MODULUS enters:
// the forward multiplies psi by A and the adjoint multiplies the
// residual by conj(A), so the phase cancels exactly.
__global__ void detector(const cf* total, const float* Idat, float* Imod,
                         cf* Rf, int Mp, int P, float invP,
                         const float* g2ptr, const float* mask) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float g2 = g2ptr[0];
    float I = 0.f;
    for (int m = 0; m < Mp; ++m) {
        cf t = total[m * P + p];
        I += t.real() * t.real() + t.imag() * t.imag();
    }
    I *= invP * g2;
    Imod[p] = I;
    // a zero mask entry drops the pixel from the update, as in the cupy
    // core's where(mask > 0, fac, 0). This has to replace rather than
    // multiply: masked pixels can hold non-finite intensities, and
    // NaN * 0 is still NaN.
    float f = 0.f;
    if (mask[p] > 0.f) f = (sqrtf(Idat[p] / fmaxf(I, 1e-12f)) - 1.f) * g2;
    for (int m = 0; m < Mp; ++m) Rf[m * P + p] = total[m * P + p] * f;
}

// grad[j,p] += sum_m conj(a[j,m,p]) * b[j,m,p]
__global__ void pair_accum(const cf* a, const cf* b, cf* grad,
                           int NP, int Mp, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;
    int j = i / P, p = i % P, MP = Mp * P;
    cf acc(0.f, 0.f);
    for (int m = 0; m < Mp; ++m) {
        int k = j * MP + m * P + p;
        acc += conj(a[k]) * b[k];
    }
    grad[i] += acc;
}

// w[j,m,p] += a[j,m,p] ; on the last order K[j,p] = sum_m |w[j,m,p]|^2
__global__ void winc_accum(const cf* a, cf* w, float* K,
                           int NP, int Mp, int P, int last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;
    int j = i / P, p = i % P, MP = Mp * P;
    float acc = 0.f;
    for (int m = 0; m < Mp; ++m) {
        int k = j * MP + m * P + p;
        cf v = w[k] + a[k];
        w[k] = v;
        acc += v.real() * v.real() + v.imag() * v.imag();
    }
    if (last) K[i] = acc;
}

// grad_P[m,p] = order-0 residual + the summed scattered adjoint
__global__ void probe_grad(const cf* rI, const cf* tI, cf* gP, int MP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MP) gP[i] = rI[i] + tI[i];
}

// c[m,p] = a[p] * b[m,p]   (the chirp has no mode axis)
__global__ void mult(const cf* a, const cf* b, cf* c, int MP, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MP) c[i] = a[i % P] * b[i];
}

// c[m,p] = conj(a[p]) * b[m,p]
__global__ void mult_conj(const cf* a, const cf* b, cf* c, int MP, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MP) c[i] = conj(a[i % P]) * b[i];
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


// Minimiser of the quartic A0 + A1 g + A2 g^2 + A3 g^3 + A4 g^4 along the
// search line. The stationary points are the real roots of the derivative
// cubic; the quartic is evaluated at each and the lowest is taken. A4 > 0
// whenever the direction reaches the detector, so a global minimum exists.
__device__ double ls_quartic(const double* A, double g) {
    return A[0] + g*(A[1] + g*(A[2] + g*(A[3] + g*A[4])));
}

__global__ void ls_gamma(const double* A, float* out) {
    if (threadIdx.x || blockIdx.x) return;
    double a3 = 4.0*A[4], a2 = 3.0*A[3], a1 = 2.0*A[2], a0 = A[1];
    if (fabs(a3) < 1e-30) {                 // degenerate: Newton on the
        out[0] = (fabs(a1) > 1e-30) ? (float)(-a0/a1) : 0.f;   // quadratic
        return;
    }
    double a = a2/a3, b = a1/a3, c = a0/a3;
    double p = b - a*a/3.0;
    double q = 2.0*a*a*a/27.0 - a*b/3.0 + c;
    double disc = q*q/4.0 + p*p*p/27.0;
    double roots[3];
    int nr;
    if (disc > 0.0) {                       // one real root
        double s = sqrt(disc);
        roots[0] = cbrt(-q/2.0 + s) + cbrt(-q/2.0 - s) - a/3.0;
        nr = 1;
    } else {                                // three real roots
        double m = 2.0*sqrt(fmax(-p/3.0, 0.0));
        double arg = (m > 1e-300) ? 3.0*q/(p*m) : 0.0;
        arg = fmin(1.0, fmax(-1.0, arg));
        double th = acos(arg)/3.0;
        for (int k = 0; k < 3; ++k)
            roots[k] = m*cos(th - 2.0943951023931953*k) - a/3.0;
        nr = 3;
    }
    double best = 0.0, bestv = 1e300;
    for (int k = 0; k < nr; ++k) {
        double v = ls_quartic(A, roots[k]);
        if (v < bestv) { bestv = v; best = roots[k]; }
    }
    out[0] = (float)best;
}

}  // extern C
"""

_NAMES = ("scatter", "scatter_conj", "cumsum_fwd", "cumsum_rev", "detector",
          "pair_accum", "winc_accum", "probe_grad", "kp_kernel", "mult",
          "mult_conj", "ls_gamma")
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

    One scan position per run(), complex64, with any number of probe
    modes, an optional detector mask, the centred background gauge and
    an optionally padded propagation grid. All buffers are preallocated;
    run() enqueues only kernels and planned FFTs, so it is CUDA-graph
    capturable as-is.

    Per-position inputs are written into the buffers between runs: O
    (the model object patches), Idat, mask and g2. Pr, Hf and Hr are
    kept as the views passed in, so an in-place probe update is seen by
    the next run() without a copy.

    The far-field convention is the orthonormal DFT: the detector kernel
    carries the 1/P of |fft_ortho|^2 itself, so the planned transforms
    all run unnormalised.
    """

    def __init__(self, O, P, I_dat, H_fwd, H_rev, M, pre=None,
                 det_shape=None):
        N, n = O.shape[0], int(H_fwd.shape[-1])
        Mp = 1 if P.ndim < 3 else P.shape[0]
        nd = int(O.shape[-1]) if det_shape is None else int(det_shape[-1])
        self.N, self.n, self.nd, self.M, self.Mp = N, n, nd, int(M), int(Mp)
        self.padded = nd != n
        self.off = (n - nd) // 2
        self.ctr = (Ellipsis, slice(self.off, self.off + nd),
                    slice(self.off, self.off + nd))
        self.Pix = n * n
        self.Pd = nd * nd
        self.MP = self.Mp * self.Pix
        self.MPd = self.Mp * self.Pd
        cdt = cp.complex64

        def _view(a, dt, shape):
            a = a.reshape(shape)
            if a.dtype != dt or not a.flags.c_contiguous:
                a = cp.ascontiguousarray(a, dtype=dt)
            return a

        self.O = _view(O, cdt, (N, nd, nd))
        self.Pr = _view(P, cdt, (Mp, nd, nd))
        self.Idat = _view(I_dat, cp.float32, (nd, nd))
        self.Hf = _view(H_fwd, cdt, (N, n, n))
        self.Hr = _view(H_rev, cdt, (N, n, n))
        self.pre = None if pre is None else _view(pre, cdt, (nd, nd))
        self.Hlast = self.Hf[N - 1]
        self.Hlast_c = self.Hr[N - 1]
        # |A|^2 of the centred background gauge; the phase cancels
        # between the forward and the adjoint, so only this is needed.
        # Device-resident, so a per-position gauge costs no host sync.
        self.g2 = cp.ones((1,), dtype=cp.float32)

        self._k = _get_kernels()

        z = lambda *s: cp.zeros(s, dtype=cdt)  # noqa: E731
        # propagation grid
        self.Opad = cp.ones((N, n, n), dtype=cdt)
        self.Prpad = z(Mp, n, n)
        self.P_f = z(Mp, n, n)
        self.cur = [z(N, Mp, n, n) for _ in range(self.M)]
        self.phi = [z(N, Mp, n, n) for _ in range(self.M)]
        self.s = z(N, Mp, n, n)
        self.buf = z(N, Mp, n, n)
        self.total = z(Mp, n, n)
        self.Rf = z(Mp, n, n)
        self.tP = z(Mp, n, n)
        self.psi = z(Mp, n, n)
        self.rpad = z(Mp, n, n)
        self.gpad = z(N, n, n)
        self.w = z(N, Mp, n, n)
        self.Kpad = cp.zeros((N, n, n), dtype=cp.float32)
        # detector frame
        self.rI = z(Mp, nd, nd)
        self.tPI = z(Mp, nd, nd)
        self.grad = z(N, nd, nd)
        self.gP = z(Mp, nd, nd)
        self.e1 = z(Mp, nd, nd)
        self.e2 = z(Mp, nd, nd)
        self.mask = cp.ones((nd, nd), dtype=cp.float32)
        self.Imod = cp.zeros((nd, nd), dtype=cp.float32)
        self.K = cp.zeros((N, nd, nd), dtype=cp.float32)
        self.KP = cp.zeros((nd, nd), dtype=cp.float32)
        # line search: the detector-frame spectrum of the current model
        # and of a search direction, plus the quartic and its minimiser
        self.Odir = cp.ones((N, n, n), dtype=cdt)
        self.Dfld = z(Mp, nd, nd)
        self.Dd = z(Mp, nd, nd)
        self.A = cp.zeros((5,), dtype=cp.float64)
        self.gamma = cp.zeros((1,), dtype=cp.float32)

        # one plan per array shape the transforms run on
        self.plan_grid = get_fft_plan(self.P_f, axes=(-2, -1))    # Mp,n,n
        self.plan_slices = get_fft_plan(self.s, axes=(-2, -1))    # N,Mp,n,n
        self.plan_frame = get_fft_plan(self.rI, axes=(-2, -1))    # Mp,nd,nd

    def _launch(self, name, size, *args):
        self._k[name](((size + _TPB - 1) // _TPB,), (_TPB,), args)

    def _fft(self, plan, x, out, inverse=False):
        """Unnormalised batched transform of x into out, under the plan
        matching their shape."""
        with plan:
            out[...] = (ifft2(x, axes=(-2, -1), norm="backward") if inverse
                        else fft2(x, axes=(-2, -1), norm="backward"))

    def _forward(self, O, Pr, incident=True):
        """Order-M forward model into self.total. incident=False drops the
        unscattered term, which is what a search direction needs."""
        N, M, Mp, Pix = self.N, self.M, self.Mp, self.Pix
        MP, NMP = self.MP, self.N * self.MP
        self._fft(self.plan_grid, Pr, self.P_f)
        self.buf[...] = self.Hf[:, None] * self.P_f[None]
        self._fft(self.plan_slices, self.buf, self.cur[0], inverse=True)
        if incident:
            self.total[...] = self.P_f
        else:
            self.total.fill(0)
        for m in range(1, M + 1):
            self._launch("scatter", NMP, O, self.cur[m - 1], self.s,
                         NMP, MP, Pix)
            self._fft(self.plan_slices, self.s, self.buf)
            self._launch("cumsum_fwd", MP, self.buf, self.Hf, self.Hr,
                         self.s, self.total, N, Mp, Pix)
            if m < M:
                self._fft(self.plan_slices, self.s, self.cur[m], inverse=True)

    def _detector_spectrum(self, out):
        """Entrance-referred self.total to the detector-frame spectrum.
        The model intensity is g2/Pd times its squared modulus, summed
        over probe modes."""
        pad, ctr, Pd = self.padded, self.ctr, self.Pd
        if pad:
            # carry to the exit plane before cropping: a crop does not
            # commute with propagation
            self.psi[...] = self.Hlast * self.total
            self._fft(self.plan_grid, self.psi, self.Rf, inverse=True)
            self.e1[...] = self.Rf[ctr]
            if self.pre is None:
                self._fft(self.plan_frame, self.e1, out)
            else:
                self._launch("mult", self.MPd, self.pre, self.e1,
                             self.e2, self.MPd, Pd)
                self._fft(self.plan_frame, self.e2, out)
        elif self.pre is None:
            out[...] = self.total
        else:
            self._fft(self.plan_grid, self.total, self.e1, inverse=True)
            self._launch("mult", self.MP, self.pre, self.e1, self.e2,
                         self.MP, self.Pix)
            self._fft(self.plan_grid, self.e2, out)

    def run(self):
        N, M, Mp = self.N, self.M, self.Mp
        Pix, Pd, MP = self.Pix, self.Pd, self.MP
        NMP, NP = N * MP, N * Pix
        pad, ctr = self.padded, self.ctr

        # onto the propagation grid
        # the object is padded with ones (vacuum outside the patch) so
        # the scatter kernel's O - 1 vanishes there; the probe with zeros
        if pad:
            self.Opad[ctr] = self.O
            self.Prpad.fill(0)
            self.Prpad[ctr] = self.Pr
            O, Pr = self.Opad, self.Prpad
        else:
            O, Pr = self.O, self.Pr

        self._forward(O, Pr, incident=True)

        # detector
        self._detector_spectrum(self.Dfld)
        self._launch("detector", Pd, self.Dfld, self.Idat, self.Imod,
                     self.e2, Mp, Pd, np.float32(1.0 / Pd),
                     self.g2, self.mask)
        # e2 is the frame-domain residual; carry it back to the entrance
        # plane, undoing whatever _detector_spectrum applied
        if self.pre is None:
            self._fft(self.plan_frame, self.e2, self.rI, inverse=True)
        else:
            self._fft(self.plan_frame, self.e2, self.e1, inverse=True)
            self._launch("mult_conj", self.MPd, self.pre, self.e1,
                         self.rI, self.MPd, Pd)
        if pad:
            self.rpad.fill(0)
            self.rpad[ctr] = self.rI
            self._fft(self.plan_grid, self.rpad, self.psi)
            self.Rf[...] = self.Hlast_c * self.psi
        elif self.pre is None:
            self.Rf[...] = self.e2          # already the Fourier residual
        else:
            self._fft(self.plan_grid, self.rI, self.Rf)

        # adjoint orders
        self.buf[...] = self.Hf[:, None] * self.Rf[None]
        self._fft(self.plan_slices, self.buf, self.phi[0], inverse=True)
        self.tP.fill(0)
        for a in range(M):
            last = a == M - 1
            self._launch("scatter_conj", NMP, O, self.phi[a], self.s,
                         NMP, MP, Pix)
            self._fft(self.plan_slices, self.s, self.buf)
            self._launch("cumsum_rev", MP, self.buf, self.Hf, self.Hr,
                         self.s, self.tP, N, Mp, Pix)
            if not last:
                self._fft(self.plan_slices, self.s, self.phi[a + 1],
                          inverse=True)

        # gradients
        grad = self.gpad if pad else self.grad
        K = self.Kpad if pad else self.K
        grad.fill(0)
        self.w.fill(0)
        for l in range(M):
            self.buf.fill(0)
            for a in range(M - l):
                self.buf += self.phi[a]
            self._launch("pair_accum", NP, self.cur[l], self.buf,
                         grad, NP, Mp, Pix)
            self._launch("winc_accum", NP, self.cur[l], self.w, K,
                         NP, Mp, Pix, np.int32(l == M - 1))
        if pad:
            self.grad[...] = self.gpad[ctr]
            self.K[...] = self.Kpad[ctr]

        # probe gradient
        # order 0: the probe reaches the detector through the same
        # exit-plane referencing the forward applies, so its adjoint is
        # the entrance-referred residual, not the frame-plane one
        self._fft(self.plan_grid, self.tP, self.psi, inverse=True)
        self.tPI[...] = self.psi[ctr] if pad else self.psi
        if pad:
            self._fft(self.plan_grid, self.Rf, self.psi, inverse=True)
            self.e1[...] = self.psi[ctr]
            order0 = self.e1
        else:
            order0 = self.rI
        self._launch("probe_grad", self.MPd, order0, self.tPI, self.gP,
                     self.MPd)
        self._launch("kp_kernel", Pd, self.O, self.KP, N, Pd)


    # Line search. The order-M model is affine in a probe step at every
    # order and affine in an object step at M = 1, so the detector field
    # along the search line is psi + gamma*d and the model intensity is
    # exactly quadratic in gamma.

    def direction_object(self, d):
        """Detector-frame spectrum of an object search direction, into
        self.Dd. The scatter kernel forms O - 1, so passing d + 1 makes it
        scatter d itself, and the unscattered term is dropped."""
        self.Odir[...] = 1.0                # d vanishes outside the patch
        self.Odir[self.ctr] = d + 1.0
        if self.padded:
            self.Prpad.fill(0)
            self.Prpad[self.ctr] = self.Pr
            Pr = self.Prpad
        else:
            Pr = self.Pr
        self._forward(self.Odir, Pr, incident=False)
        self._detector_spectrum(self.Dd)

    def direction_probe(self, dP):
        """Detector-frame spectrum of a probe search direction, into
        self.Dd. The model is linear in the probe, so this is the model
        itself evaluated at dP, incident term included."""
        if self.padded:
            self.Opad[self.ctr] = self.O
            self.Prpad.fill(0)
            self.Prpad[self.ctr] = dP
            O, Pr = self.Opad, self.Prpad
        else:
            O, Pr = self.O, dP
        self._forward(O, Pr, incident=True)
        self._detector_spectrum(self.Dd)

    def _line_coeffs(self):
        """c0, c1, c2 of Imod(gamma) - Idat = c0 + gamma*c1 + gamma^2*c2,
        summed over probe modes."""
        k = self.g2[0] / np.float32(self.Pd)
        Imod = k * (self.Dfld.real ** 2 + self.Dfld.imag ** 2).sum(axis=0)
        c0 = Imod - self.Idat
        c1 = 2.0 * k * (self.Dfld.conj() * self.Dd).real.sum(axis=0)
        c2 = k * (self.Dd.real ** 2 + self.Dd.imag ** 2).sum(axis=0)
        return Imod, c0, c1, c2

    def line_search_gaussian(self, W):
        """Exact step for the intensity Gaussian likelihood: the quartic
        in gamma is minimised at a real root of its derivative cubic."""
        _, c0, c1, c2 = self._line_coeffs()
        self.A[...] = cp.stack([
            (W * c0 * c0).sum(),
            2.0 * (W * c0 * c1).sum(),
            (W * (c1 * c1 + 2.0 * c0 * c2)).sum(),
            2.0 * (W * c1 * c2).sum(),
            (W * c2 * c2).sum()]).astype(cp.float64)
        self._launch("ls_gamma", 1, self.A, self.gamma)
        return self.gamma

    def line_search_newton(self, w):
        """One Newton step for the amplitude-l2 fidelity, whose dependence
        on gamma has no closed-form minimiser. Imod(gamma) is still exactly
        quadratic, so the derivatives at gamma = 0 are closed form."""
        Imod, _, c1, c2 = self._line_coeffs()
        Im = cp.maximum(Imod, 1e-12)
        Jp = 1.0 - cp.sqrt(self.Idat / Im)
        Jpp = 0.5 * cp.sqrt(self.Idat) * Im ** -1.5
        L1 = (w * Jp * c1).sum()
        L2 = (w * (Jpp * c1 * c1 + 2.0 * Jp * c2)).sum()
        # a non-positive curvature has no minimiser; fall back to no step
        self.gamma[0] = cp.where(L2 > 0, -L1 / cp.maximum(L2, 1e-30), 0.0)
        return self.gamma

    def advance(self, gamma, d):
        """Take the object step: the detector field moves along the line
        in closed form, so the probe search that follows needs no second
        forward evaluation of the model."""
        self.Dfld += gamma * self.Dd
        self.O += gamma * d


# Storage / output helpers
def pull_storages_to_host(containers):
    """Replace each storage's host data with its device array.

    The multislice engines keep a ``storage.gpu`` beside ``storage.data``;
    before anything is written out the two have to agree, and the device
    side is the live one.
    """
    for container in containers:
        for storage in container.S.values():
            if hasattr(storage, "gpu"):
                storage.data = storage.gpu.get()


def write_slice_file(path, objects, probes=(), extras=None,
                     description="multi-slices result details."):
    """Write the per-slice reconstruction file shared by the inter-slice
    Born engine.

    ``objects`` and ``probes`` are sequences of ptypy containers, one per
    slice; each is stored by container ID then storage ID, so a reader can
    recover the slices in order. ``extras`` is a dict of scalars recorded
    alongside (slice count, thickness, engine-specific settings).

    h5write cannot serialise ptypy's own class types, so UNSUPPORTED is
    switched to 'ignore' for the call and restored afterwards.
    """
    from ptypy import io
    from ptypy.utils import Param
    from ptypy.utils.verbose import logger

    info = Param()
    for key, value in (extras or {}).items():
        info[key] = value
    info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                    for ob in objects}
    if probes:
        info.probes = {pr.ID: {ID: S._to_dict()
                               for ID, S in pr.storages.items()}
                       for pr in probes}

    h5opt = io.h5options["UNSUPPORTED"]
    io.h5options["UNSUPPORTED"] = "ignore"
    logger.info(f"Saving to {path}")
    io.h5write(path, header={"description": description}, content=info)
    io.h5options["UNSUPPORTED"] = h5opt


# Propagation helpers
def cumulative_propagators(geom, slice_thickness, n_slices, pad=1,
                           bandlimit=True):
    """Angular-spectrum kernels H_{z_j} for the CUMULATIVE depth of every
    slice plane, on the (optionally padded) propagation grid.

    The inter-slice Born series refers every branch to the entrance plane,
    so plane j needs the kernel for the whole depth z_j rather than for one
    gap. Built through ptypy's own nearfield geometry, with the ThreePIE
    engines' alias-free band limit, at full precision.

    Returns (H_fwd, pshape): H_fwd is [n_slices, 1, 1, *pshape] complex128,
    broadcast-ready, with plane 0 the identity.
    """
    from ptypy.core import geometry
    from ptypy.custom.multislice_utils import slice_bandlimit
    from ptypy.utils import Param

    energy = float(np.ravel(geom.energy)[0])
    pshape = tuple(int(v) * int(pad) for v in geom.shape)
    z_depths = [
        sum(slice_thickness[:i])
        if isinstance(slice_thickness, (list, tuple))
        else i * slice_thickness
        for i in range(1, n_slices)
    ]

    gpar = Param()
    gpar.energy = energy
    gpar.psize = geom.resolution
    gpar.shape = pshape
    gpar.propagation = "nearfield"

    H_fwd = np.ones((n_slices, 1, 1, *pshape), dtype=np.complex128)
    for i, zi in enumerate(z_depths):
        gpar.distance = zi
        G = geometry.Geo(owner=None, pars=gpar)
        # an ownerless Geo pins its propagator to complex64; rebuild the
        # same kernel at full precision through the public helper
        prop = geometry.get_propagator(G.p, dtype=np.complex128)
        # the geometry kernel is NaN in the evanescent region
        k = np.nan_to_num(np.asarray(prop.kernel, dtype=np.complex128))
        if bandlimit:
            k *= slice_bandlimit(k.shape, geom.resolution, energy, zi)
        H_fwd[i + 1, 0, 0] = k
    return H_fwd, pshape


def farfield_chirp(propagator, shape, tol=1e-5):
    """The entrance-plane chirp of a ptypy far-field propagator, as the
    device array the core's mult/mult_conj kernels expect, or None when
    the geometry needs no chirp.

    ptypy writes its far field as

        fw(w) = post_fft * sc * fft(pre_fft * w)

    with pre_ifft = conj(post_fft) and post_ifft = conj(pre_fft). The
    core keeps only pre_fft, because post_fft has unit modulus and so
    cancels between the forward intensity and the adjoint residual, and
    because the core normalises by 1/P in the detector kernel rather
    than carrying ptypy's sc. Those are the assumptions checked here:
    a geometry that breaks them raises instead of quietly reconstructing
    the wrong thing.
    """
    pre = np.asarray(propagator.pre_fft)
    post = np.asarray(propagator.post_fft)
    pre_i = np.asarray(propagator.pre_ifft)
    post_i = np.asarray(propagator.post_ifft)

    if (np.asarray(propagator.crop_pad) != 0).any():
        raise NotImplementedError(
            "the fused CUDA kernels do not support crop_pad geometries")
    if np.abs(np.abs(post) - 1.0).max() > tol:
        raise NotImplementedError(
            "the fused CUDA kernels need a unimodular far-field "
            "post-filter; |post_fft| departs from 1 by %.2e"
            % np.abs(np.abs(post) - 1.0).max())
    if (np.abs(pre_i - np.conj(post)).max() > tol
            or np.abs(post_i - np.conj(pre)).max() > tol):
        raise NotImplementedError(
            "the fused CUDA kernels need a conjugate-inverse far-field "
            "filter pair")

    if np.abs(pre - 1.0).max() < tol:
        return None
    return cp.asarray(np.broadcast_to(pre, tuple(int(v) for v in shape)),
                      dtype=cp.complex64)
