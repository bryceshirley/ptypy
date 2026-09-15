"""
Unit tests for the MLISS pure-math core (mlisstorch.py) -- CPU, no ptypy
containers. Conventions follow the module: far-field pair = orthonormal
fft2/ifft2, model field

    psi = P + sum_j ifft2(H_rev_j * fft2(g_j * phi_j)),
    phi_j = ifft2(H_fwd_j * fft2(P)),

joint direction (h in model space, q probe direction):

    psi(t) = psi0 + t*a + t^2*b,
    a = q + S(h_j phi_j + g_j phiq_j),  b = S(h_j phiq_j).

Run:  pytest test/engine_tests/mlisstorch_test.py -q
"""

import numpy as np
import pytest
import torch

from ptypy.custom.torch.mliss_torch import (
    iss_poly_Ak,
    ml_LL,
    ml_poly_coeffs_euclid,
    ml_poly_coeffs_gaussian,
    ml_poly_coeffs_poisson,
    ml_rho,
    poly_global_min,
)
from ptypy.custom.torch.liss_torch import solve_step_cubic

torch.manual_seed(0)

N, B, M, NY, NX = 2, 2, 1, 32, 32
CD = torch.complex128
RD = torch.float64


def _ff_fw(x):
    return torch.fft.fft2(x, norm="ortho")


def _ff_bw(x):
    return torch.fft.ifft2(x, norm="ortho")


def _rand_c(*shape):
    return (torch.randn(*shape, dtype=RD)
            + 1j * torch.randn(*shape, dtype=RD)).to(CD)


def _setup():
    g = 0.3 * _rand_c(N, B, NY, NX)
    P = _rand_c(M, NY, NX)
    # unit-modulus propagators, slice 0 = identity (engine convention)
    ph = torch.rand(N, NY, NX, dtype=RD)
    H_fwd = torch.exp(2j * np.pi * ph).to(CD)
    H_fwd[0] = 1.0
    H_rev = H_fwd.conj()
    I_dat = torch.rand(B, NY, NX, dtype=RD) * 5.0
    mask = (torch.rand(B, NY, NX) > 0.1).to(RD)
    return g, P, H_fwd, H_rev, I_dat, mask


def _forward(g, P, H_fwd, H_rev):
    """psi and mode-summed intensity for slice-stacked g [N,B,NY,NX]."""
    phi = torch.fft.ifft2(H_fwd.unsqueeze(1) * torch.fft.fft2(P).unsqueeze(0))
    # phi: [N, M, NY, NX]
    acc = 0
    for j in range(N):
        acc = acc + H_rev[j] * torch.fft.fft2(
            g[j].unsqueeze(1) * phi[j].unsqueeze(0))
    psi = P.unsqueeze(0) + torch.fft.ifft2(acc)
    F = _ff_fw(psi)
    u = (F.real.square() + F.imag.square()).sum(1)
    return phi, psi, F, u


def _direct_LL(ml_type, g, P, H_fwd, H_rev, I_dat, mask):
    _, _, _, u = _forward(g, P, H_fwd, H_rev)
    return ml_LL(ml_type, u, I_dat, mask)


# --------------------------------------------------------------------- #
#  (a) gradients vs autograd Wirtinger derivatives                       #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("ml_type", ["gaussian", "poisson", "euclid"])
def test_grad_matches_autograd(ml_type):
    g0, P0, H_fwd, H_rev, I_dat, mask = _setup()

    # ---- autograd reference -------------------------------------------
    g = g0.clone().requires_grad_(True)
    P = P0.clone().requires_grad_(True)
    _, _, _, u = _forward(g, P, H_fwd, H_rev)
    m = mask.double()
    ud, Id = u.double(), I_dat.double()
    if ml_type == "gaussian":
        L = (m / (1.0 + Id) * (ud - Id).square()).sum()
    elif ml_type == "poisson":
        up = ud + 1e-6
        L = (m * (up - Id * torch.log(up))).sum()
    else:
        # euclid regularisation convention: sqrt(u + 1e-12), matching both
        # ml_rho/ml_LL and the line-search coefficients
        amod = (ud.clamp_min(0) + 1e-12).sqrt()
        L = (m * (amod - Id.clamp_min(0).sqrt()).square()).sum()
    L.backward()
    # torch's .grad for a real loss is 2 * dL/dz-bar (the full real-gradient
    # convention); the engine's manual gradient is dL/dz-bar itself, so the
    # exact relation is manual = 0.5 * .grad (verified numerically: the
    # ratio is 0.5 + 0i to machine precision).
    auto_g, auto_P = 0.5 * g.grad, 0.5 * P.grad

    # ---- manual (engine) gradient --------------------------------------
    phi, _, F, u = _forward(g0, P0, H_fwd, H_rev)
    rho = ml_rho(ml_type, u, I_dat, mask)
    r = _ff_bw(F * rho.unsqueeze(1).to(CD))
    r_f = torch.fft.fft2(r)
    man_g = torch.empty_like(g0)
    accP = 0
    for j in range(N):
        r_j = torch.fft.ifft2(H_fwd[j] * r_f)
        man_g[j] = (phi[j].unsqueeze(0).conj() * r_j).sum(1)
        accP = accP + H_rev[j] * torch.fft.fft2(
            g0[j].conj().unsqueeze(1) * r_j)
    man_P = (r + torch.fft.ifft2(accP)).sum(0)

    assert torch.allclose(man_g, auto_g, rtol=1e-8, atol=1e-10), \
        "object gradient mismatch (%s)" % ml_type
    assert torch.allclose(man_P, auto_P, rtol=1e-8, atol=1e-10), \
        "probe gradient mismatch (%s)" % ml_type


# --------------------------------------------------------------------- #
#  (b) polynomial exactness / accuracy along a joint direction           #
# --------------------------------------------------------------------- #
def _poly_setup(with_probe=True):
    g, P, H_fwd, H_rev, I_dat, mask = _setup()
    h = 0.1 * _rand_c(N, B, NY, NX)
    q = 0.1 * _rand_c(M, NY, NX) if with_probe else None

    phi, _, F0, _ = _forward(g, P, H_fwd, H_rev)
    phiq = (torch.fft.ifft2(H_fwd.unsqueeze(1)
                            * torch.fft.fft2(q).unsqueeze(0))
            if with_probe else None)

    def S(xs):
        acc = 0
        for j, x in enumerate(xs):
            acc = acc + H_rev[j] * torch.fft.fft2(x)
        return torch.fft.ifft2(acc)

    if with_probe:
        a = q.unsqueeze(0) + S(
            h[j].unsqueeze(1) * phi[j].unsqueeze(0)
            + g[j].unsqueeze(1) * phiq[j].unsqueeze(0) for j in range(N))
        b = S(h[j].unsqueeze(1) * phiq[j].unsqueeze(0) for j in range(N))
        A = _ff_fw(a)
        Bf = _ff_fw(b)
    else:
        a = S(h[j].unsqueeze(1) * phi[j].unsqueeze(0) for j in range(N))
        A = _ff_fw(a)
        Bf = None
    Ak = iss_poly_Ak(F0, A, Bf)
    return g, P, h, q, H_fwd, H_rev, I_dat, mask, Ak


def test_gaussian_polynomial_exact():
    g, P, h, q, H_fwd, H_rev, I_dat, mask, Ak = _poly_setup(with_probe=True)
    Bc = ml_poly_coeffs_gaussian(*Ak, I_dat, mask)
    for t in [-0.7, -0.2, 0.0, 0.1, 0.35, 0.8, 1.5]:
        LL_d = _direct_LL("gaussian", g + t * h, P + t * q,
                          H_fwd, H_rev, I_dat, mask)
        LL_p = float(np.polyval(np.flip(Bc), t))
        assert abs(LL_p - LL_d) <= 1e-9 * max(abs(LL_d), 1.0), \
            "gaussian polynomial not exact at t=%.2f: %r vs %r" % (t, LL_p,
                                                                   LL_d)


@pytest.mark.parametrize("ml_type", ["poisson", "euclid"])
def test_taylor_models_local_and_improving(ml_type):
    g, P, h, q, H_fwd, H_rev, I_dat, mask, Ak = _poly_setup(with_probe=True)
    if ml_type == "poisson":
        m = mask.double()
        llbase = float((m * torch.lgamma(I_dat.double() + 1.0)).sum())
        Bc = ml_poly_coeffs_poisson(*Ak, I_dat, mask, llbase)

        def direct(t):
            return llbase + _direct_LL("poisson", g + t * h, P + t * q,
                                       H_fwd, H_rev, I_dat, mask)
    else:
        Bc = ml_poly_coeffs_euclid(*Ak, I_dat, mask)

        def direct(t):
            return _direct_LL("euclid", g + t * h, P + t * q,
                              H_fwd, H_rev, I_dat, mask)

    # B[0] is the exact value at t = 0
    assert abs(Bc[0] - direct(0.0)) <= 1e-8 * max(abs(Bc[0]), 1.0)
    # B[1] is the exact first derivative at t = 0 (finite differences)
    eps = 1e-6
    d_fd = (direct(eps) - direct(-eps)) / (2 * eps)
    assert abs(Bc[1] - d_fd) <= 1e-4 * max(abs(d_fd), 1.0), \
        "%s: B[1]=%r vs FD %r" % (ml_type, Bc[1], d_fd)
    # the polynomial's chosen root improves the TRUE objective
    t, degen = poly_global_min(Bc / max(abs(Bc[0]), 1.0), 1.0, 0.0)
    assert not degen
    assert direct(t) < direct(0.0), \
        "%s: root t=%r does not improve the true LL" % (ml_type, t)


# --------------------------------------------------------------------- #
#  (c) frozen-probe gaussian quartic == solve_step_cubic invariant       #
# --------------------------------------------------------------------- #
def test_frozen_probe_matches_solve_step_cubic():
    g, P, h, q, H_fwd, H_rev, I_dat, mask, Ak = _poly_setup(with_probe=False)
    A0, A1, A2, A3, A4 = Ak
    assert A3 is None and A4 is None
    Bc = ml_poly_coeffs_gaussian(A0, A1, A2, A3, A4, I_dat, mask)
    # frozen probe: the INTENSITY polynomial is quadratic (A3 = A4 = 0), so
    # the gaussian LL is a quartic -- B[3] = 2*w*A1*A2 and B[4] = w*A2^2 are
    # legitimately nonzero, only B[5..8] vanish.
    assert np.all(Bc[5:] == 0.0) and Bc[4] != 0.0

    scale = max(abs(Bc[0]), 1.0)
    t_poly, degen = poly_global_min(Bc / scale, 1.0, 0.0)
    assert not degen

    # LISSTorch identification: e = A0 - I, v = A1/2, w = A2, omega = m/(1+I)
    e = (A0 - I_dat.double())
    v = A1 / 2.0
    w = A2
    omega = mask.double() / (1.0 + I_dat.double())
    c0 = float((omega * e * v).sum())
    c1 = float((omega * (e * w + 2.0 * v * v)).sum())
    c2 = 3.0 * float((omega * v * w).sum())
    c3 = float((omega * w * w).sum())

    def Q(a):
        r = e + (2.0 * a) * v + (a * a) * w
        return float((omega * r * r).sum())

    t_cubic = solve_step_cubic(c0, c1, c2, c3, Q, fallback=np.nan)
    assert np.isfinite(t_cubic)
    assert abs(t_poly - t_cubic) <= 1e-10 * (1.0 + abs(t_cubic)), \
        "quartic root %r != solve_step_cubic %r" % (t_poly, t_cubic)


# --------------------------------------------------------------------- #
#  (d) poly_global_min behaviour                                         #
# --------------------------------------------------------------------- #
def test_poly_global_min_degenerate_and_damped():
    # flat polynomial -> degenerate
    t, degen = poly_global_min(np.zeros(9), 1.0, 0.0)
    assert degen and t == 0.0
    # simple parabola (t-2)^2 -> root 2, damped and clipped
    Bpar = np.array([4.0, -4.0, 1.0, 0, 0, 0, 0, 0, 0])
    t, degen = poly_global_min(Bpar, 1.0, 0.0)
    assert not degen and abs(t - 2.0) < 1e-12
    t, _ = poly_global_min(Bpar, 0.5, 0.0)
    assert abs(t - 1.0) < 1e-12
    t, _ = poly_global_min(Bpar, 1.0, 0.75)
    assert abs(t - 0.75) < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
