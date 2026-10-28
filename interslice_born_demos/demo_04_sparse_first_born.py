"""
Demo 4: the sparse strong specimen on which first Born is EXACT.

The paper's overlap property: every term the order-1 truncation discards
contains the product of two layers' perturbations inside one beam cone.
The specimen places one strong (~10 rad) pure-phase particle per layer --
particles lifted from the real NanoMAX scan-434 reconstruction -- so that
scattering cones meet material on (at most) one layer and those products
vanish. The chord model g = O - 1 is then exact at order 1 despite the
strong scattering, while the tangent (slice-wise first Born) model
g = i*phi keeps its per-slice O(phi^2) linearisation error, which
sparsity cannot remove.

This is the production-core reproduction of the ORIGINAL torch-engine
experiment (ISSChordPhaseTorch / ISSTangentTorch, fixed step, campaign
figure figs-p1/sparse_chord_vs_tangent_recon.png):

  * specimen: the native PHASE_SCALE-12 simulation
    sim10rad_20260920_082901_slices-3_seed-7 (~-10 rad per particle),
    taken AS IS -- no unwrapping, rescaling or re-placement;
  * probe: the simulation's own 2-mode probe (truth h5);
  * scan: a uniform raster at the real scan's own pitch (median
    nearest-neighbour spacing of the sim ptyd's 670 positions) over the
    vacuum-padded canvas, so every particle -- corner ones included --
    is covered densely by well-centred illumination. The scan study
    behind this default: the raw spiral leaves a corner residual at the
    under-covered layer-0 particle (chord 8e-3); the padded, complete
    raster removes it (chord 2.3e-4, featureless error map), while the
    tangent model stays pinned at its linearisation floor at every
    coverage level;
  * data: exact multislice through the truth at those positions, via
    the production born_multislice_target (the sim is noise-free);
  * reconstruction: fixed-step Jacobi with the order-1 cupy core,
    alpha = 1.0 as the campaign ran (the campaign engine's 'local'
    preconditioner is replaced by the production 'max' rule, so the
    per-iteration path differs slightly; the fixed point is the same).

Writes (prefix demo_04_*, suffix _ls with --line-search):
  * demo_out/demo_04_sparse_forward_frames.png
  * demo_out/demo_04_sparse_chord_vs_tangent_recon.png  (one shared
    scale, 0 down to the minimum phase; panels use the campaign figure's
    windowed unwrap -- a full-frame unwrap of this specimen is broken by
    background phase vortices)
  * demo_out/demo_04_sparse_convergence.png, _poserr.png

--line-search swaps the fixed steps for the exact affine line search of
paper-1a-parallel-opt (amplitude loss, two Newton iterations, object
then probe); kept for comparison, off by default.

    python demo_04_sparse_first_born.py [--iters 100]
    python demo_04_sparse_first_born.py --synthetic   # no NanoMAX data
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json

import numpy as np
import cupy as cp
from scipy.ndimage import binary_dilation
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from ptypy.custom.interslice_born.isb_cupy import (
    born_detector_orders,
    born_multislice_target,
    born_series_step,
)

EPS = float(np.finfo(np.float64).eps)
# reference-palette pair of the original figure: blue = chord, orange = tangent
C_CHORD, C_TAN = '#2a78d6', '#eb6834'

SIM_DIR = ('/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/'
           'scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/'
           'sim/sim10rad_20260920_082901_slices-3_seed-7')
TRUTH_H5 = os.path.join(
    SIM_DIR, 'sim10rad_20260920_082901_slices-3_seed-7_truth_iter-0000.h5')
PTYD = os.path.join(SIM_DIR, 'sim10rad_20260920_082901_slices-3_seed-7.ptyd')
RESULTS_JSON = os.path.join(SIM_DIR, 'results.json')


# ---------------------------------------------------------------------------
#  Loading the original experiment
# ---------------------------------------------------------------------------
def load_truth(path):
    import h5py
    out = []
    with h5py.File(path, 'r') as f:
        objs = f['content/objects']
        for k in sorted(objs.keys()):
            sk = list(objs[k].keys())[0]
            d = np.array(objs[k][sk]['data'])
            out.append(d[0] if d.ndim == 3 else d)
        pr = np.array(f['content/probe/S00/data'])
    return np.stack(out), pr


def load_scan_positions(path, res):
    """The sim ptyd's scan positions as integer pixel offsets (dlow) on
    the truth canvas -- validated: position span + frame = canvas."""
    import h5py
    with h5py.File(path, 'r') as f:
        pos = np.array(f['chunks/0/positions'])
    pix = (pos - pos.min(axis=0)) / res
    return np.rint(pix).astype(int)


def real_geometry():
    """lam, res and the layer depths of the original experiment."""
    lam, res = 1.5498e-10, 16.87e-9
    z = np.array([0.0, 1580e-6, 3160e-6])
    if os.path.exists(RESULTS_JSON):
        run = json.load(open(RESULTS_JSON)).get('run', {})
        lam = run.get('lam_A', lam * 1e10) * 1e-10
        res = run.get('resolution_nm', res * 1e9) * 1e-9
        z = np.asarray(run.get('z_um', z * 1e6)) * 1e-6
    return lam, res, z


def propagators_at(z, n, lam, res):
    """H_{z_j} for arbitrary depths, same formula as the engines."""
    f = np.fft.fftfreq(n)
    a2 = ((f[:, None] * lam / res) ** 2 + (f[None, :] * lam / res) ** 2)
    kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
    H = np.ones((len(z), 1, 1, n, n), dtype=np.complex128)
    for j, zj in enumerate(z):
        if zj:
            H[j, 0, 0] = np.exp(2j * np.pi * (zj / lam) * kz) * (a2 <= 1.0)
    return H


def masked_truth_phase(truth):
    """Per-layer unwrapped truth phase, the campaign's vortex-safe way
    (analyze_fixedstep_phase.chord_phase_maps): unwrap only on the
    particle plus a dilated collar -- a FULL-FRAME unwrap of this
    specimen is rerouted by background phase vortices and shifts whole
    particles by 2 pi."""
    out = []
    for lay in truth:
        pm = np.abs(lay - 1.0) > 0.5
        roi = binary_dilation(pm, iterations=25)
        wm = np.ma.array(np.angle(lay), mask=~roi)
        u = np.asarray(unwrap_phase(wm))
        ref = roi & (np.abs(lay - 1.0) < 1e-6)
        u -= np.median(u[ref]) if ref.any() else 0.0
        out.append(np.where(roi, u, 0.0))
    return np.stack(out)


# ---------------------------------------------------------------------------
#  Synthetic fallback specimen (same construction, synthetic blobs)
# ---------------------------------------------------------------------------
def particle(n, cy, cx, r_px, depth, rng):
    yy, xx = np.mgrid[:n, :n]
    blob = np.zeros((n, n))
    for _ in range(40):
        dy, dx = (rng.random(2) - 0.5) * r_px
        rr = r_px * (0.15 + 0.25 * rng.random())
        blob -= ((yy - cy - dy) ** 2 + (xx - cx - dx) ** 2 < rr ** 2) * 1.0
    f = np.fft.fft2(blob)
    k = np.fft.fftfreq(n)
    f *= np.exp(-((k[:, None] ** 2 + k[None, :] ** 2)) * (2 * np.pi * 2.0) ** 2)
    blob = np.fft.ifft2(f).real
    return depth * blob / max(-blob.min(), 1e-9)


def make_synthetic(nc, depth, rng):
    centres = [(0.78 * nc, 0.18 * nc), (0.45 * nc, 0.72 * nc),
               (0.14 * nc, 0.68 * nc)]
    return np.stack([particle(nc, cy, cx, 0.10 * nc, depth, rng)
                     for cy, cx in centres])


def focused_probe(n, frac=0.25):
    yy, xx = np.mgrid[:n, :n] - n / 2
    r = np.hypot(yy, xx)
    amp = 0.5 * (1 - np.tanh((r - frac * n / 2) / 2.0))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        amp * np.exp(-0.5 * (r / (frac * n)) ** 2))))


# ---------------------------------------------------------------------------
#  Forward and reconstruction on top of the production cores
# ---------------------------------------------------------------------------
def ff_fw(x):
    return cp.fft.fft2(x, norm="ortho")


def ff_bw(x):
    return cp.fft.ifft2(x, norm="ortho")


def detector_intensity(O_patch, P, H):
    T = born_multislice_target(O_patch, P, H)
    return (cp.abs(T) ** 2).sum(1) / (T.shape[-2] * T.shape[-1])


def order1_field(O_model, P, H):
    D0, D = born_detector_orders(O_model, P, H, cp.conj(H), 1)
    return D0 + D[0]


def order1_intensity(O_model, P, H):
    T = order1_field(O_model, P, H)
    return (cp.abs(T) ** 2).sum(1) / (T.shape[-2] * T.shape[-1])


def eps_table(truth, phase, P, H, positions, n):
    """Order-by-order truncation error, chord and tangent, against exact
    multislice -- the ISSDiagnostics measurement."""
    N = truth.shape[0]
    num = np.zeros(N + 1)
    num_tan = 0.0
    den = 0.0
    Hr = cp.conj(H)
    for r, c in positions:
        Op = truth[:, r:r + n, c:c + n][:, None]
        phip = cp.asarray(phase[:, r:r + n, c:c + n],
                          dtype=P.dtype)[:, None]
        T = born_multislice_target(Op, P, H)
        D0, D = born_detector_orders(Op, P, H, Hr, N)
        acc = D0.copy()
        num[0] += float((cp.abs(acc - T) ** 2).sum())
        for m in range(N):
            acc = acc + D[m]
            num[m + 1] += float((cp.abs(acc - T) ** 2).sum())
        D0t, Dt = born_detector_orders(1.0 + 1j * phip, P, H, Hr, 1)
        num_tan += float((cp.abs(D0t + Dt[0] - T) ** 2).sum())
        den += float((cp.abs(T) ** 2).sum())
    return np.sqrt(num / den), float(np.sqrt(num_tan / den))


def newton_step_size(I0, c1, c2, amp_dat, rdt, iters=2):
    """A couple of Newton iterations on the amplitude-l2 loss along the
    exact quadratic I(gamma) (paper-1a-parallel-opt Eq. newton-step)."""
    gam = cp.zeros((), dtype=rdt)
    for _ in range(iters):
        Ig = cp.maximum(I0 + gam * c1 + gam * gam * c2, 1e-12)
        dI = c1 + 2.0 * gam * c2
        rIg = cp.sqrt(Ig)
        Jp = 1.0 - amp_dat / rIg
        Jpp = 0.5 * amp_dat / (Ig * rIg)
        L1 = (Jp * dI).sum()
        L2 = (Jpp * dI * dI + 2.0 * Jp * c2).sum()
        gam = cp.where(L2 > 0, gam - L1 / cp.maximum(L2, EPS), gam)
    return gam


class Recon:
    """One running reconstruction (chord or tangent), advanced an
    iteration at a time so both can be plotted in lockstep.
    'chord': unknown O, g = O - 1. 'tangent': unknown REAL phi,
    g = i*phi. Fixed Jacobi steps (default, the campaign's rule with the
    production max preconditioner), or the exact affine line search
    with line_search=True."""

    def __init__(self, mode, I, P0, positions, shape, n, H,
                 alpha, beta, line_search, cdt):
        self.mode, self.I, self.positions = mode, I, positions
        self.n, self.H = n, H
        self.Hr = cp.conj(H)
        self.alpha, self.beta, self.ls = alpha, beta, line_search
        self.cdt = cdt
        self.rdt = cp.float64 if cdt == cp.complex128 else cp.float32
        N = H.shape[0]
        self.N = N
        self.ob = (cp.ones if mode == "chord" else cp.zeros)(
            (N,) + tuple(shape), dtype=cdt)
        self.P = P0.copy()
        self.errs = []
        self.pos_err = None
        self.it = 0
        self.Pix = n * n

    def _model(self, patch):
        return patch if self.mode == "chord" else 1.0 + 1j * patch

    def step(self):
        n, N = self.n, self.N
        e0 = 0.0
        pos_err = []
        do_probe = self.it >= 2
        for (r, c), Ii in zip(self.positions, self.I):
            patch = self.ob[:, r:r + n, c:c + n][:, None]
            model = self._model(patch)
            g, K, gP, KP, _, Im = born_series_step(
                model, self.P, Ii[None], self.H, self.Hr, ff_fw, ff_bw,
                M=1, calc_probe_grad=do_probe)
            e_i = float(((cp.sqrt(cp.maximum(Im, 0))
                          - cp.sqrt(Ii)) ** 2).sum()) / Ii.size
            e0 += e_i
            pos_err.append(e_i)
            upd = g / cp.maximum(K.max(axis=(-2, -1), keepdims=True), EPS)
            if self.mode == "tangent":  # dL/ds* = -i dL/dg*, real potential
                upd = (-1j * upd).real.astype(self.cdt)
            if not self.ls:
                self.ob[:, r:r + n, c:c + n] += (self.alpha / N) * upd[:, 0]
                if do_probe:
                    self.P = self.P + (self.beta / N) * gP \
                        / cp.maximum(KP.max(), EPS)
                continue

            # ---- exact affine line search (order 1) -------------------
            dg = self._model(patch + upd) - model
            T = order1_field(model, self.P, self.H)
            Td = born_detector_orders(1.0 + dg, self.P, self.H,
                                      self.Hr, 1)[1][0]
            amp = cp.sqrt(Ii)[None]
            I0 = (cp.abs(T) ** 2).sum(1) / self.Pix
            c1 = 2.0 * (cp.conj(T) * Td).real.sum(1) / self.Pix
            c2 = (cp.abs(Td) ** 2).sum(1) / self.Pix
            gam = newton_step_size(I0, c1, c2, amp, self.rdt)
            self.ob[:, r:r + n, c:c + n] += gam * upd[:, 0]
            if do_probe:
                dP = gP / cp.maximum(KP.max(), EPS)
                TdP = order1_field(model + gam * dg, dP, self.H)
                Tu = T + gam * Td
                I0p = (cp.abs(Tu) ** 2).sum(1) / self.Pix
                c1p = 2.0 * (cp.conj(Tu) * TdP).real.sum(1) / self.Pix
                c2p = (cp.abs(TdP) ** 2).sum(1) / self.Pix
                bet = newton_step_size(I0p, c1p, c2p, amp, self.rdt)
                self.P = self.P + bet * dP
        self.errs.append(e0 / len(self.positions))
        self.pos_err = np.asarray(pos_err)
        self.it += 1


# ---------------------------------------------------------------------------
#  Figures: the campaign figure's display recipe
# ---------------------------------------------------------------------------
def display_window(truth):
    """Bounding box of all structure plus padding; the unwrap runs inside
    it, where there is no background vortex to reroute it."""
    devmax = np.abs(truth - 1.0).max(axis=0) > 1e-6
    ys, xs = np.nonzero(devmax)
    pad = 60
    return (slice(max(ys.min() - pad, 0), ys.max() + pad),
            slice(max(xs.min() - pad, 0), xs.max() + pad))


def recon_figure(truth, win, rec_c, rec_t, step_label, path,
                 title_extra=""):
    ob_c = cp.asnumpy(rec_c.ob)
    ob_t = cp.asnumpy(rec_t.ob)
    n_lay = truth.shape[0]

    def panel(row, L):
        if row == 0:
            p = np.asarray(unwrap_phase(np.angle(truth[L][win])))
        elif row == 1:
            p = np.asarray(unwrap_phase(np.angle(
                ob_c[L][win] / np.maximum(np.abs(ob_c[L][win]), 1e-9))))
        else:
            p = ob_t[L][win].real.copy()
        bg = np.abs(truth[L][win] - 1.0) < 1e-6
        if bg.any():
            p -= np.median(p[bg])
        return p

    labels = ["ground truth",
              r"chord $\delta O = e^{i\theta}-1$",
              r"tangent $\delta O = i\theta$"]
    ph = [[panel(r, L) for L in range(n_lay)] for r in range(3)]
    vmin = min(float(p.min()) for row in ph for p in row)
    fig = plt.figure(figsize=(3.6 * n_lay, 10.5), dpi=150)
    gs = GridSpec(3, n_lay, figure=fig)
    fig.suptitle(f"Sparse strong pure-phase specimen ({n_lay} layers): "
                 f"inter-slice first Born, chord vs tangent, {step_label}"
                 + title_extra + f" -- one scale, 0 to {vmin:.2f} rad")
    for i in range(3):
        for j in range(n_lay):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(ph[i][j], cmap="Greys_r", vmin=vmin, vmax=0)
            if i == 0:
                ax.set_title(f"layer {j}")
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9)
            ax.text(0.03, 0.04, f"min {ph[i][j].min():.2f} rad",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc='white', alpha=0.6, ec='none'))
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def convergence_figure(rec_c, rec_t, step_label, path):
    err_c = np.asarray(rec_c.errs)
    err_t = np.asarray(rec_t.errs)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    fig.suptitle(f"chord vs tangent convergence, {step_label}")
    it_ax = np.arange(1, err_c.size + 1)
    ax.semilogy(it_ax, err_c, color=C_CHORD,
                label=rf"chord ($\theta$ real), {step_label}")
    ax.semilogy(it_ax, err_t, color=C_TAN,
                label=rf"tangent ($\theta$ real), {step_label}")
    for e, col in ((err_c, C_CHORD), (err_t, C_TAN)):
        ax.annotate(f"{e[-1]:.3g}", (it_ax[-1], e[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=col)
    ax.set_xlabel("iteration")
    ax.set_ylabel("detector (Fourier) error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def poserr_figure(rec_c, rec_t, positions, shape, n, step_label, path):
    """The error at every scan point, drawn at the window centres."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150,
                             constrained_layout=True)
    fig.suptitle(f"per-position detector error, {step_label} "
                 f"(iteration {rec_c.it})")
    cy = np.array([r + n / 2 for r, c in positions])
    cx = np.array([c + n / 2 for r, c in positions])
    both = np.concatenate([rec_c.pos_err, rec_t.pos_err])
    norm = plt.matplotlib.colors.LogNorm(vmin=both.min(), vmax=both.max())
    for ax, rec, name in ((axes[0], rec_c, "chord"),
                          (axes[1], rec_t, "tangent")):
        sc = ax.scatter(cx, cy, c=rec.pos_err, s=14, cmap="viridis",
                        norm=norm)
        ax.set_aspect(1)
        ax.invert_yaxis()
        ax.set_title(f"{name} (mean {rec.pos_err.mean():.3g})", fontsize=10)
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=20,
                    help="the campaign's chord run is essentially "
                         "converged by ~20 full-scan iterations")
    ap.add_argument("--eps-views", type=int, default=64,
                    help="positions used for the eps table (it is a "
                         "diagnostic; the full scan takes minutes)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="object step (the campaign ran 1.0); unused "
                         "with --line-search")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="probe step; unused with --line-search")
    ap.add_argument("--precision", default="single",
                    choices=("single", "double"),
                    help="the campaign ran single")
    ap.add_argument("--line-search", action="store_true",
                    help="exact affine line search instead of the fixed "
                         "steps (off by default: the reference campaign "
                         "is the fixed-step one)")
    ap.add_argument("--plot-every", type=int, default=0,
                    help="write the figures every this many iterations "
                         "(0 = only at the end)")
    ap.add_argument("--synthetic", action="store_true",
                    help="synthetic specimen instead of the NanoMAX sim")
    ap.add_argument("--frames-only", action="store_true",
                    help="write the forward-frames figure and exit")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    CDT = cp.complex128 if args.precision == "double" else cp.complex64

    use_real = not args.synthetic and os.path.exists(TRUTH_H5)
    if use_real:
        print(f"ground truth (native, no rescaling): {TRUTH_H5}")
        truth_np, Pr = load_truth(TRUTH_H5)        # [3, 1224, 1225], [2,512,512]
        lam, res, z = real_geometry()
        n = Pr.shape[-1]
        dlow = load_scan_positions(PTYD, res)
        # vacuum-pad the canvas: window centres cannot sit nearer than
        # frame/2 to the canvas edge, so without padding a corner
        # particle can never be centred under the probe
        PAD = 200
        truth_np = np.pad(truth_np, ((0, 0), (PAD, PAD), (PAD, PAD)),
                          constant_values=1.0)
        # uniform raster over the full window-offset range, at the real
        # scan's own pitch (one pattern everywhere, border included)
        from scipy.spatial import cKDTree
        ny, nx = truth_np.shape[-2:]
        d_nn, _ = cKDTree(dlow).query(dlow, k=2)
        pitch = float(np.median(d_nn[:, 1]))
        gy = np.linspace(0, ny - n, max(2, int(round((ny - n) / pitch))
                                        + 1))
        gx = np.linspace(0, nx - n, max(2, int(round((nx - n) / pitch))
                                        + 1))
        positions = [(int(round(r)), int(round(c)))
                     for r in gy for c in gx]
        print(f"scan: {len(gy)} x {len(gx)} raster at the real scan's "
              f"pitch ({pitch:.1f} px) over the padded "
              f"{truth_np.shape[-2:]} canvas")
        print(f"probe: {Pr.shape} from the truth h5; "
              f"{len(positions)} scan positions")
        truth = cp.asarray(truth_np, dtype=CDT)
        H = cp.asarray(propagators_at(z, n, lam, res), dtype=CDT)
        P = cp.asarray(np.ascontiguousarray(Pr), dtype=CDT)
        phase = masked_truth_phase(truth_np)
    else:
        print("synthetic specimen (NanoMAX truth not found or --synthetic)")
        rng = np.random.default_rng(11)
        nc, n = 192, 96
        phase = make_synthetic(nc, 10.0, rng)
        truth_np = np.exp(1j * phase)
        truth = cp.asarray(truth_np, dtype=CDT)
        z3 = np.array([0.0, 50e-6, 100e-6])
        H = cp.asarray(propagators_at(z3, n, lam=1e-10, res=20e-9),
                       dtype=CDT)
        P = cp.asarray(focused_probe(n, frac=0.25)[None], dtype=CDT)
        stops = np.linspace(0, nc - n, 6).astype(int)
        positions = [(r, c) for r in stops for c in stops]

    shape = truth_np.shape[-2:]
    worst = 0
    for r, c in positions:
        hit = [(np.abs(phase[j, r:r + n, c:c + n]) > 0.2).any()
               for j in range(phase.shape[0])]
        worst = max(worst, sum(hit))
    print(f"{len(positions)} positions; layers per beam cone: max {worst} "
          + ("(disjoint -- chord order 1 is exact)" if worst <= 1
             else "(marginal overlap: chord order 1 is near-exact, as in "
                  "the campaign diagnostics)"))

    # ---- the ISSDiagnostics numbers over a subset of positions ----------
    k_eps = max(1, min(args.eps_views, len(positions)))
    sel = np.unique(np.linspace(0, len(positions) - 1,
                                k_eps).astype(int))
    eps, eps_tan = eps_table(truth, phase, P, H,
                             [positions[i] for i in sel], n)
    print(f"eps_M (chord) over {sel.size} positions:",
          np.array2string(eps, precision=3))
    print(f"eps_1 (tangent): {eps_tan:.3g}")

    # ---- exact multislice data over the scan positions ------------------
    print("simulating multislice data over the scan...")
    I = []
    for r, c in positions:
        Op = truth[:, r:r + n, c:c + n][:, None]
        I.append(detector_intensity(Op, P, H)[0].astype(
            cp.float64 if CDT == cp.complex128 else cp.float32))

    # ---- forward frames figure ------------------------------------------
    scores = [float(np.abs(phase[:, r:r + n, c:c + n]).max())
              for r, c in positions]
    frame_sel = [int(np.argmin(scores)), int(np.argmax(scores))]
    row_labels = ["frame through empty space", "frame through a particle"]
    frames = []
    for kk in frame_sel:
        r, c = positions[kk]
        Op = truth[:, r:r + n, c:c + n][:, None]
        phip = cp.asarray(phase[:, r:r + n, c:c + n], dtype=CDT)[:, None]
        frames.append([cp.asnumpy(I[kk]),
                       cp.asnumpy(order1_intensity(Op, P, H)[0]),
                       cp.asnumpy(order1_intensity(1.0 + 1j * phip,
                                                   P, H)[0])])
    vmax = max(f.max() for row in frames for f in row)
    norm = plt.matplotlib.colors.LogNorm(vmin=max(vmax * 1e-9, 1e-3),
                                         vmax=vmax)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), dpi=150,
                             constrained_layout=True)
    fig.suptitle("forward models through the ground truth, "
                 "same positions, one scale")
    for row in range(2):
        for col, (f, title) in enumerate(zip(frames[row], [
                "multislice", "inter-slice first Born (chord)",
                "slice-wise first Born (tangent)"])):
            ax = axes[row, col]
            im = ax.imshow(np.fft.fftshift(f), cmap="inferno", norm=norm)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02,
                 label="intensity")
    fig.savefig(os.path.join(args.out, "demo_04_sparse_forward_frames.png"))
    plt.close(fig)
    print("wrote demo_04_sparse_forward_frames.png")
    if args.frames_only:
        return

    # ---- reconstruct with both perturbations, in lockstep ---------------
    step_label = "line search" if args.line_search else "fixed step"
    suffix = "_ls" if args.line_search else ""
    win = display_window(truth_np)
    rec_c = Recon("chord", I, P, positions, shape, n, H,
                  args.alpha, args.beta, args.line_search, CDT)
    rec_t = Recon("tangent", I, P, positions, shape, n, H,
                  args.alpha, args.beta, args.line_search, CDT)
    dir_cmp = os.path.join(args.out, f"demo_04{suffix}_comparison")
    os.makedirs(dir_cmp, exist_ok=True)
    print(f"reconstructing chord + tangent in lockstep ({step_label}, "
          f"{args.iters} iterations x {len(positions)} positions)...")
    for it in range(args.iters):
        rec_c.step()
        rec_t.step()
        print(f"iteration {it + 1}: chord {rec_c.errs[-1]:.4g}, "
              f"tangent {rec_t.errs[-1]:.4g}", flush=True)
        if args.plot_every > 0 and (it + 1) % args.plot_every == 0:
            tag = f"iter_{it + 1:04d}"
            recon_figure(
                truth_np, win, rec_c, rec_t, step_label,
                os.path.join(dir_cmp, f"demo_04_recon_{tag}.png"),
                title_extra=f" -- iteration {it + 1}")
            convergence_figure(
                rec_c, rec_t, step_label,
                os.path.join(dir_cmp, "demo_04_convergence.png"))
            poserr_figure(
                rec_c, rec_t, positions, shape, n, step_label,
                os.path.join(dir_cmp, f"demo_04_poserr_{tag}.png"))

    recon_figure(truth_np, win, rec_c, rec_t, step_label, os.path.join(
        args.out, f"demo_04_sparse_chord_vs_tangent_recon{suffix}.png"))
    print(f"wrote demo_04_sparse_chord_vs_tangent_recon{suffix}.png")
    convergence_figure(rec_c, rec_t, step_label, os.path.join(
        args.out, f"demo_04_sparse_convergence{suffix}.png"))
    poserr_figure(rec_c, rec_t, positions, shape, n, step_label,
                  os.path.join(args.out,
                               f"demo_04_sparse_poserr{suffix}.png"))
    np.savez_compressed(
        os.path.join(args.out, f"demo_04{suffix}_results.npz"),
        ob_chord=cp.asnumpy(rec_c.ob).astype(np.complex64),
        ob_tangent=cp.asnumpy(rec_t.ob).astype(np.complex64),
        err_chord=np.asarray(rec_c.errs),
        err_tangent=np.asarray(rec_t.errs),
        pos_err_chord=rec_c.pos_err, pos_err_tangent=rec_t.pos_err,
        positions=np.asarray(positions), frame=n, iters=args.iters)
    print(f"wrote demo_04{suffix}_results.npz")
    print(f"final errors: chord {rec_c.errs[-1]:.3g}, "
          f"tangent {rec_t.errs[-1]:.3g}")


if __name__ == "__main__":
    main()
