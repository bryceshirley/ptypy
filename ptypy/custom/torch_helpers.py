"""
Numerical extraction of a ptypy propagator, shared by the torch engines.

Each ptypy propagator is probed numerically instead of reimplementing its FFT
conventions. Four candidate forms are built from two probes and the one that
reproduces the operator on random data is kept:

    lsi     op(w) = ifft2(fft2(w) * H)       near-field
    ilsi    op(w) = fft2(ifft2(w) * H)       near-field, reverse pairing
    chirp   op(w) = A * fft2(R * w)          far-field forward
    ichirp  op(w) = A * ifft2(R * w)         far-field reverse

numpy's convention is used throughout (fft2 unnormalised, ifft2 carrying 1/N),
which torch reproduces with the default norm='backward'. Both directions are
verified against the ptypy operator on random data, so a convention mismatch is
a startup error rather than a silently wrong reconstruction.

`extract_operator` returns a spec dict; `to_torch` moves it onto the device and
`apply_torch` applies it to a stack with trailing spatial dimensions.

Per-iteration diagnostic figure for the multislice torch engines.

Probe intensity, first slice phase and final slice phase, each with a zoom on
the centre quarter. Written as a PNG per iteration; nothing here feeds back
into a reconstruction.
"""

import numpy as np

from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import os

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.restoration import unwrap_phase


def slice_containers(ob, n, tag="o"):
    """
    n copies of the object container, under IDs no other engine has taken.

    A second multislice engine in the same run asks for the same names as the
    first, and ptypy's pool answers a duplicate container ID with an
    UnboundLocalError from _new_ptypy_object rather than with a message. That
    is what a probe warm-up stage followed by a multislice stage looks like, so
    the names are made unique here instead.
    """
    pool = getattr(ob.owner, "containers", {})
    out = []
    for i in range(n):
        ID, k = f"{ob.ID}_{tag}_{i}", 1
        while ID in pool:
            ID = f"{ob.ID}_{tag}{k}_{i}"
            k += 1
        out.append(ob.copy(ID))
    return out


def circular_support(fraction, shape, dtype, device):
    """
    ptypy's circular probe support as a torch mask, or None if disabled.

    `fraction` is p.probe_support, the radius as a fraction of the frame, the
    same convention BaseEngine.support_constraint uses. Values outside (0, 1)
    disable the support: 1 or more covers the whole frame and 0 would erase the
    probe. The torch engines run their own iteration loop and never reach
    support_constraint, so they apply this themselves.
    """
    if fraction is None or not 0 < float(fraction) < 1:
        return None
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - ny / 2.0, xx - nx / 2.0)
    mask = r < float(fraction) * min(ny, nx) / 2.0
    return torch.as_tensor(mask.astype(np.float64), dtype=dtype, device=device)


def complex_gaussian_filter(x, std, truncate=4.0):
    """
    Gaussian-blur the trailing two axes of a complex tensor.

    Matches ptypy's accelerate.base.array_utils.complex_gaussian_filter: the
    real and imaginary parts are filtered separately rather than amplitude and
    phase, so a phase wrap blurs as the smooth complex quantity it is instead of
    smearing a 2*pi step across the neighbourhood. Padding is 'reflect', scipy's
    default and the one that matters here -- this regulariser exists to tidy the
    unconstrained object border, and circular padding would blur that border
    against the far side of the canvas.

    Separable, so the cost is 2 * (8*std) taps per pixel rather than the square.
    Agrees with the numpy version to 5e-16 in the interior; the border matches
    too, which is why the padding is built by hand -- scipy's 'reflect' repeats
    the edge sample (a b c | c b a) and torch's F.pad 'reflect' does not
    (a b c | b a), which is what scipy calls 'mirror'.
    """
    if std is None or float(std) <= 0:
        return x
    std = float(std)

    *lead, ny, nx = x.shape
    r = int(truncate * std + 0.5)
    if r < 1:
        return x

    t = torch.arange(-r, r + 1, dtype=x.real.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / std) ** 2)
    k = k / k.sum()

    def sym(n):
        """Half-sample symmetric index map of length n + 2r, scipy's 'reflect'.

        Folded modulo 2n rather than reflected once, so it stays correct when
        the kernel is wider than the array -- which happens for a large std on
        a small object, and silently truncated the kernel when this reflected
        only once.
        """
        i = torch.remainder(torch.arange(-r, n + r, device=x.device), 2 * n)
        return torch.where(i >= n, 2 * n - 1 - i, i)

    # (..., ny, nx) complex -> (2 * prod(lead), 1, ny, nx) real, so one conv2d
    # covers every slice and both Cartesian components at once.
    v = torch.view_as_real(x).movedim(-1, 0).reshape(-1, 1, ny, nx)
    v = torch.nn.functional.conv2d(v.index_select(3, sym(nx)), k.view(1, 1, 1, -1))
    v = torch.nn.functional.conv2d(v.index_select(2, sym(ny)), k.view(1, 1, -1, 1))
    v = v.reshape(2, *lead, ny, nx).movedim(0, -1)
    return torch.view_as_complex(v.contiguous())


def clip_amplitude(x, interval):
    """
    Clip |x| into `interval`, leaving arg(x) alone. ptypy's clip_object.

    A transmission function of a non-amplifying sample has |O| <= 1, so an upper
    bound of 1 is physics rather than a tuning knob. The phase is untouched, so
    this constrains the object without touching what carries the signal.
    """
    if interval is None:
        return x
    lo, hi = float(interval[0]), float(interval[1])
    a = x.abs()
    # Where the object is exactly zero there is no direction to keep; the
    # clamp below would divide by zero, so leave those pixels as they are.
    safe = a.clamp(min=torch.finfo(a.dtype).tiny)
    return x * (a.clamp(lo, hi) / safe)


def op_call(op, arr):
    """
    Call a ptypy propagator, tolerating single- or double-precision input.

    complex128 is tried first, and not merely as a preference.  A propagator
    called with complex64 input carries out its FFTs in complex64, so the
    recovered transfer function is only accurate to ~1e-7 no matter what dtype
    it is stored in afterwards -- while the CPU engine calls the same
    propagator with the complex128 arrays ptypy holds at ``data_type='double'``.
    Extracting in single and comparing against a double-precision reference put
    a ~1e-6 floor under the CPU-vs-GPU agreement that looked like round-off but
    was really an artefact of the probe.  complex64 stays as a fallback for
    dtype-strict backends (pyFFTW plans are built for one dtype).
    """
    last = None
    for dt in (np.complex128, np.complex64):
        try:
            return np.asarray(op(arr.astype(dt)), dtype=np.complex128)
        except Exception as e:  # noqa: BLE001 - propagator dtype pickiness
            last = e
    raise RuntimeError(f"could not call propagator: {last}")


def apply_numpy(kind, d, w):
    if kind == "lsi":  # ifft2(fft2(w) * H)
        return np.fft.ifft2(np.fft.fft2(w) * d["H"])
    if kind == "ilsi":  # fft2(ifft2(w) * H)
        return np.fft.fft2(np.fft.ifft2(w) * d["H"])
    if kind == "chirp":  # A * fft2(R * w)
        return d["A"] * np.fft.fft2(d["R"] * w)
    if kind == "ichirp":  # A * ifft2(R * w)
        return d["A"] * np.fft.ifft2(d["R"] * w)
    raise ValueError(kind)


def extract_operator(op, shape, name="propagator", rtol=1e-5):
    """Recover a ptypy propagator exactly, without assuming its internals."""
    ny, nx = shape
    Ntot = float(ny * nx)

    d00 = np.zeros(shape, dtype=np.complex128)
    d00[0, 0] = 1.0
    ones = np.ones(shape, dtype=np.complex128)

    out00 = op_call(op, d00)
    out1 = op_call(op, ones)

    cands = [
        ("lsi", {"H": np.fft.fft2(out00)}),
        ("ilsi", {"H": Ntot * np.fft.ifft2(out00)}),
    ]
    if np.abs(out00).min() > 0:
        A = out00
        cands.append(("chirp", {"A": A, "R": np.fft.ifft2(out1 / A)}))
        cands.append(("ichirp", {"A": A, "R": np.fft.fft2(out1 / A)}))

    rng = np.random.default_rng(0)
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    ref = op_call(op, w)
    scale = max(np.abs(ref).max(), 1e-30)

    best_kind, best_d, best_err = None, None, np.inf
    errs = {}
    for kind, d in cands:
        err = np.abs(apply_numpy(kind, d, w) - ref).max() / scale
        errs[kind] = err
        if err < best_err:
            best_kind, best_d, best_err = kind, d, err

    if best_err > rtol:
        raise RuntimeError(
            "{}: numerical extraction failed. Errors by form: {}. The propagator "
            "is none of ifft2(fft2(w)*H), fft2(ifft2(w)*H), A*fft2(R*w), "
            "A*ifft2(R*w). Inspect it and hard-code the transfer function.".format(
                name, {k: f"{v:.2e}" for k, v in errs.items()}
            )
        )

    spec = {"kind": best_kind}
    spec.update(best_d)
    logger.info(f"{name} extracted as '{best_kind}', verification error {best_err:.2e}")
    return spec


def to_torch(spec, dtype, device):
    """Move an extracted spec onto the device."""
    out = {"kind": spec["kind"]}
    for k in ("H", "A", "R"):
        if k in spec:
            out[k] = torch.as_tensor(spec[k], dtype=dtype, device=device)
    return out


def apply_torch(spec, w):
    """Apply an extracted operator to a stack with trailing spatial dims."""
    k = spec["kind"]
    if k == "lsi":
        return torch.fft.ifft2(torch.fft.fft2(w) * spec["H"])
    if k == "ilsi":
        return torch.fft.fft2(torch.fft.ifft2(w) * spec["H"])
    if k == "chirp":
        return spec["A"] * torch.fft.fft2(spec["R"] * w)
    if k == "ichirp":
        return spec["A"] * torch.fft.ifft2(spec["R"] * w)
    raise ValueError(k)


def plot_progress(pr_g, ob_g, curiter, outdir, psize=(1e-6, 1e-6), crop=512,
                  engine_name=None):
    """
    Save the iteration figure for one engine state.

    pr_g        probe modes, (M, ny, nx) complex torch tensor
    ob_g        object slices, (N, ony, onx) complex torch tensor
    curiter     iteration number, used in the title and the file name
    outdir      directory for the PNG
    psize       object pixel size (dy, dx) in metres, for the axis extents
    crop        pixels trimmed from every edge of the object slices
    engine_name solver that produced the state, stated in the title so a
                folder of these figures identifies its run without the log
    """
    # deep stacks get the all-slice gallery: every slice at full panel
    # resolution, with the probe (full view + zoom) in its own figure
    if int(ob_g.shape[0]) > 2:
        return _plot_progress_multi(pr_g, ob_g, curiter, outdir, psize,
                                    crop, engine_name)

    # 1. Extract Data
    probe = (pr_g[0].abs() ** 2).cpu().numpy()
    first = ob_g[0].angle().cpu().numpy()
    final = ob_g[-1].angle().cpu().numpy()
    n_slices = int(ob_g.shape[0])

    # 2. Crop Object Slices ONLY (Keep Probe full-sized)
    d = crop // 2
    if d and 2 * d >= min(first.shape):
        d = 0  # object smaller than the crop: show it whole
    win = (slice(d, -d), slice(d, -d)) if d else (slice(None), slice(None))
    first_cropped = first[win]
    final_cropped = final[win]

    # 3. Phase Processing on Object Slices
    rec_ph_0 = unwrap_phase(first_cropped)
    rec_ph_1 = unwrap_phase(final_cropped)

    # Offset removal: the unwrapped phase sits at an arbitrary level; the
    # median is the empty background wherever most of the field is empty,
    # with no dataset-specific reference window to maintain.
    rec_ph_0 = rec_ph_0 - np.median(rec_ph_0)
    rec_ph_1 = rec_ph_1 - np.median(rec_ph_1)

    # 4. Extents & Pixel Scaling
    # Object Extent
    Ny_obj, Nx_obj = rec_ph_0.shape
    extent_obj = [0, Nx_obj * psize[1] * 1.0e6, 0, Ny_obj * psize[0] * 1.0e6]

    # Probe Extent (Uncropped)
    Ny_pr, Nx_pr = probe.shape
    extent_pr = [0, Nx_pr * psize[1] * 1.0e6, 0, Ny_pr * psize[0] * 1.0e6]

    # 5. Dynamic Zoom Region (Centered 25% window of the cropped object)
    x_min, x_max = extent_obj[0], extent_obj[1]
    y_min, y_max = extent_obj[2], extent_obj[3]

    x_center = x_min + 0.50 * (x_max - x_min)
    y_center = y_min + 0.50 * (y_max - y_min)
    zoom_width = 0.25 * (x_max - x_min)
    zoom_height = 0.25 * (y_max - y_min)

    zoom_xlim = (x_center - zoom_width / 2, x_center + zoom_width / 2)
    zoom_ylim = (y_center - zoom_height / 2, y_center + zoom_height / 2)

    # 6. Plotting Setup
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7), dpi=300)
    plt.suptitle(f"{engine_name} — Iteration {curiter} Progress"
                 if engine_name else f"Iteration {curiter} Progress")

    columns = [
        {
            "data": probe,
            "extent": extent_pr,
            "title": "Probe Intensity",
            "cmap": "viridis",
            "label": "Intensity",
        },
        {
            "data": rec_ph_0,
            "extent": extent_obj,
            "title": f"First Slice (0 of {n_slices})",
            "cmap": "Greys_r",
            "label": "",
        },
        {
            "data": rec_ph_1,
            "extent": extent_obj,
            "title": f"Final Slice ({n_slices - 1} of {n_slices})",
            "cmap": "Greys_r",
            "label": "phase shift [rad]",
        },
    ]

    for col_idx, col in enumerate(columns):
        data, extent = col["data"], col["extent"]

        # --- Top Row: Full View ---
        ax_full = axes[0, col_idx]
        ax_full.set_title(col["title"])
        im_full = ax_full.imshow(
            data, interpolation="None", cmap=col["cmap"], extent=extent
        )
        ax_full.set_xticks([])
        ax_full.set_yticks([])

        if col_idx > 0:
            rect = Rectangle(
                (zoom_xlim[0], zoom_ylim[0]),
                zoom_xlim[1] - zoom_xlim[0],
                zoom_ylim[1] - zoom_ylim[0],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax_full.add_patch(rect)

        sb_x = extent[1] * 0.70
        sb_y = extent[3] * 0.08
        ax_full.plot([sb_x, sb_x + 2.0], [sb_y, sb_y], lw=3, c="r")
        ax_full.text(
            sb_x + 1.0,
            sb_y + (extent[3] * 0.03),
            "2 μm",
            ha="center",
            c="r",
            fontweight="semibold",
            fontsize=11,
        )

        if col["label"]:
            fig.colorbar(
                im_full, ax=ax_full, fraction=0.046, pad=0.04, label=col["label"]
            )

        # --- Bottom Row: Zoom View ---
        ax_zoom = axes[1, col_idx]
        ax_zoom.set_title("zoom in")
        im_zoom = ax_zoom.imshow(
            data, interpolation="None", cmap=col["cmap"], extent=extent
        )
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])

        if col_idx == 0:
            pr_cx, pr_cy = extent_pr[1] / 2, extent_pr[3] / 2
            ax_zoom.set_xlim(pr_cx - zoom_width / 2, pr_cx + zoom_width / 2)
            ax_zoom.set_ylim(pr_cy - zoom_height / 2, pr_cy + zoom_height / 2)
        else:
            ax_zoom.set_xlim(zoom_xlim)
            ax_zoom.set_ylim(zoom_ylim)

        curr_xlim = ax_zoom.get_xlim()
        curr_ylim = ax_zoom.get_ylim()
        z_sb_x = curr_xlim[0] + (curr_xlim[1] - curr_xlim[0]) * 0.60
        z_sb_y = curr_ylim[0] + (curr_ylim[1] - curr_ylim[0]) * 0.08

        ax_zoom.plot([z_sb_x, z_sb_x + 0.5], [z_sb_y, z_sb_y], lw=3, c="r")
        ax_zoom.text(
            z_sb_x + 0.25,
            z_sb_y + ((curr_ylim[1] - curr_ylim[0]) * 0.03),
            "500 nm",
            ha="center",
            c="r",
            fontweight="semibold",
            fontsize=11,
        )

        if col["label"]:
            fig.colorbar(
                im_zoom, ax=ax_zoom, fraction=0.046, pad=0.04, label=col["label"]
            )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"iter_{curiter:04d}.png"))
    plt.close(fig)


def _plot_progress_multi(pr_g, ob_g, curiter, outdir, psize, crop,
                         engine_name):
    """
    Iteration figures for stacks of more than two slices.

    The slice gallery (iter_XXXX.png) is one full-resolution panel per
    slice -- unwrapped phase, median background removed, all slices on a
    shared colour scale so depth structure is comparable across panels.
    The probe is written to its own figure (iter_XXXX_probe.png) with a
    full view and a centred zoom, matching the two-slice probe column.
    """
    probe = (pr_g[0].abs() ** 2).cpu().numpy()
    n_slices = int(ob_g.shape[0])

    d = crop // 2
    if d and 2 * d >= min(ob_g.shape[-2:]):
        d = 0  # object smaller than the crop: show it whole
    win = (slice(d, -d), slice(d, -d)) if d else (slice(None), slice(None))

    phases = []
    for j in range(n_slices):
        ph = unwrap_phase(ob_g[j].angle().cpu().numpy()[win])
        phases.append(ph - np.median(ph))

    # shared scale from robust percentiles across the whole stack
    vmin = min(float(np.percentile(p, 1)) for p in phases)
    vmax = max(float(np.percentile(p, 99)) for p in phases)
    if vmax <= vmin:
        vmin, vmax = vmin - 1e-3, vmax + 1e-3

    Ny_obj, Nx_obj = phases[0].shape
    extent_obj = [0, Nx_obj * psize[1] * 1.0e6, 0, Ny_obj * psize[0] * 1.0e6]

    # ---- slice gallery: slices only, probe lives in its own figure ----
    ncols = min(6, n_slices)
    nrows = int(np.ceil(n_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.4 * nrows),
                             dpi=200, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"{engine_name} — Iteration {curiter} Progress"
                 if engine_name else f"Iteration {curiter} Progress")

    im = None
    for j in range(n_slices):
        ax = axes[j]
        im = ax.imshow(phases[j], interpolation="None", cmap="Greys_r",
                       vmin=vmin, vmax=vmax, extent=extent_obj)
        ax.set_title(f"slice {j} of {n_slices}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # scalebar once, on the first slice panel
    ax = axes[0]
    sb_x = extent_obj[1] * 0.70
    sb_y = extent_obj[3] * 0.08
    ax.plot([sb_x, sb_x + 2.0], [sb_y, sb_y], lw=3, c="r")
    ax.text(sb_x + 1.0, sb_y + extent_obj[3] * 0.03, "2 μm", ha="center",
            c="r", fontweight="semibold", fontsize=10)

    for k in range(n_slices, len(axes)):
        axes[k].axis("off")

    fig.colorbar(im, ax=list(axes[:n_slices]), fraction=0.02, pad=0.01,
                 shrink=0.7, label="phase shift [rad]")

    plt.savefig(os.path.join(outdir, f"iter_{curiter:04d}.png"))
    plt.close(fig)

    # ---- probe: separate figure, full view + centred zoom ----
    Ny_pr, Nx_pr = probe.shape
    extent_pr = [0, Nx_pr * psize[1] * 1.0e6, 0, Ny_pr * psize[0] * 1.0e6]
    pr_cx, pr_cy = extent_pr[1] / 2, extent_pr[3] / 2
    zoom_width = 0.25 * extent_pr[1]
    zoom_height = 0.25 * extent_pr[3]

    figp, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=200)
    figp.suptitle(f"{engine_name} — Iteration {curiter} Probe"
                  if engine_name else f"Iteration {curiter} Probe")

    im_full = ax_full.imshow(probe, interpolation="None", cmap="viridis",
                             extent=extent_pr)
    ax_full.set_title("Probe Intensity")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    rect = Rectangle((pr_cx - zoom_width / 2, pr_cy - zoom_height / 2),
                     zoom_width, zoom_height,
                     linewidth=1, edgecolor="r", facecolor="none")
    ax_full.add_patch(rect)
    sb_x = extent_pr[1] * 0.70
    sb_y = extent_pr[3] * 0.08
    ax_full.plot([sb_x, sb_x + 2.0], [sb_y, sb_y], lw=3, c="r")
    ax_full.text(sb_x + 1.0, sb_y + extent_pr[3] * 0.03, "2 μm", ha="center",
                 c="r", fontweight="semibold", fontsize=11)
    figp.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04,
                  label="Intensity")

    im_zoom = ax_zoom.imshow(probe, interpolation="None", cmap="viridis",
                             extent=extent_pr)
    ax_zoom.set_title("zoom in")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_xlim(pr_cx - zoom_width / 2, pr_cx + zoom_width / 2)
    ax_zoom.set_ylim(pr_cy - zoom_height / 2, pr_cy + zoom_height / 2)
    z_sb_x = pr_cx - zoom_width / 2 + zoom_width * 0.60
    z_sb_y = pr_cy - zoom_height / 2 + zoom_height * 0.08
    ax_zoom.plot([z_sb_x, z_sb_x + 0.5], [z_sb_y, z_sb_y], lw=3, c="r")
    ax_zoom.text(z_sb_x + 0.25, z_sb_y + zoom_height * 0.03, "500 nm",
                 ha="center", c="r", fontweight="semibold", fontsize=11)
    figp.colorbar(im_zoom, ax=ax_zoom, fraction=0.046, pad=0.04,
                  label="Intensity")

    figp.tight_layout()
    figp.savefig(os.path.join(outdir, f"iter_{curiter:04d}_probe.png"))
    plt.close(figp)


def plot_error_history(err_history, outdir, engine_name=None):
    """
    Save the convergence curve for one run, overwriting on every call so the
    file in the progress folder is always the history so far.

    Only the Fourier error is plotted: the photon and exit errors track it and
    their very different scales forced a decades-wide axis that flattened the
    one curve being read.

    err_history  sequence of per-iteration error vectors, shape (n_iter, k);
                 column 0 is the Fourier error
    outdir       directory for the PNG (error_history.png)
    engine_name  solver that produced the errors, stated in the title
    """
    arr = np.asarray(err_history, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return
    it = np.arange(1, arr.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.semilogy(it, np.maximum(arr[:, 0], 1e-30), lw=1.4)
    ax.set_xlabel("iteration")
    ax.set_ylabel("Fourier error (engine units)")
    head = f"{engine_name} — convergence" if engine_name else "convergence"
    ax.set_title("%s\nFourier error %.3g after %d iterations"
                 % (head, arr[-1, 0], arr.shape[0]), fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_history.png"))
    plt.close(fig)
