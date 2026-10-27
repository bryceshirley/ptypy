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
import sys

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


def _draw_scalebar(ax, extent, length_um):
    """Red scale bar with label, lower right of the panel."""
    x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
    sx = x0 + (x1 - x0) * 0.68
    sy = y0 + (y1 - y0) * 0.08
    ax.plot([sx, sx + length_um], [sy, sy], lw=3, c="r")
    label = ("%g μm" % length_um if length_um >= 1
             else "%d nm" % round(length_um * 1000))
    ax.text(sx + length_um / 2, sy + (y1 - y0) * 0.03, label,
            ha="center", c="r", fontweight="semibold", fontsize=11)


def _probe_centroid_um(probe, psize):
    """Intensity centroid of the probe in axis units (um, origin lower)."""
    tot = float(probe.sum())
    if tot <= 0:
        return (probe.shape[1] / 2 * psize[1] * 1e6,
                probe.shape[0] / 2 * psize[0] * 1e6)
    iy, ix = np.indices(probe.shape)
    return (float((probe * ix).sum()) / tot * psize[1] * 1e6,
            float((probe * iy).sum()) / tot * psize[0] * 1e6)


def plot_progress(pr_g, ob_g, curiter, outdir, psize=(1e-6, 1e-6), crop=512,
                  engine_name=None, cmaps=None, zoom_center=None,
                  phase_clim=None, scalebar=None, phase_zero_top=False):
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
    cmaps       optional {"probe": ..., "phase": ...} colormap overrides
                (defaults keep the historical viridis / Greys_r)
    zoom_center optional fractional (x, y) centre of the object zoom box,
                origin lower-left; default (0.5, 0.5). A list of (x, y)
                pairs sets one centre per slice column. The probe zoom
                always tracks the intensity centroid.
    phase_clim  optional explicit (vmin, vmax) for the phase panels,
                overriding the percentile limits
    scalebar    optional (full_um, zoom_um) red scale bars on the object
                panels, e.g. (2.0, 0.5) for 2 um / 500 nm
    phase_zero_top  shift each slice so the (99.5th-percentile) background
                sits at 0 and pin the colour bar top there, phase shown as
                purely negative -- the paper's x-ray figure convention
    """
    cm = {"probe": "viridis", "phase": "Greys_r"}
    if cmaps:
        cm.update(cmaps)
    # very deep stacks get the all-slice gallery: every slice at full panel
    # resolution, with the probe (full view + zoom) in its own figure
    if int(ob_g.shape[0]) > 6:
        return _plot_progress_multi(pr_g, ob_g, curiter, outdir, psize,
                                    crop, engine_name, cmaps=cm)

    # 1. Extract Data
    probe = (pr_g[0].abs() ** 2).cpu().numpy()
    n_slices = int(ob_g.shape[0])

    # 2. Crop Object Slices ONLY (Keep Probe full-sized)
    ob0 = ob_g[0].angle().cpu().numpy()
    d = crop // 2
    if d and 2 * d >= min(ob0.shape):
        d = 0  # object smaller than the crop: show it whole
    win = (slice(d, -d), slice(d, -d)) if d else (slice(None), slice(None))

    # 3. Phase Processing on Object Slices
    # Offset removal: the unwrapped phase sits at an arbitrary level; the
    # median is the empty background wherever most of the field is empty,
    # with no dataset-specific reference window to maintain.
    # Unwrapping gives the cleanest display for smooth phases, but on
    # strongly wrapped objects (a zone plate after gauge drift) the
    # unwrapper assigns 2-pi branches per island and the panel turns into
    # patchwork. Fall back to the raw wrapped phase when that happens.
    slice_ph = []
    slice_flip = []
    for j in range(n_slices):
        wrapped = ob_g[j].angle().cpu().numpy()[win]
        ph = unwrap_phase(wrapped)
        ph -= np.median(ph)
        span = np.percentile(ph, 99.5) - np.percentile(ph, 0.5)
        if span > 6 * np.pi:
            ph = wrapped - np.median(wrapped)
        flip = False
        if phase_zero_top:
            # background is at 0 (median); if the features are
            # positive-dominant, flip the DISPLAY sign so the panel reads
            # white background / darker objects -- flagged in the title.
            if abs(float(np.percentile(ph, 99.5))) \
                    > abs(float(np.percentile(ph, 0.5))):
                ph = -ph
                flip = True
        slice_ph.append(ph)
        slice_flip.append(flip)
    rec_ph_0, rec_ph_1 = slice_ph[0], slice_ph[-1]

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

    zoom_width = 0.25 * (x_max - x_min)
    zoom_height = 0.25 * (y_max - y_min)
    if zoom_center is None:
        fracs = [(0.5, 0.5)] * n_slices
    elif hasattr(zoom_center[0], "__len__"):
        fracs = [tuple(zoom_center[min(j, len(zoom_center) - 1)])
                 for j in range(n_slices)]
    else:
        fracs = [tuple(zoom_center)] * n_slices
    zoom_boxes = []
    for fx, fy in fracs:
        xc = min(max(x_min + fx * (x_max - x_min),
                     x_min + zoom_width / 2), x_max - zoom_width / 2)
        yc = min(max(y_min + fy * (y_max - y_min),
                     y_min + zoom_height / 2), y_max - zoom_height / 2)
        zoom_boxes.append((xc, yc))

    # 6. Plotting Setup
    ncols = 1 + n_slices
    fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 7), dpi=300,
                             squeeze=False)
    plt.suptitle(f"{engine_name} — Iteration {curiter} Progress"
                 if engine_name else f"Iteration {curiter} Progress")

    # Robust colour limits: percentiles, not extremes -- a handful of hot
    # pixels otherwise flattens everything readable. All slices share one
    # scale so they stay comparable.
    if phase_clim is not None:
        ph_lo, ph_hi = float(phase_clim[0]), float(phase_clim[1])
    elif phase_zero_top:
        # The background sits at 0 (median-removed). Pin the bar top just
        # above the SAMPLED BACKGROUND NOISE, not the global maximum:
        # sigma_bg from the median absolute deviation of the near-zero
        # pixels, so the background maps to the white end and anything
        # brighter (e.g. positive ridges) clips to white.
        sig = []
        for p in slice_ph:
            near = p[np.abs(p) < max(3 * np.median(np.abs(p)), 1e-6)]
            if near.size:
                sig.append(1.4826 * float(np.median(np.abs(near))))
        ph_hi = max(2.5 * min(sig) if sig else 1e-3, 1e-3)
        ph_lo = min(float(np.percentile(p, 0.5)) for p in slice_ph)
        if ph_lo >= ph_hi:
            ph_lo = ph_hi - 1e-3
    else:
        ph_lo = min(float(np.percentile(p, 1)) for p in slice_ph)
        ph_hi = max(float(np.percentile(p, 99)) for p in slice_ph)
        if ph_hi <= ph_lo:
            ph_lo, ph_hi = ph_lo - 1e-3, ph_hi + 1e-3
    pr_hi = float(np.percentile(probe, 99.9))

    columns = [
        {
            "data": probe,
            "extent": extent_pr,
            "title": "Probe Intensity",
            "cmap": cm["probe"],
            "label": "Intensity",
            "clim": (0.0, pr_hi if pr_hi > 0 else None),
        },
    ]
    for j, ph in enumerate(slice_ph):
        if j == 0:
            title = f"First Slice (0 of {n_slices})"
        elif j == n_slices - 1:
            title = f"Final Slice ({j} of {n_slices})"
        else:
            title = f"Slice {j} (of {n_slices})"
        if slice_flip[j]:
            title += " (sign flipped)"
        columns.append({
            "data": ph,
            "extent": extent_obj,
            "title": title,
            "cmap": cm["phase"],
            "label": "phase shift [rad]" if j == n_slices - 1 else "",
            "clim": (ph_lo, ph_hi),
        })

    for col_idx, col in enumerate(columns):
        data, extent = col["data"], col["extent"]

        # --- Top Row: Full View ---
        ax_full = axes[0, col_idx]
        ax_full.set_title(col["title"])
        im_full = ax_full.imshow(
            data, interpolation="None", cmap=col["cmap"], extent=extent,
            origin="lower", vmin=col["clim"][0], vmax=col["clim"][1]
        )
        ax_full.set_xticks([])
        ax_full.set_yticks([])

        if col_idx > 0:
            xc, yc = zoom_boxes[col_idx - 1]
            rect = Rectangle(
                (xc - zoom_width / 2, yc - zoom_height / 2),
                zoom_width,
                zoom_height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax_full.add_patch(rect)
            if scalebar:
                _draw_scalebar(ax_full, extent, float(scalebar[0]))

        if col["label"]:
            fig.colorbar(
                im_full, ax=ax_full, fraction=0.046, pad=0.04, label=col["label"]
            )

        # --- Bottom Row: Zoom View ---
        ax_zoom = axes[1, col_idx]
        ax_zoom.set_title("zoom in")
        im_zoom = ax_zoom.imshow(
            data, interpolation="None", cmap=col["cmap"], extent=extent,
            origin="lower", vmin=col["clim"][0], vmax=col["clim"][1]
        )
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])

        if col_idx == 0:
            pr_cx, pr_cy = _probe_centroid_um(probe, psize)
            ax_zoom.set_xlim(pr_cx - zoom_width / 2, pr_cx + zoom_width / 2)
            ax_zoom.set_ylim(pr_cy - zoom_height / 2, pr_cy + zoom_height / 2)
        else:
            xc, yc = zoom_boxes[col_idx - 1]
            ax_zoom.set_xlim(xc - zoom_width / 2, xc + zoom_width / 2)
            ax_zoom.set_ylim(yc - zoom_height / 2, yc + zoom_height / 2)
            if scalebar and len(scalebar) > 1:
                zx = ax_zoom.get_xlim()
                zy = ax_zoom.get_ylim()
                zext = [zx[0], zx[1], zy[0], zy[1]]
                _draw_scalebar(ax_zoom, zext, float(scalebar[1]))

        if col["label"]:
            fig.colorbar(
                im_zoom, ax=ax_zoom, fraction=0.046, pad=0.04, label=col["label"]
            )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"iter_{curiter:04d}.png"))
    plt.close(fig)


def _plot_progress_multi(pr_g, ob_g, curiter, outdir, psize, crop,
                         engine_name, cmaps=None):
    """
    Iteration figures for stacks of more than two slices.

    The slice gallery (iter_XXXX.png) is one full-resolution panel per
    slice -- unwrapped phase, median background removed, all slices on a
    shared colour scale so depth structure is comparable across panels.
    The probe is written to its own figure (iter_XXXX_probe.png) with a
    full view and a centred zoom, matching the two-slice probe column.
    """
    cm = {"probe": "viridis", "phase": "Greys_r"}
    if cmaps:
        cm.update(cmaps)
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
        im = ax.imshow(phases[j], interpolation="None", cmap=cm["phase"],
                       vmin=vmin, vmax=vmax, extent=extent_obj,
                       origin="lower")
        ax.set_title(f"slice {j} of {n_slices}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n_slices, len(axes)):
        axes[k].axis("off")

    fig.colorbar(im, ax=list(axes[:n_slices]), fraction=0.02, pad=0.01,
                 shrink=0.7, label="phase shift [rad]")

    plt.savefig(os.path.join(outdir, f"iter_{curiter:04d}.png"))
    plt.close(fig)

    # ---- probe: separate figure, full view + centred zoom ----
    Ny_pr, Nx_pr = probe.shape
    extent_pr = [0, Nx_pr * psize[1] * 1.0e6, 0, Ny_pr * psize[0] * 1.0e6]
    pr_cx, pr_cy = _probe_centroid_um(probe, psize)
    zoom_width = 0.25 * extent_pr[1]
    zoom_height = 0.25 * extent_pr[3]

    figp, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=200)
    figp.suptitle(f"{engine_name} — Iteration {curiter} Probe"
                  if engine_name else f"Iteration {curiter} Probe")

    pr_hi = float(np.percentile(probe, 99.9)) or None
    im_full = ax_full.imshow(probe, interpolation="None", cmap=cm["probe"],
                             extent=extent_pr, origin="lower",
                             vmin=0.0, vmax=pr_hi)
    ax_full.set_title("Probe Intensity")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    rect = Rectangle((pr_cx - zoom_width / 2, pr_cy - zoom_height / 2),
                     zoom_width, zoom_height,
                     linewidth=1, edgecolor="r", facecolor="none")
    ax_full.add_patch(rect)
    figp.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04,
                  label="Intensity")

    im_zoom = ax_zoom.imshow(probe, interpolation="None", cmap=cm["probe"],
                             extent=extent_pr, origin="lower",
                             vmin=0.0, vmax=pr_hi)
    ax_zoom.set_title("zoom in")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_xlim(pr_cx - zoom_width / 2, pr_cx + zoom_width / 2)
    ax_zoom.set_ylim(pr_cy - zoom_height / 2, pr_cy + zoom_height / 2)
    figp.colorbar(im_zoom, ax=ax_zoom, fraction=0.046, pad=0.04,
                  label="Intensity")

    figp.tight_layout()
    figp.savefig(os.path.join(outdir, f"iter_{curiter:04d}_probe.png"))
    plt.close(figp)


def plot_error_history(err_history, outdir, engine_name=None, channels=None):
    """
    Save the convergence curve for one run, overwriting on every call so the
    file in the progress folder is always the history so far.

    All recorded channels share one log axis with a legend, matching the
    campaign RAAR summary figures: column 0 fourier, column 1 photon,
    column 2 exit. On real data the three sit within a decade or two of
    each other, so the shared axis reads fine.

    err_history  sequence of per-iteration error vectors, shape (n_iter, k)
    outdir       directory for the PNG (error_history.png)
    engine_name  solver that produced the errors, stated in the title
    channels     which channels to draw: None (all recorded, the previous
                 behaviour) or an iterable / comma-separated string of names
                 from {fourier, photon, exit}
    """
    arr = np.asarray(err_history, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return
    it = np.arange(1, arr.shape[0] + 1)
    labels = ["fourier", "photon", "exit"]
    if channels is None:
        wanted = set(labels)
    else:
        if isinstance(channels, str):
            channels = channels.split(",")
        wanted = {str(c).strip().lower() for c in channels}
        unknown = wanted - set(labels)
        if unknown:
            raise ValueError("unknown error channel(s) %s; choose from %s"
                             % (sorted(unknown), labels))
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    for c in range(min(arr.shape[1], len(labels))):
        if labels[c] not in wanted:
            continue
        col = np.maximum(arr[:, c], 1e-30)
        if np.any(col > 1e-30):
            ax.semilogy(it, col, lw=1.4, label=labels[c])
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    head = f"{engine_name} — convergence" if engine_name else "convergence"
    ax.set_title("%s\nfourier %.3g after %d iterations"
                 % (head, arr[-1, 0], arr.shape[0]), fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_history.png"))
    plt.close(fig)

from datetime import datetime

from ptypy import io
from ptypy import utils as u
from ptypy.utils import Param

class _TermRecorder:
    """Capped, order-preserving buffer shared by the stdout/stderr tees."""

    def __init__(self, cap=16 << 20):
        self.parts, self.n, self.cap = [], 0, cap

    def add(self, x):
        if self.n < self.cap:
            self.parts.append(x)
            self.n += len(x)

    def dump(self):
        return "".join(self.parts)


class _TeeStream:
    """Mirror writes to the live stream and the engine's terminal log.

    Until the log path is known (the engine does not exist yet during
    Ptycho init levels 1-3) the stream runs in RECORDER mode: writes pass
    through and accumulate in a shared capped buffer. When the engine
    resolves its log file, _start_terminal_log dumps that recording into
    the file and attaches the handle, so the terminal log carries the FULL
    run output -- data import, POD creation, probe/object init -- not just
    what follows engine_initialize.
    """

    _torch_tee = True

    def __init__(self, stream, fh=None, rec=None):
        self._s, self._f, self._rec = stream, fh, rec

    def attach(self, fh):
        self._f, self._rec = fh, None

    def write(self, x):
        self._s.write(x)
        if self._f is not None:
            self._f.write(x)
            self._f.flush()
        elif self._rec is not None:
            self._rec.add(x)

    def flush(self):
        self._s.flush()
        if self._f is not None:
            self._f.flush()

    def __getattr__(self, a):
        return getattr(self._s, a)


def _repoint_log_handlers():
    """Point ptypy's stream handlers at the current (wrapped) stderr."""
    import logging
    for lg in (logging.getLogger(), logging.getLogger("ptypy")):
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) \
                    and not isinstance(h, logging.FileHandler):
                h.stream = sys.stderr


def _install_terminal_recorder():
    """Pass-through recorders on stdout/stderr, installed at import time.

    Run scripts import the engine modules before Ptycho is built, so the
    recording starts ahead of init level 1 and the engine's terminal log
    can include everything the run printed. Pure pass-through until an
    engine attaches a file; the cap stops the buffer growing without bound
    in a process that never does. A tee installed by a run script (the
    legacy _Tee pattern) is respected and nothing is wrapped over it.
    """
    cur = type(sys.stdout)
    if getattr(cur, "_torch_tee", False) or cur.__name__ == "_Tee":
        return
    rec = _TermRecorder()
    sys.stdout = _TeeStream(sys.stdout, rec=rec)
    sys.stderr = _TeeStream(sys.stderr, rec=rec)
    _repoint_log_handlers()


_install_terminal_recorder()


# ===========================================================================
#  Shared torch infrastructure
# ===========================================================================
class TorchMixin:
    """Model-agnostic torch plumbing shared by the custom engines:
    device/dtype setup, far-field operator extraction, the BaseEngine
    probe-support convention, and the progress plotting and terminal
    logging, together with the Defaults for the shared plotting and
    logging options below. Family infrastructure (ISSEngineInfra) or
    an engine mixes this in; everything model-specific stays in the
    concrete engine's own parameter tree.

    Defaults:

    [plot_interval]
    default = 0
    type = int
    help = Iterations between diagnostic PNGs, 0 disables

    [plot_dir]
    default = None
    type = str
    help = Directory for the progress figures and terminal log; None
      auto-creates a timestamped iteration_progress_<engine>_<stamp>/ in
      the CURRENT WORKING directory (not the run's output directory)

    [plot_error_channels]
    default = fourier,photon,exit
    type = str
    help = Which error channels the convergence PNG draws
    doc = Comma-separated subset of fourier, photon, exit. The error
          HISTORY always records all three; this only selects what
          error_history.png shows.

    [plot_zoom_center]
    default = None
    type = tuple
    help = Fractional (x, y) centre of the object zoom box in the progress
      figures, origin lower-left; None means the panel centre

    [plot_phase_zero_top]
    default = True
    type = bool
    help = Shift each slice's displayed phase so the background sits at 0 and
      the colour bar tops out there (phase purely negative, the x-ray paper
      figure convention)

    [plot_phase_clim]
    default = None
    type = tuple
    help = Explicit (vmin, vmax) for the phase panels; None uses robust
      percentiles

    [plot_scalebar]
    default = None
    type = tuple
    help = Red scale bars (full_um, zoom_um) on the object panels, e.g.
      (2.0, 0.5); None draws none

    [terminal_log]
    default = auto
    type = str
    help = Mirror this process's stdout/stderr into a log file
    doc = 'auto' writes terminal.log inside the progress directory when
          plotting is on, else <fslices stem>_terminal.log; an explicit
          path is used as given; '' disables. A tee already installed by
          the run script is detected and left alone.
    """

    # {"probe": ..., "phase": ...} colormaps for the progress figures;
    # subclasses may override to restyle without touching the helper.
    PLOT_CMAPS = {"probe": "inferno", "phase": "Greys_r"}

    def _stamp_plot_dir(self):
        explicit = self.p.get("plot_dir", None)
        if explicit:
            self.plot_progress_dir = str(explicit)
            return
        # Local time on purpose: this names a run directory the user reads.
        # The class name is in it so a folder of runs identifies its solver
        # without the log (type(self) so subclasses label themselves).
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        self.plot_progress_dir = \
            f"iteration_progress_{type(self).__name__}_{stamp}"

    def _setup_torch_device(self):
        """Resolve device and dtypes from p.device / p.precision."""
        if torch is None:
            raise ImportError("ISSTorch requires PyTorch")
        want = str(self.p.device)
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            want = "cpu"
        self.dev = torch.device(want)
        dbl = str(self.p.precision).lower().startswith("d")
        self.cdt = torch.complex128 if dbl else torch.complex64
        self.rdt = torch.float64 if dbl else torch.float32

    def _setup_probe_support(self):
        """Circular probe support whenever p.probe_support is set."""
        # BaseEngine's [probe_support] convention: a circular support is
        # applied whenever the fraction is set (None disables, the
        # PositionCorrectionEngine default)
        self.pr_support = circular_support(
            getattr(self.p, "probe_support", None),
            (self.fny, self.fnx),
            self.rdt,
            self.dev,
        )
        if self.pr_support is not None:
            logger.info(
                f"probe support: circular, radius fraction "
                f"{float(self.p.probe_support):.3f}"
            )

    def _setup_plot_state(self):
        """Error history, plot cadence, progress dir, terminal log, and
        the reproducibility record (run script copy, git commit)."""
        self._err_history = []
        self._plot_int = max(0, int(self.p.plot_interval))
        # Only when plotting is on: with plot_interval = 0 nothing is ever
        # written there and the empty timestamped directory is pure litter.
        if self._plot_int > 0:
            os.makedirs(self.plot_progress_dir, exist_ok=True)
            self._save_run_script()
        self._start_terminal_log()
        self._log_git_commit()

    def _save_run_script(self):
        """Copy the entry script into the progress directory, so a run
        folder records exactly what produced it."""
        import shutil
        src = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        if not src.endswith(".py") or not os.path.isfile(src):
            return
        try:
            shutil.copy2(src, os.path.join(
                self.plot_progress_dir,
                "run_script_" + os.path.basename(src)))
            logger.info("run script saved: %s" % os.path.basename(src))
        except OSError as e:
            logger.warning("could not save the run script: %s" % e)

    def _log_git_commit(self):
        """Record the ptypy git commit (and dirty state) in the run log."""
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            head = subprocess.run(
                ["git", "-C", here, "rev-parse", "--short=12", "HEAD"],
                capture_output=True, text=True, timeout=5)
            if head.returncode != 0:
                return
            # -uno: the tracked files decide dirtiness, untracked scans of
            # a large data tree would stall the engine start
            dirty = subprocess.run(
                ["git", "-C", here, "status", "--porcelain", "-uno"],
                capture_output=True, text=True, timeout=5)
            flag = " (dirty)" if dirty.stdout.strip() else ""
            logger.info("ptypy git commit: %s%s"
                        % (head.stdout.strip(), flag))
        except (OSError, subprocess.SubprocessError):
            pass

    def _start_terminal_log(self):
        """Attach the run's log file to the import-time terminal recorder.

        Engine-owned so every run is logged without each script rebuilding
        the tee, and COMPLETE: the recorder has been buffering since the
        engine module was imported, so the file starts with the full
        Ptycho init output (data import, POD creation, probe/object init),
        exactly what the legacy script-side _Tee captured. p.terminal_log
        gives the path; 'auto' resolves into the progress directory when
        plotting is on, else next to fslices; '' disables. Idempotent: a
        script's own tee, or a previous engine's attached log in this
        process, is detected and left alone.
        """
        spec = str(self.p.get("terminal_log", "auto") or "")
        if not spec:
            return
        cur = sys.stdout
        if isinstance(cur, _TeeStream):
            if cur._f is not None:
                return          # an earlier engine already logs this process
        elif getattr(type(cur), "_torch_tee", False) \
                or type(cur).__name__ == "_Tee":
            return              # the run script installed its own tee
        if spec.lower() == "auto":
            if self._plot_int > 0:
                path = os.path.join(self.plot_progress_dir, "terminal.log")
            else:
                path = os.path.splitext(str(self.p.fslices))[0] \
                    + "_terminal.log"
        else:
            path = spec
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = open(path, "a")  # noqa: SIM115 - lives for the process
        if isinstance(cur, _TeeStream):
            if cur._rec is not None:
                fh.write(cur._rec.dump())
                fh.flush()
            cur.attach(fh)
            if isinstance(sys.stderr, _TeeStream):
                sys.stderr.attach(fh)
        else:
            # recorder was never installed (module imported some unusual
            # way): fall back to teeing from here on
            sys.stdout = _TeeStream(sys.stdout, fh=fh)
            sys.stderr = _TeeStream(sys.stderr, fh=fh)
        _repoint_log_handlers()
        logger.info("terminal log: %s" % path)

    def _build_farfield(self):
        """Extracts fw (forward) and bw (backward) far-field propagators from ptypy."""
        pod = next(iter(next(iter(self.di.views.values())).pods.values()))
        shape = (self.fny, self.fnx)
        for tag, op in (("fw", pod.fw), ("bw", pod.bw)):
            spec = extract_operator(op, shape, f"farfield {tag}")
            setattr(
                self, f"_ff_{tag}", to_torch(spec, self.cdt, self.dev)
            )

    @staticmethod
    def _apply(spec, w):
        return apply_torch(spec, w)

    def _display_object(self):
        """Slice transmissions for plotting; the tangent subclass, which
        stores log O, overrides this with the exponential."""
        return self.ob_g

    def _plot_progress(self):
        plot_progress(
            self.pr_g,
            self._display_object(),
            self.curiter,
            self.plot_progress_dir,
            psize=getattr(self.p, "psize", (1e-6, 1e-6)),
            engine_name=type(self).__name__,
            cmaps=self.PLOT_CMAPS,
            zoom_center=self.p.get("plot_zoom_center", None),
            phase_clim=self.p.get("plot_phase_clim", None),
            scalebar=self.p.get("plot_scalebar", None),
            phase_zero_top=bool(self.p.get("plot_phase_zero_top", False)),
        )
        plot_error_history(
            self._err_history,
            self.plot_progress_dir,
            engine_name=type(self).__name__,
            channels=self.p.get("plot_error_channels", None),
        )



class ISSEngineInfra(TorchMixin):
    """ISS-family infrastructure on TorchMixin: object/probe tensors
    with warm starts, the cumulative branch propagators, view
    collection with flat-index gather/scatter, and the slice
    write-out. The ISS engines mix this in for the plumbing only.
    """

    def _setup_torch_infra(self):
        self._setup_torch_device()

        N = self.p.number_of_slices
        self._obs = next(iter(self.ob.storages.values()))
        self._prs = next(iter(self.pr.storages.values()))
        self.M, self.fny, self.fnx = self._prs.data.shape

        # Initialize object slices and probe.
        #
        # The start is 1 + 1e-3 * uniform, symmetry-broken so the slices are not
        # degenerate.  It is drawn from self.p.seed rather than from OS entropy:
        # a paired comparison whose two runs begin from different objects has a
        # twin-run difference that is not zero at iteration 0, which is usually
        # the quantity being measured.  seed = 0 leaves the draw free, which is
        # what the original engine did.
        self._object = slice_containers(self.ob, N)
        seed = int(self.p.seed)
        ob0 = np.ones_like(self._obs.data[0])
        rng = np.random.default_rng(seed if seed else None)
        self.ob_g = torch.stack(
            [
                torch.as_tensor(
                    ob0
                    + 1e-3
                    * (
                        (rng.random(ob0.shape) - 0.5)
                        + 1j * (rng.random(ob0.shape) - 0.5)
                    ),
                    dtype=self.cdt,
                    device=self.dev,
                )
                for _ in range(N)
            ]
        )
        logger.info(
            "object start: 1 + 1e-3 noise, seed %s"
            % (seed if seed else "free (not reproducible)")
        )
        self.pr_g = torch.as_tensor(
            np.ascontiguousarray(self._prs.data), dtype=self.cdt, device=self.dev
        )

        # Warm start: overwrite the fresh object/probe init with a previous
        # reconstruction so a run can continue where another stopped.
        ws_ob = getattr(self.p, "warm_start_h5", None)
        if ws_ob:
            import h5py as _h5
            with _h5.File(ws_ob, "r") as _f:
                objs = _f["content/objects"]
                keys = sorted(objs.keys(), key=lambda s: int(s.split("_")[-1]))
                if len(keys) != N:
                    raise ValueError(
                        "warm_start_h5 has %d slices, engine wants %d"
                        % (len(keys), N))
                for i, k in enumerate(keys):
                    arr = objs[k]["Sscan00G00"]["data"][()][0]
                    self.ob_g[i] = torch.as_tensor(
                        arr, dtype=self.cdt, device=self.dev)
            logger.info("warm-start: loaded %d object slices from %s"
                        % (N, ws_ob))
        ws_pr = getattr(self.p, "warm_start_ptyr", None)
        if ws_pr:
            import h5py as _h5
            with _h5.File(ws_pr, "r") as _f:
                g = _f["content/probe"]
                pk = list(g.keys())[0]
                Pw = g[pk]["data"][()]
            if Pw.shape != tuple(self.pr_g.shape):
                raise ValueError("warm_start_ptyr probe shape %s != %s"
                                 % (Pw.shape, tuple(self.pr_g.shape)))
            self.pr_g = torch.as_tensor(Pw, dtype=self.cdt, device=self.dev)
            logger.info("warm-start: loaded probe %s from %s"
                        % (str(Pw.shape), ws_pr))

        self._order_gen = None
        if seed:
            self._order_gen = torch.Generator(device=self.dev)
            self._order_gen.manual_seed(seed + 1)
            logger.info(f"seed = {seed}: the position order is reproducible")
        self.ony, self.onx = self.ob_g.shape[-2:]

        # Precompute Free-Space Propagators (H_z and H_{-z})
        geom = next(iter(self.ptycho.model.scans.values())).geometries[0]
        V, W = u.grids(geom.shape, geom.lam / geom.shape[0] / geom.resolution, "fft")
        a2 = V**2 + W**2
        z_depths = [
            sum(self.p.slice_thickness[:i])
            if isinstance(self.p.slice_thickness, (list, tuple))
            else i * self.p.slice_thickness
            for i in range(1, N)
        ]

        H = np.empty((max(N - 1, 1), *geom.shape), dtype=np.complex128)
        for i, zi in enumerate(z_depths):
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            H[i] = np.exp(2j * np.pi * (zi / geom.lam) * kz) * (a2 <= 1.0)

        self.H = torch.as_tensor(H[: N - 1], dtype=self.cdt, device=self.dev)
        self.Hb = self.H.conj().contiguous() if N > 1 else self.H

        # Build fully broadcastable H arrays for pure math function
        self.H_fwd = torch.ones((N, 1, 1, *geom.shape), dtype=self.cdt, device=self.dev)
        self.H_rev = torch.ones((N, 1, 1, *geom.shape), dtype=self.cdt, device=self.dev)
        if N > 1:
            self.H_fwd[1:, 0, 0] = self.H
            self.H_rev[1:, 0, 0] = self.Hb

        self._build_farfield()
        self._collect_views()

        self._setup_probe_support()
        self._setup_plot_state()

    def _collect_views(self):
        names = sorted(n for n, v in self.di.views.items() if v.active)
        rows, cols, amps, masks = [], [], [], []
        trivial = True
        for n in names:
            pod = next(iter(self.di.views[n].pods.values()))
            lo = np.asarray(pod.ob_view.dlow, dtype=np.int64)
            rows.append(np.arange(lo[0], lo[0] + self.fny))
            cols.append(np.arange(lo[1], lo[1] + self.fnx))
            I = np.asarray(self.di.views[n].data, dtype=np.float64)
            amps.append(np.sqrt(np.maximum(I, 0.0)))
            m = (
                np.ones_like(I, dtype=bool)
                if pod.mask is None
                else np.asarray(pod.mask) > 0
            )
            trivial = trivial and bool(m.all())
            masks.append(m)

        self.n_pos = len(names)
        self.rows = torch.as_tensor(np.stack(rows), dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(np.stack(cols), dtype=torch.long, device=self.dev)
        self.amp_dat = torch.as_tensor(np.stack(amps), dtype=self.rdt, device=self.dev)
        self.maskb = (
            None
            if trivial
            else torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.dev)
        )
        self.view_names = names

        # Illuminated region.  Any statistic taken over a slice is meaningless
        # outside it: most of the object array is never touched and stays at
        # exactly 1, which would drag every rms towards zero by an arbitrary
        # factor.  Read from outside the engine, by diagnostics.
        cov = torch.zeros((self.ony, self.onx), dtype=self.rdt, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            B = idx.numel()
            self._scatter(
                cov,
                self._flat(idx),
                torch.ones((B, self.fny, self.fnx), dtype=self.rdt, device=self.dev),
            )
        self.cov = cov > 0

    def _flat(self, idx):
        return (
            self.rows[idx][:, :, None] * self.onx + self.cols[idx][:, None, :]
        ).reshape(-1)

    def _gather(self, arr, flat, B):
        return arr.reshape(-1)[flat].view(B, self.fny, self.fnx)

    def _scatter(self, dst, flat, vals):
        dst.view(-1).index_put_((flat,), vals.reshape(-1), accumulate=True)

    def _model_object(self, O_patches):
        """What the forward core sees as the slice transmissions.

        The chord engine stores the transmissions O_j themselves, so this is
        the identity and the core's g = O - 1 is the chord perturbation. The
        tangent subclass stores the scattering potential s_j instead and
        returns 1 + i s_j, so the same core computes g = i s_j: the
        linearised transmission.
        """
        return O_patches

    def _grad_to_param(self, grad):
        """Chain rule from the model perturbation g to the stored parameter.

        The chord engine's parameter is O with g = O - 1, so dg/dO = 1 and
        this is the identity. The tangent subclass stores s with g = i s and
        overrides this with the conjugate factor, -i.
        """
        return grad

    def _write_slices(self):
        for i in range(self.p.number_of_slices):
            arr = self.ob_g[i].cpu().numpy()
            for st in self._object[i].storages.values():
                st.data[0][:] = arr
        self._prs.data[:] = self.pr_g.cpu().numpy()

        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        info = Param(
            number_of_slices=self.p.number_of_slices,
            slice_thickness=self.p.slice_thickness,
            alpha=self.p.get("alpha", None),
            objects={
                ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                for ob in self._object
            },
        )

        # We must tell ptypy to ignore its own custom class types when saving to HDF5
        h5opt = io.h5options["UNSUPPORTED"]
        io.h5options["UNSUPPORTED"] = "ignore"

        logger.info(f"saving slices to {self.p.fslices}")
        io.h5write(
            self.p.fslices,
            header={"description": "ISS multislice"},
            content=info,
        )

        io.h5options["UNSUPPORTED"] = h5opt



