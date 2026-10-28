"""
Per-iteration diagnostic figures for the inter-slice Born engines, any
backend.

Probe intensity, first and final slice phase, each with a zoom on the centre
quarter, plus the convergence curve. Nothing here feeds back into a
reconstruction and nothing here imports cupy: the helpers consume
any array-like exposing .abs(), .angle(), ** and .cpu().numpy() -- host
tensors directly, or host numpy arrays through the _HostView adapter below.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.restoration import unwrap_phase


class _HostView:
    """The thin tensor surface the plot helpers touch (indexing, .abs(),
    .angle(), **, .cpu().numpy(), .shape), wrapped around a host numpy
    array so any backend reuses the figures without a second copy
    of the plotting code."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, k):
        return _HostView(self.a[k])

    def __pow__(self, p):
        return _HostView(self.a ** p)

    def abs(self):
        return _HostView(np.abs(self.a))

    def angle(self):
        return _HostView(np.angle(self.a))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


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

    pr_g        probe modes, (M, ny, nx) complex array-like
    ob_g        object slices, (N, ony, onx) complex array-like
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

