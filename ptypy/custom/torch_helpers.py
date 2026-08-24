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


def plot_progress(pr_g, ob_g, curiter, outdir, psize=(1e-6, 1e-6), crop=512):
    """
    Save the iteration figure for one engine state.

    pr_g    probe modes, (M, ny, nx) complex torch tensor
    ob_g    object slices, (N, ony, onx) complex torch tensor
    curiter iteration number, used in the title and the file name
    outdir  directory for the PNG
    psize   object pixel size (dy, dx) in metres, for the axis extents
    crop    pixels trimmed from every edge of the object slices
    """
    # 1. Extract Data
    probe = (pr_g[0].abs() ** 2).cpu().numpy()
    first = ob_g[0].angle().cpu().numpy()
    final = ob_g[-1].angle().cpu().numpy()
    n_slices = int(ob_g.shape[0])

    # 2. Crop Object Slices ONLY (Keep Probe full-sized)
    d = crop // 2
    first_cropped = first[d:-d, d:-d]
    final_cropped = final[d:-d, d:-d]

    # 3. Phase Processing on Object Slices
    rec_ph_0 = unwrap_phase(first_cropped)
    rec_ph_1 = unwrap_phase(final_cropped)

    # Offset removal
    try:
        rec_ph_0 = rec_ph_0 - np.mean(rec_ph_0[124:127, 640:680])
        rec_ph_1 = rec_ph_1 - np.mean(rec_ph_1[100:250, 350:500])
    except IndexError:
        pass

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
    plt.suptitle(f"Iteration {curiter} Progress")

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
            "title": "First Object Slice",
            "cmap": "Greys_r",
            "label": "",
        },
        {
            "data": rec_ph_1,
            "extent": extent_obj,
            "title": f"Final Slice ({n_slices - 1})",
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
