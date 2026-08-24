"""
No-op extension points for the multislice torch engines.

An engine calls these at fixed places in its lifecycle and knows nothing about
what, if anything, answers. A mixin listed *before* this class in the bases
overrides one or more of them; with no such mixin the engine runs exactly as it
would without the hooks at all.

    class FirstBornTorch(stochastic.EPIE, SomeMixin, EngineHooks):

Method resolution puts SomeMixin ahead of EngineHooks, so its overrides win, and
deleting it from the bases falls back to the no-ops here. That is the point: an
experiment bolted onto an engine must be removable without editing the engine.

`SliceCorruptionMixin` below is the one mixin that currently uses them.
"""

import numpy as np

from ptypy.utils.verbose import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class EngineHooks:
    """The hooks, all inert."""

    def post_initialize(self):
        """Called at the end of engine_initialize, once the GPU state exists."""

    def pre_iteration(self):
        """Called at the top of every iteration, before the position order."""

    def extra_slices_info(self):
        """Extra provenance merged into the saved slice file."""
        return {}


# ---------------------------------------------------------------------------#
#  Deliberate corruption of one object slice                                  #
# ---------------------------------------------------------------------------#
# `corrupt_slice` damages one slice on purpose, to measure how far the damage
# travels to the others. The two forward models carry it differently, and that
# difference is the measurement:
#
#   * First Born. The slices enter as an affine sum, so no slice sits in
#     another's optical path and the per-slice normal operator is exactly
#     diagonal. A bad slice reaches another slice's gradient only through the
#     shared detector residual.
#
#   * Multislice (3PIE). The damage reaches the other slices through the
#     operator as well, and asymmetrically: an error upstream corrupts the
#     illumination every slice below it sees, while an error downstream
#     corrupts the path the residual is carried back along. Corrupting the
#     first slice and then the last, with an identical perturbation, separates
#     the two channels.
#
# What makes a corruption hard to undo is its direction, not its size. A
# perturbation finer than the shear a_c = 2 NA dz is depth resolvable, so the
# data says which slice it belongs to and the gradient removes it; one coarser
# than a_c is not, and its antisymmetric part is close to null. Size
# `corrupt_smooth` against a_c, not against the pixel.
#
# The engines share one copy: the measurement is a difference between the two,
# so any asymmetry in how they damage a slice -- the modes, the seeding, the
# statistics -- would be read as a property of the forward model.
#
# Nothing in the engines refers to this class except to list it in their bases.
# It attaches through the hooks above, so removing it leaves each engine
# running as if the experiment had never existed.


class SliceCorruptionMixin(EngineHooks):
    """
    Corruption of one slice, mixed into an engine holding its slices on the GPU.

    Listed in the bases ahead of EngineHooks, it takes over `post_initialize`,
    `pre_iteration` and `extra_slices_info`; the engine calls those three and
    names nothing here.

    The host engine must provide, before the first call:

        self.p              the corrupt_* parameters below, and number_of_slices
        self.ob_g           object slices, (N, ony, onx) complex torch tensor
        self.dev            torch device
        self.cdt, self.rdt  complex and real working dtypes
        self._lam           wavelength, m           (for corrupt_mode 'defocus')
        self._dxo           object pixel size, m    (for corrupt_mode 'defocus')
        self.curiter        current iteration

    Defaults:

    [corrupt_slice]
    default = none
    type = str
    help = Deliberately damage one slice: 'none', 'first', 'last' or an index
    doc = For measuring how far an error in one slice travels to the others. In
          the first Born model only the shared residual carries it, because the
          operator is built from H_i and phi_i and contains no object. In
          multislice it travels through the operator as well, and
          asymmetrically: upstream it corrupts the illumination, downstream the
          return path. Corrupting first and then last, with an identical
          perturbation, separates the two channels -- and the difference
          between the two engines is the asymmetry this experiment measures.

    [corrupt_mode]
    default = phase
    type = str
    help = phase, amplitude, antisym, duplicate, constant, vacuum, shift, swap, defocus
    doc = What makes a corruption hard to undo is its direction, not its size.
          A perturbation finer than the shear a_c = 2 NA dz is depth
          resolvable, so the data says which slice it belongs to and the
          gradient removes it; one coarser than a_c is not, and its
          antisymmetric part is close to null. Set corrupt_smooth against
          a_c, in object pixels.
            'phase' / 'amplitude'  smooth random field on one slice
            'antisym'    the same field added to one slice and subtracted from
                         the other end slice: the depth-differential direction
            'duplicate'  another slice's content multiplied on top of this one,
                         so the corruption is a plausible object, not noise
            'constant'   a uniform phase offset, exactly null: a control that
                         should persist untouched and change no recorded value
            'vacuum' erases the slice; 'shift' translates it by
            corrupt_amplitude pixels; 'swap' exchanges first and last;
            'defocus' propagates it by corrupt_amplitude metres. Those four
            need corrupt_at_iter > 0 to mean anything.

    [corrupt_amplitude]
    default = 1.0
    type = float
    help = Strength: rad rms, or pixels, or metres, depending on corrupt_mode
    doc = Amplitude saturates: |g| = 2|sin(phi/2)| never exceeds 2, so phi = pi
          buys only 2.7 times phi = 1. Coarsening the field keeps working.

    [corrupt_smooth]
    default = 8.0
    type = float
    help = Correlation length of the random field, in pixels
    doc = White noise sits mostly outside the transfer function and would be
          filtered away in a few iterations. Compare with the shear
          a_c = 2 NA dz, in object pixels.

    [corrupt_at_iter]
    default = 0
    type = int
    help = Iteration at which to apply the corruption; 0 means at start-up

    [corrupt_seed]
    default = 1234
    type = int
    help = Seed for the corruption field
    doc = Fixed so that corrupting the first slice and corrupting the last use
          an identical perturbation, and any asymmetry belongs to the model.
          Must match the value given to the engine it is compared against.

    """

    def _corrupt_index(self):
        c = str(self.p.corrupt_slice).lower()
        N = int(self.p.number_of_slices)
        if c in ("none", "", "off", "false"):
            return None
        if c == "first":
            return 0
        if c == "last":
            return N - 1
        try:
            return int(c) % N
        except ValueError:
            logger.warning(f"corrupt_slice = {c!r} not understood; ignoring")
            return None

    def post_initialize(self):
        """corrupt_at_iter = 0 damages the slice before the first iteration."""
        self._corruption_applied = False
        if int(self.p.corrupt_at_iter) <= 0:
            self._corrupt_once()

    def pre_iteration(self):
        """Otherwise it is damaged on reaching corrupt_at_iter, exactly once."""
        if (
            not getattr(self, "_corruption_applied", False)
            and int(self.p.corrupt_at_iter) > 0
            and int(self.curiter) >= int(self.p.corrupt_at_iter)
        ):
            self._corrupt_once()

    def _corrupt_once(self):
        self._corruption_applied = True
        if self._corrupt_index() is not None:
            self._apply_corruption()

    def _smooth_field(self, gen):
        """Zero-mean random field on the object grid, correlation length
        corrupt_smooth pixels, unit rms."""
        ny, nx = int(self.ob_g.shape[-2]), int(self.ob_g.shape[-1])
        r = torch.randn((ny, nx), generator=gen, device=self.dev, dtype=torch.float32)
        sig = float(self.p.corrupt_smooth)
        if sig > 0:
            fy = torch.fft.fftfreq(ny, device=self.dev)
            fx = torch.fft.fftfreq(nx, device=self.dev)
            k2 = fy[:, None] ** 2 + fx[None, :] ** 2
            g = torch.exp(-2.0 * (np.pi * sig) ** 2 * k2)
            r = torch.fft.ifft2(torch.fft.fft2(r.to(torch.complex64)) * g).real
        r = r - r.mean()
        return (r / r.std().clamp_min(1e-12)).to(self.rdt)

    def _apply_corruption(self):
        j = self._corrupt_index()
        if j is None:
            return
        N = int(self.p.number_of_slices)
        mode = str(self.p.corrupt_mode).lower()
        amp = float(self.p.corrupt_amplitude)
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(int(self.p.corrupt_seed))
        k = (N - 1) if j == 0 else 0

        if mode == "phase":
            self.ob_g[j] = self.ob_g[j] * torch.exp(
                1j * (amp * self._smooth_field(gen)).to(self.cdt)
            )
        elif mode in ("amplitude", "amp"):
            self.ob_g[j] = self.ob_g[j] * (1.0 + amp * self._smooth_field(gen)).to(
                self.cdt
            )
        elif mode in ("antisym", "antisymmetric"):
            fld = (amp * self._smooth_field(gen)).to(self.cdt)
            self.ob_g[j] = self.ob_g[j] * torch.exp(1j * fld)
            self.ob_g[k] = self.ob_g[k] * torch.exp(-1j * fld)
            logger.info(f"corruption: antisymmetric between slices {j} and {k}")
        elif mode == "duplicate":
            self.ob_g[j] = self.ob_g[j] * self.ob_g[k]
            logger.info(f"corruption: slice {j} multiplied by slice {k}")
        elif mode == "constant":
            self.ob_g[j] = self.ob_g[j] * np.exp(1j * amp)
        elif mode == "vacuum":
            self.ob_g[j] = torch.ones_like(self.ob_g[j])
        elif mode == "shift":
            n = round(amp)
            self.ob_g[j] = torch.roll(self.ob_g[j], shifts=(n, n), dims=(0, 1))
        elif mode == "swap":
            tmp = self.ob_g[j].clone()
            self.ob_g[j] = self.ob_g[k]
            self.ob_g[k] = tmp
            logger.info(f"corruption: slices {j} and {k} exchanged")
        elif mode == "defocus":
            ny, nx = int(self.ob_g.shape[-2]), int(self.ob_g.shape[-1])
            fy = np.fft.fftfreq(ny, d=self._dxo)
            fx = np.fft.fftfreq(nx, d=self._dxo)
            a2 = (self._lam * fy)[:, None] ** 2 + (self._lam * fx)[None, :] ** 2
            kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
            Hd = np.exp(2j * np.pi * (amp / self._lam) * kz)
            Hd[a2 > 1.0] = 0.0
            Ht = torch.as_tensor(Hd, dtype=self.cdt, device=self.dev)
            self.ob_g[j] = torch.fft.ifft2(Ht * torch.fft.fft2(self.ob_g[j]))
        else:
            logger.warning(f"corrupt_mode = {mode!r} not understood; nothing done")
            return

        logger.warning(
            f"CORRUPTION APPLIED at iteration {self.curiter}: slice {j} of {N}, "
            f"mode '{mode}', amplitude {amp:.4g}, "
            f"smooth {float(self.p.corrupt_smooth):.4g} px, "
            f"seed {int(self.p.corrupt_seed)}"
        )

    def extra_slices_info(self):
        """The corruption settings, for the slice file."""
        return {
            "corrupt_slice": str(self.p.corrupt_slice),
            "corrupt_mode": str(self.p.corrupt_mode),
            "corrupt_amplitude": float(self.p.corrupt_amplitude),
            "corrupt_smooth": float(self.p.corrupt_smooth),
            "corrupt_at_iter": int(self.p.corrupt_at_iter),
            "corrupt_seed": int(self.p.corrupt_seed),
        }
