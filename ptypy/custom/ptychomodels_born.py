"""
Inter-slice Born series engine for ptychography, backed by ptychomodels.

The order-M truncation of multislice: order 1 is the inter-slice first Born
model, order = number_of_slices is exact multislice. The sequential depth of
an update is the order rather than the slice count. Everything host-side is
in ptychomodels_base, shared with ptychomodels_multislice.
"""

import torch
from ptychomodels import (
    FirstBornLineSearch,
    born_forward,
    born_gradients,
    born_projection_step,
    centre,
    cumulative_propagators,
    modulus_projection,
)

from ptypy.custom.ptychomodels_base import PtychoModelsEngine
from ptypy.engines import register
from ptypy.utils.verbose import logger


@register()
class IntersliceBorn_ptychomodels(PtychoModelsEngine):
    """
    Defaults:

    [name]
    default = IntersliceBorn_ptychomodels
    type = str
    help =

    [order]
    default = 1
    type = int
    help = Born truncation order M
    doc = 1 is the inter-slice first Born model; M = number_of_slices is exact multislice,
          the series being nilpotent after N orders. Clamped into [1, number_of_slices].

    [background]
    default = mean
    type = str
    help = Background gauge the series expands about: 'vacuum' or 'mean'
    doc = The model expands in O_j = a_j (1 + dO_j) with a_j frozen for the update.
          'vacuum' takes a_j = 1; 'mean' takes the patch mean of each slice, so the
          truncated series carries only the deviation from the average.

    [step]
    default = epie
    type = str
    help = Step rule: 'epie', 'line_search' or 'adam'
    doc = 'epie' is the preconditioned projection step, one position at a time.
          'line_search' sets its step size by the exact minimiser along the same direction
          and needs order 1, the model being affine in the step only there. 'adam' takes
          alpha and beta as the object and probe learning rates, so 1.0 has to be lowered.

    [fidelity]
    default = amplitude
    type = str
    help = Functional the line search minimises: 'amplitude' or 'gaussian'
    doc = 'amplitude' is the masked amplitude l2 the error history reports, minimised by one
          Newton step. 'gaussian' is the intensity Gaussian likelihood, for which the step is
          exact but of a different functional from the one the direction came from.
    """

    STEPS = ("epie", "line_search", "adam")
    MODEL = "inter-slice Born"

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.article = dict(
            title="Parallel-in-depth ptychography of thick samples",
            author="B. Shirley et al.",
            journal="in preparation",
            volume=0,
            year=2026,
            page=0,
            doi="",
            comment="The inter-slice Born series reconstruction algorithm",
        )
        ptycho_parent.citations.add_article(**self.article)

    def _configure(self):
        N = int(self.p.number_of_slices)
        self._M = max(1, min(int(self.p.order), N))
        if int(self.p.order) != self._M:
            logger.warning(
                "order %d clamped to %d (number_of_slices)" % (int(self.p.order), self._M)
            )
        self._background = str(self.p.background or "mean").lower()
        if self._M > 1 and self._step == "line_search":
            raise ValueError(
                "step 'line_search' is defined at order 1 only: beyond it the detector "
                "field is a degree-M polynomial in the step and the likelihood stops "
                "being a quartic. Use 'epie' or 'adam' at order %d." % self._M
            )
        logger.info(
            "inter-slice Born series: order M = %d of N = %d%s, background %s, step %s"
            % (self._M, N, " (exact multislice)" if self._M == N else "",
               self._background, self._step)
        )

    def _build_propagators(self, H1, n_slices):
        self.H = cumulative_propagators(H1, n_slices)

    def _core_fn(self):
        return self._adam_core if self._step == "adam" else self._epie_core

    def _result_info(self, info):
        info.order = self._M
        info.background = self._background

    def _on_grid(self, patch):
        """The patch on the propagation grid, with the gauge taken over the
        patch, as (object, gauge). Outside the patch the grid is filled with
        the background a_j and not with ones: the series expands in O/a - 1,
        so ones would leave a spurious scatterer covering the whole grid."""
        gauge = centre(patch, self.H, self._background)
        if self._pad == 1:
            return patch, gauge
        grid = patch.new_empty(patch.shape[:3] + self._grid)
        grid[...] = gauge.a_slice
        grid[self._centre] = patch
        return grid, gauge._replace(dO=(grid / gauge.a_slice - 1.0).unsqueeze(1))

    @torch.no_grad()
    def _update_view(self, pos, do_probe):
        """One scan position, projection step. adam goes through
        _update_batch instead, at every batch size."""
        patch, gauge = self._on_grid(self._patches(pos))
        probe = self._probe_on_grid()
        measured = (self.amp_dat[pos] ** 2)[None]

        mask = None if self._mask_model is None else self._mask_model[pos]
        pattern, updated, dP = self._core(patch, gauge, probe, measured, mask, do_probe)
        self._accumulate_error(pos, pattern)
        if self._step == "line_search":
            updated, dP = self._searched(pos, patch, gauge, probe, updated, dP)
        self._write_patch(pos, self._crop((updated - patch)[0, 0]))
        if dP is not None:
            self._write_probe(self._crop(dP[0]))

    def _epie_core(self, patch, gauge, probe, measured, mask, do_probe):
        """Forward, projection and step, the region p.compile is given.
        Nothing here mutates ob_g or pr_g, which would rule out CUDA
        graphs."""
        psi_det = self._to_detector(born_forward(gauge, probe, self._M))
        pattern, correction = modulus_projection(psi_det, measured)
        if mask is not None:
            correction = correction * mask
        updated, dP = born_projection_step(
            patch, gauge, probe, self._M, self._from_detector(correction),
            beta_obj=float(self.p.alpha), beta_probe=float(self.p.beta),
            update_probe=do_probe,
        )
        return pattern, updated, dP

    def _searched(self, pos, patch, gauge, probe, updated, dP):
        """Replace the projection's step size by the exact minimiser along the
        same direction. The search rescales the whole direction, so alpha and
        beta play no part and the returned step may be negative."""
        weight = None if self.maskb is None else self.maskb[pos][None]
        search = FirstBornLineSearch(
            gauge, probe, (self.amp_dat[pos] ** 2)[None],
            fidelity=str(self.p.fidelity).lower(), weight=weight,
            # the crop and chirp the forward goes through, or the search
            # minimises on the propagation grid while being scored on the frame
            to_detector=self._to_detector,
        )
        direction = (updated - patch) / gauge.a_slice  # the search works in dO
        gamma = search.direction_object(direction).step()
        search.advance(gamma, direction)
        stepped = patch + gamma.view(-1, 1, 1, 1, 1) * (updated - patch)
        if dP is None:
            return stepped, None
        return stepped, search.direction_probe(dP).step().view(-1, 1, 1, 1) * dP

    def _adam_core(self, patch, gauge, probe, measured, weight):
        """The same region on the adam path."""
        return born_gradients(
            gauge, probe, self._M, measured, weight=weight,
            to_detector=self._to_detector, from_detector=self._from_detector,
        )

    def _batch_gradients(self, positions):
        """The order-M adjoint over a batch. The gauge is per position,
        centre() reducing over the patch axes."""
        patch, gauge = self._on_grid(self._patches_batch(positions))
        probe = self._probe_on_grid()
        weight = None if self.maskb is None else self.maskb[positions]
        pattern, grad_O, grad_probe = self._core(
            patch, gauge, probe, self.amp_dat[positions] ** 2, weight
        )
        for k, pos in enumerate(positions.tolist()):
            self._accumulate_error(pos, pattern[k][None])
        return self._crop(grad_O), self._crop(grad_probe)
