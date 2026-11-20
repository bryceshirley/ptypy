"""
Multislice engine for ptychography, backed by ptychomodels.

The Lie-Trotter sweep with 3PIE or Adam driving it. Everything
host-side is in ptychomodels_base, shared with the Born engine.
"""

import torch
from ptychomodels import (
    modulus_projection,
    multislice_gradients,
    multislice_sweep_exit,
    pie3_projection_step,
)

from ptypy.custom.ptychomodels_base import PtychoModelsEngine
from ptypy.engines import register
from ptypy.utils.verbose import logger


@register()
class Multislice_ptychomodels(PtychoModelsEngine):
    """
    Defaults:

    [name]
    default = Multislice_ptychomodels
    type = str
    help =

    [step]
    default = 3pie
    type = str
    help = Step rule: '3pie' or 'adam'
    doc = '3pie' is the sequential projection step; at one slice it is ePIE. 'adam' is a
          descent on the amplitude fidelity instead, taking alpha and beta as the object and
          probe learning rates, so their default of 1.0 has to be lowered.

    [beta_wave]
    default = 1.0
    type = float
    help = Coupling that carries the correction between slices
    doc = 3PIE as published uses one coefficient for both this and the probe update; set
          beta_wave = beta to reproduce that. The default leaves the transport undamped, which
          a deep stack needs to update its front slices.
    """

    STEPS = ("3pie", "adam")
    MODEL = "multislice"

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        self.article = dict(
            title="Ptychographic transmission microscopy in three dimensions "
                  "using a multi-slice approach",
            author="A. M. Maiden et al.",
            journal="J. Opt. Soc. Am. A",
            volume=29,
            year=2012,
            page=1606,
            doi="10.1364/JOSAA.29.001606",
            comment="The 3PIE reconstruction algorithm",
        )
        ptycho_parent.citations.add_article(**self.article)

    def _configure(self):
        N = int(self.p.number_of_slices)
        logger.info(
            "multislice: N = %d slice%s%s, step %s"
            % (N, "" if N == 1 else "s",
               " (no propagation; 3pie is ePIE)" if N == 1 else "", self._step)
        )

    def _core_fn(self):
        return self._adam_core if self._step == "adam" else self._pie_core

    def _on_grid(self, patch):
        """The patch on the propagation grid, padded with ones: the sweep
        multiplies by the transmission, so vacuum outside the patch is 1."""
        if self._pad == 1:
            return patch
        grid = patch.new_ones(patch.shape[:3] + self._grid)
        grid[self._centre] = patch
        return grid

    @torch.no_grad()
    def _update_view(self, pos, do_probe):
        patch = self._on_grid(self._patches(pos))
        probe = self._probe_on_grid()
        measured = (self.amp_dat[pos] ** 2)[None]

        mask = None if self._mask_model is None else self._mask_model[pos]
        pattern, updated, dP = self._core(patch, probe, measured, mask, do_probe)
        self._accumulate_error(pos, pattern)
        self._write_patch(pos, self._crop((updated - patch)[0, 0]))
        if dP is not None:
            self._write_probe(self._crop(dP[0]))

    def _pie_core(self, patch, probe, measured, mask, do_probe):
        """Forward sweep, projection and backward sweep, the region p.compile
        is given. In exit waves rather than far fields: the sweep ends in real
        space and the step wants the corrected exit wave, so a round trip to
        the far field would be four transforms that cancel."""
        exit_wave, entrances = multislice_sweep_exit(patch, probe, self.H1)
        pattern, correction = modulus_projection(self._exit_to_detector(exit_wave), measured)
        if mask is not None:
            correction = correction * mask
        updated, dP = pie3_projection_step(
            patch, exit_wave, entrances, self._exit_from_detector(correction), self.H1,
            beta_obj=float(self.p.alpha), beta_probe=float(self.p.beta),
            beta_wave=float(self.p.beta_wave), update_probe=do_probe,
        )
        return pattern, updated, dP

    def _adam_core(self, patch, probe, measured, weight):
        """The same region on the adam path."""
        return multislice_gradients(
            patch, probe, self.H1, measured, weight=weight,
            to_detector=self._exit_to_detector, from_detector=self._exit_from_detector,
        )

    def _batch_gradients(self, positions):
        patch = self._on_grid(self._patches_batch(positions))
        probe = self._probe_on_grid()
        weight = None if self.maskb is None else self.maskb[positions]
        pattern, grad_O, grad_probe = self._core(
            patch, probe, self.amp_dat[positions] ** 2, weight
        )
        for k, pos in enumerate(positions.tolist()):
            self._accumulate_error(pos, pattern[k][None])
        return self._crop(grad_O), self._crop(grad_probe)
