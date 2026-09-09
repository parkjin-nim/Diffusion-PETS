"""Isolated TimesBlock placement variants for the Reviewer #2 ETTh-128 audit.

The production Transformer and TimesBlock sources are intentionally left
untouched.  The proposed placement is inherited exactly.  Alternative
placements only re-route the already existing production TimesBlock, while the
All-Layers architecture is exposed exclusively for forward-cost profiling.
"""

import types

import torch
import torch.nn as nn

from Models.interpretable_diffusion.TimesNet import TimesBlock
from Models.interpretable_diffusion.gaussian_diffusion import (
    Diffusion_TS as ProductionDiffusionTS,
)
from Models.interpretable_diffusion.transformer import apply_timesblock_residual


LAST_BEFORE_D = "e_last_before_d"
MID_D = "e_mid_d"
E_ONLY = "e_only"
ALL_LAYERS = "all_layers_cost_only"
ALLOWED_PLACEMENTS = {LAST_BEFORE_D, MID_D, E_ONLY, ALL_LAYERS}


def _mapping(configs):
    if isinstance(configs, dict):
        return dict(configs)
    if hasattr(configs, "__dict__"):
        return vars(configs).copy()
    raise TypeError("configs must be a mapping or namespace")


class DisabledTimesBlock(nn.Module):
    """Zero-parameter telemetry sentinel for a genuinely absent TimesBlock.

    The historical training engine expects ``decoder.times_block`` to expose
    diagnostic attributes even when auxiliary support losses are disabled. A
    non-callable, zero-parameter sentinel preserves that interface without
    adding a decoder spectral module, spectral call, or trainable parameter.
    """

    def __init__(self, role):
        super().__init__()
        self.role = role
        self.register_buffer("k_logits", torch.zeros(()), persistent=False)
        self.last_k_cont = 0.0
        self.last_k_value = 0
        self.last_prefix_mask = None
        self.last_period_list = None
        self.last_period_weight = None
        self.last_raw_period_probability = None
        self.last_effective_period_weight = None
        self.last_period_score_tensor = None
        self.last_concentration_metrics = {}
        self.last_candidate_diagnostics = {}
        self.last_branch_contribution_vector = None
        self.last_aggregation_fallback_count = 0
        self.inference_period_candidates = None
        self.spectral_call_count = 0

    def forward(self, _x):
        raise RuntimeError("DisabledTimesBlock is a telemetry sentinel and must not run")


def _mid_decoder_forward(self, x, t, enc, padding_masks=None, label_emb=None):
    """D1 -> TimesBlock -> D2 -> D3."""
    x = self.blocks[0](x, enc, t, mask=padding_masks, label_emb=label_emb)
    x = apply_timesblock_residual(x, self.times_block, self.timesblock_residual_mode)
    for block_idx in range(1, len(self.blocks)):
        x = self.blocks[block_idx](
            x, enc, t, mask=padding_masks, label_emb=label_emb
        )
    return x


def _decoder_no_timesblock_forward(
    self, x, t, enc, padding_masks=None, label_emb=None
):
    """D1 -> D2 -> D3 with no decoder-side spectral module."""
    for block in self.blocks:
        x = block(x, enc, t, mask=padding_masks, label_emb=label_emb)
    return x


def _encoder_all_layers_forward(
    self, input, t, padding_masks=None, label_emb=None
):
    x = input
    for block, times_block in zip(self.blocks, self.times_blocks):
        x, _ = block(x, t, mask=padding_masks, label_emb=label_emb)
        x = apply_timesblock_residual(
            x, times_block, self.timesblock_residual_mode
        )
    return x


def _decoder_all_layers_forward(
    self, x, t, enc, padding_masks=None, label_emb=None
):
    for block, times_block in zip(self.blocks, self.times_blocks):
        x = block(x, enc, t, mask=padding_masks, label_emb=label_emb)
        x = apply_timesblock_residual(
            x, times_block, self.timesblock_residual_mode
        )
    return x


class Diffusion_TS(ProductionDiffusionTS):
    """Production diffusion model with one explicit reviewer-only placement."""

    def __init__(self, *args, **kwargs):
        raw_configs = kwargs.get("configs", {})
        configs = _mapping(raw_configs)
        placement = str(configs.get("timesblock_placement", LAST_BEFORE_D))
        if placement not in ALLOWED_PLACEMENTS:
            raise ValueError("unknown TimesBlock placement: {}".format(placement))
        super().__init__(*args, **kwargs)
        self.timesblock_placement = placement

        if placement == MID_D:
            self.model.decoder.forward = types.MethodType(
                _mid_decoder_forward, self.model.decoder
            )
        elif placement == E_ONLY:
            self.model.decoder.times_block = DisabledTimesBlock("decoder")
            self.model.decoder.forward = types.MethodType(
                _decoder_no_timesblock_forward, self.model.decoder
            )
        elif placement == ALL_LAYERS:
            self._install_all_layers(configs)

        for block in self.placement_timesblocks():
            if block.inference_period_candidates is not None:
                raise RuntimeError("frozen inference period banks are prohibited")

    def _install_all_layers(self, configs):
        configs = dict(configs)
        configs["seq_len"] = int(self.seq_length)
        configs["d_model"] = int(self.model.encoder.times_block.d_model)
        encoder = self.model.encoder
        decoder = self.model.decoder
        first_encoder = encoder.times_block
        first_decoder = decoder.times_block
        del encoder.times_block
        del decoder.times_block
        encoder.times_blocks = nn.ModuleList(
            [
                first_encoder,
                TimesBlock(configs, name="encoder_all_2"),
                TimesBlock(configs, name="encoder_all_3"),
            ]
        )
        decoder.times_blocks = nn.ModuleList(
            [
                first_decoder,
                TimesBlock(configs, name="decoder_all_2"),
                TimesBlock(configs, name="decoder_all_3"),
            ]
        )
        encoder.times_block = DisabledTimesBlock("encoder")
        decoder.times_block = DisabledTimesBlock("decoder")
        encoder.forward = types.MethodType(_encoder_all_layers_forward, encoder)
        decoder.forward = types.MethodType(_decoder_all_layers_forward, decoder)

    def placement_timesblocks(self):
        if self.timesblock_placement == ALL_LAYERS:
            return list(self.model.encoder.times_blocks) + list(
                self.model.decoder.times_blocks
            )
        blocks = [self.model.encoder.times_block]
        decoder = self.model.decoder.times_block
        if not isinstance(decoder, DisabledTimesBlock):
            blocks.append(decoder)
        return blocks

    def placement_definition(self):
        if self.timesblock_placement == LAST_BEFORE_D:
            return {
                "encoder": ["after E3 / encoder output"],
                "decoder": ["after D2 / immediately before D3"],
            }
        if self.timesblock_placement == MID_D:
            return {
                "encoder": ["after E3 / encoder output"],
                "decoder": ["after D1 / before D2"],
            }
        if self.timesblock_placement == E_ONLY:
            return {
                "encoder": ["after E3 / encoder output"],
                "decoder": [],
            }
        return {
            "encoder": ["after E1", "after E2", "after E3"],
            "decoder": ["after D1", "after D2", "after D3 / pre-output"],
        }
