"""Isolated ETTh-128 representation alternatives for Reviewer #2.

This module deliberately leaves the production TimesNet implementation
untouched.  The default period-aligned PETS-U path is inherited byte-for-byte;
only configurations that explicitly request one of the two reviewer controls
replace the encoder and decoder TimesBlocks.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.interpretable_diffusion.TimesNet import (
    TimesBlock,
    FFT_for_Period,
    branch_contribution_statistics,
    candidate_diversity,
    diagnostic_active_probability,
    masked_period_probability,
    period_softmax,
    probability_concentration,
)
from Models.interpretable_diffusion.gaussian_diffusion import (
    Diffusion_TS as ProductionDiffusionTS,
)


PERIOD_ALIGNED = "period_aligned_2d"
STFT_SPECTROGRAM = "stft_spectrogram"
FREQUENCY_AS_CHANNEL = "frequency_as_channel"
ALLOWED_STRATEGIES = {
    PERIOD_ALIGNED,
    STFT_SPECTROGRAM,
    FREQUENCY_AS_CHANNEL,
}


def _mapping(configs):
    if isinstance(configs, dict):
        return dict(configs)
    if hasattr(configs, "__dict__"):
        return vars(configs).copy()
    raise TypeError("configs must be a mapping or namespace")


class AlternativeRepresentationTimesBlock(TimesBlock):
    """TimesBlock-compatible STFT or frequency-channel representation.

    Candidate selection, learnable hard top-k, canonical spectral weights,
    output shape, and residual interface are inherited from the production
    block.  Only the construction of candidate branch tensors changes.
    """

    def __init__(self, configs, name="timesblock"):
        cfg = _mapping(configs)
        strategy = str(cfg.get("representation_strategy", ""))
        if strategy not in {STFT_SPECTROGRAM, FREQUENCY_AS_CHANNEL}:
            raise ValueError("AlternativeRepresentationTimesBlock needs an alternative strategy")
        super().__init__(cfg, name=name)
        if self.period_candidate_scope != "batch_shared":
            raise ValueError("reviewer comparison requires batch-shared candidates")
        if self.period_fft_source != "full":
            raise ValueError("reviewer comparison requires full-sequence spectral evidence")

        self.representation_strategy = strategy
        # Remove the production Inception block parameters allocated by the
        # parent constructor.  This replacement is local to this module.
        self.conv = nn.Identity()
        if strategy == STFT_SPECTROGRAM:
            self.n_fft = int(cfg.get("stft_n_fft", 32))
            self.win_length = int(cfg.get("stft_win_length", 32))
            self.hop_length = int(cfg.get("stft_hop_length", 8))
            self.stft_center = bool(cfg.get("stft_center", False))
            if (self.n_fft, self.win_length, self.hop_length, self.stft_center) != (32, 32, 8, False):
                raise ValueError("STFT configuration is frozen at 32/32/8/center=False")
            hidden = int(cfg.get("stft_hidden_channels", 332))
            if hidden != 332:
                raise ValueError("STFT hidden channels are frozen at 332")
            self.stft_processor = nn.Sequential(
                nn.Conv2d(2 * self.d_model, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, self.d_model, kernel_size=3, padding=1),
            )
            self.register_buffer(
                "stft_hann_window",
                torch.hann_window(self.win_length),
                persistent=False,
            )
        else:
            embedding = int(cfg.get("frequency_channel_embedding", 256))
            hidden = int(cfg.get("frequency_channel_hidden", 470))
            if (embedding, hidden) != (256, 470):
                raise ValueError("frequency-as-channel capacity is frozen at 256/470")
            self.frequency_channel_embedding = embedding
            self.spectral_projection = nn.Linear(2 * self.d_model, embedding)
            self.temporal_processor = nn.Sequential(
                nn.Conv1d(
                    self.d_model + embedding,
                    hidden,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv1d(hidden, self.d_model, kernel_size=3, padding=1),
            )

        self.representation_call_count = 0
        self.last_input_shape = None
        self.last_internal_representation_shape = None
        self.last_output_shape = None
        self.last_selected_frequency_indices = None
        self.last_selected_stft_bins = None

    def _representative_frequencies(self, x, periods):
        """Recover the ranked frequency that introduced every distinct period."""
        spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        ranked = torch.argsort(spectrum[1:], descending=True, stable=True) + 1
        selected = []
        for period in np.asarray(periods, dtype=np.int64).tolist():
            match = None
            for frequency in ranked.detach().cpu().tolist():
                if int(x.shape[1] // frequency) == int(period):
                    match = int(frequency)
                    break
            if match is None:
                raise RuntimeError("could not map selected period back to a frequency")
            selected.append(match)
        return torch.tensor(selected, device=x.device, dtype=torch.long)

    def _candidate_state(self, x):
        periods, scores, diagnostics = FFT_for_Period(
            x,
            self.k_max,
            candidate_scope="batch_shared",
            fft_source="full",
            history_length=self.seq_len,
            return_diagnostics=True,
        )
        frequencies = self._representative_frequencies(x, periods)
        prefix_mask = self.get_k_prefix_mask(x.device, scores.dtype)
        raw_probability = period_softmax(scores, self.period_weight_temperature)
        weighted = raw_probability * prefix_mask.detach().unsqueeze(0)
        weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weighted.dtype).tiny
        )
        surrogate = F.softmax(
            torch.log1p(scores.detach().clamp_min(0))
            / self.period_weight_temperature,
            dim=1,
        )
        surrogate = surrogate * prefix_mask.unsqueeze(0)
        surrogate = surrogate / surrogate.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(surrogate.dtype).tiny
        )
        effective = weighted + surrogate - surrogate.detach()
        aggregation_fallback = torch.zeros(
            x.shape[0], device=x.device, dtype=torch.bool
        )

        hard_mask = self._current_hard_mask.to(scores)
        effective_probability = masked_period_probability(
            raw_probability, hard_mask, 1.0e-8
        )
        diagnostic_probability, diagnostic_fallback = diagnostic_active_probability(
            effective_probability, hard_mask, 1.0e-8
        )
        hard_k = hard_mask.sum().detach()
        self.last_period_list = periods
        self.last_period_weight = scores.detach()
        self.last_raw_period_probability = raw_probability.detach()
        self.last_effective_period_weight = effective.detach()
        self.last_period_score_tensor = scores
        self.last_candidate_diagnostics = {
            **diagnostics,
            **candidate_diversity(periods, x.shape[0]),
        }
        self.last_selected_frequency_indices = frequencies.detach().cpu()
        self.last_aggregation_fallback_count = int(
            aggregation_fallback.detach().sum().item()
        )
        if scores.requires_grad:
            scores.retain_grad()

        raw_stats = probability_concentration(raw_probability, 1.0e-8)
        effective_stats = probability_concentration(
            diagnostic_probability, 1.0e-8
        )
        removed = (raw_probability * (1.0 - hard_mask).unsqueeze(0)).sum(1)
        self.last_concentration_metrics = {
            **self.last_k_diagnostics,
            "raw_entropy_mean": float(raw_stats["entropy"].detach().mean()),
            "raw_shannon_effective_count_mean": float(
                raw_stats["shannon_count"].detach().mean()
            ),
            "raw_inverse_hhi_count_mean": float(
                raw_stats["inverse_hhi_count"].detach().mean()
            ),
            "raw_maximum_weight_mean": float(
                raw_stats["maximum_weight"].detach().mean()
            ),
            "raw_top1_top2_margin_mean": float(
                raw_stats["top1_top2_margin"].detach().mean()
            ),
            "effective_entropy_mean": float(
                effective_stats["entropy"].detach().mean()
            ),
            "effective_shannon_count_mean": float(
                effective_stats["shannon_count"].detach().mean()
            ),
            "effective_inverse_hhi_count_mean": float(
                effective_stats["inverse_hhi_count"].detach().mean()
            ),
            "maximum_effective_weight_mean": float(
                effective_stats["maximum_weight"].detach().mean()
            ),
            "effective_top1_top2_margin_mean": float(
                effective_stats["top1_top2_margin"].detach().mean()
            ),
            "removed_probability_mass_mean": float(removed.detach().mean()),
            "raw_probability_retained_mean": float((1.0 - removed).detach().mean()),
            "effective_count_divided_by_hard_k": float(
                (effective_stats["shannon_count"] / hard_k.clamp_min(1))
                .detach()
                .mean()
            ),
            "active_weight_ge_0.01_mean": float(
                (effective_probability >= 0.01).sum(1).float().mean()
            ),
            "active_weight_ge_0.05_mean": float(
                (effective_probability >= 0.05).sum(1).float().mean()
            ),
            "active_weight_ge_0.10_mean": float(
                (effective_probability >= 0.10).sum(1).float().mean()
            ),
            "aggregation_fallback_count": self.last_aggregation_fallback_count,
            "diagnostic_effective_probability_fallback_count": int(
                diagnostic_fallback.detach().sum().item()
            ),
            "input_latent_mean": float(x.detach().mean()),
            "input_latent_std": float(x.detach().std(unbiased=False)),
        }
        return periods, frequencies, scores, effective, hard_mask

    def _stft_branches(self, x, frequencies):
        batch, length, channels = x.shape
        transformed = torch.stft(
            x.permute(0, 2, 1).reshape(batch * channels, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_hann_window.to(device=x.device, dtype=x.dtype),
            center=False,
            return_complex=True,
        ).reshape(batch, channels, self.n_fft // 2 + 1, -1)
        bins = torch.round(
            frequencies.to(x.dtype) * float(self.n_fft) / float(length)
        ).to(torch.long).clamp(1, self.n_fft // 2)
        selected = transformed.index_select(2, bins)
        representation = torch.cat((selected.real, selected.imag), dim=1)
        self.last_selected_stft_bins = bins.detach().cpu()
        self.last_internal_representation_shape = tuple(representation.shape)
        processed = self.stft_processor(representation)
        frames = processed.shape[-1]
        branch_batch = (
            processed.permute(0, 2, 1, 3)
            .reshape(batch * self.k_max, channels, frames)
        )
        upsampled = F.interpolate(
            branch_batch, size=length, mode="linear", align_corners=False
        )
        return (
            upsampled.reshape(batch, self.k_max, channels, length)
            .permute(0, 3, 2, 1)
            .contiguous()
        )

    def _frequency_channel_branches(self, x, frequencies):
        batch, length, channels = x.shape
        coefficients = torch.fft.rfft(x, dim=1).index_select(1, frequencies)
        representation = torch.cat(
            (coefficients.real, coefficients.imag), dim=-1
        )
        self.last_internal_representation_shape = tuple(representation.shape)
        embedding = self.spectral_projection(representation)
        temporal = (
            x.permute(0, 2, 1)
            .unsqueeze(1)
            .expand(batch, self.k_max, channels, length)
        )
        spectral = embedding.unsqueeze(-1).expand(
            batch, self.k_max, self.frequency_channel_embedding, length
        )
        merged = torch.cat((temporal, spectral), dim=2).reshape(
            batch * self.k_max,
            channels + self.frequency_channel_embedding,
            length,
        )
        processed = self.temporal_processor(merged)
        return (
            processed.reshape(batch, self.k_max, channels, length)
            .permute(0, 3, 2, 1)
            .contiguous()
        )

    def forward(self, x):
        if self.inference_period_candidates is not None:
            raise RuntimeError("frozen inference period banks are prohibited in this experiment")
        if tuple(x.shape[1:]) != (self.seq_len, self.d_model):
            raise ValueError("unexpected TimesBlock input shape")
        self.representation_call_count += 1
        self.last_input_shape = tuple(x.shape)
        _, frequencies, _, weights, hard_mask = self._candidate_state(x)
        if self.representation_strategy == STFT_SPECTROGRAM:
            branches = self._stft_branches(x, frequencies)
        else:
            branches = self._frequency_channel_branches(x, frequencies)

        contribution, stats, fallback = branch_contribution_statistics(
            branches, weights.detach(), hard_mask, self.period_weight_eps
        )
        self.last_branch_contribution_vector = contribution.detach().cpu()
        self.last_concentration_metrics.update(
            {
                "branch_contribution_entropy_mean": float(
                    stats["entropy"].detach().mean()
                ),
                "branch_contribution_shannon_count_mean": float(
                    stats["shannon_count"].detach().mean()
                ),
                "branch_contribution_inverse_hhi_count_mean": float(
                    stats["inverse_hhi_count"].detach().mean()
                ),
                "branch_contribution_maximum_fraction_mean": float(
                    stats["maximum_weight"].detach().mean()
                ),
                "branch_contribution_zero_output_fallback_count": int(
                    fallback.detach().sum().item()
                ),
            }
        )
        residual = torch.sum(branches * weights[:, None, None, :], dim=-1)
        output = x + residual
        self.last_output_shape = tuple(output.shape)
        self.last_concentration_metrics.update(
            {
                "output_latent_mean": float(output.detach().mean()),
                "output_latent_std": float(output.detach().std(unbiased=False)),
            }
        )
        return output


class Diffusion_TS(ProductionDiffusionTS):
    """Production diffusion backbone with an explicit reviewer-only strategy."""

    def __init__(self, *args, **kwargs):
        raw_configs = kwargs.get("configs", {})
        cfg = _mapping(raw_configs)
        strategy = str(cfg.get("representation_strategy", PERIOD_ALIGNED))
        if strategy not in ALLOWED_STRATEGIES:
            raise ValueError("unknown representation strategy: {}".format(strategy))
        super().__init__(*args, **kwargs)
        self.representation_strategy = strategy
        if strategy == PERIOD_ALIGNED:
            return
        cfg["seq_len"] = int(self.seq_length)
        cfg["d_model"] = int(self.model.encoder.times_block.d_model)
        self.model.encoder.times_block = AlternativeRepresentationTimesBlock(
            cfg, name="encoder"
        )
        self.model.decoder.times_block = AlternativeRepresentationTimesBlock(
            cfg, name="decoder"
        )
