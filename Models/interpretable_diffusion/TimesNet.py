import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1


def period_softmax(raw_period_score, temperature):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("period weight temperature must be positive and finite")
    return F.softmax(raw_period_score / temperature, dim=1)


def diagnostic_active_probability(values, active_mask, eps=1.0e-8):
    """Normalize detached diagnostic mass over an explicit active set."""
    if values.ndim != 2:
        raise ValueError("diagnostic values must have shape [B,K]")
    mask = active_mask.to(device=values.device, dtype=values.dtype)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(values.shape[0], -1)
    elif mask.shape[0] == 1 and values.shape[0] != 1:
        mask = mask.expand(values.shape[0], -1)
    if mask.shape != values.shape:
        raise ValueError("active_mask must have shape [K] or [B,K]")
    mask = (mask > 0).to(values.dtype)
    active_count = mask.sum(dim=1, keepdim=True)
    if torch.any(active_count == 0):
        raise ValueError("diagnostic normalization requires an active branch")
    mass = values.detach().clamp_min(0) * mask
    total = mass.sum(dim=1, keepdim=True)
    uniform = mask / active_count
    normalized = mass / total.clamp_min(torch.finfo(values.dtype).tiny)
    fallback = total <= float(eps)
    return torch.where(fallback, uniform, normalized), fallback.squeeze(1)


def branch_contribution_statistics(
    branches, effective_weight, active_mask=None, eps=1.0e-8
):
    """Report branch contribution concentration without affecting training."""
    contribution = (
        branches * effective_weight[:, None, None, :]
    ).abs().mean(dim=(1, 2))
    if active_mask is None:
        active_mask = effective_weight > 0
    fraction, fallback = diagnostic_active_probability(
        contribution, active_mask, eps
    )
    statistics = probability_concentration(fraction, eps)
    return fraction, statistics, fallback


def masked_period_probability(raw_probability, hard_mask, eps=1.0e-8):
    probability = raw_probability * hard_mask.unsqueeze(0)
    # ``eps`` is the entropy/log safety constant, not a replacement for a
    # valid (possibly very small) active probability mass.  Clamping a real
    # denominator to 1e-8 leaves rows unnormalised when the global softmax is
    # concentrated on an inactive branch.  Only protect an actual numerical
    # zero here.
    denominator_floor = torch.finfo(probability.dtype).tiny
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(
        denominator_floor
    )


def probability_concentration(probability, eps=1.0e-8):
    entropy = -(
        probability * torch.log(probability.clamp_min(eps))
    ).sum(dim=1)
    top_two = torch.topk(probability, min(2, probability.shape[1]), dim=1).values
    margin = top_two[:, 0] - (top_two[:, 1] if top_two.shape[1] > 1 else 0)
    return {
        "entropy": entropy,
        "shannon_count": torch.exp(entropy),
        "inverse_hhi_count": 1.0 / probability.square().sum(
            dim=1
        ).clamp_min(eps),
        "maximum_weight": probability.max(dim=1).values,
        "top1_top2_margin": margin,
    }


def _rank_distinct_periods(spectrum, fft_length, k):
    """Rank exactly k distinct integer periods; spectrum is [F] or [B,F]."""
    if spectrum.ndim == 1:
        spectrum = spectrum.unsqueeze(0)
        squeeze = True
    elif spectrum.ndim == 2:
        squeeze = False
    else:
        raise ValueError("spectrum must be [F] or [B,F]")
    rows = []
    fallback_count = 0
    possible = sorted(
        {fft_length // frequency for frequency in range(1, spectrum.shape[1])},
        reverse=True,
    )
    if len(possible) < k:
        raise RuntimeError(
            f"requested {k} distinct periods for FFT length {fft_length}, "
            f"but only {len(possible)} are available"
        )
    for row in spectrum:
        # Stable=True gives deterministic frequency-index tie handling.
        ranked = (torch.argsort(row[1:], descending=True, stable=True) + 1).detach().cpu().tolist()
        selected = []
        seen = set()
        for frequency in ranked:
            period = int(fft_length // frequency)
            if period not in seen:
                selected.append(period); seen.add(period)
                if len(selected) == k: break
        if len(selected) < k:
            fallback_count += 1
            period_scores = []
            for period in possible:
                if period in seen: continue
                indices = [f for f in range(1, spectrum.shape[1]) if fft_length // f == period]
                period_scores.append((float(row[indices].sum().detach()), period))
            period_scores.sort(key=lambda item: (-item[0], -item[1]))
            selected.extend(period for _, period in period_scores[:k-len(selected)])
        rows.append(selected)
    result = torch.tensor(rows, device=spectrum.device, dtype=torch.long)
    return (result[0] if squeeze else result), fallback_count


def FFT_for_Period(
    x,
    k=2,
    candidate_scope="batch_shared",
    fft_source="full",
    history_length=None,
    return_diagnostics=False,
):
    """Return exactly ``k`` unique periods ranked by global FFT strength.

    Period identities are selected by global (batch-averaged) frequency
    salience. The per-batch evidence for a selected period is the sum of all
    non-DC frequency amplitudes that map to that integer period.
    """
    B, T, _ = x.shape
    if candidate_scope not in {"batch_shared", "sample_wise"}:
        raise ValueError("period_candidate_scope must be batch_shared or sample_wise")
    if fft_source not in {"full", "history"}:
        raise ValueError("period_fft_source must be full or history")
    if history_length is None:
        history_length = T
    history_length = int(history_length)
    if history_length <= 0 or history_length > T:
        raise ValueError("period_fft_history_length must be in [1, sequence length]")
    source = x if fft_source == "full" else x[:, :history_length, :]
    fft_length = int(source.shape[1])
    xf = torch.fft.rfft(source, dim=1)
    sample_spectrum = abs(xf).mean(-1)
    selection_spectrum = (
        sample_spectrum.mean(0)
        if candidate_scope == "batch_shared"
        else sample_spectrum
    )

    if k < 1:
        raise ValueError("k must be at least 1")
    if sample_spectrum.shape[1] <= 1:
        raise RuntimeError(
            "FFT_for_Period requires at least one non-DC frequency "
            f"(FFT source length: {fft_length})"
        )
    periods, fallback_count = _rank_distinct_periods(selection_spectrum, fft_length, k)
    if candidate_scope == "batch_shared":
        scores = []
        for period in periods.detach().cpu().tolist():
            indices = [f for f in range(1, sample_spectrum.shape[1]) if fft_length // f == period]
            scores.append(sample_spectrum[:, indices].sum(dim=1))
        period_weight = torch.stack(scores, dim=1)
        period_result = periods.detach().cpu().numpy()
    else:
        score_rows = []
        for batch_index in range(B):
            scores = []
            for period in periods[batch_index].detach().cpu().tolist():
                indices = [f for f in range(1, sample_spectrum.shape[1]) if fft_length // f == period]
                scores.append(sample_spectrum[batch_index, indices].sum())
            score_rows.append(torch.stack(scores))
        period_weight = torch.stack(score_rows)
        period_result = periods
    diagnostics = {"fft_length": fft_length, "fallback_count": fallback_count}
    if return_diagnostics:
        return period_result, period_weight, diagnostics
    return period_result, period_weight


def candidate_diversity(periods, batch_size):
    if torch.is_tensor(periods):
        vectors = periods.detach().cpu().tolist()
    else:
        shared = np.asarray(periods, dtype=np.int64).tolist()
        vectors = [shared for _ in range(batch_size)]
    tuples = [tuple(int(v) for v in row) for row in vectors]
    top1 = [row[0] for row in tuples]
    counts = {value: top1.count(value) for value in sorted(set(top1))}
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(top1)
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    unique_vectors = len(set(tuples))
    jaccards = []
    for left in range(len(tuples)):
        for right in range(left + 1, len(tuples)):
            a, b = set(tuples[left]), set(tuples[right])
            jaccards.append(len(a & b) / len(a | b))
    return {
        "unique_top1_periods_in_batch": len(counts),
        "top1_period_entropy": entropy,
        "top1_mode_period": max(counts, key=counts.get),
        "top1_mode_fraction": max(counts.values()) / len(top1),
        "unique_candidate_vectors": unique_vectors,
        "candidate_vector_collision_rate": 1.0 - unique_vectors / len(tuples),
        "mean_pairwise_candidate_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
    }


class TimesBlock(nn.Module):
    def __init__(self, configs, name="timesblock"):
        super(TimesBlock, self).__init__()
        self.name = name

        if isinstance(configs, dict):
            config_value = configs.__getitem__
            temperature = configs.get("k_mask_temperature", 1.0)
            period_weight_temperature = configs.get(
                "period_weight_temperature", 1.0
            )
            period_weight_eps = configs.get("period_weight_eps", 1.0e-8)
            candidate_scope = configs.get("period_candidate_scope", "batch_shared")
            fft_source = configs.get("period_fft_source", "full")
            fft_history_length = configs.get("period_fft_history_length", self.seq_len if hasattr(self, "seq_len") else 64)
        else:
            config_value = lambda key: getattr(configs, key)
            temperature = getattr(configs, "k_mask_temperature", 1.0)
            period_weight_temperature = getattr(
                configs, "period_weight_temperature", 1.0
            )
            period_weight_eps = getattr(configs, "period_weight_eps", 1.0e-8)
            candidate_scope = getattr(configs, "period_candidate_scope", "batch_shared")
            fft_source = getattr(configs, "period_fft_source", "full")
            fft_history_length = getattr(configs, "period_fft_history_length", 64)

        self.seq_len = int(config_value("seq_len"))
        self.d_model = int(config_value("d_model"))
        self.num_kernels = int(config_value("num_kernels"))
        d_ff = int(config_value("d_ff"))
        self.period_candidate_scope = str(candidate_scope)
        self.period_fft_source = str(fft_source)
        self.period_fft_history_length = int(fft_history_length)
        if self.period_candidate_scope not in {"batch_shared", "sample_wise"}:
            raise ValueError("period_candidate_scope must be batch_shared or sample_wise")
        if self.period_fft_source not in {"full", "history"}:
            raise ValueError("period_fft_source must be full or history")
        if not 0 < self.period_fft_history_length <= self.seq_len:
            raise ValueError("period_fft_history_length must be positive and <= seq_len")

        requested_k_min = max(1, int(config_value("top_k")) // 2)
        self.factor = 2.0
        raw_k_max = int(self.factor * math.log(self.seq_len))
        possible_periods = {
            self.seq_len // frequency_index
            for frequency_index in range(1, self.seq_len // 2 + 1)
        }
        self.k_max = min(raw_k_max, len(possible_periods))
        if self.k_max < 1:
            raise ValueError(
                "TimesBlock requires a sequence length with at least one "
                "non-DC frequency and one possible period"
            )
        self.k_min = min(requested_k_min, self.k_max)
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError("expected 1 <= k_min <= k_max")

        self.k_mask_temperature = float(temperature)
        if not math.isfinite(self.k_mask_temperature) or self.k_mask_temperature <= 0:
            raise ValueError("k_mask_temperature must be a finite positive value")

        self.period_weight_temperature = float(period_weight_temperature)
        if (
            not math.isfinite(self.period_weight_temperature)
            or self.period_weight_temperature <= 0
        ):
            raise ValueError(
                "period_weight_temperature must be a finite positive value"
            )
        self.period_weight_eps = float(period_weight_eps)
        if not math.isfinite(self.period_weight_eps) or self.period_weight_eps <= 0:
            raise ValueError("period_weight_eps must be a finite positive value")

        # Keep this parameter name unchanged for checkpoint compatibility.
        self.k_logits = nn.Parameter(torch.randn(1))

        # Keep the convolution module structure/names unchanged.
        self.conv = nn.Sequential(
            Inception_Block_V1(
                self.d_model,
                d_ff,
                num_kernels=self.num_kernels,
            ),
            nn.GELU(),
            Inception_Block_V1(
                d_ff,
                self.d_model,
                num_kernels=self.num_kernels,
            ),
        )

        self.last_k_value = None
        self.last_k_cont = None
        self.last_prefix_mask = None
        self.last_period_list = None
        self.last_period_weight = None
        self.last_effective_period_weight = None
        self.last_raw_period_probability = None
        self.last_period_score_tensor = None
        self.last_concentration_metrics = None
        self.last_candidate_diagnostics = None
        self.last_k_diagnostics = None
        self._current_hard_mask = None
        # PETS-F inference controls. These are plain attributes, not model
        # parameters or buffers, so existing checkpoints remain compatible.
        self.inference_period_candidates = None
        self.inference_period_scores = None
        self.inference_periodic_gate = None

    def set_inference_period_bank(self, candidates, scores, gate=None):
        """Install a sample-wise SH64 bank for one inference batch."""
        candidates = torch.as_tensor(candidates).detach()
        scores = torch.as_tensor(scores).detach()
        if candidates.ndim != 2 or scores.ndim != 2:
            raise ValueError("inference period candidates/scores must be [B,K]")
        if candidates.shape != scores.shape or candidates.shape[1] != self.k_max:
            raise ValueError("inference period bank must match [B,k_max]")
        if torch.any(candidates < 1):
            raise ValueError("inference period candidates must be positive")
        if not torch.isfinite(scores).all() or torch.any(scores < 0):
            raise ValueError("inference period scores must be finite and nonnegative")
        if gate is not None:
            gate = torch.as_tensor(gate).detach().reshape(-1)
            if gate.shape[0] != candidates.shape[0]:
                raise ValueError("inference gate must have one value per sample")
            if not torch.isfinite(gate).all() or torch.any((gate < 0) | (gate > 1)):
                raise ValueError("inference gate must be finite in [0,1]")
        self.inference_period_candidates = candidates
        self.inference_period_scores = scores
        self.inference_periodic_gate = gate

    def clear_inference_period_bank(self):
        self.inference_period_candidates = None
        self.inference_period_scores = None
        self.inference_periodic_gate = None

    def get_k_prefix_mask(self, device, dtype):
        k_norm = torch.sigmoid(self.k_logits)
        k_cont = self.k_min + k_norm * (self.k_max - self.k_min)
        k_cont = torch.clamp(
            k_cont,
            min=float(self.k_min),
            max=float(self.k_max),
        )

        slots = torch.arange(
            1,
            self.k_max + 1,
            device=device,
            dtype=k_cont.dtype,
        )
        hard_count = torch.round(k_cont).clamp(self.k_min, self.k_max)
        hard_mask = (slots <= hard_count.detach()).to(dtype=dtype)
        soft_mask = torch.sigmoid(
            (k_cont - slots + 0.5) / self.k_mask_temperature
        ).to(dtype=dtype)
        prefix_mask = hard_mask + soft_mask - soft_mask.detach()

        # Diagnostics only: detached values never re-enter the loss path.
        self.last_k_value = int(hard_mask.sum().detach().item())
        self.last_k_cont = float(k_cont.detach().item())
        self.last_prefix_mask = hard_mask.detach().cpu()
        self._current_hard_mask = hard_mask.detach()
        nearest_boundary = torch.abs(k_cont - (torch.floor(k_cont) + 0.5))
        self.last_k_diagnostics = {
            "nearest_rounding_boundary_distance": float(nearest_boundary.detach().item()),
            "lower_clamp_distance": float((k_cont - self.k_min).detach().item()),
            "upper_clamp_distance": float((self.k_max - k_cont).detach().item()),
            "active_slot_count": int(hard_mask.sum().detach().item()),
            "inactive_slot_count": int((1 - hard_mask).sum().detach().item()),
        }
        return prefix_mask

    def forward(self, x):
        B, T, N = x.size()
        if self.inference_period_candidates is None:
            period_list, period_weight, candidate_diagnostics = FFT_for_Period(
                x, self.k_max, self.period_candidate_scope, self.period_fft_source,
                self.period_fft_history_length, return_diagnostics=True,
            )
        else:
            period_list = self.inference_period_candidates.to(
                device=x.device, dtype=torch.long
            )
            period_weight = self.inference_period_scores.to(
                device=x.device, dtype=x.dtype
            )
            if tuple(period_list.shape) != (B, self.k_max):
                raise RuntimeError(
                    "frozen inference period-bank batch does not match TimesBlock input"
                )
            candidate_diagnostics = {
                "fft_length": -1,
                "fallback_count": 0,
                "inference_period_bank_override": True,
            }
        if self.period_candidate_scope == "batch_shared":
            assert len(period_list) == self.k_max
        else:
            assert tuple(period_list.shape) == (B, self.k_max)
        assert period_weight.shape[1] == self.k_max

        prefix_mask = self.get_k_prefix_mask(
            device=x.device,
            dtype=period_weight.dtype,
        )
        self.last_period_list = period_list
        self.last_period_weight = period_weight.detach()

        def transform(grouped_x, period):
            grouped_batch = grouped_x.shape[0]
            if T % period != 0:
                length = ((T // period) + 1) * period
                pad = x.new_zeros(grouped_batch, length - T, N)
                out = torch.cat([grouped_x, pad], dim=1)
            else:
                length = T
                out = grouped_x
            out = out.reshape(grouped_batch, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            return out.permute(0, 2, 3, 1).reshape(grouped_batch, -1, N)[:, :T, :]

        if self.period_candidate_scope == "batch_shared":
            branches = torch.stack(
                [transform(x, int(period_list[i])) for i in range(self.k_max)],
                dim=-1,
            )
        else:
            # Group across both sample and ranked-slot dimensions.  This
            # avoids launching the same period convolution once per slot,
            # while index_copy preserves every (sample, slot) assignment.
            flat_periods = period_list.reshape(-1)
            flat_output = x.new_zeros(B * self.k_max, T, N)
            sample_indices = torch.arange(B, device=x.device).repeat_interleave(self.k_max)
            for period in torch.unique(flat_periods, sorted=True).detach().cpu().tolist():
                pair_indices = torch.where(flat_periods == period)[0]
                grouped_x = x.index_select(0, sample_indices.index_select(0, pair_indices))
                flat_output = flat_output.index_copy(0, pair_indices, transform(grouped_x, int(period)))
            branches = flat_output.reshape(B, self.k_max, T, N).permute(0, 2, 3, 1).contiguous()

        # Temperature is applied to the raw FFT evidence before hard masking
        # and active-weight renormalization. T=1 exactly preserves legacy
        # forward weights.
        raw_probability = period_softmax(
            period_weight,
            self.period_weight_temperature,
        )
        # Preserve the exact hard-masked forward probability, but do not let
        # its potentially tiny denominator create an unbounded duplicate STE
        # gradient to k.  k receives its gradient from the stable log-evidence
        # surrogate below; FFT scores still receive the exact raw path.
        pw = raw_probability * prefix_mask.detach().unsqueeze(0)
        denominator = pw.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(pw.dtype).tiny
        )
        pw = pw / denominator

        # Summed FFT evidence can differ by hundreds (especially period=2,
        # which collects many bins), making the required forward softmax
        # numerically one-hot in float32. Preserve that exact forward value,
        # while using a detached log-evidence distribution only as a stable
        # backward surrogate for the STE prefix mask. This does not alter the
        # FFT-selection gradient path or the forward aggregation semantics.
        surrogate_score = (
            torch.log1p(period_weight.detach().clamp_min(0))
            / self.period_weight_temperature
        )
        surrogate_pw = F.softmax(surrogate_score, dim=1)
        surrogate_pw = surrogate_pw * prefix_mask.unsqueeze(0)
        surrogate_denominator = surrogate_pw.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(torch.finfo(surrogate_pw.dtype).tiny)
        surrogate_pw = surrogate_pw / surrogate_denominator
        pw = pw + (surrogate_pw - surrogate_pw.detach())

        # Hard-masked probabilities are retained for diagnostics only.
        hard_mask = self._current_hard_mask.to(
            device=period_weight.device,
            dtype=period_weight.dtype,
        )
        effective_probability = masked_period_probability(
            raw_probability,
            hard_mask,
            1.0e-8,
        )
        raw_statistics = probability_concentration(
            raw_probability,
            1.0e-8,
        )
        hard_k_detached = hard_mask.sum().detach()
        effective_statistics = probability_concentration(
            effective_probability,
            1.0e-8,
        )

        removed_probability_mass = (
            raw_probability * (1.0 - hard_mask).unsqueeze(0)
        ).sum(dim=1)
        retained_probability = 1.0 - removed_probability_mass
        self.last_raw_period_probability = raw_probability.detach()
        self.last_effective_period_weight = pw.detach()
        self.last_period_score_tensor = period_weight
        self.last_candidate_diagnostics = {
            **candidate_diagnostics,
            **candidate_diversity(period_list, B),
        }
        if period_weight.requires_grad:
            period_weight.retain_grad()
        self.last_concentration_metrics = {
            **self.last_k_diagnostics,
            "raw_entropy_mean": float(
                raw_statistics["entropy"].detach().mean().item()
            ),
            "raw_shannon_effective_count_mean": float(
                raw_statistics["shannon_count"].detach().mean().item()
            ),
            "raw_inverse_hhi_count_mean": float(raw_statistics["inverse_hhi_count"].detach().mean().item()),
            "raw_maximum_weight_mean": float(raw_statistics["maximum_weight"].detach().mean().item()),
            "raw_top1_top2_margin_mean": float(raw_statistics["top1_top2_margin"].detach().mean().item()),
            "effective_entropy_mean": float(
                effective_statistics["entropy"].detach().mean().item()
            ),
            "effective_shannon_count_mean": float(
                effective_statistics["shannon_count"].detach().mean().item()
            ),
            "effective_inverse_hhi_count_mean": float(
                effective_statistics["inverse_hhi_count"].detach().mean().item()
            ),
            "maximum_effective_weight_mean": float(
                effective_statistics["maximum_weight"].detach().mean().item()
            ),
            "effective_top1_top2_margin_mean": float(effective_statistics["top1_top2_margin"].detach().mean().item()),
            "removed_probability_mass_mean": float(
                removed_probability_mass.detach().mean().item()
            ),
            "raw_probability_retained_mean": float(retained_probability.detach().mean().item()),
            "effective_count_divided_by_hard_k": float((effective_statistics["shannon_count"] / hard_k_detached.clamp_min(1)).detach().mean().item()),
            "active_weight_ge_0.01_mean": float((effective_probability >= .01).sum(1).float().mean().item()),
            "active_weight_ge_0.05_mean": float((effective_probability >= .05).sum(1).float().mean().item()),
            "active_weight_ge_0.10_mean": float((effective_probability >= .10).sum(1).float().mean().item()),
        }

        res = torch.sum(
            branches * pw[:, None, None, :],
            dim=-1,
        )
        if self.inference_periodic_gate is None:
            return res + x
        gate = self.inference_periodic_gate.to(
            device=x.device, dtype=x.dtype
        ).reshape(B, 1, 1)
        return x + gate * res
