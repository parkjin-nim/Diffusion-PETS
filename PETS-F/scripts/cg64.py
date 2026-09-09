"""Observed-history SH64 bank and parameter-free CG64 confidence gate."""

import math
import numpy as np
import torch

TOP_K = 7


def spectrum_probability(histories, eps=1.0e-8):
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 64:
        raise ValueError("histories must have shape [B,64,C]")
    amplitude = np.abs(np.fft.rfft(values, axis=1)).mean(axis=2)[:, 1:]
    probability = amplitude + eps
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, amplitude


def rank_distinct_periods(probability, history_length=64, k=9):
    frequency = np.arange(1, probability.shape[1] + 1)
    mapped = history_length // frequency
    possible = sorted(set(mapped.tolist()), reverse=True)
    rows = []
    for row in probability:
        selected = []
        for position in np.argsort(-row, kind="stable"):
            period = int(mapped[position])
            if period not in selected:
                selected.append(period)
                if len(selected) == k:
                    break
        selected.extend(v for v in possible if v not in selected)
        rows.append(selected[:k])
    return np.asarray(rows, dtype=np.int64)


def confidence(probability, eps=1.0e-8):
    top_mass = np.sort(probability, axis=1)[:, -TOP_K:].sum(1)
    entropy = -(probability * np.log(probability + eps)).sum(1) / math.log(
        probability.shape[1]
    )
    return top_mass * (1.0 - entropy)


def confidence_quantiles(train_histories):
    probability, _ = spectrum_probability(train_histories)
    value = confidence(probability)
    return float(np.quantile(value, 0.1)), float(np.quantile(value, 0.9))


def bank_from_histories(histories, k, q10=None, q90=None, use_gate=True):
    probability, amplitude = spectrum_probability(histories)
    periods = rank_distinct_periods(probability, 64, k)
    mapped = 64 // np.arange(1, amplitude.shape[1] + 1)
    scores = np.asarray(
        [
            [float(amplitude[row, mapped == period].sum()) for period in periods[row]]
            for row in range(len(periods))
        ],
        dtype=np.float32,
    )
    gate = None
    if use_gate:
        if q10 is None or q90 is None:
            raise ValueError("CG64 requires training-only q10/q90")
        scaled = np.clip((confidence(probability) - q10) / (q90 - q10 + 1.0e-8), 0, 1)
        gate = (0.25 + 0.75 * scaled).astype(np.float32)
    return periods, scores, gate


def install_bank(model, periods, scores, gate):
    device = next(model.parameters()).device
    periods = torch.as_tensor(periods, device=device, dtype=torch.long)
    scores = torch.as_tensor(scores, device=device, dtype=torch.float32)
    gate = None if gate is None else torch.as_tensor(gate, device=device)
    for block in (model.model.encoder.times_block, model.model.decoder.times_block):
        block.period_candidate_scope = "sample_wise"
        block.period_fft_source = "history"
        block.period_fft_history_length = 64
        block.set_inference_period_bank(periods, scores, gate)


def clear_bank(model):
    for block in (model.model.encoder.times_block, model.model.decoder.times_block):
        block.clear_inference_period_bank()
