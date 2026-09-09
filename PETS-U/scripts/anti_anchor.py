"""Frozen symmetric Anti-Anchor aggregation used by the final PETS-U-AA sampler."""

import torch

from Models.interpretable_diffusion.TimesNet import period_softmax

EPS = 1.0e-8


def harmonic_anchor_mask(periods, weights, hard_mask):
    active = hard_mask.bool().unsqueeze(0).expand(weights.shape[0], -1)
    anchor_index = weights.masked_fill(~active, -1).argmax(1, keepdim=True)
    anchor_period = periods.gather(1, anchor_index)
    ratio = torch.maximum(periods, anchor_period).to(weights.dtype) / torch.minimum(
        periods, anchor_period
    ).clamp_min(1).to(weights.dtype)
    return ((ratio - ratio.round()).abs() <= 0.05) & active, active


def halfspace_projection(residual, reference):
    dot = (residual * reference).sum(-1, keepdim=True)
    norm2 = reference.square().sum(-1, keepdim=True)
    valid = norm2 > EPS ** 2
    denominator = torch.where(valid, norm2, torch.ones_like(norm2))
    projected = torch.where(
        valid, residual + ((-dot).clamp_min(0) / denominator) * reference, residual
    )
    for _ in range(2):
        post = (projected * reference).sum(-1, keepdim=True)
        projected = torch.where(
            valid,
            projected + ((-post).clamp_min(0) / denominator) * reference,
            projected,
        )
    return projected


class TimesBlockProbe:
    def __init__(self, block):
        self.block = block
        self.conv_calls = []
        self.handles = [
            block.register_forward_pre_hook(self._pre),
            block.conv.register_forward_hook(self._conv),
            block.register_forward_hook(self._post),
        ]

    def _pre(self, _module, inputs):
        self.input = inputs[0]
        self.conv_calls = []

    def _conv(self, _module, inputs, output):
        period = int(inputs[0].shape[-1])
        transformed = output.permute(0, 2, 3, 1).reshape(
            output.shape[0], -1, output.shape[1]
        )[:, : self.input.shape[1], :]
        self.conv_calls.append((period, transformed))

    def reconstruct(self, module):
        batch, length, features = self.input.shape
        periods = torch.as_tensor(module.last_period_list, device=self.input.device).long()
        if periods.ndim == 1:
            periods = periods.unsqueeze(0).expand(batch, -1)
        kmax = periods.shape[1]
        branches = self.input.new_zeros(batch * kmax, length, features)
        flat = periods.reshape(-1)
        for period, transformed in self.conv_calls:
            positions = torch.where(flat == period)[0]
            if transformed.shape[0] != positions.numel():
                raise RuntimeError("cannot align period branches")
            branches = branches.index_copy(0, positions, transformed)
        branches = branches.reshape(batch, kmax, length, features).permute(0, 2, 3, 1)
        scores = module.last_period_score_tensor
        hard_mask = module._current_hard_mask.to(scores)
        weights = period_softmax(scores, module.period_weight_temperature)
        weights = weights * hard_mask.unsqueeze(0)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        return periods, hard_mask, weights, branches

    def _post(self, module, _inputs, output):
        periods, hard_mask, weights, branches = self.reconstruct(module)
        anchor_mask, active = harmonic_anchor_mask(periods, weights, hard_mask)
        anchor_weight = weights * anchor_mask.to(weights)
        anchor_weight = anchor_weight * (
            weights.sum(1, keepdim=True)
            / anchor_weight.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        )
        anchor = (branches * anchor_weight[:, None, None, :]).sum(-1)
        original = (
            branches * (weights * active.to(weights))[:, None, None, :]
        ).sum(-1)
        residual = (original.double() - anchor.double()).to(original.dtype)
        for _ in range(2):
            residual = residual + (original - (anchor + residual))
        safe = halfspace_projection(residual, anchor)
        return self.input + anchor + safe

    def close(self):
        for handle in self.handles:
            handle.remove()


class AntiAnchorContext:
    """Apply the frozen operator at either or both TimesBlock locations."""

    def __init__(self, model, encoder=True, decoder=True):
        if not encoder and not decoder:
            raise ValueError("Anti-Anchor requires encoder and/or decoder")
        self.hooks = []
        if encoder:
            self.hooks.append(TimesBlockProbe(model.model.encoder.times_block))
        if decoder:
            self.hooks.append(TimesBlockProbe(model.model.decoder.times_block))

    def close(self):
        for hook in self.hooks:
            hook.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
