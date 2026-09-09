#!/usr/bin/env python3
"""Final PETS-F sampler: fixed SH64 bank, CG64, reconstruction guidance, gamma=1.5."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cg64 import bank_from_histories, clear_bank, confidence_quantiles, install_bank
from pets_runtime import load_ema_model, seed_everything
from Utils.io_utils import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--draw-seeds", type=int, nargs="+", default=list(range(100000, 100010)))
    parser.add_argument("--disable-gate", action="store_true", help="SH64 ablation")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.gamma <= 0:
        raise ValueError("gamma must be positive")
    model, config, device = load_ema_model(args.config, args.checkpoint, args.gpu)
    train_cfg = copy.deepcopy(config["dataloader"]["train_dataset"])
    train_cfg["params"]["save2npy"] = False
    train_cfg["params"]["output_dir"] = str(args.output_dir / "dataset_cache")
    test_cfg = copy.deepcopy(config["dataloader"]["test_dataset"])
    test_cfg.pop("coefficient", None)
    test_cfg.pop("step_size", None)
    test_cfg.pop("sampling_steps", None)
    test_cfg["params"].update(
        predict_length=64, save2npy=False, output_dir=str(args.output_dir / "dataset_cache")
    )
    train_data = instantiate_from_config(train_cfg)
    test_data = instantiate_from_config(test_cfg)
    routing = config["model"]["params"].get("configs", {})
    use_frozen_sh64 = (
        routing.get("period_candidate_scope") == "sample_wise"
        and routing.get("period_fft_source") == "history"
        and int(routing.get("period_fft_history_length", 0)) == 64
    )
    if use_frozen_sh64:
        q10, q90 = confidence_quantiles(np.asarray(train_data.samples)[:, :64])
    loader = DataLoader(
        test_data, batch_size=config["dataloader"]["sample_size"], shuffle=False,
        num_workers=0, drop_last=False,
    )
    normalized_draws = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for draw in args.draw_seeds:
        seed_everything(draw, args.gpu)
        pieces = []
        for index, (target, mask) in enumerate(loader):
            target, mask = target.to(device), mask.to(device)
            if use_frozen_sh64:
                periods, scores, gate = bank_from_histories(
                    target[:, :64].cpu().numpy(),
                    model.model.encoder.times_block.k_max,
                    q10, q90, use_gate=not args.disable_gate,
                )
                install_bank(model, periods, scores, gate)
            try:
                generated = model.fast_sample_infill(
                    target.shape, target,
                    int(config["dataloader"]["test_dataset"]["sampling_steps"]),
                    partial_mask=mask,
                    model_kwargs={
                        "coef": float(config["dataloader"]["test_dataset"]["coefficient"]),
                        "learning_rate": float(config["dataloader"]["test_dataset"]["step_size"]),
                    },
                )
            finally:
                if use_frozen_sh64:
                    clear_bank(model)
            pieces.append(generated.detach().cpu().numpy().astype(np.float32))
            print("draw={} batch={}/{}".format(draw, index + 1, len(loader)), flush=True)
        value = np.concatenate(pieces)
        normalized_draws.append(value)
        draw_dir = args.output_dir / "draw_{}".format(draw)
        draw_dir.mkdir(parents=True, exist_ok=True)
        np.save(draw_dir / "prediction_normalized_uncalibrated.npy", value)
    ensemble = np.stack(normalized_draws).astype(np.float64)
    mean = ensemble.mean(0)
    calibrated = ensemble.copy()
    calibrated[:, :, 64:] = mean[None, :, 64:] + args.gamma * (
        ensemble[:, :, 64:] - mean[None, :, 64:]
    )
    raw_calibrated = np.stack([test_data.unnormalize(draw) for draw in calibrated])
    raw_truth = test_data.unnormalize(np.asarray(test_data.samples))
    np.save(args.output_dir / "ensemble_normalized.npy", calibrated.astype(np.float32))
    np.save(args.output_dir / "ensemble.npy", raw_calibrated.astype(np.float32))
    np.save(args.output_dir / "truth.npy", raw_truth.astype(np.float32))
    print("SAVED:", args.output_dir, "shape=", raw_calibrated.shape)


if __name__ == "__main__":
    main()
