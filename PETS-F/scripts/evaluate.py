#!/usr/bin/env python3
"""Compute core deterministic and probabilistic forecast metrics."""

import argparse
import json
from pathlib import Path
import numpy as np


def metrics(samples, truth):
    mean = samples.mean(0)
    forecast_samples = samples[:, :, 64:]
    forecast_truth = truth[:, 64:]
    sample_error = forecast_samples - forecast_truth[None]
    crps = np.abs(sample_error).mean() - 0.5 * np.abs(
        forecast_samples[:, None] - forecast_samples[None, :]
    ).mean()
    return {
        "draws": int(samples.shape[0]),
        "windows": int(samples.shape[1]),
        "ensemble_mean_mae": float(np.abs(mean[:, 64:] - forecast_truth).mean()),
        "ensemble_mean_rmse": float(np.sqrt(np.square(mean[:, 64:] - forecast_truth).mean())),
        "expected_sample_mae": float(np.abs(sample_error).mean()),
        "empirical_crps": float(crps),
        "picp_90": float(
            ((forecast_truth >= np.quantile(forecast_samples, 0.05, axis=0))
             & (forecast_truth <= np.quantile(forecast_samples, 0.95, axis=0))).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = metrics(np.load(args.ensemble), np.load(args.truth))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
