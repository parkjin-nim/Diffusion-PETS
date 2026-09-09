#!/usr/bin/env python3
"""Plot a probabilistic forecast fan chart for one window and feature."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--feature", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples, truth = np.load(args.ensemble), np.load(args.truth)
    value = samples[:, args.window, :, args.feature]
    actual = truth[args.window, :, args.feature]
    x = np.arange(128)
    lo, hi = np.quantile(value, [0.05, 0.95], axis=0)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(x, actual, color="black", label="truth")
    axis.plot(x, value.mean(0), color="#2563eb", label="PETS-F mean")
    axis.fill_between(x, lo, hi, color="#2563eb", alpha=0.22, label="90% interval")
    axis.axvline(63.5, color="gray", ls="--", lw=1)
    axis.set(xlabel="time", ylabel="value")
    axis.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
