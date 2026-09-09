#!/usr/bin/env python3
"""Create a compact real-versus-generated unconditional comparison."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=int, default=7)
    args = parser.parse_args()
    real, fake = np.load(args.real), np.load(args.fake)
    count = min(args.features, real.shape[-1])
    fig, axes = plt.subplots(count, 1, figsize=(10, 2.2 * count), sharex=True)
    axes = np.atleast_1d(axes)
    for feature, axis in enumerate(axes):
        axis.plot(real[0, :, feature], label="real", color="black", lw=1.4)
        axis.plot(fake[0, :, feature], label="PETS-U-AA", color="#2563eb", lw=1.2)
        axis.set_ylabel("f{}".format(feature))
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
