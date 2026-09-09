#!/usr/bin/env python3
"""Evaluate unconditional samples with the release's four primary metrics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Utils.generative_metrics import evaluate_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--evaluator-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    real, fake = np.load(args.real), np.load(args.fake)
    if real.ndim != 3 or fake.ndim != 3 or real.shape[1:] != fake.shape[1:]:
        raise ValueError("real/fake arrays must be [N,T,C] with equal T,C")
    metrics = evaluate_generation(real, fake, args.gpu, args.evaluator_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
