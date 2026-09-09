#!/usr/bin/env python3
"""Sample a frozen PETS-U checkpoint with canonical or symmetric Anti-Anchor aggregation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anti_anchor import AntiAnchorContext
from pets_runtime import load_ema_model, load_yaml, seed_everything
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--draw-seed", type=int, default=100000)
    parser.add_argument("--num", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--aggregation",
        choices=("canonical", "encoder-only", "decoder-only", "anti-anchor"),
        default="anti-anchor",
        help="Anti-Anchor location; anti-anchor means the accepted symmetric AA/AA method",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(args.draw_seed, args.gpu)
    model, config, _device = load_ema_model(args.config, args.checkpoint, args.gpu)
    expected = config["model"]["params"]
    samples = []
    locations = {
        "encoder-only": (True, False),
        "decoder-only": (False, True),
        "anti-anchor": (True, True),
    }
    context = (
        AntiAnchorContext(model, *locations[args.aggregation])
        if args.aggregation != "canonical" else None
    )
    try:
        for low in range(0, args.num, args.batch_size):
            size = min(args.batch_size, args.num - low)
            with torch.inference_mode():
                value = model.generate_mts(batch_size=size)
            samples.append(unnormalize_to_zero_to_one(value).cpu().numpy().astype(np.float32))
            print("sampling {}/{}".format(low + size, args.num), flush=True)
    finally:
        if context is not None:
            context.close()
    result = np.concatenate(samples)
    if result.shape != (args.num, expected["seq_length"], expected["feature_size"]):
        raise RuntimeError("unexpected sample shape {}".format(result.shape))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    print("SAVED:", args.output)


if __name__ == "__main__":
    main()
