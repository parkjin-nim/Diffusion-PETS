#!/usr/bin/env python3
"""Train the final true-unconditional PETS-U backbone."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pets_runtime import build_unconditional_config, save_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ETTh", "Energy", "AirQuality"), required=True)
    parser.add_argument("--length", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument(
        "--representation",
        choices=("period_aligned_2d", "stft_spectrogram", "frequency_as_channel"),
        default="period_aligned_2d",
    )
    parser.add_argument(
        "--placement",
        choices=("e_last_before_d", "e_mid_d", "e_only"),
        default="e_last_before_d",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tag = "{}_T{}_seed{}".format(args.dataset, args.length, args.seed)
    if args.representation != "period_aligned_2d":
        tag += "_" + args.representation
    if args.placement != "e_last_before_d":
        tag += "_" + args.placement
    output = ROOT / "PETS-U" / "artifacts" / tag
    config_path = ROOT / "PETS-U" / "configs" / (tag + ".yaml")
    config = build_unconditional_config(
        args.dataset, args.length, args.seed, output,
        representation=args.representation, placement=args.placement,
    )
    save_yaml(config_path, config)
    command = [
        sys.executable, str(ROOT / "main.py"), "--name", config["dataloader"]["train_dataset"]["params"]["name"],
        "--config_file", str(config_path), "--gpu", str(args.gpu), "--train",
        "--task", "uncond", "--seed", str(args.seed), "--save_dir", str(output),
        "--checkpoint_dir", str(output / "checkpoints"),
    ]
    if args.resume_step is not None:
        command += ["--resume_step", str(args.resume_step)]
    print("CONFIG:", config_path)
    print("COMMAND:", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
