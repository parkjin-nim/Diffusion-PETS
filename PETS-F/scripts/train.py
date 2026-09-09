#!/usr/bin/env python3
"""Train the final history-conditioned PETS-F backbone on complete 128-step windows."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pets_runtime import build_forecast_config, save_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ETTh", "Energy", "AirQuality"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--routing", choices=("BF", "BH", "SH64"), default="SH64")
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tag = "{}_{}_seed{}".format(args.dataset, args.routing, args.seed)
    output = ROOT / "PETS-F" / "artifacts" / tag
    config_path = ROOT / "PETS-F" / "configs" / (tag + ".yaml")
    config = build_forecast_config(args.dataset, args.seed, output, args.routing)
    save_yaml(config_path, config)
    command = [
        sys.executable, str(ROOT / "main.py"), "--name", config["dataloader"]["train_dataset"]["params"]["name"],
        "--config_file", str(config_path), "--gpu", str(args.gpu), "--train",
        "--task", "forecasting", "--seed", str(args.seed), "--save_dir", str(output),
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
