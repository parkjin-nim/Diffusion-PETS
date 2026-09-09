#!/usr/bin/env python3
"""Build a normalized real-window reference array for PETS-U evaluation."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import instantiate_from_config
from pets_runtime import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_config = copy.deepcopy(config["dataloader"]["train_dataset"])
    dataset_config["params"]["save2npy"] = False
    dataset_config["params"]["output_dir"] = str(args.output.parent / "dataset_cache")
    dataset = instantiate_from_config(dataset_config)
    reference = unnormalize_to_zero_to_one(np.asarray(dataset.samples)).astype(
        np.float32
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, reference)
    print("SAVED:", args.output, "shape=", reference.shape)


if __name__ == "__main__":
    main()
