"""Shared, path-safe runtime helpers for the curated Diffusion-PETS release."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from Utils.io_utils import instantiate_from_config


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "ETTh": {"name": "etth", "file": "ETTh.csv", "features": 7},
    "Energy": {"name": "energy", "file": "energy_data.csv", "features": 28},
    "AirQuality": {"name": "air_qual", "file": "air_qual.csv", "features": 9},
}


def dataset_spec(name):
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError("dataset must be ETTh, Energy, or AirQuality") from exc


def raw_dataset_path(name):
    return ROOT / "Data" / "datasets" / dataset_spec(name)["file"]


def seed_everything(seed, gpu=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.cuda.manual_seed_all(seed)


def save_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def final_split_paths(dataset):
    base = ROOT / "PETS-F" / "data" / dataset
    return base / "final_train.csv", base / "final_test_with_context64.csv"


def prepare_forecast_split(dataset, history=64):
    """Create the frozen chronological 90/10 split used by the final paper."""
    source = raw_dataset_path(dataset)
    frame = pd.read_csv(source)
    cut = int(math.ceil(0.9 * len(frame)))
    train_path, test_path = final_split_paths(dataset)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[:cut].to_csv(train_path, index=False)
    frame.iloc[cut - history :].to_csv(test_path, index=False)
    return train_path, test_path, cut


def _base_config(dataset, seed, output, sequence_length=128):
    spec = dataset_spec(dataset)
    output = Path(output).resolve()
    return {
        "model": {
            "target": "Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS",
            "params": {
                "seq_length": sequence_length,
                "feature_size": spec["features"],
                "n_layer_enc": 3,
                "n_layer_dec": 3,
                "d_model": 64,
                "timesteps": 500,
                "sampling_timesteps": 500,
                "loss_type": "l1",
                "beta_schedule": "cosine",
                "n_heads": 4,
                "mlp_hidden_times": 4,
                "attn_pd": 0.0,
                "resid_pd": 0.0,
                "kernel_size": 1,
                "padding_size": 0,
                "configs": {
                    "top_k": 7,
                    "d_ff": 128,
                    "num_kernels": 3,
                    "k_mask_temperature": 1.0,
                    "period_weight_temperature": 1.0,
                    "timesblock_residual_mode": "single",
                },
            },
        },
        "solver": {
            "base_lr": 1.0e-5,
            "max_epochs": 18000,
            "results_folder": str(output / "checkpoints"),
            "gradient_accumulate_every": 2,
            "save_cycle": 3000,
            "checkpoint_name_by_step": True,
            "ema": {"decay": 0.995, "update_interval": 10},
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5,
                    "patience": 4000,
                    "min_lr": 1.0e-6,
                    "threshold": 1.0e-4,
                    "threshold_mode": "rel",
                    "warmup_lr": 1.0e-4,
                    "warmup": 500,
                    "verbose": False,
                },
            },
        },
        "dataloader": {"batch_size": 128, "sample_size": 256, "shuffle": True},
        "fft_ablation": {"variant": "FINAL", "seed": seed},
    }


def build_unconditional_config(dataset, length, seed, output, representation=None, placement=None):
    config = _base_config(dataset, seed, output, length)
    params = config["model"]["params"]
    params["configs"].update(
        period_candidate_scope="batch_shared",
        period_fft_source="full",
        period_fft_history_length=length,
        period_weight_mode="raw_softmax",
        factor=2,
        learnable_k=True,
        corrected_ste=True,
    )
    # The accepted PETS-U runs wrote ten numbered checkpoints at 1,800-step
    # intervals; checkpoint-10.pt is therefore the final 18k checkpoint.
    config["solver"]["save_cycle"] = 1800
    config["solver"]["checkpoint_name_by_step"] = False
    if representation and representation != "period_aligned_2d":
        config["model"]["target"] = (
            "Models.interpretable_diffusion.reviewer2_representations.Diffusion_TS"
        )
        params["configs"]["representation_strategy"] = representation
    if placement and placement != "e_last_before_d":
        if representation and representation != "period_aligned_2d":
            raise ValueError("representation and placement ablations are run separately")
        config["model"]["target"] = (
            "Models.interpretable_diffusion.reviewer2_placements.Diffusion_TS"
        )
        params["configs"]["timesblock_placement"] = placement
    raw = raw_dataset_path(dataset)
    spec = dataset_spec(dataset)
    data = {
        "target": "Utils.Data_utils.real_datasets.CustomDataset",
        "params": {
            "name": spec["name"], "proportion": 1.0, "data_root": str(raw),
            "window": length, "save2npy": False, "neg_one_to_one": True,
            "seed": seed, "period": "train",
        },
    }
    config["dataloader"].update(train_dataset=data, test_dataset=copy.deepcopy(data))
    config["experiment"] = {
        "marker": "PETS_U_FINAL_R2", "true_unconditional": True,
        "training_aggregation": "FULL", "anti_anchor_training": False,
    }
    return config


def build_forecast_config(dataset, seed, output, routing="SH64"):
    train_path, test_path, _ = prepare_forecast_split(dataset)
    config = _base_config(dataset, seed, output, 128)
    scope, source, history = {
        "BF": ("batch_shared", "full", 128),
        "BH": ("batch_shared", "history", 64),
        "SH64": ("sample_wise", "history", 64),
    }[routing]
    config["model"]["params"]["configs"].update(
        period_candidate_scope=scope,
        period_fft_source=source,
        period_fft_history_length=history,
    )
    spec = dataset_spec(dataset)
    config["dataloader"]["sample_size"] = min(256, max(16, 1792 // spec["features"]))
    config["dataloader"]["train_dataset"] = {
        "target": "Utils.Data_utils.real_datasets.CustomDataset",
        "params": {
            "name": spec["name"], "proportion": 1.0, "data_root": str(train_path),
            "window": 128, "save2npy": False, "neg_one_to_one": True,
            "seed": seed, "period": "train",
        },
    }
    config["dataloader"]["test_dataset"] = {
        "target": "Utils.Data_utils.real_datasets.CustomDataset",
        "params": {
            "name": spec["name"], "proportion": 0.0, "data_root": str(test_path),
            "scaler_data_root": str(train_path), "window": 128, "save2npy": False,
            "neg_one_to_one": True, "seed": seed, "period": "test",
            "style": "separate", "distribution": "geometric",
        },
        "coefficient": 0.01,
        "step_size": 0.05,
        "sampling_steps": 200,
    }
    config["fft_ablation"]["variant"] = routing
    config["experiment"] = {
        "marker": "PETS_F_FINAL_R2", "history": 64, "forecast": 64,
        "complete_window_training": True, "future_used_for_period_bank": False,
    }
    return config


def ema_state(checkpoint):
    state = checkpoint.get("ema")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no EMA state")
    result = {
        key[len("ema_model.") :]: value
        for key, value in state.items()
        if key.startswith("ema_model.")
    }
    if not result:
        raise RuntimeError("EMA model weights are missing")
    return result


def load_ema_model(config_path, checkpoint_path, gpu):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sampling")
    device = torch.device("cuda:{}".format(gpu))
    config = load_yaml(config_path)
    model_config = copy.deepcopy(config["model"])
    model_config["params"]["configs"] = SimpleNamespace(
        **model_config["params"].get("configs", {})
    )
    model = instantiate_from_config(model_config).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    incompatible = model.load_state_dict(ema_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("checkpoint/model state mismatch")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, device
