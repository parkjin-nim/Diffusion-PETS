#!/usr/bin/env python3
"""Regenerate every runnable release config with paths rooted at this copy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pets_runtime import build_forecast_config, build_unconditional_config, save_yaml


def main():
    seeds = (0, 42, 123)
    datasets = ("ETTh", "Energy", "AirQuality")
    lengths = (64, 128, 256)

    for dataset in datasets:
        for length in lengths:
            for seed in seeds:
                output = (
                    ROOT / "PETS-U/artifacts/paper_main" / dataset
                    / ("T{}".format(length)) / ("seed_{}".format(seed))
                )
                config = build_unconditional_config(dataset, length, seed, output)
                name = "{}_T{}_seed{}.yaml".format(dataset, length, seed)
                save_yaml(ROOT / "PETS-U/configs/paper_main" / name, config)
                save_yaml(output / "config_resolved.yaml", config)

    representation_root = (
        ROOT / "PETS-U/results/ablations_raw/"
        "reviewer2_representation_etth128/training"
    )
    for representation in ("stft_spectrogram", "frequency_as_channel"):
        for seed in seeds:
            output = representation_root / representation / ("seed_{}".format(seed))
            config = build_unconditional_config(
                "ETTh", 128, seed, output, representation=representation
            )
            save_yaml(output / "config_resolved.yaml", config)

    placement_root = (
        ROOT / "PETS-U/results/ablations_raw/"
        "reviewer2_timesblock_placement_etth128_v2/training"
    )
    for placement in ("e_only", "e_mid_d"):
        for seed in seeds:
            output = placement_root / placement / ("seed_{}".format(seed))
            config = build_unconditional_config(
                "ETTh", 128, seed, output, placement=placement
            )
            save_yaml(output / "config_resolved.yaml", config)

    for dataset in datasets:
        for seed in seeds:
            output = (
                ROOT / "PETS-F/artifacts/paper_main" / dataset
                / ("seed_{}".format(seed))
            )
            config = build_forecast_config(dataset, seed, output, "SH64")
            name = "{}_SH64_seed{}.yaml".format(dataset, seed)
            save_yaml(ROOT / "PETS-F/configs/paper_main" / name, config)
            save_yaml(output / "config_resolved.yaml", config)

    for directory, routing in (("BF", "BF"), ("BH", "BH"), ("SH", "SH64")):
        output = ROOT / "PETS-F/artifacts/routing_ablation" / directory / "seed_0"
        config = build_forecast_config("ETTh", 0, output, routing)
        save_yaml(output / "config_resolved.yaml", config)

    examples = (
        ("PETS-U/configs/ETTh_T64_seed0.yaml", build_unconditional_config(
            "ETTh", 64, 0, ROOT / "PETS-U/artifacts/ETTh_T64_seed0"
        )),
        ("PETS-U/configs/ETTh_T128_seed0_stft_spectrogram.yaml", build_unconditional_config(
            "ETTh", 128, 0, ROOT / "PETS-U/artifacts/ETTh_T128_seed0_stft_spectrogram",
            representation="stft_spectrogram",
        )),
        ("PETS-U/configs/ETTh_T128_seed0_e_mid_d.yaml", build_unconditional_config(
            "ETTh", 128, 0, ROOT / "PETS-U/artifacts/ETTh_T128_seed0_e_mid_d",
            placement="e_mid_d",
        )),
        ("PETS-F/configs/ETTh_BF_seed0.yaml", build_forecast_config(
            "ETTh", 0, ROOT / "PETS-F/artifacts/ETTh_BF_seed0", "BF"
        )),
        ("PETS-F/configs/ETTh_SH64_seed0.yaml", build_forecast_config(
            "ETTh", 0, ROOT / "PETS-F/artifacts/ETTh_SH64_seed0", "SH64"
        )),
    )
    for relative, config in examples:
        save_yaml(ROOT / relative, config)

    print("CONFIGS_REGENERATED")


if __name__ == "__main__":
    main()
