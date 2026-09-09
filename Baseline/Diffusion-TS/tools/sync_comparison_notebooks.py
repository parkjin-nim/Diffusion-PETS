"""Create Diffusion-TS notebooks matching the Diffusion-PETS experiment layout."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PETS_ROOT = ROOT.parent / "Diff-pets"

DATASETS = {
    "air_qual": ("air_qual", "air_qual", "Config/air_qual/air_qual_128.yaml"),
    "drift": ("drift", "sensor_drift", "Config/drift/drift_128.yaml"),
    "electricity": ("electricity", "electricity", "Config/electricity/electricity_128.yaml"),
    "energy": ("energy", "energy", "Config/energy/energy_128.yaml"),
    "etth": ("etth", "etth", "Config/etth/etth_128.yaml"),
    "ettm": ("ettm", "ettm", "Config/ettm/ettm_128.yaml"),
    "ettm2": ("ettm2", "ettm2", "Config/ettm/ettm2_128.yaml"),
    "exchange": (
        "exchange_rate",
        "exchange_rate",
        "Config/exchange_rate/exchange_rate_128.yaml",
    ),
    "geoje": ("geoje", "geoje", "Config/geoje/geoje_128.yaml"),
    "geoje_mean": ("geoje_mean", "geoje", "Config/geoje/geoje_mean_128.yaml"),
    "illness": (
        "national_illness",
        "national_illness",
        "Config/illness/national_illness_128.yaml",
    ),
    "sines": ("sines", "sine", "Config/sines/sines_128.yaml"),
    "stocks": ("stocks", "stock", "Config/stocks/stocks_128.yaml"),
    "synth": ("synth", "synth", "Config/synth/synth_128.yaml"),
    "traffic": ("traffic", "traffic", "Config/traffic/traffic_128.yaml"),
    "weather": ("weather", "weather", "Config/weather/weather_128.yaml"),
}

TEMPLATES = {
    "Uncond": PETS_ROOT / "Uncond/air_qual/uncond_air_qual_128.ipynb",
    "forecasting": PETS_ROOT
    / "forecasting/air_qual/forecast_air_qual_128.ipynb",
    "imputation": PETS_ROOT
    / "imputation/air_qual/impute_air_qual_128_m75.ipynb",
}

FILENAMES = {
    "Uncond": "uncond_{folder}_128.ipynb",
    "forecasting": "forecast_{folder}_128.ipynb",
    "imputation": "impute_{folder}_128_m75.ipynb",
}


def transformed_notebook(task, folder, run_name, truth_name, config_path):
    notebook = json.loads(TEMPLATES[task].read_text())
    replacements = {
        "Diffusion-PETS": "Diffusion-TS",
        "Diff-pets": "Diff-ts",
        "# air_qual:": f"# {folder}:",
        'NAME = "air_qual"': f'NAME = "{run_name}"',
        'TRUTH_NAME = "air_qual"': f'TRUTH_NAME = "{truth_name}"',
        'Config/air_qual/air_qual_128.yaml': config_path,
        "SEEDS = [42, 123, 2024]": "SEEDS = [0, 42, 123]",
        "GPU_BY_SEED = {42: 1, 123: 2, 2024: 3}": (
            "GPU_BY_SEED = {0: 1, 42: 2, 123: 3}"
        ),
        "from Utils.generative_metrics import evaluate_generation": (
            "from evaluation.generative_metrics import evaluate_generation"
        ),
        '/ "checkpoints"': '/ "checkpoints_128"',
    }

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = source.splitlines(True)
        if cell.get("cell_type") == "code":
            ast.parse(source)
    return notebook


def transformed_etth_train_only_notebook(seed):
    source = (
        PETS_ROOT
        / "forecasting"
        / "etth"
        / f"forecast_etth_128_s{seed}_pets.ipynb"
    )
    notebook = json.loads(source.read_text())
    replacements = {
        "Diffusion-PETS": "Diffusion-TS",
        "Diff-pets": "Diff-ts",
        (
            'CHECKPOINT_DIR = ROOT / "artifacts" / "train-only" / NAME '
            '/ f"seed_{SEED}" / "checkpoints"'
        ): (
            'CHECKPOINT_BASE = ROOT / "artifacts" / "train-only" / NAME '
            '/ f"seed_{SEED}" / "checkpoints"\n'
            'CHECKPOINT_DIR = Path(f"{CHECKPOINT_BASE}_{SEQ_LEN}")'
        ),
        '"--checkpoint_dir", str(CHECKPOINT_DIR)': (
            '"--checkpoint_dir", str(CHECKPOINT_BASE)'
        ),
        'test_params = config["dataloader"]["test_dataset"]["params"]': (
            'config["dataloader"]["test_dataset"]["target"] = '
            '"Data.comparison_datasets.TrainScalerCustomDataset"\n'
            '    test_params = config["dataloader"]["test_dataset"]["params"]'
        ),
    }
    for cell in notebook["cells"]:
        source_text = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source_text = source_text.replace(old, new)
        cell["source"] = source_text.splitlines(True)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            ast.parse(source_text)
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if "".join(cell.get("source", [])).strip()
    ]
    return notebook


def main():
    created = []
    for folder, (run_name, truth_name, config_path) in DATASETS.items():
        for task in ("Uncond", "forecasting", "imputation"):
            target = ROOT / task / folder / FILENAMES[task].format(folder=folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            notebook = transformed_notebook(
                task,
                folder,
                run_name,
                truth_name,
                config_path,
            )
            target.write_text(json.dumps(notebook, indent=1) + "\n")
            created.append(target)
    for seed in (0, 42, 123):
        target = (
            ROOT
            / "forecasting"
            / "etth"
            / f"forecast_etth_128_s{seed}_ts.ipynb"
        )
        notebook = transformed_etth_train_only_notebook(seed)
        target.write_text(json.dumps(notebook, indent=1) + "\n")
        created.append(target)
    print(f"created {len(created)} comparison notebooks")


if __name__ == "__main__":
    main()
