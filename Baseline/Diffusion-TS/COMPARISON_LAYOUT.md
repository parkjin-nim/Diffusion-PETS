# Diffusion-PETS comparison layout

This repository keeps the original Diffusion-TS implementation under
`engine/`, `Models/`, and `Utils/` unchanged. Experiment adapters live in
`main.py`, `Data/`, `evaluation/`, `tools/`, and the task notebook folders.

## One-to-one notebooks

Each of the 16 datasets has one notebook under each task:

- `Uncond/<dataset>/uncond_<dataset>_128.ipynb`
- `forecasting/<dataset>/forecast_<dataset>_128.ipynb`
- `imputation/<dataset>/impute_<dataset>_128_m75.ipynb`

The relative paths match the standard Diffusion-PETS notebooks.

ETTh forecasting additionally provides train-only notebooks for seeds `0`,
`42`, and `123` as `forecast_etth_128_s<seed>_ts.ipynb`, matching the
corresponding `_pets.ipynb` Diffusion-PETS experiment files.

## Seeds and output paths

Unconditional training uses seeds `0`, `42`, and `123` on GPUs `1`, `2`, and
`3`. Outputs are written to:

```text
artifacts/uncond/<run_name>/seed_<seed>/
```

Conditional task outputs are written to:

```text
artifacts/forecasting/<run_name>/
artifacts/imputation/<run_name>/
```

The untouched Diffusion-TS trainer appends the sequence length to its
checkpoint directory, so 128-step checkpoints are stored under
`checkpoints_128/`. Generated arrays, truth arrays, masks, ensemble arrays,
and metric JSON files otherwise use the same names as Diffusion-PETS.

## Configuration adapter

The copied YAML files contain a nested `model.params.configs` section used by
Diffusion-PETS TimesNet. `main.py` removes only that method-specific section
before constructing Diffusion-TS. Shared model, solver, and dataset settings
remain unchanged.

## Regeneration

Run the following command after changing the standard Diffusion-PETS notebook
layout:

```bash
python tools/sync_comparison_notebooks.py
```

The script regenerates and syntax-checks 48 standard notebooks plus the three
ETTh train-only seed notebooks.
