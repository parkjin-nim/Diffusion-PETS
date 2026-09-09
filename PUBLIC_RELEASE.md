# Public release contents

This GitHub tree contains the final Diffusion-PETS implementation, runnable
training/sampling/evaluation/visualization scripts, portable experiment
configs, compact paper result tables, figures, notebooks, and baseline source.

To keep the repository reproducible and within standard GitHub limits, the
following runtime artifacts are intentionally not versioned:

- model and optimizer checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- generated/reference/prediction arrays (`*.npy`, `*.npz`);
- per-step logs, telemetry, process records, and cache files;
- duplicated raw trajectory and intermediate-ablation directories.

The included `.gitignore` enforces the same policy for future local runs.
Aggregate experimental evidence remains available as CSV/JSON/Markdown tables,
PNG/PDF figures, and executed Jupyter notebooks under `PETS-U/results/` and
`PETS-F/results/`.
