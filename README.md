# Diffusion-PETS R2

This repository is the public, path-safe code and results release
corresponding to the final revision in `docs/NEUCOM-D-26-06327_R1.pdf`.

## Layout

- `Models/`, `engine/`, `Data/`, `Utils/`, `layers/`, `Config/`, `main.py`:
  shared Diffusion-PETS backbone, data loaders, metrics, and base configs.
- `PETS-U/`: true-unconditional generation, frozen symmetric Anti-Anchor
  sampling, published ablations, and compact result tables/figures.
- `PETS-F/`: history-conditioned forecasting with SH64 routing, CG64,
  reconstruction guidance, and mean-preserving spread calibration.
- `Baseline/Diffusion-TS/` and `Baseline/DiffWave/`: comparison-source code.
- `docs/SELECTION_AUDIT.md`: paper-to-code selection and exclusion rationale.
- `PUBLIC_RELEASE.md`: public-artifact inclusion and exclusion policy.

## Environment

The revision experiments used Python 3.8 and CUDA. From this directory:

```bash
python -m pip install -r requirements.txt
```

Training and sampling require a CUDA device. Table inspection, saved-result
analysis, and plotting can run on CPU. Commands and artifact conventions are
documented separately in `PETS-U/README.md` and `PETS-F/README.md`.

## Scope

The release retains the adopted PETS-U and PETS-F mechanisms and the controls
that appear in the final paper's ablation tables. Exploratory branches that
were not adopted—MPRCL/cMPRCL, DPW-Norm/ENC-CAL, SNR/cap/relation penalties,
late activation, APC/harmonic-complement recovery, fixed-anchor pilots,
T=512, Traffic, and Sines—are intentionally absent.

GitHub does not contain model checkpoints, generated NumPy arrays, raw
trajectory dumps, or execution logs. Train the model with the documented
commands to create a checkpoint locally, then run the sampling and evaluation
commands. Published aggregate CSV/JSON tables, figures, and analysis notebooks
are retained under `PETS-U/results/` and `PETS-F/results/`.
