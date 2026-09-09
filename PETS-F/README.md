# PETS-F: history-conditioned probabilistic forecasting

PETS-F predicts 64 future steps from 64 observed steps. Training uses complete
128-step windows with sample-wise history-only routing (SH64). At inference,
the period bank is frozen from each observed history and shared by the encoder
and decoder; CG64 gates the periodic residual, reconstruction guidance enforces
the observed prefix, and gamma=1.5 calibrates only ensemble spread while
preserving its mean.

Run commands from the repository root.

## Train

```bash
python PETS-F/scripts/train.py \
  --dataset ETTh --seed 0 --gpu 0 --routing SH64
```

The script creates the frozen chronological 90/10 train/test protocol in
`PETS-F/data/`. Checkpoints are written every 3,000 steps and the final file is
`checkpoint-step-18000.pt`. Use `--resume-step 12000` (for example) to resume.
A run started by the command above writes its config and artifacts under:

```text
PETS-F/configs/ETTh_SH64_seed0.yaml
PETS-F/artifacts/ETTh_SH64_seed0/checkpoints/checkpoint-step-18000.pt
```

## Sample

```bash
python PETS-F/scripts/sample.py \
  --config PETS-F/configs/ETTh_SH64_seed0.yaml \
  --checkpoint PETS-F/artifacts/ETTh_SH64_seed0/checkpoints/checkpoint-step-18000.pt \
  --gpu 0 --draw-seeds 100000 100001 100002 100003 100004 \
  100005 100006 100007 100008 100009 --gamma 1.5 \
  --output-dir PETS-F/output/ETTh_seed0
```

Use `--disable-gate` for the SH64-without-CG64 control, and `--gamma 1.0` for
uncalibrated spread. The sampler saves the calibrated ensemble, individual
normalized draws, and held-out truth.

## Test and visualize

```bash
python PETS-F/scripts/evaluate.py \
  --ensemble PETS-F/output/ETTh_seed0/ensemble.npy \
  --truth PETS-F/output/ETTh_seed0/truth.npy \
  --output PETS-F/output/ETTh_seed0/metrics.json

python PETS-F/scripts/visualize.py \
  --ensemble PETS-F/output/ETTh_seed0/ensemble.npy \
  --truth PETS-F/output/ETTh_seed0/truth.npy --window 0 --feature 0 \
  --output PETS-F/output/ETTh_seed0/forecast.png
```

The paper uses all three datasets, seeds 0/42/123, and ten trajectories per
seed. Raw final-test arrays are omitted from GitHub; paper tables and the final
qualitative notebook/figures are under `PETS-F/results/`. Open
`PETS-F/notebooks/PETS_F_final_results.ipynb` for a CPU-only summary.

## Routing ablation

Use `--routing BF`, `BH`, or `SH64` with `train.py`. The ETTh seed-0 final
aggregate results are retained under `PETS-F/results/paper/`. Checkpoints,
trajectory arrays, telemetry, and traces are intentionally omitted from GitHub.
