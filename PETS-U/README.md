# PETS-U: true-unconditional generation

PETS-U trains on complete windows with batch-shared full-sequence routing.
The final PETS-U-AA sampler recomputes the latent period bank at each reverse
step and applies the same frozen Anti-Anchor operation to the encoder and
decoder TimesBlocks. Anti-Anchor adds no trainable parameter and requires no
retraining.

Run commands from the repository root. Supported paper
datasets are `ETTh`, `Energy`, and `AirQuality`; supported lengths are 64, 128,
and 256.

## Train

```bash
python PETS-U/scripts/train.py \
  --dataset ETTh --length 128 --seed 0 --gpu 0
```

Ten numbered checkpoints are written every 1,800 steps; `checkpoint-10.pt` is
the final 18,000-step model. Add `--resume-step 10` to resume that numbered
checkpoint. A run started by the command above writes its config and artifacts
under:

```text
PETS-U/configs/ETTh_T128_seed0.yaml
PETS-U/artifacts/ETTh_T128_seed0/checkpoints/checkpoint-10.pt
```

## Sample

Final symmetric Anti-Anchor sampling:

```bash
python PETS-U/scripts/sample.py \
  --config PETS-U/configs/ETTh_T128_seed0.yaml \
  --checkpoint PETS-U/artifacts/ETTh_T128_seed0/checkpoints/checkpoint-10.pt \
  --gpu 0 --draw-seed 100000 --num 4096 --aggregation anti-anchor \
  --output PETS-U/output/ETTh_T128_seed0_draw100000.npy
```

Use `--aggregation canonical`, `encoder-only`, or `decoder-only` for the Table
7 controls. The five paper draw seeds are 100000–100004.

## Test and visualize

```bash
python PETS-U/scripts/prepare_reference.py \
  --config PETS-U/configs/ETTh_T128_seed0.yaml \
  --output PETS-U/output/ETTh_T128_real.npy

python PETS-U/scripts/evaluate.py \
  --real PETS-U/output/ETTh_T128_real.npy \
  --fake PETS-U/output/ETTh_T128_seed0_draw100000.npy \
  --gpu 0 --evaluator-seed 0 \
  --output PETS-U/output/ETTh_T128_seed0_draw100000_metrics.json

python PETS-U/scripts/visualize.py \
  --real PETS-U/output/ETTh_T128_real.npy \
  --fake PETS-U/output/ETTh_T128_seed0_draw100000.npy \
  --output PETS-U/output/ETTh_T128_example.png
```

The paper aggregates model seeds 0/42/123, five draws per seed, and evaluator
seeds 0/42/123 for learned metrics. Published paper tables, plots, and compact
reports are under `PETS-U/results/`; raw arrays and checkpoints are omitted
from GitHub. Open
`PETS-U/notebooks/PETS_U_final_results.ipynb` for CPU-only inspection.

## Published architecture ablations

```bash
# Representation controls (Table 8)
python PETS-U/scripts/train.py \
  --dataset ETTh --length 128 --seed 0 --gpu 0 \
  --representation stft_spectrogram

# Placement controls (Table 9)
python PETS-U/scripts/train.py \
  --dataset ETTh --length 128 --seed 0 --gpu 0 --placement e_mid_d
```

The other valid control values are `frequency_as_channel` and `e_only`.
