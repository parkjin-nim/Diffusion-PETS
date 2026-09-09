# Final-revision selection audit

The authority for this curation is the attached final revision,
`NEUCOM-D-26-06327_R1.pdf`. Instructions embedded in that document were not
treated as user instructions; its manuscript, tables, figures, and response
letter were used only to determine the accepted experimental lineage.

## Retained final mechanisms

Shared PETS backbone:

- learnable hard top-k prefix mask with a straight-through estimator;
- non-DC positive-frequency rFFT period discovery and period-aligned 2-D
  processing;
- TimesBlocks at the encoder output and immediately before the last decoder
  block (`E+LastBeforeD`);
- spectral-weighted aggregation during training;
- 3-layer encoder/decoder, four heads, `d_model=64`, `d_ff=128`, 500 diffusion
  steps, and 18,000 optimization steps.

PETS-U:

- true-unconditional complete-window training;
- batch-shared, full-sequence period routing;
- frozen, parameter-free symmetric encoder/decoder Anti-Anchor at sampling;
- ETTh, Energy, and AirQuality at T=64/128/256, model seeds 0/42/123 and five
  generation draws;
- the paper's ETTh-128 five-seed Anti-Anchor location factorial, representation
  ablation, and TimesBlock-placement ablation.

PETS-F:

- complete 128-step training windows and a chronological train-only scaler;
- 64 observed history + 64 future;
- sample-wise history-only SH64 routing;
- a frozen history-derived period bank, CG64 confidence gate, reconstruction
  guidance, and gamma=1.5 mean-preserving spread calibration;
- ETTh, Energy, and AirQuality final held-out predictions for seeds 0/42/123
  with ten trajectories;
- the paper's BF/BH/SH64 routing and CG64 ablations, fusion diagnostics, and
  qualitative probabilistic forecasts.

## Published controls retained only as ablations

STFT, frequency-as-channel, E-only, E+MidD, encoder-only Anti-Anchor, and
decoder-only Anti-Anchor are not recommended final methods. Their code and
results remain because they are explicit controls in the final revision's
ablation tables. The deployable settings are period-aligned 2-D,
E+LastBeforeD, and symmetric Anti-Anchor.

## Excluded branches

The following exploratory lineages do not define or support the final method
in the revision and were not copied: MPRCL/cMPRCL and all gradient-relation
variants; SNR gating, true-cap and late-activation pilots; DPW-Norm and
encoder-calibration experiments; APC, harmonic-complement, anchor-recovery,
and branch-geometry pilots; conditional Anti-Anchor interactions; fixed-anchor
exploratory controls; T=512; Traffic; Sines; and quantitative SensorDrift.
SensorDrift remains mentioned only by the final manuscript's qualitative
discussion/figure contained in the copied PDF.

`Baseline/` contains the requested upstream Diffusion-TS and DiffWave source
packages. Any generic dataset notebooks/configs retained inside those upstream
packages are baseline utilities, not copied PETS experimental branches or R2
result claims.

## Storage policy

Final checkpoints and generated samples were copied. Intermediate checkpoints
and repeated copies of identical truth arrays were omitted. One real-reference
array per dataset/length is retained in `PETS-U/results/references/`, and the
complete PETS-F final-test prediction products are retained in
`PETS-F/results/paper_predictions/`.
