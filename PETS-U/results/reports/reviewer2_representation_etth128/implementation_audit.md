# Reviewer #2 implementation audit

- `ETTH128_ONLY_CONFIRMED`
- `PETSU_ONLY_CONFIRMED`
- `ANTI_ANCHOR_DISABLED`
- `COMMON_BACKBONE_CONFIRMED`
- `COMMON_E_LASTBEFORED_PLACEMENT_CONFIRMED`
- `COMMON_TOPK_POLICY_CONFIRMED`
- `COMMON_OBJECTIVE_CONFIRMED`
- `OUTPUT_SHAPE_PARITY_PASS`
- `PARAMETER_BUDGET_AUDITED`
- `ETTH128_STANDARD_PETSU_PARITY_PASS`

## Controlled definitions

- Period-Aligned 2D: production batch-shared full-rFFT distinct-period selection, hard top-k, 2D Inception maps, normalized spectral aggregation.
- STFT / Spectrogram: fixed `n_fft=win_length=32`, `hop=8`, Hann, `center=False`, complex real/imaginary map, parameter-matched 2D convolution.
- Frequency-as-Channel: selected complex rFFT coefficients, shared spectral projection, time broadcast, parameter-matched 1D convolution.

All variants retain the same E + before-last D placement and legacy residual interface. Anti-Anchor, AA/AA, forecasting masks, guidance, MPRCL, and PETS-F components are absent.

## Prompt/code conflict resolution

The production lineage uses optimizer base LR `1e-5` with a 500-step warm-up target of `1e-4`; all three variants retain that exact schedule. The production distinct-period definition uses integer floor division, so all variants retain it instead of changing the standard model to the prompt's conflicting `ceil` prose. The host exposes only GPUs 0–2.
