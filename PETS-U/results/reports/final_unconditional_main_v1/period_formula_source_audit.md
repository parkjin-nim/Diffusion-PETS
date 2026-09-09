# Period-formula source audit

`PERIOD_FORMULA_FLOOR_CONFIRMED`

The three protected PETS lineages implement discrete period identity with integer division. The separate branch reshape pads the temporal axis to the next multiple of that already selected period.

No generation-quality metric was inspected for this decision.

## Sources

- `/path/to/PETS-FFT-Ablation/Models/interpretable_diffusion/TimesNet.py` — `floor` — `5098d2b2a133d77458081b483ac2ccd7d964e15d1d7c2fe3f84b08d983b4b53f`
- `/path/to/user/DMs/PETS-RC/Models/interpretable_diffusion/TimesNet.py` — `floor` — `c5e4b19bd945b7758140fa33acb673b600ee3767d93b7b97794689c6de9aee6d`
- `/path/to/user/DMs/Diff-pets/Models/interpretable_diffusion/TimesNet.py` — `floor` — `0c9424c3f268474cdf5782c37af20d37e103405d543c2b8f2b3392d36655bb01`

## Git lineage

- `4aa3638ba34482165326dc9114c6c87b1ed0180e Add SH decoder period-weight calibration ablation`
- `14e03fa1a318aee5751ec34488fa2473f2d381f5 Vectorize sample-wise period execution`
- `71925dfc6aa9ebea54a14312a5f49dcd7d7e011f Stabilize learnable-k STE smoke training`
- `f2aa8435daf3233777e6d6540ad91e8ea9897af2 Add manual FFT ablation GPU runners and analysis notebook`
- `69009e5742229a87194e1f54238a2cce1632b22f Implement 2x2 FFT candidate ablation`
- `8af920ef16e3b15b62900a7680f016ac48c94c7c Baseline copied from PETS-RC before FFT candidate ablation`
