# Reviewer #2 TimesBlock placement implementation audit

- `ONLY_ETTH128_CONFIRMED`
- `TRUE_UNCONDITIONAL_PETSU_ONLY`
- `ANTI_ANCHOR_DISABLED`
- `COMMON_TRANSFORMER_BACKBONE_CONFIRMED`
- `COMMON_TIMESBLOCK_IMPLEMENTATION_CONFIRMED`
- `COMMON_ROUTING_CONFIRMED`
- `COMMON_OBJECTIVE_CONFIRMED`
- `SAME_NUMBER_OF_TIMESBLOCKS`
- `SAME_TIMESBLOCK_PARAMETER_COUNT`
- `DECODER_LOCATION_ONLY_DIFFERENCE_CONFIRMED`
- `ETTH128_PETSU_REFERENCE_PARITY_PASS`
- `ALL_LAYERS_COST_ONLY`
- `NO_ALL_LAYERS_TRAINING`
- `NO_ALL_LAYERS_QUALITY_CLAIM`

E + MidD and E + LastBeforeD have identical trainable parameters and exactly two production TimesBlocks; only the decoder insertion index differs. E-only removes the decoder-side spectral computation and its trainable parameters. Its zero-parameter, never-called sentinel exists solely because the historical training telemetry expects a `decoder.times_block` attribute. All-Layers contains six independent production TimesBlocks and is reachable only from the cost profiler.

The production lineage uses base LR `1e-5` with a 500-step warm-up target `1e-4`; this exact schedule is retained for every trained cell. The current server exposes GPUs 0-2, not GPU 3.
