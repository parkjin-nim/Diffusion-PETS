# Legacy unconditional main-table provenance

`LEGACY_MAIN_TABLE_AUDITED`

The exact revision LaTeX source containing Context-FID, Correlational, Discriminative, and Predictive cells was not present in the supplied repositories. Consequently, all non-locked historical cells are classified `LEGACY_CELL_PROVENANCE_UNRESOLVED` and none will be copied into the new table.

A LaTeX file was found at `/path/to/PETS-FFT-Ablation/reports/final_probabilistic_forecast_benchmark/paper/tables_b_to_e.tex`, but it is a probabilistic forecasting table (MAE/RMSE/CRPS), not the requested true-unconditional four-metric main table.

The controlled ETTh-128 values differ from remembered/legacy numbers whenever those values used a mixed training split, a different evaluator, pooled draw-level variation, a single checkpoint, or an unsupported sample/reference count. The new anchor fixes all of these choices and is the only ETTh-128 value eligible for exact reuse.
