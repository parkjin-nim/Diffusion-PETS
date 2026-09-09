# Qualitative probabilistic forecasting V3

## Scope and disclosure

This report uses **performance-aligned illustrative window selection**. It is not a
model-independent typical, random, representative, or unbiased window analysis. The
model seed (0) and displayed feature (index 0) were fixed before scanning windows.
The primary evidence remains the all-window, three-seed FINAL_TEST quantitative table.
The supplementary audit remains the V2 model-independent typical-window result; V2
was preserved unchanged.

## Fixed systems and protocol

The frozen FINAL_TEST systems are `TS_REC_ORIGINAL` and
`PETS_SHTRAIN_CG64_REC` with frozen mean-preserving tau=1.5 calibration. FINAL_TRAIN
is the first 90%, FINAL_TEST the last 10%, H=64, F=64, L=128, sampling steps=200.
No training or tuning was performed. Seed 0 is fixed only for visualization continuity;
paper-level claims use the three-seed FINAL_TEST aggregate.

## Selection summary

| dataset    | total_windows | pets_improves_mae_count | pets_improves_crps_count | pets_improves_both_count | eligible_count | pets_improves_mae_fraction | pets_improves_crps_fraction | pets_improves_both_fraction | eligible_fraction | selected_window_index | selected_origin_row | selected_delta_mae | selected_delta_crps | selected_joint_gain | selected_curvature_pct | selected_variation_pct | selected_range_pct | selected_boundary_jump_pct | median_eligible_joint_gain |
| ---------- | ------------- | ----------------------- | ------------------------ | ------------------------ | -------------- | -------------------------- | --------------------------- | --------------------------- | ----------------- | --------------------- | ------------------- | ------------------ | ------------------- | ------------------- | ---------------------- | ---------------------- | ------------------ | -------------------------- | -------------------------- |
| ETTh       | 1679          | 720                     | 623                      | 606                      | 72             | 0.428827                   | 0.371054                    | 0.360929                    | 0.0428827         | 104                   | 15782               | -13.3629           | -14.6176            | 13.9903             | 0.777844               | 0.488982               | 0.438952           | 0.519952                   | 13.8795                    |
| AirQuality | 872           | 379                     | 379                      | 354                      | 65             | 0.434633                   | 0.434633                    | 0.405963                    | 0.0745413         | 272                   | 8694                | -26.322            | -18.2736            | 22.2978             | 0.637615               | 0.793578               | 0.704702           | 0.543578                   | 22.2978                    |

## 10-draw selection metrics

| dataset    | system                   | model_seed | window_index | origin_row | draws | ensemble_mean_mae | ensemble_mean_rmse | expected_sample_mae | empirical_crps |
| ---------- | ------------------------ | ---------- | ------------ | ---------- | ----- | ----------------- | ------------------ | ------------------- | -------------- |
| ETTh       | TS_REC_ORIGINAL          | 0          | 104          | 15782      | 10    | 3.78037           | 5.77532            | 4.11525             | 3.17598        |
| ETTh       | PETS_SELECTED_CALIBRATED | 0          | 104          | 15782      | 10    | 3.27521           | 4.6008             | 3.63165             | 2.71173        |
| AirQuality | TS_REC_ORIGINAL          | 0          | 272          | 8694       | 10    | 216.766           | 264.244            | 228.406             | 165.993        |
| AirQuality | PETS_SELECTED_CALIBRATED | 0          | 272          | 8694       | 10    | 159.709           | 202.374            | 169.012             | 135.66         |

## Optional aggregate-alignment diagnostic

| dataset    | aggregate_delta_mae | aggregate_delta_crps | selected_delta_mae | selected_delta_crps | euclidean_distance_percentage_points |
| ---------- | ------------------- | -------------------- | ------------------ | ------------------- | ------------------------------------ |
| ETTh       | -3.12561            | -3.07018             | -13.3629           | -14.6176            | 15.4319                              |
| AirQuality | -0.237659           | -0.456144            | -26.322            | -18.2736            | 31.5888                              |

## 100-trajectory displayed-window metrics

| dataset    | system                   | model_seed | window_index | origin_row | feature_index | trajectories | ensemble_mean_mae | ensemble_mean_rmse | expected_sample_mae | empirical_crps |
| ---------- | ------------------------ | ---------- | ------------ | ---------- | ------------- | ------------ | ----------------- | ------------------ | ------------------- | -------------- |
| ETTh       | TS_REC_ORIGINAL          | 0          | 104          | 15782      | 0             | 100          | 3.79789           | 5.80227            | 4.14347             | 3.106          |
| ETTh       | PETS_SELECTED_CALIBRATED | 0          | 104          | 15782      | 0             | 100          | 3.18736           | 4.4043             | 3.54431             | 2.51           |
| AirQuality | TS_REC_ORIGINAL          | 0          | 272          | 8694       | 0             | 100          | 197.712           | 238.096            | 225.658             | 143.701        |
| AirQuality | PETS_SELECTED_CALIBRATED | 0          | 272          | 8694       | 0             | 100          | 161.69            | 204.467            | 170.936             | 133.446        |

The ordering check below reports the 10-draw and 100-draw outcomes without any
post-hoc reselection.

| dataset    | trajectories | pets_lower_mae | pets_lower_crps | pets_lower_both | delta_mae_percent | delta_crps_percent |
| ---------- | ------------ | -------------- | --------------- | --------------- | ----------------- | ------------------ |
| ETTh       | 10           | True           | True            | True            | -13.3629          | -14.6176           |
| ETTh       | 100          | True           | True            | True            | -16.0754          | -19.1885           |
| AirQuality | 10           | True           | True            | True            | -26.322           | -18.2736           |
| AirQuality | 100          | True           | True            | True            | -18.2192          | -7.13594           |

## 100-trajectory shape comparison

- **ETTh:** PETS vs TS peak-timing error 51/0, trough-timing error 0/22, slope MAE 2.263/2.4, curvature MAE 3.472/3.51, turning F1 0.788/0.500, mean spread 1.842/1.873, 50% coverage 0.234/0.297, 90% coverage 0.625/0.625. The PETS ensemble mean is smoother than its individual paths (curvature ratio 1.649).
- **AirQuality:** PETS vs TS peak-timing error 14/1, trough-timing error 0/0, slope MAE 69.76/70.72, curvature MAE 90.16/95.87, turning F1 0.714/0.686, mean spread 68.82/151.8, 50% coverage 0.172/0.156, 90% coverage 0.391/0.812. The PETS ensemble mean is smoother than its individual paths (curvature ratio 1.619).

Peak/trough timing and magnitude, local slope/curvature, derivative-sign agreement,
turning-point F1 (+/-2), ensemble spread and 50%/90% coverage are secondary diagnostics;
none were used to select the windows. `ensemble_mean_oversmooths_individual_trajectories`
compares mean absolute curvature of the mean path with that of individual paths.

## Ensemble-size convergence

| dataset    | system                   | M1_mae  | M10_mae | M100_mae | M10_to_M100_mae_change_percent | M10_to_M100_rmse_change_percent |
| ---------- | ------------------------ | ------- | ------- | -------- | ------------------------------ | ------------------------------- |
| AirQuality | PETS_SELECTED_CALIBRATED | 172.393 | 159.45  | 161.69   | 1.40514                        | 0.950512                        |
| AirQuality | TS_REC_ORIGINAL          | 267.3   | 180.36  | 197.712  | 9.62109                        | 6.74965                         |
| ETTh       | PETS_SELECTED_CALIBRATED | 3.54336 | 3.24492 | 3.18736  | -1.77369                       | -1.65885                        |
| ETTh       | TS_REC_ORIGINAL          | 4.72922 | 3.8536  | 3.79789  | -1.44575                       | -1.38877                        |

## Paper caption

Qualitative probabilistic forecasts on illustrative FINAL_TEST windows where the final PETS-F configuration yields lower ensemble-mean MAE and empirical CRPS than Diffusion-TS under a fixed seed-0 selection ensemble. To avoid extreme or best-case examples, candidate windows are restricted to the central 20--80% of ground-truth curvature, variability, amplitude range, and boundary-jump statistics, and the case closest to the median joint PETS improvement is shown. Thin curves denote 100 stochastic trajectories, the thick forecast curve their pointwise ensemble mean, and the shaded bands the 50% and 90% central prediction intervals. The quantitative FINAL_TEST comparison over all windows and three model seeds is reported separately in Table X.

## Required verdicts

- `SEED0_FIXED_BEFORE_WINDOW_SCAN`
- `FEATURES_FIXED_BEFORE_WINDOW_SCAN`
- `PERFORMANCE_ALIGNED_SELECTION_DISCLOSED`
- `NO_BEST_CASE_EXTREME_SELECTION`
- `GROUND_TRUTH_CENTRAL_20_80_FILTER_CONFIRMED`
- `MEDIAN_JOINT_GAIN_SELECTION_CONFIRMED`
- `WINDOW_FROZEN_BEFORE_100_TRAJECTORY_RUN`
- `NO_RESELECTION_AFTER_100_TRAJECTORY_RUN`
- `ONE_HUNDRED_TRAJECTORIES_CONFIRMED`
- `SAME_WINDOW_PER_DATASET_CONFIRMED`
- `SAME_TRUTH_PER_DATASET_CONFIRMED`
- `SAME_RAW_SCALE_PER_DATASET_CONFIRMED`
- `SAME_YLIM_PER_DATASET_CONFIRMED`
- `FROZEN_TAU_1P5_ONLY_CONFIRMED`
- `NO_OFFICIAL_FINAL_ARTIFACT_MUTATION`
