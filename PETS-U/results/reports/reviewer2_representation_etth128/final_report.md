# Reviewer #2 controlled ETTh-128 representation comparison

`REVIEWER2_REPRESENTATION_ETTH128_COMPLETE`

All values are three-model-seed mean ± sample SD; all quality metrics are lower-is-better.

| Representation | context_fid | correlational_score | discriminative_score | predictive_score | period_histogram_jsd | spectral_fidelity_error |
| --- | --- | --- | --- | --- | --- | --- |
| Period-Aligned 2D | 0.900467 ± 0.0463813 | 0.0775455 ± 0.00734094 | 0.148862 ± 0.0139885 | 0.115261 ± 0.000357164 | 0.014726 ± 0.00235316 | 0.00146268 ± 3.7944e-05 |
| STFT / Spectrogram | 1.05917 ± 0.0962863 | 0.105225 ± 0.0118319 | 0.174241 ± 0.0127661 | 0.116107 ± 0.00110963 | 0.0237293 ± 0.00742029 | 0.00145913 ± 9.58488e-05 |
| Frequency-as-Channel | 0.847891 ± 0.131239 | 0.126261 ± 0.00469049 | 0.165949 ± 0.0170268 | 0.117746 ± 0.00127999 | 0.0426206 ± 0.00393783 | 0.00140816 ± 6.89523e-05 |

## Efficiency

| representation | representation_key | mean_sampling_seconds_per_generated_batch | sampling_seconds_sd | mean_peak_gpu_memory_mb | peak_gpu_memory_mb | total_trainable_parameters | representation_module_parameters_combined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frequency-as-Channel | frequency_as_channel | 38.92699113488197 | 0.05919552997723771 | 3696.5831380208333 | 3696.85302734375 | 1593528 | 1149998 |
| Period-Aligned 2D | period_aligned_2d | 147.6744728386402 | 1.5640400511052215 | 580.0978190104166 | 581.43115234375 | 1591564 | 1148034 |
| STFT / Spectrogram | stft_spectrogram | 126.50600475668907 | 0.07600408297612808 | 555.59423828125 | 555.70361328125 | 1591716 | 1148186 |

## Scientific questions

- **1_period_aligned_outperforms_stft:** `{"answer": true, "winning_metrics": ["context_fid", "correlational_score", "discriminative_score", "predictive_score", "period_histogram_jsd"]}`
- **2_period_aligned_outperforms_frequency_as_channel:** `{"answer": true, "winning_metrics": ["correlational_score", "discriminative_score", "predictive_score", "period_histogram_jsd"]}`
- **3_best_period_distribution:** `"Period-Aligned 2D"`
- **4_best_spectral_fidelity:** `"Frequency-as-Channel"`
- **5_stft_wins_any_metric:** `{"answer": true, "metrics": ["curvature_fidelity_error", "turning_point_fidelity_error"]}`
- **6_frequency_channel_efficiency_advantage:** `{"answer": true, "sampling_speedup_vs_period_aligned": 3.7936266979111837, "peak_memory_ratio_vs_period_aligned": 6.372344485498669, "individual_metric_wins": ["context_fid", "spectral_fidelity_error", "slope_fidelity_error"]}`
- **7_quality_efficiency_tradeoff:** `"Frequency-as-Channel is 3.79x faster per generated batch but uses 6.37x the peak sampling memory of Period-Aligned 2D. STFT is 14.3% faster than Period-Aligned 2D. Period-Aligned 2D wins four primary metrics and period-JSD; no composite score was used."`
- **8_cycle_phase_geometry_supported:** `{"answer": "PARTIALLY_SUPPORTED", "evidence": "Period-Aligned 2D is best on period-histogram JSD and on the correlational, discriminative, and predictive scores.", "limitation": "Frequency-as-Channel is best on Context-FID and spectral fidelity, while STFT is best on curvature and turning-point fidelity; therefore the evidence supports period-distribution preservation, not universal superiority on every cycle-related metric."}`
- **9_claim_moderation:** `"Restrict the claim to a controlled ETTh-128 comparison. State that Frequency-as-Channel wins Context-FID, spectral fidelity, and slope fidelity, while STFT wins curvature and turning-point fidelity."`

## Integrity

- `ONLY_ETTH128_EXECUTED`
- `TRUE_UNCONDITIONAL_PETSU_ONLY`
- `ANTI_ANCHOR_NOT_USED`
- `NO_PETSF_COMPONENT_USED`
- `NO_DATASET_EXPANSION`
- `NO_HYPERPARAMETER_SEARCH`
- `NO_TEST_DRIVEN_TUNING`
- `SAME_BACKBONE_CONFIRMED`
- `SAME_E_LASTBEFORED_PLACEMENT_CONFIRMED`
- `SAME_TOPK_POLICY_CONFIRMED`
- `SAME_OBJECTIVE_CONFIRMED`
- `SAME_EVALUATORS_CONFIRMED`
- `PAIRED_GENERATION_PROTOCOL_CONFIRMED`
- `ALL_RESULTS_REPORTED`
- `HISTORICAL_ARTIFACTS_UNCHANGED`
