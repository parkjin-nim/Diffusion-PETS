# Response to Reviewer #2 — Comments 1 and 2

We added an intentionally focused ETTh experiment at sequence length 128. Period-Aligned 2D, a fixed complex STFT/spectrogram control, and a complex frequency-as-channel control share the same PETS-U backbone, E + before-last D placement, learnable hard top-k policy, objectives, evaluator protocol, three model seeds, and five paired generation draws. Anti-Anchor and forecasting-specific components are excluded.

| Representation | context_fid | correlational_score | discriminative_score | predictive_score | period_histogram_jsd | spectral_fidelity_error |
| --- | --- | --- | --- | --- | --- | --- |
| Period-Aligned 2D | 0.900467 ± 0.0463813 | 0.0775455 ± 0.00734094 | 0.148862 ± 0.0139885 | 0.115261 ± 0.000357164 | 0.014726 ± 0.00235316 | 0.00146268 ± 3.7944e-05 |
| STFT / Spectrogram | 1.05917 ± 0.0962863 | 0.105225 ± 0.0118319 | 0.174241 ± 0.0127661 | 0.116107 ± 0.00110963 | 0.0237293 ± 0.00742029 | 0.00145913 ± 9.58488e-05 |
| Frequency-as-Channel | 0.847891 ± 0.131239 | 0.126261 ± 0.00469049 | 0.165949 ± 0.0170268 | 0.117746 ± 0.00127999 | 0.0426206 ± 0.00393783 | 0.00140816 ± 6.89523e-05 |

The comparison is not generalized beyond ETTh-128. STFT metric wins were: `curvature_fidelity_error, turning_point_fidelity_error`; frequency-as-channel metric wins were: `context_fid, spectral_fidelity_error, slope_fidelity_error`. The period-distribution winner was `Period-Aligned 2D` and the spectral-fidelity winner was `Frequency-as-Channel`.
