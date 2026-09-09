# Response to Reviewer #2 Comment 4

We added a focused ETTh-128 true-unconditional PETS-U placement ablation. All trained variants use the same Transformer backbone, production TimesBlock, routing interface, learnable hard top-k mechanism, objective, checkpoint rule, and evaluators. E + MidD and E + LastBeforeD contain exactly two TimesBlocks and differ only in the decoder insertion index. E-only removes the decoder-side TimesBlock. All-Layers was not trained and is reported strictly as a six-TimesBlock forward-cost profile. Each trained placement uses three model seeds and five paired generation draws; Anti-Anchor and PETS-F components are excluded.

| Placement | #TB | Params | context_fid | correlational_score | discriminative_score | predictive_score |
| --- | --- | --- | --- | --- | --- | --- |
| E-only | 1 | 1017547 | 0.755185 ± 0.0249256 | 0.0962101 ± 0.00944738 | 0.152236 ± 0.00948358 | 0.115511 ± 0.0013919 |
| E + MidD | 2 | 1591564 | 0.765428 ± 0.0753103 | 0.0895244 ± 0.00931858 | 0.128306 ± 0.00248278 | 0.116172 ± 0.000935089 |
| E + LastBeforeD | 2 | 1591564 | 0.900467 ± 0.0463813 | 0.0775455 ± 0.00734094 | 0.148862 ± 0.0139885 | 0.115261 ± 0.000357164 |

E + LastBeforeD wins 2 of four metrics against E + MidD and 3 of four against E-only. All-Layers uses 3.0x as many TimesBlocks, 2.44x the parameters, 2.39x the forward latency, and 1.07x the peak memory of the proposed placement; no All-Layers quality conclusion is made.
