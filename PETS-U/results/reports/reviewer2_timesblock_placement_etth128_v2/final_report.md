# Reviewer #2 minimal TimesBlock placement ablation

`REVIEWER2_TIMESBLOCK_PLACEMENT_ETTH128_V2_COMPLETE`

All quality values are three-model-seed mean ± sample SD and lower-is-better.

| Placement | #TB | Params | context_fid | correlational_score | discriminative_score | predictive_score |
| --- | --- | --- | --- | --- | --- | --- |
| E-only | 1 | 1017547 | 0.755185 ± 0.0249256 | 0.0962101 ± 0.00944738 | 0.152236 ± 0.00948358 | 0.115511 ± 0.0013919 |
| E + MidD | 2 | 1591564 | 0.765428 ± 0.0753103 | 0.0895244 ± 0.00931858 | 0.128306 ± 0.00248278 | 0.116172 ± 0.000935089 |
| E + LastBeforeD | 2 | 1591564 | 0.900467 ± 0.0463813 | 0.0775455 ± 0.00734094 | 0.148862 ± 0.0139885 | 0.115261 ± 0.000357164 |

## Cost profile

All-Layers is not trained and has no quality claim.

| placement | timesblock_count | total_trainable_parameters | mean_denoiser_forward_latency_ms | std_denoiser_forward_latency_ms | peak_gpu_memory_mb | cost_only |
| --- | --- | --- | --- | --- | --- | --- |
| E + LastBeforeD | 2 | 1591564 | 30.3759666633606 | 0.713775193093678 | 232.4609375 | False |
| E + MidD | 2 | 1591564 | 30.248396797180177 | 0.8707010073825626 | 236.87109375 | False |
| E-only | 1 | 1017547 | 19.42383623123169 | 0.6980139744114101 | 232.5634765625 | False |
| All-Layers (cost only) | 6 | 3887632 | 72.51698677062988 | 0.670997825555335 | 248.18310546875 | True |

## Controlled comparisons

```json
{
  "last_before_d_vs_mid_d": {
    "metric_delta_mid_minus_last": {
      "context_fid": -0.13503921029013732,
      "correlational_score": 0.011978910366694129,
      "discriminative_score": -0.020555555555555577,
      "predictive_score": 0.000911053389002825
    },
    "last_winning_metrics": [
      "correlational_score",
      "predictive_score"
    ],
    "last_wins_majority": false
  },
  "last_before_d_vs_e_only": {
    "metric_delta_e_only_minus_last": {
      "context_fid": -0.14528184100046782,
      "correlational_score": 0.01866456021865208,
      "discriminative_score": 0.003373983739837383,
      "predictive_score": 0.00025054647028606436
    },
    "last_winning_metrics": [
      "correlational_score",
      "discriminative_score",
      "predictive_score"
    ],
    "last_wins_majority": true,
    "parameter_increase": 574017,
    "forward_latency_increase_ms": 10.95213043212891,
    "peak_memory_increase_mb": -0.1025390625
  },
  "all_layers_cost_vs_proposed": {
    "timesblock_count_ratio": 3.0,
    "parameter_increase": 2296068,
    "parameter_ratio": 2.442648866146759,
    "forward_latency_ratio": 2.387314536333741,
    "peak_memory_ratio": 1.0676335909931105,
    "quality_claim": "PROHIBITED_NOT_TRAINED"
  }
}
```

## Integrity

- `ONLY_ETTH128_EXECUTED`
- `TRUE_UNCONDITIONAL_PETSU_ONLY`
- `ANTI_ANCHOR_NOT_USED`
- `NO_PETSF_COMPONENT_USED`
- `ONLY_THREE_TRAINED_PLACEMENTS`
- `NO_EXTRA_PLACEMENT_SEARCH`
- `ALL_LAYERS_COST_ONLY`
- `NO_ALL_LAYERS_TRAINING`
- `NO_ALL_LAYERS_QUALITY_CLAIM`
- `NO_HYPERPARAMETER_SEARCH`
- `NO_TEST_DRIVEN_TUNING`
- `SAME_BACKBONE_CONFIRMED`
- `SAME_TIMESBLOCK_CONFIRMED`
- `SAME_ROUTING_CONFIRMED`
- `SAME_OBJECTIVE_CONFIRMED`
- `SAME_EVALUATORS_CONFIRMED`
- `MID_VS_LAST_MODULE_COUNT_MATCHED`
- `PAIRED_GENERATION_PROTOCOL_CONFIRMED`
- `ALL_RESULTS_REPORTED`
- `HISTORICAL_ARTIFACTS_UNCHANGED`
