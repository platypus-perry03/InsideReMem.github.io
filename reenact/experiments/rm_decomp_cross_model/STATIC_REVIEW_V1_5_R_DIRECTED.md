# Cross-model R-direction Search v1.5 Static Review

Status: **PASS — EXECUTION MAY BE HASH-LOCKED; NO R-DIRECTION RESULT EXISTS YET**  
Review date: `2026-09-01`

## Reviewed files

- `design_v1_5_r_directed_frozen.json`
- `run_cross_model_r_directed_v1_5.py`
- `test_cross_model_r_directed_v1_5.py`
- `freeze_execution_authorization_v1_5.py`
- `launch_r_directed_v1_5_tmux.sh`

## Symmetry audit against M-direction v1.3/v1.4

The R-direction runner preserves the M-direction implementation and changes only
the preregistered pole-specific rule and result labels.

| Stage | M-direction rule | R-direction v1.5 rule | Review |
|---|---|---|---|
| Discovery sign | `mean_M(c)<0` and `Delta>0` | `mean_R(c)>0` and `Delta>0` | PASS |
| Candidate cap | global top 5 Heads + top 5 Neurons | unchanged | PASS |
| Depth scope | all Transformer blocks; no depth quota | unchanged | PASS |
| Heldout sign | `mean_M<0`, `Delta>0` | `mean_R>0`, `Delta>0` | PASS |
| Heldout FDR | candidate-family BH `q<.05` | unchanged | PASS |
| Suppression | final prompt token, alpha 0.5/1.0 | unchanged | PASS |
| Controls | matched + three deterministic random controls | unchanged | PASS |
| Causal endpoint | reduction in absolute final-layer R/M gap | unchanged | PASS |
| Strict gate | CI, permutation FDR, dose, both controls | unchanged | PASS |

## Safety and scope review

| Check | Result |
|---|---|
| Dataset and 2,400/600 split are fixed | PASS |
| Each model uses its frozen layerwise LiReF direction | PASS |
| Model weights are never updated | PASS |
| Discovery and validation capture only final-token component values | PASS |
| Full hidden-state tensors are not persisted | PASS |
| Candidate IDs and controls are frozen before heldout inspection | PASS |
| Heldout failure cannot be replaced by another candidate | PASS |
| Zero eligible candidates is an allowed result | PASS |
| Suppression hooks are temporary and removed after inference | PASS |
| Existing Gap and M-direction outputs are never overwritten | PASS |
| `result.pdf` automatic modification is forbidden | PASS |

## Model-free verification

Command:

```text
/home/jinhyun/.conda/envs/torch/bin/python -m unittest experiments/rm_decomp_cross_model/test_cross_model_r_directed_v1_5.py -v
```

Result: **11/11 PASS**.

Tests cover whole-depth scope, depth labels without quotas, component IDs, BH-FDR,
gap-reduction sign, final-token Head suppression, final-token neuron mean ablation,
capture reset, deterministic discovery-only selection, the valid zero-candidate
outcome, four-model design integrity, and PDF-update prohibition.

## Fail-closed execution

- Execution requires exact hashes for the design, implementation, this review,
  dataset, split, four layerwise LiReF files, and every model config/index/shard.
- GPU mapping, batch size, and float32 dtype are authorization-locked.
- A direction-alignment cosine below `0.999` aborts that model.
- Existing completed output is never overwritten.

## Claim boundary

A strict PASS means that a component writes a relatively stronger R-direction
contribution on R items and supports the measured final-layer R/M representation
gap under this fixed dataset and prompt. It does not establish a reasoning neuron,
behavioral necessity, a complete circuit, or an independent replication.

## Conclusion

The v1.5 implementation is a valid directional mirror of the frozen M-direction
protocol and may be hash-locked for the four authorized base-model runs.
