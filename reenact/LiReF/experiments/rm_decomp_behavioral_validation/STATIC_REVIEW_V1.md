# Meta-Llama Behavioral Validation v1 Static Review

Status: **PASS — EXECUTION MAY BE HASH-LOCKED; NO RESULT EXISTS YET**  
Review date: `2026-09-01`

## Reviewed scope

- `design_v1_frozen.json`
- `candidate_manifest_v1_frozen.json`
- `run_meta_llama_behavioral_validation_v1.py`
- `test_behavioral_validation_v1.py`

## Frozen scientific scope

- Model: local `Meta-Llama-3-8B` base only.
- Population: the pre-existing heldout 600 rows (`M=324`, `R=276`).
- Candidate set: frozen union of 13 previously selected components (5 Heads, 8 FFN neurons).
- Candidate interventions: alpha `0.5` and `1.0`.
- Controls: one frozen matched and one frozen random same-type component per candidate, alpha `1.0`.
- Outcomes: forced-choice accuracy and normalized correct-choice probability, with log probability and margin as secondary outcomes.
- Interpretation: same-sample exploratory behavioral validation, not independent replication.

## Safety and implementation review

| Check | Result |
|---|---|
| Model weights are never modified in place | PASS |
| Intervention uses temporary forward pre-hooks only | PASS |
| Hook output changes only the selected Head block or FFN neuron | PASS |
| Intervention is limited to the final prompt token | PASS |
| Candidate Head alpha scales its pre-`o_proj` block exactly once | PASS |
| Candidate neuron alpha moves its pre-`down_proj` scalar toward the frozen reference mean | PASS |
| Hooks are removed in `finally`, and leakage is checked after every condition | PASS |
| Hidden states and attention tensors are neither requested nor saved | PASS |
| LiReF directions are neither loaded nor re-estimated | PASS |
| Candidate discovery or expansion is absent | PASS |
| Only scalar behavioral item results are written | PASS |
| Baseline solvability is evaluated before any intervention | PASS |
| Baseline failure blocks every candidate/control intervention | PASS |
| Existing run directories are never overwritten | PASS |
| `result.pdf` automatic update is forbidden | PASS |

## Model-free verification

Command:

```text
/home/jinhyun/.conda/envs/torch/bin/python -m unittest experiments/rm_decomp_behavioral_validation/test_behavioral_validation_v1.py -v
```

Result: **10/10 PASS**.

The tests cover manifest integrity, component parsing, prompt construction, Head last-token block suppression, neuron last-token mean ablation, hook cleanup, physical-condition deduplication, baseline PASS/FAIL behavior, directional test behavior, and the PDF-update prohibition.

The runner also passed Python bytecode compilation and model-free input/hash preflight. The preflight found the frozen 600-row split, 13 candidates, local model configuration/index, and all four local safetensor shards.

## Fail-closed boundaries

- A mismatch in the implementation, this review, design, candidate manifest, dataset, split, model config/index, or any model shard aborts execution.
- A mismatch in run ID, GPU mapping, dtype, or batch size aborts execution.
- If either R or M baseline fails the frozen solvability gate, the run records the failure and does not test any component.
- A surviving behavioral signal means only that suppressing that component changes answer behavior on this already-used prompt set. It does not establish a reasoning neuron, memorization store, complete circuit, or independent replication.

## Review conclusion

The implementation matches the frozen v1 design and is safe to hash-lock for the single authorized run `meta_llama_behavioral_prevalidation_20260901_01`.
