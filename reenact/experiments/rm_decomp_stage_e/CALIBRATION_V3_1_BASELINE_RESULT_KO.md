# Baseline Calibration v3.1 실행 결과

상태: **COMPLETE — ALL FAMILIES FAIL — STAGE E PILOT NOT AUTHORIZED**  
실행일: `2026-08-30`  
결과 run ID: `calv3_1_baseline_20260830_02`

## 1. 실행 및 provenance

- 공식 명칭: **protocol-authorized AI-only-audited Baseline Calibration v3.1**
- model: local `Meta-Llama-3-8B`
- dataset: frozen v3 `192 pairs / 384 prompts`
- device / batch / dtype: `cuda:1` / `8` / `float32`
- human audit: `not_performed`
- model result 범위: teacher-forced candidate score, forced-choice, one-token generation만 허용
- LiReF direction, candidate component, hidden state, hook, patching, suppression, intervention: 사용하지 않음
- Stage E Pilot: 실행하지 않음

초기 승인 run `calv3_1_baseline_20260830_01`은 system Python에 `torch`가 없어 model loading 전에 종료됐다. GPU 계산과 model forward는 발생하지 않았으며 실패 기록을 보존했다. 동일한 frozen implementation, dataset, threshold와 실행 설정을 유지한 별도 retry authorization으로 `run02`를 실행했다.

## 2. Family별 결과

| Family | A forced-choice | S forced-choice | A generation | S generation | mean(`D_k`) | `d_z` | 최종 |
|---|---:|---:|---:|---:|---:|---:|---|
| object-count | 36/64 | 36/64 | 8/64 | 26/64 | +0.1574 | +0.5170 | FAIL |
| points-balance | 35/64 | 32/64 | 3/64 | 15/64 | +0.0411 | +0.2190 | FAIL |
| temperature | 36/64 | 40/64 | 5/64 | 31/64 | -0.0006 | -0.0023 | FAIL |

`A`는 Arithmetic, `S`는 Selector 조건이다.

### 2.1 object-count

| Frozen criterion | 결과 | 판정 |
|---|---:|---|
| Arithmetic forced-choice 40–56/64 | 36/64 | FAIL |
| Selector forced-choice 40–56/64 | 36/64 | FAIL |
| forced-choice count gap ≤6 | 0 | PASS |
| Arithmetic generation 16–56/64 | 8/64 | FAIL |
| Selector generation 16–56/64 | 26/64 | PASS |
| generation count gap ≤6 | 18 | FAIL |
| `|mean(D_k)| ≤ 0.40` | 0.1574 | PASS |
| `|d_z| ≤ 0.30` | 0.5170 | FAIL |

### 2.2 points-balance

| Frozen criterion | 결과 | 판정 |
|---|---:|---|
| Arithmetic forced-choice 40–56/64 | 35/64 | FAIL |
| Selector forced-choice 40–56/64 | 32/64 | FAIL |
| forced-choice count gap ≤6 | 3 | PASS |
| Arithmetic generation 16–56/64 | 3/64 | FAIL |
| Selector generation 16–56/64 | 15/64 | FAIL |
| generation count gap ≤6 | 12 | FAIL |
| `|mean(D_k)| ≤ 0.40` | 0.0411 | PASS |
| `|d_z| ≤ 0.30` | 0.2190 | PASS |

### 2.3 temperature

| Frozen criterion | 결과 | 판정 |
|---|---:|---|
| Arithmetic forced-choice 40–56/64 | 36/64 | FAIL |
| Selector forced-choice 40–56/64 | 40/64 | PASS |
| forced-choice count gap ≤6 | 4 | PASS |
| Arithmetic generation 16–56/64 | 5/64 | FAIL |
| Selector generation 16–56/64 | 31/64 | PASS |
| generation count gap ≤6 | 26 | FAIL |
| `|mean(D_k)| ≤ 0.40` | 0.0006 | PASS |
| `|d_z| ≤ 0.30` | 0.0023 | PASS |

각 family는 모든 hard criterion을 동시에 만족해야 PASS다. 결과 확인 후 threshold, item, template 또는 family를 변경하거나 제외하지 않았다.

## 3. 진단 해석

- v2/v2.1.1의 Selector forced-choice ceiling은 재현되지 않았다. Selector는 32–40/64로 오히려 일부 family에서 하한보다 낮았다.
- forced-choice 조건 간 gap과 평균 `D_k`는 세 family 모두 기준을 만족했다. 따라서 primary margin의 평균 조건 불균형은 크게 줄었다.
- 그러나 절대 forced-choice 난이도가 대체로 너무 높았고, Arithmetic one-token generation은 모든 family에서 하한 16/64보다 낮았다.
- Arithmetic generation은 정답 대신 start operand를 자주 출력했다: object-count 19/64, points-balance 16/64, temperature 27/64. object-count와 points-balance에서는 delta 출력도 각각 16/64였다.
- Arithmetic SUB frame의 generation 정답은 object-count 1/32, points-balance 0/32, temperature 0/32로 특히 낮았다. 따라서 terminal arithmetic rule, 특히 SUB 적용이 실제 generation behavior에서 충분히 수행되지 않았다.
- 이 결과는 R/M 분리나 candidate component에 대한 결론이 아니다. Baseline behavior만 측정했으며 LiReF와 4개 candidate를 평가하지 않았다.

## 4. 최종 gate

```text
passed_families = []
failed_families = [object_count, points_balance, temperature]
```

PASS family가 없으므로 이 결과를 근거로 Stage E Pilot dataset을 만들거나 Pilot을 실행하지 않는다. 다음 연구 작업은 실패 artifact를 보존한 상태에서 새 design version을 검토하는 것이다.

## 5. 결과 artifact와 hash

- output directory: `AI/reenact/liref_outputs/rm_decomp/v3/calv3_1_baseline_20260830_02/`
- execution authorization SHA-256: `4c78148c666c2c42dd2d2200b4de4ba3ca972e0a425004abf0515190c5483e54`
- pair results SHA-256: `c4562fcfb109083b8c501ac90af64ae5a8f6e6f7f33485619a86366aaec78e6a`
- summary SHA-256: `7c23ae9ea5b6f850185e5aa9cb80855652a3dd5290917c8ab14e1b1562521b0c`
- environment/safety SHA-256: `5f8014171440660841374c827aed5b95a6e89250e4cbdb5196a91dd85f071eda`
- run manifest SHA-256: `277c4dc9a984bef70f0fc12924f663dbdcf9006269a230702e49ff4a15be6d12`

