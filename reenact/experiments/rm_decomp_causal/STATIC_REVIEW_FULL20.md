# Meta-Llama frozen 20-candidate gap validation 정적 검토

상태: **PASS — GPU SANITY/GAP EXECUTION MAY BE AUTHORIZED SEPARATELY**  
검토일: `2026-08-31`

## 고정 범위

- Stage A `detailed_candidate=true` 20개를 누락 없이 사용한다.
- Stage B 결과로 후보를 제거하거나 교체하지 않는다.
- 자연 MMLU-Pro heldout 600문항과 frozen Discovery LiReF Layer 31 방향을 사용한다.
- 각 후보는 마지막 prompt token에서 50%/100% 억제한다.
- 후보당 frozen matched control 1개와 random control 3개를 사용한다.
- permutation BH-FDR은 100% 억제 후보 20개 family에 적용한다.
- joint intervention과 synthetic mediation은 이번 run에서 실행하지 않는다.

## 코드 검토

- hook은 입력 tensor를 clone한 뒤 지정 Head block 또는 Neuron scalar의 마지막 token만 수정한다.
- optimizer, backward, parameter update, activation patching, 후보 재탐색이 없다.
- 전체 hidden state를 파일로 저장하지 않는다.
- 출력은 per-item scalar score·출력 진단값과 집계 통계다.
- 기존 `v2_c01` 결과를 덮어쓰지 않고 새 output root `c_causal_v2_c02_full20`을 사용한다.
- `gap-report`는 모델을 호출하지 않고 동결된 결과만 집계한다.
- `result.pdf`는 모델 실행 코드가 자동 수정하지 않는다.

## PASS 기준

각 후보는 아래를 모두 만족해야 strict PASS다.

1. Stage A heldout에서 Discovery와 같은 Δ 부호 및 BH-FDR `q<.05`
2. 100% 억제 gap 감소 bootstrap 95% CI 하한 `>0`
3. 후보 20개 permutation BH-FDR `q<.05`
4. `|G|`: baseline ≥ 50% 억제 ≥ 100% 억제
5. 후보-minus-matched gap 감소 CI 하한 `>0`
6. 후보-minus-random-mean gap 감소 CI 하한 `>0`

## Model-free 검증

- `python -m unittest test_causal.py -v`: **5/5 PASS**
- full-20 조건 검사: candidate 60 logical rows, control 20 test rows in fixture, joint 0
- 실제 prepare manifest: candidate conditions 60, frozen control conditions 80, baseline 1, 총 141

## 해석 제한

20개 전수 최종 시험으로 Meta-Llama 내부의 기존 Stage B 필터 편향은 제거한다. 그러나 cross-model 후보는 전역 positive-Δ top 5 Head + top 5 Neuron 방식으로 지명되었으므로, PASS 개수를 component 총량처럼 직접 비교할 수는 없다.

