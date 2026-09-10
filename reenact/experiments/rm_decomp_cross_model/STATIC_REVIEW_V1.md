# Cross-model Functional Homologue v1 정적 검토

상태: **PASS — MODEL EXECUTION MAY BE AUTHORIZED BY A SEPARATE HASH-LOCKED FILE**  
검토일: `2026-08-31`

## 검토 범위

- frozen design과 구현의 dataset, split, prompt, model, dtype, layer 범위 일치
- Discovery에서만 후보를 선택하고 held-out 결과로 후보를 교체하지 않는지 확인
- Head/Neuron 측정식과 last-token-only intervention 위치 확인
- hook이 측정 또는 동결된 억제만 수행하며 weight를 수정하지 않는지 확인
- 후보 외 추가 탐색, hidden-state 파일 저장, `result.pdf` 자동 수정이 없는지 확인
- 출력 schema, 재실행 시 overwrite 방지, hash/provenance 확인

## 판정

| 항목 | 판정 | 근거 |
|---|:---:|---|
| 모델 범위 | PASS | frozen 목록의 Mistral, OLMo, Gemma만 허용 |
| 입력 고정 | PASS | 기존 3,000문항과 기존 Discovery/held-out split만 사용 |
| 방향 누수 방지 | PASS | LiReF 방향은 Discovery 2,400문항에서 재계산; 기존 artifact는 정렬 검증에만 사용 |
| 후보 선택 누수 방지 | PASS | 각 모델에서 Discovery top-5 Head + top-5 Neuron만 동결 |
| held-out 검증 | PASS | 동결 후보 10개에만 same-sign 및 BH-FDR 적용 |
| intervention 범위 | PASS | 마지막 prompt token의 선택 component만 alpha 0.5/1.0으로 변경 |
| control | PASS | 동일 type/layer의 low-association pool에서 matched 1개 + random 3개 |
| causal gate | PASS | gap-reduction CI, permutation FDR, dose, matched/random 우월을 모두 요구 |
| architecture caveat | PASS | OLMo/Gemma의 discovery scalar를 pre-normalization screening proxy로 제한 |
| weight mutation | PASS | optimizer/backward 없음; parameter 쓰기 없음; `eval` + inference mode |
| hook cleanup | PASS | 모든 capture/intervention hook을 `finally`에서 제거 |
| 저장 범위 | PASS | scalar/statistics만 저장; full hidden/pre-O/FFN tensor 저장 없음 |
| 결과 PDF | PASS | 실행기가 `result.pdf`를 수정하지 않으며 frozen design도 금지 |
| overwrite | PASS | model summary가 있으면 재실행을 거부 |
| GPU lock | PASS | physical GPU 환경변수와 logical `cuda:0`을 authorization과 대조 |
| model lock | PASS | config, weight index, 모든 safetensor shard SHA-256 재검증 |

## Model-free test

`test_cross_model_homologue_v1.py`: **8/8 PASS**

- 상대 layer 범위
- component ID parser
- BH-FDR
- gap reduction 부호
- Head last-token block suppression
- Neuron last-token mean ablation
- deterministic Discovery-only candidate/control selection
- `result.pdf` 자동 수정 금지

## 명시적 해석 제한

이 검토는 코드가 동결된 설계를 구현하는지를 확인한다. 실험 결과가 아직 없으므로 기능적 homologue의 존재를 보증하지 않는다. 같은 번호·같은 weight-level neuron 정렬·보편적 reasoning mechanism은 본 연구의 주장 범위가 아니다.
