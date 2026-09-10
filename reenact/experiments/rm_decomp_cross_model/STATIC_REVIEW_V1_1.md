# Cross-model Functional Homologue v1.1 정적 검토

상태: **PASS — NEW V1.1 HASH-LOCKED AUTHORIZATION MAY BE CREATED**  
검토일: `2026-08-31`

## v1과의 상태 분리

- v1 model inference: **not executed**
- v1 late-15%-only authorization: **superseded and forbidden**
- v1.1: 모든 Transformer block을 검색하는 별도 frozen design
- v1 결과를 v1.1 결과와 합치거나 threshold를 이전하지 않음

## 정적 판정

| 항목 | 판정 | 확인 내용 |
|---|:---:|---|
| whole-depth 범위 | PASS | `0..num_hidden_layers-1` 모든 block 포함 |
| depth 표지 | PASS | 상대 깊이로 Early/Middle/Late만 사후 표시 |
| depth quota 없음 | PASS | 구간별 강제 선발 없음; 각 구간 후보 0개 허용 |
| Discovery 후보 동결 | PASS | 모델 전체 positive Delta에서 종류별 최대 5개; held-out 교체 금지 |
| 0-candidate 처리 | PASS | 양의 후보/held-out 생존자가 없으면 suppression 0개로 종료 가능 |
| held-out gate | PASS | same-sign 및 동결 후보 family BH `q<.05` 생존자만 억제 |
| 전체 component 출력 | PASS | Discovery와 held-out scalar 통계를 모든 layer에서 저장 |
| causal intervention | PASS | 생존 후보의 마지막 prompt token만 alpha `.5/1` 변경 |
| controls | PASS | Discovery-only 동일 type/layer low-association matched 1 + random 3 |
| causal criteria | PASS | bootstrap, permutation FDR, dose, matched/random 우월을 모두 요구 |
| architecture caveat | PASS | OLMo/Gemma contribution은 pre-normalization screening proxy로 제한 |
| model mutation | PASS | optimizer/backward/parameter write 없음; inference mode |
| hook cleanup | PASS | measurement/intervention hook을 `finally`에서 제거 |
| tensor persistence | PASS | full hidden/pre-O/z tensor 파일 저장 없음 |
| provenance | PASS | design/code/review/input/model shards를 새 v1.1 authorization에 고정 |
| GPU/dtype/batch | PASS | physical GPU 1/2/3, logical `cuda:0`, FP32, batch 4 검증 |
| 결과 PDF | PASS | `result.pdf` 자동 수정 금지 |

## Model-free test

`test_cross_model_homologue_v1.py`: **11/11 PASS**

- 모든 layer 포함
- depth band 경계
- depth와 무관한 deterministic global candidate selection
- 후보 0개 허용
- component ID parser
- BH-FDR 및 gap-reduction 부호
- Head last-token block suppression
- Neuron last-token mean ablation
- batch reset이 설치된 hook의 target dictionary identity를 유지
- hook 제거
- `result.pdf` 수정 금지

## 주장 경계

통과 결과는 동일 R/M dataset·prompt에서의 기능적 recurrence다. 같은 번호, weight-level 정렬, 보편적 reasoning mechanism, R/M 입력 feature 또는 행동 인과를 의미하지 않는다. Early/Middle/Late 분포는 전 구간 검색 결과로 보고하되 후보 수가 적다는 점을 함께 제시한다.
