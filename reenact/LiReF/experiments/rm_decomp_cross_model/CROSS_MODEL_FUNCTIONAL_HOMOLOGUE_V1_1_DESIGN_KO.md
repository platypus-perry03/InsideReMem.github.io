# Cross-model Functional Homologue Search v1.1 설계

상태: **FROZEN — WHOLE-DEPTH SEARCH — MODEL EXECUTION REQUIRES NEW HASH-LOCKED AUTHORIZATION**  
동결일: `2026-08-31`

## 변경 이유

실행되지 않은 v1은 각 모델의 후반 15% layer만 검색했다. 이는 다른 모델의
기능적 대응물이 중간층에 있을 가능성을 설계 단계에서 배제한다. v1.1은 모든
Transformer block을 열어 두고 위치를 결과로 확인한다. v1 결과는 존재하지
않으며 v1 authorization은 사용하지 않는다.

## 전체 검색과 깊이 표시

각 모델의 모든 Head와 FFN neuron을 Discovery 2,400문항에서 측정한다. 상대
깊이 `(layer_index + 1) / num_layers`로만 다음 표지를 붙인다.

- Early: `depth <= 1/3`
- Middle: `1/3 < depth <= 2/3`
- Late: `depth > 2/3`

깊이 구간별 후보 할당량은 없다. 각 component 종류에서 양의
`R mean - M mean`이 큰 순서로 모델 전체 상위 최대 5개만 고정한다. 따라서
Early/Middle/Late 중 어느 구간도 후보 0개가 될 수 있고, 한 구간에 후보가
몰릴 수도 있다. held-out 결과를 보고 후보를 교체하지 않는다.

## 독립 생존 gate와 억제

동결 후보 최대 10개를 held-out 600문항에서 검증한다. 같은 양의 부호와 후보
family BH-FDR `q < .05`를 만족한 후보만 causal suppression으로 진행한다.
생존 후보가 없으면 suppression 후보도 0개이며 그 결과를 그대로 보고한다.

생존 후보에는 last-token-only alpha `.5/1.0` 억제를 적용하고, 동일
type/layer의 low-association pool에서 Discovery 값만으로 동결한 matched 1개와
random 3개 control을 alpha `1.0`으로 비교한다. functional homologue 판정은
다음을 모두 요구한다.

1. held-out same-sign 및 후보-family BH `q < .05`
2. gap reduction bootstrap 95% CI lower bound `> 0`
3. 생존 후보 family 내 delta-G permutation BH `q < .05`
4. alpha `0 -> .5 -> 1`에서 절대 gap이 단조 감소
5. candidate-minus-matched CI lower bound `> 0`
6. candidate-minus-random-mean CI lower bound `> 0`

bootstrap/permutation은 `5,000`, seed는 `20260831`로 고정한다.

## 고정 입력과 해석 제한

- 기존 MMLU-Pro 3,000문항, 기존 Discovery 2,400 / held-out 600
- 기존 `memory_reason_score > .5` R label 및 동일 prompt
- 각 모델 자체의 Discovery LiReF 방향
- Mistral, OLMo-2, Gemma-2 base / FP32 / GPU 1,2,3
- full component 표와 depth 분포를 저장하되 full hidden tensor는 저장하지 않음
- 같은 번호 또는 weight-level correspondence를 주장하지 않음
- input feature, 보편적 reasoning mechanism, 행동 인과를 주장하지 않음
- 실험 종료 전 `result.pdf`를 자동 수정하지 않음
