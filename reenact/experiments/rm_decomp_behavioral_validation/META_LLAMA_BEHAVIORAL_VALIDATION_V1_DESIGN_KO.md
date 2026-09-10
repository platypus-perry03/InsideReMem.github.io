# Meta-Llama Behavioral Validation v1 설계 동결

상태: **FROZEN — 구현·model-free test 허용, GPU 실행은 별도 hash-lock 이후만 허용**

## 질문

기존에 R/M LiReF representation gap 감소를 보인 Meta-Llama 후보를 억제했을 때 실제 객관식 정답 행동도 변하는가?

## 대상

- 모델: local Meta-Llama-3-8B base
- 평가 집단: 기존 heldout 600문항(M 324, R 276)
- 후보: 일반 gap strict PASS 9개와 M-directed strict PASS 7개의 합집합 13개
- 각 후보의 기존 frozen matched control 1개와 random control 1개
- 같은 600문항이 후보 검증에 이미 사용됐으므로 결과는 **same-sample exploratory evidence**로만 부른다.

## 행동 측정

모든 선택지를 프롬프트에 제시하고 `A`–`J`의 한 토큰 label을 비교한다.

- forced-choice accuracy
- 선택지들 안에서 정규화한 correct-choice probability
- correct-choice log probability
- correct 대 best foil margin
- vocabulary top-1이 유효한 선택지 label인지 여부(secondary)

## Baseline gate

M과 R 각각에서 다음을 모두 만족해야 candidate suppression으로 넘어간다.

- forced-choice accuracy ≥ 0.20
- forced-choice accuracy ≥ 해당 문항들의 평균 uniform chance + 0.05
- 모든 정답 인덱스가 유효함
- A–J label이 tokenizer에서 각각 단일 continuation token임

FAIL이면 후보 intervention을 실행하지 않고 baseline 결과만 보존한다.

## Intervention

- 기존 실험과 동일하게 마지막 prompt token에서만 read/write component를 억제한다.
- 후보: α=0.5, 1.0
- matched/random control: α=1.0
- Head: 해당 query-head block을 비례 축소
- Neuron: 기존 Discovery mean으로 비례 mean-ablation

## 판정

R과 M을 별도로 분석한다. strict behavioral signal은 같은 집단에서 다음을 모두 요구한다.

1. accuracy drop bootstrap CI lower > 0
2. paired McNemar one-sided BH-q < .05
3. correct probability drop bootstrap CI lower > 0
4. probability sign-flip BH-q < .05
5. probability drop이 0% ≤ 50% ≤ 100% 억제로 단조 증가
6. candidate 효과가 matched와 random control보다 크고 두 차이의 CI lower > 0

정답 확률만 변하면 `probability-only signal`, 내부 gap만 변하고 행동 지표가 없으면 `no behavioral signal`로 보고한다.

## 주장 제한

성공해도 이 결과는 해당 프롬프트·데이터에서의 행동 기여다. 추론 뉴런, 암기 저장소, 완전한 회로, 독립 재현을 뜻하지 않는다.
