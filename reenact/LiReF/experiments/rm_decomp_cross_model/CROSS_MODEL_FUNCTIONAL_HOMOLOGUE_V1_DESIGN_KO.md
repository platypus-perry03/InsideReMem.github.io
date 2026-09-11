# Cross-model Functional Homologue Search v1 설계

상태: **FROZEN — IMPLEMENTATION·MODEL-FREE TEST·STATIC REVIEW·LISTED BASE-MODEL EXECUTION ALLOWED**  
동결일: `2026-08-31`

## 1. 연구 질문

Meta-Llama-3-8B에서 확인된 `1 FFN neuron + 3 attention heads`와 번호가 같은
부품을 찾는 것이 아니다. 다음 기능적 패턴이 다른 base model에서도 반복되는지
검증한다.

> **후반부 FFN neuron 또는 attention head가 각 모델 자체의 R/M LiReF 방향에
> 선택적으로 기여하고, held-out에서 같은 방향으로 재현되며, 그 부품을
> 억제하면 최종 R/M representation gap이 control보다 더 감소하는가?**

대상 모델과 실행 순서는 다음으로 고정한다.

1. `Mistral-7B-v0.3`
2. `OLMo-2-1124-7B`
3. `gemma-2-9b`

Instruct 모델은 사용하지 않는다.

## 2. 고정 입력

- dataset: 기존 `mmlu-pro-3000samples.json`
- R label: `memory_reason_score > 0.5`
- M label: `memory_reason_score <= 0.5`
- split: 기존 Discovery 2,400 / held-out 600
- prompt: `Q: {question}\nA: `
- 위치: 마지막 prompt token
- dtype: `float32`
- seed: `20260831`
- 모델 weight 변경·학습·저장: 금지

## 3. Discovery 범위

각 모델의 layer 수를 `L`이라 할 때 `(layer_index + 1) / L >= 0.85`인 후반부
layer만 component screen에 사용한다.

- 32-layer model: layer index `27--31`
- 42-layer model: layer index `35--41`

각 모델의 Discovery R/M 평균 차이로 layer별 unit LiReF direction을 새로
계산한다. 기존 held-out LiReF artifact가 포함하는 final block 이전 layer와
cosine alignment를 검사하며 최소값은 `0.999`로 동결한다.

## 4. Component screening 정의

Attention head:

```text
pre_o head block · (o_proj head block)^T r_hat_layer
```

FFN neuron:

```text
z_j × (down_proj[:, j] · r_hat_layer)
```

Discovery에서 `R mean - M mean > 0`인 component 중 각 종류별 상위 5개를
고정한다. 따라서 모델당 candidate family는 head 5개와 neuron 5개, 총 10개다.
held-out 결과를 보고 후보를 교체하지 않는다.

Mistral에서는 위 값이 residual addition의 직접 선형 분해다. Gemma-2와
OLMo-2는 attention/FFN 출력 뒤 post-normalization이 있으므로 위 값은
pre-normalization screening proxy다. 이 두 모델의 동형 판정은 proxy만으로
내리지 않고 실제 억제 후 최종 layer gap 변화를 필수로 요구한다.

## 5. Held-out 재현

Discovery에서 고정한 10개에 대해 다음을 계산한다.

- held-out R/M contribution difference
- Discovery와 같은 양의 부호
- candidate family 10개 내 Welch test BH-FDR `q < 0.05`

held-out는 후보 재선정에 사용하지 않고 functional-homologue 판정 조건으로만
사용한다.

## 6. 억제와 control

각 고정 candidate를 마지막 prompt token에서만 조작한다.

- head: `pre_o_head' = (1-alpha) × pre_o_head`
- neuron: `z_j' = (1-alpha)z_j + alpha × pooled_Discovery_mean(z_j)`
- candidate alpha: `0.5`, `1.0`
- baseline alpha=0은 한 번만 실행하고 모든 candidate가 공유

각 candidate에는 같은 component 종류와 같은 layer에서 Discovery
`|Delta|`가 중앙값 이하인 low-association pool을 만든다.

- matched control 1개: candidate와 writer-scale proxy가 가장 가까운 component
- random control 3개: seed로 고정한 low-association component
- control alpha: `1.0`

control은 Discovery 값만 사용해 고정하며 held-out 또는 intervention 결과를
보지 않는다.

## 7. Primary endpoint와 판정

Primary endpoint는 각 모델의 마지막 block output을 그 모델의 frozen
Discovery final-layer LiReF direction에 투영한 값이다.

```text
G = mean(score | R) - mean(score | M)
gap_reduction = |G_baseline| - |G_intervention|
```

candidate 하나가 functional homologue criterion을 통과하려면 다음을 모두
만족해야 한다.

1. held-out contribution 부호가 양수이고 candidate-family BH `q < 0.05`
2. alpha=1 gap reduction의 stratified bootstrap 95% CI lower bound `> 0`
3. candidate delta-G permutation test의 10-candidate BH `q < 0.05`
4. `|G(alpha=0)| >= |G(alpha=.5)| >= |G(alpha=1)|`
5. candidate-minus-matched gap reduction bootstrap CI lower bound `> 0`
6. candidate-minus-random-mean gap reduction bootstrap CI lower bound `> 0`

bootstrap/permutation은 각각 `5,000`회로 고정한다.

모델 수준 결과는 다음 두 단계로 보고한다.

- class recurrence: PASS head `>=1` 및 PASS neuron `>=1`
- Meta-like sparse pattern: PASS head `>=3` 및 PASS neuron `>=1`

어느 기준도 threshold를 보고 변경하지 않는다.

## 8. 허용되는 주장과 금지되는 주장

허용:

- 여러 base model에서 R/M LiReF gap에 기여하는 후반부 component class가
  기능적으로 반복되는지 여부
- 각 모델에서 발견된 candidate의 상대 layer 위치와 억제 효과

금지:

- 서로 다른 모델의 component 번호가 직접 대응한다는 주장
- weight-level neuron alignment 또는 회로 동형성 주장
- Reasoning neuron, Memorization head, 보편적 R/M mechanism 주장
- R/M을 다르게 만드는 입력 Feature를 찾았다는 주장
- 결과를 보고 후보 수·후반부 범위·threshold·control 규칙 변경

## 9. 결과 PDF 반영 시점

이번 실험이 끝나기 전에는 `result.pdf`에 cross-model 결과를 넣지 않는다.
세 모델의 결과와 해석 제한이 모두 확정된 뒤 별도로 반영한다.

