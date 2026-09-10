# Cross-model Functional Homologue Search v1.1 결과

상태: **COMPLETE — PARTIAL CROSS-MODEL RECURRENCE**  
실행일: `2026-08-31`

## 1. 질문과 범위

Meta-Llama-3-8B의 `L31N13336`, `L29H30`, `L30H6`, `L29H31`과 번호가
같은 부품을 찾은 것이 아니다. Mistral, OLMo, Gemma base 모델의 모든
Transformer block에서 다음 기능 조건을 만족하는 Head/Neuron이 반복되는지
검사했다.

> Discovery R/M 선택성 → held-out same-sign/FDR → suppression dose response →
> 최종 R/M LiReF gap 감소 → matched/random control보다 큰 효과

각 모델에서 전체 component를 측정한 뒤, depth quota 없이 양의 Discovery
기여가 큰 Head 최대 5개와 Neuron 최대 5개만 고정했다. 따라서 아래 `5개
PASS`는 전체 모델에 후보가 정확히 5개뿐이라는 뜻이 아니라, **사전에 정한
top-5 screening cap에서 5개 모두 통과했다**는 뜻이다.

## 2. 최종 판정

| Model | Held-out survivors | Strict PASS Heads | Strict PASS Neurons | PASS depth | Class recurrence | Meta-like 3H+1N |
|---|---:|---:|---:|---|:---:|:---:|
| Mistral-7B-v0.3 | 10/10 | 5/5 | 5/5 | Late H 5; Late N 4; Middle N 1 | **YES** | **YES** |
| OLMo-2-1124-7B | 10/10 | 5/5 | 5/5 | Late H 5; Late N 5 | **YES** | **YES** |
| Gemma-2-9B | 10/10 | 0/5 | 0/5 | none | **NO** | **NO** |

세 모델 모두에서 recurrence가 성립하지 않았으므로 보편적 구조 판정은
**NO**다. 다만 Mistral과 OLMo에서는 Meta-Llama와 유사하게 소수의 Head와
Neuron이 R/M representation gap에 선택적으로 기여하는 기능 패턴이 강하게
반복됐다.

## 3. 통과 component

### Mistral-7B-v0.3

- Head, Late: `L29H29`, `L29H30`, `L31H14`, `L31H13`, `L29H31`
- Neuron, Late: `L23N11041`, `L31N8035`, `L31N11179`, `L31N1186`
- Neuron, Middle: `L15N4890`

Baseline absolute gap은 `15.5031`이었다. alpha=1 suppression의 gap reduction은
Head `0.1597–0.8789`, Neuron `0.0823–0.2545`였고, 모든 후보에서 bootstrap
CI·permutation FDR·dose·matched/random control 기준을 통과했다.

### OLMo-2-1124-7B

- Head, Late: `L30H30`, `L31H0`, `L31H25`, `L29H9`, `L30H0`
- Neuron, Late: `L31N4479`, `L31N5664`, `L30N9739`, `L29N2564`, `L30N1652`

Baseline absolute gap은 `60.6453`이었다. gap reduction은 Head `0.5214–6.9722`,
Neuron `2.9486–9.1117`이었고, top-5 Head와 top-5 Neuron이 모두 strict gate를
통과했다.

### Gemma-2-9B

Discovery/held-out 선택성 후보는 Head 5개와 Neuron 5개였으며, Head 4개와
Neuron 5개가 Early, Head 1개가 Late였다. 그러나 suppression에서 strict PASS는
0개였다.

- Late `L38H8`: gap을 줄이지 않고 오히려 `32.2428` 증가
- Early 후보 다수: gap-reduction CI가 0을 포함하거나 dose/control 기준 실패
- `L06N5587`: gap-reduction·dose·control 기준은 통과했지만 permutation
  BH `q=0.2460`으로 사전 기준 실패

따라서 Gemma의 Early 선택성은 **상관 후보**로는 관찰됐지만 최종 R/M gap에
대한 안정적인 기능적 homologue로 인정하지 않는다.

## 4. 전체 깊이 검색이 바꾼 점

후반 15%만 보던 미실행 v1과 달리 v1.1은 모든 layer를 검색했다. 그 결과
Mistral의 `L15N4890` Middle 후보를 실제 strict PASS로 포착했다. Gemma에서도
Early 상관 후보가 다수 발견됐지만 causal gate에서 탈락했다.

따라서 가장 정확한 깊이 결론은 다음이다.

> **통과한 기능적 component는 대체로 후반부에 집중됐지만, 후반부에만
> 존재하지는 않았다. Mistral에서는 Middle Neuron 하나가 재현·suppression·
> control 기준을 모두 통과했다.**

## 5. 결론과 제한

> **Meta-Llama에서 관찰한 ‘소수의 Attention Head와 FFN Neuron이 R/M
> representation gap에 기여한다’는 기능 패턴은 Mistral과 OLMo에서 반복됐지만,
> Gemma에서는 엄격한 causal/control 기준으로 반복되지 않았다. 따라서 부분적
> cross-model 일반화이며 보편적 R/M mechanism의 증거는 아니다.**

이 결과는 같은 번호의 component 대응, weight-level 회로 동형성, Reasoning
Neuron, 행동 성능의 원인 또는 R/M을 구분하는 입력 Feature를 뜻하지 않는다.
OLMo/Gemma의 discovery contribution은 post-normalization 이전 screening
proxy이며, 최종 판정은 실제 final-layer endpoint suppression에 근거했다.

## 6. 무결성 검증

- Mistral full components: `459,776`
- OLMo full components: `353,280`
- Gemma full components: `602,784`
- 각 모델 intervention scalar rows: `36,600`
- summary SHA와 complete marker 일치
- candidate가 각 모델 전체 positive Discovery global top-5/type와 일치
- forbidden hidden/pre-O/z tensor persistence: 없음
- `result.pdf`: 실험 실행 중 자동 수정하지 않음

상세 원시 결과: `AI/reenact/liref_outputs/rm_decomp/cross_model_v1_1/`
