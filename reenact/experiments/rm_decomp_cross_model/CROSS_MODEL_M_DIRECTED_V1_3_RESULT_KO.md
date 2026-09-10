# Cross-model M-directed Functional Homologue Search v1.3 결과

상태: **COMPLETE — THREE MODELS EXECUTED — PARTIAL CROSS-MODEL RECURRENCE**  
실행일: `2026-08-31`

## 1. 질문과 정의

이 실험은 다른 base 모델에도 Meta-Llama-3-8B에서 관찰한 것과 기능적으로 비슷한 **M 방향 R/M-gap component**가 존재하는지 검사했다. 같은 layer/head/neuron 번호를 찾는 실험은 아니다.

각 component가 residual stream에 쓰는 벡터를 해당 layer의 frozen M→R LiReF 방향에 투영한 scalar를 `c`라고 했다.

- `mean_M(c) < 0`: M 문항에서 평균적으로 LiReF의 M 방향에 쓰임
- `Delta = mean_R(c) - mean_M(c) > 0`: M 문항에서 R 문항보다 더 M 방향으로 쓰여 R/M gap을 지지함
- `gap reduction = |G_base| - |G_suppressed| > 0`: component를 억제했을 때 최종 R/M representation gap이 감소함

따라서 최종 PASS는 단순한 음수 activation이 아니라, **M 문항에서 M 방향 기여가 재현되고 그 component를 억제하면 최종 R/M gap이 matched/random control보다 더 감소하는 경우**다. 이는 memorization 정확도나 기억 능력을 뜻하지 않는다.

## 2. 실행 범위와 엄격 판정

- 모델: `Mistral-7B-v0.3`, `OLMo-2-1124-7B`, `gemma-2-9b` base
- Discovery 2,400문항, heldout 600문항
- 모든 Transformer layer의 모든 Attention Head와 FFN Neuron 검색
- 후보: Discovery에서 `mean_M(c)<0`, `Delta>0`인 component 중 type별 전역 top 5
- heldout: 같은 방향, 후보군 BH-FDR `q<.05`
- suppression: 마지막 prompt token에서 50%/100%, bootstrap CI, permutation FDR, dose monotonicity, matched/random control 비교
- 엄격 PASS: 위 조건을 모두 만족

## 3. 전체 결과

| Model | Heldout survivors | PASS Head | PASS Neuron | PASS 위치 | Head+Neuron 반복 |
|---|---:|---:|---:|---|:---:|
| Mistral-7B-v0.3 | 10/10 | 5 | 4 | Late H 5 · Late N 1 · Middle N 3 | YES |
| OLMo-2-1124-7B | 10/10 | 3 | 4 | 모두 Late | YES |
| gemma-2-9b | 10/10 | 2 | 0 | Late H 2 | NO |

세 모델 모두에서 Head+Neuron 조합 또는 Meta-Llama형 `3 Head + 1 Neuron` 구조가 반복된 것은 아니다. 따라서 결과는 **부분적 기능 일반화**다.

## 4. 모델별 엄격 PASS component

### Mistral-7B-v0.3 — 9개

Baseline `|G|=15.5031`.

| ID | 종류·위치 | heldout M mean | heldout R mean | Delta | gap 감소 (%base) |
|---|---|---:|---:|---:|---:|
| `L29H29` | Head · Late | -0.04585 | +0.24749 | +0.29334 | 0.87888 (5.67%) |
| `L31H14` | Head · Late | -0.01253 | +0.21919 | +0.23172 | 0.15972 (1.03%) |
| `L29H31` | Head · Late | -0.06986 | +0.12933 | +0.19919 | 0.43851 (2.83%) |
| `L30H30` | Head · Late | -0.00853 | +0.16896 | +0.17749 | 0.36101 (2.33%) |
| `L29H28` | Head · Late | -0.12862 | +0.04296 | +0.17158 | 0.37266 (2.40%) |
| `L23N11041` | Neuron · Late | -0.49037 | -0.22277 | +0.26760 | 0.10907 (0.70%) |
| `L15N4890` | Neuron · Middle | -0.21814 | -0.02272 | +0.19543 | 0.08227 (0.53%) |
| `L18N11690` | Neuron · Middle | -0.17111 | -0.03909 | +0.13202 | 0.01704 (0.11%) |
| `L17N12568` | Neuron · Middle | -0.13103 | -0.01138 | +0.11965 | 0.15032 (0.97%) |

### OLMo-2-1124-7B — 7개

Baseline `|G|=60.6453`.

| ID | 종류·위치 | heldout M mean | heldout R mean | Delta | gap 감소 (%base) |
|---|---|---:|---:|---:|---:|
| `L26H15` | Head · Late | -0.08377 | +1.00618 | +1.08994 | 1.30980 (2.16%) |
| `L30H10` | Head · Late | -0.14807 | +0.69356 | +0.84163 | 0.17532 (0.29%) |
| `L31H13` | Head · Late | -0.29915 | +0.13900 | +0.43815 | 0.19420 (0.32%) |
| `L28N91` | Neuron · Late | -0.12671 | +1.13750 | +1.26421 | 3.04619 (5.02%) |
| `L30N4602` | Neuron · Late | -0.19256 | +1.07076 | +1.26332 | 3.84693 (6.34%) |
| `L30N10993` | Neuron · Late | -0.24784 | +0.80717 | +1.05501 | 0.53746 (0.89%) |
| `L29N3303` | Neuron · Late | -0.01026 | +0.83955 | +0.84981 | 1.25688 (2.07%) |

### gemma-2-9b — 2개

Baseline `|G|=322.2574`.

| ID | 종류·위치 | heldout M mean | heldout R mean | Delta | gap 감소 (%base) |
|---|---|---:|---:|---:|---:|
| `L40H13` | Head · Late | -0.01571 | +0.55823 | +0.57394 | 31.32146 (9.72%) |
| `L38H14` | Head · Late | -0.05588 | +0.41539 | +0.47127 | 11.73553 (3.64%) |

Gemma에서는 M 방향 Head는 확인됐지만 엄격 기준을 통과한 FFN Neuron은 없었다.

## 5. 결론과 제한

> **Mistral과 OLMo에서는 M 문항에서 M 방향으로 더 쓰이고 억제하면 R/M gap이 줄어드는 Head와 FFN Neuron이 함께 확인됐다. Gemma에서는 같은 기능의 Late Head 두 개만 확인됐다. 따라서 M 방향 gap-supporting component는 여러 base 모델에서 부분적으로 반복되지만, 보편적인 동일 회로나 memorization mechanism으로 확정할 수 없다.**

- Head는 특정 정보를 선택·혼합해 residual stream에 쓰는 경로이고, Neuron은 FFN의 개별 feature channel이다.
- PASS는 component의 출력 벡터가 M 문항에서 M 방향으로 더 기여하고, 억제 시 최종 R/M 표현 차이가 줄었다는 뜻이다.
- 실제 정답률·기억 성능·행동적 memorization을 측정한 결과가 아니다.
- 같은 번호 또는 weight-level 정렬을 의미하지 않는다.
- 이 실험은 R/M 차이를 유발하는 입력 Feature를 찾지 않았다.
- v1.2의 `Delta<0` 탐색은 계산 자체는 유효하지만 M-selective라는 명칭이 잘못되어 M 결과에서 제외했다. 본 결과는 수정된 v1.3 정의만 사용한다.

## 6. Provenance

- design SHA-256: `03a2cd0a51b6614f2662f1d64db39cdea2bb970530fd16abf711b8e03989c1d5`
- implementation SHA-256: `fd5c69bec7560c49245a848bb0036b5fa5d5e6df9eb832fd520207cd4fb3be23`
- static review SHA-256: `6c840a3ece7620d2c90a4ecc1c749b3536e387cc326973eaf8cc853b11a67b69`
- authorization SHA-256: `2b7dd445140b726676602cbefcaff185cc6764a3f17f527048ee0680acb369fe`
- combined summary SHA-256: `066247be601d178ac5913729a8f68f8833980243ced59a484683b6b54bc5a213`

