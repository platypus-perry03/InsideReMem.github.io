# Stage E Baseline Calibration v3 설계 초안

상태: **FROZEN — DESIGN FREEZE COMPLETE — DATASET BUILD ALLOWED, MODEL EXECUTION NOT AUTHORIZED**  
작성일: 2026-08-30  
목적: v2.1.1에서 남은 retrieval ceiling을 완화하고 answer-exposure scoring 비대칭을 condition 내부에서 통제하여, arithmetic composition과 matched selector-based retrieval을 baseline 난이도가 겹치는 조건에서 비교할 수 있는 새 Stage E calibration을 설계한다.

## 1. 출발 상태와 non-reuse

현재 authoritative status는 `STAGE_E_CURRENT_STATUS_KO.md`를 따른다.

> **Protocol-deviating diagnostic baseline run: all three families failed; official Calibration not performed.**

v2.1.1에서 확보한 것은 다음 두 진단뿐이다.

1. 세 family 모두 matched keyed-retrieval forced-choice가 64/64 ceiling에 도달했고 baseline 난이도 균형을 확보하지 못했다.
2. 별도 exploratory run에서 `L31N13336`의 condition effect가 lexical family에 따라 부호가 반전될 가능성을 관찰했다.

v2/v2.1/v2.1.1의 design, dataset, audit, result, authorization와 provenance는 수정하지 않는다. 기존 item, context, numeric block과 normalized template skeleton은 v3 Development, official Calibration, Pilot 또는 Confirmatory에서 재사용하지 않는다.

## 2. v2.1.1의 두 구조적 실패 원인

### 2.1 Terminal task 난이도 불일치

두 조건 모두 `label -> key -> record`를 거쳤지만 마지막 단계는 다음처럼 달랐다.

- arithmetic condition: 두 operand를 결합해 context에 없는 값을 계산
- keyed-retrieval condition: 선택된 record에 적힌 숫자를 그대로 출력

공통 binding을 추가하는 것만으로는 마지막 retrieval step의 ceiling을 막지 못했다.

### 2.2 Cross-condition alternative의 answer-exposure 비대칭

v2.1.1은 각 prompt에서 다른 condition의 canonical answer를 `y_alt`로 사용했다.

```text
M_R = logP(R correct: context에 없음) - logP(I answer: context에 있음)
M_I = logP(I correct: context에 있음) - logP(R answer: context에 없음)
```

따라서 answer value를 counterbalance해도 correct/alternative의 **노출 역할**은 상쇄되지 않았다. 음의 `D_k`에는 terminal task 난이도뿐 아니라 이 scoring 비대칭도 포함될 수 있다.

v3에서는 다른 condition의 정답을 primary alternative로 사용하지 않는다. 이는 v2.1.1의 교차된 correct/alternative 노출 비대칭을 크게 줄이지만, condition 사이의 절대 노출 차이까지 제거하지는 않는다.

## 3. v3 연구 질문과 operationalized feature

공식 feature 명칭 후보:

> **`arithmetic composition vs selector-guided value retrieval after matched multi-hop binding`**

두 condition은 공통으로 다음 단계를 수행한다.

1. 질문의 target label 식별
2. label과 case key의 binding 조회
3. case의 terminal instruction과 두 value key 조회
4. 공통 value ledger에서 두 숫자 조회
5. 두 candidate 중 답 결정

마지막 결정 규칙만 다르다.

- **Arithmetic condition:** 명시된 증가·감소 규칙과 operands로 값을 산출
- **Selector condition:** case seal/tag와 일치하는 entry를 선택해 그 값을 반환

Selector condition도 숫자 하나를 바로 복사하지 않는다. 두 후보를 모두 조회한 뒤 arbitrary selector binding을 해결해야 한다. 이 조작은 일반적인 Reasoning 대 Memorization 전체가 아니라, 위 두 terminal rule의 통제 비교다.

## 4. 문항 구조 예시

아래는 구조 예시이며 실제 문장, 이름, label, key와 수치는 새 dataset에서 생성한다.

```text
Ava's blue inventory uses case Kappa.
Case Kappa lists base key Luma and change key Vega; its rule is ADD.

Ava's red inventory uses case Tango.
Case Tango bears seal NERO and lists VELA -> Coda and NERO -> Mira.

Value ledger: Luma = 42, Vega = 7, Coda = 53, Mira = 47.

Arithmetic Q: What is Ava's current blue-inventory value? -> 49
Selector Q:   What is Ava's current red-inventory value?  -> 47

Answer with one Arabic numeral only.
```

필수 의미 구조:

- 두 condition 모두 target label에서 case를 찾고 두 value key를 처리해야 함
- arithmetic correct와 arithmetic alternative는 context의 독립 numeric mention으로 등장하지 않음
- selector correct와 selector alternative는 value ledger에 각각 정확히 한 번 등장함
- selector answer는 seal/tag matching 없이는 두 ledger value 중 결정할 수 없음
- pair의 context와 output instruction은 동일하고 question target만 변경

## 5. Within-condition exposure-matched scoring

각 condition은 자기 condition 안에서 exposure가 맞는 prespecified alternative를 사용한다.

```text
M_A(x) = logP(y_arithmetic_correct | x)
         - logP(y_arithmetic_alt | x)

M_S(x) = logP(y_selector_correct | x)
         - logP(y_selector_alt | x)
```

- arithmetic correct/alt: 둘 다 context에 독립 numeric mention으로 없음
- selector correct/alt: 둘 다 value ledger에 정확히 한 번 있음
- 두 alternative는 dataset 생성 전에 고정하며 결과를 보고 변경하지 않음

> Correct/alternative exposure is matched within each condition, but the absolute exposure regime still differs across conditions; therefore `D_k` is interpreted as a calibrated difficulty contrast, not as a perfectly exposure-invariant effect.

즉 이 scoring은 condition 내부의 correct-vs-alternative 비교를 공정하게 만들지만, arithmetic의 `둘 다 비노출`과 selector의 `둘 다 노출` 차이는 operationalized feature에 남는다. Calibration PASS만으로 순수 terminal-rule effect 또는 exposure-invariant effect를 주장하지 않는다.

Template-family cluster `k`의 difficulty contrast:

```text
D_k = mean_f [ M_A(x_arithmetic,f) - M_S(x_selector,f) ]
```

다른 condition의 canonical answer를 alternative로 사용하는 v2/v2.1 scoring은 폐기한다.

Primary metric 후보:

- within-condition exposure-matched candidate log-odds의 template contrast `D_k`

Secondary metrics 후보:

- condition별 candidate forced-choice accuracy
- one-token unrestricted generation accuracy
- raw canonical logP와 per-token geometric probability
- arithmetic no-change/operand diagnostic foil log-odds
- selector inactive-entry foil log-odds
- condition별 margin distribution과 template-cluster overlap

## 6. Numeric candidate-set counterbalance

각 template은 겹치지 않는 두 numeric candidate pair `P={p1,p2}`, `Q={q1,q2}`를 사용한다.

- 한 frame에서 P는 arithmetic correct/alt, Q는 selector correct/alt로 사용
- counterbalanced frame에서는 Q를 arithmetic에, P를 selector에 사용
- ADD/SUB orientation에 따라 arithmetic pair의 high/low value가 correct/alternative 역할을 교환
- selector active-entry orientation에 따라 selector pair의 두 value가 correct/alternative 역할을 교환
- 모든 값은 arithmetic/selector, correct/alternative 역할에 동일 횟수 등장

Arithmetic primary alternative는 임의의 `correct +/- offset`으로 생성하지 않는다. 다음 structured error로 고정한다.

```text
increase / warming: correct = start + delta
                    primary alt = start - delta

decrease / cooling: correct = start - delta
                    primary alt = start + delta
```

즉 **wrong-operation result를 arithmetic primary foil**로 사용한다. 이는 실제 산술 과정에서 변화 방향을 반대로 적용한 plausible error이며, selector condition의 inactive-entry value처럼 의미 있는 경쟁 후보 역할을 한다. `start`, `delta`, 임의 인접 숫자 또는 결과 확인 뒤 고른 숫자는 primary foil로 사용할 수 없다.

Operation 방향 때문에 arithmetic correct와 foil의 상대적 크기가 결정되는 것은 해당 arithmetic rule의 구조적 속성이다. 숫자 자체의 prior가 condition effect에 고정되지 않도록 numeric pair assignment, operation direction과 selector active-entry orientation의 균형을 automatic audit에서 별도로 확인한다.

각 numeric pair는 같은 parity의 두 값으로 만들고 midpoint를 `start`, half-distance를 `delta`로 사용한다.

```text
low  = start - delta
high = start + delta

ADD: correct=high, primary alt=low
SUB: correct=low,  primary alt=high
```

따라서 동일 candidate pair가 ADD와 SUB frame에서 correct/alternative 역할을 정확히 교환할 수 있다.

초기 numeric 후보값:

| 항목 | v3 제안값 |
|---|---:|
| canonical/alternative 범위 | 30–69 |
| arithmetic delta | 2–9 |
| candidate-pair 거리 | `2*delta`, 즉 4–18의 짝수 |
| pair parity | 두 값이 동일 parity; midpoint가 정수 |
| answer digit 길이 | 2자리 |
| answer continuation | locked tokenizer에서 정확히 1 token |

모든 correct, primary alternative, operand와 다른 ledger value는 collision이 없어야 한다. Wrong-operation result가 음수, 범위 밖, operand 또는 ledger value와 충돌하면 해당 numeric block을 사용하지 않고 deterministic generator가 새 block을 생성한다.

## 7. Counterbalance 구조

v3에는 다음 다섯 이진 nuisance factor가 있다.

| Column | Factor | Level 0 | Level 1 |
|---|---|---|---|
| A | numeric candidate-pair assignment | P arithmetic / Q selector | Q arithmetic / P selector |
| B | arithmetic operation | ADD / warming | SUB / cooling |
| C | selector active entry | entry 1 | entry 2 |
| D | label/case role orientation | label 1 arithmetic / label 2 selector | label 2 arithmetic / label 1 selector |
| E | channel-block order | arithmetic block first | selector block first |

ADD/SUB와 selector entry 1/2는 서로 독립된 column으로 둔다. 완전교차 32 frame 대신 다음 **8-run strength-2 orthogonal counterbalance design**, 즉 `OA(8,5,2,2)`를 사용한다.

| Frame | A | B | C | D=`A xor B` | E=`A xor C` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 0 | 1 |
| 3 | 0 | 1 | 0 | 1 | 0 |
| 4 | 0 | 1 | 1 | 1 | 1 |
| 5 | 1 | 0 | 0 | 1 | 1 |
| 6 | 1 | 0 | 1 | 1 | 0 |
| 7 | 1 | 1 | 0 | 0 | 1 |
| 8 | 1 | 1 | 1 | 0 | 0 |

이 matrix는 다음만 보장한다.

```text
각 factor level: 4:4
임의의 두 factor: 2x2 네 조합이 각각 정확히 2회
```

5개 factor의 모든 interaction을 8 run에서 분리할 수는 없다. Fractional-factorial 관점에서는 main effect와 일부 two-factor interaction이 alias될 수 있는 Resolution III 수준이며, individual interaction estimate에 사용하지 않는다. 목적은 숫자·연산 방향·selector 위치·label 역할·문장 순서 편향을 pairwise 수준에서 상쇄하는 것이다.

> **The orthogonal array is used only for counterbalancing, not for estimating main effects or factor interactions.**

Builder는 위 8행의 순서와 level mapping을 그대로 구현하고, manifest에는 matrix 자체와 SHA-256을 기록해야 한다. 임의의 다른 pairwise-balanced matrix로 교체할 수 없다.

## 8. Dataset과 독립 단위 제안

| 항목 | v3 제안값 |
|---|---:|
| lexical family | object-count / points-balance / temperature |
| family당 template family | 8 |
| template당 frame | 8 |
| 총 pair | 192 |
| 총 prompt | 384 |
| 독립 cluster | 24 |
| random seed | freeze 시 새 seed 지정 |

세 family는 연산 방향이 아니라 의미 영역으로 정의한다. 각 family의 모든 template에서 arithmetic-operation factor B를 통해 ADD/SUB를 4:4로 배치한다. Temperature도 warming/cooling을 모두 포함한다. 이로써 arithmetic correct가 항상 큰 값 또는 작은 값에 묶이지 않는다.

Lexical family별 결과를 별도로 판정하며 operation direction도 diagnostic stratum으로 보고한다. 앞선 exploratory의 `decrease/increase/temperature` 분류와 v3 semantic family를 같은 interaction으로 간주하지 않고, exploratory 결과를 근거로 family를 합치거나 사후 선택하지 않는다.

## 9. 난이도 gate 제안

v2.1 threshold는 자동 승계하지 않는다. 아래 값은 **freeze 전 검토가 필요한 신규 제안값**이다.

64 prompt/condition/family 기준:

| 지표 | 제안 gate |
|---|---:|
| candidate forced-choice | 각 condition 40–56 / 64 |
| one-token generation | 각 condition 16–56 / 64 |
| forced-choice correct-count gap | 최대 6 / 64 |
| generation correct-count gap | 최대 6 / 64 |
| `|mean_k(D_k)|` | 최대 0.40 nat |
| `|d_z|` | 최대 0.30 |

모든 hard 기준을 동시에 만족해야 family PASS다. 일부 template를 결과 확인 뒤 제외하는 규칙은 사용하지 않는다.

Threshold 근거는 다음과 같다.

- forced-choice 최소 40/64는 chance 0.5에 대해 one-sided exact binomial tail이 약 0.030으로, 단순 우연보다 높은 식별 능력을 요구한다.
- 최대 56/64는 최소 8개 오류와 12.5% headroom을 요구하여 60/64보다 ceiling에 더 엄격하다.
- one-token generation 최소 16/64는 floor를 배제하기 위한 25% 경계이고, 최대 56/64는 forced-choice와 같은 ceiling headroom을 적용한다.
- condition count gap 6/64는 10 percentage point 미만의 사전 equivalence margin이다.
- `0.40 nat`은 correct-vs-alternative odds의 condition 차이를 약 `exp(0.40)=1.49` 이내로 제한한다.
- `|d_z| <= 0.30`은 template-cluster 기준 standardized imbalance를 small-effect 범위로 제한한다.

이 경계는 v3 model output을 보기 전에 정하며 v2.1.1 결과에 맞춰 family를 통과시키기 위한 조정값이 아니다. Freeze 후 결과를 보고 완화하거나 일부 criterion을 삭제하지 않는다.

같은 부호의 `D_k` 수는 `mean(D_k)`와 `d_z`에 일부 중복되므로 hard PASS criterion에서 제외하고 descriptive diagnostic으로 보고한다. 추가 positivity/overlap 진단으로 condition별 prompt margin의 10–90 percentile 범위, template별 margin 분포와 같은 부호의 `D_k` 수를 함께 보고한다. 이 diagnostic을 보고 threshold나 family 판정을 바꾸지 않는다.

## 10. Automatic audit

모든 pair에 대해 다음을 검사한다.

1. v2/v2.1/v2.1.1과 template skeleton exact/near-duplicate가 아님
2. arithmetic correct와 primary alternative가 context numeric mention에 없음
3. selector correct와 primary alternative가 ledger에 각각 정확히 한 번 있음
4. selector tag와 두 entry가 명확하며 tag matching 없이는 정답을 결정할 수 없음
5. 두 channel 모두 label -> case -> value-key -> ledger 경로를 요구함
6. correct/alternative/operand/foil/ledger collision 없음
7. pair context byte-identical
8. question은 target label 외 동일 skeleton
9. prompt token count exact match
10. 모든 answer candidate가 정확히 one-token continuation
11. 다섯 counterbalance factor 각각 4:4
12. 모든 factor pair의 2x2 cell 각각 2회
13. numeric pair의 arithmetic/selector 역할, ADD/SUB, selector active-entry 위치와 correct/alternative 역할 균형
14. family당 8개 독립 template와 총 192 pair 확인

한 항목이라도 실패하면 audit 단계로 넘기지 않는다.

## 11. AI 및 independent human audit

Automatic audit 이후 192/192 AI linguistic pre-audit을 수행한다. 검수 항목은 다음을 포함한다.

- arithmetic rule과 canonical answer 정확성
- selector seal/tag와 active entry 정답 정확성
- 두 binding 경로의 명확성과 의미적 대칭성
- arithmetic answer-copy shortcut 부재
- selector가 단일 숫자 표면 복사가 아닌지
- target 외 질문 의미 불변
- 문법과 자연스러움
- counterbalance counterpart 동등성
- one-numeral output instruction 명확성
- 기존 dataset과 의미적 non-reuse

Official Calibration 전에는 독립 human reviewer 2명이 각각 192/192 pair를 blind review한다. 여기서 blind는 reviewer가 다음을 보지 않는다는 뜻이다.

- model score, forced-choice/generation output 또는 calibration 결과
- exploratory LiReF/component 결과
- `L31N13336` family-specific sign-reversal 가설
- 다른 reviewer의 판정과 comments
- candidate intervention 또는 Pilot 결과

Reviewer는 dataset pair, frozen linguistic-audit guide와 자신의 reviewer ID만 받는다. 불일치만 제3 reviewer가 동일한 blind 조건에서 adjudication한다. Waiver는 허용하지 않는다. 최종 FAIL 또는 NEEDS_REVISION이 하나라도 있으면 새 dataset version을 생성하고 automatic/AI/human audit을 처음부터 다시 수행한다.

## 12. Calibration과 LiReF dataset의 역할 분리

- v3 controlled dataset: terminal feature와 baseline difficulty를 통제하기 위한 Calibration/Pilot source
- 기존 LiReF dataset: R/M direction과 기존 candidate의 역사적 근거
- 사용하지 않은 held-out LiReF items: official Pilot 이후 external-validity 확인용

LiReF direction 학습 또는 candidate discovery에 사용한 동일 item을 Stage E 검증에 재사용하지 않는다. Held-out LiReF 검증을 수행할 경우 split provenance와 non-overlap audit 또는 cross-fitting을 먼저 동결한다.

## 13. 안전 및 실행 gate

이 문서는 draft이며 다음은 허용하지 않는다.

```text
design_frozen: false
dataset_generation_allowed: false
model_loading_allowed: false
baseline_calibration_execution_allowed: false
load_liref_direction: false
capture_hidden_states: false
inspect_candidate_states: false
forward_hooks: false
patching_or_suppression: false
stage_e_pilot_allowed: false
confirmatory_claim_allowed: false
```

## 14. 승인된 freeze 항목

1. [x] operationalized feature 명칭
2. [x] selector seal/tag matching 구조와 arithmetic channel의 공통 multi-hop 경로
3. [x] within-condition exposure-matched scoring과 condition 간 absolute-exposure limitation
4. [x] exact `OA(8,5,2,2)` strength-2 counterbalance matrix
5. [x] numeric range와 wrong-operation primary foil 규칙
6. [x] 192 pair / 384 prompt 규모
7. [x] 신규 acceptance threshold와 수치 근거
8. [x] same-sign 및 positivity/overlap을 descriptive diagnostic으로만 사용하는 정책
9. [x] independent reviewer 2명과 blind 절차; waiver 금지
10. [x] held-out LiReF external-validation 역할과 non-overlap 정책

이 10개 항목은 2026-08-30 사용자의 명시적 승인으로 동결했다. 이후 model 결과를 보고 threshold, matrix, foil, template/family 포함 규칙 또는 audit gate를 변경하지 않는다. 변경이 필요하면 v3 artifact를 보존하고 새 design version을 만든다.

승인 후 순서:

```text
v3 design freeze
-> deterministic builder
-> new 192-pair dataset
-> automatic audit
-> AI audit
-> independent human audit
-> baseline-only implementation/static review
-> explicit official execution authorization
-> official Baseline Calibration
```

## 15. Design freeze 기록

- freeze date: `2026-08-30`
- approved pre-freeze source SHA-256: `34d23bed83584963d0a8fd88881b1d666bb4aefd759a91453e7d4e9686081137`
- machine-readable manifest: `calibration_v3_design_frozen.json`
- frozen manifest SHA-256: `c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc`
- freeze record: `calibration_v3_design_freeze_record.json`
- freeze record SHA-256: `e0686da0facf8092dca5f914e451874429e067bb87b8f9d1e67d9bb4f048b7e1`
- manifest status: `design_frozen_execution_not_approved`
- dataset seed / bootstrap seed: `20260831`
- exact counterbalance: `OA(8,5,2,2)` matrix frozen in manifest
- human audit: independent blind reviewer 2명 × 각 192/192; waiver 불가
- dataset generation: 허용
- model loading / forward / GPU calibration: 미승인
- Stage E Pilot / intervention: 미승인
- next allowed work: deterministic v3 dataset builder implementation

Frozen manifest, threshold 또는 matrix를 수정해야 하면 기존 파일을 덮어쓰지 않는다. 변경 사유와 새 design version, 새 source/manifest hash를 생성한다.
