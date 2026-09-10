# Stage E Baseline Calibration v2.1 설계 초안

상태: **FROZEN — 설계 동결 완료, dataset 생성·실행 미승인**  
작성일: 2026-08-30  
목적: v2에서 관찰된 direct-retrieval ceiling과 generation 출력 형식 confound를 제거한 새 calibration으로, Stage E Pilot에 사용할 수 있는 통제 feature operationalization을 확보한다.

## 1. v2 결과와 artifact 보존

v2 design, dataset, implementation과 결과는 수정하거나 덮어쓰지 않는다.

- v2 frozen design SHA-256: `a8f3dad7fced945377194074f9aa12d673faff3b55c3ec45bd82e397b5a5302b`
- v2 dataset SHA-256: `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87`
- v2 pair results SHA-256: `4cdf5d57295d483ddb2ad3bfe1683442c1c72759c5d447f7a1d18999283c3668`
- v2 summary SHA-256: `cec0432036344dccfaebc60ccc82f5d9aa463171eeae8d5c0dc7d6a1d9b295ce`
- v2 판정: decrease / increase / temperature 모두 `FAIL`
- v2 human audit: `not_performed`, gate `waived_by_user`
- v2 문항·context·template·fact의 v2.1/Pilot/Confirmatory 재사용: 금지

고정된 실패 원인은 다음과 같다.

1. decrease와 increase에서 Irrelevant forced-choice accuracy가 1.00으로 ceiling에 도달했다.
2. decrease와 increase에서 Relevant의 paired-answer margin이 Irrelevant보다 일관되게 작았다.
3. temperature는 양 조건의 forced-choice가 ceiling에 걸렸다.
4. v2 generation은 숫자만 정답으로 허용하면서 최대 8 token을 생성해, `18 blue marbles`처럼 값은 맞지만 단위가 붙은 출력을 오답 처리했다.
5. 이 관찰을 이유로 v2 item을 삭제하거나 v2 threshold·normalization을 변경하지 않는다.

## 2. v2.1 연구 질문

v2.1의 operationalized feature는 다음으로 고정한다.

> **`arithmetic value transformation vs matched one-hop keyed fact retrieval`**

- **Relevant:** target label에 연결된 record를 찾은 뒤, 그 record의 start value와 delta에 한 번의 증가·감소 연산을 적용해야 한다.
- **Irrelevant:** target label에 연결된 record를 찾은 뒤, 그 record에 직접 기재된 current value를 조회해야 한다.

두 조건 모두 다음 공통 단계를 요구한다.

1. 질문의 target label 식별
2. target label과 record key의 binding 조회
3. 해당 record 선택

마지막 단계만 다르다.

- Relevant: operands에서 값을 산출
- Irrelevant: 기재된 값을 조회

따라서 v2의 단순 direct-fact copy보다 retrieval 조건을 어렵게 만들되, 결과를 순수 산술 효과나 일반적인 Reasoning/Memorization 전체의 차이라고 부르지 않는다.

## 3. 핵심 문항 구조

예시 구조이며 실제 문장·수치·key는 새 dataset에서 별도로 생성한다.

```text
Ava's red marbles are filed under record Kappa.
Record Kappa currently lists 47 marbles.
Ava's blue marbles are filed under record Tango.
Record Tango listed 38 marbles before 5 more were added.

Relevant Q:   What is Ava's current blue-marble count? -> 43
Irrelevant Q: What is Ava's current red-marble count?  -> 47
```

공통 prompt 계약:

```text
Q: {question}
Answer with one Arabic numeral only.
A: 
```

필수 구조:

```text
Relevant answer != Irrelevant answer
Relevant answer는 context의 독립 numeric mention으로 등장하지 않음
Irrelevant answer는 선택된 direct record에 정확히 한 번 등장
Relevant answer != start != delta
direct와 transformed channel 모두 label -> record key binding을 요구
Relevant/Irrelevant question은 target label 외 동일
```

## 4. 난이도 matching 규칙

v2.1은 direct condition을 단순 표면 복사로 구성하지 않는다.

- 두 channel 모두 target label과 arbitrary record key를 연결하는 문장을 가진다.
- 두 channel 모두 별도의 record 문장을 가진다.
- 두 channel의 label-to-record referential hop 수는 같다.
- direct record와 transformed record의 위치, label 역할과 key 역할을 counterbalance한다.
- 질문에는 record key가 직접 주어지지 않는다. 모델이 target label에서 올바른 record를 선택해야 한다.
- 다른 entity/record의 값은 양 조건에 동일하게 노출되므로 distractor 수가 같다.
- transformed channel에만 start/delta가 존재하는 것은 의도된 feature 구성 요소다.

이 설계는 두 조건의 모든 계산량이 같다고 주장하지 않는다. 목적은 공통 binding/retrieval 부담을 맞춘 상태에서 **마지막 arithmetic transformation 필요성**을 조작하는 것이다.

## 5. Pair invariant

pair 안에서 다음을 고정한다.

- context 전체
- entity set
- label-to-record mapping
- record key set
- direct/transformed fact와 그 순서
- operation과 operands
- 질문 문형과 출력 지시
- target label을 제외한 question token sequence
- answer digit length와 tokenizer token length
- prompt token count
- source record spans

조건에 따라 달라지는 것은 question target label, canonical answer와 answer literal exposure뿐이다.

## 6. Numeric design 승인값

| 항목 | v2.1 승인값 | 상태 |
|---|---:|---|
| canonical answer 범위 | 30–59 | 승인 |
| delta 범위 | 2–9 | 승인 |
| 최소 `|A-B|` | 3 | 승인 |
| answer digit length | 2자리로 동일 | 승인 |
| answer continuation token 수 | Meta-Llama-3 tokenizer에서 정확히 1 token | 필수 |
| template당 numeric block | 1개 | 승인 |
| numeric block seed | `20260830` | 승인 |

추가 collision 금지 규칙:

- `A != B`
- A/B 어느 것도 start 또는 delta와 같지 않음
- Relevant answer가 context의 다른 record·key 숫자와도 같지 않음
- start, delta, correct, alternative와 wrong-operation foil이 모두 다름
- record key에는 숫자를 포함하지 않은 alphabetic key만 사용함
- 한 frame 안의 두 record key는 tokenizer token 수가 같아야 함
- v2에서 사용한 numeric block을 그대로 복제하지 않음

## 7. Counterbalance 승인값

한 template family 안에서 다음을 완전 교차한다.

```text
answer orientation (2)
× label/record role orientation (2)
× record-block order (2)
= 8 frames/template family
```

### 7.1 Answer orientation

- `A_to_relevant`: Relevant=A, Irrelevant=B
- `B_to_relevant`: Relevant=B, Irrelevant=A

### 7.2 Label/record role orientation

- label 1/key 1 direct, label 2/key 2 transformed
- label 2/key 2 direct, label 1/key 1 transformed

각 label과 key가 direct/transformed 역할에 동일 횟수 등장해야 한다.

### 7.3 Record-block order

- direct mapping+record block first
- transformed mapping+record block first

mapping 문장과 해당 record 문장은 하나의 block으로 이동한다. block 안 문장 순서는 template family별로 고정하고 template 간 counterbalance한다.

## 8. Dataset 규모 승인값

| 항목 | v2.1 승인값 | 상태 |
|---|---:|---|
| lexical family | decrease / increase / temperature | 승인 |
| family당 새 template family | 8 | 승인 |
| template당 frame | 8 | 승인 |
| frame당 instantiation | 1 | 승인 |
| 총 paired item | 192 | 계산값 |
| 총 prompt | 384 | 계산값 |
| 독립 template-family cluster | 24 | 계산값 |
| dataset seed | `20260830` | 승인 |

계산은 `3 families × 8 templates × 8 frames = 192 pairs`다. v2의 6 cluster/family보다 8 cluster/family를 사용해 template-level 방향 일관성과 standardized gap이 소수 template 하나에 과도하게 좌우되는 문제를 완화한다.

모든 surface template, entity combination, record key, context와 numeric block은 v2와 새로 만들어야 한다.

## 9. Teacher-forced scoring 승인값

v2의 counterbalanced A/B scoring 원칙은 v2.1 구조에 맞춰 별도로 재승인한다. v2 결과에 맞춰 재계산하거나 threshold를 조정하는 용도로 승계하지 않는다.

각 prompt에서:

```text
M(x) = logP_seq(y_correct | x) - logP_seq(y_alt | x)
```

answer orientation을 짝지은 cell contrast:

```text
C_k,g = 1/2 * [
  (M_R - M_I)_A_to_relevant
  +
  (M_R - M_I)_B_to_relevant
]
```

template contrast:

```text
D_k = mean_g(C_k,g)
```

여기서 `g`는 label/record role × record-block order의 4개 cell이다. 이 식은 8 frame 평균과 같지만, answer orientation cancellation이 실제로 수행됐는지 별도 audit할 수 있게 한다.

Primary metric:

- counterbalanced paired-answer log-odds의 template contrast `D_k`

Secondary metrics:

- A/B forced-choice accuracy
- one-token unrestricted generation accuracy
- raw canonical sequence logP
- per-token geometric probability
- arithmetic foil 대비 log odds
- answer-orientation cancellation residual

EOS/newline은 answer score에 포함하지 않는다. Prompt와 answer를 joint tokenization한 뒤 answer suffix token만 score한다.

## 10. Generation/output contract 승인값

v2의 8-token free-form generation 규칙은 승계하지 않는다.

```text
do_sample: false
num_beams: 1
max_new_tokens: 1
accepted answer: canonical Arabic numeral 하나
```

이 규칙은 모든 canonical answer가 leading-space continuation 기준 정확히 한 tokenizer token이라는 full-dataset audit를 통과해야만 사용할 수 있다.

정규화 순서:

1. 생성 suffix 한 token만 decode
2. Unicode NFKC
3. outer whitespace 제거
4. 전체 문자열이 정규식 `[0-9]+`와 일치하는지 확인
5. canonical answer와 exact match

단위, 설명, 식, punctuation, number word와 alias는 허용하지 않는다. `max_new_tokens=1`로 정답 뒤 설명이 accuracy를 훼손하는 v2 confound를 구조적으로 차단한다. 이 metric은 **one-token unrestricted generation accuracy**로 명명하고 일반적인 장문 generation 성능으로 해석하지 않는다.

## 11. Acceptance criteria 승인값

아래 값은 v2에서 자동 승계한 값이 아니라, v2.1의 64 prompts/condition/family와 one-token output contract를 기준으로 새로 승인한 값이다.

### 11.1 절대 난이도

| 지표 | Relevant | Irrelevant | 상태 |
|---|---:|---:|---|
| A/B forced-choice accuracy | 0.60–0.95 | 0.60–0.95 | 승인 |
| one-token unrestricted generation accuracy | 0.25–0.95 | 0.25–0.95 | 승인 |

64개 prompt/condition/family에서 경계는 정수 count로 구현한다.

- forced-choice: 최소 39/64, 최대 60/64
- one-token generation: 최소 16/64, 최대 60/64
- 소수 비율은 보고용이며 PASS/FAIL은 위 count로 판정

### 11.2 조건 간 균형

| 지표 | 권고 경계 | 상태 |
|---|---:|---|
| forced-choice accuracy gap | `<= 0.10` | 승인 |
| one-token generation accuracy gap | `<= 0.10` | 승인 |
| `|mean_k(D_k)|` | `<= 0.50 nat` | 승인 |
| cluster standardized gap `|d_z|` | `<= 0.35` | 승인 |
| 같은 부호의 `D_k` | 최대 5/8 | 승인 |

64개 prompt에서는 accuracy gap을 부동소수점 반올림이 아니라 count difference `<= 6/64`로 구현한다. 이는 실제 gap `<= 0.09375`에 해당한다.

모든 기준을 lexical family별로 동시에 만족해야 해당 family가 PASS다. 결과를 보고 template를 제외하는 부분 통과 규칙은 사용하지 않는다. Raw logP와 bootstrap CI는 diagnostic이며 PASS/FAIL threshold를 바꾸는 근거로 사용하지 않는다.

`0.50 nat`, `0.35`는 v2 실패값에 맞춘 통과선이 아니라 v2.1용 허용 오차로 새로 승인된 값이다.

## 12. Cluster 통계 승인값

- 독립 단위: template family
- lexical family별 cluster 수: 8
- `d_z = mean(D_k) / sample_sd(D_k)`, `ddof=1`
- 모든 `D_k=0`: `d_z=0`
- float64 machine epsilon 이하 SD와 nonzero mean: FAIL
- NaN / infinity / missing: FAIL
- 같은 방향 수: `max(count(D_k>0), count(D_k<0))`; zero 제외
- item-level 독립표본 p-value: 계산 금지
- template cluster bootstrap: lexical family별 replacement resampling 10,000회
- bootstrap seed: `20260830`
- percentile 95% CI는 descriptive only

## 13. Automatic shortcut·structure audit

모든 항목을 통과해야 AI pre-audit로 넘긴다.

1. v2 prompt/context/template/fact와 exact 또는 near-duplicate가 아님
2. Relevant answer가 어떤 독립 numeric mention으로도 context에 없음
3. Irrelevant answer가 선택된 direct record에 정확히 한 번만 있음
4. Relevant/Irrelevant answer, start, delta와 foil collision 없음
5. 두 channel 모두 label-to-record binding이 필요함
6. pair context가 byte-identical임
7. question은 target label 외 byte-identical skeleton임
8. prompt token count exact match
9. canonical/alternative answer continuation이 각각 정확히 1 token임
10. A/B relevance 배치 4:4
11. label/key 역할 4:4
12. record-block order 4:4
13. answer magnitude 방향 4:4
14. 각 template에 8개 factorial cell이 중복 없이 존재
15. family당 독립 template 8개와 전체 192 pairs 확인

near-duplicate non-reuse 검사는 **v2 template family와 v2.1 template family 사이**에만 다음 규칙으로 적용한다.

1. Unicode NFKC와 lowercase를 적용한다.
2. 사람 이름, color label, record key와 모든 numeric mention을 각각 `<NAME>`, `<COLOR>`, `<KEY>`, `<NUM>` placeholder로 치환한다.
3. punctuation과 반복 whitespace를 정규화한 token sequence로 template skeleton을 만든다.
4. normalized template skeleton SHA-256이 v2 skeleton과 exact match이면 무조건 FAIL이다.
5. normalized word 5-gram set의 Jaccard similarity가 `>= 0.80`인 v2 template이 하나라도 있으면 near-duplicate FAIL이다.
6. 5-gram set이 비어 있는 짧은 skeleton은 exact normalized token-sequence comparison을 사용한다.

v2.1의 동일 template family 안에서 생성된 8개 counterbalance frame은 이 cross-version near-duplicate 판정 대상에서 제외한다. 다만 서로 다른 v2.1 template family끼리 skeleton hash가 같으면 독립 template가 아니므로 FAIL이다.

## 14. AI 및 human audit

AI pre-audit은 192 pair 전부에 대해 다음을 확인한다.

- 산술 정답
- direct record 정답
- binding 경로의 명확성
- Relevant answer-copy shortcut 부재
- 질문 target 외 의미 동일성
- 문법과 자연스러움
- counterbalance counterpart 동등성
- 출력 지시의 명확성

v2의 human-audit waiver는 v2.1에 승계하지 않는다. v2.1은 다음 human-audit gate를 사용한다.

- 독립 primary reviewer 2명이 각각 192/192 pair를 전수 검수한다.
- 두 reviewer는 AI pre-audit, model prediction, logP, generation, LiReF, candidate/intervention 결과와 서로의 판정을 보지 않는다.
- 두 reviewer가 모두 PASS한 pair만 최종 PASS다.
- 불일치는 제3 independent reviewer가 blind adjudication하고 사유를 기록한다.
- 최종 `FAIL` 또는 `NEEDS_REVISION`이 하나라도 있으면 dataset 실행을 금지한다.
- 수정이 필요하면 frozen dataset을 덮어쓰지 않고 새 dataset version을 생성한 뒤 automatic audit, AI pre-audit와 human audit을 처음부터 다시 수행한다.

## 15. 안전 및 non-reuse

v2.1 Baseline Calibration도 별도 구현·정적 검토·실행 승인 전에는 모델을 로드하지 않는다.

```text
baseline_behavior_only: true
capture_hidden_states: false
inspect_liref_scores: false
inspect_candidate_states: false
enable_forward_hooks: false
enable_activation_intervention: false
enable_weight_intervention: false
allow_loading_discovery_liref_direction: false
confirmatory_claim_allowed: false
reuse_in_pilot: false
reuse_in_confirmatory: false
allow_post_result_item_deletion: false
allow_post_result_threshold_change: false
model_training: false
```

Calibration item은 Pilot/Confirmatory에 재사용하지 않는다. 하나 이상의 family가 PASS하면 그 family의 **생성 규칙만** 사용해 새로운 Pilot dataset을 만든다.

## 16. v2.1을 통해 얻을 수 있는 것

Calibration PASS는 다음만 의미한다.

- arithmetic transformation과 matched keyed retrieval 조건이 모두 측정 가능한 행동 범위에 있음
- answer value, label/key 역할과 record 위치를 상쇄한 뒤 과도한 baseline 난이도 불균형이 없음
- Stage E Pilot에서 LiReF R/M projection 차이를 난이도·출력 형식 차이만으로 설명할 위험이 감소함

Calibration PASS 자체로는 R/M feature, causal component 또는 pathway를 입증하지 않는다. 이후 Pilot에서만 다음 연결을 처음 검사한다.

```text
controlled input feature
-> L31N13336 / L29H00030 / L30H00006 / L29H00031 response
-> LiReF R/M projection shift
```

그 뒤 별도 Confirmatory intervention에서 patch/suppression 효과가 재현되어야 causal chain 주장에 접근할 수 있다.

## 17. Freeze 전 승인 gate

모든 설계값과 human audit 정책이 승인됐다. 아래 항목을 machine-readable manifest에 그대로 옮기고 hash를 생성한 뒤에만 freeze가 완료된다.

- [x] operationalized feature 명칭과 keyed-retrieval 구조
- [x] 공통 prompt와 one-token output contract
- [x] numeric range / delta / collision 규칙
- [x] 3 families × 8 templates × 8 frames 규모
- [x] A/B × label/key role × record order counterbalance
- [x] primary/secondary scoring
- [x] acceptance thresholds 전부
- [x] cluster 통계와 bootstrap seed
- [x] near-duplicate audit 경계
- [x] v2.1 independent reviewer 2명 × 192/192 human audit
- [x] v2/v2.1/Pilot/Confirmatory non-reuse

승인 후 진행 순서:

1. machine-readable v2.1 frozen manifest와 SHA-256 생성
2. deterministic v2.1 dataset builder 작성
3. 새 192 pairs 생성
4. shortcut·near-duplicate·counterbalance·tokenizer audit
5. 192/192 AI pre-audit
6. 승인된 human-audit 정책 적용
7. baseline-only implementation 정적 safety review
8. 별도 execution authorization
9. Baseline Calibration v2.1 실행

Design freeze가 완료된 뒤에도 dataset/audit/implementation/별도 execution authorization gate 전에는 model forward와 GPU calibration을 금지한다. LiReF 분석과 Stage E Pilot은 v2.1 Calibration PASS 전까지 금지한다.

## 18. Design freeze 기록

- freeze date: `2026-08-30`
- 승인된 pre-freeze source document SHA-256: `793167b67d91f202fc49ffd3b8ded4f78c307e65205c9a0810f3d2abf32aae38`
- machine-readable manifest: `calibration_v2_1_design_frozen.json`
- frozen manifest SHA-256: `d0d6d19432eb48234a9729cc5f297c09963eb485bf5ec02178d504a93dc8307f`
- freeze record: `calibration_v2_1_design_freeze_record.json`
- manifest status: `design_frozen_execution_not_approved`
- human audit: 독립 reviewer 2명 × 각 192/192 pair, waiver 불가
- `execution_allowed`: `false`
- `stage_e_pilot_allowed`: `false`
- deterministic dataset builder: 생성 전
- calibration dataset: 생성 전

이후 설계 규칙, threshold 또는 human-audit 정책을 변경해야 하면 frozen manifest를 덮어쓰지 않는다. 변경 사유를 기록하고 새로운 design version과 SHA-256을 생성한다.
