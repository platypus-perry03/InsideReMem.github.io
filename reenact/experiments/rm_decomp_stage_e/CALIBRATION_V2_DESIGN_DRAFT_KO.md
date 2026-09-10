# Stage E Baseline Calibration v2 설계 초안

상태: **FROZEN — 설계 동결 완료, dataset 생성·실행 미승인**  
작성일: 2026-08-29  
목적: v1의 cross-fact answer-copy shortcut을 제거한 뒤, `transformation-dependent answer derivation vs direct-fact retrieval`을 Stage E primary feature 후보로 사용할 수 있는지 baseline 행동만으로 선별한다.

## 1. v1 보존과 폐기 범위

v1 artifact는 수정하거나 덮어쓰지 않는다.

- frozen design SHA-256: `ba9939e30b1e68c7b061e407f981c37bc2b1878055ef9074ede86f2afe5aacb0`
- v1 dataset SHA-256: `ba4c0fe2b17633c082ca666c4af7133b90a7e221efb6824bac74c99eddf2505b`
- blocking reason: 모든 Relevant 정답이 동일 context의 직접 사실에 그대로 노출되어 transformation 없이 복사 가능
- v1 calibration 실행: 금지
- v1 문항의 Pilot/Confirmatory 재사용: 금지

v1의 tokenizer·문법 검사는 통과했지만 primary feature operationalization은 실패한 것으로 기록한다.

## 2. v2 연구 질문

동일 context에서 다음 두 질문의 baseline 난이도를 비교한다. Operationalized feature의 공식 명칭은 **`transformation-dependent answer derivation vs direct-fact retrieval`**로 고정한다.

- **Relevant:** 정답을 얻으려면 한 번의 증가·감소 relation transformation을 적용해야 한다.
- **Irrelevant:** 정답은 별도의 직접 사실에서 조회할 수 있으며 transformation은 질문과 무관하다.

가설 수준에서는 single-step relation transformation의 task relevance가 내부 pathway를 변화시키는지 묻는다. 그러나 실제 v2 조작에는 정답의 직접 노출 여부도 포함되므로 이를 순수한 산술 transformation 조작이라고 부르지 않는다.

v2 calibration은 이 조작이 LiReF score를 움직이는지 검증하지 않는다. 두 조건이 모델이 풀 수 있는 범위에 있고, 특정 숫자·label·문장 위치 때문에 난이도 차이가 생기지 않는지를 검사한다.

## 3. 핵심 구조

pair 안에서 context는 동일하지만 두 조건의 정답은 다르다.

```text
Ava has 6 red marbles.
Ava had 8 blue marbles and gave away 3 blue marbles.

Relevant:   How many blue marbles does Ava have now?  -> 5
Irrelevant: How many red marbles does Ava have now?   -> 6
```

필수 조건:

```text
Relevant answer != Irrelevant answer
Relevant answer != transformation start value
Relevant answer != transformation delta
Relevant answer literal은 context에 등장하지 않음
Irrelevant answer literal은 직접 사실에 한 번 등장함
```

`answer literal exposure`가 조건 간 동일하다는 v1 규칙은 폐기한다. v2에서 정답의 직접 노출 여부는 `transformation required vs direct retrieval` 조작의 구성 요소로 취급한다. 따라서 결과를 “순수한 산술 연산 효과”로 해석해서는 안 된다.

## 4. Answer-value counterbalance

각 template family에는 numeric value block 하나를 배정한다. 각 block은 두 정수 `A`, `B`를 사용한다.

| Answer orientation | Relevant answer | Irrelevant answer |
|---|---:|---:|
| `A_to_relevant` | A | B |
| `B_to_relevant` | B | A |

예시:

```text
Block orientation 1: Relevant=5, Irrelevant=6
Block orientation 2: Relevant=6, Irrelevant=5
```

두 orientation은 같은 lexical/template family 안에 반드시 함께 존재한다. 특정 숫자의 사전 확률이 relevance 효과로 섞이지 않도록 `A`와 `B`가 Relevant/Irrelevant 정답으로 동일 횟수 등장해야 한다.

추가 규칙:

- A/B 범위는 10–29다.
- delta 범위는 2–7이다.
- `A != B`이고 `|A - B| >= 2`여야 한다.
- A/B는 tokenizer 기준 answer token 수가 같아야 한다.
- A/B의 숫자 자릿수와 answer format을 맞춘다.
- A/B 어느 것도 transformation의 start/delta와 같지 않게 한다.
- A/B의 크기 관계도 block 간 counterbalance한다. Relevant 정답이 항상 더 작거나 크면 안 된다.
- 18개 template family에는 서로 독립적으로 생성된 18개 numeric block을 하나씩 배정한다.

## 5. Label과 문장 순서 counterbalance

Answer orientation과 독립적으로 다음을 교차한다.

### 5.1 Label pair와 role

- 각 lexical family의 template 1–3: red/blue label pair
- 각 lexical family의 template 4–6: green/black label pair
- 선택된 label pair 안에서 label-role orientation을 두 번 교차한다.
  - label 1 direct / label 2 transformed
  - label 2 direct / label 1 transformed

각 label은 direct와 transformed 역할에 동일 횟수 배치한다.

### 5.2 Context sentence order

- direct fact first / transformed fact second
- transformed fact first / direct fact second

문장 순서가 answer-copy 또는 최근성 효과를 만들지 않도록 두 순서를 동일 횟수 사용한다.

### 5.3 최소 완전 교차 단위

한 template family 안에서 다음 세 factor를 완전 교차한다.

```text
answer orientation (2)
× label-role orientation (2)
× sentence order (2)
= 8 frames/template
```

## 6. Pair invariant

v2에서 유지할 invariant:

- Relevant/Irrelevant pair의 context 동일
- entity set 동일
- operation과 operands 동일
- 질문 문형 동일
- target label만 변경
- answer format 동일
- prompt token 길이 exact match
- target label token 수 동일
- source relation span 동일

v2에서 폐기할 invariant:

- 동일 canonical answer
- 동일 answer literal exposure

## 7. Primary scoring

서로 다른 정답의 raw sequence logP를 바로 빼면 숫자 prior가 섞일 수 있으므로 primary로 사용하지 않는다.

각 prompt에서 correct answer `y`와 같은 block의 counterbalanced alternative answer `y_alt`를 사전에 지정한다.

```text
M(x) = logP_seq(y_correct | x) - logP_seq(y_alt | x)
```

`M(x)`는 모델이 A/B 중 올바른 값을 얼마나 선호하는지를 나타내는 paired-answer log-odds다. A/B의 역할은 counterbalanced frame에서 교환한다.

template-family cluster `k`의 baseline difficulty contrast:

```text
D_k = (1/8) sum_f [ M(x_relevant,f) - M(x_irrelevant,f) ]
```

여기서 `f`는 answer orientation, label-role orientation과 sentence-order orientation을 완전 교차한 8개 frame이다.

Primary metric:

- counterbalanced paired-answer log-odds `D_k`

Secondary metrics:

- greedy generation exact-match accuracy
- raw canonical sequence logP
- length-normalized canonical logP
- per-token geometric mean probability
- correct answer가 A/B forced-choice에서 선택되는 비율

Forced-choice accuracy는 `M(x) > 0`이면 correct, `M(x) <= 0`이면 incorrect로 계산한다.

추가 arithmetic foil은 다음처럼 고정한다.

```text
decrease:              start, delta, start + delta
increase/temperature:  start, delta, start - delta
```

`foil != correct`, `foil != y_alt`, foil끼리 서로 다름을 모두 만족해야 한다. 충돌하면 해당 문항을 사용하지 않고 deterministic numeric-block generator가 새 block을 생성한다. 결과를 본 뒤 foil을 변경하지 않는다.

## 8. Generation 규칙

v1과 동일한 재현성 규칙을 재승인한다.

```text
do_sample: false
num_beams: 1
max_new_tokens: 8
stop: EOS 또는 첫 newline
```

정규화 순서는 다음과 같이 고정한다.

1. Unicode NFKC
2. outer whitespace strip
3. first line
4. lowercase
5. terminal `.`, `,`, `!`, `?` 제거
6. prespecified accepted-answer exact match

자동 digit↔number-word 변환은 하지 않는다. alias는 dataset freeze 전에 지정하며 결과를 본 뒤 추가하지 않는다.

## 9. Calibration acceptance criterion

다음 값은 v1 기준의 자동 승계가 아니라 v2 scoring에 맞춘 신규 승인값이다.

### 9.1 절대 난이도

| 지표 | Relevant | Irrelevant |
|---|---:|---:|
| A/B forced-choice accuracy | 0.60–0.95 | 0.60–0.95 |
| greedy generation accuracy | 0.20–0.95 | 0.20–0.95 |

### 9.2 조건 간 불균형

```text
|forced-choice accuracy_R - forced-choice accuracy_I| <= 0.10
|generation accuracy_R - generation accuracy_I| <= 0.10
|mean_k(D_k)| <= 0.40 nat
|d_z| <= 0.30
같은 부호의 D_k <= 4/6 templates
```

각 기준은 lexical family별로 모두 적용하며 하나라도 실패하면 해당 lexical family 전체를 FAIL로 판정한다. 결과를 보고 6개 template 중 일부를 제외하는 `5/6 pass` 규칙은 사용하지 않는다.

Raw canonical logP, length-normalized logP와 per-token geometric probability는 diagnostic secondary metric이며 PASS/FAIL gate에 사용하지 않는다. 결과를 본 뒤 threshold를 변경하지 않는다.

### 9.3 Cluster 통계

각 lexical family에는 독립 template-family cluster 6개가 있으며 `D_1, ..., D_6`를 계산한다.

```text
d_z = mean_k(D_k) / sample_sd_k(D_k)
```

- 표준편차는 sample standard deviation(`ddof=1`)을 사용한다.
- 모든 `D_k`가 정확히 0이면 `d_z=0`으로 정의한다.
- `sd(D_k)`가 float64 machine epsilon 이하이고 평균이 0이 아니면 FAIL이다.
- NaN, infinity 또는 missing cluster가 있으면 FAIL이다.
- 같은 방향 기준은 `D_k>0`인 cluster 수와 `D_k<0`인 cluster 수 중 큰 값으로 계산하며 0은 어느 부호에도 포함하지 않는다.
- item 144개를 독립표본으로 간주한 p-value는 계산하지 않는다.

Lexical family별 template cluster 6개를 replacement resampling하는 stratified cluster bootstrap 10,000회를 수행하고 percentile 95% CI를 descriptive 결과로 보고한다. Bootstrap seed는 `20260829`로 고정하며 CI를 보고 PASS/FAIL threshold를 변경하지 않는다.

## 10. Dataset 규모

독립 단위는 `template_family`다. item 수를 독립 표본처럼 취급하지 않는다.

| 항목 | 승인값 |
|---|---:|
| lexical family | decrease / increase / temperature |
| lexical family 수 | 3 |
| family당 template family 수 | 6 |
| template당 numeric block 수 | 1 |
| answer orientation | 2 |
| label-role orientation | 2 |
| sentence-order orientation | 2 |
| template당 frame 수 | 8 |
| frame당 instantiation 수 | 1 |
| 총 paired item 수 | 144 |
| 총 prompt 수 | 288 |
| 독립 template-family cluster 수 | 18 |
| random seed | 20260829 |

계산은 `3 families × 6 templates × 8 frames × 1 instantiation = 144 pairs`다.

## 11. Shortcut audit

모든 pair에 대해 다음을 자동 검사한다.

1. Relevant answer와 같은 값을 가진 **독립 numeric mention**이 context 어디에도 없음
2. Relevant answer가 start/delta와 다름
3. Irrelevant answer가 직접 사실에 정확히 한 번 존재
4. Relevant answer와 Irrelevant answer가 다름
5. A/B answer가 relevance 양쪽에 동일 횟수 등장
6. label의 direct/transformed 역할 균형
7. direct/transformed sentence order 균형
8. prompt token 길이 exact match
9. 질문 target 외 문자열 구조 동일
10. 동일 template 안에서 answer magnitude 방향 균형

숫자 노출 검사는 문자열 substring이 아니라 경계가 분리된 numeric mention을 parsing하여 비교한다. 예를 들어 정답 `15`를 context의 `115`와 일치한다고 판정하지 않는다.

한 항목이라도 실패하면 dataset을 human audit로 넘기지 않는다.

## 12. Human linguistic audit

AI pre-audit은 보조자료일 뿐 독립 human audit를 대체하지 않는다. 독립 reviewer 2명이 각각 144 pair 전부를 검수한다. 두 reviewer는 AI 판정, model prediction, logP, generation, LiReF, candidate/intervention 결과 및 서로의 판정을 보지 않는다.

검수자는 다음을 확인한다.

- 계산과 canonical answer 정확성
- Relevant에서 transformation이 실제로 필요한지
- Irrelevant에서 직접 조회가 가능한지
- 다른 context 숫자를 복사해 Relevant 정답을 맞힐 수 없는지
- 질문 target 외 의미 변화가 없는지
- 문법과 문장 의미가 자연스러운지
- counterbalance counterpart가 의미적으로 동등한지

두 reviewer가 모두 PASS하면 pair PASS다. 불일치는 제3 reviewer가 독립적으로 판정하고 사유를 기록한다. 최종 FAIL 또는 NEEDS_REVISION이 하나라도 있으면 해당 v2 dataset은 실행하지 않는다. 수정이 필요하면 기존 파일을 덮어쓰지 않고 v2.1 dataset으로 재생성하여 automatic audit과 human audit를 처음부터 다시 수행한다.

## 13. 안전 및 non-reuse

v2 calibration에서도 다음을 금지한다.

- model forward 이전에 design/dataset/hash 미동결 상태로 실행
- LiReF direction 로딩
- hidden state와 candidate state capture
- forward hook
- activation/weight intervention
- Calibration 문항의 Pilot/Confirmatory 재사용
- 결과를 본 뒤 item 삭제 또는 acceptance criterion 변경

Baseline Calibration은 별도 implementation·정적 safety review·명시적 execution approval 이후에만 실행한다.

## 14. 현재 가능한 주장 범위

Calibration PASS가 의미하는 것:

- v2 Relevant/Irrelevant 조건이 baseline에서 측정 가능한 난이도 범위에 있음
- answer value, label, sentence order를 counterbalance한 뒤에도 과도한 난이도 불균형이 없음

Calibration PASS만으로 주장할 수 없는 것:

- relation transformation이 LiReF R 방향의 원인임
- 기존 4개 component가 해당 feature를 전달함
- R-specific pathway를 찾았음

이 주장은 이후 Stage E Pilot과 Confirmatory pathway intervention이 필요하다.

## 15. 승인된 v2 설계값과 freeze gate

| 항목 | 승인값 |
|---|---|
| Operationalized feature | transformation-dependent answer derivation vs direct-fact retrieval |
| Answer block | A/B 10–29, delta 2–7, `A!=B`, `|A-B|>=2`, 동일 token/digit length, operand 충돌 금지 |
| Counterbalance | answer orientation 2 × label-role 2 × sentence order 2 = 8 frames/template |
| Dataset | 3 families × 6 templates × 8 frames = 144 pairs / 288 prompts |
| Instantiation / seed | frame당 1 / `20260829` |
| Primary metric | paired-answer log-odds와 template-level `D_k` |
| Arithmetic foils | start, delta, wrong-operation result; correct/y_alt/foil 중복 금지 |
| Generation | greedy, beam 1, max 8, EOS/첫 newline stop, frozen normalization |
| Acceptance | forced-choice 0.60–0.95; generation 0.20–0.95; condition gaps ≤0.10; `|mean(D_k)|≤0.40`; `|d_z|≤0.30`; same sign ≤4/6 |
| Cluster statistics | template family 독립 단위, sample-SD `d_z`, 10,000회 descriptive cluster bootstrap |
| Human audit | 독립 reviewer 2명 전수검수, 불일치만 제3 reviewer adjudication |

수치와 선택지는 모두 승인되었으며 남은 단계는 이 문서의 최종 사람 확인이다. 최종 확인 전에는 machine-readable v2 frozen manifest, dataset builder 또는 baseline calibration 코드를 작성하지 않는다.

최종 확인 후 순서:

1. v2 frozen manifest와 SHA-256 생성
2. deterministic dataset builder 작성
3. 144 pair 생성
4. shortcut·counterbalance·tokenizer audit
5. AI pre-audit
6. 독립 human audit

이 단계에서도 모델 forward와 GPU 실행은 금지한다.

## 16. Design freeze 기록

- freeze date: `2026-08-29`
- 승인된 pre-freeze source document SHA-256: `85334c4fac6b4ffb09d4cb0ffc6cc256158408cbd042604e8164574604f86e35`
- machine-readable manifest: `calibration_v2_design_frozen.json`
- frozen manifest SHA-256: `a8f3dad7fced945377194074f9aa12d673faff3b55c3ec45bd82e397b5a5302b`
- manifest status: `design_frozen_execution_not_approved`
- `execution_allowed`: `false`
- deterministic dataset builder: 생성 전
- calibration dataset: 생성 전
- model forward / GPU: 실행 금지

이후 설계 규칙과 threshold를 변경해야 하면 이 manifest를 덮어쓰지 않는다. 변경 사유를 기록하고 새로운 design version과 SHA-256을 생성한다.
