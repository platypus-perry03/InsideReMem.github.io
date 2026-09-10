# Stage E Behavioral Control v5 설계 초안

상태: **FROZEN — DATASET BUILD/AUDIT 허용, MODEL EXECUTION 미승인**  
작성일: `2026-08-30`

## 1. v4 실패 원인과 v5 목표

v4 artifact와 결과는 수정하지 않는다.

- v4에서 Selector ceiling은 제거됐다.
- 그러나 points-balance는 Arithmetic 33/64, Selector 37/64, temperature는 32/64, 37/64로 두 조건 모두 거의 chance 수준이었다.
- 정답 A/B가 32:32로 균형이었음에도 모델은 특히 points-balance에서 A를 자주 선택했다.
- 원인은 `label → case → key/tag → ledger → value → operation/selection → A/B`의 다단계 symbolic binding 부담으로 진단한다.

따라서 v5는 R/M 가설이나 Layer 31 endpoint를 바꾸지 않는다. 다음 하나만 바꾼다.

> **Arithmetic과 Selector의 terminal rule은 유지하되, 정답 이전의 binding hop을 각각 한 단계로 줄여 absolute solvability를 회복한다.**

## 2. 단순화된 조작

한 pair는 같은 context와 같은 두 숫자 후보를 사용하고 질문의 target label만 다르다.

```text
Mira's bronze score:
START=52; CHANGE=7; RULE=ADD; A=59; B=45.

Mira's silver score:
ACTIVE=BLUE; BLUE=59; RED=45; A=59; B=45.

Arithmetic question: Which choice is Mira's bronze score now?
Selector question:   Which choice is Mira's silver score now?
Answer with A or B only.
Answer:
```

- Arithmetic: `START ± CHANGE`를 계산하고 A/B에 직접 대응한다.
- Selector: `ACTIVE`가 가리키는 숫자를 직접 찾고 A/B에 대응한다.
- 이름, case key, arbitrary value key, 별도 ledger는 사용하지 않는다.
- 두 조건 모두 후보 숫자와 A/B mapping을 context에서 직접 본다.
- pair 안의 context는 완전히 동일하며 질문 target label만 바뀐다.

공식 feature 명칭:

> **single-step arithmetic transformation vs single-step explicit-tag selection under shared exposed choices**

이 조작은 Reasoning/Memorization label 자체가 아니다. 이후 독립 재현에서 frozen LiReF 방향과 연결되는지를 별도로 확인한다.

## 3. Answer exposure와 foil

숫자 block은 `start`, `delta`, `high=start+delta`, `low=start-delta`로 구성한다.

- Arithmetic correct: ADD이면 `high`, SUBTRACT이면 `low`
- Arithmetic foil: frozen wrong-operation result
- Selector mapping: 두 tag가 각각 `high`와 `low`를 직접 가리킴
- Selector correct: active tag가 가리키는 값
- Selector foil: inactive tag가 가리키는 값
- 공통 A/B mapping: `high→A, low→B` 또는 반대 방향

`high`와 `low`는 다음 위치에 각각 동일 횟수 등장한다.

1. Arithmetic A/B choice list
2. Selector tag-to-value mapping
3. Selector A/B choice list

따라서 correct/foil numeric exposure, digit length, candidate identity와 A/B mapping이 두 조건에서 동일하다. `start`와 `delta`는 Arithmetic operand로만 등장하며 이는 의도한 terminal-rule 조작의 일부다.

## 4. Numeric 규칙

동결값:

- start: 40–69
- delta: 2–9
- high/low: 두 자리 정수
- `start`, `delta`, `high`, `low` pairwise distinct
- `high != low`, `|high-low| >= 4`
- A/B continuation은 frozen Meta-Llama-3-8B tokenizer에서 각각 정확히 한 token
- numeric block은 template마다 독립 생성
- deterministic seed: `20260902`

## 5. Counterbalance

각 template은 고정된 `OA(8,5,2,2)` strength-2 matrix를 사용한다.

독립 nuisance factor:

1. Arithmetic operation: ADD / SUBTRACT
2. Selector active tag: high-value tag / low-value tag
3. Candidate assignment: high→A / high→B
4. Label-role orientation
5. Block order: Arithmetic first / Selector first

동결 matrix:

| Row | operation | active tag | candidate assignment | label role | block order |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 1 | 1 |
| 3 | 0 | 1 | 0 | 0 | 1 |
| 4 | 0 | 1 | 1 | 1 | 0 |
| 5 | 1 | 0 | 0 | 1 | 0 |
| 6 | 1 | 0 | 1 | 0 | 1 |
| 7 | 1 | 1 | 0 | 1 | 1 |
| 8 | 1 | 1 | 1 | 0 | 0 |

각 factor는 4:4이고 임의의 두 factor의 2×2 cell은 각각 두 번 등장해야 한다. Arithmetic/Selector correct choice A/B도 각각 4:4가 되어야 한다. 이 matrix는 nuisance counterbalance용이며 interaction 추정용이 아니다.

## 6. 데이터 pool과 non-reuse

### Small Calibration pool

- points-balance: 4 templates × 8 frames = 32 pairs
- temperature: 4 templates × 8 frames = 32 pairs
- 합계: 64 pairs / 128 prompts

### Sealed independent replication pool

- points-balance: 8 templates × 8 frames = 64 pairs
- temperature control: 8 templates × 8 frames = 64 pairs
- 합계: 128 pairs / 256 prompts

두 pool은 이름, label pair, wording template, numeric block, 완성 prompt와 normalized skeleton이 겹치지 않아야 한다. v2–v4 dataset 및 v3.2 Pilot 문항과도 중복하지 않는다.

Replication pool은 Calibration 결과 이전에 생성·hash-lock·감사하되, Calibration gate가 열리기 전에는 모델에 입력하지 않는다. 기존 v4 replication pool은 계속 봉인하고 v5에서 재사용하지 않는다.

## 7. Behavioral Calibration scoring

```text
M(x) = logP(correct A/B token | prompt) - logP(incorrect A/B token | prompt)
D_k = mean_8frames[M(Arithmetic) - M(Selector)]
```

각 family의 authoritative denominator는 조건당 32다.

### Hard gate

- forced-choice correct: 각 조건 `22–29 / 32`
- forced-choice condition count gap: `≤3 / 32`
- one-token A/B generation correct: 각 조건 `22–29 / 32`
- generation condition count gap: `≤3 / 32`
- `|mean_k(D_k)| ≤ 0.40 nat`
- missing/NaN/infinity: FAIL
- 모든 기준 동시 충족 시에만 family PASS

근거:

- chance 0.5에서 `P[X≥22], X~Binomial(32,0.5) ≈ 0.0251`이므로 하한은 chance 수준의 선택을 통과시키지 않는다.
- 상한 29/32는 최소 3개 오류를 요구해 ceiling을 방지한다.
- count gap 3/32는 두 조건 차이를 9.375 percentage points 이하로 제한한다.
- mean-margin 기준은 조건 간 teacher-forced imbalance를 별도로 제한한다.
- 결과를 본 뒤 threshold, item, template 또는 family를 바꾸지 않는다.

### Descriptive diagnostic

- `d_z = mean(D_k) / sample_sd(D_k)`, `ddof=1`
- cluster별 `D_k`
- template-cluster bootstrap 10,000회, seed `20260902`의 mean `D_k` 95% CI

4개 cluster의 `d_z`는 template 하나에 민감하므로 계산·보고하되 PASS/FAIL에 사용하지 않는다. Bootstrap CI도 descriptive이며 hard threshold를 사후 변경하지 않는다.

## 8. Calibration 진행 gate

- points-balance PASS: v5 sealed points-balance replication 실행 가능
- points-balance와 temperature 모두 PASS: family interaction replication도 가능
- points-balance FAIL: independent Layer 31 replication 중단
- temperature만 FAIL: temperature와 interaction은 실행하지 않고 points-balance primary만 가능
- PASS family가 있어도 자동 실행하지 않고 별도 implementation/static review/execution authorization을 요구

## 9. Independent replication endpoint

v3.2 결과를 본 뒤 정한 endpoint를 변경하지 않는다.

### Primary

- population: sealed v5 points-balance replication 64 pairs 전체
- endpoint: Meta-Llama-3-8B base Layer 31 last-prompt-token frozen LiReF projection
- pair effect: Arithmetic − Selector
- independent unit: 8 wording-template clusters
- replication rule: template-cluster bootstrap 95% CI lower bound `>0`
- bootstrap: 10,000회, seed `20260902`

### Prespecified secondary

- 두 family가 Calibration PASS인 경우 Layer 31 points-balance − temperature interaction

### Exploratory only

- Layer 0–30 trajectory
- `L31N13336`, `L29H30`, `L30H6`, `L29H31` frozen scalar contribution
- ADD/SUB 및 correct/incorrect behavioral diagnostics

Exploratory 결과로 primary endpoint, family, layer 또는 candidate를 교체하지 않는다.

## 10. Audit와 실행 제한

필수 model-free audit:

1. automatic structural/numeric/tokenizer/counterbalance/non-reuse audit
2. primary AI linguistic/semantic audit 전수
3. 별도 adversarial AI shortcut/bias audit 전수

- human audit: `not_performed`
- human-audited evidence 주장 금지
- 하나라도 FAIL이면 해당 dataset version에서 model execution 금지

현재 허용:

- 이 draft의 검토와 수정
- exact template catalog와 matrix의 model-free 검증

현재 금지:

- v5 design freeze 전 builder/dataset 생성
- model/tokenizer runtime을 이용한 forward
- GPU
- LiReF direction loading
- replication pool model use
- hidden state/candidate capture
- hook/intervention/patching/suppression

## 11. Claim limits

Calibration PASS는 두 operationalized condition이 baseline에서 풀 수 있고 과도하게 불균형하지 않다는 뜻일 뿐이다.

Independent replication 성공 시에도 다음까지만 주장한다.

> **새 points-balance 문항에서도 Arithmetic이 Selector보다 Layer 31 frozen LiReF coordinate에서 상대적으로 높게 나타났다.**

절대 R/M 분류, Reasoning neuron/head, component mediation, causal mechanism 또는 다른 모델 일반성은 주장하지 않는다.

## 12. Design freeze 결정

- 64-pair Calibration 규모: 승인
- `22–29/32`, count gap `≤3`, `|mean(D_k)|≤0.40`: hard gate로 승인
- 4-cluster `d_z`와 bootstrap CI: descriptive diagnostic으로 승인
- exact calibration/replication wording catalog와 non-reuse 검사: dataset build 전에 hash-lock
- `OA(8,5,2,2)` exact matrix: factor 4:4 및 모든 factor pair cell 2회 검증 후 동결
- AI-only audit 정책과 human-audited claim 금지: 승인

결과를 본 뒤 본 문서, threshold 또는 gate를 수정하지 않는다. 변경이 필요하면 v5 artifact를 보존하고 새 design version을 생성한다.
