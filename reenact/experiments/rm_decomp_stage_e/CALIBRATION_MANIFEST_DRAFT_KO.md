# Stage E Baseline Calibration Manifest

상태: **DESIGN PARAMETERS APPROVED — dataset·구현·audit 미완료, 실행 금지**  
문서 역할: Stage E 첫 feature를 동결하기 전에 감소·증가·온도 controlled pair의 baseline 난이도만 평가하기 위한 사전 명세다. 수치와 scoring 규칙은 2026-08-28 사용자 승인으로 동결했으며, 별도 machine-readable design manifest를 기준으로 한다. Dataset·구현·linguistic audit·명시적 실행 승인이 완료되기 전에는 calibration을 실행하지 않는다.

## 1. Calibration 목적

Calibration은 다음 두 질문에만 답한다.

1. Relevant condition이 baseline model에서 floor 수준으로 어려운가?
2. Irrelevant condition이 baseline model에서 ceiling 수준으로 쉬운가?

Calibration은 feature effect, LiReF mechanism, candidate activation 또는 pathway causality를 검증하지 않는다.

## 2. 실행 범위

### 2.1 포함

- lexical family: `decrease`, `increase`, `temperature`
- frozen prompt template: `Q: {question}\nA: `
- baseline teacher-forced answer score
- baseline greedy generation
- Relevant/Irrelevant 조건별 난이도와 paired 난이도 차이
- tokenizer length와 answer exposure 검증

### 2.2 제외

- LiReF score 및 direction projection
- hidden state 또는 residual stream 저장
- 후보 4개의 activation/output capture
- Attention weight, Q/K/V/O, FFN gate/up/z/down capture
- activation patching, suppression, steering, weight edit
- 후보 또는 feature direction 변경
- Pilot/Confirmatory claim

## 3. 강제 안전 규칙

Frozen manifest에는 다음 값을 그대로 기록해야 한다.

```text
baseline_behavior_only: true
capture_hidden_states: false
inspect_liref_scores: false
inspect_candidate_states: false
enable_forward_hooks: false
enable_activation_intervention: false
enable_weight_intervention: false
confirmatory_claim_allowed: false
reuse_in_pilot: false
reuse_in_confirmatory: false
allow_feature_reselection: false
allow_direction_change: false
model_training: false
```

실행기가 이 중 하나라도 다른 값을 받으면 시작 전에 중단해야 한다.

## 4. 고정 연구 범위

| 항목 | 값 | 상태 |
|---|---|---|
| Calibration 대상 feature | 동일 context에서 정답 산출에 single-step relation transformation이 필요한가 | 설계 승인 |
| Relevant condition | 정답을 얻기 위해 context의 변화 연산을 적용해야 함 | 고정 후보 |
| Irrelevant condition | 같은 변화 연산이 context에 있으나 정답은 직접 제시된 matched fact에서 조회 | 고정 후보 |
| Context | Relevant/Irrelevant pair 안에서 완전히 동일 | 필수 |
| Entity set | pair 안에서 동일 | 필수 |
| Correct answer | pair 안에서 동일 | 필수 |
| Answer literal exposure | 두 조건의 공통 context에 동일하게 존재 | 필수 |
| Question target | direct attribute와 transformed attribute 사이에서만 변경 | 필수 |
| Counterbalance | target label의 direct/transformed 역할을 template family 안에서 대칭 배치 | 필수 |
| Primary endpoint 후보 | Layer 31 last-prompt-token frozen LiReF score | **Calibration에서는 계산 금지; Stage E freeze 후보** |
| Layer profile | secondary diagnostic 후보 | **Calibration에서는 계산 금지** |

## 5. Seed family audit 상태

로컬 Meta-Llama-3-8B tokenizer로 실시한 사전 audit 결과다. 이는 seed 문장의 구조 검증이며 최종 calibration dataset 승인과 동일하지 않다.

| Family | Relevant 길이 | Irrelevant 길이 | 역할 반전 4조건 길이 일치 | target token | answer literal 노출 | 상태 |
|---|---:|---:|---|---|---|---|
| decrease | 40 | 40 | PASS | 각 1 token | 양쪽 공통 context에 존재 | seed 사용 가능 |
| increase | 35 | 35 | PASS | 각 1 token | 양쪽 공통 context에 존재 | seed 사용 가능 |
| temperature | 39 | 39 | PASS | 각 1 token | 양쪽 공통 context에 존재 | seed 사용 가능 |

`discount`와 `rank` family는 의미 표현 수정 전에는 calibration에 포함하지 않는다.

## 6. Dataset 규모 — 사람 승인 필요

| 항목 | 값 | 승인 상태 |
|---|---|---|
| 총 lexical family 수 | `3` | 승인 |
| family당 template family 수 | `6` | 승인 |
| template family당 counterbalance frame 수 | `4` | 승인 |
| frame당 item instantiation 수 | `2` | 승인 |
| 총 paired item 수 | `144` | 승인 |
| 총 prompt 수 | `288` | 승인 |
| 독립 template-family 최소 수 | `18` | 승인 |
| random seed | `42` | 승인 |

독립 표본 단위는 item 수가 아니라 `template_family`다. 같은 template의 숫자·이름 교체본은 같은 cluster로 기록한다.

## 7. Calibration item schema

각 pair는 최소한 다음 필드를 가져야 한다.

```text
calibration_item_id
pair_id
lexical_family
template_family_id
counterbalance_frame_id
condition                 # relevant | irrelevant
context
question
full_prompt
target_attribute
direct_attribute
transformed_attribute
operation
operands
correct_answer
accepted_answers
prespecified_foils
answer_format
answer_literal_in_context
changed_question_span
changed_question_token_ids
source_relation_span
source_relation_token_ids
prompt_token_count
human_review_status
pilot_reuse_allowed        # always false
confirmatory_reuse_allowed # always false
```

`correct_answer`, `accepted_answers`, `prespecified_foils`는 inference 전에 저장하고 결과를 본 뒤 수정하지 않는다. Teacher-forced primary에는 canonical `correct_answer` 하나만 사용하며, `accepted_answers`는 generation accuracy에만 사용한다.

## 8. Multi-token 정답 확률 계산법

### 8.1 제안 방식

정답 문자열 `y = (y_1, ..., y_T)`의 teacher-forced sequence log probability를 사용한다.

```text
logP_seq(y | x) = sum_t log p(y_t | x, y_<t)
```

- pair 안에서는 동일한 정답 문자열을 사용하므로 Relevant/Irrelevant 비교에서 answer-length bias가 동일하다.
- primary behavior score: canonical `logP_seq(correct | prompt)`
- secondary score: `logP_seq / T`와 prespecified foil set 중 최대-score foil 대비 log odds
- absolute floor/ceiling용 headroom score: `P_geo = exp(logP_seq / T)`
- foil set은 inference 전에 동결하며 결과를 보고 foil을 추가·삭제하지 않는다.

### 8.2 동결 전 결정 항목

| 항목 | 값 | 승인 상태 |
|---|---|---|
| primary score를 sequence sum으로 사용할지 | canonical-answer sequence logP sum | 승인 |
| length-normalized score의 secondary 사용 여부 | 사용 | 승인 |
| correct-versus-foil log odds의 primary/secondary 위치 | secondary; prespecified foil set의 최대 score 사용 | 승인 |
| answer 앞 leading-space 처리 | `prompt + canonical_answer` 전체를 함께 tokenize하고 answer suffix token만 score; 임의 trim 금지 | 승인 |
| EOS/newline을 answer probability에 포함할지 | 미포함 | 승인 |
| accepted answer가 여러 개인 경우 aggregate 규칙 | teacher-forced에는 미사용; generation exact-match에만 사전 열거 alias 사용 | 승인 |

## 9. Generation decoding과 정답 정규화

### 9.1 제안 decoding

재현성을 위해 greedy decoding을 기본 후보로 한다.

```text
do_sample: false
num_beams: 1
max_new_tokens: 8
stop: EOS 또는 첫 newline
```

Temperature, top-p 등 sampling parameter는 사용하지 않는 방향을 제안하지만 사람 승인 후 동결한다.

### 9.2 제안 정규화

생성 결과는 다음 순서로 처리하는 방안을 제안한다.

1. Unicode NFKC 정규화
2. 앞뒤 공백 제거
3. 첫 줄 또는 frozen answer extraction rule 적용
4. 대소문자 정규화
5. 끝의 `.`, `,`, `!`, `?`만 제거
6. item별 `accepted_answers`와 exact match

숫자 단어와 Arabic numeral을 자동으로 무제한 변환하지 않는다. 허용 alias는 item 생성 시 사전 열거한다.

| 항목 | 값 | 승인 상태 |
|---|---|---|
| max new tokens | `8` | 승인 |
| stop rule | EOS 또는 첫 newline | 승인 |
| 첫 줄만 사용할지 | 사용 | 승인 |
| 단위 포함 답의 허용 alias | item 생성 시 `accepted_answers`에 사전 열거 | 승인 |
| punctuation 제거 범위 | 정규화된 첫 줄 끝의 `.`, `,`, `!`, `?` | 승인 |
| 설명 뒤 정답을 허용할지 | 허용하지 않음; 정규화된 첫 줄 전체가 alias와 일치해야 함 | 승인 |

## 10. Floor / ceiling 기준

수치를 결과 보기 전에 동결한다.

| Criterion | Relevant | Irrelevant | 승인 상태 |
|---|---:|---:|---|
| generation accuracy floor | `20%` | `20%` | 승인 |
| generation accuracy ceiling | `95%` | `95%` | 승인 |
| mean per-token geometric probability floor | `0.20` | `0.20` | 승인 |
| mean per-token geometric probability ceiling | `0.95` | `0.95` | 승인 |
| family별 최소 통과 template 비율 | `5/6` | `5/6` | 승인 |
| 개별 template-family 허용 실패 수 | 최대 `1` | 최대 `1` | 승인 |

Accuracy와 per-token geometric probability 기준은 모두 적용한다. 하나라도 범위를 벗어나면 해당 template은 통과하지 않는다.

## 11. Relevant–Irrelevant 난이도 차이 기준

같은 pair의 Relevant와 Irrelevant baseline 행동 차이가 허용 범위를 넘으면 해당 family는 Stage E primary feature 후보에서 제외한다.

| 지표 | 허용 절댓값 | 승인 상태 |
|---|---:|---|
| paired generation accuracy gap | `≤ 10 percentage points` | 승인 |
| absolute paired mean canonical sequence logP gap | `≤ 0.30 nat` | 승인 |
| paired standardized effect `|d_z|` | `≤ 0.30` | 승인 |
| 같은 gap 방향의 template-family 최대치 | 최대 `4/6` | 승인 |

Template-family cluster `k`에 대해 다음과 같이 계산한다.

```text
D_k = mean_i_in_k[logP_seq(Relevant_i) - logP_seq(Irrelevant_i)]
d_z = mean_k(D_k) / sd_k(D_k)
```

- 모든 `D_k = 0`이면 `d_z = 0`으로 정의한다.
- `sd(D_k)`가 machine epsilon 이하인데 `mean(D_k) != 0`이면 FAIL이다.
- NaN, infinity 또는 missing cluster가 있으면 FAIL이다.
- raw logP gap, accuracy gap, `|d_z|` 세 기준을 동시에 만족해야 한다.

난이도 차이를 줄이기 위해 결과를 본 뒤 특정 item만 삭제하지 않는다. Family가 실패하면 실패로 기록하고, 수정된 operationalization은 새 calibration run으로 분리한다.

## 12. Family 채택 판정 구조

Family는 다음을 모두 만족해야 Stage E 1–4 freeze 대상으로 진입한다.

1. tokenizer exact-length 및 counterbalance audit 통과
2. answer exposure와 human linguistic audit 통과
3. Relevant floor/ceiling 기준 통과
4. Irrelevant floor/ceiling 기준 통과
5. Relevant–Irrelevant 난이도 gap 기준 통과
6. 모든 output이 baseline behavior-only 허용 범위 안에서 생성됨
7. calibration item이 Pilot/Confirmatory 제외 manifest에 기록됨

Family가 탈락해도 다른 feature나 후보를 이 calibration 결과로 새로 선택하지 않는다.

## 13. 허용 출력

- frozen calibration manifest와 hash
- item/pair/template-family manifest와 hash
- prompt token count와 target/source token indices
- teacher-forced answer score
- prespecified foil set의 최대-score foil 대비 log odds
- generated text와 normalized answer
- condition/family별 accuracy 및 score 요약
- paired difficulty gap과 cluster-aware uncertainty
- 환경·모델·tokenizer·코드 hash
- safety gate 통과/실패 log

## 14. 금지 출력

- hidden state tensor
- layer별 LiReF score
- candidate activation 또는 contribution
- Attention map과 head output
- FFN intermediate activation
- pathway ranking
- intervention·patch·rescue 결과

금지 출력이 생성되면 해당 run은 무효다.

## 15. Artifact 및 hash gate

| Artifact | SHA-256 / identity |
|---|---|
| Stage A identity | `7f0f7f020533b68e20e7b82d39dcacddad9f6e350dd599df6a40be667f8ec9f7` |
| Stage A frozen prompt | `924c8cb05f84bdcad8da83b67dc617ffbbde6e20d929bfa8b45fd451f4a468e1` |
| Model config | `2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441` |
| Model parameter set | `a86eda67086313c3b20d92a471820455c0ab6a9489db1870186980b0027bcb0b` |
| `tokenizer.json` | `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` |
| `tokenizer_config.json` | `690727b4fed286383df1c7ca5e805124cb70c6eb4529f807c7b2e60ff741da7e` |
| `special_tokens_map.json` | `462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc` |
| `generation_config.json` | `93caf96e269e32b9ee33ad78b0d76d910408d11a63a8b2c49241030836759311` |
| Calibration dataset draft | `calibration_assets/calibration_dataset_draft.json`; SHA-256 `ba4c0fe2b17633c082ca666c4af7133b90a7e221efb6824bac74c99eddf2505b` |
| Calibration tokenizer audit | `calibration_assets/calibration_tokenizer_audit.json`; SHA-256 `1acc16642fd133b2a45bd84c9ae79b21b69b93346111e9b3b051de53ad383f7a` |
| Calibration dataset manifest | `calibration_assets/calibration_dataset_manifest.json`; SHA-256 `be0922f28ee55f11e3b7b1526d48f161995e92fa5b2a0bdaed4def78dbc0092c`; human audit 대기 |
| Linguistic audit sheet | `calibration_assets/calibration_linguistic_audit.csv`; SHA-256 `95f7830cfa906f5fef63b357869b72296acb7d08a109762ed573364231b619d4` |
| Linguistic audit guide | `CALIBRATION_LINGUISTIC_AUDIT_GUIDE_KO.md`; SHA-256 `78e086c65228798badc72cfd935379e3b1b8d0b138b937f667a3788fa63bd7d7` |
| Dataset/tokenizer-audit builder | `build_calibration_dataset.py`; SHA-256 `3cfdb5e8e2183b20706ed66838f58e302d45116a284b6b5cf7d5ff26f7c735aa` |
| Frozen calibration design | `calibration_design_frozen.json`; SHA-256 `ba9939e30b1e68c7b061e407f981c37bc2b1878055ef9074ede86f2afe5aacb0` |
| Calibration implementation | `PENDING_NOT_CREATED` |

Calibration은 Discovery LiReF direction 파일을 읽을 필요가 없다. 실행 코드에서 해당 파일을 입력으로 요구하거나 열면 safety violation으로 처리한다.

## 16. 사람 승인 표

| 승인 항목 | 결정값 | 검토자 | 날짜 | 상태 |
|---|---|---|---|---|
| 문항/template 규모 | 3 family × 6 template × 4 frame × 2 instantiation = 144 pairs / 288 prompts | user (대화에서 명시 승인) | 2026-08-28 | 승인 |
| 정답 확률 계산법 | canonical sequence logP primary; length-normalized 및 prespecified-foil log odds secondary | user (대화에서 명시 승인) | 2026-08-28 | 승인 |
| generation/정규화 규칙 | greedy, max 8, EOS/newline stop, frozen normalization | user (대화에서 명시 승인) | 2026-08-28 | 승인 |
| floor/ceiling | accuracy 20–95%; mean per-token geometric probability 0.20–0.95 | user (대화에서 명시 승인) | 2026-08-28 | 승인 |
| 난이도 gap | accuracy ≤10pp; raw logP ≤0.30 nat; cluster `|d_z|≤0.30`; same-direction ≤4/6 | user (대화에서 명시 승인) | 2026-08-28 | 승인 |
| dataset linguistic audit | 144쌍 audit sheet 생성, 검수 미완료 | independent reviewer 미배정 | — | `PENDING` |
| Pilot/Confirmatory 재사용 금지 | `false / false` | user (대화에서 명시 승인) | 2026-08-28 | 승인 |
| 전체 calibration 실행 승인 | `false` | user | — | `NOT_APPROVED` |

## 17. 동결 및 실행 gate

아래 조건을 모두 만족해야 별도의 machine-readable frozen manifest를 생성할 수 있다.

- [x] 모든 설계 수치 확정
- [x] scoring·decoding·normalization 선택 확정
- [x] calibration dataset 생성 및 tokenizer audit 완료 (144 pairs / 288 prompts; automated checks 144/144 PASS)
- [ ] dataset human linguistic audit 승인
- [x] item/template non-reuse flag 확인 (`pilot=false`, `confirmatory=false`, 144/144)
- [x] safety flag 전부 고정
- [x] 기존 model·prompt·tokenizer artifact hash 검증
- [ ] 실행 코드가 baseline behavior 외 tensor를 저장하지 않는다는 정적 검토
- [x] frozen design manifest SHA-256 생성 (`ba9939e30b1e68c7b061e407f981c37bc2b1878055ef9074ede86f2afe5aacb0`)
- [ ] 사람의 명시적 execution approval 기록

이 gate 이전에는 baseline calibration을 실행하지 않는다. Gate 통과 후에도 실행 범위는 baseline behavior calibration뿐이며 Pilot로 자동 진입하지 않는다.
