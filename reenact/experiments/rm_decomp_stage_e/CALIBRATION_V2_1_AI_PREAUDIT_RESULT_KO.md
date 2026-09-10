# Stage E Baseline Calibration v2.1 AI 사전검수 결과

상태: **COMPLETE — 24 NEEDS_REVISION, DATASET 실행 금지**  
검수일: 2026-08-30  
reviewer ID: `codex_ai_preaudit_nonhuman_v2_1`

## 1. 검수 범위

- frozen v2.1 dataset: 192 pairs / 384 prompts / 24 template families
- 검수 종류: 비인간 AI 사전검수
- 검수 단위: 192개 pair 전수 및 24개 template-family counterbalance 구조
- 모델·tokenizer 로딩: 없음
- model forward / GPU / LiReF / hidden state / candidate state / hook / intervention: 없음
- 기존 frozen design, builder 및 dataset 수정: 없음

AI 사전검수는 독립 human linguistic audit를 대체하지 않는다.

## 2. 최종 판정

| 판정 | Pair 수 |
|---|---:|
| `AI_PREAUDIT_PASS` | 168 |
| `AI_PREAUDIT_NEEDS_REVISION` | 24 |
| `AI_PREAUDIT_FAIL` | 0 |
| 합계 | 192 |

자동 structural/shortcut/tokenizer/counterbalance 검사는 192/192 통과했고, 24개 template family의 2×2×2 counterbalance도 모두 통과했다. 그러나 자연어 의미와 연산 방향 검수에서 3개 template family가 `NEEDS_REVISION`으로 판정되었다.

따라서 현재 v2.1 dataset의 전체 판정은 다음과 같다.

```text
dataset_pass: false
status: ai_preaudit_needs_revision_dataset_execution_forbidden
independent_human_audit_started: false
baseline_calibration_execution_allowed: false
stage_e_pilot_allowed: false
```

## 3. 수정 필요 template family

| Lexical family | Template family | 영향 Pair 수 | 사유 |
|---|---|---:|---|
| decrease | `v21_dec_badge_index` | 8 | `a trade of N badges`가 배지를 주었는지 받았는지 명시하지 않아 subtraction 방향이 모호함 |
| temperature | `v21_temp_reservoir_register` | 8 | `Entry ... shows N degrees before gaining M degrees`에서 온도가 아니라 record entry가 온도를 얻는 것처럼 읽힘 |
| temperature | `v21_temp_capsule_account` | 8 | `Account ... records N degrees before adding M degrees`에서 account가 온도를 더하는 것처럼 읽힘 |

이 문제들은 계산값이나 counterbalance 오류가 아니라, 자동검사가 보장하지 못하는 **자연어 연산 방향과 의미 주체의 명확성** 문제다. 각 template의 8개 counterbalance frame에 동일한 문형이 사용되므로 총 24개 pair가 영향을 받는다.

## 4. 통과한 검수 항목

192개 pair에서 다음 항목은 모두 통과했다.

- 산술 canonical answer 정확성
- Relevant/Irrelevant의 동일 context 및 output contract
- 질문 target 외 구조 불변
- label → key → record → value binding 구조
- Relevant의 transformation 필요성
- Irrelevant의 matched keyed direct retrieval
- cross-fact answer-copy shortcut 부재
- A/B paired-alternative orientation
- one-Arabic-numeral output instruction
- frozen automatic pair checks
- 24개 template의 8-frame 완전교차와 균형

단, 위 3개 template family에서는 `natural_grammar_and_unambiguous_change_direction`만 `NO`로 판정했다.

## 5. 다음 gate

현행 frozen protocol은 `NEEDS_REVISION`이 하나라도 있으면 human audit와 Baseline Calibration으로 진행할 수 없다. 따라서 다음 순서를 적용한다.

1. 현재 v2.1 design, builder, dataset 및 사전검수 결과를 그대로 보존한다.
2. 수정 사유와 변경 문형을 사전 기록한 새 dataset version을 만든다.
3. 기존 v2.1 dataset을 덮어쓰거나 일부 pair만 사후 제외하지 않는다.
4. 새 version에 대해 deterministic generation, near-duplicate, shortcut, tokenizer 및 counterbalance audit를 전부 다시 수행한다.
5. 새 192개 pair를 대상으로 AI 사전검수를 처음부터 다시 수행한다.
6. AI 사전검수 192/192 PASS 후에만 독립 reviewer 2명의 human audit로 이동한다.

설계의 feature 정의나 acceptance threshold를 이번 언어 검수 결과만으로 변경할 필요는 없다. 필요한 변경은 세 template family의 문형을 명확하게 만드는 dataset-level revision이다.

## 6. Artifact 및 SHA-256

| Artifact | SHA-256 |
|---|---|
| Frozen v2.1 design manifest | `d0d6d19432eb48234a9729cc5f297c09963eb485bf5ec02178d504a93dc8307f` |
| v2.1 dataset | `cfc5a628d75e05c17d4c4ff3907f477d6f6179892453f1e2c3927c8aeed10640` |
| v2.1 dataset manifest | `074075188f09f76d0cbae7b369dae056d12183d85e6350ed3225b101526dce72` |
| v2.1 automatic audit | `f594fd5c892473a52a93a12868d7f1dad48677d744f8eaf965e2243aa53957d1` |
| AI pre-audit implementation | `cc32e5c3a22e8a875a78f590034aabf2018ef444d10705ae30d3501e53b4a925` |
| AI pre-audit guide | `e46ed01b8de19630c8685d2345e357fa5e67ec3edee1e4bc9844d36a9a2ef197` |
| AI pre-audit CSV | `71710049e5a06bf75eb85f79fe0ed3bc65d54b38300ace89767c85641312c0b1` |
| AI pre-audit summary JSON | `54cb46010b703f45d106e22e7fd2d26e4aeac48b9382a0d879a4e080648c4bcc` |

이 결과는 `arithmetic transformation vs matched one-hop keyed retrieval` 문항의 언어·구조 사전검수 결과이며, R/M 구분 또는 LiReF/component 효과에 대한 결과가 아니다.
