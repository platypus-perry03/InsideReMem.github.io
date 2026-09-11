# Stage E Baseline Calibration v2.1.1 Dataset 및 AI 사전검수 결과

> **후속 상태 정정:** 아래 상태는 dataset 생성 직후의 역사적 기록이다. 독립 human audit은 이후에도 수행되지 않았으며, 이 dataset으로 실행된 exploratory 및 baseline run은 protocol-deviating 진단/가설 생성 결과로만 취급한다. official Calibration은 수행되지 않았다. 현재 authoritative status는 `STAGE_E_CURRENT_STATUS_KO.md`를 따른다.

상태: **AI PRE-AUDIT 192/192 PASS — INDEPENDENT HUMAN AUDIT PENDING**  
작성일: 2026-08-30  
dataset version: `2.1.1`

## 1. Revision 범위

v2.1 frozen design, threshold, counterbalance, scoring 및 human-audit 정책은 변경하지 않았다. 기존 v2.1 builder와 dataset도 수정하지 않았다.

AI 사전검수에서 `NEEDS_REVISION`이었던 다음 세 transformed 문형만 새 dataset version에서 교체했다.

| Template family | v2.1.1 문형 |
|---|---|
| `v21_dec_badge_index` | `... notes that {delta} badges were given away.` |
| `v21_temp_reservoir_register` | `... after which the recorded temperature rose by {delta} degrees.` |
| `v21_temp_capsule_account` | `... after which the recorded temperature increased by {delta} degrees.` |

새 pair ID는 `calv211_` prefix를 사용한다. Parent v2.1과 비교하면 168개 pair의 문항 내용은 동일하고, 정확히 위 3 template × 8 frames에 해당하는 24개 pair의 문형만 변경되었다.

## 2. Dataset 생성 및 automatic audit

- 3 lexical families
- family당 8 template families
- template당 2×2×2 완전교차 8 frames
- 총 192 pairs / 384 prompts / 24 template-family clusters
- random seed: `20260830`
- 두 임시 디렉터리에서 독립 생성한 4개 artifact가 모두 byte-identical
- 공식 output도 검증본과 동일한 dataset/audit/manifest hash

Automatic audit 결과:

```text
pair checks:             192/192 PASS
template counterbalance: 24/24 PASS
near-duplicate audit:    PASS
one-token answer:        192/192 PASS
dataset_pass:            true
```

`trust_remote_code=false`, `local_files_only=true`로 locked tokenizer만 사용했다. tokenizer-only 격리 환경에는 PyTorch가 설치되어 있지 않았으며 model weight, forward, GPU, LiReF, hidden state, candidate state, hook 또는 intervention을 사용하지 않았다.

## 3. AI linguistic pre-audit

reviewer ID: `codex_ai_preaudit_nonhuman_v2_1_1`

| 판정 | Pair 수 |
|---|---:|
| `AI_PREAUDIT_PASS` | 192 |
| `AI_PREAUDIT_NEEDS_REVISION` | 0 |
| `AI_PREAUDIT_FAIL` | 0 |

192개 pair 전체에서 다음 항목을 다시 검사했다.

- 산술 canonical answer 정확성
- 동일 context와 prompt/output contract
- 질문 target 외 구조 불변
- label → key → record → value binding
- Relevant transformation 필요성
- Irrelevant matched keyed direct retrieval
- Relevant answer-copy shortcut 부재
- paired-alternative orientation
- one-numeral output instruction
- 자연스러운 문법과 명확한 증가·감소 방향
- 24개 template의 counterbalance counterpart 동등성

세 수정 template도 각각 8개 frame 전부 PASS했다. AI 사전검수는 독립 human audit를 대체하지 않는다.

## 4. 현재 gate

현재 공식 상태는 다음과 같다.

```text
automatic_audit: PASS
ai_preaudit: 192/192 PASS
independent_human_audit: PENDING
human_audit_waiver_allowed: false
baseline_calibration_execution_allowed: false
stage_e_pilot_allowed: false
```

독립 reviewer 2명이 각각 192/192 pair를 blind 검수해야 한다. 각 pair는 두 reviewer가 모두 PASS해야 최종 PASS이며, 불일치가 있으면 제3 independent reviewer가 blind adjudication한다. 최종 `FAIL` 또는 `NEEDS_REVISION`이 하나라도 있으면 dataset 실행은 금지된다.

## 5. Human-audit handoff

두 reviewer용 CSV를 별도로 생성했다.

- 각 CSV: 192행 / 192개 고유 pair / 전부 `PENDING`
- `reviewer_id`: 전부 공란
- AI pre-audit 및 model 결과 관련 열: 없음
- reviewer 1과 reviewer 2는 서로의 파일과 판정을 보지 않음

reviewer에게는 human-audit 지침과 자신의 CSV만 전달하며 `ai_preaudit/` artifact는 제공하지 않는다.

## 6. Artifact 및 SHA-256

| Artifact | SHA-256 |
|---|---|
| Frozen v2.1 design manifest | `d0d6d19432eb48234a9729cc5f297c09963eb485bf5ec02178d504a93dc8307f` |
| v2.1.1 revision record | `a899df454b12f1d84719a2a8d1974c023672c5c3598a4e64b070a5353e7fa087` |
| v2.1.1 builder | `271bbc94027c2f122a2202710b2beb4857003090b53b14ba4f98681b5e3d6776` |
| v2.1.1 builder verification | `4f599b975c01e0e8a0528ab3d39485e4ad26c0740ea0e70e2fc9da5ed5fd550e` |
| v2.1.1 static schema check | `40685f6aece78468f0b5bbfaffe725d9b3a523e2985be37ef66c7c1dedeb9657` |
| v2.1.1 dataset | `322d80f3c2c0723f6a4e8b0a968b30baac23e28cf68b6c60c5a7a95a5bca7420` |
| v2.1.1 automatic audit | `93cd89e6901772c87435eb9f208bd4fc310e032b660c7678c34acf044f348e3b` |
| v2.1.1 dataset manifest | `d78264b2a9742c73ad82b6ee8ec28e6d1da7ad98c7261d46dfc1fb80bc2386db` |
| v2.1.1 AI pre-audit implementation | `06d3d13ef0ed8b1fb6c37f540c2ced7d779a46375d742ce836e1d917927d0745` |
| v2.1.1 AI pre-audit CSV | `0c6de07362252dc3bce44379c81c8ca74558c455e34bd5e95fe9fd56f059ef88` |
| v2.1.1 AI pre-audit summary | `3cf6d5645134f66692395d17297eb445194565e66504ed3d14ef382fded4d339` |
| Human-audit guide | `d6f37c8f48381624f79b271e18b93feb6d1b4edf040413d997e9df826adf82ae` |
| Human-audit worksheet preparation | `9e21a9d7299a02963bb81a2cdfdec514def7653b985b5c23d1511db95fedc70b` |
| Reviewer 1 blank CSV | `90cb15cd4294513986c51c06928611fc3d9273b64ee5586f65528f4bc2082ed7` |
| Reviewer 2 blank CSV | `90cb15cd4294513986c51c06928611fc3d9273b64ee5586f65528f4bc2082ed7` |
| Human-audit handoff manifest | `9f5bfdbf9423ae1caf02b7bc533477427ec4602c7290413b8b00bb0f2a3d401e` |

이 결과는 `arithmetic transformation vs matched one-hop keyed retrieval` 통제 문항의 dataset 및 언어 검증 결과다. 아직 R/M, LiReF projection 또는 component/pathway 효과를 측정한 결과가 아니다.
