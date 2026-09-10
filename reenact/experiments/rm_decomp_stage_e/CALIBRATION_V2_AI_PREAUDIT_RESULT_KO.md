# Stage E Calibration v2 AI Pre-audit 결과

상태: **AI PRE-AUDIT PASS — INDEPENDENT HUMAN AUDIT PENDING**  
날짜: 2026-08-29  
검수자 표기: `codex_ai_preaudit_nonhuman_v2`

## 결과

| 항목 | 결과 |
|---|---:|
| 전체 pair | 144 |
| AI pre-audit PASS | 144 |
| FAIL | 0 |
| NEEDS_REVISION | 0 |
| Template family | 18 |
| Counterbalance 실패 | 0 |

전수검사에서 다음을 확인했다.

- 증가·감소 계산과 canonical answer 일치
- Relevant 정답 literal이 context에 없음
- Relevant에서 transformed attribute의 단일 연산 필요
- Irrelevant 정답은 direct fact에 정확히 한 번 존재
- v1의 cross-fact answer-copy shortcut 없음
- Relevant/Irrelevant는 동일 context를 사용하고 질문 target만 변경
- answer orientation, label role, sentence order의 2×2×2 완전교차
- 각 template에서 answer value와 magnitude 방향 균형
- 정답·대안·operand·foil 충돌 없음
- 18개 template의 문법, 시제, 단위와 질문–context 연결에 명백한 오류 없음

`v2_temp_oven_climbed`의 “temperature climbed” 표현은 다른 temperature template보다 덜 중립적으로 들릴 수 있으나 문법적이고 변화 방향이 명확하여 AI pre-audit에서는 PASS로 판정했다. 독립 human audit에서는 이 표현의 자연스러움을 다시 확인한다.

## Artifact hash

| Artifact | SHA-256 |
|---|---|
| Frozen dataset | `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87` |
| Dataset manifest | `f79a80aa7b55a546724a3f639ebed22858addc39b21aaa73c65b495a9ce3898e` |
| Automatic audit | `6e7bf7bde3965ae56f9d41cf0c82fec1b2da0414d190f32c19df45995d38ac8e` |
| AI pre-audit script | `06701674615e83c4de8aade72d110c1ad8d24d17be44fa067d13f83cd75c932f` |
| AI pre-audit guide | `30ed2105d66852f0bd2d483a3365bdc23848779235036471f97debdc099ae82b` |
| Pair-level AI pre-audit CSV | `e562910e37d88d60ca2e37acfe83fc4cfa51a49efe327dcef707cd35aacaff27` |
| AI pre-audit summary | `2bd694f7cf9331391fa89c1d5635105679590d09bd261729b6ebb3866545623a` |

## 해석 제한

이번 PASS는 문항 구조와 언어적 타당성에 대한 **비인간 사전검수** 결과다. 아래를 의미하지 않는다.

- 독립 human audit 완료
- Baseline Calibration 실행 승인
- Stage E feature 또는 pathway 검증 성공
- direct-fact retrieval을 Memorization 전체와 동일시

공식 조작명은 `transformation-dependent answer derivation vs direct-fact retrieval`이다. 이 feature가 LiReF R/M 방향과 연결되는지는 이후 Calibration과 Stage E 실험에서 별도로 확인해야 한다.

## 다음 gate

독립 human reviewer 2명이 각각 144개 pair 전부를 검수해야 한다. 그전까지 모델 forward, GPU, Baseline Calibration과 Stage E Pilot은 실행하지 않는다.
