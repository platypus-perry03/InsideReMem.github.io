# Stage E Calibration v2 독립 Human Audit Waiver

상태: **USER-WAIVED PROTOCOL DEVIATION — HUMAN AUDIT NOT PERFORMED**  
기록일: 2026-08-29  
결정 주체: 사용자 명시적 결정(`사람 검수는 건너 뛸래`)

## 1. 결정 내용

Calibration v2 frozen design이 요구한 독립 human reviewer 2명의 144-pair 전수검수를 수행하지 않고 다음 준비 단계로 이동한다.

이 결정은 다음을 의미하지 않는다.

- human audit PASS
- 독립 reviewer 승인 완료
- linguistic validity의 독립 검증 완료
- Baseline Calibration 실행 자동 승인

공식 기록에서는 human audit 상태를 계속 `not_performed`로 유지하고, gate는 `passed`가 아니라 `waived_by_user`로 구분한다.

## 2. 보존되는 근거

| Artifact | SHA-256 |
|---|---|
| Frozen v2 design | `a8f3dad7fced945377194074f9aa12d673faff3b55c3ec45bd82e397b5a5302b` |
| Dataset draft | `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87` |
| Dataset manifest | `f79a80aa7b55a546724a3f639ebed22858addc39b21aaa73c65b495a9ce3898e` |
| Automatic audit | `6e7bf7bde3965ae56f9d41cf0c82fec1b2da0414d190f32c19df45995d38ac8e` |
| AI pre-audit CSV | `e562910e37d88d60ca2e37acfe83fc4cfa51a49efe327dcef707cd35aacaff27` |
| AI pre-audit summary | `2bd694f7cf9331391fa89c1d5635105679590d09bd261729b6ebb3866545623a` |

현재 확보된 검수는 automatic audit PASS와 비인간 AI pre-audit 144/144 PASS다. 이는 독립 human audit를 대체하지 않는다.

## 3. 연구상 제한

- 문법·자연스러움·의미적 동등성에 대한 독립 검증이 없다.
- 발견되지 않은 template-level linguistic confound 가능성이 남는다.
- 이후 결과는 human-audited confirmatory evidence라고 표현할 수 없다.
- 논문·보고서에는 human audit 미수행과 본 waiver를 protocol deviation으로 공개한다.
- 가능한 경우 최종 Confirmatory 이전에 독립 human audit를 복구하는 것을 권장한다.

## 4. 다음 허용 단계

다음으로 허용되는 작업은 **baseline-only calibration implementation 작성과 정적 safety review**다.

아직 금지되는 작업:

- Baseline Calibration 실행
- 모델 forward 또는 GPU 실행
- LiReF/hidden state/candidate state 접근
- Stage E Pilot/Confirmatory 실행

실제 Baseline Calibration은 구현 hash, dataset hash, safety review 결과와 이 waiver를 함께 확인한 뒤 사용자가 별도로 명시적 실행 승인해야 한다.

