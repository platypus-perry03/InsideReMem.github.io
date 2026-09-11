# Stage E Calibration v2.1.1 AI-only Audit 지침

상태: **ACTIVE UNDER FROZEN AI-ONLY AMENDMENT**

## 검수 단위

- 192개 pair 전수
- 24개 template family 전수
- 각 template의 2 × 2 × 2 counterbalance 8개 frame 전수
- 모델 weight, prediction, logP, LiReF와 exploratory result를 검수 입력으로 사용하지 않음

## Pair 판정

각 pair에서 다음 10개 열을 `YES/NO`로 기록한다.

1. `arithmetic_answer_correct`
2. `direct_answer_correct`
3. `label_key_record_binding_clear`
4. `relevant_transformation_required`
5. `irrelevant_keyed_retrieval_valid`
6. `answer_copy_shortcut_absent`
7. `question_target_only_change`
8. `natural_grammar_and_unambiguous_direction`
9. `counterbalance_counterpart_equivalent`
10. `output_instruction_clear`

10개가 모두 `YES`인 경우에만 `AI_AUDIT_PASS`다. 구조 오류는 `AI_AUDIT_FAIL`, 문형 수정이 필요하거나 의미가 애매하면 `AI_AUDIT_NEEDS_REVISION`으로 기록한다.

## 다중 pass

- Pass A: 계산, answer exposure, binding uniqueness, question diff, output contract를 deterministic하게 검사
- Pass B: 각 template의 실제 문장과 8개 role/order/value 반전을 검토
- Pass C: 기존 pre-audit 및 adversarial review artifact의 hash와 결과 일치 여부를 확인

이 pass들은 독립 인간 검수가 아니며 AI 내부 교차검사다.

## Dataset gate

- 192/192 `AI_AUDIT_PASS`
- `FAIL=0`
- `NEEDS_REVISION=0`
- 24/24 template semantic/counterbalance PASS
- locked artifact hash 전부 일치

모두 만족해야 `ai_only_audit_gate=PASS`다. 그 뒤에도 `baseline_calibration_execution_allowed=false`를 유지하며 별도 실행 승인 전에는 모델을 로딩하지 않는다.
