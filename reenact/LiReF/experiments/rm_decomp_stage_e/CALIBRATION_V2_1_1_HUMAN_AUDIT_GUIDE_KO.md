# Stage E Calibration v2.1.1 독립 Human Linguistic Audit 지침

상태: **HUMAN AUDIT PENDING — reviewer 2명 각각 192/192 검수 필요**

## 1. 독립성과 blind 원칙

- Primary reviewer는 2명이며 각 reviewer가 192개 pair를 모두 검수한다.
- reviewer는 자신의 실제 식별자를 `reviewer_id`에 기록한다.
- 두 reviewer는 서로의 파일과 판정을 보지 않는다.
- reviewer에게 AI pre-audit, model prediction, logP, generation, LiReF, component, intervention 결과를 제공하지 않는다.
- 모델 forward와 GPU 실행 없이 문항 자체만 검수한다.
- AI pre-audit PASS는 참고하거나 복사하지 않고 각 항목을 독립적으로 판정한다.

## 2. Pair별 검수 항목

각 YES/NO 열을 빠짐없이 작성한다.

1. `arithmetic_answer_correct`: start와 delta로 Relevant answer가 정확히 계산되는가?
2. `direct_answer_correct`: Irrelevant answer가 선택된 direct record에 정확히 기재되어 있는가?
3. `label_key_record_binding_clear`: label → key → record의 연결이 명확하고 유일한가?
4. `relevant_transformation_required`: Relevant 질문은 올바른 record를 찾은 뒤 실제 덧셈/뺄셈이 필요한가?
5. `irrelevant_keyed_retrieval_valid`: Irrelevant 질문은 동일한 수준의 key binding 뒤 current value를 직접 조회하는가?
6. `answer_copy_shortcut_absent`: Relevant answer와 같은 숫자가 context에 직접 노출되지 않는가?
7. `question_target_only_change`: 두 질문은 target label 이외의 의미와 형식이 같은가?
8. `natural_grammar_and_unambiguous_direction`: 문법이 자연스럽고 증가/감소 방향 및 의미 주체가 명확한가?
9. `counterbalance_counterpart_equivalent`: 같은 template family의 8개 frame이 이름·label/key 역할·record 순서·A/B 배치 외에는 의미적으로 동등한가?
10. `output_instruction_clear`: `Answer with one Arabic numeral only.`가 명확하고 canonical answer가 그 형식에 맞는가?

## 3. review status

- 10개 항목이 모두 `YES`이면 `review_status=PASS`
- 명백히 사용할 수 없는 오류이면 `review_status=FAIL`
- 문형 또는 의미 수정이 필요하면 `review_status=NEEDS_REVISION`
- `FAIL` 또는 `NEEDS_REVISION`에는 `comments`에 구체적인 사유를 반드시 기록한다.
- 애매한 경우 임의로 PASS하지 말고 `NEEDS_REVISION`으로 판정한다.

## 4. Dataset gate

- 한 pair는 reviewer 1과 reviewer 2가 모두 PASS해야 최종 PASS다.
- 두 reviewer의 판정이 다르면 제3 independent reviewer가 blind adjudication하고 사유를 기록한다.
- 최종 `FAIL` 또는 `NEEDS_REVISION`이 하나라도 있으면 v2.1.1 dataset 실행은 금지한다.
- 수정이 필요하면 현재 dataset을 덮어쓰지 않고 새 dataset version을 생성해 automatic audit, AI pre-audit, human audit을 처음부터 다시 수행한다.
- 두 reviewer의 192/192 검수와 필요한 adjudication이 끝나기 전에는 Baseline Calibration 및 Stage E Pilot을 실행하지 않는다.

## 5. Reviewer에게 전달할 파일

- reviewer 1: `human_audit/reviewer_1/calibration_v2_1_1_human_audit_reviewer_1.csv`
- reviewer 2: `human_audit/reviewer_2/calibration_v2_1_1_human_audit_reviewer_2.csv`

각 reviewer에게는 이 지침과 자신의 CSV만 전달한다. `ai_preaudit/` 디렉터리는 전달하지 않는다.
