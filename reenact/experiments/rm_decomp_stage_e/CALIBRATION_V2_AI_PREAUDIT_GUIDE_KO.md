# Stage E Calibration v2 AI Linguistic Pre-audit 지침

상태: **FROZEN FOR AI PRE-AUDIT — 독립 human audit 대체 불가**  
대상 dataset SHA-256: `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87`  
목적: 모델 결과를 보기 전에 v2의 144개 pair가 승인된 operationalization과 언어적 조건을 만족하는지 AI가 전수 사전검수한다.

## 1. 검수 범위

각 pair에서 다음을 확인한다.

1. Relevant 정답이 `start ± delta`와 일치한다.
2. Irrelevant 정답이 direct fact에 정확히 한 번 명시된다.
3. Relevant 정답은 context의 독립 numeric mention으로 존재하지 않는다.
4. Relevant 질문은 transformed attribute를, Irrelevant 질문은 direct attribute를 묻는다.
5. 두 조건은 동일 context를 사용하고 질문 target 외 구조가 같다.
6. 정답·대안·start·delta·wrong-operation foil이 충돌하지 않는다.
7. 문법, 시제, 개체, 단위와 질문–context 연결이 자연스럽고 모호하지 않다.
8. answer orientation, label role, sentence order counterpart가 의미적으로 동등하다.

v2에서 정답 노출은 의도적으로 동일하지 않다. `transformation-dependent answer derivation vs direct-fact retrieval`에는 다음 차이가 포함된다.

- Relevant: 정답 literal 비노출 + 단일 transformation 필요
- Irrelevant: 정답 literal 직접 노출 + direct retrieval 가능

따라서 이 검수는 해당 조작을 순수 산술 효과 또는 R/M 전체 차이로 해석하지 않는다.

## 2. Pair 판정

- 모든 항목이 `YES`이면 `AI_PREAUDIT_PASS`
- 명백한 오류가 있으면 `AI_PREAUDIT_FAIL`
- 문구 또는 설계 수정이 필요하면 `AI_PREAUDIT_NEEDS_REVISION`
- 모든 판정에는 `reviewer_id=codex_ai_preaudit_nonhuman_v2`를 기록한다.
- AI 판정은 독립 human reviewer의 판정으로 재표기하지 않는다.

## 3. Dataset 판정

아래 조건을 모두 만족해야 AI pre-audit PASS다.

- 144/144 pair 작성 완료
- `AI_PREAUDIT_FAIL=0`
- `AI_PREAUDIT_NEEDS_REVISION=0`
- 18개 template family 각각 8-frame 완전교차
- 모든 automatic pair/template check PASS
- dataset 및 manifest hash 일치

## 4. 안전 규칙

- dataset, frozen design과 official human-audit 파일을 수정하지 않는다.
- 모델 weight, forward, GPU, LiReF direction, hidden state와 candidate state를 사용하지 않는다.
- AI pre-audit 결과만으로 Baseline Calibration 실행을 승인하지 않는다.
- 문제 발견 시 기존 artifact를 덮어쓰지 않고 새 dataset version 절차로 돌아간다.

## 5. 다음 gate

AI pre-audit PASS 후에도 다음이 남는다.

1. 독립 human reviewer 2명이 각각 144개 전수 검수
2. 불일치 시 제3 reviewer 판정
3. human audit 결과와 hash 기록
4. baseline-only calibration 구현 및 정적 safety review
5. 별도의 명시적 실행 승인

