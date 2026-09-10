# Stage E Calibration v2.1 AI Pre-audit 지침

상태: **FROZEN FOR AI PRE-AUDIT — independent human audit 대체 불가**  
대상 dataset SHA-256: `cfc5a628d75e05c17d4c4ff3907f477d6f6179892453f1e2c3927c8aeed10640`  
목적: model result를 보기 전에 v2.1의 192개 pair가 frozen keyed-retrieval operationalization과 언어적 조건을 만족하는지 AI가 전수 사전검수한다.

## 1. Pair 검수 항목

1. Relevant 정답이 frozen operation의 `start ± delta`와 일치한다.
2. target label에서 올바른 arbitrary record key로 binding할 수 있다.
3. Relevant record에는 start/delta가 있고 정답 literal은 context에 없다.
4. Irrelevant record에는 canonical answer가 정확히 한 번 직접 기재된다.
5. 다른 숫자나 record를 복사해서 Relevant 정답을 맞힐 수 없다.
6. 두 조건의 context는 같고 question target 외 문형이 같다.
7. A/B answer와 paired alternative가 올바르게 교환된다.
8. output instruction이 두 조건에서 동일하고 숫자 한 token만 요구한다.
9. 문법·시제·변화 방향과 record의 의미가 자연스럽고 모호하지 않다.
10. 같은 template의 8개 counterbalance frame이 의미적으로 동등하다.

## 2. 판정

- 모든 항목이 명확하면 `AI_PREAUDIT_PASS`
- 계산·binding·shortcut 등 명백한 오류가 있으면 `AI_PREAUDIT_FAIL`
- 의도는 추정할 수 있지만 문법이나 변화 방향을 수정해야 하면 `AI_PREAUDIT_NEEDS_REVISION`
- reviewer ID는 `codex_ai_preaudit_nonhuman_v2_1`로 고정한다.
- AI 결과를 human reviewer 판정으로 재표기하지 않는다.

## 3. Dataset gate

AI pre-audit PASS에는 다음이 모두 필요하다.

- 192/192 pair 판정 완료
- `FAIL=0`
- `NEEDS_REVISION=0`
- 24개 template family의 8-frame 완전교차 검토
- locked dataset/manifest/automatic-audit hash 일치

하나라도 `FAIL` 또는 `NEEDS_REVISION`이면 현재 dataset을 human audit이나 Baseline Calibration으로 넘기지 않는다. 기존 artifact를 고치지 않고 새 dataset version을 만들어 automatic audit과 AI pre-audit을 처음부터 반복한다.

## 4. Blindness와 안전

- v2.1 model prediction, logP, generation, LiReF와 component/intervention 결과를 읽지 않는다.
- model/tokenizer를 로드하지 않고 frozen JSON만 읽는다.
- dataset, design과 official human-audit 파일을 수정하지 않는다.
- AI pre-audit만으로 human gate나 execution gate를 통과시키지 않는다.
- model weight/forward/GPU/hook/hidden state를 사용하지 않는다.

## 5. 주장 범위

AI PASS는 `arithmetic value transformation vs matched one-hop keyed fact retrieval` 문항의 사전 언어·구조 검수 통과만 의미한다. Reasoning/Memorization 분리나 LiReF 연결을 입증하지 않는다.
