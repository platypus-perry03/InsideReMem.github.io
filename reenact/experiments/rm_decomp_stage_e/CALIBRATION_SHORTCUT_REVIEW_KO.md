# Stage E Calibration Shortcut 검토

상태: **BLOCKING DESIGN ISSUE — 현재 dataset으로 다음 gate 진행 금지**

## 발견된 문제

현재 모든 pair는 Relevant와 Irrelevant의 canonical answer를 같게 만들기 위해 다음 구조를 사용한다.

```text
직접 사실: red = 5
변환 사실: blue = 8 - 3
Relevant 질문: blue?       정답 5
Irrelevant 질문: red?      정답 5
```

의미적으로 Relevant 질문은 `8 - 3` 변환을 요구한다. 그러나 동일 context의 직접 사실에 정답 `5`가 그대로 적혀 있다. 따라서 모델은 relation transformation을 수행하지 않고 직접 사실의 숫자를 복사해도 정답을 낼 수 있다.

이 문제는 개별 문장의 오타가 아니라 144개 전체에 적용된 dataset-level shortcut이다.

## 기존 자동 검사와의 관계

다음 검사는 여전히 모두 통과한다.

- 계산 정답 일치
- pair 내 context 동일
- 질문 target 외 표면 구조 동일
- token 길이 일치
- counterbalance 적용
- 문법과 relation 방향의 명백한 오류 없음

하지만 이 조건들은 `Relevant에서 transformation이 실제로 필요하다`는 것을 보장하지 않는다. `answer_literal_exposure_equal` 규칙은 노출량을 맞추는 대신 정답 복사 경로를 만들었다.

## 연구 해석에 미치는 영향

현재 dataset에서 Relevant–Irrelevant 차이가 관찰되어도 다음을 구분하기 어렵다.

1. relation transformation의 task relevance
2. 질문 target label의 변경
3. 직접 사실에서 정답 숫자를 복사하는 전략
4. context 안에서 두 attribute를 binding하는 난이도

따라서 이 dataset으로 Calibration을 통과하더라도 Stage E의 primary feature를 깨끗하게 operationalize했다고 보기 어렵다.

## AI 사전검수 수정 판정

- 기존 6개 문장·계산 항목: 144/144 통과
- cross-fact answer-copy shortcut: 144/144에서 존재
- 종합 AI 판정: `AI_PREAUDIT_NEEDS_REVISION` 144개
- 독립 human audit 상태: 여전히 `PENDING`
- Baseline Calibration 실행: 금지 유지

## 수정 시 필요한 원칙

이 문제는 기존 dataset을 일부 고쳐 해결할 수 없다. 아래 frozen invariant가 서로 충돌하기 때문이다.

- pair의 context 동일
- pair의 canonical answer 동일
- Irrelevant 정답은 context의 직접 사실에서 조회
- 두 조건에서 answer literal exposure 동일

따라서 기존 dataset과 hash는 보존하고, feature operationalization을 재설계한 새 calibration design과 새 dataset version을 만들어야 한다. 결과를 본 뒤 문항을 선택적으로 삭제하는 방식은 사용하지 않는다.

새 설계를 동결하기 전에는 어떤 수정안도 공식 채택하지 않으며, 모델 forward·GPU·LiReF·Calibration·Pilot을 실행하지 않는다.
