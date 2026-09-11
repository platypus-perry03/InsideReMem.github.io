# Stage E Transformation Primitives v1.3 Forced-choice Amendment

상태: **FROZEN — v1.2 PREFLIGHT FAIL 보존 — LOCAL FORCED-CHOICE PREFLIGHT ALLOWED**  
동결일: `2026-08-31`

## 1. 수정 이유

v1.2 multi-label preflight에서 자유 생성 형식이 실패했다.

- Annotator A parse validity: 92/96
- Annotator B parse validity: 2/96
- Annotator B는 주로 `NUM=Y|RULE=Y|...`처럼 separator를 잘못 출력
- 공동 유효 문항: 2/96

따라서 v1.2 결과로 primitive 신뢰도를 평가할 수 없었다. v1.2 실패 기록과 threshold는
변경하지 않는다.

## 2. v1.3 변경점

여섯 primitive 정의, annotation 대상, blind 범위, 분석·통계·해석 규칙은 v1.2와
동일하다. 바뀌는 것은 응답 측정 방식뿐이다.

각 문항과 primitive를 별도의 binary 질문으로 제시하고, 자유 텍스트를 생성하지 않는다.
마지막 위치에서 local annotator가 single-token `Y`와 `N`에 부여한 logit을 비교한다.

```text
choice = Y if logit(Y) > logit(N), else N
margin = logit(Y) - logit(N)
```

- sampling/generation/parser/retry 없음
- 동일 문항에서 여섯 primitive가 동시에 Y일 수 있음
- tie는 N으로 판정하고 별도 count 기록
- margin은 annotation 진단으로만 저장하며 study-model endpoint와 혼동하지 않음

## 3. Preflight

v1, v1.1, v1.2 preflight에 사용된 parent-Y 문항을 모두 제외한 새 96문항을
seed `20260831`로 category-diverse 추출한다.

primitive별 gate:

- coverage `=1.00`
- raw agreement `>=0.80`
- pooled positive prevalence `0.05–0.95`
- Cohen κ `>=0.50`

prevalence 범위 밖 primitive는 `INSUFFICIENT_PREVALENCE`, agreement/κ 미달은
`UNRELIABLE`로 제외한다. 최소 2개 primitive가 `USABLE`이어야 full annotation 진행.

## 4. Full reliability와 분석

v1.2 규칙을 그대로 유지한다.

- preflight USABLE primitive만 full 결과 사용
- exact two-annotator choice agreement만 consensus
- primitive별 coverage `>=0.90`, κ `>=0.50`
- 최소 2개가 full reliability PASS해야 internal outcome 결합
- primary는 parent Transformation Y 내부 primitive Y vs N
- R/M label과 기존 covariates 조정
- Discovery selection 후에만 기존 heldout secondary check

## 5. 금지 및 해석

- 외부 API 금지
- Meta-Llama-3-8B base study-model 새 forward 금지
- hidden state, 새 후보 탐색, intervention 금지
- 결과는 post-discovery association이며 causal feature/mediation 증거가 아님

