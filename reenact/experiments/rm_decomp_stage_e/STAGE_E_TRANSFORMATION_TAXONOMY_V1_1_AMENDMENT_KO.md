# Stage E Transformation Taxonomy v1.1 Annotation Instrument Amendment

상태: **FROZEN — v1 PREFLIGHT FAIL 보존 — REVISED LOCAL PREFLIGHT ALLOWED**  
동결일: `2026-08-31`

## 1. v1 실패 원인

v1 96-item preflight는 내부 outcome을 결합하기 전에 중단됐다.

- Annotator A: 96/96을 `T=Y`로 판정
- Annotator B: 92/96을 `T=Y`로 판정
- parent T balanced accuracy: A `0.500`, B `0.521`
- subtype agreement `0.438`, kappa `0.141`
- step agreement `0.457`

새 prompt가 기존 검증된 parent `transformation_required`를 재현하지 않고
multiple-choice option 평가 전반을 Transformation으로 확장한 instrument failure다.
LiReF, component 또는 R/M outcome은 보지 않았다.

## 2. v1.1 변경 범위

Parent Transformation을 재판정하지 않는다. Natural Feature Discovery v1.2의
두 annotator exact consensus를 authoritative parent feature로 유지한다.

- parent `T=Y` 895문항만 subtype/step annotation 대상
- parent `T=N`은 분석 baseline `TYPE=NONE`, `STEPS=0`으로 고정
- parent `T=UNC`는 subtype/step 분석에서 제외
- annotator는 parent Y item만 받지만 R/M label, 정답, category/source/split,
  LiReF/component 값과 이전 subtype 판정을 보지 않음

Subtype taxonomy와 분석 endpoint/count/FDR/heldout 규칙은 v1 design에서 바꾸지
않는다.

## 3. 명확화된 subtype 규칙

- 숫자 또는 수학식을 실제로 계산·비교·변환해야 하면 공식 이름이 등장해도
  `ARITH`를 사용한다.
- `FORMAL`은 핵심 operation이 비수치적인 도메인 규칙·법칙·문법·법률 규칙·
  절차 적용일 때 사용한다.
- `CAUSAL`은 원인·기전·개입·counterfactual에서 결과를 추적할 때 사용한다.
- `LOGIC`은 그 밖의 조건·명제·부정·양화·관계 추론이다.
- `MIXED`는 둘 이상이 독립적으로 필수이고 dominant를 정할 수 없을 때만 사용한다.

Step은 conceptual dependent operation을 센다.

- 한 공식에 값을 대입해 계산하는 전체 과정: `1`
- 중간값을 구한 뒤 그 값을 다른 규칙에 적용: `2`
- 식 내부의 곱셈·덧셈 각각, option 확인, 사실 회상: 별도 step 아님

출력:

```text
TYPE=ARITH|LOGIC|FORMAL|CAUSAL|MIXED|UNC;STEPS=1|2|3P|UNC
```

## 4. Disjoint preflight와 gate

v1 preflight 48 parent-Y를 제외하고, 다른 parent-Y 96문항을 category-diverse하게
seed `20260831`로 고정 추출한다.

Hard gate:

- annotator별 parse validity `>=0.99`
- joint-valid coverage `>=0.98`
- subtype raw agreement `>=0.65`, Cohen kappa `>=0.50`
- step raw agreement `>=0.65`, quadratic weighted kappa `>=0.55`
- annotator별 한 subtype 또는 step level 최대 비율 `<=0.95`

하나라도 실패하면 전수 annotation과 internal outcome 결합을 실행하지 않는다.

## 5. Full consensus와 reliability

- subtype과 steps 각각 exact two-annotator agreement만 사용
- disagreement/UNC는 해당 field `UNC`
- full joint-valid coverage `>=0.90`
- subtype kappa `>=0.50`
- step weighted kappa `>=0.55`

모두 통과해야 기존 frozen scalar와 결합한다. v1의 failed preflight 및 code를
덮어쓰지 않으며 v1.1 파일명을 사용한다.

## 6. 변하지 않는 제한

- AI-only annotation
- 기존 heldout 600은 secondary check이며 독립 confirmatory set이 아님
- 외부 API, 새 study-model forward, hidden-state 저장, 후보 탐색, intervention 금지
- association을 causal feature 또는 component mediation으로 표현 금지

