# Stage E Transformation Primitives v1.2 설계

상태: **FROZEN — LOCAL PREFLIGHT ALLOWED — INTERNAL OUTCOME MERGE BLOCKED UNTIL RELIABILITY PASS**  
동결일: `2026-08-31`

## 1. 목적

Natural Feature Discovery에서 발견한 `transformation_required=Y/N`을 유지하면서,
Transformation 내부의 어떤 구체적 operation이 Layer 31과 기존 component 반응에
연결되는지 탐색한다.

이 분석은 다음을 주장하지 않는다.

- primitive가 R/M을 인과적으로 결정한다.
- primitive가 기존 component를 매개한다.
- 기존 heldout 600이 독립 confirmatory set이다.

## 2. v1.1 실패에서 바꾸는 점

v1.1처럼 `ARITH / LOGIC / FORMAL / CAUSAL / MIXED` 중 하나만 고르게 하지 않는다.
각 문항에 아래 여섯 축을 독립적인 Y/N으로 붙인다. 여러 축이 동시에 Y일 수 있다.

1. `NUM`: 실제 수치 계산·수치 비교·단위 변환이 필요한가?
2. `RULE`: 공식·법칙·문법·법률·분류·절차 규칙을 적용하는가?
3. `REL`: 둘 이상의 주어진 관계나 정보를 결합해야 하는가?
4. `COND`: 조건·명제·부정·양화·경우 분기 논리를 적용하는가?
5. `CAUS`: 원인·기전·개입·counterfactual에서 결과를 추적하는가?
6. `INTER`: 파생된 중간 결과를 만든 뒤 그것을 다음 판단에 사용해야 하는가?

출력 문법:

```text
NUM=Y|N;RULE=Y|N;REL=Y|N;COND=Y|N;CAUS=Y|N;INTER=Y|N
```

## 3. Annotation 대상과 blind 범위

- authoritative parent `transformation_required=Y` 895문항만 local annotation
- parent N 1,739문항은 분석에서 여섯 primitive 모두 0인 no-transformation
  baseline으로 사용
- parent UNC 366문항은 제외
- annotator 입력: anonymous ID, question, options만 사용
- 미노출: 정답, R/M label, `memory_reason_score`, split, source/category,
  LiReF/component 값, 다른 annotator 출력
- annotator: local Meta-Llama-3-8B-Instruct와 local Mistral-7B-Instruct-v0.3
- 외부 API 사용 금지

## 4. Disjoint preflight

이전 v1 및 v1.1 preflight에 사용된 parent-Y 문항을 모두 제외하고 category-diverse
96문항을 seed `20260831`로 고정 추출한다.

### 형식 gate

- annotator별 parse validity `>=0.99`
- joint-valid coverage `>=0.98`

### primitive별 gate

- raw agreement `>=0.80`
- pooled positive prevalence가 `0.05–0.95`이면 Cohen κ `>=0.50`
- pooled positive prevalence가 범위 밖이면 `INSUFFICIENT_PREVALENCE`로 제외
- raw agreement 또는 κ 기준 미달이면 `UNRELIABLE`로 제외

최소 2개 primitive가 `USABLE`이어야 full annotation으로 진행한다. 하나의 primitive가
실패했다는 이유로 신뢰도 높은 다른 primitive까지 폐기하지 않지만, 실패한 primitive는
full annotation과 internal analysis에서 사용하지 않는다.

## 5. Full reliability와 consensus

- preflight에서 `USABLE`인 primitive만 full annotation 결과를 사용
- primitive별 exact two-annotator agreement만 consensus Y/N으로 사용
- disagreement는 UNC
- primitive별 joint-valid coverage `>=0.90`
- primitive별 Cohen κ `>=0.50`
- full에서도 기준을 통과한 primitive만 분석 허용
- 분석 가능한 primitive가 2개 미만이면 internal outcome 결합 중단

## 6. 분석

### Primary population

기존 Discovery 2,400문항의 parent Transformation Y 중 primitive consensus가 있는 문항.

### Primary contrast

각 primitive에 대해 parent-Y 내부에서:

> primitive Y vs primitive N

을 비교한다. R/M label, token length, option count, numeric presence, source를 공변량으로
사용한다. 이 비교는 broad Transformation Y/N 차이를 primitive 효과로 다시 포장하지
않기 위한 것이다.

Primary endpoints:

- Layer 31 frozen LiReF projection
- `L29H30` frozen LiReF-direction contribution
- `L30H6` frozen LiReF-direction contribution

Secondary endpoints:

- `L31N13336`
- `L29H31`
- parent N baseline과의 descriptive comparison
- primitive 간 co-occurrence

Discovery에서 primitive별 Y/N 각각 최소 50개가 있어야 primary test에 포함한다.
모든 usable primitive × 3 primary endpoint에 Benjamini-Hochberg FDR `q<0.05`를 적용한다.

## 7. Heldout와 stopping rule

- Discovery에서 선택된 primitive-endpoint pair만 기존 heldout 600에서 secondary check
- heldout primitive Y/N 각각 최소 15개
- 같은 방향, BH `q<0.05`, 95% CI가 0 제외일 때 `heldout-supported association`
- heldout은 taxonomy가 기존 전체 3,000문항으로 개발되므로 독립 confirmatory가 아님

Preflight FAIL 또는 full reliability FAIL이면 R/M·LiReF·component 값과 결합하지 않는다.
결과를 본 뒤 threshold를 낮추거나 문항·primitive를 사후 선택하지 않는다.

## 8. 금지

- OpenAI 또는 다른 외부 API 호출
- Meta-Llama-3-8B base의 새 forward/hidden-state 추출
- 새 neuron/head 탐색
- patching, suppression, intervention
- causal feature, mediation, general mechanism 주장

