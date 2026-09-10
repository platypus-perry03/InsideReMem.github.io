# Stage E Transformation Taxonomy v1.1 사전검수 결과

상태: **COMPLETE — PREFLIGHT FAIL — FULL ANNOTATION BLOCKED**  
실행일: `2026-08-31`  
run ID: `stagee_transformation_taxonomy_v1_1_20260831_01`

## 1. 목적과 범위

기존 Natural Feature Discovery v1.2에서 두 로컬 annotator가 exact consensus로
확정한 `transformation_required=Y` 895문항만 대상으로 Transformation subtype과
conceptual step 수를 세분화하기 위한 도구를 검사했다.

- subtype: `ARITH / LOGIC / FORMAL / CAUSAL / MIXED`
- step: `1 / 2 / 3+`
- preflight: 이전 v1 parent-Y preflight와 겹치지 않는 96문항
- annotator: local Meta-Llama-3-8B-Instruct, local Mistral-7B-Instruct-v0.3
- annotator 입력: 익명 ID, question, options만 사용
- R/M label, `memory_reason_score`, LiReF, component scalar, split/source/category: 미노출
- 외부 API: 미사용

## 2. Frozen gate 결과

| 기준 | 결과 | 판정 |
|---|---:|---|
| Annotator A parse validity | 96/96 = 1.000 | PASS |
| Annotator B parse validity | 43/96 = 0.448 | **FAIL** |
| joint-valid coverage | 43/96 = 0.448 | **FAIL** |
| subtype raw agreement | 0.651 | PASS |
| subtype Cohen κ | 0.135 | **FAIL** |
| step raw agreement | 0.209 | **FAIL** |
| step weighted κ | 0.299 | **FAIL** |
| non-degenerate output | PASS | PASS |

사전 동결한 모든 gate를 통과하지 못했으므로 `full_annotation_allowed=false`다.

## 3. 관찰된 실패 양상

- Annotator A subtype: ARITH 68, MIXED 22, LOGIC 3, CAUSAL 3
- Annotator A step: 1단계 18, 2단계 71, 3+단계 7
- Annotator B의 valid subtype: ARITH 33, MIXED 10
- Annotator B의 valid step: 1단계 13, 3+단계 30
- Annotator B의 53개 invalid 출력은 주로 `TYPE=ARITH|LOGIC`처럼 단일 dominant
  subtype 대신 복수 subtype을 반환한 경우였다.

이는 단순 parser 문제만은 아니다. 두 annotator는 유효 출력에서도 conceptual
step을 주로 `2`와 `3+`로 다르게 해석했고 subtype κ도 낮았다. 따라서 invalid
복수 subtype을 사후에 `MIXED`로 재코딩해 gate를 우회하지 않는다.

## 4. Stopping rule 적용

- 895문항 full annotation: **미실행**
- 기존 R/M label 및 `memory_reason_score`와 결합: **미실행**
- Layer 31, `L29H30`, `L30H6` 및 secondary component 분석: **미실행**
- heldout 분석: **미실행**
- intervention: **미실행**

따라서 이 결과는 `transformation_required` 가설의 실패가 아니라,
**AI-only subtype/step annotation instrument가 충분히 신뢰롭지 않았다는 실패**다.

## 5. 현재 authoritative status

> `transformation_required` Y/N association은 기존 결과로 유지되지만, 그 세부
> subtype과 단계 수는 신뢰성 기준을 통과하지 못해 아직 분석할 수 없다.

새 시도를 한다면 현재 v1.1 결과를 수정하거나 threshold를 낮추지 않고 별도
v1.2 annotation design으로 시작해야 한다.

