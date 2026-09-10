# Stage E Transformation Observable Stratification v1 설계

상태: **FROZEN — PRESERVED-SCALAR ANALYSIS ALLOWED**  
동결일: `2026-08-31`

## 목적

새 AI annotation 없이 기존 `transformation_required`가 사실상 숫자 계산 문항을
표시하는 것인지, 숫자가 있는 문항 안에서도 Layer 31과 기존 component 반응을
추가로 구분하는지 확인한다.

## 사전 관찰과 분석 범위

기존 3,000문항의 deterministic `has_numeric` 기준에서 Transformation Y는 숫자 문항에
강하게 집중되어 있다. 이 count는 설계 동결 전에 확인한 descriptive diagnostic이다.

- 전체 Transformation Y: 895
- 그중 숫자 포함: 869
- 숫자 미포함: 26

따라서 non-numeric Transformation을 논리·규칙 변환의 일반 증거로 사용하지 않는다.

## Primary analysis

- population: `has_numeric=1`
- feature: 기존 two-annotator consensus `transformation_required`
- Discovery 2,400에서 먼저 분석
- endpoints: Layer 31, `L29H30`, `L30H6`
- R/M label, log token length, option count, source 조정
- 각 Transformation level 최소 100문항
- 세 endpoint에 BH FDR `q<0.05`
- Discovery에서 선택된 endpoint만 기존 heldout 600에서 확인
- heldout 각 level 최소 50문항, 같은 방향, BH `q<0.05`, CI가 0 제외

이 분석이 유지되면 `Transformation association이 단순한 숫자 문자 존재만으로 설명되지
않는다`고 말할 수 있다. 계산·규칙·논리 중 무엇이 원인인지는 말할 수 없다.

## Secondary descriptive analysis

- numeric/non-numeric × Transformation count
- category × Transformation count/prevalence
- non-numeric subset의 endpoint estimate는 표로 저장하되 Y가 26개뿐이므로 추론적
  결론이나 feature 선택에 사용하지 않음
- `L31N13336`, `L29H31`은 descriptive endpoint

## 제한

- post-discovery observational analysis
- 기존 heldout은 독립 confirmatory가 아님
- 외부 API, 새 model forward, hidden state, 후보 탐색, intervention 없음
- causal feature, mediation, broad R/M mechanism 주장 금지

