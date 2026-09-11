# Stage E Transformation Taxonomy v1 설계

상태: **FROZEN — LOCAL BLIND ANNOTATION/ANALYSIS ALLOWED — EXTERNAL API·INTERVENTION 금지**  
동결일: `2026-08-31`

## 1. 목적과 지위

Natural Feature Discovery v1에서 `transformation_required`는 기존 heldout에서도
R label, Layer 31 LiReF 및 기존 component 반응과 함께 증가했다. 그러나
Transformation과 R label이 거의 중첩되어 어떤 종류의 변환이 신호를 설명하는지
알 수 없었다.

이번 분석은 기존 MMLU-Pro 3,000문항을 다시 사용해 다음을 탐색한다.

> R/M label을 통제한 뒤에도 산술·논리·형식 규칙·인과 변환의 종류와 변환 단계
> 수가 Layer 31 및 사전 지정 component 반응과 연관되는가?

이 분석은 parent feature를 세분화하는 **post-discovery taxonomy analysis**다.
기존 heldout 600은 subtype 정의 이전에 내부 결과가 전혀 사용되지 않은 새
독립 dataset이 아니므로 `secondary heldout replication check`로만 표현한다.
causal, mediation, confirmatory 또는 cross-model evidence로 표현하지 않는다.

## 2. 고정 데이터와 endpoint

- 문항: 기존 `mmlu-pro-3000samples.json` 3,000문항
- split: 기존 Discovery 2,400 / heldout 600을 변경 없이 사용
- parent Transformation: Natural Feature Discovery v1.2의 두 annotator exact
  consensus를 보존
- R/M: 저장된 `memory_reason_score > 0.5` 기준을 분석 단계에서만 사용
- OpenAI/GPT-4o API: 새 호출 금지
- study model forward: 새 실행하지 않음
- 기존 보존 scalar만 사용

Primary internal endpoints:

1. frozen Layer 31 LiReF projection
2. `L29H00030` frozen LiReF-direction contribution
3. `L30H00006` frozen LiReF-direction contribution

Secondary endpoints:

- `L31N13336`
- `L29H00031`
- R/M label prevalence는 설명용으로 보고하되 internal primary test에 포함하지 않음

## 3. Blind taxonomy codebook

Annotator에게 제공하는 값은 `question`, `options`, 익명 ID뿐이다. R/M label,
정답, category/source/split, 기존 Transformation 판정, LiReF/component 값과 이전
결과는 제공하지 않는다.

### 3.1 Transformation 여부

- `Y`: prompt의 정보에 arithmetic, logical, formal/rule-based 또는 causal
  operation을 적용해야 답을 얻을 수 있음
- `N`: 하나의 사실·정의·명시 정보가 option에 직접 대응함
- `UNC`: question/options만으로 안정적으로 판정할 수 없음

### 3.2 Dominant subtype

- `ARITH`: 핵심 terminal operation이 수치 계산·수치 비교·비율·단위 변환임
- `LOGIC`: 명제, 조건, 부정, 양화, 경우 분기 또는 관계를 질적으로 추론함
- `FORMAL`: 도메인 공식·법칙·문법·법률 규칙·알고리즘 절차를 입력에 적용하며,
  핵심이 단순 수치 계산이나 인과 추적은 아님
- `CAUSAL`: 원인→기전→결과 또는 intervention/counterfactual의 결과를 추적함
- `MIXED`: 둘 이상의 subtype이 답에 필수이고 dominant subtype을 정할 수 없음
- `NONE`: Transformation이 필요하지 않음
- `UNC`: 안정적으로 판정할 수 없음

우선순위 규칙:

1. 최종 핵심 operation이 수치 계산이면 `ARITH`
2. 원인·기전·개입의 결과 추적이 핵심이면 `CAUSAL`
3. 도메인 고유 공식·규칙·절차 적용이 핵심이면 `FORMAL`
4. 그 밖의 조건·명제·관계 추론이면 `LOGIC`
5. 둘 이상이 독립적으로 필수이고 위 우선순위로도 dominant를 정할 수 없을
   때만 `MIXED`

### 3.3 Transformation step count

- `0`: Transformation 없음
- `1`: 한 번의 변환 operation
- `2`: 첫 변환 결과가 다음 변환의 입력이 되는 두 dependent operations
- `3P`: 세 번 이상의 dependent operations
- `UNC`: 안정적으로 셀 수 없음

사실 회상, 문제 읽기, option 확인, 같은 계산의 기계적 전개는 별도 step으로
세지 않는다.

출력 grammar:

```text
T=Y|N|UNC;TYPE=ARITH|LOGIC|FORMAL|CAUSAL|MIXED|NONE|UNC;STEPS=0|1|2|3P|UNC
```

## 4. Local AI annotation과 preflight

- Annotator A: local `Meta-Llama-3-8B-Instruct`
- Annotator B: local `Mistral-7B-Instruct-v0.3`
- greedy, `do_sample=false`, beam 1, 최대 32 new tokens
- parser 실패 시 같은 prompt와 format reminder로 최대 2회 재시도하며 의미 규칙은
  바꾸지 않음
- 두 annotator는 독립 실행하며 서로의 판정을 보지 않음
- AI-only annotation이며 human annotation으로 표현 금지

Preflight는 기존 parent consensus `Y` 48개와 `N` 48개를 source/category 다양성을
유지해 deterministic seed `20260831`로 뽑는다. 내부 scalar와 R/M label은 선택과
annotation에 사용하지 않는다.

전수 annotation 허용 hard gate:

- annotator별 parse validity `>=0.99`
- annotator별 parent Transformation balanced accuracy `>=0.80`
- annotator 간 T raw agreement `>=0.80`, Cohen's kappa `>=0.60`
- parent-Y 문항에서 subtype raw agreement `>=0.55`, kappa `>=0.40`
- parent-Y subtype joint-valid coverage `>=0.80`
- step raw agreement `>=0.55`, weighted kappa `>=0.45`
- 한 subtype 또는 한 step level이 전체 유효 판정의 `>0.95`를 차지하지 않음

하나라도 실패하면 전수 annotation과 내부 결합 분석을 실행하지 않는다. 형식
문제 수정이 필요하면 기존 artifact를 덮어쓰지 않고 새 instrument version을
만든다.

## 5. Consensus와 full reliability

- 새 `T`, subtype, steps는 각각 두 annotator exact agreement만 사용
- disagreement 또는 어느 한쪽 `UNC`는 해당 field를 `UNC` 처리
- subtype 분석 item:
  - 기존 parent T=Y
  - 새 annotator 두 명 모두 T=Y
  - 같은 non-UNC subtype에 합의
- NONE baseline item:
  - 기존 parent T=N
  - 새 annotator 두 명 모두 T=N
  - `TYPE=NONE`, `STEPS=0`에 합의
- strength 분석 item:
  - subtype 분석 조건을 만족
  - 두 annotator가 동일한 `1/2/3P`에 합의

Full reliability gate:

- T kappa `>=0.60`
- parent-Y subset subtype kappa `>=0.40`
- step weighted kappa `>=0.45`
- parent-Y subtype 및 step joint-valid coverage 각각 `>=0.70`

실패하면 subtype/internal association은 보고하지 않고 annotation reliability
failure로 종료한다.

## 6. 분석

Internal endpoint는 각 split 안에서 표준화한다. source fixed effects, log token
length, option count, numeric mention과 frozen R/M label을 공변량으로 포함한 HC3
선형모형을 사용한다.

### 6.1 Subtype contrast

각 subtype(`ARITH`, `LOGIC`, `FORMAL`, `CAUSAL`, `MIXED`)을 NONE baseline과
별도로 비교한다. 다른 subtype은 해당 comparison에서 제외한다.

- Discovery count gate: subtype `>=50`, NONE `>=100`
- heldout count gate: subtype `>=15`, NONE `>=30`
- primary family: 5 subtype × 3 primary endpoint = 최대 15 tests
- Discovery two-sided BH-FDR `q<0.05`
- heldout에는 Discovery count/FDR를 통과한 pair만 동결해 평가
- heldout support: 같은 부호, 95% CI 0 제외, selected-test BH `q<0.05`

### 6.2 Step-intensity contrast

Transformation item 안에서만 `1,2,3P`를 ordinal `1,2,3`으로 두고 선형 trend를
검사한다. 따라서 broad T-vs-N 차이를 step trend로 다시 세지 않는다.

- Discovery/heldout에서 각 step level 최소 `30/10`
- 3 primary endpoints를 별도 BH family로 평가
- heldout support 규칙은 subtype과 동일

Secondary 두 component와 R/M prevalence는 descriptive/exploratory로 보고하며
primary selection이나 성공 판정을 바꾸지 않는다.

## 7. 해석 제한

가능한 최대 주장:

> 특정 Transformation subtype 또는 더 많은 dependent transformation steps가
> 기존 R/M label을 통제한 뒤에도 frozen internal endpoint와 연관되며, 기존
> heldout에서 같은 방향의 secondary check를 보였다.

금지:

- subtype이 R/M을 만든다는 인과 주장
- component mediation 또는 Reasoning neuron/head 주장
- 기존 heldout을 새로운 독립 confirmatory dataset으로 표현
- 결과를 보고 subtype, step 정의, count gate 또는 endpoint 변경
- intervention, suppression, patching
- 외부 API 호출
