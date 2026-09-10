# Stage E Natural R/M Feature Discovery v1 설계

상태: **FROZEN — AI annotation 및 read-only scalar extraction 승인, intervention 금지**  
동결일: `2026-08-30`

## 1. 목적

Stage E v2–v5 synthetic behavioral control은 Meta-Llama-3-8B base의 문제 수행
불안정성 때문에 입력 feature를 독립적으로 검증하지 못했다. 이번 분석은 새
synthetic task를 만들지 않고, Stage A–C에 사용한 기존 MMLU-Pro 3,000문항에서
다음 탐색 질문을 다룬다.

> 어떤 자연 문항 특성이 R/M label, frozen LiReF projection 및 기존 4개
> component의 LiReF-direction contribution과 함께 변하는가?

이 분석은 **feature discovery/association study**이며 원인, mediation 또는
Reasoning/Memorization mechanism을 검증하지 않는다.

## 2. 고정 데이터와 분할

- 모델/대상: `Meta-Llama-3-8B base`
- 문항: Stage A와 동일한 `mmlu-pro-3000samples.json`
- R/M 기준: `memory_reason_score > 0.5`이면 R, 그 외 M
- Discovery: 기존 train 2,400문항
- Validation: 기존 heldout 600문항
- split을 다시 만들거나 item을 제외하지 않는다.
- Discovery에서 feature/endpoint pair를 선택하고 Validation에서는 재선택하지
  않고 동일한 pair만 평가한다.

## 3. Blind feature codebook

두 AI annotator에게는 `question`과 `options`만 제공한다. 다음 값은 제공하지
않는다.

- R/M label 및 `memory_reason_score`
- 정답/answer index 및 chain-of-thought
- category, source dataset 및 discovery/validation split
- LiReF, candidate response, 행동 결과 및 기존 Stage E 결과

동결 feature는 여섯 개다.

1. `answer_mode`
   - `DER`: prompt의 정보와 규칙을 사용해 답을 유도해야 함
   - `RET`: 주로 학습된 사실·정의·연관을 직접 회상하면 됨
   - `MIX`: 회상과 유도가 모두 실질적으로 필요함
   - `UNC`: 질문만으로 안정적으로 판정할 수 없음
2. `transformation_required`
   - 주어진 값을 계산하거나 논리·형식·인과 규칙으로 변환해야 하면 `Y`
3. `composition_required`
   - 서로 구별되는 둘 이상의 premise/constraint를 결합해야 하면 `Y`
4. `multi_step_required`
   - 앞 단계 결과가 다음 단계 입력이 되는 의존적 추론이 둘 이상이면 `Y`
5. `external_knowledge_required`
   - prompt/options에 없는 domain fact·정의·연관 지식이 필요하면 `Y`
6. `answer_directness`
   - 하나의 명시 정보 또는 하나의 회상 사실로 바로 option을 고르면 `DIR`
   - 변환·결합·다단계 처리가 필요하면 `IND`

각 binary feature는 `Y/N/UNC`, directness는 `DIR/IND/UNC`를 사용한다.
문항의 실제 정답을 맞히는 것이 아니라 **요구되는 해결 방식**을 판정한다.

## 4. AI annotation과 consensus

- Annotator A: local `Meta-Llama-3-8B-Instruct`
- Annotator B: local `Mistral-7B-Instruct-v0.3`
- deterministic greedy decoding, sampling 금지
- parser 실패는 최대 2회 형식 재시도 후 `UNC`
- 두 annotator의 feature별 raw agreement와 Cohen's kappa를 결과에 보고
- 두 판정이 같으면 consensus로 사용
- 불일치는 local `gemma-2-9b-it`가 동일 blind input과 두 판정만 보고 adjudicate
- adjudicator도 R/M label·정답·내부 결과를 보지 않는다.
- AI-only annotation이며 human-annotated evidence라고 표현하지 않는다.

## 5. 내부 endpoint

Primary internal endpoint:

- 마지막 Transformer block raw output의 frozen Layer 31 LiReF projection

기존 고정 component endpoint:

- `L31N13336`
- `L29H00030`
- `L30H00006`
- `L29H00031`

네 component 모두 raw activation이 아니라 frozen same-layer LiReF-direction
scalar contribution인 `total_contribution`을 사용한다. Discovery 값은 Stage B,
Validation 값은 pre-Stage-C의 보존된 natural-response table에서 읽는다.

Layer 31 item scalar는 기존 cache에 최종 block output이 없으므로 frozen Stage A
direction을 사용한 read-only forward hook으로 다시 추출한다. hook은 값을
수정하지 않고 scalar만 기록하며 hidden vector를 파일에 저장하지 않는다.
Layer 0–30 scalar는 secondary trajectory diagnostic으로만 저장한다.

## 6. 분석

Consensus annotation에서 다음 여섯 분석 feature를 만든다.

- `mode_derivation_vs_retrieval`: `DER=1`, `RET=0`; `MIX/UNC` 제외
- `transformation_required`: `Y=1`, `N=0`; `UNC` 제외
- `composition_required`: 동일
- `multi_step_required`: 동일
- `external_knowledge_required`: 동일
- `answer_indirect`: `IND=1`, `DIR=0`; `UNC` 제외

각 split에서 결과별로 source fixed effects, log token length, option count와 numeric
mention 여부를 포함한 선형모형을 사용한다. binary R/M label에는 exploratory
linear-probability coefficient를 사용하고 unadjusted odds ratio를 함께 보고한다.
HC3 standard error와 95% CI를 계산한다.

Discovery의 6 features × 6 endpoints(R/M label + Layer31 + 4 components), 총 36개
검정 전체에 BH-FDR를 적용한다. 다음을 모두 만족한 feature만 Validation 후보다.

1. present/absent 각각 50문항 이상
2. R/M label association의 Discovery `q < 0.05`
3. 적어도 한 internal endpoint association의 Discovery `q < 0.05`

Validation에서는 동결된 feature-label 및 feature-internal endpoint pair만
검정한다. 부호 재현, two-sided BH `q < 0.05`, 95% CI의 0 제외를 모두
Validation support로 기록한다. Validation present/absent가 각각 15 미만이면
`insufficient_support`로 기록한다.

## 7. 해석 제한

이번 결과로 말할 수 있는 최대 범위:

> 특정 자연 문항 feature가 기존 데이터에서 R/M label과 내부 endpoint에 함께
> 연관되며, 기존 heldout split에서도 같은 방향으로 재현되었다.

말할 수 없는 것:

- 해당 feature가 R/M representation의 원인임
- 네 component가 feature를 매개함
- 특정 component가 Reasoning 또는 Memorization 기능을 담당함
- MMLU-Pro의 category/source/style 차이를 완전히 제거함
- Stage E synthetic calibration 실패가 해결됨

Validation-supported feature가 있으면 새로운 문항·template/source를 사용한
독립 재현 또는 controlled manipulation을 별도로 설계한다. intervention은 그
후 별도 설계와 승인 없이는 수행하지 않는다.

## 8. 허용 및 금지

허용:

- blind AI annotation
- 보존된 scalar response table 읽기
- read-only forward hook을 통한 layer projection scalar 추출
- model-free 통계, audit 및 report 생성

금지:

- weight/activation 수정
- patching, suppression, amplification
- 후보 추가 탐색
- 결과를 본 뒤 feature 정의·문항·threshold 변경
- Validation 결과를 이용한 feature 재선택
- raw hidden state 또는 전체 component tensor 저장

