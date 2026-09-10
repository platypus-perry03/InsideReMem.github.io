# Stage E Transformation 독립 자연문항 재현 v2 설계

상태: **FROZEN — ANNOTATION PREFLIGHT/POOL CONSTRUCTION ALLOWED — STUDY MODEL EXECUTION NOT AUTHORIZED**  
동결일: `2026-08-31`

## 1. 목적

Natural Feature Discovery v1에서 `transformation_required`가 Layer 31 LiReF,
`L29H00030`, `L30H00006` 및 R/M label과 함께 증가했다. 그러나 Validation의
transformation 문항 172개 중 170개가 기존 R label이어서 feature와 label을
분리하지 못했다.

본 연구는 기존 3,000문항과 겹치지 않는 MMLU-Pro 자연문항에서 다음을
검사한다.

> 별도로 판정한 reasoning-intensity 수준을 통제하고도 transformation 요구가
> frozen LiReF 및 사전 지정 component의 R-direction contribution과 연관되는가?

이 연구는 observational item-independent same-benchmark replication이다.
causal manipulation, mediation 또는 cross-model generalization 실험이 아니다.

## 2. 중요한 label 제한

원 논문의 R/M label은 GPT-4o가 부여한 `memory_reason_score`다. 공개된 로컬
artifact에는 기존 3,000문항의 score만 있고, 새 후보 문항에는 이 값이 없다.

따라서 새 문항에 부여하는 값은 공식 R/M label이 아니라
**`local_reasoning_score_proxy`**로 부른다. 두 local instruct model이 논문
Appendix A의 공개 scoring prompt를 독립적으로 적용한다. 이 proxy가 기존
GPT-4o score와 사전 calibration gate를 통과하지 못하면 재현 dataset을 만들지
않는다. 통과하더라도 "GPT-4o R/M label을 재현했다"고 주장하지 않는다.

## 3. 후보 문항 풀과 non-reuse

- source: local MMLU-Pro test split 12,032문항
- 제외: `mmlu-pro-3000samples.json`의 모든 question ID/문자열
- 추가 제외: `mmlu-pro-600samples.json`의 모든 question ID/문자열
- ID 또는 normalized question text가 한 번이라도 겹치면 제외
- 남은 원문 풀 내부의 normalized-question 중복도 첫 항목만 유지
- deterministic non-reuse/dedup audit 후 후보: 8,656문항
- 기존 synthetic Calibration/Pilot/replication 문항도 재사용하지 않는다.
- 모델 prompt는 Stage A와 같은 `Q: {question}\nA: `를 유지한다.

후보 풀 생성은 question/options/answer/category/src metadata만 사용하며 LiReF,
component response 또는 새 study-model hidden state를 읽지 않는다.

## 4. 분리된 blind annotation

두 annotator는 각각 다음 두 작업을 **서로 다른 prompt/pass**로 수행한다.

1. 논문 Appendix A 기반 0.0--1.0 reasoning-score proxy
2. `transformation_required = Y/N/UNC`

Annotator에게 제공:

- question
- options

차단:

- 기존/새 R/M label 및 score
- 정답, answer index, CoT
- category, src, split
- LiReF/component/model score
- 다른 annotator 판정

모델:

- annotator A: local `Meta-Llama-3-8B-Instruct`
- annotator B: local `Mistral-7B-Instruct-v0.3`
- decoding: greedy, temperature 0, sampling 없음

Reasoning proxy consensus:

- 두 score 모두 parse 가능
- 두 score의 절대차 `<=0.25`
- 두 score가 모두 `>0.5`이면 proxy-R
- 두 score가 모두 `<=0.5`이면 proxy-M
- 그 외 `UNC`
- consensus continuous score는 두 값의 산술평균

Transformation consensus:

- 두 annotator가 정확히 `Y` 또는 정확히 `N`으로 일치할 때만 사용
- disagreement/UNC는 제외

## 5. Annotation preflight gate

내부 endpoint를 보기 전에 기존 heldout 600 중 192문항을 score bin과 source로
고정 추출해 annotation instrument를 검증한다. 이 문항은 최종 재현 표본에
포함되지 않는다.

모두 만족해야 full candidate annotation을 허용한다.

- annotator별 score parse success `>=0.99`
- annotator별 original GPT-4o score와 Spearman `>=0.60`
- 두 annotator 평균 proxy와 original score Spearman `>=0.70`
- threshold 0.5 기준 ensemble balanced accuracy `>=0.70`
- annotator 간 continuous-score Spearman `>=0.70`
- annotator 간 binary R/M Cohen's kappa `>=0.60`
- annotator별 transformation parse success `>=0.99`
- 두 annotator transformation 동시 유효 coverage `>=0.98`
- transformation raw agreement `>=0.80`
- transformation Cohen's kappa `>=0.60`

하나라도 실패하면 instrument를 결과에 맞춰 수정하지 않고 현재 version을
BLOCKED로 보존한다.

## 6. 2 x 2 feasibility와 표본

Full annotation 후 다음 네 셀을 만든다.

| local proxy | Transformation Y | Transformation N |
|---|---:|---:|
| proxy-R | R/T | R/no-T |
| proxy-M | M/T | M/no-T |

- 각 셀 최소 가용 문항: 96
- 하나라도 96 미만이면 내부 endpoint를 추출하지 않고 BLOCKED
- 최종 목표: 96문항/셀, 총 384문항
- random seed: `20260831`
- 96개의 4문항 matched block을 구성하며 각 block은 네 셀을 하나씩 포함
- category는 block 안에서 exact match
- source exact match를 우선하되 불가능하면 불일치 수를 기록
- tie-break: question ID lexical order 후 frozen seed
- 선택 비용: token length, option count, numeric-mention indicator의 거리

표본 freeze 전 audit:

- 384 unique items, 셀당 정확히 96
- 기존 3,000/600과 ID 및 normalized question text 비중복
- category distribution이 네 셀에서 동일
- prompt hash 생성
- 내부 outcome 미사용 확인
- 각 nuisance variable의 cell-pair standardized mean difference를 보고

## 7. Frozen endpoint와 분석

Primary endpoint:

1. Layer 31 raw block output dot frozen Layer-31 LiReF
2. `L29H00030` frozen same-layer LiReF-direction scalar contribution
3. `L30H00006` frozen same-layer LiReF-direction scalar contribution

Secondary endpoint:

- `L31N13336`
- `L29H00031`
- Layer 0--30 LiReF trajectory

전체 hidden vector/pre-O tensor는 저장하지 않고 item-level scalar만 저장한다.
Hook은 read-only이며 forward output을 변경하지 않는다.

Primary estimand은 네 셀에 동일 가중한 transformation average marginal contrast다.

```text
0.5 * [(proxy-R,T) - (proxy-R,no-T)]
+ 0.5 * [(proxy-M,T) - (proxy-M,no-T)]
```

Model에는 proxy label, transformation, interaction, matched-block effect를 넣는다.
Primary 세 endpoint의 p-value에 BH-FDR `q<0.05`를 적용하며, 방향은 Discovery와
같은 양수로 사전 지정한다. Interaction은 별도 descriptive 결과다.

재현 신호 기준은 endpoint별로 다음을 모두 만족하는 것이다.

- average marginal contrast `>0`
- 95% cluster-bootstrap CI가 0을 제외
- primary-test BH q `<0.05`

Bootstrap은 matched block을 10,000회 resample하고 seed `20260831`을 사용한다.

## 8. 해석 제한과 gate

- preflight 또는 2x2 feasibility 실패 시 study model/LiReF/component 실행 금지
- dataset freeze 후 item/template 제외 금지
- 결과 확인 후 threshold/endpoint/방향 변경 금지
- intervention, suppression, patching 금지
- `local_reasoning_score_proxy`를 공식 R/M label로 표현 금지
- 성공해도 transformation의 causal effect 또는 component mediation 주장 금지
- human audit은 수행하지 않으며 `AI-only annotated`로 명시

실제 study-model scalar extraction은 implementation, model-free test, static review,
locked input hash와 별도 execution authorization 이후에만 허용한다.
