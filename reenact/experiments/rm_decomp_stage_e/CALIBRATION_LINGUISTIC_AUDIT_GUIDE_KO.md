# Stage E Calibration Linguistic Audit 지침

상태: **REVIEW PREPARATION — 검수 결과 미입력**  
대상: `calibration_assets/calibration_linguistic_audit.csv`의 144개 pair  
목적: 모델 결과를 보기 전에 controlled pair가 의도한 단일 조작을 구현했는지 사람이 확인한다.

## 1. 검수 독립성과 blind 조건

- 검수자는 baseline 점수, 생성 정답, LiReF score, 후보 activation 및 intervention 결과를 보지 않는다.
- 검수에는 context, Relevant/Irrelevant 질문, canonical answer, 사전 지정 foil만 제공한다.
- 모든 144개 pair를 검수한다. 결과가 좋아 보이는 문항만 선택하지 않는다.
- `reviewer_id`에는 실제 검수자 식별자를 기록한다. AI의 자동 점검은 독립 human audit를 대체하지 않는다.

## 2. CSV 열별 판정

각 pair에 대해 아래 열을 `YES` 또는 `NO`로 채운다.

| 열 | `YES` 기준 |
|---|---|
| `fact_math_correct` | context의 직접 사실과 단일 증가·감소 계산이 모두 canonical answer와 일치함 |
| `identical_context_confirmed` | 두 조건의 context가 문자 단위로 동일함 |
| `single_feature_change_confirmed` | 질문 target만 바뀌며, 의도한 차이는 relation transformation의 task relevance뿐임 |
| `natural_grammar` | 두 질문과 context가 자연스럽고 의미가 모호하지 않음 |
| `relevance_labels_correct` | Relevant는 계산이 필요하고 Irrelevant는 동일 답을 직접 조회함 |
| `answer_exposure_equal` | canonical answer literal이 두 조건의 공통 context에 동일하게 노출됨 |

추가 기록:

- `reviewer_id`: 검수자 식별자
- `review_status`: `PASS`, `FAIL`, `NEEDS_REVISION` 중 하나
- `comments`: `NO` 또는 모호성의 구체적 이유

## 3. Pair 판정 규칙

- 여섯 개 필드가 모두 `YES`이면 `PASS`다.
- 하나라도 명백히 `NO`이면 `FAIL`이다.
- 문구 수정 후 재검수가 필요하면 `NEEDS_REVISION`이다.
- 빈 값이 있거나 reviewer가 기록되지 않은 pair는 미검수로 취급한다.

## 4. Dataset 승인 규칙

- 144개 pair가 모두 검수되어야 한다.
- `FAIL` 또는 `NEEDS_REVISION`이 하나라도 있으면 현재 dataset manifest는 승인하지 않는다.
- 오류 수정은 모델 결과를 보기 전에 template-family 단위로 수행하고, dataset 전체를 새로 생성한다.
- 재생성 후 dataset, tokenizer audit, linguistic audit sheet 및 manifest의 SHA-256을 모두 갱신한다.
- 수정 전 dataset의 hash와 실패 사유를 보존한다. 조용히 문항을 삭제하거나 교체하지 않는다.

## 5. 이미 완료된 자동 검사

자동 tokenizer audit은 다음 항목에서 144/144 PASS다.

- Relevant/Irrelevant prompt token 길이 일치
- 질문 target label이 각각 1 token
- canonical answer literal이 공통 context에 존재
- canonical answer가 transformation의 시작값·변화량과 우연히 같지 않음
- 사전 지정 foil이 canonical answer와 다름
- 각 pair에 최소 2개 foil 존재
- source relation span에 대응하는 tokenizer token이 존재

자동 검사는 문법·자연스러움·의미적 단일 조작을 보증하지 않으므로 human audit가 별도로 필요하다.

## 6. 검수 완료 후 허용되는 다음 작업

검수 완료만으로 baseline calibration 실행이 승인되지는 않는다. 다음 gate도 별도로 필요하다.

1. 검수 완료 CSV와 승인 요약의 hash 기록
2. baseline-only calibration 구현
3. 구현에 대한 정적 safety review
4. 구현 및 dataset manifest hash 기록
5. 사용자의 명시적 execution approval

그 전에는 모델 forward, GPU calibration, LiReF 계산, hidden-state capture 및 intervention을 실행하지 않는다.
