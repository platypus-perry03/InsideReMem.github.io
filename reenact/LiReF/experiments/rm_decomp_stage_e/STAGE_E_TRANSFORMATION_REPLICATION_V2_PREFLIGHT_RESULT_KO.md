# Stage E Transformation 독립 자연문항 재현 v2 Annotation Preflight 결과

상태: **BLOCKED — LOCAL R/M PROXY RELIABILITY FAIL — INTERNAL REPLICATION NOT RUN**  
실행일: `2026-08-31`

## 1. 목적

Natural Feature Discovery v1의 `transformation_required` association이 기존 R/M
label과 거의 겹친 문제를 해결하기 위해, 기존 3,000/600문항과 겹치지 않는
MMLU-Pro 자연문항에서 다음 네 셀을 구성하려 했다.

- proxy-R / transformation Y
- proxy-R / transformation N
- proxy-M / transformation Y
- proxy-M / transformation N

그러나 새 문항에는 원 논문의 GPT-4o `memory_reason_score`가 공개되어 있지
않다. 따라서 두 local instruct model의 score가 기존 공개 score를 충분히
근사하는지 내부 endpoint를 보기 전에 preflight했다.

## 2. 비중복 후보 풀

- MMLU-Pro test 원문: 12,032
- 기존 3,000 및 600 파일과 ID 또는 normalized text가 겹치는 문항 제외
- 남은 풀 내부의 중복 normalized text도 제거
- 최종 blind 후보 풀: **8,656문항**
- candidate-pool automatic audit: **PASS**

이 단계에서는 study model, LiReF, candidate component 및 hidden state를 사용하지
않았다.

## 3. v2 attempt 01

기존 heldout에서 score 구간별 48개씩 총 192개를 blind preflight로 사용했다.

- annotator A score parse: `0/192`
- annotator B score parse: `126/192`
- 공동 parse: `0/192`
- annotator A가 `SCORE: 0.7` 형식을 사용해 frozen `SCORE=...` parser와 불일치
- annotator B는 설명이 길어 final score가 generation limit 전에 나오지 않은
  항목이 많았음

따라서 attempt 01은 `FAIL_INSTRUMENT_OUTPUT_CONTRACT`로 보존했다. 전체 후보
annotation과 내부 분석은 실행하지 않았다.

## 4. v2.1 disjoint preflight

attempt 01에 사용하지 않은 heldout 160문항을 네 score 구간에서 40개씩 새로
선정했다. Score는 설명 없는 한 줄 출력으로 바꾸고, Transformation은 이전
Natural Feature Discovery에서 κ=0.737을 보인 v1.1 multi-feature codebook을
재사용했다.

### 결과

| Gate | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| score parse A | 1.000 | >=0.99 | PASS |
| score parse B | 0.994 | >=0.99 | PASS |
| A vs original score Spearman | 0.488 | >=0.60 | **FAIL** |
| B vs original score Spearman | 0.640 | >=0.60 | PASS |
| ensemble vs original Spearman | 0.657 | >=0.70 | **FAIL** |
| ensemble binary balanced accuracy | 0.730 | >=0.70 | PASS |
| annotator score Spearman | 0.495 | >=0.70 | **FAIL** |
| annotator binary κ | 0.084 | >=0.60 | **FAIL** |
| Transformation parse A | 1.000 | >=0.99 | PASS |
| Transformation parse B | 0.931 | >=0.99 | **FAIL** |
| Transformation joint coverage | 0.931 | >=0.98 | **FAIL** |
| Transformation agreement | 0.866 | >=0.80 | PASS |
| Transformation κ | 0.706 | >=0.60 | PASS |

모든 gate를 동시에 만족하지 못했으므로 v2.1도 **FAIL**이다.

## 5. 해석

형식 교정 후에도 두 local annotator의 reasoning score와 binary R/M 판정은 서로
충분히 일치하지 않았다. 특히 binary κ=0.084이므로 local ensemble을 새 문항의
R/M label처럼 사용하는 것은 정당화할 수 없다.

따라서 현재 근거로 가능한 결론은 다음과 같다.

> 새 8,656문항 풀은 비중복 조건을 만족하지만, 공식 GPT-4o R/M score를 대체할
> 신뢰도 높은 local proxy를 확보하지 못해 R/M x Transformation 네 셀 재현
> dataset을 만들 수 없다.

이는 기존 `transformation_required` association의 반증이 아니다. label
decoupling을 검증할 측정 도구가 preflight를 통과하지 못한 것이다.

## 6. 실행하지 않은 것

- 8,656문항 full annotation
- 384문항 2x2 replication dataset 생성
- Meta-Llama-3-8B base study-model 실행
- LiReF direction 로딩
- Layer 31 / `L29H00030` / `L30H00006` 측정
- secondary candidate 및 layer trajectory
- intervention / suppression / patching

## 7. 다음 선택지

현재 protocol을 임의로 완화하지 않는다. 다음 중 하나를 새 version에서
사전 결정해야 한다.

1. 원 논문과 같은 GPT-4o scoring을 새 후보에 수행할 수 있는 별도 승인·비용·API
   환경을 마련한다.
2. independent human panel로 reasoning-intensity와 transformation을 판정한다.
3. R/M proxy를 포기하고 category/source/길이/난이도를 맞춘 Transformation
   Y/N observational replication으로 질문을 좁힌다. 이 경우 "R/M label과
   분리했다"는 주장은 할 수 없다.

어느 선택이든 현재 v2/v2.1 artifact와 threshold를 덮어쓰지 않는다.
