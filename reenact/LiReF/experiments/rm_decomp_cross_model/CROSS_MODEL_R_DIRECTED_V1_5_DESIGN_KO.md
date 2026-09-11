# Cross-model R-direction Component Search v1.5 설계

상태: **FROZEN — MODEL EXECUTION REQUIRES HASH-LOCKED AUTHORIZATION**  
동결일: `2026-09-01`

## 질문

각 base 모델의 모든 Transformer layer에서 다음 조건을 만족하는 Attention
Head와 FFN Neuron이 존재하는가?

1. R 문항에서 contribution 평균이 R 방향이다: `mean_R(c) > 0`.
2. M 문항보다 R 문항에서 더 R 방향이다: `Delta=mean_R(c)-mean_M(c) > 0`.
3. heldout에서도 같은 조건과 candidate-family FDR 기준을 통과한다.
4. 마지막 prompt token에서 50%·100% 억제하면 최종 절대 R/M gap이 줄어든다.
5. gap 감소가 matched control과 세 random control의 평균보다 크고 dose response를 보인다.

## M-direction과의 대칭성

이 설계는 M-direction v1.3/v1.4의 방향 조건만 반전한다.

- M-direction: `mean_M(c)<0` 및 `Delta>0`
- R-direction: `mean_R(c)>0` 및 `Delta>0`

dataset, 2,400/600 split, LiReF 축, 전체-depth 검색, type별 최대 5개,
control 선정, FDR, bootstrap/permutation, suppression 위치와 강도는 동일하다.

## 모델과 후보 지명

- `Meta-Llama-3-8B`
- `Mistral-7B-v0.3`
- `OLMo-2-1124-7B`
- `gemma-2-9b`
- 전체 layer, Head/Neuron 각각 최대 5개, depth quota 없음
- Discovery 결과만으로 후보·control 동결
- heldout 결과를 본 뒤 후보 교체 금지
- 조건을 만족하는 후보가 없으면 `0개`로 기록

## 주장 제한

통과 후보는 R 문항에서 R 방향 residual 성분을 상대적으로 더 기록하고,
억제하면 측정된 R/M representation gap을 줄이는 component다. 이는 reasoning
능력의 원인, 정답 행동의 필수 부품, 완전한 reasoning circuit 또는 독립
재현을 의미하지 않는다.

모델 실행 중 기존 Gap/M-direction 결과와 `result.pdf`는 수정하지 않는다.
