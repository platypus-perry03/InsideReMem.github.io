# Meta-Llama M 방향 component 탐색 v1.4 설계

상태: **FROZEN — MODEL EXECUTION REQUIRES HASH-LOCKED AUTHORIZATION**  
동결일: `2026-08-31`

## 질문

Meta-Llama-3-8B의 모든 Transformer layer에서 다음 조건을 만족하는
Attention Head와 FFN Neuron이 존재하는가?

1. M 문항에서 contribution 평균이 M 방향이다: `mean_M(c) < 0`.
2. R 문항보다 M 문항에서 더 M 방향이다: `mean_R(c)-mean_M(c) > 0`.
3. heldout에서도 같은 조건과 FDR 기준을 통과한다.
4. 마지막 prompt token에서 50%·100% 억제하면 최종 R/M gap이 줄어든다.
5. gap 감소가 matched control과 random controls보다 크고 dose response를 보인다.

## 후보 지명

- 전체 32개 layer 검색
- Head와 Neuron 각각 최대 5개
- depth quota 없음
- Discovery 결과만으로 후보와 control 동결
- validation 결과를 본 뒤 후보 교체 금지

## 주장 제한

통과 후보는 M 문항에서 M 방향 residual 성분을 상대적으로 더 쓰고,
억제하면 측정된 R/M representation gap을 줄이는 component다.
이는 memorization 정답을 저장한 부품, memorization 능력의 원인,
또는 완전한 memorization circuit을 의미하지 않는다.
