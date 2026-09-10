# Cross-model M-selective Functional Homologue Search v1.2 설계

상태: **FROZEN — MODEL EXECUTION REQUIRES HASH-LOCKED AUTHORIZATION**  
동결일: `2026-08-31`

## 목적

v1.1은 `Delta = mean_R - mean_M > 0`인 R-selective component만 검색했다.
v1.2는 같은 데이터·split·모델·통계·suppression 기준을 유지하면서 반대쪽
꼬리인 `Delta < 0`만 검색한다.

> **M 문항에서 R 문항보다 LiReF R 방향 기여가 작거나 M 방향 기여가 큰
> Head/Neuron 중, 억제했을 때 최종 절대 R/M gap이 안정적으로 감소하는
> component가 있는가?**

## 동결 규칙

- 모델: Mistral-7B-v0.3 / OLMo-2-1124-7B / gemma-2-9b base
- 전체 Transformer layer 검색
- Discovery negative `Delta`가 작은 순서로 Head 최대 5, Neuron 최대 5
- depth quota 없음; Early/Middle/Late는 결과 표지
- heldout에서 같은 음의 부호 및 후보-family BH `q<.05`
- 마지막 prompt token에서만 50%·100% suppression
- bootstrap CI, permutation FDR, dose monotonicity, matched/random control 우월을 모두 요구
- hidden/pre-O/z 전체 tensor 저장, weight update, 후보 교체 금지
- 실행 중 `result.pdf` 자동 수정 금지

## 해석 제한

이 실험은 v1.1 R-side 결과를 본 뒤 설계한 대칭적 확장이다. 따라서 결과는
`exploratory M-selective functional contributors`로 보고하며, 독립 확인이나
보편적 memorization circuit으로 부르지 않는다.
