# Cross-model M-directed Functional Homologue Search v1.3 설계

상태: **FROZEN — MODEL EXECUTION REQUIRES HASH-LOCKED AUTHORIZATION**  
동결일: `2026-08-31`

## 정정된 질문

LiReF 축은 M에서 R로 향한다. component contribution을 `c`라고 할 때,
M-side writer는 단순히 `mean_R(c)-mean_M(c)<0`인 부품이 아니다.

> **M 문항에서 실제 contribution이 M 방향(`mean_M(c)<0`)이고, R 문항보다
> 더 M 방향(`mean_R(c)-mean_M(c)>0`)이며, 억제하면 최종 절대 R/M gap이
> 줄어드는 Head/Neuron이 여러 base 모델에서 존재하는가?**

## 동결 규칙

- 전체 layer, type별 최대 5개, depth quota 없음
- Discovery에서 `memory_mean_discovery < 0` 및 `Delta_discovery > 0`
- 위 조건 안에서 `Delta_discovery`가 큰 순으로 선정
- heldout에서도 `memory_mean_validation < 0`, `Delta_validation > 0`, BH `q<.05`
- 마지막 prompt token 50%·100% suppression
- bootstrap·permutation FDR·dose·matched/random control 기준은 v1.1과 동일
- v1.2 raw 결과는 보존하되 M evidence로 사용하지 않음
- 실행 중 PDF 자동 수정 금지

## 주장 제한

통과 후보는 M 문항에서 M 방향 residual 성분을 상대적으로 더 기록하고
최종 R/M 표현 gap을 지지한 component다. 이는 memorization 능력이나
memorization circuit의 확인을 의미하지 않는다.
