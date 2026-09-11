# Cross-model M-directed v1.3 정적 검토

상태: **PASS — HASH-LOCKED EXECUTION AUTHORIZATION MAY BE CREATED**  
검토일: `2026-08-31`

## 정정과 버전 분리

- v1.2 계산 결과는 보존하되 `negative-Delta diagnostic`으로만 해석
- v1.3은 별도 study/design/runner/authorization/output root 사용
- M-directed 정의를 `mean_M(c)<0` 및 `mean_R(c)-mean_M(c)>0`로 명시

## 정적 판정

| 항목 | 판정 | 확인 내용 |
|---|:---:|---|
| group mean 계산 | PASS | Head/Neuron contribution의 Discovery M/R 평균을 별도 저장 |
| M 방향 | PASS | 후보는 `memory_mean_discovery < 0` 필수 |
| gap 지지 | PASS | 후보는 `Delta_discovery > 0` 필수 |
| 후보 동결 | PASS | 위 조건 내 global Delta top-5/type, depth quota 없음 |
| heldout | PASS | M 평균 음수·Delta 양수 same-sign와 후보-family BH `q<.05` |
| intervention | PASS | 마지막 prompt token, candidate `.5/1`, control `1` |
| causal criteria | PASS | CI·permutation FDR·dose·matched/random control 모두 요구 |
| 기존 로직 대칭성 | PASS | group mean 저장·후보 gate 외 inference/statistics는 v1.1과 동일 |
| 모델 변경 금지 | PASS | inference mode; optimizer/backward/weight write 없음 |
| tensor persistence | PASS | full hidden/pre-O/z 저장 없음 |
| 출력/PDF | PASS | 새 출력 경로, overwrite 거부, 실행 중 PDF 수정 금지 |

## Model-free test

`test_cross_model_m_directed_v1_3.py`: **11/11 PASS**

## 주장 경계

통과 후보는 M 문항에서 실제 M 방향 residual 성분을 상대적으로 더 기록하고,
억제 시 최종 절대 R/M LiReF gap을 control보다 더 줄인 component다. 이는
memorization 능력, memorization circuit, 행동 원인 또는 독립 재현이 아니다.
