# Cross-model M-selective v1.2 정적 검토

상태: **PASS — HASH-LOCKED EXECUTION AUTHORIZATION MAY BE CREATED**  
검토일: `2026-08-31`

## v1.1과의 분리

- v1.1 R-selective 결과와 출력 경로는 변경하지 않음
- v1.2는 별도 study ID, design, runner, authorization, output root 사용
- v1.2는 v1.1 완료 후 설계된 exploratory symmetric extension으로 명시

## 정적 판정

| 항목 | 판정 | 확인 내용 |
|---|:---:|---|
| whole-depth 범위 | PASS | 모든 Transformer block 포함 |
| 후보 방향 | PASS | `Delta_discovery < 0`만 허용, 가장 음수인 순으로 type별 최대 5개 |
| depth quota 없음 | PASS | Early/Middle/Late 강제 선발 없음 |
| heldout gate | PASS | Discovery·heldout 모두 음수이며 후보-family BH `q<.05` |
| R-side 대칭성 | PASS | 후보 부호·same-sign 필드 외 discovery/validation/intervention 통계 동일 |
| intervention | PASS | 마지막 prompt token, candidate alpha `.5/1`, control alpha `1` |
| controls | PASS | 같은 type/layer의 low-association matched 1 + random 3 |
| causal criteria | PASS | gap-reduction CI, permutation FDR, dose, matched/random 우월 모두 요구 |
| 모델 변경 금지 | PASS | inference mode; optimizer/backward/weight write 없음 |
| hook 안전성 | PASS | clone 후 지정 component만 수정, `finally`에서 hook 제거 |
| tensor 저장 제한 | PASS | full hidden/pre-O/z tensor 파일 저장 없음 |
| 출력 분리 | PASS | `cross_model_m_selective_v1_2` 새 경로, 기존 결과 overwrite 거부 |
| PDF | PASS | 모델 실행 중 `result.pdf` 자동 수정 금지; 결과 완료 후 별도 수동 갱신 |

## Model-free test

`test_cross_model_m_selective_v1_2.py`: **11/11 PASS**

- 음의 Discovery tail deterministic global top-5/type
- 양의 후보만 있을 때 0-candidate 허용
- whole-depth와 depth band 경계
- component ID parser
- BH-FDR와 gap-reduction 부호
- Head/Neuron last-token suppression
- batch reset hook dictionary 보존과 hook 제거
- frozen design의 PDF 자동 수정 금지

## 주장 경계

통과 후보는 동일 R/M dataset·prompt에서 M 문항에 상대적으로 선택적인
residual writer이며, 억제했을 때 최종 절대 R/M LiReF gap을 control보다 더
줄인 component다. 이는 memorization 능력, memorization circuit, 독립 재현,
같은 번호의 모델 간 대응 또는 행동 인과를 의미하지 않는다.
