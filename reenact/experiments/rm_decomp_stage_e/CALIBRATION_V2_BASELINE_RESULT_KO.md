# Stage E Baseline Calibration v2 실행 결과

상태: **COMPLETE — CALIBRATION FAIL, STAGE E PILOT 미승인**  
실행일: 2026-08-30  
run ID: `calv2_baseline_20260830_02`

## 1. 실행 범위

- frozen v2 dataset: 144 pairs / 288 prompts
- model: local Meta-Llama-3-8B
- device: `cuda:1` (요청한 `cuda:0`은 실행 직전 가용 메모리가 부족하여 동일 사양의 비어 있는 GPU로 대체)
- batch size: 8
- dtype: float32
- human linguistic audit: `not_performed`
- human audit gate: `waived_by_user`
- 허용 출력: baseline teacher-forced score와 greedy generation behavior뿐
- LiReF, hidden state, candidate state, hook, activation/weight intervention: 사용하지 않음

`calv2_baseline_20260830_01`은 sandboxed Python이 CUDA를 볼 수 없어 모델 로드 전에 종료됐다. 모델 로드·forward·GPU 계산은 모두 발생하지 않았으며, 실패 상태 파일을 보존했다. 실제 calibration 결과는 별도 승인 파일로 동결한 `run02`에서 생성했다.

## 2. 최종 판정

| Lexical family | 판정 | 핵심 실패 지표 |
|---|---|---|
| decrease | FAIL | forced-choice gap 0.25, Irrelevant forced-choice 1.00, mean `D_k=-1.9223`, `d_z=-1.0106`, same sign 5/6, generation 양쪽 0 |
| increase | FAIL | forced-choice gap 0.125, Irrelevant forced-choice 1.00, mean `D_k=-1.5056`, `d_z=-2.3484`, same sign 6/6, generation 양쪽 0 |
| temperature | FAIL | forced-choice Relevant 1.00 / Irrelevant 0.9792, generation gap 0.8542 (Relevant 0.8542 / Irrelevant 0) |

사전 동결된 모든 기준을 동시에 적용했을 때 통과한 family는 없다. 결과를 보고 item/template를 제외하거나 threshold를 변경하지 않았다.

## 3. 해석

- decrease와 increase는 A/B forced-choice에서 direct-retrieval 조건이 ceiling(1.00)에 도달했고, Relevant 조건보다 일관되게 쉬웠다. 음의 `D_k`는 transformation-dependent Relevant prompt의 correct-vs-alternative margin이 Irrelevant보다 작았음을 뜻한다.
- temperature는 paired-answer log-odds 불균형 기준은 통과했지만, forced-choice가 양쪽 모두 ceiling에 걸렸고 greedy generation은 Relevant와 Irrelevant 사이에 큰 형식/행동 차이를 보였다.
- generation 0은 언제나 산술 정답을 모르기 때문만은 아니다. decrease/increase에서 모델은 `18 blue marbles` 같은 정답+단위 출력을 자주 생성했지만 frozen exact-match 규칙상 숫자만 허용된 항목에서는 오답 처리됐다. 이 현상 역시 결과 확인 뒤 normalization을 바꿔 재판정하지 않고, 현재 run의 FAIL에 그대로 포함한다.
- 따라서 이번 결과는 "R/M이 구분되지 않는다"는 결론이 아니다. Baseline Calibration만 수행했으며 LiReF R/M projection이나 후보 component는 측정하지 않았다.

## 4. 안전 및 provenance

- execution authorization SHA-256: `e3cac9f2224ea4c3b7424bdaeda8709f5767986d46ab5e2e1c7ff55ba1b772b1`
- implementation SHA-256: `78681b8e2b778e4cb7f224baff82a6eb9a4f67eaa7dcf139d13097e9131b9884`
- static safety review SHA-256: `fa9fd731d70712b11439e22359abbc76ce42ad4b2118c20d9e632f9f3b38358d`
- pair results SHA-256: `4cdf5d57295d483ddb2ad3bfe1683442c1c72759c5d447f7a1d18999283c3668`
- summary SHA-256: `cec0432036344dccfaebc60ccc82f5d9aa463171eeae8d5c0dc7d6a1d9b295ce`
- environment/safety SHA-256: `5c45f24810863965e40702d91d8e7712a9325fea6e62c76b999051aa6b5575bb`
- run manifest SHA-256: `97f6fbf1a8fb5294495aa9896df4114d8016ab956e4b6f11c2e9695e5b704f36`

## 5. 다음 gate

현재 frozen 규칙에서는 PASS family가 0개이므로 이 calibration dataset을 근거로 Stage E Pilot을 시작하지 않는다. 다음 작업은 실패 원인을 기록한 뒤 새 버전의 operationalization 또는 scoring/generation 설계를 **model 결과와 분리하여 사전 명세**하는 것이다. 새 설계는 기존 v2 artifact를 덮어쓰지 않고 새 version, dataset, audit와 execution approval을 가져야 한다.

Human audit waiver가 적용된 결과이므로 이 run을 `human-audited`라고 표현하지 않는다.
