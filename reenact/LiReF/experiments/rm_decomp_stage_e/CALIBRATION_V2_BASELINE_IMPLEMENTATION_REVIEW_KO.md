# Stage E Calibration v2 Baseline 구현 및 정적 검토

상태: **IMPLEMENTED — STATIC SAFETY REVIEW PASS — EXECUTION NOT APPROVED**  
작성일: 2026-08-29

## 구현 범위

`run_calibration_v2_baseline.py`는 승인 이후 다음 baseline behavior만 측정하도록 구현했다.

- canonical answer teacher-forced sequence log probability
- paired alternative answer sequence log probability
- paired-answer margin `M(x)`
- A/B forced-choice accuracy
- greedy generation과 frozen normalization에 따른 exact-match accuracy
- raw/length-normalized logP와 per-token geometric probability
- template-level `D_k`, lexical-family별 `d_z`, same-sign count
- 10,000회 descriptive template-cluster bootstrap CI
- frozen floor/ceiling 및 condition-gap 기준에 따른 family-level PASS/FAIL

LiReF score, hidden state, candidate activation, attention/FFN intermediate, hook, patching과 weight intervention은 구현 범위에서 제외했다.

## 실행 차단 구조

1. `--preflight-only`와 `--self-test`는 Python 표준 라이브러리만 사용한다.
2. `torch`와 `transformers`는 승인 이후 도달 가능한 `load_runtime()` 내부에서만 lazy import한다.
3. `--execute`에는 별도 authorization JSON이 필수다.
4. authorization은 implementation, static review, dataset, manifest와 waiver hash를 모두 일치시켜야 한다.
5. `explicit_execution_approval=true`, `execution_allowed=true`, 승인자와 승인 시각이 모두 없으면 중단한다.
6. 출력 run directory가 이미 존재하면 덮어쓰지 않고 중단한다.
7. 모델과 tokenizer는 `trust_remote_code=false`, `local_files_only=true`로만 로드한다.
8. 모델은 eval/float32로 고정하고 hidden-state/attention 출력을 명시적으로 끈다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| Source AST parse | PASS |
| Runtime import가 `load_runtime`에만 존재 | PASS |
| Hook/weight edit/tensor 저장 호출 부재 | PASS |
| LiReF direction 참조 부재 | PASS |
| Hidden state/attention output 비활성화 | PASS |
| 승인 검증이 model load보다 먼저 실행 | PASS |
| Frozen input preflight | PASS |
| Normalization/statistics self-test | PASS |
| 승인 파일 없는 `--execute` 차단 | PASS |
| 정적 검토 중 model/tokenizer/GPU 사용 | 없음 |

## Artifact hash

| Artifact | SHA-256 |
|---|---|
| Dataset builder | `58bdf5d6b6294513c4f282b129586d090e3519b6589cf425849b912140980757` |
| Frozen v2 dataset | `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87` |
| Baseline implementation | `78681b8e2b778e4cb7f224baff82a6eb9a4f67eaa7dcf139d13097e9131b9884` |
| Static review implementation | `03fe9353d788cd7df98ac1078c57c7cf5d2f32af4795dceace3e6444fbe1921d` |
| Static safety review report | `fa9fd731d70712b11439e22359abbc76ce42ad4b2118c20d9e632f9f3b38358d` |
| Human-audit waiver | `52e01895394f48c7e9391b43042953f23173df89d37585d34ad1b70e4d60752f` |

## 아직 결정·승인할 값

- 실행 `run_id`
- 명시적 CUDA device(`cuda:N`)
- batch size(1–8)
- 사용자의 별도 `explicit_execution_approval`

`calibration_v2_execution_authorization_draft.json`은 위 값이 비어 있고 승인값이 `false`인 비실행 초안이다. 이 파일 자체로는 모델 로딩이나 Calibration 실행을 허용하지 않는다.

## 현재 다음 gate

구현과 정적 검토는 끝났다. 다음은 artifact hash, waiver와 실행 parameter를 확인한 뒤 사용자가 **별도로 명시적 실행 승인**하는 단계다. 그전까지 Baseline Calibration, Stage E Pilot과 모델/GPU 실행은 금지한다.
