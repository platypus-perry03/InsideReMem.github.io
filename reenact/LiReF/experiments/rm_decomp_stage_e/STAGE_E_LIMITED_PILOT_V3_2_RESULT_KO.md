# Stage E limited same-sample Pilot v3.2 실행 결과

상태: **COMPLETE — LIMITED SAME-SAMPLE PILOT; INTERVENTION 미승인**  
실행일: `2026-08-30`  
run ID: `stagee_limited_pilot_v3_2_20260830_01`

## 1. 실행 범위

- model: local Meta-Llama-3-8B
- dataset: frozen v3 dataset 전체 `192 pairs / 384 prompts`
- family: object-count / points-balance / temperature 전부 포함
- device / batch / dtype: `cuda:1` / `8` / `float32`
- evidence class: `limited same-sample Pilot`
- primary contrast: `Arithmetic - Selector`
- independent unit: template family cluster `24개`
- cluster bootstrap: family 안에서 template replacement resampling `10,000회`, seed `20260831`

Calibration 결과를 보고 family, template 또는 item을 제외하지 않았고 correct-only primary 분석도 하지 않았다.

## 2. Primary 결과

양의 값은 Arithmetic 조건이 Selector보다 frozen LiReF의 Reasoning 방향에 더 정렬되었거나, 해당 component가 그 방향으로 더 큰 scalar contribution을 보였다는 뜻이다. `95% CI excludes 0`은 frozen 설계상 **Pilot signal**일 뿐 통계적 확증이나 PASS 판정이 아니다.

### 2.1 Overall — 24 template clusters

| Endpoint | Arithmetic − Selector | 95% cluster-bootstrap CI | `d_z` | Pilot signal |
|---|---:|---:|---:|---|
| Layer 31 LiReF projection | +0.3359 | [-0.1948, +0.7906] | +0.2441 | NO |
| `L31N13336` contribution | -0.00728 | [-0.04652, +0.02514] | -0.0721 | NO |
| `L29H30` contribution | -0.000035 | [-0.001004, +0.000895] | -0.0139 | NO |
| `L30H6` contribution | +0.001259 | [-0.000330, +0.002898] | +0.2948 | NO |
| `L29H31` contribution | +0.000096 | [-0.000421, +0.000628] | +0.0697 | NO |

**Overall에서는 5개 primary endpoint 모두 CI가 0을 포함했다.** 따라서 전체 192 pair를 합친 수준에서 Arithmetic과 Selector의 일관된 내부 차이를 확보했다고 말할 수 없다.

### 2.2 Family-level Pilot signals

| Family | Endpoint | Arithmetic − Selector | 95% CI | 해석 범위 |
|---|---|---:|---:|---|
| points-balance | Layer 31 LiReF | +0.7724 | [+0.3267, +1.1817] | Arithmetic이 더 R 방향인 Pilot signal |
| temperature | `L31N13336` | -0.03002 | [-0.04457, -0.01345] | Arithmetic에서 해당 R-direction contribution이 더 작음 |
| points-balance | `L29H30` | -0.000940 | [-0.001945, -0.000063] | 크기가 작은 음의 component Pilot signal |

나머지 family × endpoint 조합은 CI가 0을 포함했다. 특히 `L30H6`과 `L29H31`은 어떤 family에서도 Pilot signal을 보이지 않았다.

### 2.3 Family interaction

사전에 보고하도록 정한 interaction 중 CI가 0을 제외한 것은 하나였다.

```text
Layer 31 LiReF:
points-balance − temperature = +1.2336
95% CI = [+0.5971, +1.8579]
```

이는 Arithmetic−Selector의 Layer 31 LiReF 차이가 semantic family에 따라 달라질 가능성을 지지하는 Pilot signal이다. 다른 endpoint의 family interaction CI는 모두 0을 포함했다.

과거 exploratory 결과에서 파생된 `L31N13336` directional interaction 두 개는 재현 기준을 충족하지 못했다.

- object-count − temperature: `-0.01410`, CI `[-0.09105, +0.04071]`
- points-balance − temperature: `+0.08230`, CI `[-0.00689, +0.15977]`

따라서 기존의 family-specific sign-reversal 가설을 확인했다고 말할 수 없다.

## 3. Behavioral 제한과 secondary 진단

Frozen v3.1 Calibration은 세 family 모두 FAIL이었고 behavioral equivalence를 확보하지 못했다. 이번 Pilot에서도 Arithmetic one-token generation correct는 `16/192`뿐이었다. 따라서 family-level 내부 신호가 산술 성공, 실패 또는 난이도 차이와 완전히 독립적이라고 볼 수 없다.

ADD/SUB, Arithmetic correct/incorrect와 behavioral margin correlation은 frozen 설계에 따라 secondary diagnostic으로만 저장했다. 이 subset 결과로 primary population을 바꾸거나 후보를 선택하지 않는다.

## 4. 현재 말할 수 있는 것

현재 근거는 다음 정도로 제한한다.

> **전체 192 pair에서 공통된 내부 차이는 확인되지 않았지만, Layer 31 LiReF의 Arithmetic−Selector effect가 semantic family에 따라 달라질 가능성과 일부 family-specific component signal이 관찰됐다.**

특히 points-balance에서는 Layer 31이 Arithmetic에서 더 R 방향이었고, temperature에서는 `L31N13336` contribution이 반대 방향이었다.

## 5. 말할 수 없는 것

- Arithmetic이 일반적으로 R representation을 만든다는 주장
- `L31N13336`이 Reasoning neuron 또는 R/M mediator라는 주장
- 기존 4개 component가 공통 R/M feature를 전달한다는 주장
- independent replication 또는 confirmatory evidence
- component mediation 또는 causal mechanism

이번 run은 같은 Calibration 문항을 사용한 limited same-sample Pilot이며 behavioral Calibration도 FAIL 상태다.

## 6. Safety와 provenance

- prompt scalar rows: `384`
- pair difference rows: `192`
- template cluster rows: `24`
- output hash 검증: 모두 PASS
- capture hook 제거: PASS
- raw hidden/candidate tensor 저장: 없음
- candidate 추가·재선정: 없음
- patching / suppression / intervention: 없음
- execution authorization SHA-256: `db5645b30d932e8ffd4fe458746befc0815d4aebd92f77d133b1e0390616deb8`
- implementation SHA-256: `b51e0ed5e5500fffdd98c79057fe1aabbe619fa967e33affdb9a674be5e10455`
- static review SHA-256: `e62de0a136998c26a6b01ff7079a2e8d74eabe236df24e3e15fc74d24fc9159e`
- run manifest SHA-256: `0d8d7a922461597df5028e6a3da22cf44385aea7f1d6fc4b65a0395612043f0d`
- summary SHA-256: `63afba0b776165c9d02c5866ecef7d082244f1765e6d9e57d6ff0a0fd95baa98`

## 7. 다음 gate

이 결과만으로 intervention을 자동 승인하지 않는다. 먼저 behavioral limitation과 overall null을 포함해 Pilot 결과를 검토하고, 후속 작업을 한다면 별도의 사전 설계·독립 데이터·실행 승인이 필요하다.
