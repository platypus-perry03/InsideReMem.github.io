# Stage E Independent Replication v4 Behavioral Calibration 결과

상태: **COMPLETE — BOTH FAMILIES FAIL — INDEPENDENT REPLICATION BLOCKED**  
실행일: `2026-08-30`  
run ID: `stagee_v4_calibration_20260830_01`

## 1. 실행 범위

- 모델: local `Meta-Llama-3-8B` base
- calibration pool만 사용: 128 pairs / 256 prompts
- family: points-balance / temperature, 각 64 pairs
- device / batch / dtype: `cuda:1` / `8` / `float32`
- 측정: A/B teacher-forced margin과 한 토큰 greedy A/B 출력
- independent replication pool: 접근하지 않음
- LiReF, 후보 component, hidden state, hook, intervention: 사용하지 않음
- human audit: `not_performed`; human-audited evidence가 아님

## 2. Frozen 기준 판정

| Family | Arithmetic FC | Selector FC | Arithmetic generation | Selector generation | mean `D_k` | `d_z` | 최종 |
|---|---:|---:|---:|---:|---:|---:|---|
| points-balance | 33/64 | 37/64 | 33/64 | 37/64 | -0.01093 | -0.32391 | **FAIL** |
| temperature | 32/64 | 37/64 | 32/64 | 37/64 | -0.01009 | -0.48141 | **FAIL** |

Frozen count range는 forced-choice 40–60/64, generation 32–60/64다. 두 조건의 count gap, generation range와 mean `D_k`는 대체로 균형적이었지만 다음 hard criterion을 충족하지 못했다.

- points-balance: Arithmetic·Selector forced-choice가 모두 하한 미달
- temperature: Arithmetic·Selector forced-choice가 모두 하한 미달, `|d_z| > 0.35`

모든 기준을 동시에 충족해야 family PASS이므로 `passed_families=[]`다. 결과 확인 후 threshold, item, template 또는 family를 변경하지 않았다.

## 3. 실패의 의미

v2/v3 계열의 direct-retrieval ceiling은 제거됐다. 그러나 이번에는 두 조건이 모두 너무 어려워져 모델이 A/B를 거의 chance 수준으로 선택했다.

- points-balance generation 선택: Arithmetic `A=51, B=13`; Selector `A=45, B=19`
- temperature generation 선택: Arithmetic `A=38, B=26`; Selector `A=39, B=25`
- 정답 A/B 배치는 각 조건에서 정확히 32:32였으므로, 특히 points-balance에는 강한 A 선택 편향이 남았다.
- Arithmetic ADD가 SUBTRACT보다 높았지만 두 조건 모두 task solution evidence가 충분하지 않다.

따라서 이번 결과는 “두 조건의 난이도가 완벽히 같아졌다”는 뜻도, “R/M 차이가 없다”는 뜻도 아니다. 정확한 진단은 다음과 같다.

> **Absolute answer exposure와 ceiling은 통제했지만, prompt의 다단계 symbolic binding 부담이 너무 커서 두 조건 모두 안정적으로 풀리지 않았다.**

## 4. 재현 gate

Frozen v4 설계는 points-balance Calibration PASS를 primary Layer 31 independent replication의 필수 조건으로 정했다. points-balance가 FAIL했으므로:

- primary independent replication: **실행 금지 유지**
- points-balance–temperature interaction replication: **실행 금지 유지**
- Layer 0–30 탐색, 후보 4개 측정: **실행하지 않음**
- intervention / patching / suppression: **실행하지 않음**

이미 생성·감사된 replication pool은 결과를 보지 않은 채 보존한다. 이번 FAIL을 이유로 해당 pool을 사후 실행하지 않는다.

## 5. Provenance

- design SHA-256: `0382a059f2ac3578446e772939a10dc6911d11b7a90bb4cb0f7bd78ed5ebe106`
- calibration dataset SHA-256: `e4b660057b8103533c3303c8defc8a3b03268fac036ff3b8232c9e20662f6ded`
- implementation SHA-256: `299d8531c6ec31eb1f09890456c0b1b3ae77da3915f71dfdf75527325ec7f28f`
- static review SHA-256: `48ba9d7f8da91c8c349a68d8f74bb6545eb535ec9a2e524c54da71e32bc73b97`
- execution authorization SHA-256: `ecb7542a5a06f4f743dbf9d0959c93a5f910e229184edf3d7c14b0da6ab6142f`
- pair results SHA-256: `6cf2ff393c0f80d4c44aa5aacf3cd261768e1557a5f5d4c90758cd6f68934ce7`
- summary SHA-256: `ee2ce66cb00e3b2cc9209e49683134c8f7024203748cbfd9d78665b3310f115a`
- run manifest SHA-256: `1458c7c9a83a3c2a9bde07b21b309871b3c4a1b6584d271237fe0748889cbbc0`

## 6. 다음 연구 판단

같은 문제를 더 복잡하게 만드는 방향은 피한다. 후속 설계가 필요하면 다음을 사전 명세해야 한다.

1. A/B 선택 편향을 줄이되 A/B 역할 1:1 counterbalance는 유지
2. arithmetic과 selector 모두의 binding 단계 수를 줄여 absolute solvability를 회복
3. 정답 숫자 노출과 meaningful foil 통제는 유지
4. 새 calibration 문항으로 behavioral gate를 다시 확인
5. gate 통과 전에는 보존된 independent replication pool이나 Layer 31 재현 실험을 실행하지 않음

