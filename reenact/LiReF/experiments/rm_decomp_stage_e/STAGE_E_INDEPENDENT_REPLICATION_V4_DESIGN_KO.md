# Stage E Independent Replication v4 설계

상태: **FROZEN — DATASET BUILD/AUDIT ALLOWED, MODEL EXECUTION NOT AUTHORIZED**  
동결일: `2026-08-30`

## 1. 목적

Limited same-sample Pilot v3.2에서 관찰된 다음 신호를 기존 192문항과 겹치지 않는 새 문항으로 검증한다.

> **Primary:** points-balance에서 Layer 31 frozen LiReF projection의 `Arithmetic - Selector`가 0보다 큰가?

추가로 points-balance와 temperature의 차이가 재현되는지 사전 지정 secondary interaction으로 확인한다. 이번 연구는 새 component나 좋은 layer를 탐색하는 실험이 아니다.

## 2. 두 개의 독립 pool

- Calibration pool: `2 families × 8 templates × 8 frames = 128 pairs / 256 prompts`
- Replication pool: `2 families × 8 templates × 8 frames = 128 pairs / 256 prompts`
- 두 pool 사이 template ID, 이름, label, numeric block과 완성 prompt 중복 금지
- v2/v2.1/v2.1.1/v3의 item ID, 완성 prompt와 normalized template skeleton 재사용 금지
- Calibration 문항은 replication model run에 사용하지 않음

## 3. 조작

두 조건은 동일 context와 동일 A/B 출력 계약을 사용한다.

- Arithmetic: start와 delta를 찾아 ADD/SUB를 수행한 뒤 A/B 후보 중 결과를 선택
- Selector: active tag를 따라 두 값 중 하나를 찾은 뒤 A/B 후보 중 결과를 선택

두 조건 모두 correct와 alternative numeric value가 context에 한 번씩 노출된다. 따라서 v3의 `Arithmetic 비노출 vs Selector 노출` 차이를 제거한다.

```text
Answer with A or B only.
Answer:
```

정답 continuation은 Meta-Llama-3-8B tokenizer에서 정확히 한 token인 ` A` 또는 ` B`다.

## 4. Counterbalance

각 template은 고정된 `OA(8,6,2,2)` strength-2 matrix의 8 frame을 사용한다.

| Row | ADD/SUB | selector entry | arithmetic A/B | selector A/B | label role | block order |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 0 | 1 | 1 |
| 3 | 0 | 1 | 0 | 1 | 0 | 1 |
| 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| 5 | 1 | 0 | 0 | 1 | 1 | 0 |
| 6 | 1 | 0 | 1 | 1 | 0 | 1 |
| 7 | 1 | 1 | 0 | 0 | 1 | 1 |
| 8 | 1 | 1 | 1 | 0 | 0 | 0 |

각 factor는 4:4이고 임의의 factor pair의 네 2×2 cell은 각각 두 번 등장한다. 이 matrix는 nuisance counterbalance용이며 interaction factor 추정용이 아니다.

## 5. Numeric 및 foil

- start: 40–69
- delta: 2–9
- Arithmetic correct: ADD=`start+delta`, SUB=`start-delta`
- Arithmetic foil: frozen wrong-operation result
- Selector correct/foil: 서로 다른 두 ledger value
- 네 candidate value는 모두 두 자리, 서로 다르고 start/delta와 충돌하지 않음
- Arithmetic/Selector correct choice A/B는 독립 counterbalance
- seed: `20260901`

## 6. Calibration scoring과 gate

각 prompt에서:

```text
M(x) = logP(correct choice token | prompt) - logP(incorrect choice token | prompt)
D_k = mean_8frames[M(arithmetic) - M(selector)]
```

Family마다 64 pairs를 사용하며 다음을 모두 만족해야 PASS다.

- forced-choice correct: 각 조건 `40–60 / 64`
- forced-choice condition count gap: `≤6 / 64`
- one-token A/B generation: 각 조건 `32–60 / 64`
- generation condition count gap: `≤6 / 64`
- `|mean_k(D_k)| ≤ 0.40 nat`
- `|d_z| ≤ 0.35`, sample SD `ddof=1`
- missing/NaN/infinity: FAIL
- 결과를 보고 item/template 제거 또는 threshold 변경 금지

진행 규칙:

- points-balance PASS: primary replication 실행 가능
- points-balance와 temperature 모두 PASS: 사전 interaction 검증 가능
- points-balance FAIL: independent replication model run 중단
- temperature만 FAIL: temperature/interaction model run 금지, points-balance primary만 가능

## 7. Independent replication

### Primary

- population: replication points-balance 전체 64 pairs
- endpoint: Layer 31 last-prompt-token frozen LiReF projection
- pair effect: Arithmetic − Selector
- independent unit: 8 template clusters
- success rule: template-cluster bootstrap 95% CI lower bound `>0`
- bootstrap: 10,000회, seed `20260901`

### Prespecified secondary

- 양 family가 Calibration PASS인 경우 Layer 31 `points-balance − temperature` interaction
- success label이 아니라 `replicated secondary signal`로만 보고

### Exploratory only

- Layer 0–30 frozen LiReF projection trajectory
- `L31N13336`, `L29H30`, `L30H6`, `L29H31` scalar contributions
- behavior correct/incorrect와 ADD/SUB diagnostics
- exploratory 결과로 primary를 교체하거나 candidate/layer를 새로 선택하지 않음

## 8. Audit 정책

- automatic structural/tokenizer/non-reuse audit 전수
- primary AI linguistic audit 전수
- 별도 adversarial AI-only audit 전수
- human audit: `not_performed`
- human-audited evidence 주장 금지
- audit 하나라도 FAIL이면 해당 dataset version에서 model execution 금지

## 9. Claim limits

Replication primary가 성공해도 다음만 주장한다.

> 새 points-balance 문항에서도 Arithmetic이 Selector보다 Layer 31 frozen LiReF coordinate에서 더 높게 나타났다.

Reasoning neuron, R/M 분류 경계, component mediation, causal mechanism, 모델 일반성은 주장하지 않는다. 이번 데이터는 Meta-Llama-3-8B base 한 모델에 대한 검증이다.

## 10. 현재 permission

- deterministic builder, dataset 생성, model-free audit: 허용
- tokenizer CPU 사용: 허용
- model loading/forward/GPU: 미승인
- LiReF runtime loading: 미승인
- intervention/patching/suppression: 미승인

다음 순서는 builder → 두 pool 생성 → automatic/AI audits → Calibration implementation/static review → 별도 execution authorization이다.
