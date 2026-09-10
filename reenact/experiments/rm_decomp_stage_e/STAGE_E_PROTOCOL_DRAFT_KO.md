# Stage E — Feature-to-Pathway Mechanism Protocol

상태: **DRAFT — 사람 검토·동결 전, Pilot 구현 금지**  
목적: Stage C에서 causal contribution을 보인 4개 component를 고정하고, 이들이 어떤 controlled input feature를 어떤 내부 경로로 전달해 LiReF representation과 행동에 영향을 주는지 검증한다.

## 0. 범위와 현재 증거

현재까지 확립된 범위는 다음과 같다.

- Stage A: 자연 R/M 데이터의 representation gap과 연관된 layer/module/head/neuron 위치를 찾았다.
- Stage B: 일부 후보가 relation semantics와 task relevance 변화에 민감했다. 이는 feature sensitivity이지 R/M 분리 원인의 증명이 아니다.
- Stage C: 5개 후보를 마지막 prompt token에서 억제했고 4개가 frozen criterion에 따라 자연 R/M gap 감소를 보였다. 이는 해당 측정치에 대한 causal contribution 근거이며 일반 reasoning/memorization의 필요성 증명은 아니다.
- Stage D: 관계 조합 요구와 parametric factual-memory dependence를 2×2로 조작했으나 primary feature criterion과 4개 후보 patch criterion이 모두 실패했다. 해당 실행은 두 명의 blind human audit가 없는 `v2_d07_exploratory`였으므로 confirmatory claim을 허용하지 않는다. 같은 operationalization은 Stage E에서 재사용하지 않는다.

Stage E는 A–D를 다시 탐색하거나 새로운 전체 후보 ranking을 만드는 단계가 아니다.

## 1. 연구 질문

주 질문은 다음과 같다.

> 어떤 controlled input feature 변화가 기존 causal 후보의 특정 input/output pathway state를 변화시키며, 그 state 변화가 frozen LiReF direction의 R/M representation과 선택적 행동 변화로 전달되는가?

이를 네 질문으로 분리한다.

1. **Feature effect:** 해당 feature만 바꾸면 LiReF score와 행동이 사전 방향으로 변하는가?
2. **Candidate response:** 그 변화가 기존 4개 후보 state를 재현성 있게 바꾸는가?
3. **Pathway contribution:** 후보 내부의 특정 pathway를 억제·교체하면 feature effect가 control보다 더 감소하는가?
4. **Rescue:** 손상된 pathway에 clean feature-conditioned state를 복원하면 잃어버린 effect가 회복되는가?

## 2. Stage E 인과 가설 구조

검증할 구조는 다음과 같다.

```text
controlled input feature
        ↓
detector/router 후보 state
        ↓
writer 후보 state
        ↓
frozen LiReF score / |R-M gap|
        ↓
correct-answer probability / generation accuracy
```

`detector`, `router`, `writer`는 분석적 역할명이다. 해당 경로의 feature response, intervention attenuation, control specificity, rescue가 모두 확인되기 전에는 기능이 확정된 것으로 쓰지 않는다.

### 2.1 증거 수준

| 관찰 | 허용되는 해석 |
|---|---|
| feature와 state의 상관/평균 차이 | feature association 또는 sensitivity |
| randomized pathway intervention 후 effect 감소 | 측정한 effect에 대한 causal contribution |
| 한 pathway 억제로 effect가 완전히 사라짐 | 해당 조건·개입 범위 안에서 necessity 후보; 일반 necessity 아님 |
| 반대 condition state patch로 effect 이동 | transfer evidence; 이것만으로 sufficiency 확정 금지 |
| pathway 손상 후 clean state reinjection으로 회복 | 해당 state가 손실 효과를 매개한다는 rescue evidence |
| representation만 변화 | 행동 mechanism 확정 금지 |
| 행동만 변화 | LiReF representation mediation 확정 금지 |

## 3. 고정 artifact와 후보

### 3.1 재사용할 frozen artifact

Stage E는 아래 Stage A artifact를 그대로 읽고 새 데이터로 direction을 재계산하지 않는다.

| Artifact | 경로 | SHA-256 / identity |
|---|---|---|
| Stage A input identity | `AI/reenact/liref_outputs/rm_decomp/v2/a_core/checkpoints/input_manifest.json` | file `5a60f6878a2ba4e9b2df7571bde7fd381edc0832c4a8b8ee0a9131971ee51e0b`; identity `7f0f7f020533b68e20e7b82d39dcacddad9f6e350dd599df6a40be667f8ec9f7` |
| Discovery LiReF directions | `AI/reenact/liref_outputs/rm_decomp/v2/a_core/checkpoints/discovery_liref_directions.pt` | `55647779ecf44a33143f66800af9ae3b2767d34b99b8877abd3711b6bba6adf6` |
| Stage A candidate manifest | `AI/reenact/liref_outputs/rm_decomp/v2/a_core/manifests/candidate_manifest.json` | `244b7397790fc71224ed77aafb4b4a1f267cd2dbebe67336b8560d67acbb52b9` |
| Stage C causal design | `AI/reenact/liref_outputs/rm_decomp/v2/c_causal_v2_c01/manifests/causal_design.json` | `96db9e6dae8d1a6ec75ac7933b19ac657b41a1306260948dd8d4e4ce7e6cd697` |
| Model config | `AI/reenact/models/Meta-Llama-3-8B/config.json` | `2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441` |
| Model parameter set | Stage C causal design에 기록된 4 shard | combined checksum `a86eda67086313c3b20d92a471820455c0ab6a9489db1870186980b0027bcb0b` |
| Prompt | `Q: {question}\nA: ` | `924c8cb05f84bdcad8da83b67dc617ffbbde6e20d929bfa8b45fd451f4a468e1` |

실행 전 이 값 중 하나라도 다르면 중단한다. Stage E 문서 동결 시 이 표 자체도 manifest에 복제한다.

### 3.2 고정 후보 4개

후보 ID의 layer/head/neuron은 모두 **0-based model index**다.

| 후보 | 종류 | 0-based 위치 | Stage C card SHA-256 | Stage E pathway 단위 |
|---|---|---|---|---|
| `L31N13336` | FFN neuron | layer 31, neuron 13336 | `643ca419bd3cd33b5afe6d2ab4c9871c7eb95db25f1bb91542e417f6ae0cbbe7` | gate row, up row, gated scalar, down column/output |
| `L29H00030` | Attention head | layer 29, query head 30 | `9b5a9a7df44599255f0af527c6f7e953e9375825580e922a89fd57950ba5769e` | Q head 30, shared K/V group 7, pre-O head block 30, O block 30 |
| `L30H00006` | Attention head | layer 30, query head 6 | `4aeba68ca9b7635d7056895030dc9e1d8a04467516fb95fda02baef97b66c19d` | Q head 6, shared K/V group 1, pre-O head block 6, O block 6 |
| `L29H00031` | Attention head | layer 29, query head 31 | `1465c335c27d820e70fab2b86c32f3b17ecfbf4eb1d07154659d6f630ccdc626` | Q head 31, shared K/V group 7, pre-O head block 31, O block 31 |

Stage E에서 이 후보를 교체하거나 결과가 좋은 새 후보를 추가하지 않는다. 새 후보 탐색이 필요해지면 별도 exploratory study로 분리한다.

문서 승인 뒤 이 4개 ID와 위 card hash만 담은 Stage E candidate manifest를 생성하고 그 hash를 frozen design에 기록한다. 기존 Stage C의 5개 후보 set hash를 4개 후보 set hash로 잘못 재사용하지 않는다.

## 4. Feature 가설 생성과 동결

### 4.1 가설 생성에 허용되는 자료

- Stage A Discovery 자연 문항에서 4개 후보의 activation/output이 큰·작은 사례
- Stage B/B-extension 결과와 실패 결과를 포함한 candidate card
- Stage C의 gap 및 behavior effect
- 선행논문에서 제안한 feature 범주
- 연구자가 사전에 정의한 계산 요구 또는 정보 의존성

Stage A Validation, 새 controlled Confirmatory 결과, Pilot의 item별 유리한 사례는 새 feature 선택에 사용하지 않는다.

### 4.2 가설 manifest에 필수인 항목

각 feature 가설은 다음을 결과 보기 전에 기록한다.

- `feature_id`, 정의, 이론적 이유
- 연결을 예상하는 후보와 pathway; 반응하지 않을 것으로 예상하는 후보/control
- 원본/변형 condition과 바뀌는 단일 요소
- 반드시 같아야 하는 요소: 정답, entity/fact chain, domain, 답 형식, 난이도, 정보량 등
- feature source span과 tokenizer token index 생성 규칙
- LiReF effect의 사전 방향과 candidate-state effect의 사전 방향
- competing hypothesis와 이를 구분할 negative control
- Stage D operationalization과의 차이
- Pilot용 template family 목록
- confirmatory로 진입하는 객관적 gate
- 작성자, 승인자, timestamp, manifest hash

### 4.3 가설 개수

한 후보에 임의로 많은 feature를 붙이지 않는다. 후보당 최대 가설 수와 전체 primary family 수는 사람이 Pilot 전에 고정한다. Pilot 실패 후 같은 run에서 새 가설을 추가하지 않는다.

## 5. Feature operationalization 기준

하나의 feature pair는 가능한 한 다음 조건을 만족해야 한다.

1. **Targeted change:** 목표 feature만 다르고 나머지 의미 구조는 같다.
2. **Answer invariance:** 두 condition의 정답과 허용 답 문자열이 같다. feature 자체가 답을 바꾸는 가설은 별도 contrast로 분리한다.
3. **Token control:** exact-token-length matched primary subset을 사전에 만들고, 전체 set은 secondary로 둔다.
4. **Span annotation:** 바뀐 span, 동일 source span, answer span, distractor span을 문자 offset과 token index로 저장한다.
5. **Task relevance:** 목표 feature가 답 산출에 필요한 condition과 불필요한 matched condition을 구분한다.
6. **Lexical diversity:** 같은 의미 feature를 서로 다른 template family와 lexical realization으로 구현한다.
7. **Difficulty calibration:** baseline correct-answer probability와 generation accuracy가 바닥·천장 효과를 피하는 범위에 있는지 Pilot에서 확인한다.
8. **No leakage:** 정답 문자열이나 정답을 직접 연상시키는 표면 단서가 한 condition에만 나타나지 않는다.
9. **Human audit:** Confirmatory 문항은 model result를 보지 않은 독립 reviewer가 fact, 문법, 단일-feature 변경, 정답 동일성, relevance를 검수한다.

Stage D에서 사용한 `supplied/direct/composition/parametric` A–D template를 문구만 바꿔 재사용하지 않는다. 같은 개념을 다시 시험하려면 feature의 조작 단위, matched control, 길이 처리, source span, 예상 mechanism 중 무엇이 실질적으로 달라졌는지 manifest에 명시해야 한다.

## 6. 개입 token 위치

### 6.1 마지막 prompt token

Stage A/C와 직접 연결되는 primary readout/intervention 위치다.

- layer별 residual output `h_out[layer, -1]`을 frozen direction에 projection한다.
- FFN output과 Attention pre-O/O output이 최종 question representation을 쓰는지 본다.
- 마지막 token intervention은 source feature를 어디서 읽었는지 증명하지 않는다.

### 6.2 source token/span

Feature가 입력되는 위치를 추적하는 위치다.

- controlled manifest의 문자 span을 frozen tokenizer로 token index에 매핑한다.
- Q는 마지막 prompt token query와 source-token query를 구분해 기록한다.
- K/V는 annotated source span에서만 patch하는 조건과 전체 prefix에서 patch하는 diagnostic 조건을 분리한다.
- FFN gate/up은 source span token별 activation과 마지막 prompt token activation을 별도 분석한다.

### 6.3 사용 순서

1. last-token readout으로 feature effect와 Stage C endpoint 연결을 확인한다.
2. source-span×layer patch로 feature 정보가 후보 layer에 도달하는 위치를 추적한다.
3. source-span pathway를 손상한 뒤 last-token candidate/output과 LiReF score 변화를 본다.
4. source-span clean state를 복원해 last-token effect가 회복되는지 본다.

## 7. FFN pathway 분석

Llama-3 SwiGLU의 layer `l`, neuron `j`에 대해 다음 값을 분리한다.

```text
g_lj(x) = gate_proj_l[j, :] · x
u_lj(x) = up_proj_l[j, :] · x
z_lj(x) = SiLU(g_lj(x)) · u_lj(x)
m_lj(x) = down_proj_l[:, j] · z_lj(x)
```

`L31N13336`에서:

- `gate_proj[13336, :]`: detector 입력 후보 A
- `up_proj[13336, :]`: detector 입력 후보 B
- `z_13336`: 두 입력이 결합된 activation
- `down_proj[:, 13336]` 및 `m_13336`: writer 후보

### 7.1 분리 개입

- gate-only activation patch: recipient의 `g`만 donor의 `g`로 보간하고 `u`는 유지
- up-only activation patch: `u`만 patch하고 `g`는 유지
- gate+up patch: 두 preactivation을 같은 donor에서 함께 patch
- gated-scalar patch: `z`만 patch해 input decomposition과 비교
- writer-output patch: `m` 또는 down-projected neuron output만 patch
- temporary weight intervention: gate row, up row, down column을 각각 또는 사전 정의 조합으로 개입

Patch 보간은 공통적으로 다음 형태를 쓴다.

```text
a_patch(alpha) = (1 - alpha) * a_recipient + alpha * a_donor
```

`alpha` dose set은 Pilot 전에 고정한다. Gate/up을 서로 다른 donor에서 섞는 실험은 primary에서 금지하고 diagnostic으로만 분리한다.

## 8. Attention pathway 분석과 GQA 통제

모델은 query head 32개, key/value head 8개인 GQA 구조다. 한 KV group은 query head 4개가 공유한다.

### 8.1 후보별 공유 구조

| 후보 | Q/O 단위 | 공유 K/V group | 같은 group의 query heads |
|---|---:|---:|---|
| `L29H00030` | head 30 | group 7 | 28, 29, 30, 31 |
| `L29H00031` | head 31 | group 7 | 28, 29, 30, 31 |
| `L30H00006` | head 6 | group 1 | 4, 5, 6, 7 |

따라서 layer 29의 두 후보는 K/V pathway를 공유한다. K 또는 V 개입 결과를 `H30만의 효과`, `H31만의 효과`라고 표현하면 안 된다.

### 8.2 역할 후보와 개입 단위

- Q head block: 어떤 query가 정보를 요청하는지에 대한 router 후보
- K shared-group block: source token이 어떤 query에 선택되는지에 대한 shared router 후보
- V shared-group block: source information content를 제공하는 shared writer-input 후보
- pre-O query-head block: attention-weighted head state
- O projection의 해당 head column block: residual stream writer 후보

### 8.3 필수 GQA controls

- Q/O single-head intervention과 K/V whole-shared-group intervention을 별도 family로 보고한다.
- K/V group intervention은 같은 group의 noncandidate sibling query heads를 control로 함께 측정한다.
- layer 29 H30과 H31의 joint Q/O intervention을 single-head 결과와 비교한다.
- 동일 크기의 다른 KV group intervention을 matched/random control로 둔다.
- attention weight만 보지 않고 pre-O state, O-projected output, LiReF projection을 함께 측정한다.

## 9. 데이터 설계

### 9.1 자연 데이터

Stage A Discovery 자연 R/M 문항은 feature 가설 생성과 기술 통계에만 사용한다. 이 데이터에서 보인 pattern은 confirmatory proof가 아니다.

Stage A Validation은 이미 A–C 평가에 사용됐으므로 Stage E의 새 controlled Confirmatory sample로 간주하지 않는다.

### 9.2 Controlled pair

각 `fact_chain_id × template_family × feature_condition` 조합에 대해 저장한다.

- 원문, condition, 목표 feature, changed span
- invariant facts와 correct answer
- tokenizer token length 및 exact-match 여부
- domain, difficulty/calibration stratum, 답 형식
- source/answer/distractor span token indices
- donor eligibility와 금지 donor
- human review 결과
- 생성 코드·schema·manifest hash

### 9.3 독립 단위

같은 template에서 숫자/entity만 바꾼 item을 독립 표본으로 세지 않는다.

- primary cluster: `template_family`
- factual feature가 포함된 경우 추가 cluster: `fact_chain_id`
- 통계: cluster bootstrap 또는 사전 지정 mixed-effects model
- 보고: `n_template_family`, `n_fact_chain`, `n_pair`, `pairs_per_cluster`

## 10. Pilot과 Confirmatory 분리

| 항목 | Pilot | Confirmatory |
|---|---|---|
| 목적 | variance, feasibility, calibration, implementation sanity | frozen hypothesis의 최종 검증 |
| template family | Pilot 전용 | 새 held-out family |
| fact/entity chain | Pilot 전용 | 겹치지 않는 held-out chain |
| 문항 재사용 | 금지 | 해당 없음 |
| feature 추가 | 다음 별도 run 설계에만 반영 | 실행 후 금지 |
| threshold 결정 | Pilot aggregate variance만 사용 가능 | 변경 금지 |
| 사람 audit | 내부 검수 가능 | model result를 보지 않은 blind audit 필수 |

Pilot의 유리한 item을 Confirmatory에 옮기지 않는다. Confirmatory manifest가 동결되고 human approval가 기록되기 전에는 GPU inference를 실행하지 않는다.

## 11. 주요 지표와 수식

### 11.1 LiReF score

Stage A Discovery에서 고정한 layer별 unit direction을 `r_hat_l`이라 한다.

```text
S_l(x) = h_out,l(x, last_prompt_token)^T r_hat_l
```

자연 R/M primary endpoint는 Stage C와 같은 layer 31 last-token score다.

```text
G_nat = mean[S_31(x) | R] - mean[S_31(x) | M]
```

Primary separation magnitude는 `|G_nat|`다. signed `G_nat`도 함께 보고한다.

### 11.2 Controlled feature effect

Feature `F`의 pair별 방향을 사전에 `sigma_F ∈ {-1, +1}`로 고정한다.

```text
e_F,l(i) = sigma_F * (S_l(x_i,F1) - S_l(x_i,F0))
E_F,l = cluster_mean[e_F,l(i)]
```

`E_F,l > 0`이 사전 예상 방향이 되도록 orientation만 정하며, 결과를 보고 `sigma_F`를 뒤집지 않는다.

### 11.3 Candidate response

- FFN: `Δg`, `Δu`, `Δz`, `Δ(m^T r_hat_l)`
- Attention: `ΔQ`, source-span attention distribution, `Δpre-O`, `Δ(O-head^T r_hat_l)`
- 각 값은 paired standardized effect와 cluster CI를 보고한다.

### 11.4 Pathway attenuation

Pathway `P` intervention 후 feature effect를 `E_F,l^P`라 한다.

```text
A_F,P,l = E_F,l^base - E_F,l^P
attenuation_fraction = A_F,P,l / E_F,l^base
```

분모가 primary feature criterion을 통과하고 사전 최소 절댓값보다 클 때만 fraction을 해석한다. 0으로 clipping하지 않고 signed 값과 CI를 그대로 보고한다.

자연 gap intervention은 다음과 같다.

```text
Delta_abs_G(P) = |G_nat^P| - |G_nat^base|
```

감소 방향은 `Delta_abs_G(P) < 0`이다.

### 11.5 Transfer

F1 donor state를 F0 recipient에 넣었을 때:

```text
T_F,P,l = sigma_F * (S_l(x_F0[P <- state_F1]) - S_l(x_F0))
```

반대 방향 F0→F1도 독립적으로 보고한다. Transfer는 경로가 정보를 운반할 수 있다는 증거이며 이것만으로 feature mechanism의 sufficiency를 확정하지 않는다.

### 11.6 행동 지표

- teacher-forced correct-answer sequence log probability
- strongest foil 대비 correct-answer log odds
- exact generation accuracy 및 정규화된 허용 답 accuracy
- multi-token answer는 첫 token 확률만 쓰지 않고 answer sequence 전체를 primary로 사용
- representation effect와 behavior effect는 별도 표와 correction family로 보고

## 12. 대조군과 donor control

### 12.1 Component controls

- **sham:** hook과 tensor copy는 수행하지만 원래 값을 다시 넣음
- **matched:** 같은 layer/type에서 Stage A contribution scale, activation variance, output norm이 유사한 noncandidate
- **random:** 같은 layer/type에서 seed로 비복원 추출한 noncandidate
- **wrong-layer:** 같은 component index 또는 같은 shape의 인접/사전 지정 다른 layer pathway

Matched distance, random seed, control 수는 Pilot 전에 고정한다. Control 결과를 보고 교체하지 않는다.

### 12.2 Donor controls

| Donor | 정의 | 목적 |
|---|---|---|
| self/sham | 같은 item·condition의 자기 state | patch 구현 자체의 무효 효과 확인 |
| same-condition | 같은 feature level의 matched realization state | donor 교체와 일반 state mismatch 효과 확인 |
| matched cross-condition | 같은 chain·answer의 반대 feature condition state | primary feature transfer |
| shuffled-chain | 다른 fact/entity chain의 목표-condition state | 정답·entity leakage와 content mismatch 확인 |
| wrong-feature | 목표가 아닌 feature만 다른 state | feature specificity 확인 |

Donor와 recipient는 가능하면 같은 정답, token length stratum, template-family 역할을 맞춘다. 불가능한 경우 해당 pair는 primary donor analysis에서 제외한다.

## 13. 성공 기준 구조

숫자 threshold는 이 DRAFT에서 임의로 정하지 않는다. 단, 동결할 criterion의 구조는 다음으로 제한한다.

### 13.1 Primary feature

다음을 모두 만족해야 한다.

- 사전 예상 방향
- cluster-aware 95% CI가 0을 제외
- 사전 정의 BH-FDR family에서 `q < q_primary`
- paired standardized effect `|d_z| ≥ SESOI_feature`
- template-family sign consistency `≥ consistency_feature`
- exact-token-length-matched subset에서 동일 방향과 최소 effect 충족
- 행동 feature라면 baseline calibration criterion 통과

### 13.2 Pathway causal contribution

- primary feature criterion이 먼저 통과
- candidate pathway attenuation/transfer의 CI가 0을 제외하고 FDR 통과
- `|d_z| ≥ SESOI_pathway`
- attenuation 또는 transfer fraction이 사전 최소치 이상
- sham, matched, random, wrong-layer의 개별/최대 효과보다 큼
- dose가 사전 순서대로 monotonic하거나 사전 contrast를 통과
- global-disruption threshold를 넘지 않음

### 13.3 Rescue

- pathway 손상으로 primary feature effect가 먼저 유의하게 감소
- clean state reinjection이 손실분의 `rescue_fraction_min` 이상 회복
- rescue CI가 0을 제외하고 FDR 통과
- same-condition, shuffled-chain, wrong-feature donor rescue보다 큼
- representation rescue와 behavior rescue를 별도로 판정

### 13.4 선택성

`R-specific` 또는 `M-specific`은 다음을 모두 만족한 후에만 사용한다.

- 목표 조건에서 candidate pathway intervention effect가 criterion 통과
- 반대 조건 effect보다 사전 최소 차이 이상 큼
- difference-in-differences CI가 0을 제외하고 FDR 통과
- difficulty/length/domain-matched subset에서 재현
- 전체 언어 능력 붕괴로 설명되지 않음

### 13.5 Multiple testing family

BH-FDR family는 적어도 다음을 분리한다.

1. primary feature effect
2. candidate-state response
3. pathway attenuation
4. transfer
5. rescue
6. representation endpoint
7. behavior endpoint
8. Attention GQA shared-group diagnostic

각 family에 들어갈 candidate×feature×pathway×dose contrast 목록을 Confirmatory manifest에 열거한다.

## 14. Pathway intervention 순서

각 feature는 다음 gate를 순서대로 통과한다.

1. baseline feature effect와 calibration 확인
2. 4개 후보 state response 측정
3. source-span×layer 정보 유입 위치 확인
4. candidate pathway별 activation suppression/patch
5. matched/random/wrong-layer 및 donor controls 비교
6. 여러 dose에서 attenuation/transfer 확인
7. joint pathway는 individual 결과를 본 뒤 고르는 것이 아니라 사전 지정 조합만 실행
8. representation과 behavior endpoint 동시 보고
9. global disruption diagnostic 통과 시 rescue 진입

앞 gate가 실패하면 다음 confirmatory gate를 실행하지 않는다. 실패한 feature에서 다른 후보를 사후 선택하지 않는다.

## 15. Activation rescue와 제한적 weight intervention

### 15.1 Activation rescue

Primary rescue는 다음 순서를 따른다.

```text
clean feature condition
→ candidate pathway를 억제/손상
→ LiReF 및 행동 effect 감소 확인
→ 같은 item의 clean feature-conditioned pathway state reinjection
→ effect 회복 측정
```

Independent rescue는 단순 reverse transfer를 중복해서 세지 않는다. 손상 intervention과 reinjection intervention이 분리돼야 한다.

### 15.2 Weight intervention

- FFN: gate row, up row, down column을 분리
- Attention: Q/O는 query-head block, K/V는 shared KV-group block
- 개입은 in-memory temporary copy에만 적용
- exact weight rule, norm constraint, dose는 Pilot 전에 하나로 고정
- 실행 전 parameter checksum, 개입 중 delta checksum, 원복 후 checksum을 저장
- 원복 checksum이 baseline과 다르면 해당 run은 무효

ROME형 rank-one edit는 factual feature와 FFN writer가 앞 단계에서 모두 통과했을 때만 허용한다. Attention 후보에는 ROME update를 직접 적용하지 않는다.

### 15.3 Global disruption diagnostics

- next-token `KL(base || intervention)`
- top-1 token change rate
- logit RMS change
- 일반 control question accuracy와 answer log probability
- prompt perplexity 또는 사전 지정 language-model quality metric

각 허용 threshold는 Pilot 전에 동결한다.

## 16. 실패 판정

다음 중 하나면 해당 claim은 FAIL이다.

- feature effect가 primary criterion을 통과하지 못함
- exact-length subset에서 방향이 뒤집히거나 최소 effect 미달
- candidate response는 있으나 pathway intervention이 control보다 크지 않음
- pathway effect는 있으나 broad disruption threshold 초과
- transfer만 있고 attenuation 또는 rescue가 없음
- 손상 효과가 없는데 rescue fraction만 계산함
- rescue가 shuffled/wrong-feature donor와 구분되지 않음
- representation만 바뀌고 preregistered behavior claim은 실패
- Confirmatory 후 hypothesis, pair, threshold, correction family를 수정함
- parameter checksum 원복 실패

실패 결과는 후보 교체나 기준 완화 없이 그대로 보고한다. 새로운 operationalization은 새 Stage E run으로 시작한다.

## 17. 가능한 주장

Criterion이 실제로 통과한 범위 안에서만 다음 표현을 허용한다.

- 특정 controlled feature가 이 모델·prompt 조건에서 frozen LiReF score를 사전 방향으로 이동시켰다.
- 특정 기존 candidate pathway state가 해당 feature 변화에 민감했다.
- 해당 pathway intervention이 feature-conditioned LiReF/behavior effect를 control보다 선택적으로 약화해 causal contribution 근거를 보였다.
- clean state reinjection이 손실 효과를 부분 회복해 pathway mediation 근거를 보였다.
- 양 조건 및 control 대비 선택성이 검증된 경우에만 `R-selective` 또는 `M-selective` pathway라고 표현한다.

## 18. 금지되는 주장

- `이 neuron은 reasoning neuron이다`, `이 head는 memorization head다`
- `R/M 분리의 유일한 원인이다`
- 부분 attenuation을 일반적 necessity로 표현
- transfer 하나를 sufficiency 증명으로 표현
- attention weight가 높다는 이유만으로 정보 전달 mechanism 확정
- shared K/V group 효과를 단일 query head 효과로 표현
- LiReF representation 변화만으로 실제 정답 행동 mechanism 확정
- 한 모델·한 dataset·한 prompt의 결과를 모든 LLM으로 일반화
- weight edit 성공을 원래 자연 계산의 저장 위치 증명으로 동일시

## 19. Stage D Feature 재사용 규칙

Stage D에서 실패한 두 operationalization을 그대로 재사용하지 않는다.

- 관계 조합 요구: 기존 A/B composition template의 길이·단계 구조만 반복 금지
- parametric factual-memory dependence: 기존 supplied/parametric contrast의 정보량·prompt 구조 반복 금지

같은 상위 개념을 다시 다룰 수는 있으나 다음을 모두 충족해야 한다.

1. 실패 원인을 구체적으로 기술한다.
2. 새로운 manipulation unit과 competing control을 제시한다.
3. exact-token-length matching과 source-span 정의를 개선한다.
4. Stage D template family와 fact chain을 재사용하지 않는다.
5. `stage_d_operationalization_delta`를 manifest에 기록하고 사람이 승인한다.

## 20. Pilot 진입 전 체크리스트

아래가 모두 체크되기 전에는 Pilot 코드를 작성하지 않는다.

- [ ] `LITERATURE_METHOD_COMPARISON_KO.md` 사람 검토 완료
- [ ] 본 protocol의 연구 질문·claim boundary 승인
- [ ] 4개 후보와 artifact hash 확인
- [ ] frozen LiReF direction 재사용 및 재계산 금지 확인
- [ ] primary feature hypothesis와 competing control 승인
- [ ] Stage D operationalization delta 승인
- [ ] source span/token mapping 규칙 승인
- [ ] FFN/Attention/GQA intervention tensor 단위 승인
- [ ] donor pool과 donor matching 규칙 승인
- [ ] component control matching 규칙·seed 승인
- [ ] primary endpoint와 contrast formula 승인
- [ ] SESOI, dose, CI, FDR, consistency, rescue threshold 승인
- [ ] Pilot sample size와 template/fact-chain 수 승인
- [ ] behavior answer scoring·generation rule 승인
- [ ] global disruption threshold 승인
- [ ] Pilot/Confirmatory non-overlap 규칙 승인
- [ ] weight edit 허용 gate와 원복 checksum 절차 승인
- [ ] 모든 승인 내용을 hash 가능한 frozen design manifest로 옮김

## 21. Pilot 구현 전 반드시 사람이 결정할 미결정 항목

다음 항목은 현재 문서에서 의도적으로 숫자나 선택지를 임의 확정하지 않았다.

1. **Primary feature 가설:** 무엇을 첫 Pilot에서 시험할지, 후보당 최대 몇 개를 허용할지
2. **Feature 방향:** 각 feature의 `sigma_F`와 R/M 해석 범위
3. **Operationalization:** exact pair/template, invariants, task-relevance control, Stage D와의 실질적 차이
4. **Primary endpoint:** layer 31을 단일 primary로 둘지, 사전 지정 layer profile을 공동 primary로 둘지
5. **SESOI:** feature, pathway, behavior 각각의 최소 표준화 효과크기
6. **표본수:** Pilot의 template family·fact chain·pair 수 및 Confirmatory power target
7. **Dose:** activation/weight intervention의 `alpha` 집합과 monotonic contrast
8. **Attenuation/transfer 기준:** 최소 fraction 및 control 대비 최소 차이
9. **Rescue 기준:** 최소 recovery fraction과 representation/behavior 동시 통과 여부
10. **Calibration:** baseline 정답 확률·accuracy·paraphrase consistency 허용 범위
11. **Control matching:** matched distance 변수·허용 거리·control 수·random seed
12. **Donor matching:** same-answer 요구, length 허용오차, donor 부족 시 제외 규칙
13. **통계:** cluster bootstrap 대 mixed-effects model, replicate 수, FDR `q`와 family membership
14. **Global disruption:** KL, top-1 change, logit RMS, 일반 accuracy 저하의 허용 한계
15. **Weight rule:** block scaling, mean clamp, matched-norm perturbation, 제한적 rank-one edit 중 pathway별 허용 방식
16. **Human audit:** Confirmatory reviewer 수, blind 절차, 합의 기준, 재검수 규칙
17. **ROME 원문 고정:** 공식 PDF 로컬 저장 경로와 SHA-256
18. **Stage E run naming:** Pilot/Confirmatory run ID와 output root

이 항목이 승인되고 frozen design manifest의 hash가 생성된 뒤에만 Pilot 구현을 시작한다.
