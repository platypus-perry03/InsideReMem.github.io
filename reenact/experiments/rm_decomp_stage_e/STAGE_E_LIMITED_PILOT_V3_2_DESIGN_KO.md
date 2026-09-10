# Stage E limited same-sample Pilot v3.2 상세 설계

상태: **FROZEN — IMPLEMENTATION/STATIC REVIEW ALLOWED — MODEL EXECUTION NOT AUTHORIZED**  
동결일: `2026-08-30`

## 1. 목적과 증거 수준

이 Pilot은 frozen v3 Calibration dataset의 전체 192 pair를 다시 사용하여 Arithmetic과 Selector 조건이 기존 LiReF direction 및 네 candidate의 scalar contribution을 선택적으로 변화시키는지 본다.

이 설계는 Calibration v3.1의 세 family FAIL을 변경하지 않는다. 결과는 **limited same-sample Pilot evidence**이며 independent replication, confirmatory evidence, mediation 또는 causal mechanism으로 해석하지 않는다.

## 2. 분석 모집단

- primary population: frozen v3 dataset 전체 `192 pairs / 384 prompts`
- semantic families: `object_count`, `points_balance`, `temperature`
- family당 8 template family, template당 8 pair
- primary 분석에서 correct-only filtering 금지
- 결과를 보고 family, template 또는 item 제외 금지
- 새 candidate 추가·재선정·ranking 금지

## 3. 고정 artifact

| Artifact | SHA-256 |
|---|---|
| v3 dataset | `d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1` |
| v3 design | `c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc` |
| Pilot continuation amendment v3.2 | `3e9d14c676c96ee6d950761d39faa176c0e9267d613856d1bd57ff826d41e5fe` |
| Baseline pair results | `c4562fcfb109083b8c501ac90af64ae5a8f6e6f7f33485619a86366aaec78e6a` |
| Stage A Discovery LiReF directions | `55647779ecf44a33143f66800af9ae3b2767d34b99b8877abd3711b6bba6adf6` |
| Stage A candidate manifest | `244b7397790fc71224ed77aafb4b4a1f267cd2dbebe67336b8560d67acbb52b9` |
| Stage C causal design | `96db9e6dae8d1a6ec75ac7933b19ac657b41a1306260948dd8d4e4ce7e6cd697` |
| Pilot v3.2 candidate manifest | `e9d20904967512fa73581d528eb3026a3142f1a5d2c8b0f878e847dbc7eeb233` |
| model config | `2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441` |

Discovery direction은 새 데이터로 다시 추정하거나 부호를 뒤집지 않는다. `r_hat_l`은 Stage A의 Reasoning mean minus Memorization mean을 정규화한 방향이다.

## 4. Readout 위치

- 입력은 dataset의 frozen `full_prompt` 문자열 그대로이며 정답 token을 붙이지 않는다.
- tokenizer는 local Meta-Llama-3-8B tokenizer, left padding을 사용한다.
- readout은 각 prompt의 마지막 non-padding prompt token이다. frozen prompt는 `A: `로 끝나며 tensor index는 `[:, -1, :]`이다.
- residual endpoint는 0-based layer 31 block output이다.
- candidate contribution은 candidate가 위치한 동일 layer의 frozen direction을 사용한다.

## 5. Primary scalar 정의

### 5.1 Layer 31 LiReF projection

```text
S_31(x) = h_out,31(x,last_prompt_token) dot r_hat_31
```

pair `i`의 condition effect:

```text
d_LiReF(i) = S_31(x_i,Arithmetic) - S_31(x_i,Selector)
```

양수는 Arithmetic condition이 Selector보다 frozen R 방향으로 더 이동했음을 뜻한다.

### 5.2 FFN neuron `L31N13336`

```text
z_31,13336(x) = SiLU(gate_proj[13336] x) * up_proj[13336] x
p_31,13336 = down_proj[:,13336] dot r_hat_31
c_L31N13336(x) = z_31,13336(x) * p_31,13336
```

Primary response는 `c(Arithmetic)-c(Selector)`다. signed `z`와 `|z|` 차이는 secondary diagnostic이다.

### 5.3 Attention candidates

candidate layer `l`, query head `h`에 대해:

```text
q_lh = O_proj[:, head_block(h)]^T r_hat_l
c_lh(x) = pre_O_lh(x,last_prompt_token) dot q_lh
```

Primary response는 candidate별 `c(Arithmetic)-c(Selector)`다. pre-O head L2 norm 차이는 secondary diagnostic이다. Attention weight는 primary metric이 아니다.

대상은 `L29H00030`, `L30H00006`, `L29H00031`뿐이다.

## 6. Aggregation

endpoint `j`의 pair effect를 `d_ij`라 한다.

1. pair: 같은 context의 `Arithmetic - Selector`
2. template cluster: 같은 template의 8 frame 평균
3. family: 8 template-cluster 평균
4. overall: 세 family mean의 동일가중 평균. 각 family cluster 수가 8로 같으므로 24 cluster 단순평균과 같아야 하며 두 계산의 수치 일치를 검사한다.

item 192개를 독립표본으로 취급한 p-value는 계산하지 않는다.

## 7. Family interaction

각 endpoint에 대해 세 family effect와 아래 세 pairwise interaction contrast를 전부 보고한다.

```text
I(object_count, points_balance)
I(object_count, temperature)
I(points_balance, temperature)
```

`I(f,g)=E_f-E_g`다. 특정 interaction만 결과를 보고 선택하지 않는다.

이전 exploratory 결과에서 유도된 사전 가설은 다음 하나다.

```text
L31N13336:
E_object_count - E_temperature > 0
E_points_balance - E_temperature > 0
```

이는 exploratory-derived hypothesis이며 독립 confirmatory 가설이 아니다. 나머지 endpoint의 family interaction은 방향을 정하지 않은 descriptive Pilot 결과다.

## 8. 통계와 Pilot signal 표기

- independent unit: template family
- family cluster 수: 8
- overall cluster 수: 24
- bootstrap: family 안에서 template cluster를 replacement resampling
- overall bootstrap: 각 family에서 8 cluster를 따로 재표집한 뒤 세 family mean을 동일가중
- repetitions: `10,000`
- seed: `20260831`
- CI: percentile 95%
- family 및 overall에 sample-SD(`ddof=1`) `d_z`와 cluster sign count 보고
- interaction은 두 bootstrap family mean의 차이와 95% CI를 보고
- NaN, infinity, missing pair/cluster는 run FAIL

Pilot signal은 `cluster-bootstrap 95% CI가 0을 제외`할 때 표시할 수 있지만, 여러 endpoint를 탐색하는 same-sample Pilot이므로 confirmatory significance 또는 PASS로 부르지 않는다. p-value, BH-FDR, hard success threshold 및 candidate ranking은 사용하지 않는다.

## 9. Secondary diagnostics

다음은 primary 전체표본 결과를 제시한 뒤에만 보고한다.

- ADD vs SUB의 family별 effect
- Arithmetic one-token generation correct vs incorrect
- Arithmetic forced-choice correct vs incorrect
- baseline arithmetic margin과 internal pair effect의 template-cluster Spearman correlation
- FFN signed `z`, absolute `z`; Attention pre-O L2 norm

secondary subset은 primary 결과를 대체하지 않으며 subset별 p-value나 candidate selection에 사용하지 않는다. behavior는 frozen Baseline pair result를 `pair_id`로 join하며 다시 정의하지 않는다.

## 10. 출력 schema

- prompt scalar rows: 384
- pair difference rows: 192
- template-cluster rows: 24
- family summary: 3 family × 5 primary endpoints
- overall summary: 5 primary endpoints
- interaction table: 5 endpoints × 3 family contrasts
- secondary diagnostics
- behavioral Calibration summary와 internal result의 병렬 기록
- environment, implementation, authorization, input/output hash provenance

raw hidden-state tensor, raw pre-O tensor, raw FFN intermediate tensor와 checkpoint는 저장하지 않는다. 저장 가능한 것은 scalar와 집계값뿐이다.

## 11. 안전 및 다음 gate

read-only capture hook은 scalar 계산을 위해 필요한 layer 31 output, 후보 layer pre-O 및 L31N13336 `z`에만 제한한다. intervention hook, tensor replacement, patching, suppression, amplification 및 weight modification은 금지한다.

현재 허용:

- Pilot implementation
- model-free test
- static safety/schema review

현재 금지:

- model loading/forward/GPU Pilot 실행
- direction 재추정
- candidate 추가 탐색
- raw state 저장
- intervention/patching/suppression
- independent/confirmatory/mediation/causal claim

별도의 hash-locked execution authorization 전에는 Pilot을 실행하지 않는다.

