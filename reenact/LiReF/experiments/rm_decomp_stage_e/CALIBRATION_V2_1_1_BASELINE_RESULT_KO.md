# Stage E v2.1.1 protocol-deviating diagnostic baseline 결과

상태: **COMPLETE — PROTOCOL-DEVIATING DIAGNOSTIC RUN; ALL THREE FAMILIES FAILED; OFFICIAL CALIBRATION NOT PERFORMED**  
실행일: 2026-08-30  
run ID: `calv211_baseline_20260830_02`

> **Authoritative interpretation:** 이 실행은 human audit을 수행하지 않은 채 진행되었으므로 공식 Baseline Calibration이 아니다. 아래 PASS/FAIL 계산은 frozen threshold를 진단적으로 적용한 결과이며, 공식 Calibration family 판정으로 사용하지 않는다. 현재 통합 상태는 `STAGE_E_CURRENT_STATUS_KO.md`를 따른다.

## 1. 실행 전 gate

- frozen v2.1.1 dataset: 192 pairs / 384 prompts / 24 template families
- automatic audit: PASS
- AI-only linguistic audit: 192/192 PASS, 10개 기준 1,920/1,920 YES
- human audit: `not_performed`
- human audit gate: `not_satisfied`
- waiver requested by user: `true`
- waiver valid under frozen protocol: `false`
- human-audited evidence: `false`
- implementation static safety review: PASS
- 사용자는 protocol-deviating diagnostic baseline-only model run만 명시적으로 승인했으며, 이는 frozen protocol의 official Calibration authorization을 대체하지 않는다.

AI-only amendment와 사용자 실행 승인은 실제 baseline-only model run의 범위를 통제했지만, frozen v2.1.1 설계가 요구한 독립 human audit를 충족하지는 못했다. 따라서 이 실행은 human audit PASS를 주장하지 않는 **AI-only-audited protocol-deviating diagnostic baseline run**이다.

## 2. 실행 범위

- model: local Meta-Llama-3-8B
- physical GPU: `cuda:1`; `CUDA_VISIBLE_DEVICES=1` 아래 logical device `cuda:0`
- batch size: 8
- dtype: float32
- generation: greedy, `max_new_tokens=1`
- 출력: canonical/alternative/foil teacher-forced log probability와 one-token generation behavior
- LiReF direction loading: 없음
- hidden state / candidate state capture: 없음
- forward hook: 없음
- activation/weight intervention: 없음
- model training: 없음

`calv211_baseline_20260830_01`은 sandbox에서 CUDA runtime을 볼 수 없어 모델 로드 전에 종료됐다. 해당 실패는 별도 상태 및 provenance 파일로 보존했으며 model load, forward, GPU 계산, calibration 결과 생성은 없었다. 실제 결과는 새 authorization으로 동결한 `run02`에서만 생성했다.

## 3. 진단적 threshold 판정

사전 동결된 기준을 진단적으로 모두 적용했을 때 통과한 lexical family는 없다. 이는 세 family의 공식 Calibration 판정이 아니라, 동일 기준을 사용해 실패 양상을 기록한 결과다.

| Family | Forced-choice R / I | Generation R / I | count gap FC / Gen | mean `D_k` | `d_z` | same sign | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| decrease | 52/64 / 64/64 | 47/64 / 45/64 | 12 / 2 | -1.2344 | -1.0074 | 7/8 | FAIL |
| increase | 51/64 / 64/64 | 42/64 / 57/64 | 13 / 15 | -1.3346 | -1.0508 | 6/8 | FAIL |
| temperature | 60/64 / 64/64 | 59/64 / 32/64 | 4 / 27 | -0.8947 | -0.6055 | 6/8 | FAIL |

동결 기준은 다음과 같다.

- forced-choice condition별 39–60/64
- one-token generation condition별 16–60/64
- forced-choice와 generation의 R/I correct-count gap 각각 6 이하
- `|mean(D_k)| <= 0.50 nat`
- `|d_z| <= 0.35`
- 같은 부호의 `D_k` 최대 5/8
- 모든 기준 동시 충족

### 3.1 decrease

- Relevant forced-choice는 52/64로 범위 안이지만 Irrelevant는 64/64로 ceiling을 초과했다.
- forced-choice count gap은 12로 허용치 6을 초과했다.
- `mean(D_k)=-1.2344`, `d_z=-1.0074`, 같은 부호 7/8로 세 cluster 기준도 모두 실패했다.
- generation은 47/64 대 45/64로 두 조건의 절대 범위와 gap 기준을 통과했다.

### 3.2 increase

- Relevant forced-choice는 51/64이나 Irrelevant는 64/64 ceiling이다.
- forced-choice gap 13과 generation gap 15가 모두 허용치 6을 초과했다.
- `mean(D_k)=-1.3346`, `d_z=-1.0508`, 같은 부호 6/8로 cluster 기준도 모두 실패했다.

### 3.3 temperature

- forced-choice gap은 4로 통과하지만 Irrelevant가 다시 64/64 ceiling이다.
- generation은 Relevant 59/64, Irrelevant 32/64로 각각의 절대 범위는 통과했으나 gap 27로 크게 실패했다.
- `mean(D_k)=-0.8947`, `d_z=-0.6055`, 같은 부호 6/8로 cluster 기준도 모두 실패했다.

## 4. 해석 범위

세 family 모두에서 Irrelevant paired forced-choice가 64/64에 도달했다. 음의 `D_k`는 arithmetic-transformation Relevant prompt의 correct-vs-alternative margin이 matched keyed-retrieval Irrelevant prompt보다 평균적으로 작았음을 뜻한다. 즉 v2.1.1의 keyed retrieval은 v2의 단순 direct fact보다 구조적으로 복잡해졌지만, frozen 기준이 요구한 baseline 난이도 균형까지 확보하지는 못했다.

v2에서 문제가 되었던 generation 출력 형식은 해결됐다. 여섯 condition-family cell 모두 64/64가 유효한 one-token numeral 형식이었으므로, 이번 FAIL을 정답+단위 출력이나 normalization 실패로 설명할 수 없다.

이 결과는 R/M separation, LiReF projection 또는 `L31N13336`을 포함한 candidate component 효과에 대한 판정이 아니다. 이번 진단 실행은 baseline behavior만 측정했다. 앞선 exploratory 결과는 별도 가설 생성 기록으로만 남으며 이번 threshold 계산이나 family 판정에 사용하지 않았다.

## 5. 사후 검증 및 provenance

- pair result count: 192, unique pair ID: 192
- prompt result count: 384
- family별 pair count: 64 / 64 / 64
- template-family count: 24
- NaN / infinity: 0
- valid one-token numeral generation: 384/384
- 결과 의존 item/template 제외: 없음
- threshold 변경: 없음
- forbidden tensor 또는 activation artifact: 없음

| Artifact | SHA-256 |
|---|---|
| execution authorization | `7e8441c71a3c609c2b30b846da217368068086f9e25bc68475ffad635c6a260d` |
| implementation | `fbe0e731e1fe0a8ee72db9d1359b813666354e7911339e75191bf6f18978894c` |
| static safety review | `f9ddc7769597395eaa1c7e9a5eeb5359acfa536584851135962d7d7b7d411666` |
| dataset | `322d80f3c2c0723f6a4e8b0a968b30baac23e28cf68b6c60c5a7a95a5bca7420` |
| AI-only audit CSV | `5fdf08675169aec87b42fca33b3b42e564b2e4df43cb68f6c4662563a5f62244` |
| pair results | `70b09d60a9d682cd236897e180020b0c110b88120acfb623d2e30b4ef770616d` |
| calibration summary | `ab5a58203e9547da80de266cd303a121560a39283ea2ab8b0a75766fe3453e08` |
| environment and safety | `986f5a755448cfc0ea561dbe8f3e82eaa2ca7097f0db40f6c5281b9c93a6867f` |
| run manifest | `be79a2f8348253c565b8652abe010343853465b8c1c12830180bf7db249b567e` |
| status | `c0cff4d06a08ae1481e561abf313bbb1223fd8bbcd4494901afd0eeb4685f0e0` |

## 6. 다음 gate

진단 기준상 PASS family가 0개이며, official Calibration은 수행되지 않았다. 따라서 이 dataset이나 그 family를 근거로 공식 Stage E Pilot, family 사후 선택, patching 또는 suppression을 진행하지 않는다. 현재 authoritative status는 다음과 같다.

> **Protocol-deviating diagnostic baseline run: all three families failed; official Calibration not performed.**

후속 작업은 v2.1.1을 수정하는 것이 아니라 새 design version부터 시작한다. 새 dataset의 automatic/AI/human audit와 official Baseline Calibration을 거쳐 PASS family가 생긴 경우에만, 그 생성 규칙으로 별도의 Pilot dataset을 만든다. 기존 item, threshold 또는 결과 파일을 덮어쓰거나 결과를 본 뒤 일부 template만 선택하지 않는다.
