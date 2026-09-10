# R/M 내부 분해 및 위치 추적 — Stage A

이 디렉터리는 LiReF 재현 결과에서 관찰된 Reasoning(R) / Memorization(M)
질문의 representation 차이가 Meta-Llama-3-8B Base 내부의 어느 Layer와
component에서 나타나는지 분석하는 **A — Internal Decomposition &
Localization** 구현을 담고 있다.

모델을 학습하거나 weight를 변경하지 않는다. 고정된 pretrained model에서
마지막 prompt token의 raw residual stream을 추출하고, R/M 평균 차이를
Residual Carry, Attention, FFN, Attention Head 및 FFN Neuron 수준으로
분해한다.

## 연구 질문

Stage A가 답하려는 질문은 다음과 같다.

> PCA/LiReF에서 관찰된 R/M representation gap은 Transformer Layer를
> 통과하면서 어떻게 변하며, 어떤 Attention Head와 FFN Neuron의 직접
> 출력 차이가 그 gap과 크게 연결되는가?

Stage A는 내부 위치를 찾는 상관적·구조적 분석이다. 후보를 직접 제거하거나
증폭하는 causal intervention은 수행하지 않는다.

## 데이터와 실험 조건

- 모델: `Meta-Llama-3-8B Base`
- 데이터: MMLU-Pro 3,000 samples
- R label: `memory_reason_score > 0.5`
- M label: `memory_reason_score <= 0.5`
- Discovery: 기존 `train` 2,400개
  - Reasoning 1,103개
  - Memorization 1,297개
- Validation: 기존 `heldout` 600개
  - Reasoning 276개
  - Memorization 324개
- Prompt: `Q: {question}\nA: `
- 분석 위치: 마지막 prompt token
- dtype: FP32
- 모델 상태: `eval`, `inference_mode`, parameter 완전 고정

Discovery/Validation은 모델 학습을 위한 train/test split이 아니다.
Discovery에서는 LiReF 방향과 후보를 정하고, Validation에서는 이를 다시
고르지 않고 그대로 검증한다.

## 분석 방법

각 decoder module에서 다음 값을 직접 hook으로 추출한다.

```text
h_out = h_in + attention_out + mlp_out
```

- `h_in`: 이전 Layer에서 전달된 residual
- `attention_out`: 현재 Attention이 추가한 출력
- `mlp_out`: 현재 FFN이 추가한 출력
- `h_out`: 현재 decoder module의 raw 출력 residual

각 Layer의 Discovery R/M 방향은 다음과 같이 계산한다.

```text
d_layer = mean(Reasoning hidden state) - mean(Memorization hidden state)
r_hat_layer = d_layer / ||d_layer||
```

Discovery에서 계산한 `r_hat_layer`는 Validation에서도 그대로 사용한다.

전체 분석은 다음 순서로 진행된다.

1. Discovery R/M direction 계산 및 고정
2. Discovery Layer/Head/Neuron component 분석
3. Validation 전체 component 분석
4. Discovery 후보 고정
5. Validation 재현성 평가
6. Welch 검정, Cohen's d, BH-FDR 계산
7. Discovery/Validation ranking 안정성 분석
8. 표, 그림, candidate manifest 생성

## 후보의 의미

Head와 Neuron의 기여도는 Discovery LiReF 방향으로 투영한 직접 출력의
R/M 평균 차이로 정의한다.

```text
Delta = mean_R(component contribution) - mean_M(component contribution)
```

- `Delta > 0`: R-M 출력 차이가 +LiReF 방향과 정렬됨
- `Delta < 0`: R-M 출력 차이가 LiReF 방향과 반대이거나 gap을 상쇄함
- `Delta ≈ 0`: 해당 방향에 대한 R/M 출력 차이가 작음

양수 후보를 곧바로 “Reasoning Head/Neuron”, 음수 후보를
“Memorization Head/Neuron”이라고 부르면 안 된다.

후보는 Discovery 결과만으로 선택한다.

- 전체 sign별 Top 20
- 각 Layer sign별 Top 3
- Stage B 상세 분석용 sign별 Top 5

Validation 재현 후보는 다음 두 조건을 모두 만족해야 한다.

1. Discovery와 Validation의 Delta 부호가 동일함
2. Validation BH-FDR `q < 0.05`

## 검증 장치

Full 실행 전에 Reasoning 25개와 Memorization 25개로 sanity 검사를
수행한다. 다음 항목이 모두 통과해야 Full 실행이 허용된다.

- 기존 LiReF hidden-state cache와 일치
- 32개 Layer direction 유효성
- Layer vector/scalar 재구성
- Attention Head 합산 재구성
- FFN Neuron 합산 재구성
- SwiGLU gated activation 재계산
- FFN vector 재구성
- FP32 및 마지막 token 정렬
- Head/Neuron Welch p-value 유효 비율 99% 이상

또한 config, 코드, model config, dataset, split, prompt, 실행 환경 및
direction artifact의 hash를 저장한다. 현재 실행과 hash가 다른 checkpoint는
재사용하지 않는다.

## 실행 방법

반드시 sanity를 먼저 실행한다.

```bash
cd /home/jinhyun/prj_ws/jiho/AI/reenact
bash experiments/rm_decomp/run.sh sanity
bash experiments/rm_decomp/run.sh full
```

SSH 또는 VS Code 연결이 끊겨도 실행을 유지하려면 다음처럼 tmux를 사용한다.

```bash
tmux new -s rm_decomp_a
cd /home/jinhyun/prj_ws/jiho/AI/reenact
bash experiments/rm_decomp/run.sh full 2>&1 | tee liref_outputs/rm_decomp/v2/a_core/full_run.log
```

- tmux 분리: `Ctrl-b`를 누른 뒤 `d`
- tmux 재접속: `tmux attach -t rm_decomp_a`
- 실행 명령이 끝나면 tmux 세션도 종료될 수 있다.

## 최종 결과

최종적으로 사용할 결과는 **v2**이다.

```text
liref_outputs/rm_decomp/v2/a_core/
├── sanity/                              # sanity 결과 및 PASS gate
├── checkpoints/                         # 방향 및 component 분석 checkpoint
├── tables/                              # Layer/Head/Neuron 통계
├── figures/                             # PNG/PDF 그림
├── manifests/
│   ├── candidate_manifest.json          # Stage B 고정 후보
│   └── model_contract.json
├── layer_mean_difference_vectors.pt     # 실제 Layer별 R-M 벡터
├── summary.json                         # 전체 실행 요약
├── stage_status.json                    # 최종 PASS/ERROR
└── full_run.log
```

주요 표는 다음과 같다.

```text
tables/
├── layer_vector_geometry.csv
├── layer_scalar_decomposition.csv
├── discovery_head_statistics.csv.gz
├── validation_head_statistics.csv.gz
├── discovery_neuron_statistics.csv.gz
├── validation_neuron_statistics.csv.gz
├── head_candidate_validation.csv
├── neuron_candidate_validation.csv
└── ranking_stability.csv
```

## v2 실행 결과 요약

- Stage A 상태: `PASS`
- Discovery/Validation의 32개 Layer direction: 모두 유효
- Attention Head 전체: 1,024개
- Validation Head p-value: 1,024/1,024개 유효
- FFN Neuron 전체: 458,752개
- Validation Neuron p-value: 458,751/458,752개 유효
- Discovery 고정 Head 후보: 195개
- Validation 재현 Head 후보: 193개
- Discovery 고정 Neuron 후보: 225개
- Validation 재현 Neuron 후보: 225개

재현 후보가 많다는 사실은 이 데이터의 R/M label 차이가 component
contribution 수준에서도 안정적으로 반복된다는 의미이다. 이것만으로 실제
Reasoning/Memory 기능이나 causal mechanism이 증명되는 것은 아니다.

## v1과 v2

`v1`은 representation 추출과 재구성에는 문제가 없었지만, Welch 자유도
계산에서 아주 작은 양의 분모를 일반 score epsilon과 비교하여 계산 가능한
p-value 다수가 N/A 처리되는 문제가 있었다.

`v2`에서 해당 조건을 “finite하고 0보다 큰가”로 수정하고, sanity에 통계
유효 비율 검사를 추가했다. 연구 및 Stage B에서는 반드시 `v2` 결과만
사용한다.

이 README는 v2 실행 완료 후 한국어로 갱신되었다. 실행 당시 코드와 입력의
정확한 hash는 `v2/a_core/checkpoints/input_manifest.json`에 보존되어 있다.
README 수정 이후 동일한 v2 경로로 재실행하면 hash gate가 실행을 거부할 수
있으며, 새 분석은 새 run ID와 output 경로를 사용하는 것이 안전하다.

## 결과 해석 범위

Stage A 결과로 말할 수 있는 것:

- R/M residual 평균 차이가 Layer를 따라 어떻게 변하는지
- Residual Carry, Attention, FFN 중 어디에서 차이가 추가되는지
- 어떤 Head와 Neuron에서 큰 LiReF-aligned R/M differential contribution이
  관찰되는지
- Discovery 후보가 Validation에서도 같은 방향과 BH-FDR 기준으로
  재현되는지

Stage A 결과만으로 말할 수 없는 것:

- 특정 Head/Neuron이 R/M separation의 원인이라는 주장
- 특정 Neuron이 실제 Reasoning을 담당한다는 주장
- 특정 Head가 실제 Memorization을 담당한다는 주장
- LiReF가 유일한 R/M mechanism이라는 주장
- R/M representation이 본질적으로 1차원이라는 주장

현재 데이터에서는 수학 문제와 factual question의 분야, 숫자, 어휘,
문장 구조, 길이 및 prompt 형식 차이가 완전히 통제되지 않았다. 따라서
Stage A 후보는 “R/M 기능 후보”가 아니라 우선 “고정된 데이터에서 R/M
representation gap과 연결된 후보”로 해석해야 한다.

다음 단계인 **B — Candidate Characterization**에서는 v2의 frozen
candidate manifest를 그대로 사용해 후보 Head가 어떤 source token 정보와
연결되고, 후보 Neuron이 통제된 입력 변화에 어떻게 반응하는지 분석한다.
