# Inside ReMem: Reasoning × Memorization Representation Decomposition

> **왜 Reasoning 질문과 Memorization 질문은 LLM 내부 표현에서 서로 다른
> 방향으로 갈리는가?**

이 프로젝트는 LiReF에서 관찰된 Reasoning(R) / Memorization(M) 문항의
representation separation을 Transformer 내부 계산으로 분해하는 연구이다.
PCA에서 두 집단이 나뉜다는 현상을 확인하는 데서 끝나지 않고, 어느 block에서
차이가 형성되고 어떤 Attention Head와 FFN Neuron이 그 차이에 기여하는지
추적한 뒤 component suppression으로 검증한다.

## 한눈에 보는 연구

```text
R 질문 / M 질문
        ↓
어느 block부터 표현이 달라지는가?
        ↓
Attention과 FFN 중 무엇이 차이를 추가하는가?
        ↓
어떤 head / neuron이 크게 기여하는가?
        ↓
그 component를 억제하면 R/M separation이 실제로 약해지는가?
        ↓
그 계산과 연결된 parameter 및 모델 행동은 무엇인가?
```

쉽게 말하면, **R/M이 PCA에서 갈리는 현상을 보고 끝내는 것이 아니라 모델
내부의 어떤 부품이 그 차이를 만들어 내는지 찾는 연구**이다.

## 반드시 구분할 개념

이 연구의 R/M label은 모델이 실제로 해당 문항을 “추론해서 풀었다” 또는
“외워서 풀었다”는 행동 판정이 아니다. MMLU-Pro 문항의
`memory_reason_score`를 기준으로 나눈 **문항 요구 특성(task demand) label**이다.

따라서 다음 두 문장은 서로 다르다.

- 이 연구가 직접 검증하는 것: R/M label과 연결된 내부 representation이 어떤
  component의 계산으로 형성되는가?
- 별도 행동 실험이 필요한 것: 해당 component가 실제 reasoning 성공이나
  memorization 행동에 필수적인가?

R/M probe AUROC가 높더라도 곧바로 “reasoning을 수행했다”는 뜻은 아니며,
component를 억제해 R/M decodability가 낮아져도 우선은 **linearly decodable
R/M information이 감소했다**고 제한해서 해석한다.

## 출발 관측

### 실험 조건

- 모델: Meta-Llama-3-8B Base
- 데이터: MMLU-Pro 3,000문항
- R label: `memory_reason_score > 0.5`
- M label: `memory_reason_score <= 0.5`
- Discovery / held-out: 2,400 / 600문항
- Prompt: `Q: {question}\nA: `
- 분석 위치: 마지막 prompt token

### 확인된 layerwise 결과

| 위치/지표 | Held-out 결과 | 의미 |
|---|---:|---|
| Block 1 output AUROC | **0.9292** | 첫 block 출력부터 R/M label이 선형적으로 잘 구분됨 |
| Block 5 output AUROC | **0.9782** | 얕은 층에서 이미 강한 decodability가 형성됨 |
| Block 12 cosine gap | **0.5275** | 측정한 layer별 LiReF cosine gap의 peak-state |

R/M prompt의 마지막 token이 같으므로 Llama의 마지막 위치 token embedding은
같다. 그런데 Block 1 출력부터 held-out AUROC가 0.9292이다. 이 결과는
**문항 전체 문맥의 차이가 첫 Transformer block에서 마지막-token
representation으로 전달된다**는 강한 단서다.

하지만 여기서 바로 “Block 1 Attention이 R/M separation의 원인이다”라고
결론 내리면 안 된다. Self-attention은 구조상 앞선 질문 token을 마지막 위치로
전달할 수 있는 첫 cross-token 경로이고, MLP는 Attention 이후 달라진 값을
증폭·변환할 수 있다. 따라서 현재의 정확한 표현은 다음과 같다.

> **Block 1 Attention이 초기 R/M separation을 형성하고 MLP가 이를
> 변환·증폭하는지 검증한다.**

## 핵심 분석 설계

### 1. Formation — Block 1

첫 block에서는 다음 다섯 값을 마지막 prompt token에서 비교한다.

```text
resid_pre
   ↓ Attention
attn_out → resid_mid
   ↓ MLP
mlp_out  → resid_post
```

Block 1의 `resid_pre`가 R/M에서 같고 `resid_mid`부터 차이가 커지면,
Attention이 초기 contextual separation을 residual stream에 전달했다는
관찰 근거가 된다. 이후 `mlp_out`을 통해 MLP가 그 차이를 얼마나 추가했는지
측정한다.

Attention이 주 기여자로 확인될 경우 query-head별 residual contribution을
분해한다. Llama-3-8B는 GQA를 사용하므로 KV head가 아니라 `o_proj`를 거쳐
residual output을 구성하는 **query-head contribution**을 분석 단위로 삼는다.

### 2. Amplification — `L*`

Block 12의 gap이 가장 크다는 사실과 어느 block이 separation을 가장 많이
추가했다는 주장은 다르다. 증폭 block `L*`는 Discovery 2,400문항 안에서만
cross-fitting으로 선정하고 held-out 600문항은 최종 검증에만 사용한다.

기존 layerwise scan처럼 layer마다 서로 다른 LiReF 방향을 사용한 gap의 단순
차이는 representation 변화와 측정 방향 변화를 함께 포함할 수 있다. 따라서
각 candidate block에서는 Discovery의 `resid_post`로 block-specific readout
direction을 고정하고, 같은 방향에서 다음을 비교한다.

```text
ΔS_l = S(resid_post_l) - S(resid_pre_l)
```

Primary metric은 사전에 고정하며, normalized centroid separation과 held-out
가능한 probe 지표는 보조적으로 보고한다. `L*`가 정해지면 Attention/MLP,
head/neuron 순서로 기여를 좁힌다.

### 3. Peak representation — Block 12

Block 12는 현재 측정에서 cosine gap이 가장 큰 **peak-state block**이다.
이를 자동으로 amplifier라고 부르지 않는다. 이 위치에서는 누적된 강한 R/M
state가 어떤 component contribution으로 구성되는지를 보조적으로 분석한다.

### 4. Causal validation

Discovery에서 고른 top-k head/neuron과 matched/random control에 동일한
suppression을 적용하고 held-out에서 다음을 함께 평가한다.

- 고정 LiReF direction의 projection gap
- normalized centroid separation
- 기존 fixed probe의 AUROC
- ablated Discovery에 새로 학습하고 ablated held-out에 평가한 retrained probe
- 개입 강도에 따른 dose response와 control 대비 효과

| 결과 | 제한된 해석 |
|---|---|
| Fixed ↓ / Retrained 유지 | 기존 geometry가 회전·재배치됐을 가능성 |
| Fixed ↓ / Retrained ↓ | linearly decodable R/M information 감소 가능성 |
| Fixed 유지 / Retrained 유지 | 후보가 핵심 mediator가 아닐 가능성 |
| 행동만 ↓ / representation 유지 | R/M encoding과 downstream use가 분리됐을 가능성 |

## 완료된 component 분석 결과

### Component localization 및 suppression

전체 layer에서 후보를 탐색한 뒤 동일한 held-out, dose, control 및 통계 기준으로
검증한 component 수는 다음과 같다.

| Base 모델 | R/M Gap | R-direction | M-direction |
|---|---:|---:|---:|
| Meta-Llama-3-8B | 9 (Head 4 + Neuron 5) | 10 (Head 5 + Neuron 5) | 7 (Head 4 + Neuron 3) |
| Mistral-7B-v0.3 | 10 (Head 5 + Neuron 5) | 9 (Head 5 + Neuron 4) | 9 (Head 5 + Neuron 4) |
| OLMo-2-1124-7B | 10 (Head 5 + Neuron 5) | 10 (Head 5 + Neuron 5) | 7 (Head 3 + Neuron 4) |
| Gemma-2-9B | 0 | 0 | 2 (Head 2) |

Meta-Llama의 Gap 분석에서는 Stage A 후보 20개를 모두 같은 held-out, dose,
control, 통계 기준으로 검증했고 9개가 strict PASS했다. Cross-model 및
Direction 분석은 각 모델의 전체 layer를 검색하고 유형별 최대 5개를 후보로
지명했다.

위 숫자는 사전 정의된 후보 집합에서 검증된 수이지 모델에 존재하는 전체 R/M
component 수가 아니다. 모델별 동일 번호의 부품을 찾았다는 뜻도 아니며,
선택성·방향·억제 효과가 비슷한 **기능적 후보**를 찾은 결과다.

### Behavioral validation

- Meta-Llama R/M-direction 후보의 중복 제거 후 고유 후보: **13개**
- 외부 객관식 평가에서 안정적인 정답률 감소를 보인 strict signal: **0개**
- 정답 확률만 소폭 변한 Head: `L29H31`, `L30H6`, `L31H3`
- confirmation에서 MGSM baseline gate가 실패해 추가 개입과 나머지 세 모델
  실행은 중단

즉 특정 Head와 Neuron이 R/M 내부 표현 차이에 기여한다는 결과는 확인했지만,
이 부품들이 외부 객관식 정답 선택에 필수적이라는 증거는 확인하지 못했다.

### 탐색적 입력 특징 분석

기존 MMLU-Pro 문항에서 `transformation_required`가 R label과 Layer 31 LiReF
반응에 연관되는 현상을 발견했다. 그러나 Transformation 문항 대부분이 R
문항이어서 두 효과를 분리하기 어렵다. 현재 결과는 인과적 feature가 아니라
**탐색적 association**으로만 해석한다.

독립 재현용 synthetic task v4·v5는 behavioral calibration을 통과하지 못해
Layer 31 독립 재현과 intervention을 실행하지 않았다.

## 현재 결론

> **여러 Base 모델에서 특정 Head와 Neuron이 R/M 표현 차이와 각 방향에
> 기여하는 현상은 확인했지만, 보편적인 단일 R/M mechanism이나 실제 정답
> 선택에 필수적인 component는 확인하지 못했다.**

따라서 현재 핵심 과제는 새로운 neuron 이름을 더 붙이는 것이 아니라,
Block 1의 초기 형성과 train-only로 선정한 `L*`의 증폭을 동일 방향·동일
평가 규칙으로 분해하고 causal ablation으로 확인하는 것이다.

## 연구 진행 순서

```text
1. Block 1 formation
        ↓
2. Cross-fitted amplification block L* 선정
        ↓
3. Attention/MLP → Head/Neuron localization
        ↓
4. Held-out causal validation
        ↓
5. 관련 parameter localization
        ↓
6. Behavioral/generalization extension
```

GSM-Symbolic 등 systematic generalization 평가는 의미 있는 후속 축이지만,
“왜 PCA에서 R/M이 갈리는가?”라는 핵심 질문보다 먼저 neuron/weight 연구의
전제처럼 두지 않는다.

## 실험 디렉터리 안내

| 디렉터리 | 단계와 역할 |
|---|---|
| `experiments/rm_decomp/` | Stage A: layer/module/head/neuron 분해 및 위치 추적 |
| `experiments/rm_decomp_b/` | Stage B: 동결 후보의 입력 특징 민감도 분석 |
| `experiments/rm_decomp_b_extension/` | 관계·task relevance 및 control 확장 |
| `experiments/rm_decomp_pre_stage_c/` | Stage C 이전 진단과 입력 검증 |
| `experiments/rm_decomp_causal/` | Stage C: component suppression |
| `experiments/rm_decomp_cross_model/` | Gap/R/M-direction cross-model 분석 |
| `experiments/rm_decomp_behavioral_validation/` | 외부 객관식 행동 검증 |
| `experiments/rm_decomp_feature_causal/` | 입력 feature 및 state patching 탐색 |
| `experiments/rm_decomp_stage_e/` | calibration과 후속 feature 연구 |

각 단계의 설계, 실행 gate와 해석 제한은 해당 디렉터리의 README 또는
`*_DESIGN_KO.md`, `*_RESULT_KO.md`를 기준으로 한다.

## 파일 구조

```text
reenact/
├── README.md                         # 전체 연구 질문·결과·진행 순서
├── AGENTS.md                         # 재현 프로젝트 공통 작업 원칙
├── liref/                            # LiReF 원 구현 및 재현 문서
│   ├── README.md
│   ├── REPRODUCTION_KR.md
│   ├── STUDY.md
│   └── reasoning_representation/
├── experiments/                     # 단계별 R/M mechanism 실험
│   ├── rm_decomp/
│   ├── rm_decomp_b/
│   ├── rm_decomp_b_extension/
│   ├── rm_decomp_pre_stage_c/
│   ├── rm_decomp_causal/
│   ├── rm_decomp_cross_model/
│   ├── rm_decomp_behavioral_validation/
│   ├── rm_decomp_feature_causal/
│   └── rm_decomp_stage_e/
├── scripts/                          # LiReF·robustness 분석 유틸리티
├── pdf/                              # 발표 자료와 생성 소스
├── run_liref_hidden_states.sh        # hidden-state 추출 진입점
├── run_mgsm_language_robustness.sh   # MGSM 언어 강건성 진입점
├── models/                           # 로컬 모델; Git 제외
├── liref_models/                     # 로컬 모델 링크/복사본; Git 제외
└── liref_outputs/                    # cache·표·그림·log; Git 제외
```

대용량 모델, dataset, hidden-state cache 및 output은 로컬에서만 관리한다.
공개 저장소로 내보낼 때는 blind annotation item, private key와 answer key도
제외해야 한다.

## 실행 예시

Stage A는 반드시 sanity gate를 먼저 실행한다.

```bash
cd /home/jinhyun/prj_ws/jiho/AI/reenact
bash experiments/rm_decomp/run.sh sanity
bash experiments/rm_decomp/run.sh full
```

장시간 실행은 각 디렉터리의 tmux launcher와 frozen config/authorization을
사용한다. 기존 output과 hash가 충돌할 수 있으므로 새 분석은 새 run ID와
output 경로를 사용하는 것이 안전하다.

## 주요 문서

- [Stage A 내부 분해](experiments/rm_decomp/README.md)
- [Stage B 후보 특성 분석](experiments/rm_decomp_b/README.md)
- [Stage B 확장](experiments/rm_decomp_b_extension/README.md)
- [Stage D feature causal 분석](experiments/rm_decomp_feature_causal/README.md)
- [Stage E 현재 상태](experiments/rm_decomp_stage_e/STAGE_E_CURRENT_STATUS_KO.md)
- [LiReF 재현 가이드](liref/REPRODUCTION_KR.md)
- 발표 자료: `pdf/ReMem.pptx`

## Reference

이 연구는 다음 LiReF 논문과 공개 구현을 출발점으로 한다.

- Yihuai Hong et al., *The Reasoning-Memorization Interplay in Language Models
  Is Mediated by a Single Direction*, Findings of ACL 2025,
  [arXiv:2503.23084](https://arxiv.org/abs/2503.23084)
- [Official implementation](https://github.com/yihuaihong/Linear_Reasoning_Memory_Features)
