# Stage E 선행논문 방법 비교

상태: **DRAFT — 사람 검토 전, 동결되지 않음**  
목적: 선행논문의 결론을 LiReF의 R/M에 그대로 대입하지 않고, Stage E의 `입력 Feature → 내부 pathway → LiReF representation → 행동` 검증에 필요한 방법만 선별한다.

## 1. 비교 원칙

- 선행논문의 `reasoning`, `knowledge`, `memorization/generalization` 개념은 LiReF의 R/M label과 동일하지 않다.
- 후보 선정, activation 개입, weight 개입, 평가 지표를 분리해서 본다.
- 상관관계, causal contribution, necessity, sufficiency를 구분한다.
- ROME 논문의 `Causal Tracing`과 `Rank-One Model Editing`은 서로 다른 방법으로 분리한다.
- Stage E는 새 후보 전체 탐색보다 Stage C를 통과한 기존 4개 후보의 내부 경로를 우선 분석한다.

## 2. 한눈에 보는 비교

| 방법 | 원 논문의 분석 대상 | 후보 선정 핵심 | 개입 핵심 | 원 논문이 직접 지지하는 수준 | Stage E의 사용 위치 |
|---|---|---|---|---|---|
| Reasoning Neurons | Llama2-7B의 GSM8K few-shot CoT 산술 추론 | 생성 단계·layer별 FFN activation 상위 neuron을 모은 뒤 value vector의 vocabulary projection과 개념 annotation | 선정 neuron의 value vector에 Gaussian noise | 선정 집합이 해당 산술 CoT 정확도에 기여한다는 집합 수준 근거 | FFN `gate/up` activation 후보 해석과 제한적 neuron intervention 설계 참고 |
| Knowledge Neurons | BERT의 PARAREL factual cloze | 정답 확률에 대한 FFN intermediate neuron의 integrated gradients와 paraphrase 간 교집합 | activation 0/2배, second FFN linear-layer value slot 수정 | 특정 cloze factual expression에 대한 neuron 기여 및 제한적 편집 근거 | factual feature 가설, paraphrase 안정성, FFN `down` writer 후보 개입 참고 |
| Mem/Gen Neurons | GPT-2 및 LoRA LLaMA-3.2-3B의 synthetic memorization/generalization 조건 | paired activation의 NMD와 label correlation | 상위 neuron hidden activation에 signed NMD shift | 해당 synthetic task에서 행동 steering이 가능하다는 근거 | paired contrast, activation steering, random-control 설계 참고 |
| ROME — Causal Tracing | GPT 계열의 factual association recall | clean/corrupted/restored run의 token×layer average indirect effect | subject embedding corruption 후 특정 hidden/MLP/attention state 복원 | 특정 factual prediction을 매개하는 activation 위치에 대한 causal contribution 근거 | source span×layer 경로 추적 및 activation rescue 참고 |
| ROME — Weight Editing | GPT 계열의 단일 factual association | causal trace로 정한 mid-layer MLP와 subject key | MLP projection에 rank-one update | 특정 fact를 일반화·특이성을 유지하며 바꿀 수 있다는 weight-level 편집 근거 | factual writer가 먼저 검증된 경우의 제한적·일시적 FFN writer edit 참고 |

## 3. 방법별 상세 비교

### 3.1 Reasoning Neurons

| 항목 | 내용 |
|---|---|
| 논문명 | Daking Rai, Ziyu Yao. *An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs*. ACL 2024. |
| 분석 대상 | Llama2-7B, GSM8K, 8-shot CoT, 산술 추론 생성 과정의 FFN sub-update |
| 후보 선정 방법 | 각 예시·decoding step·layer에서 activation coefficient가 큰 상위 10개 neuron을 수집한다. 각 neuron의 value vector를 vocabulary space로 projection하고, seed-token filter 뒤 GPT-4로 사전 정의된 7개 산술 개념을 annotation한다. |
| 개입 위치 | 발견된 FFN neuron의 value vector. 논문 실험은 생성 전 과정의 선정 neuron 집합을 대상으로 한다. |
| 개입 방식 | 발견 neuron의 value vector에 Gaussian noise를 더하고, 같은 수의 random neuron corruption과 비교한다. |
| 주요 평가 지표 | GSM8K few-shot CoT exact-answer accuracy, neuron activation 빈도, projected vocabulary token의 개념 일치 |
| 성공 기준 | 논문은 preregistered threshold보다 targeted corruption과 random corruption의 정확도 감소 비교를 제시한다. 보고값은 no corruption 16.83%, reasoning-neuron corruption 4.54%, random-neuron corruption 11.37%이다. 이 수치를 Stage E 성공 기준으로 재사용하지 않는다. |
| causal claim 수준 | 선정한 **neuron 집합의 corruption이 해당 산술 CoT 행동에 causal contribution을 보인다**. 단일 neuron의 필요성·충분성이나 일반 reasoning mechanism을 입증하지 않는다. |
| 주요 한계 | 산술 CoT에 국한되고 baseline 정확도가 낮다. 후보 의미가 vocabulary projection, seed lexicon, GPT-4 annotation에 의존한다. value noise는 activation detector와 output writer를 분리하지 않는다. random ablation 자체도 성능을 낮춘다. |
| 그대로 적용 가능한 부분 | input-dependent activation과 input-independent output direction을 구분하는 관점, targeted 대 random corruption, 생성 token별 activation 기록 |
| 수정이 필요한 부분 | Llama-3 SwiGLU에서는 `gate_proj`와 `up_proj`의 곱을 neuron activation으로 정의해야 한다. GPT annotation을 확인 증거로 사용하지 않고 controlled feature intervention을 primary evidence로 삼는다. 기존 4개 후보를 바꾸는 discovery 절차로 사용하지 않는다. |
| Stage E에서 사용할 위치 | `L31N13336`의 `gate/up` 반응을 feature detector **후보**로 분석하고 `down[:,j]` writer 후보와 분리하는 데 사용 |

### 3.2 Knowledge Neurons

| 항목 | 내용 |
|---|---|
| 논문명 | Damai Dai et al. *Knowledge Neurons in Pretrained Transformers*. ACL 2022. |
| 분석 대상 | BERT의 PARAREL fill-in-the-blank factual cloze; masked answer token 위치의 FFN intermediate neuron |
| 후보 선정 방법 | neuron 값을 0에서 실제 activation까지 변화시키며 correct-answer probability gradient를 적분한다(20-step Riemann approximation). 한 사실의 여러 paraphrase에서 threshold를 넘고 일정 비율 이상 공유되는 neuron만 남긴다. |
| 개입 위치 | answer를 예측하는 masked token의 FFN intermediate activation 및 해당 neuron에 대응하는 second FFN linear-layer value slot |
| 개입 방식 | activation을 0으로 suppress하거나 2배 amplify한다. 제한적 fact update에서는 second FFN layer의 value slot을 old/new answer embedding 방향으로 수정한다. |
| 주요 평가 지표 | correct-answer probability change, paraphrase 간 neuron overlap, fact-update change/success rate, intra/inter-relation perplexity 변화 |
| 성공 기준 | 논문은 knowledge-attribution neuron과 activation baseline/random neuron의 상대 효과를 보고한다. 평균 suppression 시 정답 확률 29.03% 감소, amplification 시 31.17% 증가를 보였지만, 이는 Stage E의 사전 성공 기준이 아니다. |
| causal claim 수준 | 특정 BERT cloze 사실 표현에 neuron activation이 기여하며 제한적 edit가 가능하다는 proof-of-concept. 일반적인 memorization 기능의 필요성·충분성은 아니다. |
| 주요 한계 | encoder-only BERT, GELU FFN, single-token cloze 중심이다. attribution threshold와 paraphrase 교집합에 민감하다. editing은 소수 사례의 preliminary study이며 Llama SwiGLU와 구조가 다르다. |
| 그대로 적용 가능한 부분 | correct-answer probability 기반 attribution, 동일 사실의 paraphrase 안정성, activation suppression/amplification, specificity 평가 |
| 수정이 필요한 부분 | autoregressive multi-token answer에는 sequence log probability를 사용한다. `gate/up/down`을 분리하고, factual recall과 LiReF M label을 동일시하지 않는다. weight edit는 writer가 먼저 인과적으로 확인된 경우에만 사용한다. |
| Stage E에서 사용할 위치 | factual/parametric-memory feature 가설의 보조 방법, `L31N13336`의 `down[:,j]` writer 후보 intervention, 행동 specificity 진단 |

### 3.3 Mem/Gen Neurons

| 항목 | 내용 |
|---|---|
| 논문명 | Ko-Wei Huang et al. *Neuron-Level Differentiation of Memorization and Generalization in Large Language Models*. EMNLP 2025. |
| 분석 대상 | from-scratch GPT-2와 LoRA fine-tuned LLaMA-3.2-3B의 synthetic in-context inference, arithmetic addition, ciphertext 조건 |
| 후보 선정 방법 | paired memorization/generalization sample activation의 Neuron-wise Mean Difference(NMD)와 behavior label에 대한 Pearson correlation 절댓값으로 neuron을 ranking한다. |
| 개입 위치 | GPT-2는 post-FFN LayerNorm 뒤, LLaMA는 FFN·LoRA·residual normalization 뒤 hidden neuron; 주로 상위 5–10% neuron |
| 개입 방식 | 선택 neuron에 `h_i ← h_i + α·sign(ρ_i)·|NMD_i|` 형태의 inference-time shift를 적용하고 반대 행동으로의 steering을 측정한다. |
| 주요 평가 지표 | NMD 분포, correlation ranking, 목표 memorization/generalization output으로의 행동 전환율, random intervention, seed/task 간 안정성 |
| 성공 기준 | targeted intervention이 원하는 행동 전환을 만들고 random shift보다 일관되게 커지는지 본다. 논문 hyperparameter를 Stage E의 threshold로 그대로 사용하지 않는다. |
| causal claim 수준 | 선택 hidden-neuron state에 대한 intervention이 해당 synthetic task의 행동을 steer할 수 있다는 causal-control 근거. 해당 neuron이 자연 R/M의 유일한 저장 위치라는 근거는 아니다. |
| 주요 한계 | 원 논문의 memorization/generalization 정의가 LiReF R/M과 다르다. fine-tuning/LoRA와 synthetic task 효과가 섞일 수 있다. NMD·correlation은 후보 선정 단계에서 상관적이며, broad hidden-state shift는 detector와 writer를 구분하지 않는다. |
| 그대로 적용 가능한 부분 | paired condition difference, signed bidirectional steering, random control, seed/task transfer 평가 |
| 수정이 필요한 부분 | 새 neuron ranking에 쓰지 않고 기존 4개 후보 state의 조건 차이를 확인하는 데만 쓴다. 자연 R/M과 controlled feature effect를 분리하고, pathway-level intervention과 rescue를 추가한다. |
| Stage E에서 사용할 위치 | feature-conditioned candidate-state contrast, dose-response, matched/random steering control 설계 |

### 3.4 ROME — Causal Tracing

| 항목 | 내용 |
|---|---|
| 논문명 | Kevin Meng et al. *Locating and Editing Factual Associations in GPT*. NeurIPS 2022 — activation localization 부분 |
| 분석 대상 | GPT factual prompt `(subject, relation) → object`의 token×layer hidden state, MLP, attention 경로 |
| 후보 선정 방법 | clean run, subject embedding을 Gaussian noise로 손상한 corrupted run, 손상 상태에서 특정 token×layer의 clean activation만 복원한 run을 비교해 indirect effect를 계산한다. |
| 개입 위치 | subject span embedding corruption; 각 token×layer hidden state 또는 MLP/attention activation restoration. 원 논문에서는 중간 layer의 마지막 subject token MLP와 후반 layer 마지막 token이 두드러졌다. |
| 개입 방식 | corruption 후 clean state restoration. `IE = P_restore(correct object) − P_corrupt(correct object)`를 token×layer별로 계산한다. |
| 주요 평가 지표 | total effect, average indirect effect, correct-object probability recovery, token×layer heatmap과 95% CI |
| 성공 기준 | corrupted prediction이 충분히 손상되고 특정 restoration site가 correct-object probability를 재현성 있게 회복하는가. 원 논문 효과값을 Stage E threshold로 이식하지 않는다. |
| causal claim 수준 | 특정 activation state가 factual prediction의 손실을 부분적으로 매개한다는 causal contribution 근거. 해당 위치의 weight가 사실의 유일한 저장소라는 증명은 아니다. |
| 주요 한계 | corruption 분포와 patch site 정의에 의존한다. clean activation은 복수 정보가 섞인 high-dimensional state다. localization 결과가 최적 edit layer를 보장하지 않는다는 후속 비판도 있다. |
| 그대로 적용 가능한 부분 | clean/corrupt/restore 삼중 실행, source-span×layer 추적, indirect effect와 rescue 논리 |
| 수정이 필요한 부분 | Stage E에서는 임의 Gaussian noise만 쓰지 않고 feature-matched counterfactual corruption을 우선한다. last prompt token과 source span을 분리한다. component 및 GQA 공유 경로 control을 둔다. LiReF score와 행동을 별도 endpoint로 보고한다. |
| Stage E에서 사용할 위치 | feature가 candidate로 들어오는 source span, layer, attention/FFN 경로를 추적하는 activation patching 및 rescue |

### 3.5 ROME — Rank-One Model Editing

| 항목 | 내용 |
|---|---|
| 논문명 | Kevin Meng et al. *Locating and Editing Factual Associations in GPT*. NeurIPS 2022 — weight editing 부분 |
| 분석 대상 | factual association을 쓰는 것으로 가정한 mid-layer MLP projection matrix |
| 후보 선정 방법 | causal tracing에서 factual recall site를 정하고 subject representation에서 key를 만들며, desired object를 생성하도록 target value를 최적화한다. |
| 개입 위치 | 선택한 한 mid-layer MLP의 output projection. 원 논문은 MLP를 linear associative memory로 보고 rank-one update를 적용한다. |
| 개입 방식 | 새 key–value association을 만족하면서 기존 mapping 간섭을 줄이도록 covariance를 이용한 rank-one weight update를 계산한다. |
| 주요 평가 지표 | edit efficacy, paraphrase generalization, neighborhood specificity; CounterFact에서는 efficacy/generalization/specificity score와 magnitude |
| 성공 기준 | 원하는 새 object가 원 object보다 선호되고, paraphrase에서도 일반화되며, 관련 없는 neighbor fact 손상이 제한적인가. 이 기준은 factual rewriting 기준이지 R/M pathway 기준이 아니다. |
| causal claim 수준 | 선택 MLP weight에 제한된 update로 특정 factual association을 바꿀 수 있다는 weight-level intervention 근거. R/M feature 경로의 necessity·sufficiency를 자동으로 뜻하지 않는다. |
| 주요 한계 | single-fact counterfactual editing이 중심이다. edit 가능 위치와 activation localization 위치가 반드시 일치하지 않을 수 있다. 영구 parameter edit는 collateral damage와 해석 혼선을 만들 수 있다. Attention Q/K/V/O 분석법이 아니다. |
| 그대로 적용 가능한 부분 | key–value writer 관점, efficacy/generalization/specificity의 동시 평가, covariance-regularized local edit |
| 수정이 필요한 부분 | Stage E에서는 factual writer가 먼저 확인된 경우에만 사용한다. edit는 메모리에서 일시 적용하고 종료 후 checksum으로 완전 원복한다. R/M separation, behavior selectivity, KL/top-1/general-task 손상을 함께 측정한다. |
| Stage E에서 사용할 위치 | `L31N13336`의 `down` writer 후보가 factual feature를 선택적으로 매개한다는 선행 증거가 있을 때만 제한적으로 적용. Attention 후보에는 ROME를 직접 적용하지 않는다. |

## 4. Stage E 적용 매핑

| Stage E 질문 | 주 방법 | 보조 방법 | 사용하지 않는 해석 |
|---|---|---|---|
| 어떤 controlled feature가 기존 후보 state를 바꾸는가? | paired NMD/activation contrast | Knowledge Neurons식 answer attribution | activation 차이만으로 `R-specific/M-specific` 선언 |
| feature 정보가 어느 token×layer에서 후보로 들어오는가? | Causal Tracing식 matched corruption/restoration | source-span ablation | 높은 attention weight만으로 정보 전달 확정 |
| FFN neuron의 input sensing과 output writing을 나눌 수 있는가? | SwiGLU `gate/up` 대 `down` 분리 개입 | Reasoning/Knowledge Neurons 관점 | `gate/up=detector`, `down=writer`를 검증 전 사실로 표현 |
| Attention head의 routing과 writing을 나눌 수 있는가? | Q/K 대 V/O pathway intervention | donor patch와 GQA group control | shared K/V 효과를 단일 query head 효과로 표현 |
| pathway가 LiReF와 행동에 기여하는가? | attenuation + dose-response + controls | bidirectional transfer | transfer 하나로 sufficiency 확정 |
| 손상된 효과가 동일 pathway state로 회복되는가? | activation rescue | 제한적 weight edit 후 reinjection | 부분 회복을 완전 necessity로 표현 |

## 5. 문헌에서 직접 가져오지 않을 것

- 각 논문의 Top-K, noise scale, 상위 5–10%, 0/2배 activation, edit layer를 Stage E 기본값으로 복사하지 않는다.
- `reasoning neuron`, `knowledge neuron`, `memorization neuron`이라는 명칭을 기존 4개 후보에 미리 붙이지 않는다.
- vocabulary projection이나 attention weight를 primary causal evidence로 사용하지 않는다.
- Causal Tracing에서 위치가 높게 나왔다는 이유만으로 그 위치의 weight가 최적 edit target이라고 가정하지 않는다.
- LiReF representation 변화와 정답 행동 변화를 같은 endpoint로 합치지 않는다.

## 6. 원문 및 버전 고정 상태

| 자료 | 기준 원문 | 로컬 파일 | SHA-256 / 상태 |
|---|---|---|---|
| Reasoning Neurons | [ACL Anthology 2024.acl-long.387](https://aclanthology.org/2024.acl-long.387/) | `pdf/1(Reasoning)FFN_neuron_2024.pdf` | `9d919805e194d8669c5fd4b510a22f84842bdc008bb76343dbaa3a221be6f265` |
| Knowledge Neurons | [ACL Anthology 2022.acl-long.581](https://aclanthology.org/2022.acl-long.581/) | `pdf/2(Memory)FFN_neuron_2022.pdf` | `0576f133236aa9fb69a4bd7379f9cc3434091f64a02aba038763e5349ebd51de` |
| Mem/Gen Neurons | [ACL Anthology 2025.emnlp-main.812](https://aclanthology.org/2025.emnlp-main.812/) | `pdf/3(Mem-Gen)FFN_neuron_2025.pdf` | `0d6bc8b7b24b9b8349bdaba4e69003462479331a206773479016591332ff2d9b` |
| ROME / Causal Tracing | [NeurIPS 2022 official paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) | 없음 | **동결 전 공식 PDF를 로컬에 저장하고 SHA-256 기록 필요** |

## 7. 문헌 검토 결론

Stage E에는 한 논문을 통째로 재현하는 방법보다 다음 조합이 적절하다.

1. paired activation 차이로 feature 반응을 측정한다.
2. matched source-feature corruption과 token×layer restoration으로 정보 유입 경로를 찾는다.
3. 기존 causal 후보 안에서 FFN `gate/up/down`, Attention `Q/K/V/O`를 역할 후보별로 분리한다.
4. pathway suppression/patching의 LiReF attenuation과 행동 변화를 control·dose와 함께 검증한다.
5. activation rescue로 동일 pathway의 매개 여부를 확인한다.
6. factual writer가 먼저 확인된 FFN에 한해 제한적 weight edit를 고려한다.

이 문서는 방법 선택 근거이지 Stage E 성공 판정 명세가 아니다. 성공 판정과 실행 gate는 `STAGE_E_PROTOCOL_DRAFT_KO.md`에서 정의한다.
