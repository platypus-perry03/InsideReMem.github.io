## 1. 연구 목표

### 핵심 문제

LLM은 다양한 Reasoning benchmark에서 높은 성능을 보이지만,
학습 문제에서 변형된 문제를 만나면 성능이 떨어지는 경우가 있다.

논문에서는 이러한 현상의 원인 중 하나로,
모델이 문제를 실제로 추론하기보다 학습 데이터에서 본 패턴을 기억하여 재사용하는
"Reasoning Parrots"일 가능성을 제시하며, 모델이 Memorization에 지나치게 의존할 수 있다고 본다.

따라서 이 연구의 핵심 질문은 다음과 같다.

- LLM 내부에서 Reasoning과 Memorization은 서로 다른 activation 특징으로 나타나는가?
- 두 행동 사이의 전환을 설명하는 공통적인 내부 방향이 존재하는가?
- 해당 방향에 직접 개입하면 모델의 Reasoning 또는 Memory 사용을 조절할 수 있는가?
- 이러한 개입이 단순 benchmark 성능 향상이 아니라 실제 reasoning generalization 향상으로 이어지는가?

### LiReF 가설

- LLM의 Reasoning과 Memorization 전환은 Residual Stream 내 하나의 선형 방향으로 나타난다고 가정함
- 논문은 Reasoning과 Memorization 사이의 내부 방향을 Linear Reasoning Feature(LiReF)로 정의
- 이 방향을 조절함으로써 모델의 Reasoning과 Memorization 사이의 행동을 제어할 수 있다고 가정함

### 논문의 핵심

1. Reasoning과 Memory 질문은 Residual Steam Activation에서 구분됨
2. 이 차이는 하나의 선형 방향인 LiReF로 표현할 수 있음
3. LiReF는 특정 데이터셋이나 언어에만 존재하는 특징이 아니라
   여러 task, domain, language에서 공통적으로 나타남
4. inference 과정에서 LiReF를 강화하거나 억제하면
   모델의 Reasoning / Memory 행동을 실제로 변화시킬 수 있음
5. Reasoning 방향을 강화하면 변형된 문제에서도 성능 향상되어
   모델의 Generalizable Reasoning 능력이 강화될 가능성이 있음

## 2. 실험 흐름

LiReF 실험의 전체적인 흐름은 다음과 같음

### Step 1. Reasoning / Memory 데이터 구성

질문들을 크게 두 그룹으로 구분함.

- `D_Reasoning`
- `D_Memory`

MMLU-Pro의 경우 GPT-4o를 judge로 사용하여
각 질문에 0~1 사이의 Reasoning score를 부여한함.

- Score > 0.5 → `MMLU-Pro-R`
- Score ≤ 0.5 → `MMLU-Pro-M`

다른 benchmark는 데이터셋 성격에 따라
Reasoning 또는 Memory 그룹에 배치함.

### Step 2. Residual Stream Activation 추출

각 질문을 모델에 입력하고,
모델이 첫 번째 Answer Token을 생성하기 직전의 상태를 분석

특히 User Input의 마지막 Token `x_T`에 대해
모든 Transformer Layer의 Residual Stream Activation을 추출

각 Layer의 Activation을 다음과 같이 생각할 수 있다.

`h^(l)(x_T)`

하나의 질문에 대해 Layer별 Hidden representation을 얻음

### Step 3. LiReF 계산

각 Layer에서 Reasoning 질문들의 평균 Activation과 Memory 질문들의 평균 Activation을 계산함

그 두 평균의 차이를 해당 Layer의 LiReF로 정의

![](./images/STUDY/1786166500655.png)

따라서 각 Transformer Layer마다 하나의 LiReF 방향이 존재한다.

### Step 4. Representation 분석

추출한 Activation과 LiReF를 이용하여
Reasoning과 Memory가 실제로 구분되는지 분석

논문에서는 주요하게 다음 분석을 수행함

- PCA
- Logistic ReGression boundary
- LiReF와 HIdden State의 Cosine Similarity
- LiReF Projection Value 분석

### Step 5. LiReF Intervention

LiReF가 단순히 REasoning / Memory와 상관관계를 가지는 특징인지,
실제로 모델 행동을 조절하는 인과적 특징인지 확인함

Inference 과정에서 질문의 모든 token에 대해
특정 Layer의 residual steam activation에 LiReF 방향을 더한다.

`h' = h + αr`

- Reasoning task : positiva α
- Memory task : negative α

이후 intervention 전후의 task perfomance를 비교한다.

### Step 6. Generalization 확인

단순히 기존 문제의 성능만 좋아지는 것이 아니라
실제 Reasoning generalization이 좋아지는지도 확인함

이를 위해 GSM8K를 변형하여 만든
`GSM-Symbolic`을 사용함

GSM-Symbolic은 같은 문제 구조를 유지하면서
숫자나 조건 등을 변경한 문제로 단순히 기존 GSM8K 답을 기억하는 전략으로 해결하기 어렵다.

LiReF Intervention 후 GSM-Symbolic 성능까지 향상되는지를 통해
Reasoning generazlization 효과를 분석함

## 3. 데이터셋

논문에서는 Reasoning과 Memory를 비교하기 위해
여러 종류의 QA benchmark를 함께 사용한다.

### 3.1 Reasoning-intensive Datasets

#### MMLU-Pro-R

원본 데이터셋 :

`MMLU-Pro`

MMLU-Pro는 STEM, Humanities, Social Science 등을 포함하는
다양한 분야의 multiple-choice benchmark이다.

논문에서는 GPT-4o를 이용하여
각 MMLU-Pro 질문에 REasoning Score를 부여한다.

Reasoning Score가 0.5보다 큰 질문을 `MMLU-Pro-R` 로 정의하고, `D_Reasoning`에 포함한다.

MMLU-Pro 전체가 Reasoning dataset인 것이 아닌,
논문에서 자체적으로 REasoning / Memory subset으로 분리함

#### GSMBK

초등학교 수준의 수학 work problem 데이터셋이다.

다순 사실 회상보다는 여러 단계의 계산과
문제 해결 과정이 필요하기 때문에 `D_Reasoning`에 포함된다.

논문에서는 LiReF가 수학적 Reasoning에서도
동일하게 나타나는지 확인하기 위해 사용한다.

#### MGSM

GSM8K의 Multilingual Version이다.

GSM8K 문제를 여러 언어로 확장하여
LLM의 Multilingual Mathematical Reasoning 능력을 평가한다.

논문에서는 MGSM을 `D_Reasoning`에 포함한다.

MGSM을 사용하는 중요한 이유는
LiReF가 영어에만 특화된 특징인지,
아니면 언어를 넘어 REasoning 자체와 관련된 특징인지 확인하기 위해서이다.

### 3.2 Memory-intensive Datasets

#### MMLU-Pro-M

MMLU-Pro 질문 중 GPT-4o REasoning Score가 0.5 이하인 질문들이다.
`MMLU-Pro-M`으로 정의하고, `D_Memory`에 포함한다.

이 질문들은 비교적 직접적인 지식 회상이나
기억된 지식을 이용하는 성격이 강하다고 판단된 문제들이다.

#### PopQA

LLM의 factual knowledge를 평가하는
open-domain QA 데이터셋이다.

주로 entity와 관련된 사실적 지식을 묻기 때문에
논문에서는 Memory-intensive dataset으로 사용하며, `D_Memory`에 포함된다.

#### C-Eval-H

C-Eval은 중국어 기반의 종합 평가 benchmark이다.

논문에서는 전체 C-Eval 중
Humanities 영역을 사용하며, 이를 `C-Eval-H`라고 부른다.

Humanities 관련 지식 회상 성격이 강하기 때문에 `D_Memory`에 포함된다.

또한 중국어 dataset을 사용함으로써
LiReF가 영어 문제에만 나타나는 현상이 아닌지도 확인할 수 있다.

### 3.3 추가 분석 데이터셋

#### MBPP

Coding task 데이터셋이다.

논문에서는 coding이 Reasoning만 요구하거나
Memory만 요구하는 작업이라기보다
두 능력을 함께 필요로 하는 task라고 본다.

Figure 5에서는 MBPP activation이
Reasoning과 Memory 두 영역의 경계 부근에 위치하는지 확인한다.

![1786168455464](images/STUDY/1786168455464.png)

#### HumanEval

Code generation benchmark이다.

MBPP와 마찬가지로
REasoning과 Memory가 함께 필요한 task의 예시로 사용된다.

PCA 공간에서 두 극단의 중간 영역에 나타나는지를 분석한다.

#### GSM-Symbolic

GSM8K를 기반으로 만든 reasoning generalization benchmark이다.

100개의 GSM8K 문제 template을 선택하고,
각 template에서 수치나 조건 등을 변경하여
50개의 새로운 instance를 생성한다.

총 5,000개의 문제로 구성된다.

논문에서 LiReF intervention이
단순히 기준 GSM8K 문제에 대한 memorization을 강화하는 것이 아니라
새롭게 변형된 문제에도 적용되는 generalizable reasoning을
향상시키는지 평가하기 위해 사용한다.

### 데이터셋 그룹 요약


| 구분           | 데이터셋     | 주요 목적                                        |
| -------------- | ------------ | ------------------------------------------------ |
| Reasoning      | MMLU-Pro-R   | 다양한 분야의 reasoning                          |
| Reasoning      | GSM8K        | 영어 수학 reasoning                              |
| Reasoning      | MGSM         | 다국어 수학 reasoning                            |
| Memory         | MMLU-Pro-M   | 상대적으로 memory 중심 MMLU-Pro 문제             |
| Memory         | PopQA        | factual knowledge recall                         |
| Memory         | C-Eval-H     | 중국어 humanities knowledge                      |
| Mixed          | MBPP         | Coding에서 Reasoning + Memory 분석               |
| Mixed          | HumanEval    | Coding에서 Reasoning + Memory 분석               |
| Generalization | GSM-Symbolic | 변형된 수학 문제에 대한 reasoning generalization |

---

## 4. Repository 파일 서멸

> 이 부분은 논문 내용뿐 아니라 공식 LiReF repository에서 현재 확인되는 파일 구조를 기준으로 정리함
>
> 세부 Call 및 함수 역할은 실제 코드를 분석하면서 추가

### `README.md`

공식 repository의 실행 방법과
LiReF 재현을 위한 기본적인 안내를 제공한다.

현재 \`REPRODUCTION\_KR.md\`는 이 README의 내용을
한국어로 정리하기 위한 별도의 문서이다.

### `dataset.zip`

논문 재현에 사용되는 데이터 파일들이 포함되어 있다.

압축 해제 후 실제 내부 구조와 각 파일의 field를 확인하여
본 문서의 데이터셋 부분에 추가로 기록한다.

### `reasoning_representation/LiReFs_storing_hs.ipynb`

모델에 Reasoning / Memory 질문을 입력하고
hidden state를 추출하는 과정과 관련된 notebook이다.

LiReF 계산 및 representation 분석에 필요한
Layer별 activation을 준비하는 단게에 해당한다.

세부적으로 다음 항목을 코드 분석을 통해 확인해야 한다.

- 모델 로딩 방식
- dataset 로딩 방식
- prompt 구성
- 마지막 token 선택 방법
- hidden state의 shape
- layer indexing
- 결과 저장 형식
- 결과 저장 경로

### `reasoing_representation/Figures_Interp_Reason&Memory.ipynb`

추출된 hidden representation을 이용하여
논문의 representation 분석 Figure를 생성하는 notebook이다.

현재 확인해야 할 주요 내용은 다음과 같다.

- PCA 계산
- Reasoning / Memory 시각화
- Logistic Regression boundary
- LiReF 방향 표시
- Cosine Similarity 분석
- 어떤 Cell이 논문의 어떤 Figure와 연결되는지

### `reasoning_representation/Intervention/features_intervention.py`

LiReF를 이용하여 모델 residual stream에
inference-time intervention을 수행하는 코드이다.

논문의 Section 4와 직접적으로 연결된다.

추후 코드 분석 시 다음을 확인해야 한다.

- LiReF vector loading / calculation
- intervention layer
- activation 수정 방식
- α 적용 방식
- generation 과정
- accuracy 계산 방법

### `reasoning_representation/Intervention/utils.py`

Intervention 실험에서 공통으로 사용하는
utility 함수가 정의되어 있는 파일이다.

구체적인 함수의 역할은 실제 코드를 분석하면서 추가한다.

---

## 5. 모델

논문에서는 하나의 모델에서만 LiReF를 분석하지 않고
서로 다른 네 개의 LLM family를 사용한다.

각 family에 대해 :

- Base model
- Instruction-tuned model

을 함께 분석한다.

### 5.1 LLaMA3-8B

- LLaMA3-8B-base
- LLaMA3-8B-Instruct

논문의 PCA와 intervention 실험에서는 주로 Base model을 사용하고,
Figure 3에서는 Base와 Instruct 모델의
Layer별 LiReF activation pattern을 비교한다.

### 5.2 Gemma2-9B

- Gemma2-9B-base
- Gemma2-9B-Instruct

LLaMA와 다른 model family에서도
LiReF가 동일하게 나타나는지 확인하기 위해 사용한다.

### 5.3 Mistral-7B-v0.3

- Mistral-7B-v0.3-base
- Mistral-7B-v0.3-=Instruct

다른 모델들과 마찬가지로
PCA, Layer-wise LiReF 분석 및 intervention 실험에 사용된다.

### 5.4 OLMo2-7B

- OLMo2-7B-base
- OLMo2-7B-Instruct

네 번째 독립적인 model family로 사용되어
LiReF 현상이 특정 모델 architecture 또는 training recipe에만
의존하는지를 확인함.

### 모델 사용 목적

네 개의 서로 다른 model family를 사용함으로써
논문은 LiReF가 특정 모델 하나에서만 발견되는 특징이 아니라
여러 LLM에서 공통적으로 존재할 가능성을 검증한다.

Figure 2에서는 네 개 Base model의 hidden state를 PCA로 비교한다.

![1786170697884](images/STUDY/1786170697884.png)

Figure 3에서는 각 model family의 Base / Instruct model을 함께 비교한다.

![1786170675391](images/STUDY/1786170675391.png)

특히 LLaMA3-8B, Gemma2-9B, Mistral-7B-v0.3의 경우
Base와 Instruct 모델에서 Layer별 cosine similarity pattern이 상당히 유사하게 나타났다.

논문에서는 이를 통해 LiReF가 instruction tuning 이후 새롭게 만들어졌다기보다,
pre-training 과정에서 이미 형성되었을 가능성을 제시한다.
