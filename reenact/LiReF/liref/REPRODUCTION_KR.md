# LiReF 재현 준비 가이드

## 저장소 개요

이 저장소는 **The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction** 논문의 실험에 사용된 데이터와 코드를 제공한다.

- 논문 게재 정보: ACL 2025 Findings
- Arxiv: https://arxiv.org/abs/2503.23084
- 공개 코드: https://github.com/yihuaihong/Linear_Reasoning_Memory_Features

## 연구 개요

대규모 언어 모델(LLM)은 여러 추론 benchmark에서 뛰어난 성능을 보이지만, 암기한 학습 예제에 지나치게 의존하여 보지 못한 질문에 일반화하지 못할 수 있다. 이 연구는 텍스트 생성 중 LLM이 추론과 암기 사이에서 전환하는 동작을 기계론적으로 이해하기 위해, 모델의 residual stream에서 실제 추론과 기억 회상의 균형을 조절하는 선형 feature 집합을 식별한다.

이 feature들은 추론 task와 기억 의존도가 높은 task를 구분하며, 조작을 통해 추론 task의 모델 성능에 인과적으로 영향을 줄 수 있다. 또한 이 추론 feature에 개입하면 모델이 답변을 생성할 때 문제 해결에 가장 관련된 능력을 더 정확하게 활성화하는 데 도움이 됨을 보인다. 이를 통해 LLM의 추론과 기억을 작동시키는 내부 메커니즘을 이해하고, 더 견고하고 해석 가능한 생성형 AI 시스템을 개발하기 위한 단서를 제공한다.

## 재현에 사용되는 항목과 역할


| 항목                                                            | README.md에 명시된 역할                                   |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| `dataset.zip`                                                   | 실험 재현에 사용할 dataset 압축 파일                      |
| `./reasoning_representation/LiReFs_storing_hs.ipynb`            | 특정 task에서 모델의 hidden states를 저장                 |
| `./reasoning_representation/Figures_Interp_Reason&Memory.ipynb` | PCA와 기타 figure를 생성                                  |
| `Intervention`                                                  | intervention experiment를 실행하기 위해 이동하는 디렉터리 |
| `features_intervention.py`                                      | intervention experiment를 실행하는 Python 파일            |

## 실행 순서

아래 순서대로 실행한다.

### 1. Dataset 압축 해제

`dataset.zip`을 압축 해제한다.

```sh
unzip dataset.zip
```

### 2. 특정 task의 모델 hidden states 저장

다음 notebook을 실행한다.

```text
./reasoning_representation/LiReFs_storing_hs.ipynb
```

### 3. PCA 및 기타 figure 생성

다음 notebook을 실행한다.

```text
./reasoning_representation/Figures_Interp_Reason&Memory.ipynb
```

### 4. Intervention experiment 실행

다음 명령어를 실행한다.

```sh
cd Intervention
python features_intervention.py
```

## 인용

```bibtex
@misc{hong2025reasoningmemorizationinterplaylanguagemodels,
      title={The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction}, 
      author={Yihuai Hong and Dian Zhou and Meng Cao and Lei Yu and Zhijing Jin},
      year={2025},
      eprint={2503.23084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.23084}, 
}
```
