# Behavioral Validation v2 정적 검토

상태: **PASS — model-free tests와 hash-lock 이후 GPU 실행 허용**

검토 범위:

- 외부 task-family 정의는 LiReF와 동일하게 C-Eval-H=M, GSM8K/GSM-Symbolic/MGSM=R로 고정했다.
- 이 표지는 문항별 `memory_reason_score`가 아니라 dataset family label임을 결과에 명시한다.
- 네 데이터셋에서 각각 100개를 고정하고 primary 70 / confirmation 30으로 결과 전에 분리했다.
- GSM-Symbolic의 100개 원본 GSM8K ID는 GSM8K 표본에서 제외했다.
- 모든 문제는 A–D forced-choice 형식이며 정답 위치는 결정론적으로 counterbalance한다.
- 후보와 matched/random control은 완료된 R-/M-directed strict 결과에서만 가져온다.
- 억제는 기존 component 실험과 동일한 마지막 prompt token 위치, alpha 0.5/1.0을 사용한다.
- baseline gate를 dataset별로 적용하며 하나라도 실패하면 해당 모델 intervention을 중단한다.
- Meta-Llama primary가 strict accuracy+probability 기준을 통과할 때만 다른 세 모델을 실행한다.
- probability-only이면 Meta confirmation을 먼저 실행하며 confirmation strict 신호 없이는 확장하지 않는다.
- 자동 `result.pdf` 수정은 금지한다. 완료 후 결과를 사람이 확인한 뒤 문서에 반영한다.

검토 한계:

- 수학 데이터의 숫자 정답을 네 선택지로 변환했으므로 원래 자유생성 benchmark와 동일한 평가는 아니다.
- task family와 R/M pole이 중첩되므로 결과는 외부 task-family 행동 기여이며 보편적 R/M 행동 메커니즘을 뜻하지 않는다.
