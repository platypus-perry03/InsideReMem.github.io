# Stage D — Causal Feature Identification

Stage A–C에서 동결된 네 component 앞단의 계산 feature를 찾기 위한 2×2 실험이다.

- `A`: supplied + direct (country→capital 한 단계 사실 제공)
- `B`: supplied + composition (entity→country→capital 두 단계 사실 제공)
- `C`: parametric + direct (문맥 없이 country→capital 회상)
- `D`: parametric + composition (문맥 없이 entity→country→capital 회상·조합)

각 fact chain은 네 조건 모두에서 같은 정답과 같은 선택지를 사용한다. 현재 허용 범위는
`prepare → calibrate → freeze → pilot → audit-package`이며, 두 명의 blind human audit가
PASS하기 전 confirmatory inference는 실행할 수 없다.

긴 GPU 단계는 `run_tmux.sh`로 실행한다.

## AI-audited exploratory continuation

두 명의 blind human audit 없이 사용자가 명시적으로 계속 진행하도록 선택한 실행은
정식 confirmatory와 분리한다. `run_feature_patching.py`와
`exploratory_config.json`은 `v2_d07_exploratory`에만 결과를 쓰며,
`confirmatory_claim_allowed=false`를 고정한다.

- `prepare`: d06 AI 감사와 frozen component/control 확인
- `baseline`: A/B/C/D의 LiReF score와 component state 저장
- `patch`: 4개 후보, 16개 control, joint-4의 8방향 state patch
- `report`: frozen feature/patch 기준 적용 및 weight-rescue eligibility 판정

GPU 단계는 `run_exploratory_tmux.sh baseline|patch`로 실행한다.
