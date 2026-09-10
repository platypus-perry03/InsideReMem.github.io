# Stage E exploratory gate-bypass 관찰 결과

상태: **COMPLETE — PROTOCOL-DEVIATING EXPLORATORY RESULT, 공식 Stage E Pilot 주장 금지**

실행일: 2026-08-30

run ID: `stage_e_exploratory_gate_bypass_20260830_01`

## 1. 실행 성격과 범위

사용자의 명시적 요청에 따라 human audit과 Baseline Calibration gate를 통과하지 않은 상태에서 별도의 exploratory 실행을 수행했다. 기존 v2.1.1 dataset, human-audit CSV와 frozen artifact는 수정하지 않았다.

이번 실행에서 측정한 것은 다음뿐이다.

- v2.1.1의 192 pair / 384 prompt에 대한 baseline scalar behavior
- Layer 31 last-prompt-token의 frozen LiReF R/M direction projection
- 기존 4개 후보의 scalar state와 LiReF-aligned contribution
  - `L31N13336`
  - `L29H00030`
  - `L30H00006`
  - `L29H00031`

수행하지 않은 작업:

- activation patching 또는 suppression
- weight/activation intervention
- candidate reselection
- LiReF direction 재추정
- raw hidden state 또는 candidate-state tensor 저장
- 공식 Pilot PASS/FAIL 또는 confirmatory claim

양의 LiReF 차이는 Relevant 조건이 frozen R 방향으로 이동했음을 뜻한다. 모든 차이는 `Relevant - Irrelevant`이고, CI는 template-family cluster를 재표집한 10,000회 descriptive bootstrap percentile 95% CI다.

## 2. 핵심 결과

전체 24개 template cluster를 합치면 일관된 R 방향 이동은 관찰되지 않았다.

| 지표 | 평균 차이 | 95% CI | `d_z` | 부호 분포 |
|---|---:|---:|---:|---:|
| paired-answer margin | -1.1546 | [-1.6429, -0.6524] | -0.8990 | 음수 19/24 |
| one-token generation accuracy | +0.0729 | [-0.0833, +0.2292] | +0.1829 | 양수 11, 음수 10, 0 3 |
| Layer 31 LiReF projection | -0.3591 | [-2.1976, +1.4362] | -0.0776 | 양수 13, 음수 11 |
| `L31N13336` contribution | +0.1138 | [+0.0179, +0.2101] | +0.4618 | 양수 18/24 |
| `L29H00030` contribution | -0.0032 | [-0.0085, +0.0015] | -0.2544 | 양수 11, 음수 13 |
| `L30H00006` contribution | +0.0035 | [-0.0016, +0.0084] | +0.2751 | 양수 18, 음수 6 |
| `L29H00031` contribution | -0.0008 | [-0.0021, +0.0005] | -0.2394 | 양수 9, 음수 15 |

paired-answer margin이 전반적으로 음수라는 것은 Relevant arithmetic condition에서 correct-vs-alternative margin이 matched keyed-retrieval condition보다 낮았음을 뜻한다. 따라서 두 조건 사이에는 여전히 행동 난이도 차이가 있으며, 이는 LiReF/component 차이의 해석을 제한한다.

## 3. Family별 이질성

| Family | Layer 31 LiReF 차이 | 95% CI | `L31N13336` contribution | 95% CI | 행동상 특징 |
|---|---:|---:|---:|---:|---|
| decrease | +1.1788 | [-0.4176, +2.8484] | +0.2132 | [+0.1226, +0.3072] | margin -1.2344; generation 차이는 불명확 |
| increase | +2.2182 | [-0.4101, +4.6440] | +0.2523 | [+0.1190, +0.4129] | margin -1.3346; generation -0.2344 |
| temperature | -4.4742 | [-7.3654, -1.6774] | -0.1241 | [-0.2376, -0.0208] | margin -0.8947; generation +0.4219 |

- `decrease`와 `increase`는 평균적으로 Relevant가 R 방향으로 이동했지만, 각 family의 bootstrap CI가 0을 포함한다.
- `temperature`는 반대로 Relevant가 M 방향으로 이동했고 CI도 0을 포함하지 않았다.
- `L31N13336`의 LiReF-aligned contribution은 decrease/increase에서 8/8 template 모두 양수였지만 temperature에서는 음수로 반전됐다.
- Attention 후보들은 전반적으로 작고 혼합된 효과였다. 예외적으로 `L29H00030`은 temperature에서 -0.0115, CI [-0.0220, -0.0039], `L29H00031`은 increase에서 -0.0022, CI [-0.0038, -0.0005]였다.

따라서 `arithmetic transformation vs matched keyed retrieval`이 lexical family와 무관하게 하나의 공통 R/M feature로 작동한다는 패턴은 관찰되지 않았다. 현재 가장 뚜렷한 관찰은 `L31N13336`이 decrease/increase와 temperature에서 반대 방향으로 반응한다는 family-specific interaction이다.

## 4. 해석 한계

이 결과는 다음 이유로 공식 Stage E Pilot evidence가 아니다.

1. 독립 human audit이 완료되지 않았다.
2. v2.1.1 Baseline Calibration을 실행하지 않아 난이도 균형을 확인하지 않았다.
3. Calibration용 문항을 exploratory screen에 재사용했다.
4. frozen Stage E Pilot protocol과 독립 Pilot dataset이 없다.
5. 관찰형 scalar readout만 측정했으며 causal intervention을 하지 않았다.

특히 behavioral margin 불균형과 family별 LiReF 방향 반전 때문에, 현재 신호를 곧바로 “feature가 R/M representation을 만든다”거나 “기존 후보가 이를 매개한다”고 해석할 수 없다.

## 5. 결론과 다음 gate

이번 gate-bypassed screen의 결론은 다음과 같다.

> **통제 feature에 대한 내부 반응은 존재하지만, 공통된 R 방향 효과가 아니라 강한 lexical-family 이질성으로 나타났다. `L31N13336`은 가장 반응성이 큰 후보이나 방향이 family에 따라 반전된다.**

따라서 이 결과만으로 patch/suppression을 시작하거나 특정 family를 사후 선택하지 않는다. 다음의 엄밀한 단계는 human audit과 Baseline Calibration을 완료한 뒤, 사전에 동결한 독립 Pilot dataset에서 family interaction을 명시적으로 검증하는 것이다. Exploratory intervention이 필요하다면 이번 결과와 분리된 새 설계·승인 기록이 먼저 필요하다.

## 6. 실행 및 무결성 기록

- device: physical `cuda:1` (`CUDA_VISIBLE_DEVICES=1` 환경에서 visible `cuda:0`)
- dtype: float32
- batch size: 8
- prompt/pair/template rows: 384 / 192 / 24
- 비유한 numeric value: 0
- 누락 또는 불완전 pair: 0
- intervention performed: false
- raw tensor output: 없음
- design SHA-256: `52bafaefd7d2bebec338299821a352b85265f61cd5c803c30ac83d9b30bd86ce`
- implementation SHA-256: `7389e00a4e007edc175b9c7df766a832975717432ad08217c77588cb3738724b`
- static review SHA-256: `237681074f70fd39ac9ed30f74383469c10a098a4554f3c2fbb1c78569494dfa`
- run02 authorization SHA-256: `875793e500fdd71c1f91366677bb77423580e402eb798cffc9ebba60e52976e9`
- run manifest SHA-256: `60209919492712f5fb388b81367ee53acdba1fd7c9896d628c0e9ea1600ffc31`
- prompt metrics SHA-256: `017925305dfeb74de6422dc5824c402f5b6e0c828ef7a223b49168028f8ee9db`
- pair differences SHA-256: `12820ed146c17bf1375009404e45376ccb042fb9a15c763a5333b94b117a38dc`
- cluster effects SHA-256: `b3c3028d37b2ad5bfc12d07a0249631287a6169ae8458e577fe13ff57de787d4`
- exploratory summary SHA-256: `6862ae6faa48c3c18f57ba50c50957bad53578528169d917a6e65a7c95ddad05`
- post-run validation SHA-256: `f6fa524bccbcee88ecaf4b18bbdee628eeee8d003471d4c4c0770f85d75ac884`

첫 승인 파일을 사용한 attempt 1은 authorization schema의 root `batch_size` 누락으로 모델 forward 전에 종료됐다. 실패 기록 SHA-256은 `754094e021cc368fde8d85d6ae7886e840756537fa6f7dbdffbbe414afc425ea`이며, 결과는 생성되지 않았다.
