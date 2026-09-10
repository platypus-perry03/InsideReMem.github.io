# LiReF Stage B — Candidate Characterization

Stage A에서 동결한 Attention Head 10개와 FFN Neuron 10개가 어떤 입력 feature 변화에 선택적으로 민감한지 분석한다. 이 단계의 결과는 feature sensitivity이며, reasoning/memorization의 인과적 메커니즘이라는 주장을 허용하지 않는다.

## 실행 원칙

- Stage A와 동일한 Meta-Llama-3-8B Base, 데이터 split, `Q: {question}\nA: ` prompt, 마지막 prompt token을 사용한다.
- inference는 `config.json`의 `gpu_id` 하나만 사용한다. 기본값은 GPU 1이다.
- 후보는 Stage A `detailed_candidate=true` 20개로 고정한다.
- natural 분석은 가설 생성 전용이다. confirmatory 결과를 보기 전에 사람이 가설과 controlled pair를 승인해야 한다.
- pilot과 confirmatory는 template family 단위로 완전히 분리한다.
- `sanity`는 eager attention source decomposition과 Stage A total-head reconstruction을 검증한다. 실패하면 source-token 해석과 confirmatory attention 분석을 중단한다.

## 순서

```bash
cd /home/jinhyun/prj_ws/jiho/AI/reenact
bash experiments/rm_decomp_b/run.sh prepare
bash experiments/rm_decomp_b/run.sh sanity
bash experiments/rm_decomp_b/run.sh natural
```

`natural`이 끝나면 `manifests/feature_hypothesis_manifest.json` 초안을 사람이 검토한다. 승인 전에는 이후 단계가 실행되지 않는다.

이 초안에는 20개 후보가 모두 남아 있으며 natural association은 가설 생성용 기술통계일 뿐이다. 후보별 confirmatory hypothesis는 최대 3개이다. 선택한 hypothesis에는 명세의 모든 필드를 채우고 각 `approved`, 최상위 `approved`를 `true`로 설정하며 reviewer와 freeze timestamp를 기록한다.

```bash
bash experiments/rm_decomp_b/run.sh freeze_hypotheses
```

그다음 `template_rules.json`에 규칙 기반 pair를 작성한다. 각 rule은 `controlled_pair_manifest.csv`의 전체 schema를 따르며 `split`은 `pilot` 또는 `confirmatory`이다. 숫자 span과 tokenizer length는 생성 시 다시 계산되고, 사용자가 적은 값과 다르면 중단한다.

```bash
bash experiments/rm_decomp_b/run.sh generate_pilot
# 생성된 pilot manifest를 사람이 검수하여 human_validated/approved/reviewer_id를 기록
bash experiments/rm_decomp_b/run.sh approve_pilot
bash experiments/rm_decomp_b/run.sh pilot
```

Pilot은 variance와 runtime만 추정한다. 이후 `confirmatory_design_draft.json`에 pilot 결과를 근거로 표본수를 기록하고 승인한다. 프로그램도 동결된 alpha, power, minimum effect, hypothesis 수, 예상 제외율로 최소 template-family 수를 계산하며, 더 작은 설계는 거부한다. Confirmatory rule은 pilot과 다른 template family여야 하고 각 hypothesis에 동결된 수만큼 존재해야 한다.

```bash
bash experiments/rm_decomp_b/run.sh freeze_confirmatory
bash experiments/rm_decomp_b/run.sh confirmatory
bash experiments/rm_decomp_b/run.sh report
```

## 계산 및 통계

- Neuron의 primary endpoint는 paired `Δz`이다. `Δc = pΔz`는 deterministic secondary interpretation이라 별도 유의성 검정을 하지 않는다.
- Head primary endpoint는 paired total-head contribution 변화이다.
- Head source contribution은 모든 source token을 합쳐 reconstruction gate를 검사한다. 파일 크기를 제한하기 위해 token table에는 절댓값 Top-K와 사전 feature span token을 저장하고, span table에는 전체 token 합을 저장한다.
- 통계 관측 단위는 pair가 아니라 template family이다. percentile bootstrap CI, sign-flip p-value, Cohen's `d_z`, 사전 정의 family별 BH-FDR를 사용한다.
- matched/random control specificity는 `candidate pair effect - control pair effect`로 계산한다.

장시간 실행은 tmux에서 수행한다.

```bash
tmux new -s liref_stage_b
cd /home/jinhyun/prj_ws/jiho/AI/reenact
mkdir -p liref_outputs/rm_decomp/v2/b_characterization_v2_b02/logs
bash experiments/rm_decomp_b/run.sh prepare 2>&1 | tee liref_outputs/rm_decomp/v2/b_characterization_v2_b02/logs/prepare.log
```

분리: `Ctrl-b`, `d`  / 재접속: `tmux attach -t liref_stage_b`

## 승인 파일

- `feature_hypothesis_manifest.json`: 각 confirmatory hypothesis에 `approved=true`, reviewer와 승인 시각이 필요하다.
- `template_rules.json`: 규칙 기반 pair 원본이다. `approved=true`로 바꾸기 전에 문법, 의미, 단일 feature 변경 및 expected answer를 검수한다.
- `controlled_pair_manifest.csv`: confirmatory의 모든 행이 `approved=true`이고 `human_validated=true`여야 한다.

Pilot 결과는 분산 및 실행시간 추정에만 사용하며 최종 CI, p-value 또는 논문 주장에 재사용하지 않는다.

## 현재 해석 한계

완료 후에도 말할 수 있는 것은 “고정된 모델·prompt·마지막 token 조건에서 후보가 특정 controlled feature 변화에 선택적으로 민감했다”까지이다. “reasoning neuron/head”, R/M separation의 원인, 행동 차이의 원인이라는 결론은 별도 component intervention을 거쳐야 한다.
