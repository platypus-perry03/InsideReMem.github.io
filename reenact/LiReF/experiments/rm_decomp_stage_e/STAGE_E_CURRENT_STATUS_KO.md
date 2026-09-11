# Stage E 현재 authoritative status

기준일: `2026-08-31`

> **Behavioral Control v5는 absolute solvability에 실패했다. 이후 기존 MMLU-Pro R/M 문항의 blind natural-feature discovery에서 `transformation_required`가 R label, Layer 31 LiReF 및 기존 4개 component와 연관되고 heldout split에서 재현됐지만, label overlap 때문에 association-only evidence로 제한한다.**

이 문서는 Stage E의 현재 해석 상태를 통일하기 위한 기준 기록이다. 기존 frozen design, dataset, audit, authorization, raw result, hash 및 provenance artifact는 수정하거나 소급 재분류하지 않는다. 과거 문서에 적힌 당시 gate와 실행 이력은 역사적 기록으로 보존하되, 현재 연구적 해석과 발표에서는 이 문서를 우선한다.

- human audit: `not_performed`
- human audit gate: `not_satisfied`
- 당시 사용자 waiver 요청: 실행 경위로만 보존하며 v2.1.1 frozen protocol상 유효한 gate 통과로 인정하지 않음

## 1. 현재 확보한 것

- **v3.1 Baseline 결과:** frozen v3.1 기준으로 object-count, points-balance, temperature 세 family가 모두 FAIL했다. `passed_families=[]`이며 결과 확인 후 threshold나 item/template/family를 변경하지 않았다.
- **진단 결과:** v2.1.1 protocol-deviating baseline run에서 decrease, increase, temperature 세 family가 모두 사전 기준을 충족하지 못했다. 특히 matched keyed-retrieval 조건의 forced-choice ceiling과 조건 간 baseline 난이도 불균형이 남았다.
- **탐색 가설:** 별도의 protocol-deviating exploratory run에서 `L31N13336`의 condition effect가 lexical family에 따라 부호가 반전될 가능성을 관찰했다.

두 결과 모두 진단 또는 가설 생성 수준이다. 공식 Calibration PASS나 Stage E causal evidence로 사용하지 않는다.

## 2. 아직 확보하지 못한 것

- Baseline Calibration PASS family
- independent Stage E Pilot
- component mediation
- patching 또는 suppression 효과
- causal mechanism

따라서 `L31N13336`을 Reasoning neuron, R/M 결정 neuron 또는 R/M pathway mediator로 부르지 않는다. Arithmetic transformation이 R representation을 만든다고도 주장하지 않는다.

## 3. 다음 고정 순서

```text
Natural Feature Discovery 완료
-> transformation/R-M이 완전히 겹치지 않는 새 자연 문항 독립 표본
-> blind annotation 및 behavioral/source matching
-> frozen Layer 31 / L29H30 / L30H6 우선 재현
-> 재현될 경우에만 controlled manipulation 설계
-> 별도 승인 후 component intervention
```

- 기존 v2–v5 dataset, threshold 또는 결과 파일을 수정하지 않는다.
- v5 sealed replication pool은 계속 봉인하며 natural-feature 분석에 재사용하지 않는다.
- natural-feature Discovery/Validation에 사용한 3,000문항은 다음 독립 재현 표본에 재사용하지 않는다.
- `transformation_required` 외 feature를 현재 결과를 보고 추가 선택하지 않는다.
- 독립 재현 전 patching 또는 suppression으로 진행하지 않는다.

## 4. 현재 진행 작업

- frozen design: `CALIBRATION_V3_DESIGN_DRAFT_KO.md`
- machine-readable manifest: `calibration_v3_design_frozen.json`
- frozen manifest SHA-256: `c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc`
- 상태: `DESIGN FROZEN — EXECUTION NOT AUTHORIZED`
- 핵심 변경: selector-guided matched retrieval과 exposure-matched within-condition scoring
- deterministic builder: `build_calibration_v3_dataset.py`
- builder SHA-256: `b3ba73e505a1c081a1180dada8193411b7ca4309d01a6283ee07513a3f125b94`
- builder 검증: 임시 경로에서 2회 생성 결과 byte/hash 동일, automatic audit PASS
- 검증 기록: `calibration_v3_builder_verification.json`
- 공식 v3 dataset: 생성 완료 (`192 pairs / 384 prompts`)
- dataset SHA-256: `d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1`
- automatic audit: `PASS`
- AI audit: `192/192 PASS`
- AI audit 결과: `CALIBRATION_V3_DATASET_AND_AI_AUDIT_RESULT_KO.md`
- additional adversarial AI-only review: `192/192 PASS`
- additional review 기록: `calibration_v3_ai_only_additional_review.json`
- independent human audit: `not_performed`
- 기존 frozen v3 human gate: 충족하지 않음; 기존 v3 artifact는 수정하지 않음
- v3.1 AI-only audit-policy amendment: `FROZEN`
- v3.1 manifest SHA-256: `15ce7892f12e00360b07ce533188249229dc24409119bd96e4c2c25c1bf8f9de`
- human-audited claim: 금지
- Baseline Calibration v3.1 implementation: 완료
- implementation: `run_calibration_v3_1_baseline.py`
- implementation SHA-256: `0a91b55dec112efba829e9e8039ca8e0082c08e73ebbe2d96ebe49a669abd6bb`
- model-free unit tests: `5/5 PASS`
- static safety/schema review: `PASS`
- static review artifact: `calibration_v3_assets/calibration_v3_1_baseline_static_safety_review.json`
- static review SHA-256: `f9fb29033a7a50cbaec396d462d50232eda6df6b3cd4cc78373280cf6f64c3d7`
- implementation record: `calibration_v3_1_baseline_implementation_record.json`
- model/tokenizer loading, forward, GPU 사용: 수행하지 않음
- Baseline Calibration v3.1 execution authorization: 생성·동결 완료
- authorization: `calibration_v3_1_execution_authorization_frozen.json`
- authorization SHA-256: `34032a0ac93fd342da2780a947a889400200418c62370ab2d7ea1abacfb1534f`
- authorized run ID: `calv3_1_baseline_20260830_01`
- authorized device / batch / dtype: `cuda:1` / `8` / `float32`
- Baseline Calibration run01: system Python에 `torch`가 없어 model loading 전에 실패; failure record 보존
- Baseline Calibration 결과 run: `calv3_1_baseline_20260830_02`
- protocol-authorized AI-only-audited Baseline Calibration v3.1: 실행 완료
- 결과: object-count / points-balance / temperature 모두 `FAIL`
- passed families: 없음
- 결과 문서: `CALIBRATION_V3_1_BASELINE_RESULT_KO.md`
- model forward / GPU: Baseline behavior 측정에만 사용
- LiReF, candidate state, hidden state, hook, intervention: 사용하지 않음
- Baseline v3.1 완료 시점의 Stage E Pilot 상태: 미승인·미실행이었음; 이후 v3.2 amendment·authorization을 거쳐 run01 실행
- Pilot continuation amendment: `stage_e_pilot_continuation_amendment_v3_2_frozen.json`
- amendment SHA-256: `3e9d14c676c96ee6d950761d39faa176c0e9267d613856d1bd57ff826d41e5fe`
- continuation 분류: `limited same-sample Pilot`; independent/confirmatory Pilot 아님
- 대상: frozen v3 dataset의 세 family·192 pair 전부; 사후 family/template/item 선택 금지
- Calibration FAIL 판정: 변경하지 않음
- Pilot design/specification, implementation, static review: 허용
- amendment 동결 당시 Pilot model execution과 LiReF/candidate scalar 접근은 미승인이었으며, 이후 별도 v3.2 authorization으로 run01에 한해 승인·실행
- intervention, patching, suppression: 미승인
- limited same-sample Pilot v3.2 detailed design: 동결 완료
- detailed design: `stage_e_limited_pilot_v3_2_design_frozen.json`
- detailed design SHA-256: `b8b74d59975230589adae9ec91eeb2e98cefcb80e818f41f15bad45e3f58e3b3`
- candidate manifest SHA-256: `e9d20904967512fa73581d528eb3026a3142f1a5d2c8b0f878e847dbc7eeb233`
- primary scalar: 후보 output의 frozen same-layer LiReF-direction contribution
- primary population: 전체 192 pair; correct-only filtering 금지
- limited same-sample Pilot v3.2 implementation: 완료
- implementation: `run_stage_e_limited_pilot_v3_2.py`
- implementation SHA-256: `b51e0ed5e5500fffdd98c79057fe1aabbe619fa967e33affdb9a674be5e10455`
- model-free unit tests: `8/8 PASS`
- static safety/schema/read-only-hook review: `PASS`
- static review artifact: `stage_e_limited_pilot_v3_2_static_safety_review.json`
- static review SHA-256: `e62de0a136998c26a6b01ff7079a2e8d74eabe236df24e3e15fc74d24fc9159e`
- implementation record: `stage_e_limited_pilot_v3_2_implementation_record.json`
- implementation record SHA-256: `1ab06a82abb22d248d365767733f71e8ceaabd9e40dafd5af10a988d0e7a813d`
- 구현·검토 중 model/tokenizer/CUDA/LiReF direction 로딩, forward, hook 설치: 수행하지 않음
- limited Pilot execution authorization: 생성·동결 완료
- authorization: `stage_e_limited_pilot_v3_2_execution_authorization_frozen.json`
- authorization SHA-256: `db5645b30d932e8ffd4fe458746befc0815d4aebd92f77d133b1e0390616deb8`
- authorized run ID: `stagee_limited_pilot_v3_2_20260830_01`
- authorized device / batch / dtype: `cuda:1` / `8` / `float32`
- authorization schema validation: `PASS`
- Stage E limited same-sample Pilot run01: 실행 완료
- run ID: `stagee_limited_pilot_v3_2_20260830_01`
- output: `AI/reenact/liref_outputs/rm_decomp/v3_2/stagee_limited_pilot_v3_2_20260830_01`
- run manifest SHA-256: `0d8d7a922461597df5028e6a3da22cf44385aea7f1d6fc4b65a0395612043f0d`
- 결과 무결성: 384 prompt / 192 pair / 24 template cluster 및 모든 output hash `PASS`
- overall primary endpoint: 5개 모두 cluster-bootstrap 95% CI가 0 포함
- family-level Pilot signal: points-balance Layer 31 LiReF `+0.7724`; temperature `L31N13336` `-0.03002`; points-balance `L29H30` `-0.000940`
- family interaction Pilot signal: Layer 31 LiReF points-balance − temperature `+1.2336`
- 기존 `L31N13336` family-specific directional interaction 가설: CI 기준 미확인
- 결과 문서: `STAGE_E_LIMITED_PILOT_V3_2_RESULT_KO.md`
- raw state tensor 저장, 후보 재선정, intervention: 수행하지 않음
- intervention / patching / suppression: 미승인
- 다음 허용 연구 작업: Pilot 결과 및 behavioral limitation 검토; 후속 intervention 또는 independent Pilot은 별도 사전 설계와 실행 승인 필요

## 5. Independent replication v4 최신 상태

- 목적: v3.2에서 관찰한 points-balance Layer 31 상대 이동을 완전히 새로운 문항에서 재현하는지 확인
- frozen design: `stage_e_independent_replication_v4_design_frozen.json`
- design SHA-256: `0382a059f2ac3578446e772939a10dc6911d11b7a90bb4cb0f7bd78ed5ebe106`
- calibration / replication pool: 각각 128 pairs / 256 prompts, item·template·numeric block 완전 비중복
- automatic audit: 두 pool 모두 PASS
- primary AI audit: 두 pool 각각 128/128 PASS
- adversarial AI audit: 두 pool 각각 16/16 templates PASS
- human audit: `not_performed`; human-audited evidence 주장 금지
- behavioral Calibration implementation / 5 model-free tests / static review: PASS
- Calibration run ID: `stagee_v4_calibration_20260830_01`
- points-balance: FC 33/64 vs 37/64, generation 33/64 vs 37/64, mean `D_k=-0.01093`, `d_z=-0.32391`, **FAIL**
- temperature: FC 32/64 vs 37/64, generation 32/64 vs 37/64, mean `D_k=-0.01009`, `d_z=-0.48141`, **FAIL**
- `passed_families=[]`
- 해석: retrieval ceiling은 제거됐지만 다단계 symbolic binding으로 두 조건 모두 chance 근처까지 어려워짐
- independent replication pool: 모델에 사용하지 않았으며 봉인 상태로 보존
- primary Layer 31 replication / interaction replication / layer trajectory / 후보 component 측정: **미실행**
- intervention / patching / suppression: **미승인·미실행**
- 결과 문서: `STAGE_E_REPLICATION_V4_CALIBRATION_RESULT_KO.md`
- 다음 허용 작업: v4 실패 원인을 바탕으로 더 단순하고 풀 수 있는 새 behavioral-control 설계를 사전 명세; 기존 threshold나 item을 사후 변경하지 않음

## 6. Behavioral Control v5 최신 상태

- v5 목표: v4의 case/key/ledger binding을 제거하고 single-step Arithmetic과 explicit-tag Selector만 남김
- design SHA-256: `033969f85f9f982ff686c22f4bdd3977baae1dcc76a4f91b537e758a8fd98982`
- Calibration pool: 64 pairs / 128 prompts
- sealed replication pool: 128 pairs / 256 prompts
- 두 pool deterministic/automatic/non-reuse audit: PASS
- primary AI audit: Calibration 64/64, replication 128/128 PASS
- adversarial AI audit: Calibration 8/8, replication 16/16 templates PASS
- human audit: `not_performed`; human-audited evidence 아님
- Calibration implementation / 5 unit tests / static review: PASS
- run ID: `stagee_v5_calibration_20260830_01`
- points-balance: Arithmetic 17/32, Selector 18/32, mean `D_k=-0.04697`, **FAIL**
- temperature: Arithmetic 17/32, Selector 21/32, mean `D_k=-0.02126`, **FAIL**
- `d_z`: 계산·보고했으나 frozen policy대로 hard gate에 미사용
- `passed_families=[]`; primary/interaction replication gate 모두 닫힘
- v5 replication pool, LiReF, Layer 31, 후보 component, intervention: 미실행
- 진단 수정: multi-hop binding만이 floor의 유일 원인은 아님. Base model의 zero-shot paired-context A/B task 수행과 일부 A-position prior가 함께 문제
- 결과 문서: `STAGE_E_BEHAVIORAL_CONTROL_V5_CALIBRATION_RESULT_KO.md`
- 다음 허용 작업: 새 대규모 dataset 이전에 single-block, A/B mapping, paired-context, prompting 요소를 분리한 소규모 format diagnostic을 사전설계

## 7. Natural R/M Feature Discovery v1 최신 상태

- 목적: synthetic A/B task를 반복하지 않고 기존 MMLU-Pro 3,000문항에서 네 후보와 함께 움직이는 자연 입력 feature 탐색
- split: 기존 Discovery 2,400 / Validation 600 그대로 사용
- annotation input: question/options만 제공; R/M label, 정답, category/source/split 및 내부 결과 차단
- annotation: local AI 2개, AI-only evidence
- v1.0 instrument: `answer_mode=MIX` 3,000/3,000 퇴화로 무효 보존
- v1.1 instrument preflight: 두 annotator 모두 48/48 형식 유효, 분포 gate PASS
- v1.2 consensus: 두 annotator exact agreement만 사용; disagreement는 UNC
- 신뢰도 gate: Cohen's κ≥0.60
- gate 통과 feature: `transformation_required` 하나(agreement 0.878, κ=0.737)
- Discovery: transformation이 R label, Layer 31 및 기존 4개 component 모두와 양의 association; 36-test global BH 기준 통과
- Validation: 사전 고정한 label/Layer31/4-component pair 모두 같은 부호, CI 0 제외, selected-test BH q<0.05
- authoritative 분류: `validation-supported natural association`, causal feature 아님
- 핵심 제한: Validation transformation item의 170/172가 R이고 non-transformation은 45/350만 R이어서 feature와 label이 거의 겹침
- post-hoc label-adjusted Validation에서 유지: Layer 31, `L29H00030`, `L30H00006`
- post-hoc label-adjusted Validation에서 불확실: `L31N13336`, `L29H00031`
- intervention / mediation / Reasoning-neuron 주장: 금지
- 결과 문서: `STAGE_E_NATURAL_FEATURE_DISCOVERY_V1_RESULT_KO.md`
- 다음 허용 작업: transformation과 R/M label이 완전히 겹치지 않는 새로운 자연 문항의 독립 재현 설계; intervention은 재현 전 보류

## 8. Transformation 독립 자연문항 재현 v2 preflight 최신 상태

- 기존 3,000/600문항 및 원문 내부 normalized-text 중복을 제거한 새 MMLU-Pro 후보 풀: 8,656문항
- candidate-pool automatic audit: PASS
- 원 GPT-4o `memory_reason_score`는 새 문항에 공개되지 않아 local reasoning-score proxy를 사전 검증
- v2 attempt01: score output contract 실패; A 0/192, B 126/192 parse, 공동 parse 0
- v2.1: attempt01과 겹치지 않는 160문항 preflight 수행
- v2.1 score format gate는 해결됐으나 reliability gate 실패
  - ensemble vs original score Spearman 0.657 (기준 0.70)
  - annotator 간 score Spearman 0.495 (기준 0.70)
  - annotator 간 binary κ 0.084 (기준 0.60)
- Transformation agreement 0.866, κ 0.706이었으나 B parse와 joint coverage gate 실패
- authoritative status: **LOCAL R/M PROXY RELIABILITY FAIL — INTERNAL REPLICATION NOT RUN**
- full 8,656 annotation / 384-item dataset / study model / LiReF / component 측정: 미실행
- 다음 작업은 새 권한과 설계 선택 필요: GPT-4o score 확보, human panel, 또는 R/M decoupling 주장을 포기한 matched Transformation-only replication
- 결과 문서: `STAGE_E_TRANSFORMATION_REPLICATION_V2_PREFLIGHT_RESULT_KO.md`
