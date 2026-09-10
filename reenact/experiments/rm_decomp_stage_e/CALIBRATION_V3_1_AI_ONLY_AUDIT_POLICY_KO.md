# Stage E Calibration v3.1 AI-only Audit Policy Amendment

상태: **FROZEN — AI-ONLY AUDIT POLICY LOCKED — MODEL EXECUTION NOT AUTHORIZED**  
작성일: 2026-08-30

## 1. Amendment 목적

기존 frozen v3의 dataset 설계, 문항, scoring, threshold 및 counterbalance는 수정하지 않는다. 사용자가 독립 human audit 대신 비인간 AI 전수검수로 진행하기로 결정했으므로, audit gate만 별도 version `v3.1`로 사전 명세한다.

기존 v3 artifact는 덮어쓰거나 소급 재분류하지 않는다. 기존 v3에서 `human_audit_waiver_allowed=false`였다는 사실과 human audit이 수행되지 않았다는 기록도 그대로 보존한다.

## 2. 변경되지 않는 항목

- operationalized feature
- 공식 v3 dataset 192 pairs / 384 prompts
- 24 template families와 8-frame `OA(8,5,2,2)`
- numeric design과 wrong-operation arithmetic foil
- within-condition candidate log-odds scoring
- family별 acceptance criteria와 cluster statistics
- Calibration 문항의 Pilot/Confirmatory non-reuse
- 결과 확인 후 threshold, item, template 또는 family 변경 금지

Authoritative dataset SHA-256:

`d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1`

## 3. v3.1 AI-only audit gate

다음 네 조건을 모두 만족해야 Baseline Calibration 구현 및 정적검토 단계로 이동할 수 있다.

1. frozen builder automatic audit 전체 PASS
2. primary nonhuman AI linguistic/semantic audit 192/192 PASS
3. 별도 adversarial AI-only 전수검수 192/192 PASS
4. 두 AI 검수 모두 model score, generation, LiReF, candidate/component 결과를 보지 않은 상태에서 수행

현재 hash-locked evidence:

- automatic audit SHA-256: `e904fcec13b97bf9d09afc707aedd2b37c100c911b560f2b88f0ed270e654e26`
- primary AI audit summary SHA-256: `d5e1743e27aa7e78e6879a5d6902cabd7998774844e67d1eb29de9133a70827a`
- adversarial AI-only review SHA-256: `94dcf1ea437ab2a16e30b2475778372e7c78e0e325c7f288d5a6a547e209f712`

세 evidence는 모두 PASS이므로 v3.1 AI-only audit gate는 충족된 것으로 기록한다.

## 4. Human audit 상태와 주장 제한

- independent human audit: `not_performed`
- human-audited evidence: `false`
- v3 frozen human gate가 충족됐다는 주장: 금지
- `independent reviewers confirmed the items`라는 표현: 금지
- 향후 결과 보고 시 `AI-only audited controlled dataset`이라고 명시

AI-only audit는 문항의 계산·binding·문법·counterbalance를 강하게 검수하지만, 독립 인간 판단의 대체 증거로 주장하지 않는다. 이 변경으로 인한 방법론적 한계는 발표·보고서·논문에 공개한다.

## 5. 실행 gate

이 amendment freeze만으로 모델 실행을 허용하지 않는다.

다음 순서를 고정한다.

```text
v3.1 audit-policy freeze
-> Baseline Calibration v3 implementation
-> static safety/schema review
-> implementation/dataset/amendment hash lock
-> separate explicit execution authorization
-> protocol-authorized AI-only-audited Baseline Calibration
```

현재 금지:

- model loading / forward / GPU execution
- LiReF direction 또는 candidate state loading
- hidden-state capture
- hook, patching, suppression, intervention
- Stage E Pilot

## 6. Calibration 이후

사전 동결된 모든 family-level 기준을 동시에 적용한다. PASS family가 없으면 Pilot으로 진행하지 않는다. PASS family가 있으면 Calibration item을 재사용하지 않고 새로운 Pilot dataset을 생성한다.

이 amendment는 Baseline Calibration을 R/M 결과나 causal mechanism으로 바꾸지 않는다. R/M 연결성은 이후 별도 Pilot에서만 검사한다.

## 7. Freeze 기록

- approved pre-freeze policy SHA-256: `688c54a4cc4b2b292b78bf01b4beb04aa57a8ae34cc3bc6aba38a5043c148e17`
- frozen manifest: `calibration_v3_1_ai_only_audit_policy_frozen.json`
- frozen manifest SHA-256: `15ce7892f12e00360b07ce533188249229dc24409119bd96e4c2c25c1bf8f9de`
- execution allowed: `false`
- next allowed work: Baseline Calibration v3 implementation 및 static review
