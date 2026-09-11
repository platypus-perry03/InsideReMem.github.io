# Stage E Calibration v3 Dataset 및 AI Audit 결과

상태: **OFFICIAL DATASET CREATED — AUTOMATIC AUDIT PASS — AI AUDIT 192/192 PASS — INDEPENDENT HUMAN AUDIT PENDING**  
기준일: `2026-08-30`

## 1. 수행 범위

- frozen design과 hash-locked deterministic builder로 공식 v3 dataset 생성
- `object_count / points_balance / temperature` 각 64 pair
- 총 24 template families × 8 OA frames = 192 pairs / 384 prompts
- automatic shortcut·collision·counterbalance·tokenizer·non-reuse audit
- 비인간 AI reviewer의 192/192 전수 linguistic/semantic audit

모델 weight, model forward, GPU, LiReF direction, hidden/candidate state, hook 및 intervention은 사용하지 않았다.

## 2. Automatic audit

판정: **PASS**

- 192개 모든 pair-level check PASS
- 각 family ADD/SUB 32:32
- 정확한 frozen `OA(8,5,2,2)` matrix 사용
- 각 factor 4:4 및 모든 factor pair의 2×2 cell 각 2회
- arithmetic correct/foil context 비노출
- selector correct/foil ledger 내 각 1회 노출
- wrong-operation arithmetic primary foil 192/192 확인
- canonical/alternative answer one-token contract 192/192 확인
- prompt token length 및 target-label token count pair match 192/192 확인
- v2/v2.1/v2.1.1 template non-reuse PASS
- prior template 대비 최대 normalized 5-gram Jaccard: `0.054054...` (`FAIL >= 0.80`)

## 3. AI audit

판정: **192/192 PASS**

각 pair에 대해 다음을 전수 확인했다.

- arithmetic 정답과 frozen wrong-operation foil의 계산 정확성
- ADD/SUB 및 temperature WARM/COOL 방향의 명확성
- selector active tag → entry → value key → ledger value 결합 정확성
- 양 조건의 label → case → value-key → ledger 경로 유효성
- pair context와 prompt contract 동일성
- 질문 target label 외 구조 동일성
- arithmetic transformation 필요성과 answer-copy shortcut 부재
- selector matched retrieval 정답/경쟁 entry의 직접 근거 존재
- correct/foil/operand/ledger collision 부재
- one Arabic numeral 출력 지시의 명확성
- frozen OA row와 factor metadata 일치
- 24개 template의 문법·자연스러움·의미 명확성
- template별 8개 counterbalance counterpart의 구조적 동등성

이 AI audit는 automatic audit을 보강하는 비인간 검수이며 독립 human audit를 대체하지 않는다.

## 4. Artifact hash

- frozen design SHA-256: `c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc`
- builder SHA-256: `b3ba73e505a1c081a1180dada8193411b7ca4309d01a6283ee07513a3f125b94`
- static schema check SHA-256: `c697b9c3e37b297fb8a9d900d9fc88d22b5453e91551e9439c0b84bddfb33d55`
- official dataset SHA-256: `d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1`
- automatic audit SHA-256: `e904fcec13b97bf9d09afc707aedd2b37c100c911b560f2b88f0ed270e654e26`
- dataset manifest SHA-256: `a157ec7bd463c739ee046f9e3f85a08d2e3ebb7dc6a794a63757653c93fad822`
- AI audit implementation SHA-256: `1a64a2f481b599b8a5314e8d284ae402dc599d03eb455edd02e61a0773fded71`
- AI audit CSV SHA-256: `2695b1a8d1020fe9ba58de81fed14501826508e547c4260130275ca09776f3cc`
- AI audit summary SHA-256: `d5e1743e27aa7e78e6879a5d6902cabd7998774844e67d1eb29de9133a70827a`

## 5. 현재 gate

- AI audit: `PASS (192/192)`
- independent human audit: `not_started`
- human-audit waiver: `not_allowed`
- Baseline Calibration execution: `not_authorized`
- Stage E Pilot: `not_authorized`

다음 허용 작업은 독립 human reviewer 2명이 각각 192/192 pair를 blind audit하는 것이다. 두 reviewer가 모두 PASS하고 필요한 adjudication이 완료되기 전에는 Baseline Calibration 실행 승인으로 이동하지 않는다.

이 결과는 controlled dataset의 검수 결과다. R/M separation, LiReF 방향 이동, component mediation 또는 causal mechanism을 입증하지 않는다.
