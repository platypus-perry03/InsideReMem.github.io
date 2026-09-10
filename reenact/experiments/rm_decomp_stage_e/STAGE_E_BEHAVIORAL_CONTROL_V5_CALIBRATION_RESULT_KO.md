# Stage E Behavioral Control v5 Calibration 결과

상태: **COMPLETE — BOTH FAMILIES FAIL — REPLICATION BLOCKED**  
실행일: `2026-08-30`  
run ID: `stagee_v5_calibration_20260830_01`

## 1. 목적과 실행 범위

v4의 multi-hop binding floor를 해결하기 위해 case/key/ledger를 제거하고 single-step Arithmetic과 explicit-tag Selector로 단순화했다.

- 모델: local Meta-Llama-3-8B base
- Calibration: 64 pairs / 128 prompts
- family당 32 pairs
- device / batch / dtype: `cuda:1` / `8` / `float32`
- A/B teacher-forced margin과 한 토큰 greedy 출력만 측정
- sealed replication pool, LiReF, candidate component, hidden state, hook, intervention: 미사용
- human audit: `not_performed`; human-audited evidence가 아님

## 2. Frozen hard-gate 결과

| Family | Arithmetic FC | Selector FC | Arithmetic generation | Selector generation | mean `D_k` | 최종 |
|---|---:|---:|---:|---:|---:|---|
| points-balance | 17/32 | 18/32 | 17/32 | 18/32 | -0.04697 | **FAIL** |
| temperature | 17/32 | 21/32 | 17/32 | 21/32 | -0.02126 | **FAIL** |

Hard range는 각 조건 22–29/32이며 gap은 최대 3/32다.

- points-balance: gap과 mean `D_k`는 PASS했지만 네 absolute solvability count가 모두 하한 미달
- temperature: 네 solvability count가 모두 하한 미달이고 condition count gap도 4로 FAIL
- `d_z`: points-balance -2.5592, temperature -0.1747로 계산·보고했지만 frozen rule대로 PASS/FAIL에는 사용하지 않음
- 결과를 보고 threshold, item, template 또는 family를 변경하지 않음

## 3. 추가 진단

- points-balance Arithmetic 출력: A 23회 / B 9회
- points-balance Selector 출력: A 24회 / B 8회
- temperature Arithmetic 출력: A 17회 / B 15회
- temperature Selector 출력: A 17회 / B 15회
- 각 조건의 정답 A/B는 정확히 16:16으로 counterbalance됨

points-balance에서는 강한 A 선택 편향이 남았다. temperature에서는 위치 편향은 크게 줄었지만 17/32와 21/32로 사전 solvability 하한에 도달하지 못했다.

## 4. v4 진단의 수정

v5에서 binding hop을 크게 줄였는데도 floor가 지속됐다. 따라서 v4 실패를 multi-hop symbolic binding 하나만으로 설명할 수 없다.

현재 근거가 지지하는 더 좁은 진단은 다음과 같다.

> **Meta-Llama-3-8B base가 이 zero-shot paired-context A/B instruction 형식에서 terminal rule을 안정적으로 수행하지 못하며, 일부 wording/family에서는 A 위치 prior도 사용한다.**

이는 Arithmetic/Selector 내부 차이나 Layer 31 가설의 반증이 아니다. Behavioral task를 안정적으로 수행하지 못했기 때문에 그 가설을 독립적으로 시험하지 못한 것이다.

## 5. Frozen stopping rule

`passed_families=[]`이므로:

- points-balance independent Layer 31 replication: 미실행
- points-balance–temperature interaction: 미실행
- Layer trajectory와 후보 4개 scalar: 미실행
- intervention / patching / suppression: 미실행
- v5 replication pool: 모델에 사용하지 않은 봉인 상태로 보존

## 6. Provenance

- design SHA-256: `033969f85f9f982ff686c22f4bdd3977baae1dcc76a4f91b537e758a8fd98982`
- Calibration dataset SHA-256: `18005ac5c5733ab389f1b9f8d4850a671f4bdfa0f0c4755ddd62a43e4348922a`
- sealed replication dataset SHA-256: `b5bd9b083ebe7319bef02ebd30c064e9b9fe2efe85539b668ea8d1946f719dd3`
- implementation SHA-256: `0a40acef8c0198b741ad3dee57181f4ee2c4380f66bca90b6b9535fd00f591d7`
- static review SHA-256: `ff39bdbfa64bb9d5850e4ba4d1d4ab08e057dca0f5634adc90d982fc2432dd23`
- authorization SHA-256: `1787504766c1ba962cde71c9bb8118dc9e5a6cdd8ede5402975b9f2becc8d9e3`
- pair results SHA-256: `6e00a4d1133fedf079dc38d75c98204f6358a01251150455f4022ea741d022bb`
- summary SHA-256: `fea030f9933c95acbc3b88c43cb16349efdbf4810eb174c4d129464251067f16`
- run manifest SHA-256: `5a0a453bd4127c25d71f37ab9b52455306535b54745e97c4195a8333e93b5a45`

## 7. 다음 설계에서 분리해 확인할 것

새 버전이 필요하면 큰 dataset을 바로 만들지 말고 model behavior만 보는 소규모 format diagnostic을 먼저 사전설계한다.

1. 단일 block·단일 질문에서 ADD/SUB와 tag selection 각각의 solvability 확인
2. 숫자 대신 A/B를 출력하는 mapping 부담을 별도로 확인
3. shared paired context가 단일 context보다 성능을 낮추는지 확인
4. zero-shot base-model instruction 형식과 minimal worked-example 형식을 사전 분리
5. format을 고른 뒤 새 Calibration/replication item을 생성하고 진단 문항은 재사용하지 않음

