# Stage E Calibration v2.1.1 AI-only Audit 정책 변경 기록

상태: **FROZEN AMENDMENT — AI-ONLY AUDIT 허용, Calibration 실행은 별도 승인 필요**

변경일: 2026-08-30

## 1. 변경 이유

사용자가 v2.1 frozen design의 독립 human reviewer gate를 사용하지 않고, 해당 검수 항목을 AI가 전수검사하도록 명시적으로 요청했다.

기존 `calibration_v2_1_design_frozen.json`과 v2.1.1 dataset artifact는 수정하거나 덮어쓰지 않는다. 이 문서는 audit gate에만 적용되는 별도 amendment다. 설계, dataset 내용, scoring, acceptance threshold, counterbalance와 non-reuse 규칙은 변경하지 않는다.

## 2. 공식 기록 방식

다음 표현을 고정한다.

```text
human_audit: not_performed
human_audit_gate: waived_by_user
human_audited_evidence: false
ai_only_audit: required
ai_audited_evidence: true only after every AI audit pass completes
```

AI를 `human_reviewer_1` 또는 `human_reviewer_2`로 기록하지 않는다. 기존 reviewer CSV를 자동으로 채우거나 human PASS로 변환하지 않는다.

## 3. AI-only audit gate

다음 네 단계가 모두 PASS해야 한다.

1. frozen automatic audit 192/192 및 24/24 PASS
2. 기존 model-blind AI linguistic pre-audit 192/192 PASS
3. 새 structured AI-only audit에서 각 pair의 10개 항목 192/192 PASS
4. 24개 template-family adversarial semantic review PASS

Pair별 10개 항목:

1. 산술 정답 정확성
2. direct answer 정확성
3. label → key → record binding의 명확성과 유일성
4. Relevant transformation 필요성
5. Irrelevant matched keyed retrieval 타당성
6. Relevant answer-copy shortcut 부재
7. 질문 target 외 구조 동일성
8. 문법 자연스러움과 변화 방향의 명확성
9. 8-frame counterbalance counterpart 동등성
10. one-Arabic-numeral output instruction 명확성

하나라도 `FAIL` 또는 `NEEDS_REVISION`이면 audit gate는 FAIL이다. 해당 dataset을 사후 부분 삭제하거나 고쳐 쓰지 않고 새 dataset version을 만든다.

## 4. 독립성 한계

복수 AI pass는 오류 탐지 관점을 분리하기 위한 절차이며 독립 인간 reviewer를 구성하지 않는다. 동일 AI 또는 동일 계열 모델이 앞선 결과를 알 수 있으므로 `independent review`, `blind human review` 또는 `human-audited`라고 표현하지 않는다.

## 5. 후속 허용 범위

AI-only audit가 완전히 PASS하면 다음 단계인 Baseline Calibration 구현·정적 safety review·별도 execution authorization 준비로 이동할 수 있다.

AI-only audit PASS 자체는 다음을 허용하지 않는다.

- 즉시 model forward 또는 GPU Calibration 실행
- Stage E Pilot 자동 진입
- Calibration 문항의 Pilot 재사용
- confirmatory claim
- exploratory 결과를 이용한 family 사후 선택
- patching 또는 suppression

Baseline Calibration은 별도 실행 승인이 있어야 하며, 결과에는 `human_audit=not_performed`, `gate=waived_by_user`, `ai_only_audited=true`를 기록한다.

## 6. 해석 한계

이 변경으로 얻는 것은 AI-only audited controlled dataset이지 human-audited dataset이 아니다. Calibration 또는 Pilot 결과를 논문·발표에서 사용할 경우 human 검수가 없었다는 제한을 명시한다. Confirmatory evidence로 승격하려면 별도의 사전 설계와 검수 정책이 필요하다.

## 7. 보존되는 parent artifact

- parent frozen design SHA-256: `d0d6d19432eb48234a9729cc5f297c09963eb485bf5ec02178d504a93dc8307f`
- dataset SHA-256: `322d80f3c2c0723f6a4e8b0a968b30baac23e28cf68b6c60c5a7a95a5bca7420`
- automatic audit SHA-256: `93cd89e6901772c87435eb9f208bd4fc310e032b660c7678c34acf044f348e3b`
- existing AI pre-audit CSV SHA-256: `0c6de07362252dc3bce44379c81c8ca74558c455e34bd5e95fe9fd56f059ef88`
- existing AI pre-audit summary SHA-256: `3cf6d5645134f66692395d17297eb445194565e66504ed3d14ef382fded4d339`
- supplemental adversarial review SHA-256: `a6a4ab82812a2603174b6bebc739d9488be1a97bd2da8c87f5ca74b36fd0fbdd`

