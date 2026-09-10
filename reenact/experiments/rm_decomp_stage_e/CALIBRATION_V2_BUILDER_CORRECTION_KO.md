# Calibration v2 Builder 정적 안전성 보완 기록

상태: **IMPLEMENTATION CORRECTION COMPLETE — AUTOMATIC AUDIT PASS, HUMAN AUDIT PENDING**  
날짜: 2026-08-29

## 보완 사유

첫 v2 dataset draft 생성 후 builder 검토에서 다음 두 보완점이 확인되었다.

1. Dataset-only tokenizer 로딩에 `trust_remote_code=True`가 남아 있었다.
2. Builder가 frozen manifest의 key를 실제로 읽기는 했지만, 실행 전에 전체 required schema를 명시적으로 검사하는 별도 static compatibility gate가 없었다.

Dataset 의미 규칙의 변경은 아니다. 정적 안전성과 fail-fast 검증을 강화하는 구현 수정이다.

## 보완 전 artifact

| Artifact | SHA-256 |
|---|---|
| Frozen v2 design | `a8f3dad7fced945377194074f9aa12d673faff3b55c3ec45bd82e397b5a5302b` |
| Builder | `6c0264f737854dba46152bdcd965bbe240172e2feb0da0ed5094373168491e14` |
| Dataset draft | `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87` |
| Automatic audit | `6e7bf7bde3965ae56f9d41cf0c82fec1b2da0414d190f32c19df45995d38ac8e` |
| Dataset manifest | `1798f5fa133700dd1ac675c6315e8d037dec9ecc88a06928db0676fb13f8921a` |

보완 전 결과는 baseline calibration, Pilot 또는 Confirmatory에 사용할 수 없다.

## 적용할 수정

- `trust_remote_code=False`
- `local_files_only=True` 유지
- model/tokenizer 실행 전 `--validate-only` static schema gate 추가
- frozen manifest의 required section/key/type/value compatibility 검사
- schema-check report와 SHA-256을 새 dataset manifest에 기록
- 같은 seed의 두 임시 출력이 byte-identical인지 재확인
- 공식 draft 재생성 후 dataset 의미 hash가 유지되는지 비교

Frozen design은 수정하지 않는다. Dataset 의미 hash가 달라지면 단순 implementation correction으로 처리하지 않고 원인을 조사하여 새 dataset version 여부를 결정한다.

## 보완 완료 결과

- Frozen manifest와 builder의 static schema compatibility: **PASS**
- tokenizer 설정: `trust_remote_code=False`, `local_files_only=True`
- 동일 seed 독립 생성 2회: schema report, dataset, automatic audit, manifest 모두 **byte-identical**
- 생성 규모: **144 pairs / 288 prompts / 18 template families**
- Automatic audit: **PASS**
- 모델 weight 로딩, forward, GPU, LiReF/hidden-state 접근: **없음**
- 현재 gate: **AI pre-audit 및 독립 human audit 대기**
- Baseline Calibration 실행: **금지 유지**

| 보완 후 Artifact | SHA-256 |
|---|---|
| Frozen v2 design | `a8f3dad7fced945377194074f9aa12d673faff3b55c3ec45bd82e397b5a5302b` |
| Corrected builder | `58bdf5d6b6294513c4f282b129586d090e3519b6589cf425849b912140980757` |
| Static schema check | `02be72dfa1df9da07781cba0aedada546c4cd1ba8eb54431912eac16209f9251` |
| Dataset draft | `c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87` |
| Automatic audit | `6e7bf7bde3965ae56f9d41cf0c82fec1b2da0414d190f32c19df45995d38ac8e` |
| Dataset manifest | `f79a80aa7b55a546724a3f639ebed22858addc39b21aaa73c65b495a9ce3898e` |

Dataset draft와 automatic audit hash가 보완 전과 동일하므로 이번 수정은 문항 의미나 audit 판정을 변경하지 않았다.
