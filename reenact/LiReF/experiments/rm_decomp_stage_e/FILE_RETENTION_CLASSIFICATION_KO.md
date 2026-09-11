# `rm_decomp_stage_e` 파일 보존 분류

작성일: 2026-08-30  
최종 재점검: 2026-08-30 (`v3 official dataset + AI audit` 이후)  
대상 snapshot: 파일 108개 / 디렉터리 15개 / 약 8.12 MB  
상태: **legacy Python source 15개 제거 완료 — 현재 v3 source 2개 유지**

> **2026-08-30 정리 실행:** 사용자의 명시적 요청에 따라 v1, v2, v2.1, v2.1.1 및 protocol-deviating exploratory Python source 15개를 제거했다. 삭제 전 파일명과 SHA-256은 `legacy_code_removal_record.json`에 기록했다. 아래의 이전 버전 코드 관련 KEEP/ARCHIVE 목록은 삭제 전 분류의 역사적 설명이며, 해당 source는 현재 존재하지 않는다. Dataset, result, manifest, audit, authorization, failure 및 hash/provenance artifact는 유지했다.

## 분류 원칙

- **KEEP:** 현재 결론의 근거, frozen artifact, 최신 실행 코드, 또는 최신 코드가 경로/hash로 직접 참조하는 dependency
- **ARCHIVE:** 현재 실행에는 필요하지 않지만 설계 변경·실패·waiver의 연구 provenance로 보존할 이전 버전
- **DELETE CANDIDATE:** 원본에서 재생성 가능하고 연구 provenance가 없는 cache만 해당

`ARCHIVE`는 즉시 이동한다는 뜻이 아니다. 일부 코드는 상대경로와 SHA-256 lock을 사용하므로, 실제 이동 전에는 version bundle을 만들고 dependency를 다시 점검해야 한다.

## KEEP

### 0. 현재 v3 공식 dataset 및 audit pipeline

- `CALIBRATION_V3_DESIGN_DRAFT_KO.md`
- `CALIBRATION_V3_DATASET_AND_AI_AUDIT_RESULT_KO.md`
- `STAGE_E_CURRENT_STATUS_KO.md`
- `build_calibration_v3_dataset.py`
- `run_ai_linguistic_audit_v3.py`
- `calibration_v3_design_frozen.json`
- `calibration_v3_design_freeze_record.json`
- `calibration_v3_builder_verification.json`
- `calibration_v3_assets/` 전체

v3 builder는 v2, v2.1, v2.1.1 dataset hash를 직접 검증하여 non-reuse audit을 수행하므로 세 이전 dataset 계열도 현시점에는 삭제하거나 이동하지 않는다.

### 1. 현재 해석과 발표의 직접 근거

- `LITERATURE_METHOD_COMPARISON_KO.md`
- `STAGE_E_PROTOCOL_DRAFT_KO.md`
- `CALIBRATION_SHORTCUT_REVIEW_KO.md`
- `CALIBRATION_V2_BASELINE_RESULT_KO.md`
- `CALIBRATION_V2_1_DESIGN_DRAFT_KO.md`
- `CALIBRATION_V2_1_1_AI_ADVERSARIAL_REVIEW_KO.md`
- `CALIBRATION_V2_1_1_DATASET_AND_AI_PREAUDIT_RESULT_KO.md`
- `CALIBRATION_V2_1_1_AI_ONLY_AUDIT_AMENDMENT_KO.md`
- `CALIBRATION_V2_1_1_AI_ONLY_AUDIT_GUIDE_KO.md`
- `CALIBRATION_V2_1_1_AI_ONLY_AUDIT_RESULT_KO.md`
- `CALIBRATION_V2_1_1_BASELINE_RESULT_KO.md`
- `STAGE_E_EXPLORATORY_GATE_BYPASS_RESULT_KO.md`
- 이 분류 문서 `FILE_RETENTION_CLASSIFICATION_KO.md`

### 2. 현재 v2.1.1 dataset/audit pipeline

- `build_calibration_v2_1_1_dataset.py`
- `build_calibration_v2_1_dataset.py`
- `prepare_calibration_v2_1_1_human_audit.py`
- `run_ai_linguistic_preaudit_v2_1.py`
- `run_ai_linguistic_preaudit_v2_1_1.py`
- `run_ai_only_audit_v2_1_1.py`
- `calibration_v2_1_design_frozen.json`
- `calibration_v2_1_design_freeze_record.json`
- `calibration_v2_1_builder_verification.json`
- `calibration_v2_1_1_dataset_revision_record.json`
- `calibration_v2_1_1_builder_verification.json`
- `calibration_v2_1_1_ai_only_audit_policy_frozen.json`
- `calibration_v2_1_1_ai_only_audit_static_review.json`
- `calibration_v2_1_assets/` 전체
- `calibration_v2_1_1_assets/` 전체
- `CALIBRATION_V2_1_1_HUMAN_AUDIT_GUIDE_KO.md`

빈 reviewer CSV도 삭제하지 않는다. 실제 human audit이 수행되지 않았다는 provenance이자 향후 audit 재개용 worksheet다.

### 3. 현재 Baseline Calibration 및 Exploratory 재현

- `run_calibration_v2_1_1_baseline.py`
- `review_calibration_v2_1_1_baseline_static.py`
- `calibration_v2_1_1_execution_authorization_frozen_run02.json`
- `run_stage_e_exploratory_gate_bypass.py`
- `stage_e_exploratory_gate_bypass_design.json`
- `stage_e_exploratory_gate_bypass_execution_authorization_run02.json`
- `stage_e_exploratory_gate_bypass_postrun_validation.json`
- `stage_e_exploratory_gate_bypass_static_review.json`

### 4. 겉보기에는 구버전이지만 현재 코드의 hash-locked dependency

- `build_calibration_v2_dataset.py`
- `calibration_v2_assets/calibration_v2_dataset_draft.json`
- `calibration_v2_assets/calibration_v2_dataset_manifest.json`
- `calibration_v2_assets/calibration_v2_automatic_audit.json`
- `run_calibration_v2_baseline.py`
- `review_calibration_v2_baseline_static.py`
- `calibration_v2_assets/calibration_v2_baseline_static_safety_review.json`
- `calibration_v2_design_frozen.json`
- `calibration_v2_design_freeze_record.json`

v2.1 builder는 v2 dataset hash를, v2.1.1 builder는 v2.1 builder/dataset/audit/pre-audit hash를, v2.1.1 baseline runner는 v2 baseline implementation/static-review hash를 직접 검증한다. 이 파일을 이동하거나 수정하면 재현 gate가 깨진다.

## ARCHIVE

### 1. v1 calibration 계열

- `CALIBRATION_LINGUISTIC_AUDIT_GUIDE_KO.md`
- `CALIBRATION_MANIFEST_DRAFT_KO.md`
- `build_calibration_dataset.py`
- `run_ai_linguistic_preaudit.py`
- `calibration_design_frozen.json`
- `calibration_assets/` 전체

v1은 answer-copy shortcut을 발견한 실패 provenance이므로 삭제하지 않는다.

### 2. v2의 이전 검토·AI audit·authorization 문서

- `CALIBRATION_V2_AI_PREAUDIT_GUIDE_KO.md`
- `CALIBRATION_V2_AI_PREAUDIT_RESULT_KO.md`
- `CALIBRATION_V2_BASELINE_IMPLEMENTATION_REVIEW_KO.md`
- `CALIBRATION_V2_BUILDER_CORRECTION_KO.md`
- `CALIBRATION_V2_DESIGN_DRAFT_KO.md`
- `CALIBRATION_V2_HUMAN_AUDIT_WAIVER_KO.md`
- `run_ai_linguistic_preaudit_v2.py`
- `calibration_v2_assets/ai_preaudit/` 전체
- `calibration_v2_assets/calibration_v2_static_schema_check.json`
- `calibration_v2_execution_authorization_draft.json`
- `calibration_v2_execution_authorization_freeze_record.json`
- `calibration_v2_execution_authorization_freeze_record_run02.json`
- `calibration_v2_execution_authorization_frozen.json`
- `calibration_v2_execution_authorization_frozen_run02.json`

위 목록 중 `calibration_v2_assets/`의 KEEP 항목은 archive 이동 대상에서 제외한다.

### 3. v2.1에서 v2.1.1로 넘어가기 전 검토 문서

- `CALIBRATION_V2_1_AI_PREAUDIT_GUIDE_KO.md`
- `CALIBRATION_V2_1_AI_PREAUDIT_RESULT_KO.md`

`calibration_v2_1_assets/`는 v2.1.1 builder의 hash-locked parent이므로 문서상 과거 버전이어도 실제 파일은 KEEP이다.

### 4. 실패한 실행 시도와 superseded authorization

- `calibration_v2_1_1_attempt01_failure.json`
- `calibration_v2_1_1_execution_authorization_frozen.json`
- `stage_e_exploratory_gate_bypass_attempt01_failure.json`
- `stage_e_exploratory_gate_bypass_execution_authorization.json`

실패가 model forward 이전이었음을 증명하는 provenance이므로 삭제하지 않는다.

## DELETE CANDIDATE

현재 삭제 후보는 없다.

재점검에서 다음 항목이 모두 0개임을 확인했다.

- `__pycache__/`, `*.pyc`
- `.DS_Store`, `.ipynb_checkpoints/`
- `*.tmp`, `*.bak`, editor swap/backup
- `.cache/`, `.matplotlib/`
- 0-byte 파일

과거에 분류했던 Python bytecode는 이미 정리되어 있다. 남아 있는 파일은 현재 v3 pipeline, hash-locked dependency 또는 실패·waiver·실행 provenance에 해당한다.

## 실제 정리 시 권장 구조

추후 사용자가 이동을 승인하면 다음처럼 분리하는 것이 안전하다.

```text
rm_decomp_stage_e/
  current/                 # v2.1.1 + exploratory + 현재 결과
  dependencies/            # hash-locked v2/v2.1 parent artifact
  archive/v1/
  archive/v2/
  archive/v2_1_review/
  archive/failed_attempts/
```

단, 현재 코드는 기존 상대경로를 전제로 하므로 실제 이동 전 dependency path 수정, 새 static review 및 새 hash 기록이 필요하다.
