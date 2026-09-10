# Stage E 제한적 Pilot 진행 Amendment v3.2

상태: **FROZEN — LIMITED SAME-SAMPLE PILOT CONTINUATION APPROVED — MODEL EXECUTION NOT AUTHORIZED**  
동결일: `2026-08-30`

## 1. 변경하지 않는 사실

Baseline Calibration v3.1의 공식 판정을 소급 변경하지 않는다.

```text
object-count: FAIL
points-balance: FAIL
temperature: FAIL
passed_families: []
```

따라서 v3.1 dataset이 완전한 behavioral equivalence를 달성했다거나 Calibration을 PASS했다고 표현하지 않는다. Human audit도 수행되지 않았으므로 human-audited evidence라고 부르지 않는다.

## 2. 제한적 진행 근거

- forced-choice 조건 간 count gap은 세 family 모두 frozen 기준을 통과했다: `0`, `3`, `4`.
- `|mean(D_k)|`는 세 family 모두 통과했다: `0.1574`, `0.0411`, `0.0006`.
- v2/v2.1.1에서 나타난 Selector ceiling은 제거됐다. Selector forced-choice는 `36/64`, `32/64`, `40/64`였다.
- 반면 절대 forced-choice 하한, Arithmetic generation, generation condition gap과 object-count `d_z` 때문에 전체 hard gate는 실패했다.

따라서 v3 dataset을 완전한 난이도 동등화 자료가 아니라 **controlled-but-imperfect contrast**로만 사용한다.

## 3. 기존 non-reuse 규칙의 명시적 제한 override

기존 v3 design과 dataset manifest의 `reuse_in_pilot=false` 및 `pilot_reuse_allowed=false`를 조용히 무시하지 않는다. 이 amendment는 frozen v3 dataset 192 pair 전부를 아래 목적의 동일표본 Pilot에 한해서만 재사용하도록 명시적으로 허용한다.

- 분석 유형: **limited same-sample hypothesis-testing Pilot**
- independent Pilot: 아님
- confirmatory evidence: 아님
- external validation: 아님
- 대상: 세 family와 192 pair 전부
- Calibration 결과를 근거로 한 family/template/item 제외: 금지
- primary analysis를 정답 문항에 한정: 금지

이 동일표본 Pilot에서 얻은 효과는 독립 재현으로 해석하지 않는다. 독립 Pilot 또는 confirmatory claim에는 별도의 새 non-overlapping dataset이 필요하다.

## 4. Pilot에서 허용할 질문

전체 192 pair를 먼저 사용하여 다음을 평가한다.

1. Arithmetic과 Selector 조건의 LiReF projection 차이
2. 기존 네 candidate의 조건 차이
   - `L31N13336`
   - `L29H00030`
   - `L30H00006`
   - `L29H00031`
3. condition × semantic family interaction

보조·진단 분석으로만 다음 stratification을 허용한다.

- Arithmetic behavior correct vs incorrect
- ADD vs SUB
- family별 behavior와 internal effect의 관계

이 분석은 primary 전체표본 결과를 대체하거나 사후 subset만 선택하기 위한 용도로 사용하지 않는다.

## 5. 해석 제한

허용되는 결론은 통제됐지만 불완전한 contrast에서 LiReF와 기존 candidate가 선택적으로 반응하는지에 대한 Pilot evidence다.

다음 주장은 금지한다.

- 완벽히 난이도가 동일한 R/M task를 확보했다.
- Arithmetic transformation이 R representation을 만든다.
- 특정 candidate가 R/M을 결정하거나 매개한다.
- broad R/M causal mechanism을 입증했다.
- 같은 192문항 분석을 independent replication 또는 confirmatory evidence로 부른다.

모든 Pilot 결과는 v3.1 behavioral failure, 특히 Arithmetic generation과 SUB 실패와 함께 보고한다.

## 6. 현재 실행 gate

이 amendment는 Pilot 진입 원칙과 설계·구현 작업만 승인한다.

```text
pilot design/specification: allowed
pilot implementation: allowed
static safety/schema review: allowed
Pilot model execution: not authorized
intervention: not authorized
patching/suppression: not authorized
confirmatory claim: not authorized
```

다음 순서는 Pilot design freeze → implementation → static review → 별도 execution authorization이다. Intervention은 Pilot 결과를 확인한 뒤 별도 사전 명세와 승인이 필요하다.

