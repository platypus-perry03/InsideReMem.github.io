# Stage E Transformation Deep-Dive 현재 결과

상태: **COMPLETE — LOCAL FINE-ANNOTATION UNRELIABLE — OBSERVABLE STRATIFICATION COMPLETE**  
실행일: `2026-08-31`

## 1. 질문

기존 `transformation_required` association이 구체적으로 무엇을 나타내는지 확인했다.

1. 계산·규칙·관계·조건·인과·중간결과로 신뢰성 있게 세분화할 수 있는가?
2. 단순히 숫자가 포함된 문제라서 내부 반응이 높았던 것인가?

외부 API와 Meta-Llama-3-8B base의 새 forward는 사용하지 않았다.

## 2. Local fine-annotation 결과

### v1.2 자유형 multi-label

- Annotator A parse: 92/96
- Annotator B parse: 2/96
- 공동 유효: 2/96
- 결과: **FAIL**, 895문항 full annotation 미실행

### v1.3 teacher-forced Y/N

자유 생성과 parser를 제거하고 각 primitive별 `logit(Y)-logit(N)`으로 판정했다.

- 두 annotator coverage: 96/96
- `CAUS`: agreement 0.563, κ 0.170 → UNRELIABLE
- `COND`, `NUM`: positive prevalence 0.984 → INSUFFICIENT_PREVALENCE
- `RULE`, `REL`, `INTER`: positive prevalence 1.000 → INSUFFICIENT_PREVALENCE
- usable primitive: 0개
- 결과: **FAIL**, 895문항 full annotation 및 internal outcome 결합 미실행

즉 형식 문제를 없애도 두 local annotator가 세부 primitive를 신뢰성 있게 구분하지
못했다. threshold를 낮추거나 거의 전부 Y인 feature를 사후 채택하지 않았다.

## 3. 새 annotation 없이 확인한 데이터 구조

기존 확정 3,000문항에서 `transformation_required=Y`는 숫자 포함 문항에 매우 집중됐다.

- 전체: 895개 중 869개 숫자 포함 (`97.1%`)
- Discovery: 723개 중 704개 (`97.4%`)
- Validation: 172개 중 165개 (`95.9%`)
- non-numeric Transformation Y: 전체 26개뿐

따라서 현재 `transformation_required`를 arithmetic·logic·formal·causal 전반의 일반
Transformation이라고 해석할 수 없다. 이 dataset에서는 주로 **수치 계산/변환 요구**를
포착하는 operational feature에 가깝다.

## 4. Numeric-only 분석

숫자가 포함된 문항끼리만 비교하고 R/M label, token length, option count, source를
조정했다.

### Discovery

| Endpoint | T=N / T=Y | 표준화 β | 95% CI | BH q | 판정 |
|---|---:|---:|---:|---:|---|
| Layer 31 LiReF | 407 / 704 | +0.3430 | [+0.1921, +0.4939] | 0.0000091 | selected |
| `L29H30` | 407 / 704 | +0.3631 | [+0.2687, +0.4574] | 2.82e-13 | selected |
| `L30H6` | 407 / 704 | +0.3989 | [+0.2768, +0.5210] | 3.29e-10 | selected |

### 기존 heldout secondary check

| Endpoint | T=N / T=Y | 표준화 β | 95% CI | BH q | 판정 |
|---|---:|---:|---:|---:|---|
| Layer 31 LiReF | 102 / 165 | +0.4807 | [+0.1162, +0.8452] | 0.0300 | **supported** |
| `L29H30` | 102 / 165 | +0.2500 | [-0.0066, +0.5065] | 0.0713 | not supported |
| `L30H6` | 102 / 165 | +0.2961 | [-0.0260, +0.6182] | 0.0713 | not supported |

숫자 문자 존재 자체를 같게 제한해도 Layer 31 association은 Discovery와 기존 heldout에서
유지됐다. 두 attention head는 방향은 같았지만 heldout CI와 다중비교 기준을 통과하지
못했다.

## 5. 중요한 제한

Numeric-only에서도 Transformation과 R label의 중첩은 매우 강하다.

- Discovery numeric T=Y: 704개 중 691개 R
- Validation numeric T=Y: 165개 중 163개 R

회귀에서 R/M label을 조정했지만 M+Transformation 반례가 매우 적다. 따라서 현재
결과로 `수치 변환이 Layer 31을 만든다`거나 `R/M을 좌우한다`고 말할 수 없다.
또한 기존 heldout은 feature 발견에 사용된 3,000문항의 일부이므로 독립 confirmatory가
아니다.

## 6. 현재 가장 안전한 결론

> **현재 LiReF 데이터의 `transformation_required`는 주로 수치 계산·변환 요구를
> 포착한다. 숫자가 포함된 문항끼리 비교해도 Layer 31 R-direction projection과의
> association은 유지됐지만, R label과의 강한 중첩 때문에 독립적·인과적 특징으로는
> 아직 확정할 수 없다.**

## 7. Provenance

- observable design SHA-256: `32bd2f6b14dc1ec280956d55e25f4b8dfb7a1ed83261371ebcd11af7bb0f61f1`
- observable implementation SHA-256: `faa9b245b243975d42f5a2a6a65fb00c135e85bdfde1a9f4fe3e9418d4ccb43a`
- v1.2 preflight SHA-256: `99025cd863884910fa8586272eb3df0661f677fcbe8ca37c231118c9568ce0c2`
- v1.3 preflight SHA-256: `03ec456ffe74db23b5fe6edceb1bdd47361055e61724cad60d7a60bca35a8083`
- Discovery numeric table SHA-256: `e45a4a1ebb9ab90a2d560b68d7e249e7b6ae8a6a0ee17425b155646bec48dc2b`
- heldout table SHA-256: `038649331e1cd3a2502eea32ab8f8d67ba660fd168ec312b3a215ade8369763b`
- summary SHA-256: `96b0f0c2fc2358792d25ae4b3a23d68c0f3f19b2c4faf24ed6456ac4a01c296a`

