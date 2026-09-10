# Stage E Natural R/M Feature Discovery v1 결과

상태: **COMPLETE — ONE VALIDATION-SUPPORTED ASSOCIATION FOUND — CAUSAL CLAIM NOT ALLOWED**  
실행일: `2026-08-31`

## 1. 질문

Stage A–C에서 찾은 기존 4개 component가 R/M label 자체가 아니라 어떤 자연
문항 특성과 함께 움직이는지 탐색했다.

- `L31N13336`
- `L29H00030`
- `L30H00006`
- `L29H00031`

새 synthetic dataset을 만들지 않고 Stage A와 같은 MMLU-Pro 3,000문항 및
기존 Discovery 2,400 / Validation 600 split을 사용했다.

## 2. Blind AI annotation

두 annotator에게는 질문과 선택지만 제공했다. R/M label, 정답, category,
source, split, LiReF 및 component 결과는 제공하지 않았다.

초기 v1.0 annotator는 `answer_mode=MIX` 3,000/3,000으로 붕괴해 annotation
instrument 실패로 보존했다. 내부 outcome을 보기 전에 예시와 `MIX` 판정 규칙을
명확히 한 v1.1 amendment를 동결했고, 두 annotator 모두 48-item preflight를
통과했다.

전수 annotation의 두 annotator 간 신뢰도:

| Feature | Raw agreement | Cohen's κ | feature 선택 사용 |
|---|---:|---:|---|
| transformation required | 0.878 | **0.737** | 허용 |
| answer directness | 0.687 | 0.416 | 금지 |
| answer mode | 0.577 | 0.350 | 금지 |
| composition required | 0.609 | 0.230 | 금지 |
| external knowledge required | 0.527 | 0.095 | 금지 |
| multi-step required | 0.648 | 0.042 | 금지 |

세 번째 Gemma adjudicator는 preflight를 통과했지만 full 실행에서 dynamic-shape
CUDAGraph 재컴파일로 종료 시간이 비정상적으로 늘어 중단했다. 더 중요한 점은
제3 모델 판정이 낮은 원 annotator 신뢰도를 해결하지 못한다는 것이다. 따라서
내부 outcome을 보기 전에 v1.2 amendment를 동결하고, **두 annotator가 정확히
일치한 feature 값만 사용하며 disagreement는 UNC로 제외**했다. κ≥0.60인
`transformation_required`만 feature 후보 선택에 허용했다.

## 3. Discovery 결과

source fixed effects, log token length, option count와 numeric mention을 통제한
Discovery 2,400문항 분석에서 `transformation_required`가 선택 기준을 통과했다.
내부 endpoint는 표준화했으므로 아래 β는 outcome SD 단위다.

| Endpoint | β | 95% CI | global BH q |
|---|---:|---:|---:|
| R label | +0.3079 | [+0.2348, +0.3809] | 2.25e-15 |
| Layer 31 LiReF | +0.5157 | [+0.3889, +0.6424] | 1.76e-14 |
| `L31N13336` | +0.2396 | [+0.0236, +0.4556] | 0.0357 |
| `L29H00030` | +0.4450 | [+0.3621, +0.5279] | 1.05e-23 |
| `L30H00006` | +0.5452 | [+0.4358, +0.6546] | 8.11e-21 |
| `L29H00031` | +0.3156 | [+0.2419, +0.3893] | 1.01e-15 |

이 결과만으로 feature를 확정하지 않고, 위 여섯 pair를 Validation 전에
`discovery_selection_frozen.json`에 동결했다.

## 4. Heldout Validation 결과

Validation 600문항 중 두 annotator가 transformation에 합의한 522문항을
사용했다(`N=350`, `Y=172`). Discovery에서 고정한 모든 pair가 같은 부호,
95% CI 0 제외 및 selected-test BH q<0.05를 만족했다.

| Endpoint | β | 95% CI | selected BH q |
|---|---:|---:|---:|
| R label | +0.3648 | [+0.2138, +0.5157] | 1.3e-5 |
| Layer 31 LiReF | +0.6890 | [+0.3976, +0.9805] | 1.3e-5 |
| `L31N13336` | +0.6441 | [+0.1228, +1.1654] | 0.0156 |
| `L29H00030` | +0.4210 | [+0.2252, +0.6168] | 4.4e-5 |
| `L30H00006` | +0.5369 | [+0.2911, +0.7826] | 4.3e-5 |
| `L29H00031` | +0.2291 | [+0.0576, +0.4005] | 0.0107 |

따라서 frozen selection rule상 validation-supported feature는 하나다.

> **Transformation이 필요하다고 두 AI annotator가 합의한 자연 문항은 R label,
> Layer 31 LiReF 및 기존 4개 component의 R-direction contribution과 함께
> 증가하는 연관 패턴을 보였고, 기존 heldout split에서도 같은 방향으로
> 재현됐다.**

## 5. 가장 중요한 post-hoc 진단

이 결과에는 강한 구조적 제한이 있다. Validation에서:

- transformation 없음: R 45/350 (`12.9%`)
- transformation 있음: R 170/172 (`98.8%`)

즉 transformation annotation이 R/M label과 거의 겹친다. 네 component 자체도
R/M gap으로 선택됐으므로 primary association의 일부는 이 label overlap에서
자동으로 생길 수 있다.

결과 확인 후 수행한 별도 **post-hoc label-adjusted diagnostic**에서는:

| Endpoint | Validation adjusted β | post-hoc BH q | 해석 |
|---|---:|---:|---|
| Layer 31 LiReF | +0.5457 | 0.00048 | 유지 |
| `L29H00030` | +0.3221 | 0.00395 | 유지 |
| `L30H00006` | +0.3941 | 0.00562 | 유지 |
| `L31N13336` | +0.5121 | 0.105 | 불확실 |
| `L29H00031` | +0.1561 | 0.095 | 불확실 |

R 문항 내부에서는 Layer 31과 `L29H00030`의 association만 post-hoc BH 기준을
통과했다. M 문항 내부에는 transformation item이 2개뿐이라 분석할 수 없다.
따라서 이 진단은 label overlap 우려를 줄이지만 제거하지 않으며 primary 또는
confirmatory evidence로 승격하지 않는다.

## 6. 현재 결론

현재 가장 유력한 자연 입력 feature 후보는:

> **explicit transformation requirement — 주어진 정보를 계산·논리·형식·인과
> 규칙으로 변환해야 하는 요구**

이다.

그러나 현재 말할 수 있는 것은 **validation-supported association**까지다.

- transformation이 R/M representation을 만든다는 인과 주장은 금지
- `L31N13336`을 transformation 또는 Reasoning neuron으로 부르는 것 금지
- 네 component가 feature를 매개한다는 주장 금지
- MMLU-Pro source/category/style 차이를 제거했다고 주장 금지
- human-annotated evidence라고 표현 금지

## 7. 다음 단계

새로운 자연 문항에서 transformation의 유무가 R/M label과 완전히 겹치지 않도록
표본을 구성하고, category/source/길이/난이도를 맞춘 독립 재현이 필요하다.
우선순위 endpoint는 frozen Layer 31 LiReF, `L29H00030`, `L30H00006`이다.
`L31N13336`과 `L29H00031`은 label-adjusted 결과가 불확실하므로 secondary로
둔다. 독립 재현 전 intervention은 진행하지 않는다.

## 8. 핵심 provenance

- design manifest SHA-256: `b9d26af3d0fd25e7bc5cc7f816f2ee9391ee3434e8a6cd1ecbad021fad818996`
- consensus annotation SHA-256: `868ff52a77cb8e31ff994ed9425ad97ad91cd61da84cf82fad8c01f82340410d`
- Layer scalar table SHA-256: `abe3884494a7c9c390b5545b2451ae1d3c3fb6710b8d1fc1b10fdd761bf9b738`
- Discovery table SHA-256: `0fea4a7b518017ffb6c5b4e0b1fc57dc505883bfdf63b1c6ff089e97efce8230`
- frozen selection SHA-256: `865c0450f51e2977cc012abad4d4125a4ff62e0bb7999c3830d6722555bdb2f2`
- Validation table SHA-256: `226e5209e440d9951875c728e8ea6a8195d5bc91f50cb9d4a000f22f4783b2cf`
- final audit status: `PASS`

