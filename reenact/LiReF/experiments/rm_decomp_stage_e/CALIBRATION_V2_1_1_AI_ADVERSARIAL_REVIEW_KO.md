# Stage E Calibration v2.1.1 추가 AI Adversarial Review

상태: **NON-HUMAN SUPPLEMENTAL REVIEW PASS — HUMAN GATE 미충족**  
검수일: 2026-08-30  
검수자 성격: 이전 AI pre-audit 결과를 본 동일 AI의 추가 검수

## 1. 판정

v2.1.1의 192개 pair와 24개 template family를 다시 검토한 결과 새로운 `FAIL` 또는 `NEEDS_REVISION` 후보는 발견하지 못했다.

```text
reviewed pairs: 192/192
additional blocking issues: 0
template families reviewed: 24/24
human gate satisfied: false
```

이 검수는 동일 AI가 수행한 보조 검수이므로 독립 human reviewer 1 또는 2의 판정으로 사용할 수 없다.

## 2. 추가 전수검사 항목

192개 pair 모두에서 다음을 재검사했다.

- Relevant/Irrelevant가 동일 context를 사용하는지
- direct/transformed mapping에 해당 label과 key가 정확히 한 번씩 있는지
- direct/transformed key 및 label이 서로 다른지
- direct record에는 정답 numeric mention 하나만 있는지
- transformed record에는 start와 delta 두 numeric mention이 있는지
- Irrelevant answer가 direct record 값과 정확히 일치하는지
- Relevant answer가 context에 literal로 노출되지 않는지
- Relevant answer가 frozen operation과 일치하는지
- 질문의 target label이 정확히 한 번 나타나는지
- 두 prompt가 one-Arabic-numeral output contract를 유지하는지

모든 항목이 192/192 통과했다.

## 3. 방향 표현 재검토

### Decrease

`removed`, `used`, `handed out`, `went missing`, `were given away`, `redeemed`, `taken`, `recycled`가 각각 감소 방향을 명확하게 나타낸다. 수정된 badge 문형은 누가 받았는지 불분명했던 `trade` 표현을 제거했으므로 subtraction 방향이 명확하다.

### Increase

`received`, `added`, `purchased`, `collected`, `an issue of ... more`, `delivered`, `placed there`, `stocked`가 증가 방향을 나타낸다. Coupon 문형의 `an issue of ... more coupons`는 다소 형식적인 표현이지만 증가 의미는 명시적이다.

### Temperature

`a rise`, `the reading increased`, `warming by`, `the recorded temperature rose`, `an increase`, `the recorded temperature increased`, `a gain`, `it rose`가 모두 addition 방향을 나타낸다. 수정된 reservoir와 capsule 문형은 record entry/account가 아니라 `the recorded temperature`를 변화의 주체로 명시한다.

## 4. Counterbalance 및 의미 대칭성

- 각 template의 answer orientation 2 × label/key role 2 × record-block order 2가 완전교차한다.
- A/B는 Relevant/Irrelevant answer로 각각 4회씩 배치된다.
- 두 label과 두 key는 direct/transformed 역할을 각각 4회씩 갖는다.
- direct-first와 transformed-first가 각각 4회다.
- 질문은 target label만 바뀌며 output instruction은 동일하다.
- 세 수정 문형은 8개 frame 전체에서 동일한 의미를 유지한다.

## 5. 남는 해석 한계

문항 표면에는 `label → key → record → value` 경로가 명확히 존재한다. 다만 mapping 문장과 해당 record 문장이 서로 인접하므로 모델이 실제 내부 계산에서 key token을 반드시 사용했다고 Calibration 결과만으로 입증할 수는 없다. 따라서 이후 결과는 다음 범위로 해석한다.

> **matched keyed-retrieval 구조를 가진 통제 조건에서의 행동 차이**

다음처럼 더 강하게 주장하지 않는다.

> 모델이 실제로 arbitrary key를 내부적으로 따라가서 검색했다.

이 제한은 두 조건의 구조적 대칭이나 Baseline Calibration의 난이도 비교를 무효화하지 않지만, Pilot에서 component/pathway의 key-binding mechanism을 해석할 때 유지해야 한다.

## 6. Human-audit 보존 확인

- reviewer 1 CSV: 192/192 `PENDING`, reviewer ID 공란
- reviewer 2 CSV: 192/192 `PENDING`, reviewer ID 공란
- reviewer 판정 열: 전부 미작성
- AI/model/LiReF 결과 열: 없음

따라서 현재 공식 gate는 계속 `INDEPENDENT_HUMAN_AUDIT_PENDING`이다. Baseline Calibration과 Stage E Pilot은 허용되지 않는다.
