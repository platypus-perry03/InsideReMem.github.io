# v1.2 `M-selective` 명칭 정정

상태: **COMPUTATION COMPLETE — LABEL/INTERPRETATION CORRECTED — NOT M-COMPONENT EVIDENCE**  
정정일: `2026-08-31`

v1.2의 계산과 raw output은 정상이며 변경하지 않는다. 다만 후보 정의
`Delta = mean_R(c) - mean_M(c) < 0`을 `M-selective`라고 부른 것은 부정확하다.

- `Delta < 0`: 해당 component의 local contribution이 관찰된 R/M gap을
  반대 방향으로 민다는 뜻
- M-directed writer: M 문항에서 contribution 자체가 M 방향이어야 하므로
  `mean_M(c) < 0`이 필요
- M-selective gap-supporting writer: 추가로 `mean_R(c)-mean_M(c) > 0`이어야 함

따라서 v1.2의 OLMo strict PASS 4개는 `negative-Delta functional candidates`로만
보존하며 M component로 보고하지 않는다. 올바른 M-directed 탐색은 별도 v1.3
설계에서 수행한다.
