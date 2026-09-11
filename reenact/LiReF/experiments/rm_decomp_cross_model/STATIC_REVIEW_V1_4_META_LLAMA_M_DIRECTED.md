# Meta-Llama M-directed v1.4 static review

상태: **PASS — MODEL EXECUTION NOT YET AUTHORIZED**  
검토일: `2026-08-31`

## 확인 항목

- Meta-Llama-3-8B base만 허용한다.
- 전체 32개 Transformer block을 검색한다.
- candidate는 Discovery의 `mean_M(c)<0` 및 `Delta>0`에서 type별 최대 5개다.
- heldout score를 후보 교체에 사용하지 않는다.
- 억제는 마지막 prompt token의 frozen candidate/control에만 적용한다.
- Head는 pre-`o_proj` query-head block, Neuron은 pre-`down_proj` channel만 수정한다.
- 모델 weight·전체 hidden state를 저장하지 않는다.
- 50%·100% dose, bootstrap, permutation FDR, matched/random control을 적용한다.
- 결과 PDF는 실행 코드가 자동 수정하지 않는다.

## 판정

코드는 read-only discovery/validation hook과 제한된 suppression hook만 사용하며,
출력은 scalar·통계·manifest로 제한된다. 별도의 hash-locked authorization 이후
실행 가능하다.
