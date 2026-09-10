# LiReF Stage B Extension (`v2_b03_extension`)

기존 `v2_b02`의 후보 20개와 control을 그대로 고정하고, 다음 한계를 별도의 held-out run에서 점검한다.

- additive/subtractive 관계 효과가 여러 새 lexical realization에서 재현되는가?
- 관계 polarity 효과가 답에 필요한 관계에서 불필요한 관계보다 큰가?
- 숫자 값 변화가 답에 필요한 숫자에서 불필요한 숫자보다 큰가?
- digit/number-word 효과가 prompt token length를 동일하게 맞춘 뒤에도 남는가?
- factual entity cue 변화가 답에 필요한 경우에 선택적으로 커지는가?

핵심 통계 단위는 개별 문장 변형이 아니라 `semantic_context_cluster`이다. 관계 lexical form과 질문 표현은 cluster 내부 반복으로 평균한다. 이 실험은 feature sensitivity를 다루며 인과성은 주장하지 않는다.

실행:

```bash
./run.sh prepare
./run.sh sanity
./run.sh pilot
./run.sh freeze_confirmatory
./run.sh confirmatory
./run.sh report
```

Pilot과 confirmatory는 context ID, lexical family, 문장 frame이 겹치지 않도록 생성한다. 모든 생성 문장은 구조 검사를 통과하지만, 독립적인 제3자 human linguistic audit는 별도 절차다.
