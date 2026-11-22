# D/T/M 인용 예측 벤치마크

이 실험은 D/T/M의 사실 정확도가 아니라, 숨긴 인용 관계를 예측하는 데 D/T/M이 유용한지 측정합니다. 기존 자연어 추천 및 기존 graph_benchmark 실행과 독립됩니다.

## 실행

프로젝트 루트 PowerShell에서 실행합니다. output은 존재하지 않는 새 폴더여야 합니다.

```powershell
# 데이터 정렬·분할만 확인 (GPU 불필요)
.\venv\Scripts\python.exe -m benchmarks.citation --audit-only --output output/citation_audit_run2

# D/T/M만 임베딩하여 캐시 준비 (기존 로컬 Qwen3-Embedding-4B NF4, CUDA 필요)
.\venv\Scripts\python.exe -m benchmarks.citation --prepare-only --output output/citation_prepare_run1

# 동일 분할로 OGBN / DTM / 속성 없는 그래프 모델 비교
.\venv\Scripts\python.exe -m benchmarks.citation --output output/citation_benchmark_run1

# 검증용 소규모 학습 테스트
.\venv\Scripts\python.exe -m unittest tests.test_citation_benchmark -v
```

캐시는 output/citation_dtm_embeddings에 문서 입력·모델 식별자별로 보관합니다. 중단 후 새 output 경로로 재실행하면 완료된 임베딩을 재사용합니다. 논문 텍스트를 넣는 기존 Qwen 카탈로그는 D/T/M 전용 입력과 달라 재사용하지 않습니다. 입력은 Domain / Task / Method 세 필드이며 모든 태그를 사용합니다. 512토큰 초과는 조용히 자르지 않고 실패합니다. 추출 모델이나 외부 API를 호출하지 않습니다.

## 비교군

- ogbn: 원본 OGBN 128차원 속성 → 학습 가능한 128차원 투영 → 인용 GNN.
- dtm: D/T/M 전용 Qwen 2560차원 속성 → 학습 가능한 128차원 투영 → 같은 인용 GNN.
- graph_only: 텍스트 속성 없이 학습 가능한 논문 ID 임베딩 → 같은 인용 GNN.

GNN은 학습 인용만 양방향으로 전파하는 2단계 정규화 집계와 잔차 연결을 사용합니다. 최종 source/target 투영을 분리해 방향 있는 인용을 점수화합니다. 원본 속성은 고정하고 투영·GNN·디코더를 BPR로 학습합니다. 입력 차원이 달라 입력 투영의 파라미터 수는 다릅니다. 엄밀한 D/T/M 추출 효과만의 비교에는 추후 같은 Qwen의 제목·초록 입력 비교군도 필요합니다.

## 평가 프로토콜

1. 마스터 node_idx를 키로 OGBN 속성·카테고리·인용을 정렬합니다. 기존 hetero graph의 잘못 정렬된 topic 연결을 사용하지 않습니다.
2. 자기 인용·중복을 제거하고 무방향 논문 쌍을 seed=42로 80/10/10 분할합니다. 같은 쌍의 역방향 인용도 같은 분할에 둡니다. 방향 있는 에지 수는 정확한 80/10/10과 조금 다를 수 있습니다.
3. 학습 80%만 message passing 및 양성 학습에 사용합니다. validation/test 관계는 그래프에 넣지 않습니다. 별도의 콜드스타트 노드 분할은 하지 않는 transductive 실험입니다.
4. 음성 샘플은 학습 인용과 자기 자신만 제외해 생성합니다. validation/test 라벨을 학습 중 참조하지 않으므로 숨겨진 실제 인용이 음성으로 뽑힐 가능성이 있습니다. 세 비교군은 같은 seed와 규칙을 사용합니다.
5. validation NDCG@20으로 최적 epoch를 고른 뒤 test를 평가합니다. test를 튜닝에 사용하지 않습니다. 검증 시 학습 인용을, 테스트 시 학습+검증 인용을 추천 후보에서 제외합니다. 자기 논문도 제외합니다.
6. 각 소스 논문에서 코퍼스 전체를 후보로 순위를 계산합니다. 실제 숨겨진 인용이 있는 소스만 평가합니다. Recall/NDCG/MRR@10,20을 논문별 평균하고, 카테고리 불균형을 고려해 평가 소스가 존재하는 카테고리별 평균의 macro 평균도 기록합니다. 동점은 로컬 논문 번호 순입니다.
7. DTM−OGBN 차이에 대해 동일 소스 논문을 짝지은 bootstrap 2,000회 95% 구간을 계산합니다. 이는 단일 분할·학습 seed에 대한 질의 표본 변동성이며 학습 seed 변동성을 포함하지 않습니다.

이는 무작위 인용 holdout이며 미래 인용 예측이나 공식 OGB node classification 평가가 아닙니다. 코퍼스 밖 논문은 후보 및 정답에 포함되지 않습니다. 미관측 비인용을 비관련으로 간주하는 한계가 있습니다. D/T/M이 유효하더라도 모든 관련 논문 쌍에 실제 인용이 존재하는 것은 아닙니다.

## 산출물

- manifest.json: 입력 해시, 분할 해시, 모델 식별자, 설정, 데이터 감사 결과.
- split.json: 마스터와 정렬된 원본 node ID 및 로컬 인용 분할.
- summary.json: 실험군별 최적 epoch, validation 점수, test 평균 및 category macro.
- paired_deltas.json: DTM−OGBN 점수 차이와 bootstrap 구간.
- *_per_paper.json: 소스 논문 번호·카테고리·정답 인용·추천 순위·논문별 점수. 로컬 번호는 split.json의 node_ids 및 마스터 행과 대응합니다. 논문 하나의 결과도 여기서 확인할 수 있습니다.
- *_model.pt: validation에서 선택된 모델 가중치.

실험군 개별 실행은 --arms ogbn 또는 --arms dtm으로 선택할 수 있습니다. paired_deltas는 한 번의 실행에서 두 실험군을 모두 평가했을 때 생성됩니다. 하나의 실행 결과만으로 확정적인 유효성 결론을 내리지 말고, 동일한 설정을 고정해 여러 seed 결과도 확인하세요.
