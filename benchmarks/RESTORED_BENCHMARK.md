# 가져온 균형 데이터 재평가

`python -m benchmarks.restored`는 가져온 원본 샘플 manifest, GPT-5.6-Luna D/T/M, 저자 정보, 기존 0/1/2 판정을 연결한다. 추가 유료 API 호출은 없다. Qwen3-Embedding-4B NF4는 로컬 캐시에서만 로딩한다.

기본 실행 결과: output/balanced_benchmark_20261117.

1. restore: 공식 OGB 원본 ID 매핑·라벨·128차원 피처·인용으로 표본을 검증·복원한다. 균형 할당 함수의 쿼터와 실제 분포가 일치해야 한다. 카테고리별 원본 보유 수가 부족한 경우 가능한 전량을 포함하고 나머지 쿼터를 재분배한다. D/T/M·저자·텍스트·논문 ID와 판정셋 완전성을 검사한다.
2. encode: D/T/M 전용, 제목·초록, 제목·초록+D/T/M을 같은 Qwen 모델로 임베딩한다. 문서별 입력·모델 해시 캐시를 재사용한다. 입력은 기존 코드와 동일한 abstract[:1800], 최대 512토큰이다. D/T/M 실패 1편은 빈 속성 및 0벡터로 유지한다. 기존 text-embedding-3-small 1536차원 캐시는 Qwen 공간에 섞지 않는다.
3. nl: 가져온 원래 split을 보존하고 test 48개(24개 소재 논문)를 평가한다. 모든 비교군이 동일한 완전 판정 후보 풀에서 재정렬한다. 전체 코퍼스 검색 성능이 아니다. 원본 소재를 포함하는 원래 조건과, 원본 소재를 추천·정답·PPR 시드에서 제외하는 조건을 모두 기록한다. 그래프 중간 연결 노드로는 남길 수 있다. 부트스트랩은 같은 논문에서 나온 두 질의의 상관을 반영해 소재 논문 단위로 2,000회 계산한다.
4. citation: 논문 쌍 단위 인용 8:1:1 분할, train 관계만 전파에 사용한다. OGBN 속성, D/T/M 속성, 속성 없는 ID 임베딩에 같은 GNN을 사용한다. seed=42, 200 epoch, validation NDCG@20으로 모델 선택. 기존 실험과 동일한 train-only 음성 필터 규칙이다. 전체 후보에서 test 인용을 평가한다. --citation-device cpu로 로컬 Qwen 임베딩(GPU)과 독립된 자원에서 실행할 수 있으며, 세 실험군은 동일 device를 사용한다.
5. activate: 두 평가가 모두 완료된 뒤 현재 활성 master, sample, citation/author 그래프, topic 정렬 수정된 hetero graph, knowledge vocab, Qwen catalog 및 D/T/M 번들을 교체한다. 이전 활성 파일은 previous_active_dataset.zip에 보관한다. 이전 벤치마크 결과 폴더는 변경하지 않는다. active_dataset.json에 현재 데이터 출처를 기록한다.

```powershell
.\venv\Scripts\python.exe -u -m benchmarks.restored --stage restore
.\venv\Scripts\python.exe -u -m benchmarks.restored --stage encode
.\venv\Scripts\python.exe -u -m benchmarks.restored --stage nl
.\venv\Scripts\python.exe -u -m benchmarks.restored --stage citation --citation-device cpu
.\venv\Scripts\python.exe -u -m benchmarks.restored --stage activate
.\venv\Scripts\python.exe -m benchmarks.plot_restored
```

가져온 판정 데이터는 GPT-5.6-Luna가 D/T/M 메타데이터를 참고하는 생성 코드와 연결되어 있고 소재 논문은 relevance=2로 고정된다. 사람 정답이 아니며 D/T/M 비교에 유리한 판정 편향 가능성이 있다. 따라서 기존 GPT6 Luna 블라인드 판정 100개 실험과 숫자를 직접 비교해 균형화만의 효과라고 주장하면 안 된다. 이번 평가에서는 추가 라벨 생성·튜닝·모델 학습을 자연어 판정셋에 수행하지 않는다. train/validation 판정은 유지하고 test로 하이퍼파라미터를 선택하지 않는다.

테스트 질의는 가져온 생성 코드의 영어 질의이다. 이전 한국어 50개·영어 50개 블라인드 평가와 언어 구성도 다르다. 활성 교체 후 기본 추천기에서 캐시만 사용한 오프라인 평가도 확인한다.

교체 범위에는 기본 추천기의 Qwen 카탈로그, 원래 48개 테스트 질의·의도·키워드의 Qwen 캐시, Qwen 생성 재개 체크포인트, 제목초록 번들도 포함한다. 기존 1536차원 자료와 기존 활성 파일은 ZIP 백업에 보관한다.
