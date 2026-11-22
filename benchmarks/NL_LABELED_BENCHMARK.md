# 2차 실험: GPT6 Luna 판정 자연어 추천 벤치마크

주 지표는 NDCG@10으로 결과를 보기 전에 고정했다. 질의 100개는 카테고리 순환 표집으로 선택한 서로 다른 논문 100편의 제목·초록에서 생성한다. 한국어/영어 각 50개이고, 소재 논문은 추천 후보에서 제외한다. 질의 생성과 판정에는 D/T/M을 주지 않는다.

- 0: 무관하거나 피상적인 주제 중복만 있음.
- 1: 부분 관련 또는 유용한 배경 연구, 핵심 요구 일부 미충족.
- 2: 핵심 연구 요구에 직접 관련.

판정은 GPT6 Luna가 query+title+abstract만 보고 수행한다. 방법·점수·순위·인용·D/T/M은 숨기고 후보 순서를 무작위화한다. 각 방법의 전체 16,000편 검색 상위 20개를 합집합으로 모으고 무작위 대조군 5개를 추가한다. 모든 방식에 공통의 판정 풀을 사용하고 Top-20 판정률은 100%여야 한다. 따라서 NDCG는 판정 풀의 이상 순위를 기준으로 하고 Recall은 알려진 풀 내 양성에 대해서만 계산한다. 라벨은 LLM silver labels이며 사람 gold labels가 아니다.

비교군은 제목·초록 Qwen, D/T/M 전용 Qwen, 기존 제목·초록+D/T/M Qwen, 기존 결합+인용 PPR, 기존 결합+저자 PPR, 기존 결합+인용·저자 PPR, 현재 의도 분석·재정렬 코드이다. PPR은 학습된 GNN이 아니다. 기존 시스템의 하이퍼파라미터를 고정해 test 라벨로 튜닝하지 않는다. 현재 의도 분석 모델도 GPT6 Luna로 사용한다. OGBN 128차원에는 자연어 인코더가 없어 직접적인 자연어 내적 비교군으로 쓰지 않는다.

Qwen 모델과 양자화·512토큰 제한을 공유한다. 제목·초록은 기존 카탈로그와 같은 abstract[:1800] 전처리를 사용하고 길면 512토큰에서 잘린다. 기존 결합 카탈로그의 D/T/M도 잘렸을 수 있으므로 text vs combined를 순수 D/T/M 추가 실험으로 과장하면 안 된다. D/T/M 전용은 1차 실험의 입력 해시 검증된 캐시를 재사용한다. 새 title/abstract 임베딩은 문서별 캐시로 재개한다. 소량의 초기 행은 기존 단건 어댑터로 계산했고 나머지는 동일 Qwen의 메모리 제한 배치 추론을 사용한다.

각 질의를 단위로 bootstrap 2,000회 95% 구간을 기록한다. 다중 비교 보정은 하지 않는다. 한국어/영어별 성능도 별도 기록한다. 사전 선택한 10개 질의는 후보 순서와 배치를 바꿔 같은 모델로 재판정하여 일치율과 Cohen kappa를 기록한다. 이는 반복 신뢰도이고 사람 정확도가 아니다.

```powershell
# API키는 OPENAI_API_KEY 환경변수로 전달. 코드·결과에 저장하지 않는다.
.\venv\Scripts\python.exe -u -m benchmarks.nl_labeled --stage api
.\venv\Scripts\python.exe -u -m benchmarks.nl_labeled --stage encode
.\venv\Scripts\python.exe -u -m benchmarks.nl_labeled --stage rank
.\venv\Scripts\python.exe -u -m benchmarks.nl_labeled --stage judge
.\venv\Scripts\python.exe -u -m benchmarks.nl_labeled --stage evaluate
.\venv\Scripts\python.exe -m benchmarks.plot_nl_labeled
```

기본 결과 경로는 output/nl_benchmark_gpt6_luna_run1. API 완료 응답별 토큰 사용량은 usage.json, 합계는 usage_totals.json에 저장한다. query_rows와 judgment_chunks는 개별 성공 호출마다 저장하므로 재실행 시 완료된 호출을 반복하지 않는다. 인증 파일을 사용하는 경우 읽자마자 삭제한다.

원본 manifest, queries.json, rankings.json, qrels.json, per_query_metrics.json, summary.json, judge_consistency.json과 figures를 함께 보관한다. qrels의 reason은 판정 근거이며 사람이 검토할 수 있다. 이후 사람 검토를 거쳐 라벨을 수정한다면 새 버전의 평가셋을 만들어 모든 비교군을 함께 재평가해야 한다.

지표 정의: Recall과 MRR은 relevance >= 1을 관련으로 보고, Strong Recall은 relevance = 2만 사용한다. NDCG는 gain = 2^relevance - 1 (0, 1, 3)을 사용한다. 관련 정답이 없는 질의는 제외하지 않고 지표를 0으로 포함한다. 언어별 질의는 서로 다른 소재에서 생성했으므로 언어 인과 효과 비교로 해석하지 않는다.

이번 코퍼스 토큰 검사: 16,000편 중 결합 입력 전체가 512토큰에 들어가는 논문은 15,957편(99.73%), D/T/M이 전부 뒤로 밀리는 논문은 20편, 일부 잘리는 논문은 23편이다. 세부 결과는 input_truncation_audit.json에 저장한다.

보고서 추가 의존성: 기존 CUDA 환경에서 python -m pip install -r requirements-benchmark-reports.txt. 기존 torch 휠을 교체하지 않는다. 라벨 HTML 검토 화면은 python -m benchmarks.review_nl_labels로 생성한다. API JSON 모드에서 일부 응답이 형식 검증에 실패해 남은 요청에는 엄격한 JSON schema를 적용했다. 모든 채택 라벨은 동일한 후보 ID·점수 형식 검사를 통과한 응답이며 판정 기준은 변경하지 않았다.
