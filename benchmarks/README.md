# 텍스트·인용·저자 벤치마크 v3

## 기존 임베딩으로 정답 없이 추천 결과 비교

프로젝트 루트의 `compare-recommendations.bat`을 실행한 뒤 질의를 입력하면 기본 브라우저에
네 방식의 결과를 나란히 표시합니다. 기존 16,000편의 임베딩을 재사용하며 논문을 재임베딩하지
않습니다. 새 질의만 동일한 로컬 Qwen 모델로 임베딩합니다. 첫 실행에는 GPU가 필요하고
같은 질의는 모델/질의별 캐시를 재사용합니다. OpenAI API와 0/1/2 라벨은 필요하지 않습니다.

```powershell
.\venv\Scripts\python.exe -m benchmarks compare --query "그래프 신경망을 활용한 논문 추천" --top-k 10 --open
```

`--query`를 여러 번 지정하거나, 문자열 또는 `{ "query": "..." }` 배열을 담은
`--queries-json FILE`을 전달하면 한 보고서에서 질의를 전환할 수 있습니다.
`--output`은 매번 새로운 폴더를 지정해야 하며 생략하면 새 시각 기반 폴더를 만듭니다.
`--config`에는 기존 벤치 설정 파일을 사용할 수 있습니다. 이 모드에서는 split/mode/variants/ks와
bootstrap 설정을 사용하지 않고 항상 동일 임베딩으로 네 방법을 비교합니다.

결과는 `report.html`과 `comparison.json`입니다. HTML은 서버나 인터넷 없이 열 수 있으며,
초록/D/T/M 펼치기, 제목·초록 필터, 질의 전환, JSON 다운로드, 기준선 대비 순위 변화와
Top-K 중복 개수를 제공합니다. 새 질의를 계산하는 화면이 아니라 생성 당시의 결과 보고서이므로
다른 질의는 배치 파일이나 CLI로 다시 실행합니다.

**기존 임베딩은 제목·초록+D/T/M 입력으로 생성됐고 512토큰 제한 때문에 뒤쪽이 잘릴 수 있습니다.**
따라서 여기서 `임베딩만`은 D/T/M 없는 텍스트를 뜻하지 않습니다. 네 비교군 모두 이 동일한
기존 벡터를 사용하며 D/T/M 포함/미포함 효과는 비교할 수 없습니다. 정답이 없어 Recall/NDCG를
계산하지 않으며, 중복 개수·순위 변화는 관련성 품질 점수가 아닙니다. 그래프는 아래 설명한
PageRank 방식이며 GNN이 아닙니다.

기존 PT 그래프는 `node_idx`로 연결하고, 마스터의 `paper_id`와 임베딩 행 정렬을 검증합니다.
코퍼스 밖 에지는 제외하고 그 수를 보고서 JSON의 graph_stats에 기록합니다. 저자 그래프는
마스터의 authors 필드가 아닌 `author_paper_edges.pt`를 사용하므로 authors가 비어 있어도 동작합니다.
필요하면 `--master`, `--embeddings`, `--citations`, `--authors`로 경로를 변경할 수 있습니다.

기존 자연어 검색·LightGCN 코드와 독립된 실험 시스템입니다. 실제 데이터가 없어도
합성 예제로 입출력과 평가를 확인할 수 있습니다. 결과는 성능 주장에 사용할 수 없습니다.

## 실험군

| 이름 | 문서 임베딩 입력 | 그래프 |
|---|---|---|
| `text` | 제목 + 초록 | 없음 |
| `text_dtm` | 같은 제목 + 같은 초록 + D/T/M | 없음 |
| `text_citation` | `text`와 동일한 벡터 | 인용 |
| `text_author` | `text`와 동일한 벡터 | 저자–논문 |
| `text_citation_author` | `text`와 동일한 벡터 | 인용 + 저자–논문 |

첫 번째 비교는 `text` vs `text_dtm`, 두 번째 비교는 `text` vs 나머지 그래프 3종입니다.
그래프 실험에는 D/T/M, 주제 노드, 질의 의도 LLM, 기존 재정렬 보너스를 사용하지 않습니다.
문서 순서·질의·정답·평가 후보를 공유하며, 모든 실험군의 질의 벡터는 정확히 동일합니다.

## 바로 실행: GPU/API 없는 합성 예제

프로젝트 루트의 PowerShell에서 실행합니다. 출력 경로는 새 폴더여야 합니다.

```powershell
.\venv\Scripts\python.exe -m benchmarks demo --output output/bench_v3_demo
.\venv\Scripts\python.exe -m benchmarks validate --data output/bench_v3_demo/data
.\venv\Scripts\python.exe -m benchmarks run --data output/bench_v3_demo/data --embeddings output/bench_v3_demo/embeddings --config output/bench_v3_demo/config.json --output output/bench_v3_demo/results
.\venv\Scripts\python.exe -m unittest tests.test_benchmark -v
```

`demo`는 5편의 논문·2개 질의·수작업 합성 벡터를 생성합니다. `allow_demo=true`인
전용 설정으로만 실행되며, 실제 모델 성능과 무관합니다. NumPy/SciPy만 필요합니다.
현재 프로젝트 가상환경에는 두 패키지가 설치되어 있습니다.

## 실제 데이터 입력 규격

하나의 데이터 폴더에 아래 JSON 파일을 준비합니다. ID는 문자열 또는 정수이며 내부에서
문자열로 통일합니다. 배열 순서는 임베딩 메타데이터에 기록하므로 변경 후에는 재준비해야 합니다.

`papers.json`: 모든 검색 대상 논문. 메타데이터 중 임베딩에 들어가는 필드는 아래뿐입니다.

```json
[
  {"paper_id":"p0", "title":"Graph retrieval", "abstract":"A study of graph retrieval.",
   "dtm":{"domain":["Information retrieval"], "task":["Paper discovery"], "method":["Graph neural networks"]}},
  {"paper_id":"p1", "title":"Another paper", "abstract":"Another scientific abstract.",
   "dtm":{"domain":[], "task":[], "method":[]}}
]
```

`dtm`는 `text_dtm` 실험에만 필요합니다. 비어 있는 리스트는 허용하지만 누락된 객체는
오류입니다. D/T/M을 정답 질의나 평가 라벨을 참고해 생성하면 안 됩니다.

`queries.json`: 원문 질의와 고정 분할. 선택적 `exclude_paper_ids`로 자기 논문 등 제외.

```json
[{"query_id":"q0", "query":"Find research on graph-based paper retrieval", "split":"test", "exclude_paper_ids":[]}]
```

분할은 `train`, `validation`, `test`입니다. 모델/가중치/temperature/seed_k/인용 방향은
validation에서 결정한 뒤 고정하고 test를 한 번 평가하세요. 같은 숨은 원본 논문에서
생성한 질의는 같은 분할에 배치해야 합니다. 이 시스템은 질의를 자동 생성하거나 분할하지 않습니다.

`qrels.json`: relevance는 정수 0/1/2. 1 이상을 관련, 2를 강한 관련으로 계산합니다.

```json
[{"query_id":"q0", "candidate_paper_ids":["p0","p1"],
  "labels":[{"paper_id":"p0","relevance":2},{"paper_id":"p1","relevance":0}]}]
```

`citations.json`: `[인용하는 논문, 인용되는 논문]` 쌍의 배열.

```json
[["p0","p1"]]
```

`authorships.json`: `[저자 ID, 논문 ID]` 쌍의 배열. 저자 ID는 동명이인을 구별한 값이어야 합니다.

```json
[["a0","p0"],["a0","p1"]]
```

중복 에지는 하나로 합치며 자기 인용 에지는 제거합니다. 코퍼스 밖 논문 ID는 조용히
버리지 않고 오류로 처리합니다. 필요한 그래프 관계 파일이 없으면 해당 실험은 실패합니다.
텍스트 실험만 선택하면 그래프 파일은 필요하지 않습니다. 모든 관계를 검사하는
`validate` 명령에는 두 파일 모두 필요하며, 관계가 없으면 빈 배열을 넣습니다.

선택적 `graph_manifest.json`에는 그래프 생성 시점·출처·분할/컷오프 정책을 기록하세요.
실행 매니페스트에 해당 파일 해시도 저장합니다. **정적 코퍼스 검색은 코퍼스 전체의 알려진
관계를 사용할 수 있지만, 미래 추천/인용 예측 실험은 시점 이후 관계와 평가 대상 정답 에지를
제외한 그래프를 별도로 만들어야 합니다.** 이 코드는 시간 누수를 자동으로 제거하지 않습니다.

## 실제 Qwen 임베딩 준비 및 실행

기존 `output/nl_recommendation_qwen4b_nf4`의 임베딩을 자동 재사용하지 않습니다.
새 입력 규격과 입력 해시를 갖는 전용 임베딩을 준비합니다. 기존 작업은 변경하지 않습니다.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-local-embeddings.txt
.\venv\Scripts\python.exe -m benchmarks validate --data subdataset/bench_v3
.\venv\Scripts\python.exe -m benchmarks prepare --data subdataset/bench_v3 --output output/bench_v3_embeddings
.\venv\Scripts\python.exe -m benchmarks run --data subdataset/bench_v3 --embeddings output/bench_v3_embeddings --config benchmarks/configs/default.json --output output/bench_v3_test_run1
```

`prepare`만 GPU와 공식 모델 파일이 필요합니다. `run`은 저장된 벡터를 CPU에서 평가하며
모델 로딩/API 호출을 하지 않습니다. `--no-dtm`으로 text만 준비할 수 있으며 그 경우
실행 설정의 variants에서도 `text_dtm`을 빼야 합니다. 새 임베딩은 문서별로 캐시하여
중단 후 재개할 수 있습니다. 모델/프롬프트/역할/입력/토큰 제한을 캐시 키에 포함합니다.

현재 Qwen 어댑터의 제한은 512토큰입니다. D/T/M이 뒤에서 잘려 비교가 무의미해지는 것을
막기 위해, **D/T/M 포함 입력까지 512토큰 안에 들어오도록 검증하고 초과하면 중단**합니다.
이 경우 데이터 준비 단계에서 해당 초록을 줄이고 두 실험에 동일한 초록을 사용하세요.
원문 전체 초록이 반드시 필요하면 어댑터의 토큰 제한과 모델 식별자를 함께 변경해야 합니다.

## 그래프 점수와 결합 방식

이번 버전의 그래프 방법은 **학습 없는 query-seeded Personalized PageRank**입니다.
LightGCN/GNN 학습 성능으로 해석하면 안 됩니다. 추후 학습 모델 실험을 추가할 기반입니다.

1. 전체 논문의 텍스트 cosine을 temperature softmax로 확률화합니다.
2. 텍스트 상위 seed_k 논문의 점수를 정규화해 질의별 시작 분포를 만듭니다.
3. 인용은 기본 양방향 걷기, 저자는 논문→저자→논문의 2단계 걷기로 전파합니다.
   `citation_direction=outgoing`이면 인용하는 논문에서 인용되는 논문으로만 전파합니다.
4. 전체 결합은 각 논문에 존재하는 관계 유형에 동일한 비중을 줍니다. 단순 에지 개수에
   비례해 인용/저자 중 한 관계가 압도하지 않도록 합니다. 저자 걷기는 원래 논문으로의
   복귀도 포함하며, 논문 수가 많은 저자에서는 확률이 그 저자의 여러 논문으로 분산됩니다.
5. 고립 노드의 질량은 시작 분포로 돌려보냅니다. restart는 매 반복의 재시작 확률입니다.
6. 최종 점수 = `(1-graph_weight)*전체 텍스트 분포 + graph_weight*전파 결과`.

후보 제한은 그래프 전파 **이후**에 적용하므로 모든 실험이 같은 그래프와 코퍼스를 봅니다.
제외 논문은 seed/최종 결과에서 빠지지만 그래프의 중간 연결 노드로는 남습니다.
동점은 문자열 paper_id 오름차순으로 해소합니다. 그래프 벡터를 Qwen 질의 벡터와 직접
내적하지 않습니다. 수렴하지 않으면 결과와 summary에 표시하므로 검사 후 반복 한도를 늘리세요.

## 평가와 결과

- `mode=pooled`(기본): 모든 실험을 동일한 완전 라벨 후보 풀에서 재정렬합니다.
  candidate_paper_ids와 labels가 정확히 일치해야 합니다. 전체 검색 성능으로 주장하면 안 됩니다.
- `mode=full`: 전체 코퍼스에서 검색합니다. 미판정 논문은 0으로 간주하며 known-positive Recall과
  judged_fraction을 함께 기록합니다. 불완전 qrels에 따른 편향이 있으므로 충분한 판정 풀이 필요합니다.
- NDCG의 이상 순위는 **검색된 결과가 아닌 전체 qrels**에서 계산합니다.
- Recall@K, Strong Recall@K, NDCG@K, MRR@K, 판정 커버리지를 질의별 계산 후 macro 평균합니다.
  관련 정답이 없는 질의의 관련 지표는 0이며 개수를 summary에 별도 기록합니다.
- 각 실험의 text 대비 차이는 같은 질의끼리 비교하고, 질의 단위 paired bootstrap 95% 구간을
  기록합니다. 합성/소규모 질의에서 이 구간으로 유의미한 성능 주장을 하면 안 됩니다.

출력 폴더:

| 파일 | 내용 |
|---|---|
| `summary.json` | 실험별 평균, text 대비 차이/구간, 수렴 실패/무정답 질의 수 |
| `metrics.csv` | 질의 × 실험 × K 지표 |
| `rankings.json` | 상위 max(K) 결과, cosine, 최종 점수, 전파 반복 수, 순위 계산 시간 |
| `run_manifest.json` | 실제 설정, 데이터·벡터·코드 해시, 모델 정보, 실행 환경 |

시간은 벡터 준비·그래프 로딩을 제외한 개별 순위 계산 시간입니다. 모든 실험군의
그래프 가중치는 같은 값으로 비교합니다. 그래프 유형별 개별 최적화나 자동 튜닝은 하지 않습니다.
데이터 변경·정렬 변경·D/T/M 포함 여부가 다른 임베딩은 입력 해시 검사에서 거부합니다.

테스트는 5개 실험 실행, 재현성, 라벨과 순위 분리, D/T/M 격리, 그래프 질량 보존,
고립 노드, 관계별 전파 차이, 중복 에지, 잘못된 ID/캐시/불완전 후보 라벨,
graph_weight=0 기준선 일치, 평가식과 합성 벡터 보호를 검사합니다.
