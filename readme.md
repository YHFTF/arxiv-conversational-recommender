# 자연어 논문 탐색 프로젝트

목표는 자연어 추천에서 시작해 후속 대화로 연구 조건을 좁혀가는 논문 탐색입니다.
현재는 단일 질의 검색, 로컬 Qwen 임베딩, 인용·저자 관계 비교, 별도의 그래프 학습 벤치마크를 제공합니다.
대화 조건 누적 관리는 아직 구현 전입니다.

## 실행 안내

| 작업 | 실행 / 문서 |
|---|---|
| 기존 임베딩으로 네 추천 방식 비교 | `compare-recommendations.bat` |
| 대시보드 | `start-dashboard.bat` |
| Windows 환경 | [설정 안내](docs/setup/local.md) |
| Qwen NF4 임베딩 생성 | [임베딩 안내](docs/setup/embeddings.md) |
| 검색·그래프 비교 CLI / 데이터 규격 | [벤치마크 안내](benchmarks/README.md) |
| 가져온 균형 표본·0/1/2 판정 재평가 | [복원 및 벤치마크 안내](benchmarks/RESTORED_BENCHMARK.md) |
| 그래프 모델 학습·평가 | `venv\Scripts\python.exe code/graph_benchmark/run_benchmark.py --help` |
| Docker 설치 | [설치 안내](docs/setup/installation.md), [상세 가이드](docs/setup/project-guide.md) |

PowerShell 명령은 프로젝트 루트에서 실행합니다.

```powershell
# 정답 라벨 없이 현재 임베딩으로 추천 결과 비교
.\venv\Scripts\python.exe -m benchmarks compare --query "저전력 FPGA 기반 신경망 추론 가속" --open

# 자동 테스트: GPU/API 호출 없음
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

기존 임베딩은 제목·초록+D/T/M 입력이며 512토큰 제한이 있습니다. 네 비교군은 같은 벡터를 공유합니다.
D/T/M 포함 여부 비교는 별도 벡터가 필요합니다. 검색 비교의 PageRank와 그래프 학습 실험은 별개입니다.

## CPU D/T/M·그래프 검색

임베딩·온라인 LLM 없이 원문 질문과 D/T/M·인용 그래프로 검색하는 별도 CLI입니다.
기존 대시보드 검색은 변경하지 않습니다. 기본 `guarded` 모드는 한국어 분야 용어 사전·제외 조건·D/T/M 주제 필터와 작은 CPU 순위 모델을 사용합니다.
전력 수치·논문 유형 등은 제목·초록에서 초기화 시 추출한 보조 메타데이터로 확인하며, 근거가 없으면 확인 불가로 표시합니다. 일반적인 한국어 번역 기능은 아닙니다.
`--mode learned`는 조건 필터를 적용하지 않은 이전 모델, `--mode baseline`은 원래 PPR 비교군입니다.
`guarded`는 제목·초록 BM25 보조 검색을 기본으로 사용합니다. D/T/M 태그 누락을 원문의 개념 일치로 보완하고, 기존 모델 순위와 원문 BM25 순위를 RRF로 결합합니다. `--no-text-bm25`로 이전 D/T/M 전용 방식과 비교할 수 있습니다.
추천 기본 개수는 3편입니다. `output/balanced_benchmark_20261117/route_consensus/selected.json`이 있으면 validation에서 선택한 경로 중복 가점을 적용합니다. D/T/M·그래프·원문 검색의 발견 여부와 각 순위를 결과에 표시하며, `--no-route-consensus`로 가점을 끌 수 있습니다. 교집합만 강제로 반환하지는 않습니다.
모델과 선택 파일이 없다면 아래 세 단계로 생성합니다.

```powershell
.\venv\Scripts\python.exe -m benchmarks.train_dtm_ranker --stage features
.\venv\Scripts\python.exe -m benchmarks.train_dtm_ranker --stage train
.\venv\Scripts\python.exe -m benchmarks.train_dtm_ranker --stage test
.\venv\Scripts\python.exe code/retrieval/run_dtm_graph.py --query "graph neural network node classification"
.\venv\Scripts\python.exe code/retrieval/run_dtm_graph.py --query "FPGA로 신경망 추론을 가속하고 전력 소모와 처리량을 측정한 논문. 서베이는 제외해줘."
# 원문 BM25 보조 검색 끄기
.\venv\Scripts\python.exe code/retrieval/run_dtm_graph.py --no-text-bm25 --query "graph neural network node classification"
# 학습 순위 모델 없는 원래 PPR 비교군
.\venv\Scripts\python.exe code/retrieval/run_dtm_graph.py --mode baseline --query "graph neural network node classification"
```

이 명령은 복원 데이터 `output/balanced_benchmark_20261117/data`를 사용합니다.
상세 결과와 한계는 `output/balanced_benchmark_20261117/learned_dtm_graph/REPORT.md`에 기록합니다.
조건 처리 개선과 재검토 결과는 `output/balanced_benchmark_20261117/guarded_dtm_graph/REPORT.md`에 기록합니다.

## 폴더 지도

```text
benchmarks/                  검색·그래프 비교 CLI, 점수 계산, HTML 보고서
tests/                       자동 테스트
code/
  embeddings/                로컬 임베딩 및 모델 다운로드
  retrieval/                 자연어 검색 / 의도 기반 재정렬
  graph_benchmark/           그래프 모델 학습·평가, BPR/GCN/GraphSAGE 기준 모델
  models/                    LightGCN V2/V4 모델 정의
  data_collection/           샘플링·메타데이터·인용·저자 수집
  data_analysis/             D/T/M 추출·정규화, 평가셋·이종 그래프 생성
dashboard/                   대시보드 서버 및 UI
notebooks/                   Colab 실행 노트북
docs/
  setup/                     환경 설정 및 실행
  data/                      데이터 구성·재구축
  design/                    모델·검색 설계 메모
  history/                   과거 작업 기록
  assets/recommendation/     발표용 이미지·SVG와 내보내기 도구
subdataset/                  원본·전처리 데이터 (Git 제외)
output/                      임베딩·모델·보고서·캐시 (Git 제외)
```

LightGCN V2는 V4의 비교 기준으로 실제 사용하므로 유지합니다. V2/V4는 모델 종류를 구별하는 이름입니다.
`.env`, `venv`, `.dashboard-data`, 데이터·학습 결과는 보존합니다.

이전 경로와 삭제 기준은 [구조 변경 기록](docs/history/2026-10-28-folder-cleanup.md)을 참고하세요.
