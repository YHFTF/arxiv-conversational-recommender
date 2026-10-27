# 2026-10-28 폴더 정리

실행 코드와 자동 테스트를 분리하고, 버전 번호 대신 용도로 폴더를 구별했습니다.
구버전으로 보이더라도 실행 의존성이 있는 모델은 유지했습니다.

## 주요 경로 변경

| 이전 | 현재 |
|---|---|
| `benchmark_v3/` | `benchmarks/` (`python -m benchmarks`) |
| `benchmark_v3/test_*.py` | `tests/test_*.py` |
| `code/local_embeddings.py` | `code/embeddings/local_embeddings.py` |
| `code/download_qwen_weights.py` | `code/embeddings/download_qwen_weights.py` |
| `code/test/run_nl_recommendation.py` | `code/retrieval/run_nl_recommendation.py` |
| `code/test/run_benchmark_v2.py` | `code/graph_benchmark/run_benchmark.py` |
| `code/test/utils_v2.py` | `code/graph_benchmark/utils.py` |
| `code/test/config.py`, `baselines/` | `code/graph_benchmark/` 내부 동일 파일 |
| `code/model/` | `code/models/` |
| `code/scripts/build_hetero_garaph_v2.py` | `code/data_analysis/build_hetero_graph.py` |
| `colab/` | `notebooks/` |
| `Benchmark_Colab_V2.ipynb` | `Graph_Benchmark.ipynb` |
| `LOCAL-SETUP.md` | `docs/setup/local.md` |
| `LOCAL-EMBEDDINGS.md` | `docs/setup/embeddings.md` |
| `SETUP.md` | `docs/setup/installation.md` |
| `presentation-assets/` | `docs/assets/recommendation/` |

기존 readme의 설치·모델·파이프라인 설명은 `docs/setup/project-guide.md`로 보존하고,
루트 readme는 실행 경로와 폴더 지도로 줄였습니다. 연구 문서는 data/design/history로 분류했습니다.

## 삭제와 유지 기준

- Python 바이트코드 캐시 83개를 제거했습니다. 원본이 이미 없는 예전 학습·추론 모듈의 캐시도 포함합니다.
- 비어 있는 `code/scripts` 디렉터리를 제거했습니다.
- V2 LightGCN은 그래프 벤치마크의 비교 모델로 사용하므로 삭제하지 않았습니다.
- BPR/GCN/GraphSAGE, 데이터 수집, D/T/M 추출·정규화, 기존 자연어 검색은 재사용 경로가 있어 유지했습니다.
- 임베딩, 모델 가중치, raw/전처리 데이터, 기존 HTML 보고서, 사용자 설정, 대시보드 데이터는 보존했습니다.
- query cache의 `benchmark_v3_query_cache` 이름은 기존 벡터 재사용을 위해 유지했습니다.
- Colab의 삭제된 train/inference 명령을 현재 그래프 벤치마크·자연어 검색 명령으로 대체했습니다.
- 그래프 생성기의 특정 개인 PC 경로를 현재 프로젝트 기준으로 바꿨습니다.

## 복구

정리 전 소스 스냅샷: `output/maintenance/cleanup_20261028_000634.zip`.
Git에서 추적 중인 파일과 미커밋 소스를 함께 포함하고, `.env`, 데이터, 가상환경,
대시보드 개인 데이터는 포함하지 않았습니다. zip 내부 경로는 이동 전 프로젝트 루트 기준입니다.
수정 중이던 대시보드·문서 변경은 이동 또는 경로 변경을 제외하고 유지했습니다.

이미 생성한 결과 파일의 provenance와 해시는 과거 실행 기록이므로 소급 수정하지 않습니다.
원격 GitHub/Colab 링크는 이 정리가 원격 저장소에 반영된 뒤 새 경로에서 동작합니다.

## 검증

- 이동 후 자동 테스트 19개 통과 (입력 규격·평가·관계별 비교 및 경로 참조 검사).
- 검색 비교, 그래프 학습 CLI, 자연어 검색 CLI, 로컬 임베딩 CLI의 도움말 로딩 통과.
- 기존 논문 임베딩·질의 캐시로 실제 네 방법 추천 비교 재실행 성공.
- 검증 보고서: `output/maintenance/cleanup_verification/report.html`.
- 대시보드 문서 검색은 하위 폴더를 재귀 탐색하도록 변경했습니다.
- 실행 중인 대시보드는 종료하지 않았으므로 새 경로 설정은 재시작 후 반영됩니다.
