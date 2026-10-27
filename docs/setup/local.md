# Windows 로컬 GPU 환경

전역 CUDA Toolkit 12.4와 기본 Python 3.11의 PyTorch 2.6.0+cu124를 공유합니다.
프로젝트 의존성은 `venv`에 설치했으며, 다른 프로젝트의 NumPy/pandas 버전은 바꾸지 않습니다.

## 실행

프로젝트 폴더에서 `start-dashboard.bat`을 실행하고 http://localhost:18080 에 접속합니다.
이미 대시보드가 실행 중이면 추가 실행할 필요가 없습니다. 직접 실행한 창에서는 Ctrl+C로 종료합니다.
이번 설정에서 백그라운드로 실행한 대시보드는 프로젝트 폴더의 PowerShell에서 다음 명령으로 종료합니다.

```powershell
taskkill /PID (Get-Content .dashboard-data/launcher.pid) /T /F
```

로그는 `.dashboard-data/server.log`, `.dashboard-data/server-error.log`에 있습니다.
이 로컬 실행에는 Docker가 필요하지 않습니다.

PowerShell에서 CLI 작업을 할 때는 활성화 없이 다음 Python을 사용해도 됩니다.

```powershell
.\venv\Scripts\python.exe code/graph_benchmark/run_benchmark.py --help
.\venv\Scripts\python.exe code/retrieval/run_nl_recommendation.py --help
```

`start-dashboard.bat`은 `.env`를 읽고 하위 작업도 같은 가상환경의 Python으로 실행합니다.
API 키는 `.env`의 `OPENAI_API_KEY`에 직접 입력하세요. 이 파일은 Git에서 제외됩니다.
`GITHUB_TOKEN`은 비공개 GitHub 이슈를 조회할 때만 필요합니다.

## 의존성 복구

```powershell
python -m venv --system-site-packages venv
.\venv\Scripts\python.exe -m pip install -r requirements-docker.txt
.\venv\Scripts\python.exe -m pip check
```

기본 `requirements.txt`는 CPU용 PyTorch를 지정하므로 GPU 환경에서는 사용하지 않습니다.
`--system-site-packages` 환경은 전역 Python/PyTorch를 사용하므로 해당 설치를 삭제하면 복구가 필요합니다.

## 별도 입력

학습과 추천에는 README에 안내된 전처리 데이터가 필요합니다.
환경 설치만으로 데이터와 API 키가 생성되지는 않습니다.

- `subdataset/arxiv_master_final.json`
- `subdataset/build_hetero_graph_v2.pt`
- `output/knowledge_meta.json`
- `output/knowledge_meta_embeddings.pt` (자연어 추천 v2)

GPU 확인 스크립트: `..\verify_cuda.py`

## 설치 검증 (2026-10-27)

- `pip check`: 의존성 충돌 없음
- 실제 전처리 그래프와 메타데이터 로딩 성공
- LightGCN V4 GPU 순전파·역전파 성공: 출력 크기 `(38410, 128)`
- 벤치마크 CLI 로딩과 도움말 출력 성공
- 대시보드와 health/git/tasks/scripts/storage API: HTTP 200
- OpenAI API 키는 미설정이며, 유료 API 호출과 전체 학습은 실행하지 않았습니다.

