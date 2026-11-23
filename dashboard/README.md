# Team Dashboard

## 실제 논문 추천

사이드바의 **실사용**은 CPU 검색으로 최대 10편을 찾습니다. Ollama가 후보의 제목·초록을 5편씩 검토하며, 질문의 주제와 필수 조건에 대한 근거가 확인된 논문을 기존 검색 순서로 최대 3편 표시합니다. 검토한 편수와 근거가 확인된 편수를 함께 표시합니다. 확인된 후보에 가장 많이 등장하는 저자를 최대 3명 추천하고, 논문 수가 같으면 가장 높은 순위의 근거 논문이 앞선 저자를 우선합니다.

같은 탭에서 후속 질문으로 첫 질문의 근거가 확인된 후보를 좁힐 수 있습니다. 처음 화면에 표시된 3편보다 확인된 후보가 많으면 그 논문들도 후속 검토에 포함됩니다. Ollama가 주제와 적용된 조건을 참고해 후속 질문을 해석하고, 제목·초록에서 근거 구절을 확인합니다. 예를 들어 “실험에서 쓴 전력이 5와트를 넘지 않는 연구 위주로 남겨줄 수 있어?”라고 요청할 수 있습니다.

후속 조건을 추가하거나 해제할 때 남은 논문으로 저자를 다시 추천합니다. 조건 옆의 `×`로 해제할 수 있습니다. 첫 질문에 포함된 조건이나 주제를 바꾸려면 **새 탐색**을 시작합니다. 서버를 다시 시작하거나 최근 64개 작업에서 벗어나면 이전 탐색 기록은 만료됩니다.

논문 추천을 사용하려면 Ollama를 실행하고 `qwen3.5:9b` 모델을 미리 받아 둡니다. `OLLAMA_RECOMMEND_MODEL`로 모델을 지정할 수 있습니다. 기본 Ollama 주소는 `http://127.0.0.1:11434`입니다. Docker Compose는 `http://host.docker.internal:11434`를 기본으로 사용하며, Ollama에 접속할 수 있는 주소를 `OLLAMA_RECOMMEND_URL`로 지정할 수 있습니다.

첫 요청에서 D/T/M·원문 BM25 색인과 작은 순위 모델을 준비하고, 이후 요청에서 재사용합니다. `${OUTPUT_DIR}/e5_small/improved/deployment.json`이 있으면 검증된 E5 하이브리드 설정을 로드합니다. 이때 원래 질문과 정규화 검색어를 multilingual-e5-small로 CPU 인코딩하고, 의미·BM25·그래프 후보를 합친 뒤 검증에서 선택한 순위 모델과 설정으로 재정렬합니다. 그다음 선정한 후보를 Ollama에 보내 검토합니다. 배포 설정이 없는 환경에서는 기존 임베딩 없는 검색으로 후보를 찾습니다.

데이터·기존 모델은 `${OUTPUT_DIR}/balanced_benchmark_20261117` 아래에, E5 문서 벡터·선택 설정·재정렬 모델은 `${OUTPUT_DIR}/e5_small`에 필요합니다. E5 모델 스냅샷은 `${OUTPUT_DIR}/hf_cache`에 있어야 하며 런타임에 다운로드하지 않습니다. 문서 ID 순서, 모델 리비전 및 선택 설정/모델 해시를 검증하며, 불일치는 오류로 표시합니다. 처음 실행은 색인과 모델 준비에 시간이 걸립니다.

`POST /api/cpu-recommend`에 `{"query":"..."}`를 보내면 작업 ID를 반환하고 `GET /api/cpu-recommend/{id}`로 대기·준비·검색·완료·오류 상태를 확인합니다. 완료 결과는 `result.results`에 논문을, `result.recommended_authors`에 저자 추천을 담습니다. CPU 작업은 한 번에 하나씩 실행하며 최대 네 요청까지만 대기/실행을 허용합니다. 질문과 결과는 메모리에만 보관하고 최근 최대 64건을 유지합니다. 경량 이미지에서 ML 의존성을 생략했다면 추천을 사용하려면 프로젝트 CPU ML 패키지 설치가 필요합니다.

이전 탐색을 이어가려면 완료된 작업 ID를 `previous_id`로 보내면서 `query`로 후속 조건을 추가하거나 `remove_filter`로 적용된 조건을 해제합니다. `result.conversation`에는 주제, 첫 검색 후보 수, 남은 후보 수, 적용 중인 후속 조건이 들어갑니다.

개발 실행: 프로젝트 가상환경으로 `python dashboard/server.py`. 기존 서버가 실행 중이면 코드 적용을 위해 재시작합니다.

Git 기록, 브랜치, GitHub 이슈, 팀 메모, 학습 및 벤치마크 실행을 관리하는 대시보드입니다.

```bash
docker compose -f docker-compose.dashboard.yml up --build
```

브라우저에서 `http://localhost:8080`을 엽니다. 포트는 호스트의 모든 IPv4 인터페이스(`0.0.0.0:8080`)에 공개되므로, 같은 로컬 네트워크에서는 `http://<호스트-IP>:8080`으로 접속할 수 있습니다. 기본 빌드는 모델 학습과 벤치마크에 필요한 CUDA용 ML 패키지를 모두 설치합니다. 대시보드만 빠르게 실행하는 경량 이미지가 필요하면 `INSTALL_ML_DEPS=false`로 빌드합니다.

기본 이미지는 CUDA 12.4용 PyTorch를 포함하며 GPU가 없으면 CPU로 폴백합니다. NVIDIA Container Toolkit이 설치된 GPU 호스트에서는 GPU 오버레이를 함께 실행합니다.

```bash
docker compose -f docker-compose.dashboard.yml -f docker-compose.gpu.yml up --build
```

Windows PowerShell:

```powershell
$env:INSTALL_ML_DEPS='false'
docker compose -f docker-compose.dashboard.yml up --build
```

선택 환경 변수는 `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `OPENAI_API_KEY`, `DASHBOARD_PORT`, `OUTPUT_STORAGE_PATH`, `ARTIFACT_STORAGE_PATH`입니다. 팀 메모는 `dashboard-data` 볼륨에 유지됩니다. Pull은 `--ff-only`로 실행합니다.

`OUTPUT_STORAGE_PATH`에는 호스트, Google Drive 또는 NAS에 마운트된 공유 결과 폴더를 지정할 수 있습니다. 대시보드는 `code/**/*.py`를 새로고침할 때마다 다시 탐색하므로 새 스크립트를 위해 이미지를 재빌드할 필요가 없습니다. 브랜치 전환은 미커밋 변경이 없는 경우에만 허용됩니다.
