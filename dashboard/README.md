# Team Dashboard

## 실제 논문 추천

사이드바의 **실사용**은 CPU 검색을 실행합니다. 자연어 질문으로 논문 최대 3편을 찾고, 해석한 검색어·제외 조건·경로별 순위·초록·조건 근거를 표시합니다. 같은 검색의 상위 10편 후보에서 저자별로 서로 다른 논문 수를 세어 저자 최대 3명을 추천합니다. 논문 수가 같으면 가장 높은 순위의 근거 논문이 앞선 저자를 우선합니다. 저자 카드는 후보 논문 수, 근거 논문 최대 3편, OpenAlex 프로필을 표시합니다. 후속 질문의 조건은 자동 누적되지 않습니다.

첫 요청에서 D/T/M·원문 BM25 색인과 작은 순위 모델을 준비하고, 이후 요청에서 재사용합니다. `${OUTPUT_DIR}/e5_small/improved/deployment.json`이 있으면 검증된 E5 하이브리드 설정을 로드합니다. 이때 원래 질문과 정규화 검색어를 multilingual-e5-small로 CPU 인코딩하고, 의미·BM25·그래프 후보를 합친 뒤 검증에서 선택한 순위 모델/설정으로 재정렬합니다. 명시적 제외 조건은 유지하며, 주제 표현 부재는 점수와 확인 필요 표시로 반영합니다. LLM API와 GPU는 호출하지 않습니다. 배포 설정이 없는 환경에서는 기존 임베딩 없는 검색을 사용합니다.

데이터·기존 모델은 `${OUTPUT_DIR}/balanced_benchmark_20261117` 아래에, E5 문서 벡터·선택 설정·재정렬 모델은 `${OUTPUT_DIR}/e5_small`에 필요합니다. E5 모델 스냅샷은 `${OUTPUT_DIR}/hf_cache`에 있어야 하며 런타임에 다운로드하지 않습니다. 문서 ID 순서, 모델 리비전 및 선택 설정/모델 해시를 검증하며, 불일치는 오류로 표시합니다. 처음 실행은 색인과 모델 준비에 시간이 걸립니다.

`POST /api/cpu-recommend`에 `{"query":"..."}`를 보내면 작업 ID를 반환하고 `GET /api/cpu-recommend/{id}`로 대기·준비·검색·완료·오류 상태를 확인합니다. 완료 결과는 `result.results`에 논문을, `result.recommended_authors`에 저자 추천을 담습니다. CPU 작업은 한 번에 하나씩 실행하며 최대 네 요청까지만 대기/실행을 허용합니다. 질문과 결과는 메모리에만 보관하고 최근 최대 64건을 유지합니다. 경량 이미지에서 ML 의존성을 생략했다면 추천을 사용하려면 프로젝트 CPU ML 패키지 설치가 필요합니다.

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
