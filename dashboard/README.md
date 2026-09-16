# Team Dashboard

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

선택 환경 변수는 `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `OPENAI_API_KEY`, `DASHBOARD_PORT`입니다. 팀 메모는 `dashboard-data` 볼륨에 유지됩니다. Pull은 `--ff-only`로 실행합니다.
