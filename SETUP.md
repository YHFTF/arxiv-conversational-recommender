# Windows 원클릭 설치

## 준비 사항

- Git for Windows
- Docker Desktop (Docker Compose v2 포함)
- 실행 중인 Docker Desktop

## 설치

1. 프로젝트를 설치할 빈 폴더를 만듭니다.
2. `setup.bat` 하나만 그 폴더에 넣습니다.
3. `setup.bat`을 더블클릭합니다.

배치파일은 다음 작업을 자동으로 수행합니다.

1. 현재 폴더에서 Git 저장소를 초기화합니다.
2. `origin`을 이 프로젝트의 GitHub 저장소로 설정합니다.
3. 원격 `main` 브랜치를 가져와 로컬 `main`에 연결합니다.
4. GHCR에서 공통 Docker 이미지를 pull합니다.
5. 대시보드 컨테이너를 실행하고 `http://localhost:8080`을 엽니다.

환경 변수나 포트를 바꾸려면 프로젝트 루트에 `.env`를 만들 수 있습니다.

```dotenv
DASHBOARD_PORT=8080
OPENAI_API_KEY=
GITHUB_TOKEN=
OUTPUT_STORAGE_PATH=./output
ARTIFACT_STORAGE_PATH=
```

`OUTPUT_STORAGE_PATH`는 로컬 폴더뿐 아니라 Google Drive 또는 NAS처럼 호스트에 마운트된 공유 경로로 지정할 수 있습니다. 외부 저장소가 준비되면 해당 경로를 컨테이너에 마운트하고 `ARTIFACT_STORAGE_PATH`에 컨테이너 내부 경로를 설정해 대시보드에서 가져오기/내보내기를 사용할 수 있습니다.

중지할 때는 프로젝트 폴더에서 다음 명령을 실행합니다.

```powershell
docker compose -f docker-compose.dashboard.yml down
```

## 이미지 최초 배포

`main`에 관련 파일이 push되면 GitHub Actions가 이미지를
`ghcr.io/yhftf/arxiv-conversational-recommender-dashboard:main`으로 배포합니다.
첫 실행 전 GitHub 저장소의 **Actions** 탭에서 `Publish dashboard image`가 성공했는지
확인하고, 패키지 접근 권한을 **Public**으로 설정해야 로그인 없이 pull할 수 있습니다.
