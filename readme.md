# LLM-Enhanced Paper Recommendation System

LLM이 논문에서 추출한 Domain·Task·Method 지식과 논문 인용, 저자, 토픽 그래프를 결합한 하이브리드 논문 추천 시스템입니다. OGBN-Arxiv에서 샘플링한 약 1.6만 편의 논문을 대상으로 LightGCN 기반 추천, 콜드 스타트 평가, 자연어 질의를 지원합니다.

## 초기 세팅

### Windows 원클릭 설치 (권장)

준비물:

- Git for Windows
- Docker Desktop(Docker Compose v2 포함)
- 실행 중인 Docker Desktop

설치 순서:

1. 프로젝트를 설치할 빈 폴더를 만듭니다.
2. 이 저장소의 `setup.bat` 파일 하나만 해당 폴더에 넣습니다.
3. `setup.bat`을 더블클릭합니다.

스크립트가 저장소의 `main` 브랜치를 내려받고, GHCR의 대시보드 이미지를 받아 실행한 뒤 브라우저에서 <http://localhost:8080>을 엽니다.

> 기존 Git 저장소에서 실행하면 로컬 변경 사항을 덮어쓰지 않고 `git merge --ff-only`로만 업데이트합니다. fast-forward가 불가능하면 작업을 중단합니다.

### 데이터 파일 설치

학습·벤치마크·추천 실행에는 Git 저장소에 포함되지 않은 전처리 데이터가 필요합니다.

- [전처리 데이터 다운로드(Google Drive)](https://drive.google.com/file/d/1MjrUjslFcvrxOJ9gB77AvpBvvzXvoNom/view?usp=sharing)

압축을 푼 뒤 프로젝트 루트의 `subdataset/` 및 `output/`에 파일을 배치합니다. 주요 필수 파일은 다음과 같습니다.

```text
subdataset/
├── arxiv_master_final.json
└── build_hetero_graph_v2.pt
output/
├── knowledge_meta.json
└── knowledge_meta_embeddings.pt   # 자연어 추천 v2 사용 시
```

벤치마크를 실행하면 모델 가중치는 `output/benchmark/`에 생성됩니다.

### 환경 변수

Docker Compose 실행 설정은 프로젝트 루트의 `.env`에서 변경할 수 있습니다.

```dotenv
DASHBOARD_PORT=8080
OPENAI_API_KEY=
GITHUB_TOKEN=
GITHUB_REPOSITORY=YHFTF/arxiv-conversational-recommender
REQUIRE_GPU=true
OUTPUT_STORAGE_PATH=./output
ARTIFACT_STORAGE_PATH=
```

- `OPENAI_API_KEY`: LLM 전처리, 메타 임베딩 생성, 자연어 추천에 필요합니다.
- `GITHUB_TOKEN`: 비공개 저장소의 GitHub 이슈를 대시보드에서 조회할 때 필요합니다.
- `REQUIRE_GPU`: 기본값은 `true`이며, CUDA GPU가 없으면 대시보드의 학습·벤치마크 실행을 차단합니다. CPU 실행을 허용하려면 `false`로 설정합니다.
- `OUTPUT_STORAGE_PATH`: Docker와 호스트가 함께 사용할 결과 디렉터리입니다. Google Drive·NAS처럼 호스트에 마운트된 경로도 지정할 수 있습니다.
- `ARTIFACT_STORAGE_PATH`: 대시보드에서 가져오기/내보내기를 수행할 외부 저장소의 컨테이너 내부 경로입니다. 저장소가 준비되기 전에는 비워 둡니다.

Python 스크립트를 직접 실행할 때는 현재 셸에도 API 키를 설정합니다.

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
```

### 실행 확인 및 종료

```powershell
docker compose -f docker-compose.dashboard.yml ps
docker compose -f docker-compose.dashboard.yml down
```

팀 메모는 Docker의 `dashboard-data` 볼륨에 보존됩니다.

## 다른 설치 방법

### Docker로 직접 실행

```bash
docker compose -f docker-compose.dashboard.yml up -d
```

NVIDIA GPU를 학습 컨테이너에 연결하려면 NVIDIA Container Toolkit이 설치된 환경에서 GPU 오버레이를 함께 사용합니다.

```bash
docker compose -f docker-compose.dashboard.yml -f docker-compose.gpu.yml up -d
```

### Python 개발 환경

Python 3.11 환경을 권장합니다. 먼저 운영체제와 CUDA 버전에 맞는 PyTorch를 설치한 뒤 공통 의존성을 설치합니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

CUDA 12.4 Linux 환경에서는 저장소에 고정된 Docker용 의존성을 사용할 수 있습니다.

```bash
python -m pip install -r requirements-docker.txt
```

대시보드만 로컬에서 실행하려면:

```bash
python -u dashboard/server.py
```

## 주요 실행 방법

모든 명령은 프로젝트 루트에서 실행합니다.

### V4 지식 그래프 모델 학습

```bash
python code/model/train_v4_knowledge_bpr.py
```

학습된 가중치는 `output/lightgcn_v4_knowledge_bpr.pt`에 저장됩니다. 이 스크립트는 100 epoch 동안 학습하며 CUDA가 없으면 CPU를 사용합니다.

### 통합 벤치마크 v2

```bash
python code/test/run_benchmark_v2.py
```

다음 5개 모델을 동일한 분할에서 비교합니다.

- BPR-MF
- GCN + BPR
- GraphSAGE + BPR
- LightGCN (BPR)
- LightGCN + Knowledge (Ours)

벤치마크 v2는 학습 그래프에서 콜드 스타트 논문의 모든 연결을 격리한 뒤 Recall@20과 NDCG@20을 평가합니다. 특정 모델만 실행하거나 기존 가중치를 무시하려면 다음 옵션을 사용합니다.

```bash
python code/test/run_benchmark_v2.py --only "LightGCN + Knowledge (Ours)"
python code/test/run_benchmark_v2.py --force-retrain
```

결과 JSON과 모델 가중치는 `output/benchmark/`에 저장됩니다.

### 자연어 논문 추천 v2

자연어 질의에서 Domain·Task·Method를 추출하고, 메타데이터 사전과 시맨틱 매칭한 뒤 V4 지식 임베딩 공간에서 논문을 추천합니다.

```bash
python code/test/run_nl_inference_v2.py --top_k 5 --threshold 0.35
```

필요 조건:

- `OPENAI_API_KEY`
- `output/knowledge_meta_embeddings.pt`
- 벤치마크 v2가 만든 `output/benchmark/benchmark_lightgcn__knowledge_(ours)_v2.pt`

메타 임베딩 파일이 없다면 아래 명령으로 생성할 수 있습니다. OpenAI Embeddings API 비용이 발생합니다.

```bash
python code/test/word_embedding.py
```

### 제목 기반 추천

```bash
python code/test/run_inference.py --query "graph neural network" --top_k 5
```

이 명령은 `run_benchmark.py`가 만든 해당 모델 가중치를 사용합니다.

## 대시보드 기능

- 현재 브랜치, 최근 커밋, 변경 파일 확인
- GitHub 이슈 조회
- 팀 메모 작성 및 보관
- 프로젝트 문서 조회
- V4 학습과 통합 벤치마크 작업 실행 및 로그 확인
- 학습 6개 버전과 벤치마크 2개 버전 선택 실행
- 새로 추가된 `code/**/*.py` 자동 탐색 및 컨테이너 실행
- 브랜치별 커밋 확인, 작업 브랜치 전환 및 Pull
- 파일시스템 기반 외부 Output 저장소 가져오기/내보내기

현재 자연어 추천 항목은 대화형 CLI만 제공하며, 대시보드의 실사용 추천 UI는 구현 예정입니다.

## 프로젝트 구조

```text
.
├── code/
│   ├── data_collection/     # 샘플링, 저자·인용 데이터 수집
│   ├── data_analysis/       # LLM 지식 추출과 비용 산정
│   ├── model/               # LightGCN 모델, 학습 및 기존 추론
│   ├── scripts/             # 이종 그래프 생성과 데이터 검사
│   └── test/                # 벤치마크, 자연어 추천, 베이스라인
├── dashboard/               # 팀 대시보드 서버와 정적 UI
├── docs/                    # 모델 및 실험 문서
├── colab/                   # Colab 벤치마크/워크벤치 노트북
├── subdataset/              # 전처리 그래프와 논문 메타데이터(별도 다운로드)
├── output/                  # 지식 사전, 임베딩, 모델 가중치와 결과
├── docker-compose.dashboard.yml
├── docker-compose.gpu.yml
└── setup.bat
```

## 모델 개요

V4 모델은 논문 인용 그래프뿐 아니라 다음 정보를 하나의 추천 공간에 반영합니다.

- 논문–논문 인용 관계
- 저자–논문 작성 관계
- 논문–토픽 관계
- LLM이 추출한 Domain·Task·Method 지식 임베딩

벤치마크 v2는 일반 테스트와 별도로 전체 논문의 10%를 콜드 스타트 노드로 선정하고, 관련 에지를 학습 그래프에서 제거해 신규 논문 추천 성능을 측정합니다.

## 데이터 파이프라인

전처리 데이터를 새로 만들 때의 주요 단계는 다음과 같습니다.

1. `code/data_collection/forest_fire.py`: OGBN-Arxiv 그래프를 약 1.6만 노드로 샘플링
2. `code/data_collection/author_mapper.py`: OpenAlex에서 저자 정보 수집
3. `code/data_analysis/llm_keyword_extraction.py`: 제목과 초록에서 LLM 지식 추출
4. `code/data_collection/author_paper_edges.py`, `paper_paper_edge.py`: 그래프 에지 생성
5. `code/scripts/build_hetero_garaph_v2.py`: V4용 이종 그래프와 지식 사전 생성

전체 파이프라인은 OGB 원본 데이터, OpenAlex/OpenAI 네트워크 호출, API 키와 비용이 필요합니다. 모델 실험만 하려면 위의 전처리 데이터를 내려받아 사용하는 편이 간단합니다.

## 문제 해결

- 대시보드가 열리지 않으면 `docker compose -f docker-compose.dashboard.yml logs dashboard`로 로그를 확인합니다.
- 학습 버튼에서 CUDA 오류가 나면 GPU 오버레이와 NVIDIA Container Toolkit 설정을 확인하거나 `.env`에서 `REQUIRE_GPU=false`로 CPU 실행을 허용합니다.
- `FileNotFoundError`가 발생하면 `subdataset/`과 `output/`의 필수 파일 경로를 확인합니다.
- 자연어 추천에서 가중치를 찾지 못하면 먼저 `run_benchmark_v2.py`를 실행합니다.
- `setup.bat`의 이미지 pull이 실패하면 GHCR 패키지가 공개 상태인지, 또는 `ghcr.io` 로그인이 필요한지 확인합니다.
