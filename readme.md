# LLM-Enhanced Paper Recommendation System

이 프로젝트는 기존의 협업 필터링(Collaborative Filtering) 한계를 극복하기 위해 **LLM(Large Language Model)이 추출한 핵심 키워드**를 활용하는 **하이브리드 논문 추천 시스템**입니다.

OGBN-Arxiv 데이터셋을 기반으로 하며, 단순히 논문의 ID만 학습하는 것이 아니라 논문의 내용(Context)을 이해하여 사용자의 연구 관심사에 맞는 논문을 정교하게 추천합니다.

## 📂 프로젝트 데이터 (Download)

이 프로젝트에서 사용된 전처리 완료 데이터(1.6만 개 샘플 및 매핑 파일 등)는 아래 링크에서 다운로드할 수 있습니다.

  * **데이터 다운로드 링크:** [Google Drive - Project Data Files](https://drive.google.com/file/d/1MjrUjslFcvrxOJ9gB77AvpBvvzXvoNom/view?usp=sharing)

> **참고:** 다운로드한 파일들은 프로젝트의 `dataset/` 또는 `subdataset/` 폴더 경로에 위치시켜야 코드가 정상적으로 작동합니다.

## 💻 상세 코드 설명 (Code Descriptions)

이 프로젝트는 크게 데이터 수집, 데이터 분석(LLM 활용), 그리고 추천 모델 프로토타이핑의 3단계로 구성되어 있습니다.

### 1. Data Collection (`code/data_collection/`)
거대 그래프 데이터를 다루기 쉬운 크기로 줄이고, 추천 시스템을 위한 '유저' 개념을 생성하는 단계입니다.

* **`forest_fire.py`**
    * **기능:** 16.9만 개의 노드를 가진 OGBN-Arxiv 그래프를 Forest Fire Sampling 기법을 사용하여 1.6만 개(약 10%)로 다운샘플링합니다.
    * **특징:** 원본 그래프의 위상적 특성(Topological Structure)을 최대한 유지하며 `ogbn_arxiv_16k_ffs_sample.pt` 파일을 생성합니다.
* **`author_mapper.py`**
    * **기능:** OpenAlex API를 활용하여, 샘플링된 논문의 실제 저자(Author) 정보를 수집합니다.
    * **특징:** MAG ID 기반으로 매핑하며, 수집된 저자 데이터는 추후 Cold-Start 문제를 해결하기 위한 유저 프로필로 사용됩니다.
* **`user_gen(TF-IDF).py`**
    * **기능:** 저자 이름과 논문 텍스트(TF-IDF 상위 키워드)를 결합하여 추천 시스템의 학습 데이터인 '유저-아이템 상호작용(Interaction)'을 생성합니다.
    * **특징:** 저자가 작성한 논문의 키워드 히스토리를 분석하여 유저의 관심사 프로필을 구축합니다.
* **`loader.py`**
    * **기능:** OGBN-Arxiv 데이터셋을 안전하게 로드하기 위한 유틸리티 스크립트입니다 (PyTorch 보안 패치 포함).

### 2. Data Analysis & LLM (`code/data_analysis/`)
단순한 텍스트 매칭을 넘어, LLM을 통해 논문의 문맥을 이해하고 분류하는 핵심 단계입니다.

* **`llm_keyword_extraction.py`**
    * **기능:** GPT-4o-mini를 사용하여 각 논문의 제목과 초록에서 핵심 키워드(Feature) 5개를 추출합니다.
    * **특징:** `asyncio`를 활용한 비동기 처리로 대량의 데이터를 빠르게 처리하며, 중간 저장 및 에러 핸들링 로직이 포함되어 있습니다.
* **`llm_pred_label.py`**
    * **기능:** LLM이 논문의 제목/초록뿐만 아니라 **'인용된 이웃 논문들의 카테고리 분포(Graph Context)'**를 참고하여 논문의 카테고리를 예측합니다.
    * **특징:** 단순 분류를 넘어 그래프 정보를 프롬프트에 주입(Context Injection)했을 때의 정확도 향상을 실험합니다.
* **`cost_measure.py`**
    * **기능:** 전체 데이터를 처리하기 전, 샘플 데이터를 통해 LLM API의 예상 비용(토큰 사용량)을 산출합니다.
* **`lable_check.py` / `num_lable.py`**
    * **기능:** 데이터셋의 레이블 분포를 확인하고 매핑 정보를 검증하는 데이터 무결성 검사 도구입니다.

### 3. Prototype & Modeling (`code/prototype/`)
구축된 데이터를 바탕으로 실제 추천 알고리즘을 수행하고 평가합니다.

* **`data1(llm).py` (메인 모델)**
    * **기능:** **Content-Aware Matrix Factorization** 모델을 구현하여 논문을 추천합니다.
    * **구조:** `User Embedding` + `Item Embedding` + **`LLM Keyword Embedding`**을 결합한 하이브리드 구조입니다.
    * **실행:** 모델 학습 후, 인터랙티브 모드로 진입하여 랜덤한 유저를 선택하고 실제 추천 결과를 실시간으로 확인할 수 있습니다.
* **`data1(TF-IDF).py` (베이스라인)**
    * **기능:** LLM 대신 TF-IDF로 추출한 키워드를 사용하는 베이스라인 모델입니다. LLM 기반 모델과의 성능 비교를 위해 사용됩니다.

## 📁 프로젝트 구조 (Directory Structure)

```bash
root/
├── code/
│   ├── data_collection/
│   │   ├── loader.py              # 원본 데이터 로드
│   │   ├── forest_fire.py         # 그래프 샘플링
│   │   ├── author_mapper.py       # OpenAlex API 저자 수집
│   │   └── user_generator_v2.py   # 유저-아이템 상호작용 생성
│   ├── data_analysis/
│   │   └── llm_keyword_extraction.py # LLM 키워드 추출 (Async)
│   └── prototype/
│       └── test_with_partial_data.py # 추천 모델 학습 및 인터랙티브 테스트
├── dataset/
│   └── ogbn_arxiv/                # 원본 데이터셋
├── subdataset/
│   └── ogbn_arxiv_16k_ffs_sample.pt # 샘플링된 데이터
├── output/
│   ├── llm_extraction_results.json  # LLM 추출 결과
│   ├── final_user_interactions.csv  # 학습용 데이터
│   └── author_data_openalex.json    # 저자 정보
├── .env                           # API Key 설정
└── requirements.txt
```

## ⚙️ 설치 및 환경 설정 (Installation)

1.  **필수 라이브러리 설치:**

    ```bash
    pip install torch pandas numpy openai python-dotenv scikit-learn tqdm aiohttp
    ```

2.  **환경 변수 설정 (.env):**
    프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 입력하세요.

    ```text
    OPENAI_API_KEY=sk-proj-your-api-key-here...
    ```

## 🏃‍♂️ 실행 가이드 (Usage)

### 1\. 데이터 파이프라인 구축

데이터 수집부터 전처리까지 순서대로 실행합니다. (구글 드라이브 데이터를 다운로드했다면 생략 가능)

```bash
# 1. 그래프 샘플링
python code/data_collection/forest_fire.py

# 2. 저자 정보 매핑 (OpenAlex)
python code/data_collection/author_mapper_openalex.py

# 3. LLM 키워드 추출 (가장 중요)
python code/data_analysis/llm_keyword_extraction.py

# 4. 학습용 데이터셋 생성
python code/data_collection/user_generator_v2.py
```

### 2\. 모델 학습 및 테스트

추천 모델을 학습시키고, 대화형 인터페이스로 추천 결과를 확인합니다.

```bash
python code/prototype/test_with_partial_data.py
```

  * 학습이 완료되면 `Enter` 키를 눌러 랜덤한 저자를 선택하고 추천 논문을 확인할 수 있습니다.
  * 종료하려면 `Ctrl+C`를 입력하세요.

## 📊 모델 성능 예시

**Target User:** Anton van den Hengel (Computer Vision 연구자)

  * **실제 관심사:** Visual Question Answering, Semantic Segmentation
  * **추천 결과:**
    1.  Image Denoising using Encoder-Decoder (Score: 61.9%)
    2.  Classification of Distorted Images (Score: 9.7%)
  * **분석:** 컴퓨터 비전 및 딥러닝 아키텍처와 관련된 논문을 정확하게 추천하며, 관련 없는 분야(게임, 텍스트 등)는 낮은 점수로 필터링함.