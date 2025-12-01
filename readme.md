# LLM-Enhanced Paper Recommendation System

이 프로젝트는 기존의 협업 필터링(Collaborative Filtering) 한계를 극복하기 위해 **LLM(Large Language Model)이 추출한 핵심 키워드**를 활용하는 **하이브리드 논문 추천 시스템**입니다.

OGBN-Arxiv 데이터셋을 기반으로 하며, 단순히 논문의 ID만 학습하는 것이 아니라 논문의 내용(Context)을 이해하여 사용자의 연구 관심사에 맞는 논문을 정교하게 추천합니다.

## 📂 프로젝트 데이터 (Download)

이 프로젝트에서 사용된 전처리 완료 데이터(1.6만 개 샘플 및 매핑 파일 등)는 아래 링크에서 다운로드할 수 있습니다.

  * **데이터 다운로드 링크:** [Google Drive - Project Data Files](https://drive.google.com/file/d/1MjrUjslFcvrxOJ9gB77AvpBvvzXvoNom/view?usp=sharing)

> **참고:** 다운로드한 파일들은 프로젝트의 `dataset/` 또는 `subdataset/` 폴더 경로에 위치시켜야 코드가 정상적으로 작동합니다.

## 🚀 주요 기능 (Key Features)

1.  **그래프 샘플링 (Forest Fire Sampling):**
      * 거대 그래프 데이터(169k 노드)를 학습 효율성을 위해 위상적 특성을 유지하며 1.6만 개(16k)로 경량화했습니다.
2.  **LLM 기반 특성 추출 (Feature Extraction):**
      * GPT-4o-mini를 활용하여 논문의 제목과 초록에서 **'Domain, Task, Method'** 관점의 핵심 키워드 5개를 추출합니다.
      * 비동기(Asyncio) 처리를 통해 대용량 데이터를 고속으로 처리합니다.
3.  **저자 데이터 매핑 (User Profiling):**
      * OpenAlex API를 사용하여 논문의 실제 저자 정보를 수집하고, 이를 추천 시스템의 '유저(User)'로 정의하여 Cold-Start 문제를 해결했습니다.
4.  **Content-Aware Matrix Factorization:**
      * `User ID` + `Item ID` + \*\*`Keyword Embeddings`\*\*를 결합한 하이브리드 모델을 구현했습니다.
      * 단순히 같이 본 논문이 아니라, **내용이 유사한 논문**을 추천하여 추천의 설득력(Explainability)을 높였습니다.

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