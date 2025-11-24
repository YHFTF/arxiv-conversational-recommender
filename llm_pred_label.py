# llm_topic_classifier.py
# -------------------------------------------------------
# Classify OGBN-Arxiv paper abstracts into arXiv CS categories (cs.xx)
# Output: JSON with label_idx, category, reasoning
# -------------------------------------------------------

import os
import json
import random
import sys
import torch
import pandas as pd
import numpy as np
from openai import OpenAI
import time


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
GLOBAL_START = time.time()

# -------------------------------------------------------
# 0. Path Configuration
# -------------------------------------------------------
FFS_LOAD_FILE = "ogbn_arxiv_16k_ffs_sample.pt"
NODE_TO_ID_MAP_PATH = "dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv"
TITLEABS_TSV_PATH = "titleabs.tsv"
LABEL_MAPPING_PATH = "dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz"

SAMPLE_COUNT = 100
OUTPUT_JSON_FILE = "topic_prediction_results.json"

LLM_MODEL = "gpt-4o-mini"
SIMULATION_MODE = False

# -------------------------------------------------------
# 1. Fix torch.load for PyTorch 2.6 security change
# -------------------------------------------------------
_real_torch_load = torch.load

def _torch_load_with_weights_only_false(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false


# -------------------------------------------------------
# 2. Load FFS sample node indices (+ labels)
# -------------------------------------------------------

def load_sample_nodes_and_labels(path):
    if not os.path.exists(path):
        print(f"⛔ ERROR: {path} not found. Run main_sampler.py first.")
        sys.exit(1)
    data = torch.load(path)

    indices = data["indices"]               # 16,000개 노드 인덱스 (리스트)
    labels_tensor = data["labels"].squeeze()  # (16000,) 텐서라고 가정

    # node_index -> label_idx 로 매핑
    node_to_label = {
        int(node_idx): int(labels_tensor[i].item())
        for i, node_idx in enumerate(indices)
    }
    return indices, node_to_label


print("📌 Loading FFS sample file...")
sampled_nodes, node_to_label = load_sample_nodes_and_labels(FFS_LOAD_FILE)

# 샘플링
target_indices = random.sample(sampled_nodes, SAMPLE_COUNT)
print(f"✅ Sampled {len(target_indices)} nodes.\n")


# -------------------------------------------------------
# 3. Load node → paper_id mapping
# -------------------------------------------------------

def load_node_to_paperid_map(path):
    try:
        df = pd.read_csv(path, header=0, dtype={"idx": np.int32, "paper id": str})
        return df["paper id"].tolist()
    except Exception as e:
        print("⛔ Mapping load error:", e)
        sys.exit(1)


node_id_list = load_node_to_paperid_map(NODE_TO_ID_MAP_PATH)
print("📌 Node-to-paperID mapping loaded.\n")


# -------------------------------------------------------
# 4. Load titles & abstracts from TSV
# -------------------------------------------------------

def extract_title_abstracts(tsv_path, node_id_list, target_idx_list):
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["paper id", "title", "abstract"],
        dtype={"paper id": str},
    ).set_index("paper id")

    title_abs_list = []
    for idx in target_idx_list:
        pid = node_id_list[idx]
        if pid not in df.index:
            title_abs_list.append(("ERROR: TITLE NOT FOUND", "ERROR: ABSTRACT NOT FOUND"))
        else:
            row = df.loc[pid]
            title = row["title"]
            abstract = row["abstract"]
            if isinstance(abstract, str):
                abstract = abstract.replace("Abstract", "", 1).strip()
            title_abs_list.append((title, abstract))
    return title_abs_list


title_abs_texts = extract_title_abstracts(TITLEABS_TSV_PATH, node_id_list, target_indices)
print("📌 Extracted titles & abstracts.\n")


# -------------------------------------------------------
# 5. Load label → arxiv category mapping
# -------------------------------------------------------

def load_label_mapping(path):
    df = pd.read_csv(path, compression="gzip")
    return df["arxiv category"].tolist()


category_list = load_label_mapping(LABEL_MAPPING_PATH)
print(f"📌 Loaded {len(category_list)} arXiv categories.\n")

# -------------------------------------------------------
# 5-1. arXiv CS 카테고리 설명 (필요시 자유롭게 수정/추가)
# -------------------------------------------------------

CATEGORY_DESCRIPTIONS = {
    # arxiv cs na – 수치해석 (cs.NA)
    # 수치 알고리즘, 과학/공학 계산, 부동소수점 오차 분석 등
    "arxiv cs na": "Numerical Analysis (cs.NA): numerical algorithms, scientific computing, floating-point error analysis.",

    # arxiv cs mm – 멀티미디어 (cs.MM)
    # 오디오/비디오, 멀티미디어 처리 및 생성, 멀티모달 콘텐츠
    "arxiv cs mm": "Multimedia (cs.MM): audio, video, and multimodal content analysis and generation.",

    # arxiv cs lo – 컴퓨터과학의 논리 (cs.LO)
    # 형식 논리, 검증, 증명 이론, 정형 기법
    "arxiv cs lo": "Logic in Computer Science (cs.LO): formal methods, verification, proof theory, logical systems.",

    # arxiv cs cy – 컴퓨터와 사회 (cs.CY)
    # ICT와 사회 영향, 프라이버시, 윤리, 정책, 디지털 사회
    "arxiv cs cy": "Computers and Society (cs.CY): social impact of computing, policy, ethics, privacy, digital society.",

    # arxiv cs cr – 암호 및 보안 (cs.CR)
    # 암호 프로토콜, 공격/방어, 프라이버시, 안전한 시스템
    "arxiv cs cr": "Cryptography and Security (cs.CR): cryptographic protocols, system security, privacy, secure computation.",

    # arxiv cs dc – 분산/병렬/클러스터 컴퓨팅 (cs.DC)
    # 분산 시스템, 클라우드, 합의, 병렬 처리
    "arxiv cs dc": "Distributed, Parallel, and Cluster Computing (cs.DC): distributed systems, cloud, consensus, parallelism.",

    # arxiv cs hc – 인간-컴퓨터 상호작용 (cs.HC)
    # 사용자 인터페이스, UX, 사용성 평가, 인터랙션 기법
    "arxiv cs hc": "Human-Computer Interaction (cs.HC): user interfaces, usability, interaction techniques, UX.",

    # arxiv cs ce – 계산 공학/금융/과학 (cs.CE)
    # 공학/금융/과학 분야의 고성능 계산 응용
    "arxiv cs ce": "Computational Engineering, Finance, and Science (cs.CE): high-performance computing in engineering, finance, science.",

    # arxiv cs ni – 네트워킹 및 인터넷 아키텍처 (cs.NI)
    # 네트워크 프로토콜, 라우팅, 트래픽 엔지니어링, SDN
    "arxiv cs ni": "Networking and Internet Architecture (cs.NI): network protocols, routing, traffic engineering, SDN.",

    # arxiv cs cc – 계산 복잡도 (cs.CC)
    # 복잡도 계층, 하한/상한, 효율성 한계
    "arxiv cs cc": "Computational Complexity (cs.CC): complexity classes, lower bounds, limits of efficient computation.",

    # arxiv cs ai – 인공지능 (cs.AI)
    # 지능형 에이전트, 계획, 지식 표현, 추론
    "arxiv cs ai": "Artificial Intelligence (cs.AI): intelligent agents, planning, reasoning, knowledge representation.",

    # arxiv cs ma – 멀티에이전트 시스템 (cs.MA)
    # 에이전트 상호작용, 협동/경쟁, 게임이론적 다중 주체
    "arxiv cs ma": "Multiagent Systems (cs.MA): interacting agents, cooperation, negotiation, game-theoretic multi-agent settings.",

    # arxiv cs gl – 일반 문헌 (cs.GL)
    # 컴퓨터과학 전반에 대한 에세이, 튜토리얼, 리뷰 등
    "arxiv cs gl": "General Literature (cs.GL): surveys, essays, tutorials, and general-interest computer science works.",

    # arxiv cs ne – 신경/진화 컴퓨팅 (cs.NE)
    # 신경망 이론, 신경 진화, 진화 알고리즘
    "arxiv cs ne": "Neural and Evolutionary Computing (cs.NE): neural network theory, neuroevolution, evolutionary algorithms.",

    # arxiv cs sc – 기호 계산 (cs.SC)
    # 심볼릭 연산, 컴퓨터 대수 시스템, 수학 표현 조작
    "arxiv cs sc": "Symbolic Computation (cs.SC): symbolic algebra, computer algebra systems, manipulation of mathematical expressions.",

    # arxiv cs ar – 하드웨어 아키텍처 (cs.AR)
    # 프로세서/가속기 설계, 마이크로아키텍처, 시스템 구조
    "arxiv cs ar": "Hardware Architecture (cs.AR): processor and accelerator design, microarchitecture, system organization.",

    # arxiv cs cv – 컴퓨터 비전 및 패턴인식 (cs.CV)
    # 이미지/비디오 이해, 객체 탐지, 분할, 인식
    "arxiv cs cv": "Computer Vision and Pattern Recognition (cs.CV): image and video understanding, detection, segmentation.",

    # arxiv cs gr – 그래픽스 (cs.GR)
    # 렌더링, 애니메이션, 기하 모델링, 시각화
    "arxiv cs gr": "Graphics (cs.GR): rendering, animation, geometric modeling, visualization.",

    # arxiv cs et – 신흥 기술 (cs.ET)
    # 새롭거나 비전통적인 컴퓨팅 기술, 실험적 시스템
    "arxiv cs et": "Emerging Technologies (cs.ET): novel or unconventional computing technologies and experimental systems.",

    # arxiv cs sy – 시스템 및 제어 (cs.SY)
    # 제어 이론, 동적 시스템, 사이버-물리 시스템
    "arxiv cs sy": "Systems and Control (cs.SY): control theory, dynamical systems, cyber-physical systems.",

    # arxiv cs cg – 계산 기하 (cs.CG)
    # 기하 알고리즘, 공간 데이터 구조, 기하적 계산
    "arxiv cs cg": "Computational Geometry (cs.CG): geometric algorithms, spatial data structures, geometric computation.",

    # arxiv cs oh – 기타 컴퓨터과학 (cs.OH)
    # 기존 카테고리에 잘 맞지 않는 기타 CS 주제
    "arxiv cs oh": "Other Computer Science (cs.OH): computer science topics not covered by other specific categories.",

    # arxiv cs pl – 프로그래밍 언어 (cs.PL)
    # 언어 설계, 타입 시스템, 컴파일러, 정적 분석
    "arxiv cs pl": "Programming Languages (cs.PL): language design, type systems, compilers, static analysis.",

    # arxiv cs se – 소프트웨어 공학 (cs.SE)
    # 요구분석, 설계, 테스트, 유지보수, 개발 프로세스
    "arxiv cs se": "Software Engineering (cs.SE): software design, testing, maintenance, development processes and tools.",

    # arxiv cs lg – 머신러닝 (cs.LG)
    # 지도/비지도/강화학습, 딥러닝, 표현 학습
    "arxiv cs lg": "Machine Learning (cs.LG): supervised, unsupervised, and reinforcement learning, deep and representation learning.",

    # arxiv cs sd – 사운드 (cs.SD)
    # 오디오 신호 처리, 음향 모델링, 음악 정보 처리
    "arxiv cs sd": "Sound (cs.SD): audio signal processing, acoustics, music information retrieval.",

    # arxiv cs si – 사회/정보 네트워크 (cs.SI)
    # 소셜 네트워크 분석, 그래프 마이닝, 온라인 관계망
    "arxiv cs si": "Social and Information Networks (cs.SI): social network analysis, graph mining, online communities.",

    # arxiv cs ro – 로보틱스 (cs.RO)
    # 로봇 제어, 지각, 내비게이션, 매니퓰레이션
    "arxiv cs ro": "Robotics (cs.RO): robot control, perception, navigation, and manipulation.",

    # arxiv cs it – 정보 이론 (cs.IT)
    # 정보량, 채널 용량, 부호 이론, 압축
    "arxiv cs it": "Information Theory (cs.IT): information measures, channel capacity, coding theory, compression.",

    # arxiv cs pf – 성능 (cs.PF)
    # 시스템/네트워크/애플리케이션 성능 분석 및 모델링
    "arxiv cs pf": "Performance (cs.PF): performance evaluation, benchmarking, and modeling of systems and networks.",

    # arxiv cs cl – 계산과 언어 (cs.CL, 자연어처리)
    # 자연어 처리, 번역, 언어 모델, 대화 시스템
    "arxiv cs cl": "Computation and Language (cs.CL): natural language processing, translation, language models, dialogue.",

    # arxiv cs ir – 정보 검색 (cs.IR)
    # 검색엔진, 랭킹, 질의 확장, 추천
    "arxiv cs ir": "Information Retrieval (cs.IR): search engines, ranking algorithms, retrieval models, recommendation.",

    # arxiv cs ms – 수학 소프트웨어 (cs.MS)
    # 수치/기호 연산을 위한 수학 소프트웨어, 라이브러리
    "arxiv cs ms": "Mathematical Software (cs.MS): software and libraries for numerical or symbolic mathematical computation.",

    # arxiv cs fl – 형식언어 및 오토마타 (cs.FL)
    # 형식 언어 이론, 오토마타, 문법, 구문 분석
    "arxiv cs fl": "Formal Languages and Automata Theory (cs.FL): formal languages, automata, grammars, parsing.",

    # arxiv cs ds – 데이터 구조 및 알고리즘 (cs.DS)
    # 기본/고급 자료구조, 알고리즘 설계/분석
    "arxiv cs ds": "Data Structures and Algorithms (cs.DS): design and analysis of data structures and algorithms.",

    # arxiv cs os – 운영체제 (cs.OS)
    # 커널, 스케줄링, 메모리/자원 관리, 가상화
    "arxiv cs os": "Operating Systems (cs.OS): kernels, scheduling, memory and resource management, virtualization.",

    # arxiv cs gt – 게임이론과 계산 (cs.GT)
    # 알고리즘적 게임이론, 메커니즘 디자인, 전략적 상호작용
    "arxiv cs gt": "Computer Science and Game Theory (cs.GT): algorithmic game theory, mechanism design, strategic interaction.",

    # arxiv cs db – 데이터베이스 (cs.DB)
    # 데이터 모델링, 질의 처리, 트랜잭션, 분산 DB
    "arxiv cs db": "Databases (cs.DB): data modeling, query processing, transactions, distributed databases.",

    # arxiv cs dl – 디지털 라이브러리 (cs.DL)
    # 디지털 아카이빙, 메타데이터, 검색/탐색 서비스
    "arxiv cs dl": "Digital Libraries (cs.DL): digital archiving, metadata, indexing, search and access services.",

    # arxiv cs dm – 이산수학 (cs.DM)
    # 그래프 이론, 조합론, 이산 구조 및 그 알고리즘
    "arxiv cs dm": "Discrete Mathematics (cs.DM): graph theory, combinatorics, discrete structures and related algorithms.",
}

# -------------------------------------------------------
# 6. Create classification prompt (Title + Abstract, TOP-3)
# -------------------------------------------------------

def create_topic_prompt(title, abstract, category_list):
    category_lines = "\n".join(
        [
            f"{i}: {cat} - {CATEGORY_DESCRIPTIONS.get(cat, 'General computer science topic related to ' + cat)}"
            for i, cat in enumerate(category_list)
        ]
    )

    system_prompt = (
        "You are an AI assistant that performs single-label classification of research papers "
        "into one of the given arXiv computer science categories. "
        "Use both the title and abstract to determine the topic. "
        "You must choose exactly one category index for each candidate and you must not invent new categories. "
        "Return ONLY valid JSON, with no extra text or formatting. "
        "For each reasoning, you must paraphrase the topic and avoid reusing exact words or technical phrases "
        "from the title or abstract whenever possible. "
        "Each 'reasoning' field must About 500 words characters."
    )

    user_prompt = f"""
Title:
\"\"\"{title}\"\"\"

Abstract:
\"\"\"{abstract}\"\"\"

You must select the TOP 3 most likely categories from the list below.
Rank them from most likely (first) to less likely (third).

Category List (index: category):
{category_lines}

Output format (MUST be valid JSON only, no markdown, no explanation):

{{
  "candidates": [
    {{
      "label_idx": <integer>,
      "category": "<string>",
      "reasoning": "<explanation About 500 words characters; do not copy phrases or words from the title or abstract>"
    }},
    {{
      "label_idx": <integer>,
      "category": "<string>",
      "reasoning": "<explanation About 500 words characters; do not copy phrases or words from the title or abstract>"
    }},
    {{
      "label_idx": <integer>,
      "category": "<string>",
      "reasoning": "<explanation About 500 words characters; do not copy phrases or words from the title or abstract>"
    }}
  ]
}}
""".strip()

    return system_prompt, user_prompt

# -------------------------------------------------------
# 7. LLM Call
# -------------------------------------------------------

def call_llm_for_topic(title, abstract, model=LLM_MODEL, simulation=False):
    system_prompt, user_prompt = create_topic_prompt(title, abstract, category_list)

    if simulation:
        # 시뮬레이션 모드일 때는 top-1만 대충 생성
        idx = random.randint(0, len(category_list) - 1)
        dummy = {
            "label_idx": idx,
            "category": category_list[idx],
            "reasoning": "시뮬레이션 결과입니다.",
        }
        return {
            "label_idx": dummy["label_idx"],
            "category": dummy["category"],
            "reasoning": dummy["reasoning"],
            "candidates": [dummy],  # top-1만 넣어둠
        }, 0, 0

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,   # 분류 태스크는 0 추천
            top_p=1.0,
        )

        content = response.choices[0].message.content
        raw = json.loads(content)  # {"candidates": [ {...}, {...}, {...} ]}

        candidates = raw.get("candidates", [])
        if not candidates or not isinstance(candidates, list):
            raise ValueError("LLM output does not contain a valid 'candidates' list")

        # 가장 가능성이 높은 1개 (첫 번째)를 메인 예측으로 사용
        primary = candidates[0]

        result = {
            "label_idx": primary["label_idx"],
            "category": primary["category"],
            "reasoning": primary["reasoning"],
            "candidates": candidates,  # top-3 전체를 그대로 보관
        }

        usage = response.usage
        return result, usage.prompt_tokens, usage.completion_tokens

    except Exception as e:
        print(f"⛔ LLM Error: {e}")
        return None, 0, 0


# -------------------------------------------------------
# 8. Run classification for all samples
# -------------------------------------------------------
print("📌 Starting LLM classification...\n")

classification_start = time.time()   #  분류 루프 시작 시간
total_llm_time = 0.0                 #  LLM 호출에 소요된 총 시간
llm_call_count = 0                   #  실제 LLM 호출 횟수

topic_results = []

for i, (title, abs_text) in enumerate(title_abs_texts):
    node_idx = target_indices[i]
    print(f"[{i+1}/{len(title_abs_texts)}] Processing node {node_idx}...")

    if "ERROR" in abs_text:
        print(" → Abstract missing, skipping.")
        continue

    llm_start = time.time()
    result, in_tok, out_tok = call_llm_for_topic(title, abs_text, simulation=SIMULATION_MODE)
    llm_end = time.time()

    # LLM 에러 방지
    if result is None:
        print("   → LLM error, skipping.")
        continue

    total_llm_time += (llm_end - llm_start)
    llm_call_count += 1

    # 🔹 primary(pred@1) + top-3 후보
    pred_label_idx = result["label_idx"]
    pred_category = result["category"]
    reasoning = result["reasoning"]
    candidates = result.get("candidates", [])  # [{"label_idx":..., "category":..., "reasoning":...}, ...]

    # 🔹 원래 레이블 찾기
    true_label_idx = node_to_label.get(int(node_idx), None)
    if true_label_idx is not None:
        true_category = category_list[true_label_idx]
    else:
        true_category = "UNKNOWN"

    # 🔹 콘솔에 예측 vs 정답 출력
    print(f"   → Pred@1: {pred_label_idx} ({pred_category})")
    if candidates:
        # top-3 요약 출력
        top3_str = ", ".join(
            [f"{c['label_idx']}({c['category']})" for c in candidates[:3]]
        )
        print(f"   → Top-3: {top3_str}")

    if true_label_idx is not None:
        print(f"   → True: {true_label_idx} ({true_category})")
        print(f"   → Match@1: {pred_label_idx == true_label_idx}")
        # Top-3 안에 정답이 있는지
        in_top3 = any(
            (c.get("label_idx") == true_label_idx) for c in candidates[:3]
        )
        print(f"   → In Top-3: {in_top3}")
    else:
        print("   → True: UNKNOWN (not found in FFS label map)")

    topic_results.append({
        "node_index": node_idx,
        "title": title,
        "abstract": abs_text,
        "pred_label_idx": pred_label_idx,
        "pred_category": pred_category,
        "reasoning": reasoning,
        "topk_candidates": candidates,  # 🔹 top-3 전체 저장
        "true_label_idx": true_label_idx,
        "true_category": true_category,
        "is_correct_top1": (
            true_label_idx is not None and pred_label_idx == true_label_idx
        ),
        "is_correct_top3": (
            true_label_idx is not None and any(
                (c.get("label_idx") == true_label_idx) for c in candidates[:3]
            )
        ),
        "input_tokens": in_tok,
        "output_tokens": out_tok
    })

print("\n✅ Classification completed.\n")
classification_end = time.time()

# -------------------------------------------------------
# 8.1. Compute Top-1 / Top-3 accuracy
# -------------------------------------------------------

valid_results = [r for r in topic_results if r["true_label_idx"] is not None]

if valid_results:
    total = len(valid_results)
    top1_correct = sum(1 for r in valid_results if r["is_correct_top1"])
    top3_correct = sum(1 for r in valid_results if r["is_correct_top3"])

    top1_acc = top1_correct / total
    top3_acc = top3_correct / total

    print("📊 Evaluation summary (excluding UNKNOWN labels):")
    print(f"   → Samples: {total}")
    print(f"   → Top-1 accuracy: {top1_correct}/{total} ({top1_acc:.3%})")
    print(f"   → Top-3 accuracy: {top3_correct}/{total} ({top3_acc:.3%})")
else:
    print("⚠️ No valid labels found for accuracy computation.")

# -------------------------------------------------------
# 8.2. Confusion analysis: which categories are confused?
# -------------------------------------------------------
from collections import Counter, defaultdict

# true_label이 있는 샘플만 사용
valid_results = [r for r in topic_results if r["true_label_idx"] is not None]

# 1) Top-3 miss 샘플 수집
top3_miss_samples = []
for r in valid_results:
    t_idx = r["true_label_idx"]
    top3_indices = [
        c["label_idx"]
        for c in r.get("topk_candidates", [])[:3]
        if "label_idx" in c
    ]

    if t_idx not in top3_indices:  # 🔹 Top-3 어디에도 정답이 없는 경우
        top3_miss_samples.append(r)

total = len(valid_results)
miss_cnt = len(top3_miss_samples)
print(f"\n❌ Top-3 miss: {miss_cnt} / {total} ({miss_cnt/total:.3%})")

# 2) Top-3 miss 샘플 몇 개 예시 출력
print("\n📌 Example Top-3 miss samples (up to 10):")
for r in top3_miss_samples[:10]:
    t_idx = r["true_label_idx"]
    true_cat = category_list[t_idx]
    p_idx = r["pred_label_idx"]
    pred_cat = category_list[p_idx]

    top3 = r.get("topk_candidates", [])[:3]
    top3_str = ", ".join(
        f"{c['label_idx']}({category_list[c['label_idx']]})"
        for c in top3
        if "label_idx" in c
    )

    print("-" * 80)
    print(f" True: {t_idx} ({true_cat})")
    print(f" Pred@1: {p_idx} ({pred_cat})")
    print(f" Top-3: {top3_str}")
    # 제목/초록을 저장해 뒀다면 아래처럼 같이 봐도 좋음
    if "pred_title" in r:
        print(f" Title: {r['pred_title']}")
    if "pred_abstract" in r:
        print(f" Abstract: {r['pred_abstract'][:200]}...")  # 너무 길면 앞부분만

# 3) Top-3 miss에 대해서 (True → Pred@1) 혼동 통계
pair_counter_top3_miss = Counter()
for r in top3_miss_samples:
    t_idx = r["true_label_idx"]
    p_idx = r["pred_label_idx"]
    pair_counter_top3_miss[(t_idx, p_idx)] += 1

print("\n📊 Top-3 miss confusion pairs (True → Pred@1):")
for (t_idx, p_idx), cnt in pair_counter_top3_miss.most_common(20):
    true_cat = category_list[t_idx]
    pred_cat = category_list[p_idx]
    print(f" ❌ {t_idx:2d} ({true_cat})  →  {p_idx:2d} ({pred_cat}) : {cnt}")

# 4) 카테고리별 Top-3 miss 비율 (어떤 정답 카테고리가 특히 어려운지)
per_class_stats = defaultdict(lambda: {"total": 0, "top3_miss": 0})

for r in valid_results:
    t_idx = r["true_label_idx"]
    per_class_stats[t_idx]["total"] += 1

for r in top3_miss_samples:
    t_idx = r["true_label_idx"]
    per_class_stats[t_idx]["top3_miss"] += 1

print("\n📊 Per-category Top-3 miss rate (by true label):")
for t_idx, stats in sorted(per_class_stats.items(), key=lambda x: x[0]):
    total_c = stats["total"]
    miss_c = stats["top3_miss"]
    if total_c == 0:
        continue
    miss_rate = miss_c / total_c
    true_cat = category_list[t_idx]
    print(f" {t_idx:2d} ({true_cat}): miss {miss_c}/{total_c} ({miss_rate:.1%})")


# -------------------------------------------------------
# 9. Save results to JSON
# -------------------------------------------------------
print(f"📌 Saving results → {OUTPUT_JSON_FILE}")

with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(topic_results, f, ensure_ascii=False, indent=4)

GLOBAL_END = time.time()  # ⬅️ 전체 실행 종료 시간

print("✅ All results saved.")
print(f"Total records: {len(topic_results)}")

# -------------------------------------------------------
# 10. Time summary
# -------------------------------------------------------
total_runtime = GLOBAL_END - GLOBAL_START
classification_runtime = classification_end - classification_start
avg_llm_time = (total_llm_time / llm_call_count) if llm_call_count > 0 else 0.0

print("\n⏱️ Time summary:")
print(f"   → Total runtime (script): {total_runtime:.2f} seconds")
print(f"   → Classification loop:    {classification_runtime:.2f} seconds")
print(f"   → LLM calls total:        {total_llm_time:.2f} seconds over {llm_call_count} calls")
print(f"   → Avg time per LLM call:  {avg_llm_time:.2f} seconds")
print(f"   → Avg time per sample:    {classification_runtime/len(title_abs_texts):.2f} seconds/sample")