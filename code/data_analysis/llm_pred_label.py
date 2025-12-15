# llm_topic_classifier.py
# -------------------------------------------------------
# Classify OGBN-Arxiv paper abstracts into arXiv CS categories (cs.xx)
# Output: JSON with label_idx, category, reasoning
# -------------------------------------------------------

import os
import json
import random
import sys
import time
import asyncio
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
import torch
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
import networkx as nx
from collections import Counter, defaultdict
from llm_costing import LLMCostTracker

# 비동기 OpenAI 클라이언트
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
GLOBAL_START = time.time()

# -------------------------------------------------------
# 0. Path Configuration
# -------------------------------------------------------

# 현재 파일 위치를 기준으로 프로젝트 루트를 계산
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 데이터 경로 (다른 스크립트와 동일한 구조 사용)
FFS_LOAD_FILE = os.path.join(project_root, "subdataset", "ogbn_arxiv_16k_ffs_sample.pt")
NODE_TO_ID_MAP_PATH = os.path.join(
    project_root, "dataset", "ogbn_arxiv", "mapping", "nodeidx2paperid.csv"
)
TITLEABS_TSV_PATH = os.path.join(project_root, "subdataset", "titleabs.tsv")
LABEL_MAPPING_PATH = os.path.join(
    project_root,
    "dataset",
    "ogbn_arxiv",
    "mapping",
    "labelidx2arxivcategeory.csv.gz",
)

SAMPLE_COUNT = 100
PROJECT_SIZE = 16000  # 전체 노드 수 기준 예상 비용 산정용

# LLM 비동기 호출 동시 처리 개수
MAX_CONCURRENT_REQUESTS = 10

# 결과 저장 디렉토리 및 파일 (project_root/output/)
OUTPUT_DIR = os.path.join(project_root, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 코스트 측정 모드 플래그 (run_costing.py 에서만 설정)
LLM_COSTING_MODE = os.getenv("LLM_COSTING_MODE")

# 코스트 측정 모드일 때는 타임스탬프/별도 접두어를 사용해서 기존 결과를 덮어쓰지 않음
if LLM_COSTING_MODE:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_JSON_FILE = os.path.join(
        OUTPUT_DIR, f"topic_prediction_results_cost_{ts}.json"
    )
else:
    OUTPUT_JSON_FILE = os.path.join(OUTPUT_DIR, "topic_prediction_results.json")

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

    indices = data["indices"]                 # 16,000개 노드 인덱스 (리스트)
    labels_tensor = data["labels"].squeeze()  # (16000,) 텐서라고 가정

    # node_index -> label_idx 매핑
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
# 2-1. Build citation graph from local ogbn-arxiv edge list
# -------------------------------------------------------

def build_citation_graph():
    """
    로컬에 이미 내려받아 둔 ogbn-arxiv edge 리스트
    (dataset/ogbn_arxiv/raw/edge.csv.gz)를 사용해서
    인용 그래프를 NetworkX DiGraph로 만들어준다.

    노드: 논문 인덱스 (0 ~ num_nodes-1)
    엣지: source(인용하는 논문) -> target(인용 당하는 논문)
    """
    edge_path = os.path.join(project_root, "dataset", "ogbn_arxiv", "raw", "edge.csv.gz")
    if not os.path.exists(edge_path):
        print(f"⛔ ERROR: edge file not found at {edge_path}")
        sys.exit(1)

    print(f"📌 Loading local ogbn-arxiv edges from {edge_path} ...")

    # edge.csv.gz 형식: source,target 두 컬럼 (OGB 기본 포맷)
    edges_df = pd.read_csv(edge_path, compression="gzip", header=None)
    if edges_df.shape[1] < 2:
        print("⛔ ERROR: edge.csv.gz does not have at least 2 columns (source, target).")
        sys.exit(1)

    src = edges_df.iloc[:, 0].astype(int).tolist()
    dst = edges_df.iloc[:, 1].astype(int).tolist()

    G = nx.DiGraph()
    # 노드 수는 최대 인덱스 + 1 로 추정
    max_node_idx = max(max(src), max(dst))
    G.add_nodes_from(range(max_node_idx + 1))
    G.add_edges_from(zip(src, dst))

    print(f"✅ Citation graph built from local file. #nodes={G.number_of_nodes()}, #edges={G.number_of_edges()}")
    return G


citation_graph = build_citation_graph()
print("📌 Citation graph ready.\n")


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
    df = (
        pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=["paper id", "title", "abstract"],
            dtype={"paper id": str},
        )
        .set_index("paper id")
    )

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


title_abs_texts = extract_title_abstracts(
    TITLEABS_TSV_PATH, node_id_list, target_indices
)
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
# 5-1. arXiv CS 카테고리 설명 (간단 설명)
#     (키 문자열은 실제 category_list 안에 들어오는 값에 맞게 조정 필요)
# -------------------------------------------------------

CATEGORY_DESCRIPTIONS = {
    "arxiv cs na": "Numerical Analysis (cs.NA): numerical algorithms, scientific computing, floating-point error analysis.",
    "arxiv cs mm": "Multimedia (cs.MM): audio, video, and multimodal content analysis and generation.",
    "arxiv cs lo": "Logic in Computer Science (cs.LO): formal methods, verification, proof theory, logical systems.",
    "arxiv cs cy": "Computers and Society (cs.CY): social impact of computing, policy, ethics, privacy, digital society.",
    "arxiv cs cr": "Cryptography and Security (cs.CR): cryptographic protocols, system security, privacy, secure computation.",
    "arxiv cs dc": "Distributed, Parallel, and Cluster Computing (cs.DC): distributed systems, cloud, consensus, parallelism.",
    "arxiv cs hc": "Human-Computer Interaction (cs.HC): user interfaces, usability, interaction techniques, UX.",
    "arxiv cs ce": "Computational Engineering, Finance, and Science (cs.CE): high-performance computing in engineering, finance, science.",
    "arxiv cs ni": "Networking and Internet Architecture (cs.NI): network protocols, routing, traffic engineering, SDN.",
    "arxiv cs cc": "Computational Complexity (cs.CC): complexity classes, lower bounds, limits of efficient computation.",
    "arxiv cs ai": "Artificial Intelligence (cs.AI): intelligent agents, planning, reasoning, knowledge representation.",
    "arxiv cs ma": "Multiagent Systems (cs.MA): interacting agents, cooperation, negotiation, game-theoretic multi-agent settings.",
    "arxiv cs gl": "General Literature (cs.GL): surveys, essays, tutorials, and general-interest computer science works.",
    "arxiv cs ne": "Neural and Evolutionary Computing (cs.NE): neural network theory, neuroevolution, evolutionary algorithms.",
    "arxiv cs sc": "Symbolic Computation (cs.SC): symbolic algebra, computer algebra systems, manipulation of mathematical expressions.",
    "arxiv cs ar": "Hardware Architecture (cs.AR): processor and accelerator design, microarchitecture, system organization.",
    "arxiv cs cv": "Computer Vision and Pattern Recognition (cs.CV): image and video understanding, detection, segmentation.",
    "arxiv cs gr": "Graphics (cs.GR): rendering, animation, geometric modeling, visualization.",
    "arxiv cs et": "Emerging Technologies (cs.ET): novel or unconventional computing technologies and experimental systems.",
    "arxiv cs sy": "Systems and Control (cs.SY): control theory, dynamical systems, cyber-physical systems.",
    "arxiv cs cg": "Computational Geometry (cs.CG): geometric algorithms, spatial data structures, geometric computation.",
    "arxiv cs oh": "Other Computer Science (cs.OH): computer science topics not covered by other specific categories.",
    "arxiv cs pl": "Programming Languages (cs.PL): language design, type systems, compilers, static analysis.",
    "arxiv cs se": "Software Engineering (cs.SE): software design, testing, maintenance, development processes and tools.",
    "arxiv cs lg": "Machine Learning (cs.LG): supervised, unsupervised, and reinforcement learning, deep and representation learning.",
    "arxiv cs sd": "Sound (cs.SD): audio signal processing, acoustics, music information retrieval.",
    "arxiv cs si": "Social and Information Networks (cs.SI): social network analysis, graph mining, online communities.",
    "arxiv cs ro": "Robotics (cs.RO): robot control, perception, navigation, and manipulation.",
    "arxiv cs it": "Information Theory (cs.IT): information measures, channel capacity, coding theory, compression.",
    "arxiv cs pf": "Performance (cs.PF): performance evaluation, benchmarking, and modeling of systems and networks.",
    "arxiv cs cl": "Computation and Language (cs.CL): natural language processing, translation, language models, dialogue.",
    "arxiv cs ir": "Information Retrieval (cs.IR): search engines, ranking algorithms, retrieval models, recommendation.",
    "arxiv cs ms": "Mathematical Software (cs.MS): software and libraries for numerical or symbolic mathematical computation.",
    "arxiv cs fl": "Formal Languages and Automata Theory (cs.FL): formal languages, automata, grammars, parsing.",
    "arxiv cs ds": "Data Structures and Algorithms (cs.DS): design and analysis of data structures and algorithms.",
    "arxiv cs os": "Operating Systems (cs.OS): kernels, scheduling, memory and resource management, virtualization.",
    "arxiv cs gt": "Computer Science and Game Theory (cs.GT): algorithmic game theory, mechanism design, strategic interaction.",
    "arxiv cs db": "Databases (cs.DB): data modeling, query processing, transactions, distributed databases.",
    "arxiv cs dl": "Digital Libraries (cs.DL): digital archiving, metadata, indexing, search and access services.",
    "arxiv cs dm": "Discrete Mathematics (cs.DM): graph theory, combinatorics, discrete structures and related algorithms.",
}


# -------------------------------------------------------
# 6. Citation-based trend context builder
# -------------------------------------------------------

def build_citation_trend_context(
    node_idx,
    citation_graph,
    node_to_label,
    category_list,
    max_neighbors=50,
):
    """
    주어진 노드(node_idx)에 대해 인용 그래프 이웃의 레이블 분포를 요약해서
    LLM 프롬프트에 넣기 좋은 자연어 텍스트로 만든다.
    - node_to_label: node_idx -> label_idx (FFS 샘플에서 로드한 것)
    """
    if citation_graph is None:
        return "No citation graph information is available."

    if node_idx not in citation_graph:
        return "This paper node does not exist in the citation graph."

    # 1-hop 이웃: 인용하는 논문(후행), 인용 당하는 논문(선행) 모두 사용
    out_neighbors = list(citation_graph.successors(node_idx))   # node_idx -> neighbor
    in_neighbors = list(citation_graph.predecessors(node_idx))  # neighbor -> node_idx

    neighbors = out_neighbors + in_neighbors
    if not neighbors:
        return "This paper appears as an isolated node with no direct citation neighbors."

    # 너무 많으면 앞에서 max_neighbors개만 사용
    if len(neighbors) > max_neighbors:
        neighbors = neighbors[:max_neighbors]

    # 이웃들 중 label을 알고 있는 것만 사용 (FFS 샘플 기준)
    neighbor_labels = []
    for n in neighbors:
        lbl = node_to_label.get(int(n))
        if lbl is not None:
            neighbor_labels.append(lbl)

    if not neighbor_labels:
        return (
            "This paper has citation neighbors, but their labels are not available "
            "in the sampled set. Treat it as graph-connected but unlabeled neighbors."
        )

    # 레이블 → 카테고리 이름으로 카운트
    cnt = Counter(neighbor_labels)
    total = sum(cnt.values())

    # 비율이 큰 카테고리 상위 몇 개만 요약
    top_k = 5
    most_common = cnt.most_common(top_k)

    parts = []
    for label_idx, c in most_common:
        category_name = category_list[label_idx]
        ratio = c / total
        parts.append(f"{category_name} (count={c}, ratio={ratio:.2f})")

    summary_line = "; ".join(parts)

    context_text = (
        "Graph-based trend evidence:\n"
        f"- Total labeled citation neighbors considered: {total}\n"
        f"- Dominant neighbor categories: {summary_line}\n"
        "Use this as additional evidence about which research community and trend this paper belongs to."
    )
    return context_text


# -------------------------------------------------------
# 6. Create classification prompt (Title + Abstract + Graph context, TOP-3)
# -------------------------------------------------------

def create_topic_prompt(title, abstract, category_list, citation_context=None):
    category_lines = "\n".join(
        [
            f"{i}: {cat} - "
            f"{CATEGORY_DESCRIPTIONS.get(cat, 'General computer science topic related to ' + cat)}"
            for i, cat in enumerate(category_list)
        ]
    )

    system_prompt = (
        "You are an AI assistant that performs single-label classification of research papers "
        "into one of the given arXiv computer science categories. "
        "Use both the title and abstract to determine the topic. "
        "You can also use the provided graph-based trend evidence as a hint about the research community. "
        "You must choose exactly one category index for each candidate and you must not invent new categories. "
        "Return ONLY valid JSON, with no extra text or formatting. "
        "For each reasoning, you must paraphrase the topic and avoid reusing exact words or technical phrases "
        "from the title or abstract whenever possible. "
        "Each 'reasoning' field must be about 500 characters in length."
    )

    extra_context_block = ""
    if citation_context is not None:
        extra_context_block = f"\n\nAdditional graph-based context:\n{citation_context}\n"

    user_prompt = f"""
Title:
\"\"\"{title}\"\"\"

Abstract:
\"\"\"{abstract}\"\"\"\
{extra_context_block}

You must classify this paper into exactly one of the following arXiv CS categories (index and name):

{category_lines}

Return ONLY a JSON object of the form:
{{
  "top_k": [
    {{
      "label_idx": <integer>,
      "category": "<string>",
      "reasoning": "<explanation about 500 characters; do not copy phrases or words from the title or abstract>"
    }},
    {{
      "label_idx": <integer>,
      "category": "<string>",
      "reasoning": "<explanation about 500 characters; do not copy phrases or words from the title or abstract>"
    }},
    {{
      "label_idx": <integer>,
      "category": "<string>",
      "reasoning": "<explanation about 500 characters; do not copy phrases or words from the title or abstract>"
    }}
  ]
}}
""".strip()

    return system_prompt, user_prompt


# -------------------------------------------------------
# 7. LLM Call (async)
# -------------------------------------------------------


async def call_llm_for_topic(
    title,
    abstract,
    node_idx=None,
    citation_graph=None,
    node_to_label=None,
    model=LLM_MODEL,
    simulation=False,
    citation_context=None,
):
    # citation_context가 안 넘어온 경우, 여기서 계산
    if (
        citation_context is None
        and (node_idx is not None)
        and (citation_graph is not None)
        and (node_to_label is not None)
    ):
        citation_context = build_citation_trend_context(
            node_idx=node_idx,
            citation_graph=citation_graph,
            node_to_label=node_to_label,
            category_list=category_list,
            max_neighbors=50,
        )

    system_prompt, user_prompt = create_topic_prompt(
        title,
        abstract,
        category_list,
        citation_context=citation_context,
    )

    if simulation:
        # 시뮬레이션 모드일 때는 top-1만 대충 생성
        idx = random.randint(0, len(category_list) - 1)
        dummy = {
            "label_idx": idx,
            "category": category_list[idx],
            "reasoning": "시뮬레이션 결과입니다.",
        }
        result = {
            "label_idx": dummy["label_idx"],
            "category": dummy["category"],
            "reasoning": dummy["reasoning"],
            "candidates": [dummy],
            "citation_context": citation_context,
        }
        return result, 0, 0

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            top_p=1.0,
        )

        content = response.choices[0].message.content
        raw = json.loads(content)  # Expect {"top_k": [...]} or {"candidates": [...]}

        # 프롬프트는 top_k를 요구하지만, 혹시 candidates로 나올 수도 있으니 둘 다 허용
        candidates = raw.get("top_k") or raw.get("candidates")
        if not candidates or not isinstance(candidates, list):
            raise ValueError(
                "LLM output does not contain a valid 'top_k' or 'candidates' list"
            )

        primary = candidates[0]

        result = {
            "label_idx": primary["label_idx"],
            "category": primary["category"],
            "reasoning": primary["reasoning"],
            "candidates": candidates,          # 내부에서는 candidates로 통일
            "citation_context": citation_context,
        }

        usage = response.usage
        return result, usage.prompt_tokens, usage.completion_tokens

    except Exception as e:
        print(f"⛔ LLM Error: {e}")
        return None, 0, 0


# -------------------------------------------------------
# 8. Run classification for all samples (async)
# -------------------------------------------------------
print("📌 Starting LLM classification...\n")

classification_start = time.time()   # 분류 루프 시작 시간
total_llm_time = 0.0                 # LLM 호출에 소요된 총 시간
llm_call_count = 0                   # 실제 LLM 호출 횟수

topic_results = []

# 공통 비용 측정 트래커 (run_costing.py 로 실행될 때만 활성화)
cost_tracker = (
    LLMCostTracker(
        project_size=PROJECT_SIZE,
        # keyword 추출과 동일한 단가 사용 (필요시 변경 가능)
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    )
    if LLM_COSTING_MODE
    else None
)


async def process_single_sample(idx, title, abs_text, node_idx, sem):
    """단일 샘플을 비동기 처리."""
    async with sem:
        if "ERROR" in abs_text:
            print(f" → Abstract missing for node {node_idx}, skipping.")
            return None, 0.0

        # 현재 노드에 대한 citation 기반 context (저장도 하고, 프롬프트에도 사용)
        citation_context = build_citation_trend_context(
            node_idx=node_idx,
            citation_graph=citation_graph,
            node_to_label=node_to_label,
            category_list=category_list,
            max_neighbors=50,
        )

        llm_start = time.time()
        result, in_tok, out_tok = await call_llm_for_topic(
            title,
            abs_text,
            node_idx=node_idx,
            citation_graph=citation_graph,
            node_to_label=node_to_label,
            model=LLM_MODEL,
            simulation=SIMULATION_MODE,
            citation_context=citation_context,
        )
        llm_end = time.time()

        if result is None:
            print(f"   → LLM error, skipping node {node_idx}.")
            return None, 0.0

        # 🔹 primary(pred@1) + top-3 후보
        pred_label_idx = result["label_idx"]
        pred_category = result["category"]
        reasoning = result["reasoning"]
        candidates = result.get("candidates", [])

        # 🔹 원래 레이블 찾기
        true_label_idx = node_to_label.get(int(node_idx), None)
        if true_label_idx is not None:
            true_category = category_list[true_label_idx]
        else:
            true_category = "UNKNOWN"

        # 🔹 콘솔 출력
        print(f"   → Node {node_idx} Pred@1: {pred_label_idx} ({pred_category})")
        if candidates:
            top3_str = ", ".join(
                [f"{c['label_idx']}({c['category']})" for c in candidates[:3]]
            )
            print(f"   → Top-3: {top3_str}")

        if true_label_idx is not None:
            print(f"   → True: {true_label_idx} ({true_category})")
            print(f"   → Match@1: {pred_label_idx == true_label_idx}")
            in_top3 = any(
                (c.get("label_idx") == true_label_idx) for c in candidates[:3]
            )
            print(f"   → In Top-3: {in_top3}")
        else:
            print("   → True: UNKNOWN (not found in FFS label map)")

        record = {
            "node_index": node_idx,
            "title": title,
            "abstract": abs_text,
            "pred_label_idx": pred_label_idx,
            "pred_category": pred_category,
            "reasoning": reasoning,
            "topk_candidates": candidates,  # top-3 전체 저장
            "true_label_idx": true_label_idx,
            "true_category": true_category,
            "is_correct_top1": (
                true_label_idx is not None and pred_label_idx == true_label_idx
            ),
            "is_correct_top3": (
                true_label_idx is not None
                and any(
                    (c.get("label_idx") == true_label_idx)
                    for c in candidates[:3]
                )
            ),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "citation_context": citation_context,
        }

        return record, (llm_end - llm_start)


async def run_classification():
    global total_llm_time, llm_call_count

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for i, (title, abs_text) in enumerate(title_abs_texts):
        node_idx = target_indices[i]
        print(f"[{i+1}/{len(title_abs_texts)}] Scheduling node {node_idx}...")
        tasks.append(process_single_sample(i, title, abs_text, node_idx, sem))

    completed = 0
    for coro in asyncio.as_completed(tasks):
        record, duration = await coro
        if record is None:
            continue

        topic_results.append(record)
        total_llm_time += duration
        llm_call_count += 1

        # 공통 비용 트래커에 토큰 사용량 기록 (코스트 모드에서만)
        if cost_tracker is not None:
            cost_tracker.add_call(
                input_tokens=record["input_tokens"],
                output_tokens=record["output_tokens"],
                meta={"node_index": record["node_index"]},
            )
        completed += 1
        print(f"   → Completed {completed}/{len(tasks)} samples.")


asyncio.run(run_classification())

print("\n✅ Classification completed.\n")
classification_end = time.time()

# 비용 측정 종료 및 요약 출력 (코스트 모드에서만)
if cost_tracker is not None:
    cost_tracker.finalize()
    cost_tracker.print_summary(label="Topic Prediction Cost (sample-based projection)")


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

    if t_idx not in top3_indices:  # Top-3 어디에도 정답이 없는 경우
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
    # 제목/초록 보고 싶으면 아래 필드도 추가해서 쓸 수 있음
    # print(f" Title: {r['title']}")
    # print(f" Abstract: {r['abstract'][:200]}...")

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

GLOBAL_END = time.time()  # 전체 실행 종료 시간

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
