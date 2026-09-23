"""Build a reproducible AI-generated natural-language recommendation dataset.

This tool assumes that the rebuilt master file and the D/T/M extraction result
exist.  It produces three independent assets:

* a text-embedding corpus for semantic retrieval;
* user-style research queries anchored to a hidden relevant paper; and
* LLM relevance labels for a fixed candidate pool per query.

The query and judgement stages are intentionally resumable.  They are an
offline evaluation-set construction workflow, not an online recommender.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = ROOT / "subdataset" / "arxiv_master_final.json"
DEFAULT_DTM = ROOT / "output" / "dtm_luna_16k" / "gpt-5.6-luna.json"
DEFAULT_CITATIONS = ROOT / "subdataset" / "paper_paper_edges.pt"
DEFAULT_OUTPUT = ROOT / "output" / "nl_recommendation_data"
DEFAULT_LLM = "gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

QUERY_SYSTEM = """You design realistic research-paper discovery queries.
Write queries as a researcher would type into a recommendation system.  The
query must describe a research need, not a paper title, author, venue, year, or
identifier.  It must be grounded only in the supplied paper metadata and must
not claim a result that the source does not support."""

JUDGE_SYSTEM = """You are grading academic-paper recommendation relevance.
Judge whether each candidate would satisfy the research need expressed by the
query.  Use title, abstract, and supplied D/T/M metadata only.  Do not reward a
paper merely because it shares generic machine-learning terms.  Score 2 for a
directly relevant paper, 1 for useful partial/background relevance, and 0 for
irrelevant.  Return every supplied node_idx exactly once."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 기반 자연어 추천 데이터/평가셋 생성")
    parser.add_argument("--stage", nargs="+", choices=("embeddings", "queries", "judgments", "all"), default=["all"])
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--dtm-path", type=Path, default=DEFAULT_DTM)
    parser.add_argument("--citation-path", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_LLM, help="질의·판정 모델")
    parser.add_argument("--judge-model", default=None, help="relevance 판정 모델 (기본값: --model)")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--query-count", type=int, default=300)
    parser.add_argument("--queries-per-paper", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--embedding-checkpoint-interval", type=int, default=4,
                        help="임베딩 배치 몇 개마다 재개용 JSON 체크포인트를 저장할지")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-abstract-chars", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="API 호출 없이 입력 및 후보 풀만 검증")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    # A process-specific temporary name prevents a stale/manual resume from
    # corrupting an otherwise valid checkpoint through a shared ``.tmp`` file.
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_terms(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def load_papers(master_path: Path, dtm_path: Path) -> list[dict[str, Any]]:
    """Load the canonical graph ordering and attach successful D/T/M results."""
    master = read_json(master_path, None)
    if not isinstance(master, list):
        raise FileNotFoundError(f"마스터 파일이 필요합니다: {master_path}")
    dtm_rows = read_json(dtm_path, None)
    if not isinstance(dtm_rows, list):
        raise FileNotFoundError(f"D/T/M 결과가 필요합니다: {dtm_path}")
    dtm_by_node = {
        int(row["node_idx"]): row for row in dtm_rows
        if row.get("status") == "success" and "node_idx" in row
    }
    papers: list[dict[str, Any]] = []
    seen: set[int] = set()
    for master_row in master:
        node_idx = int(master_row["node_idx"])
        dtm = dtm_by_node.get(node_idx)
        title = str(master_row.get("title", "")).strip()
        abstract = str(master_row.get("abstract", "")).strip()
        if node_idx in seen or not dtm or not title or not abstract or title == "Unknown":
            continue
        knowledge = dtm.get("knowledge", master_row.get("knowledge", {}))
        papers.append({
            "node_idx": node_idx,
            "paper_id": str(master_row.get("paper_id", dtm.get("paper_id", node_idx))),
            "title": title,
            "abstract": abstract,
            "knowledge": {key: normalize_terms(knowledge.get(key, [])) for key in ("domain", "task", "method")},
        })
        seen.add(node_idx)
    if len(papers) < 2:
        raise ValueError("제목·초록·성공한 D/T/M이 연결된 논문이 2편 이상 필요합니다.")
    papers.sort(key=lambda row: row["node_idx"])
    return papers


def paper_text(paper: dict[str, Any], max_abstract_chars: int) -> str:
    tags = "; ".join(term for key in ("domain", "task", "method") for term in paper["knowledge"][key])
    return f"Title: {paper['title']}\nAbstract: {paper['abstract'][:max_abstract_chars]}\nD/T/M: {tags}"


def paper_signature(papers: Iterable[dict[str, Any]]) -> str:
    value = "".join(f"{p['node_idx']}:{p['paper_id']}\n" for p in papers)
    return hashlib.sha256(value.encode()).hexdigest()


def split_for_node(node_idx: int, seed: int) -> str:
    """Stable paper-level split: no anchor paper can appear in two query splits."""
    value = int(hashlib.sha256(f"{seed}:{node_idx}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def build_inverted_terms(papers: list[dict[str, Any]]) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = defaultdict(list)
    for paper in papers:
        for values in paper["knowledge"].values():
            for term in values:
                inverted[term.casefold()].append(paper["node_idx"])
    return inverted


def load_citations(path: Path, valid_nodes: set[int]) -> dict[int, set[int]]:
    if not path.exists():
        return {}
    edges = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(edges, torch.Tensor) or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"인용 엣지 형식이 올바르지 않습니다: {path}")
    neighbors: dict[int, set[int]] = defaultdict(set)
    for src, dst in edges.t().tolist():
        if src in valid_nodes and dst in valid_nodes:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return neighbors


def candidate_pool(anchor: dict[str, Any], papers_by_node: dict[int, dict[str, Any]], inverted: dict[str, list[int]],
                   citations: dict[int, set[int]], count: int, seed: int) -> list[int]:
    """Create a fixed pool without leaking a D/T/M variant into evaluation.

    Citation neighbours provide hard academic neighbours; the rest are random.
    D/T/M overlap is intentionally not used because the pool must stay neutral
    when Raw-DTM and normalized-DTM recommenders are compared.
    """
    if count < 4:
        raise ValueError("--candidate-count는 4 이상이어야 합니다.")
    rng = random.Random(f"{seed}:{anchor['node_idx']}")
    selected = [anchor["node_idx"]]
    related: set[int] = set(citations.get(anchor["node_idx"], set()))
    related.discard(anchor["node_idx"])
    related = {node for node in related if node in papers_by_node}
    related_list = sorted(related)
    rng.shuffle(related_list)
    selected.extend(related_list[: count - 1])
    remaining = [node for node in papers_by_node if node not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def query_prompt(anchor: dict[str, Any], query_count: int, max_abstract_chars: int) -> str:
    return f"""Create {query_count} distinct natural-language literature-search queries for the supplied anchor paper.

Requirements:
- Each query is 12-45 words, in English, and expresses an information need.
- Do not copy a title phrase longer than 3 consecutive words.
- Do not mention authors, venue, year, paper IDs, or that an anchor paper exists.
- Vary emphasis across problem, application, and method only when supported.
- The anchor must be a direct relevance-2 result for every query.

Return only JSON: {{"queries": [{{"query": "...", "intent": "..."}}]}}.

Anchor metadata:
{paper_text(anchor, max_abstract_chars)}"""


def judgement_prompt(query: str, candidates: list[dict[str, Any]], max_abstract_chars: int) -> str:
    rendered = "\n\n".join(
        f"node_idx={p['node_idx']}\n{paper_text(p, max_abstract_chars)}" for p in candidates
    )
    return f"""Query: {query}

Score the following candidates.  Return only JSON in this exact form:
{{"labels": [{{"node_idx": 123, "relevance": 0}}]}}

Candidates:
{rendered}"""


async def call_json(client: AsyncOpenAI, model: str, system: str, user: str) -> tuple[dict[str, Any], dict[str, int]]:
    # This offline batch has a different query/candidate payload on every call.
    # Explicit mode with no breakpoints disables implicit prompt-cache writes,
    # which would otherwise add cache-write charges without useful cache reads.
    request: dict[str, Any] = {
        "model": model,
        "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": 900,
        "prompt_cache_options": {"mode": "explicit"},
    }
    if model.startswith("gpt-5"):
        request["reasoning"] = {"effort": "none"}
    else:
        request["temperature"] = 0
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.responses.create(**request)
            raw = response.output_text or "{}"
            usage = response.usage
            return json.loads(raw), {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            }
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI JSON 요청이 4회 실패했습니다: {last_error}")


async def build_queries(client: AsyncOpenAI, papers: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.output_dir / "queries.json"
    existing = {} if args.overwrite else {row["query_id"]: row for row in read_json(path, [])}
    anchors = random.Random(args.seed).sample(papers, min(args.query_count, len(papers)))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(anchor: dict[str, Any]) -> list[dict[str, Any]]:
        prefix = f"q-{anchor['node_idx']}-"
        if all(f"{prefix}{i}" in existing for i in range(args.queries_per_paper)):
            return []
        async with semaphore:
            started = time.perf_counter()
            data, usage = await call_json(client, args.model, QUERY_SYSTEM,
                                          query_prompt(anchor, args.queries_per_paper, args.max_abstract_chars))
        rows = []
        for i, value in enumerate(data.get("queries", [])[:args.queries_per_paper]):
            query = str(value.get("query", "")).strip()
            if query:
                rows.append({"query_id": f"{prefix}{i}", "anchor_node_idx": anchor["node_idx"],
                             "anchor_paper_id": anchor["paper_id"], "query": query,
                             "intent": str(value.get("intent", "")).strip(), "model": args.model,
                             "split": split_for_node(anchor["node_idx"], args.seed),
                             "latency_seconds": round(time.perf_counter() - started, 3), "usage": usage})
        if len(rows) != args.queries_per_paper:
            raise ValueError(f"anchor {anchor['node_idx']}의 질의 수가 부족합니다.")
        return rows

    pending = [anchor for anchor in anchors if not all(
        f"q-{anchor['node_idx']}-{i}" in existing for i in range(args.queries_per_paper)
    )]
    for offset in range(0, len(pending), args.concurrency):
        batch_rows = await asyncio.gather(*(one(anchor) for anchor in pending[offset:offset + args.concurrency]))
        for rows in batch_rows:
            for row in rows:
                existing[row["query_id"]] = row
        atomic_json(path, [existing[key] for key in sorted(existing)])
        print(f"[queries] {min(offset + args.concurrency, len(pending))}/{len(pending)}")
    return [existing[key] for key in sorted(existing)]


async def build_judgments(client: AsyncOpenAI, queries: list[dict[str, Any]], papers_by_node: dict[int, dict[str, Any]],
                          inverted: dict[str, list[int]], citations: dict[int, set[int]], args: argparse.Namespace) -> None:
    path = args.output_dir / "judgments.json"
    existing = {} if args.overwrite else {row["query_id"]: row for row in read_json(path, [])}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(query_row: dict[str, Any]) -> dict[str, Any] | None:
        if query_row["query_id"] in existing:
            return None
        anchor = papers_by_node[query_row["anchor_node_idx"]]
        ids = candidate_pool(anchor, papers_by_node, inverted, citations, args.candidate_count, args.seed)
        candidates = [papers_by_node[node] for node in ids]
        started = time.perf_counter()
        raw_labels: dict[int, int] = {}
        usage: dict[str, int] = {}
        for attempt in range(3):
            async with semaphore:
                data, usage = await call_json(client, args.judge_model, JUDGE_SYSTEM,
                                              judgement_prompt(query_row["query"], candidates, args.max_abstract_chars))
            raw_labels = {int(item["node_idx"]): int(item["relevance"]) for item in data.get("labels", [])
                          if isinstance(item, dict) and str(item.get("node_idx", "")).isdigit() and item.get("relevance") in (0, 1, 2)}
            if set(raw_labels) == set(ids):
                break
        if set(raw_labels) != set(ids):
            raise ValueError(f"{query_row['query_id']}: 3회 재시도 후에도 후보 풀 라벨이 완전하지 않습니다.")
        # The anchor was used to generate this query; preserving it as direct relevance
        # prevents a single judge mistake from removing the only guaranteed positive.
        raw_labels[anchor["node_idx"]] = 2
        return {"query_id": query_row["query_id"], "query": query_row["query"], "anchor_node_idx": anchor["node_idx"],
                "candidate_node_indices": ids, "labels": [{"node_idx": node, "relevance": raw_labels[node]} for node in ids],
                "model": args.judge_model, "latency_seconds": round(time.perf_counter() - started, 3), "usage": usage}

    pending = [query for query in queries if query["query_id"] not in existing]
    for offset in range(0, len(pending), args.concurrency):
        rows = await asyncio.gather(*(one(query) for query in pending[offset:offset + args.concurrency]))
        for row in rows:
            if row:
                existing[row["query_id"]] = row
        atomic_json(path, [existing[key] for key in sorted(existing)])
        print(f"[judgments] {min(offset + args.concurrency, len(pending))}/{len(pending)}")


async def build_embeddings(client: AsyncOpenAI, papers: list[dict[str, Any]], args: argparse.Namespace) -> None:
    checkpoint = args.output_dir / "embedding_rows.json"
    existing = {} if args.overwrite else {int(row["node_idx"]): row for row in read_json(checkpoint, [])}
    pending = [paper for paper in papers if paper["node_idx"] not in existing]
    for batch_number, offset in enumerate(range(0, len(pending), args.embedding_batch_size), start=1):
        batch = pending[offset:offset + args.embedding_batch_size]
        response = await client.embeddings.create(model=args.embedding_model,
                                                  input=[paper_text(p, args.max_abstract_chars) for p in batch])
        for paper, item in zip(batch, response.data):
            existing[paper["node_idx"]] = {"node_idx": paper["node_idx"], "paper_id": paper["paper_id"], "embedding": item.embedding}
        if batch_number % args.embedding_checkpoint_interval == 0 or offset + len(batch) == len(pending):
            atomic_json(checkpoint, [existing[key] for key in sorted(existing)])
        print(f"[embeddings] {min(offset + len(batch), len(pending))}/{len(pending)}")
    rows = [existing[paper["node_idx"]] for paper in papers]
    torch.save({"model": args.embedding_model, "node_indices": torch.tensor([r["node_idx"] for r in rows]),
                "paper_ids": [r["paper_id"] for r in rows], "embeddings": torch.tensor([r["embedding"] for r in rows], dtype=torch.float32),
                "corpus_signature": paper_signature(papers)}, args.output_dir / "paper_text_embeddings.pt")


async def main() -> None:
    args = parse_args()
    args.judge_model = args.judge_model or args.model
    if args.concurrency < 1 or args.embedding_batch_size < 1 or args.embedding_checkpoint_interval < 1:
        raise ValueError("concurrency, embedding-batch-size, embedding-checkpoint-interval은 1 이상이어야 합니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = load_papers(args.master_path, args.dtm_path)
    papers_by_node = {paper["node_idx"]: paper for paper in papers}
    manifest = {"paper_count": len(papers), "corpus_signature": paper_signature(papers), "master_path": str(args.master_path),
                "dtm_path": str(args.dtm_path), "seed": args.seed}
    atomic_json(args.output_dir / "manifest.json", manifest)
    inverted = build_inverted_terms(papers)
    citations = load_citations(args.citation_path, set(papers_by_node))
    stages = set(("embeddings", "queries", "judgments") if "all" in args.stage else args.stage)
    if args.prepare_only:
        if "judgments" in stages:
            sample = papers[0]
            candidate_pool(sample, papers_by_node, inverted, citations, args.candidate_count, args.seed)
        print(f"입력 검증 완료: {len(papers)} papers, citation-neighbor nodes={len(citations)}")
        return
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 필요합니다. --prepare-only는 키 없이 실행할 수 있습니다.")
    client = AsyncOpenAI()
    if "embeddings" in stages:
        await build_embeddings(client, papers, args)
    queries = read_json(args.output_dir / "queries.json", [])
    if "queries" in stages:
        queries = await build_queries(client, papers, args)
    if "judgments" in stages:
        if not queries:
            raise SystemExit("judgments 단계에는 먼저 queries 단계를 실행해야 합니다.")
        await build_judgments(client, queries, papers_by_node, inverted, citations, args)


if __name__ == "__main__":
    asyncio.run(main())
