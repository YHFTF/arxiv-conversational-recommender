"""An intent-aware, zero-shot natural-language paper and author retriever.

The query planner separates required, optional, incidental, and excluded
concepts before embedding. Ranking combines semantic similarity with the
catalog's D/T/M knowledge and explicit query constraints; author ranking uses
both evidence quality and the number of relevant authored papers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "output" / "nl_recommendation_data"
DEFAULT_MASTER = ROOT / "subdataset" / "arxiv_master_final.json"
DEFAULT_CITATIONS = ROOT / "subdataset" / "paper_paper_edges.pt"
DEFAULT_AUTHOR_EDGES = ROOT / "subdataset" / "author_paper_edges.pt"
DEFAULT_AUTHOR_MAPPING = ROOT / "output" / "author_remapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="자연어 Query → Paper 검색 프로토타입")
    parser.add_argument("--query", help="단발성 추천 질의 (생략하면 대화형 모드)")
    parser.add_argument("--evaluate", action="store_true", help="AI 라벨 고정 후보 풀에서 평가")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test",
                        help="--evaluate 시 사용할 질의 분할 (기본: test)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=100,
                        help="전체 검색 뒤 재정렬할 상위 후보 수")
    parser.add_argument("--facet-candidate-depth", type=int, default=30,
                        help="각 must facet에서 추가로 가져올 후보 수 (기본: 30)")
    parser.add_argument("--dtm-weight", type=float, default=0.04,
                        help="추출된 D/T/M과 구축 데이터 간 일치 가중치 (기본: 0.04)")
    parser.add_argument("--intent-analysis", action=argparse.BooleanOptionalAction, default=True,
                        help="LLM으로 검색 의도와 키워드 중요도를 분석합니다 (기본: 사용)")
    parser.add_argument("--intent-model", default=os.getenv("INTENT_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--intent-workers", type=int, default=4,
                        help="평가 시 의도 분석 동시 API 호출 수 (기본: 4)")
    parser.add_argument("--must-weight", type=float, default=0.08,
                        help="필수 키워드 일치/누락 가중치 (기본: 0.08)")
    parser.add_argument("--should-weight", type=float, default=0.03,
                        help="선호 키워드 일치 가중치 (기본: 0.03)")
    parser.add_argument("--exclude-penalty", type=float, default=0.12,
                        help="제외 키워드 일치 패널티 (기본: 0.12)")
    parser.add_argument("--semantic-term-threshold", type=float, default=0.30,
                        help="필수·선호 용어의 의미 일치가 시작되는 cosine 유사도 (기본: 0.30)")
    parser.add_argument("--exclude-semantic-threshold", type=float, default=0.45,
                        help="제외 용어의 의미 일치가 시작되는 cosine 유사도 (기본: 0.45)")
    parser.add_argument("--semantic-temperature", type=float, default=0.04,
                        help="facet cosine 차이를 확대하는 sigmoid 온도 (기본: 0.04)")
    parser.add_argument("--graph-diversity-weight", type=float, default=0.0,
                        help="실험용: 이미 선택된 결과의 인용 이웃 중복 패널티 (기본: 0)")
    parser.add_argument("--author-top-k", type=int, default=0,
                        help="함께 출력할 저자 추천 수. 0이면 저자 추천을 생략합니다 (기본: 0)")
    parser.add_argument("--author-evidence-count", type=int, default=3,
                        help="저자 적합도에 평균낼 상위 저작 수 (기본: 3)")
    parser.add_argument("--author-relevance-depth", type=int, default=500,
                        help="저자의 관련 논문 수를 셀 전체 논문 순위 깊이 (기본: 500)")
    parser.add_argument("--author-max-weight", type=float, default=0.4,
                        help="저자 대표 논문(max R) 가중치 (기본: 0.4)")
    parser.add_argument("--author-mean-weight", type=float, default=0.4,
                        help="저자 상위 관련 논문 평균 가중치 (기본: 0.4)")
    parser.add_argument("--author-count-weight", type=float, default=0.2,
                        help="정규화된 관련 논문 수 가중치 (기본: 0.2)")
    parser.add_argument("--author-count-cap", type=int, default=10,
                        help="관련 논문 수 가점이 포화되는 논문 수 (기본: 10)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--citation-path", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--author-edge-path", type=Path, default=DEFAULT_AUTHOR_EDGES)
    parser.add_argument("--author-mapping-path", type=Path, default=DEFAULT_AUTHOR_MAPPING)
    parser.add_argument("--query-embedding-cache", type=Path, default=None,
                        help="평가 질의 임베딩 캐시(.pt). 기본: data-dir/query_embeddings.pt")
    parser.add_argument("--refresh-query-cache", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def cache_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in value)


def progress(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)


def normalise_text(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "search_intent": {"type": "string", "enum": [
            "discover", "compare", "implement", "evaluate", "survey", "find_expert"]},
        "rewritten_query": {"type": "string"},
        "must_terms": {"type": "array", "items": {"type": "string"}},
        "should_terms": {"type": "array", "items": {"type": "string"}},
        "low_priority_terms": {"type": "array", "items": {"type": "string"}},
        "exclude_terms": {"type": "array", "items": {"type": "string"}},
        "domain_terms": {"type": "array", "items": {"type": "string"}},
        "task_terms": {"type": "array", "items": {"type": "string"}},
        "method_terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["search_intent", "rewritten_query", "must_terms", "should_terms",
                 "low_priority_terms", "exclude_terms", "domain_terms", "task_terms", "method_terms"],
    "additionalProperties": False,
}


def analyse_query_intent(client: OpenAI, model: str, query: str) -> dict[str, Any]:
    """Convert a natural-language request into retrieval constraints."""
    response = client.responses.create(
        model=model,
        instructions=(
            "You are a scientific-search query planner. Interpret the user's actual research need. "
            "Return concise English retrieval phrases even when the query is in another language. "
            "must_terms are indispensable concepts explicitly required by the user; do not overuse them. "
            "should_terms improve relevance but are optional. low_priority_terms are incidental context and "
            "must be omitted from rewritten_query. exclude_terms are only explicit negative constraints. "
            "Classify useful concepts into domain, task, and method terms. The rewritten query must preserve "
            "the intent and must terms, and should read like a compact paper abstract search query."
        ),
        input=query,
        text={"format": {"type": "json_schema", "name": "query_intent",
                         "strict": True, "schema": INTENT_SCHEMA}},
    )
    intent = json.loads(response.output_text)
    for key in ("must_terms", "should_terms", "low_priority_terms", "exclude_terms",
                "domain_terms", "task_terms", "method_terms"):
        intent[key] = list(dict.fromkeys(
            normalise_text(str(term)) for term in intent.get(key, []) if str(term).strip()))
    intent["rewritten_query"] = str(intent.get("rewritten_query") or query).strip()
    return intent


def plain_query_intent(query: str) -> dict[str, Any]:
    return {"search_intent": "discover", "rewritten_query": query, "must_terms": [],
            "should_terms": [], "low_priority_terms": [], "exclude_terms": [],
            "domain_terms": [], "task_terms": [], "method_terms": []}


def build_embedding_query(intent: dict[str, Any]) -> str:
    required = "; ".join(intent["must_terms"])
    return (f"{intent['rewritten_query']}\nRequired concepts: {required}"
            if required else intent["rewritten_query"])


def intent_terms(intent: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        term for key in ("must_terms", "should_terms", "exclude_terms")
        for term in intent[key]))


def embed_query_plan(client: OpenAI, model: str,
                     intent: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Embed the rewritten query and all constraint terms in one API request."""
    terms = intent_terms(intent)
    vectors = embed_texts(client, model, [build_embedding_query(intent), *terms])
    return vectors[0], {term: vector for term, vector in zip(terms, vectors[1:])}


def load_catalog(master_path: Path, embedding_path: Path) -> tuple[list[dict[str, Any]], torch.Tensor, str]:
    embedded = torch.load(embedding_path, weights_only=False, map_location="cpu")
    node_indices = [int(node) for node in embedded["node_indices"].tolist()]
    vectors = F.normalize(embedded["embeddings"].float(), dim=1)
    if len(node_indices) != vectors.shape[0]:
        raise ValueError("논문 node_idx와 임베딩 행 수가 일치하지 않습니다.")
    master = {int(row["node_idx"]): row for row in read_json(master_path)}
    missing = [node for node in node_indices if node not in master]
    if missing:
        raise ValueError(f"임베딩 논문 {len(missing)}개가 master 파일에 없습니다.")
    papers = []
    for node in node_indices:
        row = master[node]
        knowledge = row.get("knowledge", {})
        terms = []
        knowledge_terms: dict[str, list[str]] = {"domain": [], "task": [], "method": []}
        if isinstance(knowledge, dict):
            for key in ("domain", "task", "method"):
                # ``graph`` is frequency-filtered for GNN construction. Search
                # should retain normalized and raw phrases that graph filtering
                # intentionally removed, while still accepting graph spellings.
                values: list[Any] = []
                for source in (knowledge, knowledge.get("normalized", {}), knowledge.get("graph", {})):
                    if isinstance(source, dict) and isinstance(source.get(key), list):
                        values.extend(source[key])
                knowledge_terms[key] = list(dict.fromkeys(
                    normalise_text(str(value)) for value in values if str(value).strip()))
                terms.extend(knowledge_terms[key])
        papers.append({"node_idx": node, "paper_id": str(row.get("paper_id", node)),
                       "title": str(row.get("title", "")), "abstract": str(row.get("abstract", "")),
                       "terms": sorted(set(terms)), "knowledge": knowledge_terms,
                       "search_text": normalise_text(" ".join([
                           str(row.get("title", "")), str(row.get("abstract", "")), *terms]))})
    return papers, vectors, str(embedded["model"])


def load_neighbours(path: Path, valid_nodes: set[int]) -> dict[int, set[int]]:
    neighbours: dict[int, set[int]] = defaultdict(set)
    if not path.exists():
        return neighbours
    edges = torch.load(path, weights_only=False, map_location="cpu")
    for src, dst in edges.t().tolist():
        if src in valid_nodes and dst in valid_nodes:
            neighbours[src].add(dst)
            neighbours[dst].add(src)
    return neighbours


def load_authors(edge_path: Path, mapping_path: Path, valid_nodes: set[int]) -> dict[int, dict[str, Any]]:
    """Load author--paper edges keyed by local author node index.

    The edge file is deliberately the source of authorship rather than the
    ``authors`` field in the master JSON, so this retrieval path uses the same
    graph relation as the heterogeneous graph models.
    """
    if not edge_path.exists() or not mapping_path.exists():
        return {}
    edges = torch.load(edge_path, weights_only=False, map_location="cpu")
    if not isinstance(edges, torch.Tensor) or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("저자 엣지는 shape [2, num_edges] Tensor여야 합니다.")
    mapping = read_json(mapping_path)
    names = mapping.get("idx_to_name", {})
    ids = mapping.get("idx_to_id", {})
    authors: dict[int, dict[str, Any]] = {}
    for author_idx, paper_node in edges.t().tolist():
        author_idx, paper_node = int(author_idx), int(paper_node)
        if paper_node not in valid_nodes:
            continue
        key = str(author_idx)
        authors.setdefault(author_idx, {
            "author_idx": author_idx,
            "author_id": str(ids.get(key, "")),
            "author_name": str(names.get(key, "Unknown")),
            "paper_nodes": [],
        })["paper_nodes"].append(paper_node)
    return authors


def semantic_facet_strengths(terms: list[str], term_vectors: dict[str, torch.Tensor],
                             candidate_vectors: torch.Tensor, threshold: float,
                             temperature: float) -> torch.Tensor:
    """Return calibrated [candidate, facet] semantic match strengths."""
    if not terms:
        return torch.empty((candidate_vectors.shape[0], 0), dtype=torch.float32)
    vectors = torch.stack([term_vectors[term] for term in terms])
    similarities = torch.mm(F.normalize(candidate_vectors.float(), dim=1),
                            F.normalize(vectors.float(), dim=1).t())
    return torch.sigmoid((similarities - threshold) / temperature)


def query_signal_scores(intent: dict[str, Any], term_vectors: dict[str, torch.Tensor],
                        candidate_vectors: torch.Tensor, must_weight: float, should_weight: float,
                        exclude_penalty: float, semantic_threshold: float,
                        exclude_semantic_threshold: float,
                        temperature: float) -> tuple[torch.Tensor, list[dict[str, float]]]:
    must_facets = semantic_facet_strengths(
        intent["must_terms"], term_vectors, candidate_vectors, semantic_threshold, temperature)
    should_facets = semantic_facet_strengths(
        intent["should_terms"], term_vectors, candidate_vectors, semantic_threshold, temperature)
    exclude_facets = semantic_facet_strengths(
        intent["exclude_terms"], term_vectors, candidate_vectors, exclude_semantic_threshold, temperature)
    # Required conditions are conjunctive: a single weak facet pulls down the
    # geometric mean and receives a logarithmic penalty that other facets
    # cannot compensate for.
    if must_facets.shape[1]:
        must = torch.exp(torch.log(must_facets.clamp_min(1e-6)).mean(dim=1))
        must_min = must_facets.min(dim=1).values
        must_adjustment = must_weight * torch.log(must.clamp_min(1e-6))
    else:
        must = torch.ones(candidate_vectors.shape[0], dtype=torch.float32)
        must_min = torch.ones_like(must)
        must_adjustment = torch.zeros_like(must)
    should = (should_facets.mean(dim=1) if should_facets.shape[1]
              else torch.zeros(candidate_vectors.shape[0], dtype=torch.float32))
    # One strong excluded concept is sufficient to penalize a paper.
    excluded = (exclude_facets.max(dim=1).values if exclude_facets.shape[1]
                else torch.zeros(candidate_vectors.shape[0], dtype=torch.float32))
    adjustments = must_adjustment + should_weight * should - exclude_penalty * excluded
    details = [{"must_coverage": float(must[position]),
                "must_min": float(must_min[position]),
                "should_coverage": float(should[position]),
                "exclude_coverage": float(excluded[position])}
               for position in range(candidate_vectors.shape[0])]
    return adjustments, details


def dtm_bonus(intent: dict[str, Any], papers: list[dict[str, Any]], indices: list[int]) -> torch.Tensor:
    values = []
    for index in indices:
        matched, total = 0.0, 0
        for kind in ("domain", "task", "method"):
            query_terms = intent[f"{kind}_terms"]
            if not query_terms:
                continue
            total += len(query_terms)
            paper_terms = papers[index]["knowledge"][kind]
            matched += sum(any(query_term in paper_term or paper_term in query_term
                               for paper_term in paper_terms) for query_term in query_terms)
        values.append(matched / total if total else 0.0)
    return torch.tensor(values, dtype=torch.float32)


def relevance_scores(intent: dict[str, Any], query_vector: torch.Tensor,
                     term_vectors: dict[str, torch.Tensor], papers: list[dict[str, Any]],
                     paper_vectors: torch.Tensor, indices: list[int], dtm_weight: float,
                     must_weight: float, should_weight: float, exclude_penalty: float,
                     semantic_threshold: float, exclude_semantic_threshold: float,
                     temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    candidate_vectors = paper_vectors[indices]
    text_scores = torch.mv(candidate_vectors, F.normalize(query_vector.float(), dim=0))
    keyword_scores, details = query_signal_scores(
        intent, term_vectors, candidate_vectors, must_weight, should_weight, exclude_penalty,
        semantic_threshold, exclude_semantic_threshold, temperature)
    dtm_scores = dtm_bonus(intent, papers, indices)
    return text_scores + keyword_scores + dtm_weight * dtm_scores, text_scores, dtm_scores, details


def rerank(intent: dict[str, Any], query_vector: torch.Tensor, term_vectors: dict[str, torch.Tensor],
           papers: list[dict[str, Any]], paper_vectors: torch.Tensor,
           neighbours: dict[int, set[int]], top_k: int, candidate_depth: int, facet_candidate_depth: int,
           dtm_weight: float,
           graph_diversity_weight: float, must_weight: float, should_weight: float,
           exclude_penalty: float, semantic_threshold: float, exclude_semantic_threshold: float,
           temperature: float, restrict_to: list[int] | None = None) -> list[dict[str, Any]]:
    if top_k < 1 or candidate_depth < top_k:
        raise ValueError("top-k는 1 이상이고 candidate-depth 이하여야 합니다.")
    if restrict_to is None:
        scores = torch.mv(paper_vectors, F.normalize(query_vector.float(), dim=0))
        depth = min(candidate_depth, len(papers))
        candidate_indices = torch.topk(scores, depth).indices.tolist()
        # Whole-query retrieval can hide papers that strongly satisfy one
        # indispensable condition. Retrieve per-must-facet candidates too,
        # then let conjunctive reranking decide the final order.
        facet_depth = min(facet_candidate_depth, len(papers))
        if facet_depth:
            for term in intent["must_terms"]:
                facet_scores = torch.mv(paper_vectors, F.normalize(term_vectors[term].float(), dim=0))
                candidate_indices.extend(torch.topk(facet_scores, facet_depth).indices.tolist())
            candidate_indices = list(dict.fromkeys(candidate_indices))
    else:
        candidate_indices = restrict_to
    combined, text_scores, bonuses, signal_details = relevance_scores(
        intent, query_vector, term_vectors, papers, paper_vectors, candidate_indices, dtm_weight,
        must_weight, should_weight, exclude_penalty, semantic_threshold,
        exclude_semantic_threshold, temperature)
    remaining = {index: position for position, index in enumerate(candidate_indices)}
    selected: list[int] = []
    results = []
    while remaining and len(selected) < top_k:
        best_index, best_score, best_penalty = -1, -float("inf"), 0.0
        for index, position in remaining.items():
            connected = any(papers[index]["node_idx"] in neighbours.get(papers[chosen]["node_idx"], set())
                            for chosen in selected)
            penalty = graph_diversity_weight if connected else 0.0
            score = float(combined[position]) - penalty
            if score > best_score:
                best_index, best_score, best_penalty = index, score, penalty
        position = remaining.pop(best_index)
        selected.append(best_index)
        results.append({"catalog_index": best_index, "node_idx": papers[best_index]["node_idx"],
                        "paper_id": papers[best_index]["paper_id"], "title": papers[best_index]["title"],
                        "text_score": round(float(text_scores[position]), 6),
                        "dtm_bonus": round(float(bonuses[position]), 6),
                        "intent_adjustment": round(float(combined[position] - text_scores[position]
                                                         - dtm_weight * bonuses[position]), 6),
                        **{key: round(value, 6) for key, value in signal_details[position].items()},
                        "graph_penalty": round(best_penalty, 6), "score": round(best_score, 6)})
    return results


def recommend_authors(intent: dict[str, Any], query_vector: torch.Tensor, term_vectors: dict[str, torch.Tensor],
                      papers: list[dict[str, Any]], paper_vectors: torch.Tensor,
                      authors: dict[int, dict[str, Any]], top_k: int,
                      evidence_count: int, relevance_depth: int, max_weight: float, mean_weight: float,
                      count_weight: float, count_cap: int, dtm_weight: float, must_weight: float,
                      should_weight: float, exclude_penalty: float, semantic_threshold: float,
                      exclude_semantic_threshold: float, temperature: float) -> list[dict[str, Any]]:
    """Rank authors by best work, repeated expertise, and relevant output."""
    if top_k <= 0 or not authors:
        return []
    if evidence_count < 1:
        raise ValueError("author-evidence-count는 1 이상이어야 합니다.")
    all_indices = list(range(len(papers)))
    scores, text_scores, _, _ = relevance_scores(
        intent, query_vector, term_vectors, papers, paper_vectors, all_indices, dtm_weight,
        must_weight, should_weight, exclude_penalty, semantic_threshold,
        exclude_semantic_threshold, temperature)
    depth = min(relevance_depth, len(papers))
    relevant_ranking = torch.topk(scores, depth).indices.tolist()
    relevant_indices = set(relevant_ranking)
    def bounded_relevance(index: int) -> float:
        # Keep relevance comparable across queries. Query-local min-max scaling
        # would make the best paper look perfect even for a weak query match.
        return min(1.0, max(0.0, float(scores[index])))

    catalog_by_node = {paper["node_idx"]: index for index, paper in enumerate(papers)}
    ranked: list[dict[str, Any]] = []
    for author in authors.values():
        paper_indices = [catalog_by_node[node] for node in author["paper_nodes"] if node in catalog_by_node]
        related_indices = [index for index in paper_indices if index in relevant_indices]
        if not related_indices:
            continue
        best_indices = sorted(related_indices, key=lambda index: float(scores[index]), reverse=True)[:evidence_count]
        bounded_scores = [bounded_relevance(index) for index in best_indices]
        max_relevance = bounded_scores[0]
        mean_top_k_raw = sum(bounded_scores) / len(bounded_scores)
        mean_confidence = min(1.0, len(related_indices) / evidence_count)
        # Zero-padding missing Top-K evidence is equivalent to multiplying the
        # observed mean by n/K. A one-paper author therefore cannot claim the
        # same repeated-expertise confidence as an author with K papers.
        mean_top_k = mean_top_k_raw * mean_confidence
        count_score = math.log1p(min(len(related_indices), count_cap)) / math.log1p(count_cap)
        max_component = max_weight * max_relevance
        mean_component = mean_weight * mean_top_k
        count_component = count_weight * count_score
        evidence = [{"node_idx": papers[index]["node_idx"], "paper_id": papers[index]["paper_id"],
                     "title": papers[index]["title"], "text_score": round(float(text_scores[index]), 6),
                     "relevance_score": round(float(scores[index]), 6),
                     "bounded_relevance": round(bounded_relevance(index), 6)}
                    for index in best_indices]
        ranked.append({"author_idx": author["author_idx"], "author_id": author["author_id"],
                       "author_name": author["author_name"],
                       "score": round(max_component + mean_component + count_component, 6),
                       "max_relevance": round(max_relevance, 6),
                       "mean_top_k_raw": round(mean_top_k_raw, 6),
                       "mean_confidence": round(mean_confidence, 6),
                       "mean_top_k": round(mean_top_k, 6),
                       "count_score": round(count_score, 6), "max_component": round(max_component, 6),
                       "mean_component": round(mean_component, 6),
                       "count_component": round(count_component, 6),
                       "related_paper_count": len(related_indices), "paper_count": len(paper_indices),
                       "evidence_papers": evidence})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> torch.Tensor:
    response = client.embeddings.create(model=model, input=texts)
    return torch.tensor([item.embedding for item in response.data], dtype=torch.float32)


def load_or_embed_intent_terms(client: OpenAI, model: str, intents: dict[str, dict[str, Any]],
                               cache_path: Path, refresh: bool) -> dict[str, torch.Tensor]:
    required_terms = list(dict.fromkeys(term for intent in intents.values() for term in intent_terms(intent)))
    cached: dict[str, torch.Tensor] = {}
    if cache_path.exists() and not refresh:
        value = torch.load(cache_path, weights_only=False, map_location="cpu")
        if value.get("model") == model:
            cached = value.get("vectors", {})
    pending = [term for term in required_terms if term not in cached]
    progress("term-embedding", f"cache hit={len(required_terms) - len(pending)}/{len(required_terms)}, "
                               f"pending={len(pending)}")
    for start in range(0, len(pending), 128):
        batch = pending[start:start + 128]
        progress("term-embedding", f"API batch {start + 1}-{start + len(batch)}/{len(pending)}")
        vectors = embed_texts(client, model, batch)
        cached.update({term: vector for term, vector in zip(batch, vectors)})
    if pending:
        torch.save({"model": model, "vectors": cached}, cache_path)
        progress("term-embedding", f"saved: {cache_path.name}")
    return {term: cached[term] for term in required_terms}


def ndcg_at_k(labels: list[int], k: int) -> float:
    gains = [((2 ** label) - 1) / torch.log2(torch.tensor(rank + 2.0)).item()
             for rank, label in enumerate(labels[:k])]
    ideal = sorted(labels, reverse=True)
    ideal_gains = [((2 ** label) - 1) / torch.log2(torch.tensor(rank + 2.0)).item()
                   for rank, label in enumerate(ideal[:k])]
    return sum(gains) / sum(ideal_gains) if sum(ideal_gains) else 0.0


def evaluate(args: argparse.Namespace, client: OpenAI, papers: list[dict[str, Any]], vectors: torch.Tensor,
             embedding_model: str, neighbours: dict[int, set[int]]) -> dict[str, Any]:
    queries = read_json(args.data_dir / "queries.json")
    if args.split != "all":
        queries = [row for row in queries if row["split"] == args.split]
    judgments = {row["query_id"]: row for row in read_json(args.data_dir / "judgments.json")}
    queries = [row for row in queries if row["query_id"] in judgments]
    if not queries:
        raise ValueError("평가할 질의·판정 쌍이 없습니다.")
    progress("evaluate", f"split={args.split}, queries={len(queries)}, intent_model={args.intent_model}")
    intents: dict[str, dict[str, Any]] = {}
    model_component = cache_component(args.intent_model)
    intent_cache_path = args.data_dir / f"query_intents.{model_component}.json"
    if args.intent_analysis and intent_cache_path.exists() and not args.refresh_query_cache:
        cached_intents = read_json(intent_cache_path)
        if cached_intents.get("model") == args.intent_model:
            intents = cached_intents.get("intents", {})
    # Import the former shared cache once so existing completed work is not lost.
    legacy_intent_path = args.data_dir / "query_intents.json"
    if (args.intent_analysis and not intents and legacy_intent_path.exists()
            and not args.refresh_query_cache):
        legacy = read_json(legacy_intent_path)
        if legacy.get("model") == args.intent_model:
            intents = legacy.get("intents", {})
            write_json_atomic(intent_cache_path, {"model": args.intent_model, "intents": intents})
    if args.intent_analysis:
        pending_intents = [row for row in queries if row["query_id"] not in intents]
        progress("intent", f"cache hit={len(queries) - len(pending_intents)}/{len(queries)}, "
                            f"pending={len(pending_intents)}, workers={args.intent_workers}")
        failures: list[tuple[str, Exception]] = []
        if pending_intents:
            with ThreadPoolExecutor(max_workers=args.intent_workers) as executor:
                futures = {executor.submit(analyse_query_intent, client, args.intent_model, row["query"]): row
                           for row in pending_intents}
                for completed, future in enumerate(as_completed(futures), start=1):
                    row = futures[future]
                    try:
                        intents[row["query_id"]] = future.result()
                        # Main-thread atomic checkpoint: interruption never
                        # discards already completed API responses.
                        write_json_atomic(intent_cache_path,
                                          {"model": args.intent_model, "intents": intents})
                        progress("intent", f"{completed}/{len(pending_intents)} complete: {row['query_id']}")
                    except Exception as exc:
                        failures.append((row["query_id"], exc))
                        progress("intent", f"{completed}/{len(pending_intents)} FAILED: "
                                           f"{row['query_id']} ({exc})")
            if failures:
                failed_ids = ", ".join(query_id for query_id, _ in failures[:5])
                raise RuntimeError(f"의도 분석 {len(failures)}건 실패: {failed_ids}") from failures[0][1]
    else:
        intents = {row["query_id"]: plain_query_intent(row["query"]) for row in queries}
        progress("intent", "disabled; using plain queries")
    cache_name = (f"intent_query_embeddings.{model_component}.pt"
                  if args.intent_analysis else "query_embeddings.pt")
    cache_path = args.query_embedding_cache or args.data_dir / cache_name
    cached: dict[str, torch.Tensor] = {}
    embedding_queries = {row["query_id"]: build_embedding_query(intents[row["query_id"]]) for row in queries}
    if cache_path.exists() and not args.refresh_query_cache:
        value = torch.load(cache_path, weights_only=False, map_location="cpu")
        if value.get("model") == embedding_model:
            cached = {key: vector for key, vector in value.get("vectors", {}).items()
                      if value.get("queries", {}).get(key) == embedding_queries.get(key)}
    pending = [row for row in queries if row["query_id"] not in cached]
    progress("query-embedding", f"cache hit={len(queries) - len(pending)}/{len(queries)}, pending={len(pending)}")
    if pending:
        for start in range(0, len(pending), 128):
            batch = pending[start:start + 128]
            progress("query-embedding", f"API batch {start + 1}-{start + len(batch)}/{len(pending)}")
            values = embed_texts(client, embedding_model, [embedding_queries[row["query_id"]] for row in batch])
            cached.update({row["query_id"]: vector for row, vector in zip(batch, values)})
        torch.save({"model": embedding_model, "queries": embedding_queries, "vectors": cached}, cache_path)
        progress("query-embedding", f"saved: {cache_path.name}")
    all_term_vectors = load_or_embed_intent_terms(
        client, embedding_model, intents, args.data_dir / "intent_term_embeddings.pt",
        args.refresh_query_cache)
    catalog_by_node = {paper["node_idx"]: index for index, paper in enumerate(papers)}
    ks = [k for k in (5, 10, 20) if k <= args.top_k]
    metrics = {k: {"recall": [], "strong_recall": [], "ndcg": []} for k in ks}
    for query_number, query in enumerate(queries, start=1):
        judgment = judgments[query["query_id"]]
        candidate_indices = [catalog_by_node[node] for node in judgment["candidate_node_indices"] if node in catalog_by_node]
        intent = intents[query["query_id"]]
        term_vectors = {term: all_term_vectors[term] for term in intent_terms(intent)}
        ranked = rerank(intent, cached[query["query_id"]], term_vectors, papers, vectors, neighbours,
                        args.top_k, len(candidate_indices), args.facet_candidate_depth,
                        args.dtm_weight, args.graph_diversity_weight,
                        args.must_weight, args.should_weight, args.exclude_penalty,
                        args.semantic_term_threshold, args.exclude_semantic_threshold,
                        args.semantic_temperature, candidate_indices)
        labels = {item["node_idx"]: item["relevance"] for item in judgment["labels"]}
        ranked_labels = [labels[row["node_idx"]] for row in ranked]
        all_labels = list(labels.values())
        for k in ks:
            cutoff = min(k, len(ranked_labels))
            relevant = sum(label >= 1 for label in all_labels)
            strong = sum(label == 2 for label in all_labels)
            metrics[k]["recall"].append(sum(label >= 1 for label in ranked_labels[:cutoff]) / relevant if relevant else 0.0)
            metrics[k]["strong_recall"].append(sum(label == 2 for label in ranked_labels[:cutoff]) / strong if strong else 0.0)
            metrics[k]["ndcg"].append(ndcg_at_k(ranked_labels, cutoff))
        if query_number == 1 or query_number % 10 == 0 or query_number == len(queries):
            progress("ranking", f"{query_number}/{len(queries)}")
    return {"queries": len(queries), "split": args.split, "candidate_pool": 32,
            "intent_analysis": args.intent_analysis, "intent_model": args.intent_model if args.intent_analysis else None,
            "metrics": {f"@{k}": {name: round(sum(values) / len(values), 4) for name, values in value.items()}
                        for k, value in metrics.items()}}


def print_results(query: str, rows: list[dict[str, Any]], papers: list[dict[str, Any]]) -> None:
    print(f"\n질문: {query}\n")
    for rank, row in enumerate(rows, start=1):
        paper = papers[row["catalog_index"]]
        abstract = " ".join(paper["abstract"].split())
        if len(abstract) > 300:
            abstract = abstract[:297].rstrip() + "..."
        print(f"{rank}. {row['title']}")
        print(f"   최종 {row['score']:.3f} | 의미 유사도 {row['text_score']:.3f} "
              f"| 의도 보정 {row['intent_adjustment']:+.3f} | 논문 ID {row['paper_id']}")
        print(f"   {abstract}\n")


def print_intent(intent: dict[str, Any]) -> None:
    print(f"검색 의도: {intent['search_intent']}")
    print(f"재작성 질의: {intent['rewritten_query']}")
    print(f"필수: {', '.join(intent['must_terms']) or '-'}")
    print(f"선호: {', '.join(intent['should_terms']) or '-'}")
    print(f"저중요: {', '.join(intent['low_priority_terms']) or '-'}")
    print(f"제외: {', '.join(intent['exclude_terms']) or '-'}\n")


def print_authors(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("추천 저자:")
    for rank, row in enumerate(rows, start=1):
        evidence = "; ".join(item["title"] for item in row["evidence_papers"])
        print(f"{rank}. {row['author_name']} | 적합도 {row['score']:.3f} "
              f"| 관련 저작 {row['related_paper_count']}편 / 전체 {row['paper_count']}편")
        print(f"   대표 {row['max_component']:.3f} + 반복 전문성 {row['mean_component']:.3f} "
              f"+ 관련 연구량 {row['count_component']:.3f}")
        print(f"   근거 논문: {evidence}")
    print()


def interactive_search(args: argparse.Namespace, client: OpenAI, papers: list[dict[str, Any]], vectors: torch.Tensor,
                       embedding_model: str, neighbours: dict[int, set[int]], authors: dict[int, dict[str, Any]]) -> None:
    print("자연어 논문 추천을 시작합니다. 질문을 입력하세요.")
    print("명령: :quit 또는 :q 종료, :help 도움말\n")
    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n추천을 종료합니다.")
            return
        if query in {":quit", ":q"}:
            print("추천을 종료합니다.")
            return
        if query == ":help":
            print("연구 주제, 문제, 방법을 자연어로 입력하세요. 예: 저전력 FPGA에서 센서 신호 추론을 하는 방법\n")
            continue
        if not query:
            continue
        try:
            if args.intent_analysis:
                progress("intent", "API request started")
                intent = analyse_query_intent(client, args.intent_model, query)
                progress("intent", "complete")
            else:
                intent = plain_query_intent(query)
                progress("intent", "disabled; using plain query")
            progress("embedding", f"API request started ({1 + len(intent_terms(intent))} texts)")
            query_vector, term_vectors = embed_query_plan(client, embedding_model, intent)
            progress("embedding", "complete")
            rows = rerank(intent, query_vector, term_vectors, papers, vectors, neighbours,
                          args.top_k, args.candidate_depth, args.facet_candidate_depth,
                          args.dtm_weight, args.graph_diversity_weight, args.must_weight,
                          args.should_weight, args.exclude_penalty, args.semantic_term_threshold,
                          args.exclude_semantic_threshold, args.semantic_temperature)
            print_intent(intent)
            print_results(query, rows, papers)
            print_authors(recommend_authors(
                intent, query_vector, term_vectors, papers, vectors, authors, args.author_top_k,
                args.author_evidence_count, args.author_relevance_depth, args.author_max_weight,
                args.author_mean_weight, args.author_count_weight, args.author_count_cap, args.dtm_weight,
                args.must_weight, args.should_weight, args.exclude_penalty, args.semantic_term_threshold,
                args.exclude_semantic_threshold, args.semantic_temperature))
        except Exception as exc:
            print(f"검색 요청에 실패했습니다: {exc}\n")


def main() -> None:
    args = parse_args()
    if any(value < 0 for value in (args.dtm_weight, args.graph_diversity_weight, args.must_weight,
                                   args.should_weight, args.exclude_penalty, args.author_max_weight,
                                   args.author_mean_weight, args.author_count_weight)) \
            or args.author_top_k < 0:
        raise ValueError("재정렬 가중치와 author-top-k는 음수일 수 없습니다.")
    if args.author_evidence_count < 1 or args.author_relevance_depth < 1 or args.author_count_cap < 1:
        raise ValueError("저자 추천의 논문 수 관련 옵션은 1 이상이어야 합니다.")
    if args.intent_workers < 1:
        raise ValueError("intent-workers는 1 이상이어야 합니다.")
    if args.facet_candidate_depth < 0:
        raise ValueError("facet-candidate-depth는 음수일 수 없습니다.")
    if not 0.0 <= args.semantic_term_threshold < 1.0:
        raise ValueError("semantic-term-threshold는 0 이상 1 미만이어야 합니다.")
    if not 0.0 <= args.exclude_semantic_threshold < 1.0:
        raise ValueError("exclude-semantic-threshold는 0 이상 1 미만이어야 합니다.")
    if args.semantic_temperature <= 0.0:
        raise ValueError("semantic-temperature는 0보다 커야 합니다.")
    papers, vectors, embedding_model = load_catalog(args.master_path, args.data_dir / "paper_text_embeddings.pt")
    neighbours = load_neighbours(args.citation_path, {paper["node_idx"] for paper in papers})
    authors = (load_authors(args.author_edge_path, args.author_mapping_path, {paper["node_idx"] for paper in papers})
               if args.author_top_k else {})
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 필요합니다.")
    client = OpenAI()
    if args.evaluate:
        if args.query:
            raise SystemExit("--evaluate와 --query는 함께 사용할 수 없습니다.")
        print(json.dumps(evaluate(args, client, papers, vectors, embedding_model, neighbours), ensure_ascii=False, indent=2))
    elif args.query:
        if args.intent_analysis:
            progress("intent", "API request started")
            intent = analyse_query_intent(client, args.intent_model, args.query)
            progress("intent", "complete")
        else:
            intent = plain_query_intent(args.query)
            progress("intent", "disabled; using plain query")
        progress("embedding", f"API request started ({1 + len(intent_terms(intent))} texts)")
        query_vector, term_vectors = embed_query_plan(client, embedding_model, intent)
        progress("embedding", "complete")
        rows = rerank(intent, query_vector, term_vectors, papers, vectors, neighbours,
                      args.top_k, args.candidate_depth, args.facet_candidate_depth,
                      args.dtm_weight, args.graph_diversity_weight, args.must_weight,
                      args.should_weight, args.exclude_penalty, args.semantic_term_threshold,
                      args.exclude_semantic_threshold, args.semantic_temperature)
        author_rows = recommend_authors(
            intent, query_vector, term_vectors, papers, vectors, authors, args.author_top_k,
            args.author_evidence_count, args.author_relevance_depth, args.author_max_weight,
            args.author_mean_weight, args.author_count_weight, args.author_count_cap, args.dtm_weight,
            args.must_weight, args.should_weight, args.exclude_penalty, args.semantic_term_threshold,
            args.exclude_semantic_threshold, args.semantic_temperature)
        print(json.dumps({"query": args.query, "intent": intent, "embedding_model": embedding_model,
                          "results": rows, "recommended_authors": author_rows}, ensure_ascii=False, indent=2))
    else:
        interactive_search(args, client, papers, vectors, embedding_model, neighbours, authors)


if __name__ == "__main__":
    main()
