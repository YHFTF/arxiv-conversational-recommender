"""Unlabelled, qualitative comparison using the existing legacy Qwen catalog."""
from pathlib import Path
import sys

import numpy as np

from .core import Graph, digest, file_hash, normalize_rows, prior_from_scores, read, write
from .runner import configuration

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"text": "임베딩만", "text_citation": "임베딩 + 인용",
           "text_author": "임베딩 + 저자", "text_citation_author": "임베딩 + 인용 + 저자"}


def load_existing(master_path, embedding_path, citation_path, author_path):
    import torch
    sys.path.insert(0, str(ROOT / "code" / "embeddings"))
    from local_embeddings import IDENTITY
    embedded = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if embedded.get("model") != IDENTITY:
        raise ValueError("This adapter requires the existing versioned Qwen NF4 catalog; model mismatch")
    ids = list(map(str, embedded["paper_ids"]))
    nodes = list(map(int, embedded["node_indices"].tolist()))
    vectors = normalize_rows(embedded["embeddings"].float().numpy())
    if len(set(ids)) != len(ids) or len(set(nodes)) != len(nodes) or len(nodes) != len(ids) or len(ids) != len(vectors) or vectors.shape[1] != 2560:
        raise ValueError("Invalid existing catalog IDs/embedding dimensions")
    master_rows = read(master_path)
    master = {int(row["node_idx"]): row for row in master_rows}
    if len(master) != len(master_rows):
        raise ValueError("Duplicate master node_idx")
    papers = []
    for node, pid in zip(nodes, ids):
        if node not in master or str(master[node]["paper_id"]) != pid:
            raise ValueError(f"Master/catalog alignment mismatch at node {node}")
        papers.append(master[node])
    node_to_id = dict(zip(nodes, ids))
    def edges(path, author=False):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 2 or value.dtype not in (torch.int32, torch.int64) or (value < 0).any():
            raise ValueError(f"Expected nonnegative integer [2,E] graph tensor: {path}")
        pairs = value.T.tolist()
        kept = [(f"author:{source}" if author else node_to_id[source], node_to_id[target])
                for source, target in pairs if target in node_to_id and (author or source in node_to_id)]
        return kept, {"input_edges": len(pairs), "kept_edges": len(kept), "outside_catalog": len(pairs) - len(kept)}
    citations, cs = edges(citation_path)
    authors, aus = edges(author_path, True)
    graph = Graph(None, ids, citation_edges=citations, author_edges=authors)
    stats = {"papers": len(ids), "citation": cs, "authorship": aus,
             "papers_with_citations": int(graph.cite_active.sum()),
             "papers_with_authors": int(graph.author_active.sum()),
             "authors": graph.membership.shape[1]}
    return papers, ids, vectors, graph, stats, embedded["model"]


def rank_methods(cosine, graph, ids, cfg, top_k):
    prior = prior_from_scores(cosine, cfg["temperature"])
    baseline = sorted(range(len(ids)), key=lambda i: (-prior[i], ids[i]))
    seed = np.zeros(len(ids))
    picked = baseline[:cfg["seed_k"]]
    seed[picked] = prior[picked]
    seed /= seed.sum()
    baseline_ranks = {index: i + 1 for i, index in enumerate(baseline)}
    results = {}
    for method in METHODS:
        iterations, converged = 0, True
        propagated = np.zeros(len(ids))
        if method != "text":
            propagated, iterations, converged = graph.diffuse(
                seed, "citation" in method, "author" in method,
                cfg["restart"], cfg["tolerance"], cfg["max_iter"])
            scores = (1 - cfg["graph_weight"]) * prior + cfg["graph_weight"] * propagated
        else:
            scores = prior
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))[:top_k]
        results[method] = {"name": METHODS[method], "iterations": iterations, "converged": converged,
                           "overlap_with_baseline": len(set(order) & set(baseline[:top_k])),
                           "rows": [{"index": i, "paper_id": ids[i], "rank": rank + 1,
                                     "baseline_rank": baseline_ranks[i],
                                     "rank_change": baseline_ranks[i] - rank - 1,
                                     "cosine": float(cosine[i]), "score": float(scores[i]),
                                     "text_prior": float(prior[i]), "graph_probability": float(propagated[i])}
                                    for rank, i in enumerate(order)]}
    return results


def compare(args):
    papers, ids, vectors, graph, stats, identity = load_existing(
        args.master, args.embeddings, args.citations, args.authors)
    if args.top_k < 1 or args.top_k > len(ids):
        raise ValueError("top-k must be between 1 and corpus size")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    queries = list(args.query or [])
    if args.queries_json:
        rows = read(args.queries_json)
        queries += [row if isinstance(row, str) else row["query"] for row in rows]
    if not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("Supply at least one nonempty --query or --queries-json")
    queries = list(dict.fromkeys(q.strip() for q in queries))
    cfg = configuration(args.config)
    cache_dir = ROOT / "output" / "benchmark_v3_query_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = None
    reports = []
    for query in queries:
        cache = cache_dir / (digest({"model": identity, "query": query, "role": "query"}) + ".npy")
        if cache.exists():
            q = np.load(cache, allow_pickle=False)
        else:
            if encoder is None:
                from local_embeddings import LocalEmbedder
                encoder = LocalEmbedder()
            q = encoder.encode([query], query=True).numpy()[0]
            with cache.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, q, allow_pickle=False)
            cache.with_suffix(".tmp").replace(cache)
        if q.shape != (vectors.shape[1],):
            raise ValueError("Invalid cached query dimension")
        cosine = vectors @ normalize_rows(q[None, :])[0]
        methods = rank_methods(cosine, graph, ids, cfg, args.top_k)
        for method in methods.values():
            for row in method["rows"]:
                index = row.pop("index")
                paper = papers[index]
                row.update(title=paper["title"], abstract=paper.get("abstract", ""),
                           authors=paper.get("authors", []), dtm=paper.get("knowledge", {}),
                           node_idx=int(paper["node_idx"]),
                           citation_neighbors=int(graph.citation.getrow(index).nnz),
                           graph_authors=int(graph.membership.getrow(index).nnz))
        reports.append({"query": query, "methods": methods})
        print(f"Compared 4 methods: {query}", flush=True)
    payload = {"schema": 1, "model": identity, "input_description": "기존 제목·초록+D/T/M 입력 / 512토큰 제한, 뒤쪽 잘림 가능",
               "limitations": ["정답 라벨 없음: Recall/NDCG를 계산하지 않음", "D/T/M 포함 여부 비교는 불가능: 현재 임베딩 한 종류만 사용",
                               "그래프는 학습 없는 PageRank 전파이며 GNN이 아님", "네 방법 모두 동일한 기존 D/T/M 포함 임베딩 사용"],
               "graph_stats": stats, "config": cfg, "top_k": args.top_k, "queries": reports,
               "input_sha256": {name: file_hash(path) for name, path in
                   [("master", args.master), ("embeddings", args.embeddings), ("citations", args.citations), ("authors", args.authors)]}}
    output.mkdir(parents=True, exist_ok=True)
    write(output / "comparison.json", payload)
    from .report import render
    render(output / "report.html", payload)
    print(f"Report: {(output / 'report.html').resolve()}", flush=True)
