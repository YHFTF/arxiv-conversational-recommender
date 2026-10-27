import csv
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from .core import (Graph, VARIANTS, file_hash, identifier, load_dataset,
                   metrics, prior_from_scores, read, write)
from .embeddings import load_bundle

DEFAULTS = {"variants": list(VARIANTS), "split": "test", "mode": "pooled", "ks": [5, 10, 20],
            "seed_k": 50, "temperature": 0.05, "graph_weight": 0.4, "restart": 0.2,
            "tolerance": 1e-9, "max_iter": 200, "citation_direction": "undirected",
            "bootstrap_samples": 1000, "random_seed": 42, "allow_demo": False}


def configuration(path=None):
    supplied = read(path) if path else {}
    if set(supplied) - DEFAULTS.keys():
        raise ValueError(f"Unknown config keys: {set(supplied) - DEFAULTS.keys()}")
    cfg = {**DEFAULTS, **supplied}
    if not cfg["variants"] or len(set(cfg["variants"])) != len(cfg["variants"]) or not set(cfg["variants"]) <= set(VARIANTS):
        raise ValueError("Invalid variants")
    if cfg["split"] not in ("train", "validation", "test", "all") or cfg["mode"] not in ("pooled", "full"):
        raise ValueError("Invalid split or mode")
    if not cfg["ks"] or any(type(k) is not int or k < 1 for k in cfg["ks"]):
        raise ValueError("ks must be positive integers")
    for name in ("seed_k", "max_iter", "bootstrap_samples"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < cfg["restart"] <= 1 or not 0 <= cfg["graph_weight"] <= 1 or not cfg["temperature"] > 0 or not cfg["tolerance"] > 0:
        raise ValueError("Invalid diffusion parameters")
    if cfg["citation_direction"] not in ("undirected", "outgoing"):
        raise ValueError("Invalid citation direction")
    if any(not np.isfinite(cfg[key]) for key in ("restart", "graph_weight", "temperature", "tolerance")):
        raise ValueError("Non-finite configuration")
    return cfg


def run(data, embeddings, output, cfg):
    data, embeddings, output = Path(data), Path(embeddings), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory for each run")
    papers, queries, qrels = load_dataset(data)
    paper_ids = list(papers)
    index = {pid: i for i, pid in enumerate(paper_ids)}
    selected = [(i, qid, q) for i, (qid, q) in enumerate(queries.items())
                if cfg["split"] == "all" or q["split"] == cfg["split"]]
    if not selected or any(qid not in qrels for _, qid, _ in selected):
        raise ValueError("Every selected query needs judgments; no silent skipping")
    bundles = {"text": load_bundle(embeddings / "text", papers, queries, False)}
    if "text_dtm" in cfg["variants"]:
        bundles["text_dtm"] = load_bundle(embeddings / "text_dtm", papers, queries, True)
        if bundles["text"][2]["model"] != bundles["text_dtm"][2]["model"] or not np.array_equal(bundles["text"][1], bundles["text_dtm"][1]):
            raise ValueError("D/T/M ablation must use the same model and exact query vectors")
    if not cfg["allow_demo"] and any(b[2]["model"].get("backend") == "synthetic" for b in bundles.values()):
        raise ValueError("Synthetic vectors require allow_demo=true")
    relations = set()
    if any("citation" in v for v in cfg["variants"]):
        relations.add("citation")
    if any("author" in v for v in cfg["variants"]):
        relations.add("author")
    graph = Graph(data, paper_ids, cfg["citation_direction"], relations)
    records, rankings = [], []
    for qi, qid, query in selected:
        labels = {identifier(x["paper_id"]): x["relevance"] for x in qrels[qid]["labels"]}
        excluded = {identifier(x) for x in query.get("exclude_paper_ids", [])}
        allowed = np.array([pid not in excluded for pid in paper_ids])
        if not allowed.any():
            raise ValueError("Query excludes entire corpus")
        if cfg["mode"] == "pooled":
            if "candidate_paper_ids" not in qrels[qid]:
                raise ValueError("Pooled evaluation requires an explicitly fully judged candidate pool")
            candidates = set(map(identifier, qrels[qid]["candidate_paper_ids"])) - excluded
        else:
            candidates = set(paper_ids) - excluded
        if not candidates:
            raise ValueError("Empty retrieval candidates")
        for variant in cfg["variants"]:
            started = time.perf_counter()
            p, q, _ = bundles["text_dtm" if variant == "text_dtm" else "text"]
            cosine = p @ q[qi]
            # Exclusions cannot seed the walk; the graph itself remains query-independent.
            prior = np.zeros(len(papers))
            prior[allowed] = prior_from_scores(cosine[allowed], cfg["temperature"])
            seed_order = sorted(np.flatnonzero(allowed), key=lambda i: (-prior[i], paper_ids[i]))[:cfg["seed_k"]]
            seed = np.zeros_like(prior)
            seed[seed_order] = prior[seed_order]
            seed /= seed.sum()
            iterations, converged = 0, True
            scores = prior
            if variant not in ("text", "text_dtm"):
                propagated, iterations, converged = graph.diffuse(
                    seed, "citation" in variant, "author" in variant,
                    cfg["restart"], cfg["tolerance"], cfg["max_iter"])
                scores = (1 - cfg["graph_weight"]) * prior + cfg["graph_weight"] * propagated
            ranking = sorted(candidates, key=lambda pid: (-scores[index[pid]], pid))
            elapsed = (time.perf_counter() - started) * 1000
            rankings.append({"query_id": qid, "variant": variant, "converged": converged,
                             "iterations": iterations, "latency_ms": elapsed,
                             "results": [{"paper_id": pid, "score": float(scores[index[pid]]),
                                          "cosine": float(cosine[index[pid]])}
                                         for pid in ranking[:max(cfg["ks"])]]})
            for k in cfg["ks"]:
                records.append({"query_id": qid, "variant": variant, "k": k,
                                "positive_count": sum(v > 0 for v in labels.values()),
                                **metrics(ranking, labels, k)})
    summary, comparisons = summarize(records, cfg)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(output / "rankings.json", rankings)
    write(output / "summary.json", {"config": cfg, "queries": len(selected), "papers": len(papers),
          "metrics": summary, "paired_deltas_vs_text": comparisons,
          "nonconverged": sum(not r["converged"] for r in rankings),
          "zero_positive_queries": sum(not any(x["relevance"] > 0 for x in qrels[qid]["labels"]) for _, qid, _ in selected),
          "evaluation_note": "Pooled reranking" if cfg["mode"] == "pooled" else "Full corpus; unjudged treated as zero; recall denominator is known positives only"})
    names = ["papers.json", "queries.json", "qrels.json"]
    names += [name for name in ("citations.json", "authorships.json", "graph_manifest.json") if (data / name).exists()]
    write(output / "run_manifest.json", {"schema": 1, "config": cfg,
          "graph_method": "query-seeded personalized PageRank; not a trained GNN",
          "input_sha256": {name: file_hash(data / name) for name in names},
          "implementation_sha256": {name: file_hash(Path(__file__).with_name(name))
                                    for name in ("core.py", "runner.py", "embeddings.py")},
          "embedding_manifests": {key: value[2] for key, value in bundles.items()},
          "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}})
    print(f"Completed {len(selected)} queries x {len(cfg['variants'])} variants: {output}")


def summarize(records, cfg):
    summary, comparisons = {}, {}
    rng = np.random.default_rng(cfg["random_seed"])
    for variant in cfg["variants"]:
        summary[variant] = {}
        for k in cfg["ks"]:
            rows = [r for r in records if r["variant"] == variant and r["k"] == k]
            names = ("recall", "strong_recall", "ndcg", "mrr", "judged_fraction")
            summary[variant][str(k)] = {name: float(np.mean([r[name] for r in rows])) for name in names}
            if variant != "text" and "text" in cfg["variants"]:
                base = {r["query_id"]: r for r in records if r["variant"] == "text" and r["k"] == k}
                for name in ("recall", "ndcg", "mrr"):
                    delta = np.array([r[name] - base[r["query_id"]][name] for r in rows])
                    # Paired query bootstrap; bounded memory independent of sample count.
                    samples = [float(rng.choice(delta, len(delta), replace=True).mean()) for _ in range(cfg["bootstrap_samples"])]
                    comparisons[f"{variant}/{name}@{k}"] = {"mean_delta": float(delta.mean()),
                        "ci95": list(map(float, np.quantile(samples, [0.025, 0.975])))}
    return summary, comparisons
