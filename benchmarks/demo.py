"""Synthetic wiring fixture, explicitly excluded from real benchmark results."""
from pathlib import Path
import numpy as np

from .core import load_dataset, write
from .embeddings import inputs, save_bundle
from .runner import DEFAULTS


def create(folder):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Demo destination must be empty")
    data, embeddings = folder / "data", folder / "embeddings"
    write(data / "papers.json", [
        {"paper_id": f"p{i}", "title": title, "abstract": abstract,
         "dtm": {"domain": [domain], "task": [], "method": []}}
        for i, title, abstract, domain in [
            (0, "Graph paper retrieval", "Neural graph methods for finding papers.", "retrieval"),
            (1, "Citation recommendation", "Finding related research using citations.", "retrieval"),
            (2, "Marine biology", "The ecology of coral reefs.", "biology"),
            (3, "Author research networks", "Researcher relationships for paper discovery.", "retrieval"),
            (4, "Isolated document", "An unconnected scientific document.", "other")]])
    write(data / "queries.json", [
        {"query_id": "q0", "query": "Find papers about graph retrieval", "split": "test"},
        {"query_id": "q1", "query": "Research on marine biology", "split": "test"}])
    write(data / "qrels.json", [
        {"query_id": "q0", "candidate_paper_ids": [f"p{i}" for i in range(5)],
         "labels": [{"paper_id": f"p{i}", "relevance": g} for i, g in enumerate([2, 2, 0, 1, 0])]},
        {"query_id": "q1", "candidate_paper_ids": [f"p{i}" for i in range(5)],
         "labels": [{"paper_id": f"p{i}", "relevance": g} for i, g in enumerate([0, 0, 2, 0, 0])]}])
    write(data / "citations.json", [["p0", "p1"], ["p1", "p3"]])
    write(data / "authorships.json", [["a0", "p0"], ["a0", "p3"], ["a1", "p1"], ["a2", "p2"]])
    write(data / "graph_manifest.json", {"scope": "synthetic", "note": "No real performance conclusions"})
    papers, queries, _ = load_dataset(data)
    p = np.array([[1, 0, 0], [.7, .3, 0], [0, 1, 0], [.6, .4, 0], [0, 0, 1]], dtype=np.float32)
    q = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    model = {"identity": "synthetic-fixture-v1", "backend": "synthetic"}
    for dtm in (False, True):
        docs, texts = inputs(papers, queries, dtm)
        vectors = p.copy()
        if dtm:
            vectors[1] = [.9, .1, 0]
        save_bundle(embeddings / ("text_dtm" if dtm else "text"), papers, queries, docs, texts, vectors, q, model, dtm)
    write(folder / "config.json", {**DEFAULTS, "allow_demo": True, "ks": [1, 3, 5], "seed_k": 2})
    print(f"Synthetic fixture created: {folder}")
