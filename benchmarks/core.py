from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

VARIANTS = ("text", "text_dtm", "text_citation", "text_author", "text_citation_author")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip():
        raise ValueError(f"Invalid identifier: {value!r}")
    return str(value)


def unique_rows(rows, key):
    result = {}
    for row in rows:
        value = identifier(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    if not result:
        raise ValueError(f"Empty {key} collection")
    return result


def document_text(paper, dtm=False):
    # No shared legacy formatter: text ablations cannot accidentally inherit D/T/M.
    title, abstract = paper.get("title"), paper.get("abstract")
    if not isinstance(title, str) or not title.strip() or not isinstance(abstract, str) or not abstract.strip():
        raise ValueError("Every paper needs a nonempty title and abstract")
    text = f"Title: {title.strip()}\nAbstract: {abstract.strip()}"
    if dtm:
        knowledge = paper.get("dtm")
        if not isinstance(knowledge, dict):
            raise ValueError("text_dtm needs an explicit dtm object for every paper (empty lists allowed)")
        parts = []
        for key in ("domain", "task", "method"):
            terms = knowledge.get(key, [])
            if not isinstance(terms, list) or any(not isinstance(t, str) for t in terms):
                raise ValueError(f"dtm.{key} must be a list of strings")
            parts.append(f"{key}: " + "; ".join(terms))
        text += "\nD/T/M: " + " | ".join(parts)
    return text


def load_dataset(folder, need_labels=True):
    folder = Path(folder)
    papers = unique_rows(read(folder / "papers.json"), "paper_id")
    queries = unique_rows(read(folder / "queries.json"), "query_id")
    for paper in papers.values():
        document_text(paper)
    for query in queries.values():
        if not isinstance(query.get("query"), str) or not query["query"].strip():
            raise ValueError("Empty query")
        if query.get("split") not in ("train", "validation", "test"):
            raise ValueError("Every query needs split=train/validation/test")
        excluded = [identifier(p) for p in query.get("exclude_paper_ids", [])]
        if not set(excluded) <= papers.keys():
            raise ValueError("Excluded paper is not in corpus")
    labels = unique_rows(read(folder / "qrels.json"), "query_id") if need_labels else {}
    for qid, row in labels.items():
        if qid not in queries:
            raise ValueError(f"Unknown qrels query: {qid}")
        seen = set()
        for item in row["labels"]:
            pid = identifier(item["paper_id"])
            grade = item["relevance"]
            if pid not in papers or pid in seen or type(grade) is not int or not 0 <= grade <= 2:
                raise ValueError("Invalid/duplicate qrel; relevance must be integer 0/1/2")
            if grade > 0 and pid in {identifier(x) for x in queries[qid].get("exclude_paper_ids", [])}:
                raise ValueError("Positive qrel is excluded from retrieval")
            seen.add(pid)
        if not seen:
            raise ValueError("Empty judgments")
        if "candidate_paper_ids" in row:
            pool = [identifier(p) for p in row["candidate_paper_ids"]]
            if len(pool) != len(set(pool)) or set(pool) != seen:
                raise ValueError("Candidate pool must have exactly one judgment for each candidate")
    return papers, queries, labels


def normalize_rows(matrix):
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or (norms <= 0).any():
        raise ValueError("Embedding matrix must have finite nonzero rows")
    return matrix / norms


def stochastic(matrix):
    matrix = matrix.tocsr().astype(np.float64)
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    return sparse.diags(1 / np.where(sums > 0, sums, 1)) @ matrix


class Graph:
    """Relation-normalized walks; authors remain bipartite (no dense paper clique)."""
    def __init__(self, folder, paper_ids, citation_direction="undirected", relations=("citation", "author"),
                 citation_edges=None, author_edges=None):
        index = {pid: i for i, pid in enumerate(paper_ids)}
        n = len(index)
        self.citation = sparse.csr_matrix((n, n))
        self.membership = sparse.csr_matrix((n, 0))
        if "citation" in relations:
            pairs = set()
            for source, target in (read(Path(folder) / "citations.json") if citation_edges is None else citation_edges):
                source, target = identifier(source), identifier(target)
                if source not in index or target not in index:
                    raise ValueError("Citation endpoint outside corpus")
                if source == target:
                    continue
                pairs.add((index[source], index[target]))
                if citation_direction == "undirected":
                    pairs.add((index[target], index[source]))
            if pairs:
                r, c = zip(*sorted(pairs))
                self.citation = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
        if "author" in relations:
            pairs, authors = set(), {}
            for author, paper in (read(Path(folder) / "authorships.json") if author_edges is None else author_edges):
                author, paper = identifier(author), identifier(paper)
                if paper not in index:
                    raise ValueError("Authorship paper outside corpus")
                authors.setdefault(author, len(authors))
                pairs.add((index[paper], authors[author]))
            if pairs:
                r, c = zip(*sorted(pairs))
                self.membership = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, len(authors))).tocsr()
        self.citation = stochastic(self.citation)
        self.paper_author = stochastic(self.membership)
        self.author_paper = stochastic(self.membership.T)
        self.cite_active = np.asarray(self.citation.sum(axis=1)).ravel() > 0
        self.author_active = np.asarray(self.membership.sum(axis=1)).ravel() > 0

    def walk(self, mass, citation, author, seed):
        # Equal weight for each available relation per paper, independent of edge counts.
        counts = citation * self.cite_active.astype(int) + author * self.author_active.astype(int)
        scaled = mass / np.maximum(counts, 1)
        out = np.zeros_like(mass)
        if citation:
            out += self.citation.T @ scaled
        if author:
            out += self.author_paper.T @ (self.paper_author.T @ scaled)
        return out + mass[counts == 0].sum() * seed

    def diffuse(self, seed, citation, author, restart=0.2, tolerance=1e-9, max_iter=200):
        value = seed.copy()
        for step in range(max_iter):
            updated = restart * seed + (1 - restart) * self.walk(value, citation, author, seed)
            if np.abs(updated - value).sum() <= tolerance:
                return updated, step + 1, True
            value = updated
        return value, max_iter, False


def prior_from_scores(scores, temperature):
    values = np.exp((scores - scores.max()) / temperature)
    return values / values.sum()


def metrics(ranked, labels, k):
    relevant = {pid for pid, grade in labels.items() if grade > 0}
    strong = {pid for pid, grade in labels.items() if grade == 2}
    top = ranked[:k]
    gains = [labels.get(pid, 0) for pid in top]
    ideal = sorted(labels.values(), reverse=True)[:k]
    dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(gains))
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return {"recall": len(set(top) & relevant) / len(relevant) if relevant else 0.0,
            "strong_recall": len(set(top) & strong) / len(strong) if strong else 0.0,
            "ndcg": dcg / idcg if idcg else 0.0,
            "mrr": next((1 / (i + 1) for i, pid in enumerate(top) if pid in relevant), 0.0),
            "judged_fraction": sum(pid in labels for pid in top) / len(top) if top else 0.0}
