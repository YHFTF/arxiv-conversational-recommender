"""Versioned, input-bound embeddings. Benchmarks never invoke an API."""
from pathlib import Path

import numpy as np

from .core import digest, document_text, load_dataset, normalize_rows, read, write


def inputs(papers, queries, use_dtm):
    return ([document_text(p, use_dtm) for p in papers.values()],
            [q["query"] for q in queries.values()])


def prepare(data, output, include_dtm=True):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "embeddings"))
    from local_embeddings import LocalEmbedder, IDENTITY, INSTRUCTION
    papers, queries, _ = load_dataset(data, need_labels=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoder = LocalEmbedder()
    model = {"identity": IDENTITY, "query_instruction": INSTRUCTION, "backend": "qwen_nf4"}
    _, texts = inputs(papers, queries, False)
    # Reject silent truncation: otherwise appended D/T/M can disappear entirely,
    # or the two arms can contain different portions of the abstract.
    for dtm in ([False, True] if include_dtm else [False]):
        docs, _ = inputs(papers, queries, dtm)
        for pid, text in zip(papers, docs):
            if len(encoder.tokenizer(text)["input_ids"]) > 512:
                raise ValueError(f"Paper {pid} exceeds 512 tokens (dtm={dtm}). Shorten its abstract in papers.json identically for both arms; no silent truncation is allowed.")
    for qid, text in zip(queries, texts):
        if len(encoder.tokenizer(f"Instruct: {INSTRUCTION}\nQuery: {text}")["input_ids"]) > 512:
            raise ValueError(f"Query {qid} exceeds the 512-token limit including instruction")
    # Shared query embeddings across all five arms; query construction never sees qrels or D/T/M.
    query_vectors = encode_resumable(encoder, texts, True, output / "query_rows", model)
    for dtm in ([False, True] if include_dtm else [False]):
        docs, _ = inputs(papers, queries, dtm)
        name = "text_dtm" if dtm else "text"
        vectors = encode_resumable(encoder, docs, False, output / (name + "_rows"), model)
        save_bundle(output / name, papers, queries, docs, texts, vectors, query_vectors, model, dtm)


def encode_resumable(encoder, texts, query, folder, model):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(texts):
        # Cache identity covers exact input, role, prompts, model/revision, quantization and limits.
        path = folder / (digest({"text": text, "query": query, "model": model}) + ".npy")
        if path.exists():
            value = np.load(path, allow_pickle=False)
        else:
            value = encoder.encode([text], query=query).numpy()[0]
            with path.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
        if value.shape != (2560,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid cached embedding: {path}")
        rows.append(value)
        if (i + 1) % 32 == 0 or i + 1 == len(texts):
            print(f"[{'query' if query else 'document'}] {i + 1}/{len(texts)}", flush=True)
    return normalize_rows(np.stack(rows))


def save_bundle(path, papers, queries, docs, texts, vectors, query_vectors, model, dtm):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "vectors.tmp").open("wb") as handle:
        np.savez_compressed(handle, papers=normalize_rows(vectors), queries=normalize_rows(query_vectors))
    (path / "vectors.tmp").replace(path / "vectors.npz")
    from .core import file_hash
    write(path / "manifest.json", {"schema": 1, "model": model, "include_dtm": dtm,
          "paper_ids": list(papers), "query_ids": list(queries),
          "document_hash": digest(docs), "query_hash": digest(texts),
          "vectors_sha256": file_hash(path / "vectors.npz")})


def load_bundle(path, papers, queries, dtm):
    from .core import file_hash
    path = Path(path)
    meta = read(path / "manifest.json")
    docs, texts = inputs(papers, queries, dtm)
    expected = {"schema": 1, "include_dtm": dtm, "paper_ids": list(papers), "query_ids": list(queries),
                "document_hash": digest(docs), "query_hash": digest(texts),
                "vectors_sha256": file_hash(path / "vectors.npz")}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Embedding provenance/input mismatch: {path}")
    with np.load(path / "vectors.npz", allow_pickle=False) as arrays:
        p, q = arrays["papers"], arrays["queries"]
    if p.ndim != 2 or q.ndim != 2 or len(p) != len(papers) or len(q) != len(queries) or p.shape[1] != q.shape[1]:
        raise ValueError("Invalid embedding shape")
    return normalize_rows(p), normalize_rows(q), meta
