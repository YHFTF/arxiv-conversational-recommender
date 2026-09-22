"""Conservatively reduce D/T/M sparsity without an LLM/API call.

Raw Luna tags are never deleted.  This script adds normalized and graph-safe
tags to the master file, then builds ``knowledge_meta.json`` only from terms
that occur often enough to learn an embedding.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
KINDS = ("domain", "task", "method")

# Only exact, unambiguous aliases belong here.  Semantic synonym decisions are
# deliberately excluded until a later reviewed/API-assisted phase.
ALIASES = {
    "domain": {"ml": "machine learning", "nlp": "natural language processing", "cv": "computer vision"},
    "task": {},
    "method": {
        "cnn": "convolutional neural network", "cnns": "convolutional neural network",
        "rnn": "recurrent neural network", "rnns": "recurrent neural network",
        "lstm": "long short term memory", "lstms": "long short term memory",
        "gnn": "graph neural network", "gnns": "graph neural network",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API 없이 D/T/M 표기 정규화 및 희소 태그 필터링")
    parser.add_argument("--master-path", type=Path, default=ROOT / "subdataset/arxiv_master_final.json")
    parser.add_argument("--meta-path", type=Path, default=ROOT / "output/knowledge_meta.json")
    parser.add_argument("--mapping-path", type=Path, default=ROOT / "output/knowledge_normalization_map.json")
    parser.add_argument("--report-path", type=Path, default=ROOT / "output/knowledge_normalization_report.json")
    parser.add_argument("--min-frequency", type=int, default=2, help="그래프 지식 노드로 유지할 최소 등장 횟수")
    return parser.parse_args()


def canonicalize(raw: Any, kind: str) -> str:
    value = unicodedata.normalize("NFKC", str(raw)).casefold().strip()
    value = value.replace("&", " and ").replace("_", " ").replace("/", " ")
    value = re.sub(r"[-‐‑–—]+", " ", value)
    value = re.sub(r"[^a-z0-9+ ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = ALIASES[kind].get(value, value)
    words = value.split()
    if words:
        last = words[-1]
        # Conservative English singularization for a final noun only.
        if last.endswith("ies") and len(last) > 4:
            words[-1] = last[:-3] + "y"
        elif last.endswith("s") and len(last) > 3 and not last.endswith(("ss", "us", "is", "ics")):
            words[-1] = last[:-1]
    return " ".join(words)


def unique(values: Any, kind: str) -> list[str]:
    result, seen = [], set()
    for value in values if isinstance(values, list) else []:
        term = canonicalize(value, kind)
        if term and term not in seen:
            result.append(term)
            seen.add(term)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.min_frequency < 1:
        raise ValueError("--min-frequency는 1 이상이어야 합니다.")
    with args.master_path.open(encoding="utf-8") as handle:
        master: list[dict[str, Any]] = json.load(handle)
    counts = {kind: Counter() for kind in KINDS}
    raw_to_normalized = {kind: {} for kind in KINDS}
    for row in master:
        knowledge = row.get("knowledge", {})
        for kind in KINDS:
            normalized = unique(knowledge.get(kind, []), kind)
            for raw in knowledge.get(kind, []) if isinstance(knowledge.get(kind), list) else []:
                raw_to_normalized[kind][str(raw)] = canonicalize(raw, kind)
            counts[kind].update(normalized)
    retained = {kind: {term for term, count in counts[kind].items() if count >= args.min_frequency} for kind in KINDS}
    meta = {f"{kind}s": {term: index for index, term in enumerate(sorted(retained[kind]), start=1)} for kind in KINDS}
    for row in master:
        knowledge = row.setdefault("knowledge", {})
        normalized = {kind: unique(knowledge.get(kind, []), kind) for kind in KINDS}
        knowledge["normalized"] = normalized
        knowledge["graph"] = {kind: [term for term in normalized[kind] if term in retained[kind]] for kind in KINDS}
    mapping = {
        kind: [{"raw": raw, "normalized": normalized, "frequency": counts[kind].get(normalized, 0),
                "retained_for_graph": normalized in retained[kind]}
               for raw, normalized in sorted(raw_to_normalized[kind].items(), key=lambda pair: pair[0].casefold())]
        for kind in KINDS
    }
    report = {
        "min_frequency": args.min_frequency,
        "paper_count": len(master),
        "vocabulary": {
            kind: {"normalized_terms": len(counts[kind]), "graph_terms": len(retained[kind]),
                   "tag_coverage": sum(count for term, count in counts[kind].items() if term in retained[kind]),
                   "total_tags": sum(counts[kind].values())}
            for kind in KINDS
        },
    }
    write_json(args.master_path, master)
    write_json(args.meta_path, meta)
    write_json(args.mapping_path, mapping)
    write_json(args.report_path, report)
    for kind in KINDS:
        data = report["vocabulary"][kind]
        print(f"{kind}: {data['normalized_terms']:,} -> {data['graph_terms']:,} terms | "
              f"coverage={data['tag_coverage']}/{data['total_tags']}")


if __name__ == "__main__":
    main()
