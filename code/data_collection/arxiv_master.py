"""Map balanced-sample papers to OpenAlex metadata and Luna D/T/M output."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="16k 표본과 Luna D/T/M을 정렬해 마스터를 생성")
    parser.add_argument("--sample-path", type=Path, default=ROOT / "subdataset/ogbn_arxiv_16k_ffs_sample.pt")
    parser.add_argument("--text-path", type=Path, default=ROOT / "subdataset/titleabs.tsv")
    parser.add_argument("--author-path", type=Path, default=ROOT / "output/author_data_openalex.json")
    parser.add_argument("--dtm-path", type=Path, default=ROOT / "output/dtm_luna_16k/gpt-5.6-luna.json")
    parser.add_argument("--master-path", type=Path, default=ROOT / "subdataset/arxiv_master_final.json")
    parser.add_argument("--meta-path", type=Path, default=ROOT / "output/knowledge_meta.json")
    parser.add_argument("--report-path", type=Path, default=ROOT / "output/knowledge_mapping_report.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def unique_terms(value: Any) -> list[str]:
    result, seen = [], set()
    for item in value if isinstance(value, list) else []:
        term = str(item).strip()
        if term and term.casefold() not in seen:
            result.append(term)
            seen.add(term.casefold())
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_texts(path: Path, wanted_ids: set[str]) -> dict[str, tuple[str, str]]:
    texts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            paper_id, sep, content = line.rstrip("\n").partition("\t")
            if not sep or paper_id not in wanted_ids:
                continue
            title, sep, abstract = content.partition("\t")
            if sep and title.strip() and abstract.strip():
                texts[paper_id] = (title.strip(), abstract.strip())
    return texts


def vocabulary(values: list[str]) -> dict[str, int]:
    return {term: i for i, term in enumerate(sorted(set(values), key=str.casefold), start=1)}


def main() -> None:
    args = parse_args()
    sample = torch.load(args.sample_path, weights_only=False, map_location="cpu")
    sample_nodes = [int(node) for node in sample["indices"]]
    if len(sample_nodes) != len(set(sample_nodes)):
        raise ValueError("표본에 중복 node_idx가 있습니다.")
    expected = set(sample_nodes)
    authors = {int(row["node_idx"]): row for row in read_json(args.author_path)}
    dtm_rows = {int(row["node_idx"]): row for row in read_json(args.dtm_path)}
    for name, rows in (("OpenAlex", authors), ("Luna D/T/M", dtm_rows)):
        missing, unexpected = expected - set(rows), set(rows) - expected
        if missing or unexpected:
            raise ValueError(f"{name}와 표본 불일치: missing={len(missing)}, unexpected={len(unexpected)}")
    paper_ids = {str(authors[node]["paper_id"]).strip() for node in sample_nodes}
    texts = load_texts(args.text_path, paper_ids)
    if missing := paper_ids - set(texts):
        raise ValueError(f"제목·초록 누락 {len(missing)}편: {sorted(missing)[:3]}")

    master, all_domains, all_tasks, all_methods = [], [], [], []
    statuses: Counter[str] = Counter()
    for local_idx, node_idx in enumerate(sample_nodes):
        author, dtm = authors[node_idx], dtm_rows[node_idx]
        paper_id = str(author["paper_id"]).strip()
        status = str(dtm.get("status", "missing"))
        source = dtm.get("knowledge", {}) if status == "success" else {}
        knowledge = {key: unique_terms(source.get(key, [])) for key in ("domain", "task", "method")}
        all_domains.extend(knowledge["domain"])
        all_tasks.extend(knowledge["task"])
        all_methods.extend(knowledge["method"])
        title, abstract = texts[paper_id]
        master.append({
            "local_idx": local_idx, "node_idx": node_idx, "paper_id": paper_id,
            "title": title, "abstract": abstract, "authors": author.get("authors", []),
            "knowledge": knowledge,
            "dtm": {"model": dtm.get("model"), "status": status, "error": dtm.get("error")},
        })
        statuses[status] += 1
    meta = {"domains": vocabulary(all_domains), "tasks": vocabulary(all_tasks), "methods": vocabulary(all_methods)}
    report = {
        "paper_count": len(master),
        "sample_order_preserved": [row["node_idx"] for row in master] == sample_nodes,
        "dtm_status_counts": dict(statuses),
        "empty_knowledge_papers": sum(not any(row["knowledge"].values()) for row in master),
        "vocabulary_counts": {key: len(value) for key, value in meta.items()},
    }
    write_json(args.master_path, master)
    write_json(args.meta_path, meta)
    write_json(args.report_path, report)
    print(f"마스터: {len(master):,}편 | 표본 순서 보존={report['sample_order_preserved']}")
    print(f"D/T/M: {dict(statuses)} | 빈 지식: {report['empty_knowledge_papers']}")
    print(f"사전: D={len(meta['domains']):,}, T={len(meta['tasks']):,}, M={len(meta['methods']):,}")


if __name__ == "__main__":
    main()
