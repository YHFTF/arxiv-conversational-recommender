"""Extract citation edges whose two endpoints belong to the 16k sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="표본 내부 OGBN-Arxiv 인용 엣지 추출")
    parser.add_argument("--sample-path", type=Path, default=ROOT / "subdataset/ogbn_arxiv_16k_ffs_sample.pt")
    parser.add_argument("--edge-path", type=Path, default=ROOT / "dataset/ogbn_arxiv/raw/edge.csv.gz")
    parser.add_argument("--output-path", type=Path, default=ROOT / "subdataset/paper_paper_edges.pt")
    parser.add_argument("--report-path", type=Path, default=ROOT / "output/paper_paper_edge_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = torch.load(args.sample_path, weights_only=False, map_location="cpu")
    sampled_nodes = {int(node) for node in sample["indices"]}
    if len(sampled_nodes) != len(sample["indices"]):
        raise ValueError("표본 indices에 중복 노드가 있습니다.")

    # OGB raw edge.csv.gz has no header and stores original OGB node indices.
    edges = pd.read_csv(args.edge_path, compression="gzip", header=None, names=["src", "dst"], dtype="int64")
    inside = edges["src"].isin(sampled_nodes) & edges["dst"].isin(sampled_nodes)
    sub_edges = edges.loc[inside, ["src", "dst"]].drop_duplicates().sort_values(["src", "dst"])
    if sub_edges.empty:
        raise RuntimeError("표본 내부 인용 엣지가 0개입니다. 표본과 원본 edge 파일을 확인하세요.")
    edge_index = torch.tensor(sub_edges.to_numpy().T.copy(), dtype=torch.long)
    if not torch.isin(edge_index, torch.tensor(sorted(sampled_nodes))).all():
        raise AssertionError("표본 밖 노드가 인용 엣지에 포함되었습니다.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(edge_index, args.output_path)
    report = {
        "sampled_papers": len(sampled_nodes),
        "source_edges": len(edges),
        "internal_citation_edges": int(edge_index.size(1)),
        "unique_source_papers": int(edge_index[0].unique().numel()),
        "unique_target_papers": int(edge_index[1].unique().numel()),
        "edge_node_ids": "original_ogbn_node_idx",
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"저장: {args.output_path}")


if __name__ == "__main__":
    main()
