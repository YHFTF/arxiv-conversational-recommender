"""OpenAlex 저자 수집 결과에서 저자–논문 연결을 생성한다."""
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "subdataset" / "ogbn_arxiv_16k_ffs_sample.pt"
AUTHOR_DATA_PATH = ROOT / "output" / "author_data_openalex.json"
EDGE_PATH = ROOT / "subdataset" / "author_paper_edges.pt"
AUTHOR_MAPPING_PATH = ROOT / "output" / "author_remapping.json"


def main() -> None:
    sample = torch.load(SAMPLE_PATH, weights_only=False, map_location="cpu")
    sample_nodes = {int(node) for node in sample["indices"]}
    with AUTHOR_DATA_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    records_by_node = {int(record["node_idx"]): record for record in records}
    missing = sample_nodes - set(records_by_node)
    unexpected = set(records_by_node) - sample_nodes
    if missing or unexpected:
        raise ValueError(
            f"저자 데이터가 새 표본과 일치하지 않습니다: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    # OpenAlex author ID를 그래프의 안정적인 저자 노드 ID로 사용한다.
    author_names: dict[str, str] = {}
    pairs: set[tuple[str, int]] = set()
    for node_idx, record in records_by_node.items():
        for author in record.get("authors", []):
            author_id = str(author.get("author_id") or "").strip()
            if not author_id:
                continue
            author_names.setdefault(author_id, str(author.get("author_name") or "Unknown"))
            pairs.add((author_id, node_idx))

    if not pairs:
        raise RuntimeError("저자–논문 연결이 없습니다.")

    author_ids = sorted(author_names)
    author_to_local = {author_id: index for index, author_id in enumerate(author_ids)}
    ordered_pairs = sorted(pairs, key=lambda pair: (author_to_local[pair[0]], pair[1]))
    edge_index = torch.tensor(
        [[author_to_local[author_id], node_idx] for author_id, node_idx in ordered_pairs], dtype=torch.long
    ).t().contiguous()

    EDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(edge_index, EDGE_PATH)
    with AUTHOR_MAPPING_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "id_to_idx": author_to_local,
                "idx_to_id": {index: author_id for author_id, index in author_to_local.items()},
                "idx_to_name": {index: author_names[author_id] for author_id, index in author_to_local.items()},
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"저자 데이터: {len(records_by_node):,}편")
    print(f"고유 저자: {len(author_ids):,}명")
    print(f"저자–논문 연결: {edge_index.size(1):,}개")
    print(f"저자 없는 논문: {sum(not record.get('authors') for record in records_by_node.values()):,}편")
    print(f"저장: {EDGE_PATH}")
    print(f"저장: {AUTHOR_MAPPING_PATH}")


if __name__ == "__main__":
    main()
