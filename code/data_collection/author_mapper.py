"""균형 표본 논문의 OpenAlex 저자 정보를 안정 ID로 수집한다."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "subdataset" / "ogbn_arxiv_16k_ffs_sample.pt"
MAPPING_PATHS = [
    ROOT / "dataset" / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv.gz",
    ROOT / "dataset" / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv",
]
DEFAULT_OUTPUT = ROOT / "output" / "author_data_openalex.json"
API_URL = "https://api.openalex.org/works"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex 저자 수집")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--mailto", default=os.getenv("OPENALEX_MAILTO", ""))
    parser.add_argument("--limit", type=int, help="연결 검증용 최대 논문 수")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_targets(limit: int | None) -> list[tuple[int, str]]:
    mapping_path = next((path for path in MAPPING_PATHS if path.exists()), None)
    if mapping_path is None:
        raise FileNotFoundError("nodeidx2paperid.csv(.gz)를 찾을 수 없습니다.")
    sample = torch.load(SAMPLE, weights_only=False, map_location="cpu")
    indices = [int(index) for index in sample["indices"]]
    mapping = pd.read_csv(mapping_path)
    mapping.columns = [str(column).strip().lower() for column in mapping.columns]
    mapping = mapping.rename(columns={"node idx": "node_idx", "paper id": "paper_id"})
    if not {"node_idx", "paper_id"}.issubset(mapping.columns):
        raise ValueError(f"매핑 파일 컬럼 오류: {mapping.columns.tolist()}")
    node_to_paper = dict(zip(mapping.node_idx.astype(int), mapping.paper_id.astype(str)))
    missing = [node for node in indices if node not in node_to_paper]
    if missing:
        raise ValueError(f"표본 노드 {len(missing)}개의 paper_id가 없습니다.")
    targets = [(node, node_to_paper[node]) for node in indices]
    return targets[:limit] if limit else targets


def load_records(path: Path, overwrite: bool) -> dict[int, dict[str, Any]]:
    if overwrite or not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return {int(record["node_idx"]): record for record in json.load(file)}


def save_records(path: Path, records: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump([records[key] for key in sorted(records)], file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def parse_authors(authorships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    authors = []
    for position, authorship in enumerate(authorships, start=1):
        author = authorship.get("author") or {}
        author_id = author.get("id")
        if author_id:
            authors.append({
                "author_id": str(author_id),
                "author_name": author.get("display_name") or "Unknown",
                "author_position": authorship.get("author_position") or position,
            })
    return authors


def fetch_batch(session: requests.Session, batch: list[tuple[int, str]], mailto: str) -> dict[int, list[dict[str, Any]]]:
    requested = {str(paper_id): node for node, paper_id in batch}
    params = {"filter": "ids.mag:" + "|".join(requested), "per-page": len(batch), "select": "ids,authorships"}
    if mailto:
        params["mailto"] = mailto
    for attempt in range(4):
        try:
            response = session.get(API_URL, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            result = {node: [] for node, _ in batch}
            for work in response.json().get("results", []):
                mag_id = str((work.get("ids") or {}).get("mag") or "")
                if mag_id in requested:
                    result[requested[mag_id]] = parse_authors(work.get("authorships") or [])
            return result
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"OpenAlex 요청 실패: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError("OpenAlex 재시도 한도 초과")


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= 100:
        raise SystemExit("--batch-size는 1~100 사이여야 합니다.")
    if not SAMPLE.exists():
        raise SystemExit(f"샘플 파일 없음: {SAMPLE}")
    targets = load_targets(args.limit)
    records = load_records(args.output, args.overwrite)
    pending = [(node, paper) for node, paper in targets if node not in records]
    print(f"저자 수집: 전체 {len(targets):,}편 / 남은 작업 {len(pending):,}편")
    with requests.Session() as session:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            authors_by_node = fetch_batch(session, batch, args.mailto)
            for node, paper_id in batch:
                records[node] = {"node_idx": node, "paper_id": paper_id, "authors": authors_by_node[node]}
            save_records(args.output, records)
            done = min(start + len(batch), len(pending))
            found = sum(bool(records[node]["authors"]) for node, _ in targets if node in records)
            print(f"  {done:,}/{len(pending):,} 완료 | 저자 매칭 {found:,}/{len(targets):,}")
            time.sleep(0.15)
    selected = {node for node, _ in targets}
    if selected - set(records):
        raise RuntimeError("일부 표본의 저자 결과가 저장되지 않았습니다.")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
