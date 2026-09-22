"""동일한 논문 표본에서 D/T/M 추출 모델을 비교한다.

기본 실행은 gpt-5-nano, gpt-4o-mini, gpt-5.6-luna 각각에 같은 30편을
보낸다. 결과는 모델별로 체크포인트 저장되므로 중단 후 같은 명령으로 재개한다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "subdataset" / "ogbn_arxiv_16k_ffs_sample.pt"
MAPPING_PATH = ROOT / "dataset" / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv.gz"
TEXT_PATH = ROOT / "subdataset" / "titleabs.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "dtm_model_comparison"
MODELS = ("gpt-5-nano", "gpt-4o-mini", "gpt-5.6-luna")

SYSTEM_PROMPT = """You are a careful academic metadata annotator building a reusable knowledge graph vocabulary.
Extract only concepts explicitly supported by the supplied title and abstract. Return concise, canonical English terms, not explanatory phrases.
Do not infer a method merely because it is common in the field.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D/T/M 모델 비교 실험")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--prepare-only", action="store_true", help="표본/검토 파일만 만들고 API를 호출하지 않음")
    parser.add_argument("--overwrite", action="store_true", help="기존 모델 결과를 무시하고 재호출")
    return parser.parse_args()


def normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def validate_dtm(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError("JSON object가 아닙니다.")
    return {key: normalize_list(value.get(key)) for key in ("domain", "task", "method")}


def load_papers(sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("--sample-size는 1 이상이어야 합니다.")
    sample = torch.load(SAMPLE_PATH, weights_only=False, map_location="cpu")
    sampled_nodes = [int(node) for node in sample["indices"]]
    if sample_size > len(sampled_nodes):
        raise ValueError(f"표본 크기는 최대 {len(sampled_nodes)}입니다.")

    mapping = pd.read_csv(MAPPING_PATH)
    mapping.columns = [str(column).strip().lower() for column in mapping.columns]
    mapping = mapping.rename(columns={"node idx": "node_idx", "paper id": "paper_id"})
    node_to_paper = dict(zip(mapping.node_idx.astype(int), mapping.paper_id.astype(str)))
    selected_nodes = sorted(random.Random(seed).sample(sampled_nodes, sample_size))
    selected_paper_ids = {node_to_paper[node] for node in selected_nodes}

    texts: dict[str, tuple[str, str]] = {}
    with TEXT_PATH.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) >= 2 and parts[0] in selected_paper_ids:
                texts[parts[0]] = (parts[1], parts[2] if len(parts) == 3 else "")
    papers = []
    for node_idx in selected_nodes:
        paper_id = node_to_paper[node_idx]
        title, abstract = texts.get(paper_id, ("", ""))
        if not title or not abstract:
            raise ValueError(f"Title/Abstract 매핑 실패: node_idx={node_idx}, paper_id={paper_id}")
        papers.append({"node_idx": node_idx, "paper_id": paper_id, "title": title, "abstract": abstract})
    return papers


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_or_create_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = args.output_dir / "sample_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest["sample_size"] != args.sample_size or manifest["seed"] != args.seed:
            raise ValueError("기존 manifest의 seed/sample-size가 다릅니다. 다른 output-dir를 사용하세요.")
        return manifest["papers"]
    papers = load_papers(args.sample_size, args.seed)
    signature = hashlib.sha256(
        "".join(f"{paper['node_idx']}:{paper['paper_id']}" for paper in papers).encode()
    ).hexdigest()
    save_json(manifest_path, {"sample_size": args.sample_size, "seed": args.seed, "signature": signature, "papers": papers})
    return papers


def prompt_for(paper: dict[str, Any], max_chars: int) -> str:
    content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"[:max_chars]
    return f"""Extract Domain, Task, and Method from this paper for a controlled vocabulary.

Rules:
- Each tag must be a canonical noun phrase of 1-4 words.
- Return no duplicate, near-duplicate, parent/child, explanatory, or sentence-like tags.
- Do not include datasets, benchmarks, time complexity, results, evaluation procedures, or generic words such as "analysis", "study", "approach", or "algorithm" alone.
- domain: 1-2 broad research fields.
- task: 1-2 concrete research problems or objectives.
- method: 0-2 explicitly named techniques, models, or algorithms only. Return [] if none is explicitly named.
- Use [] when the source does not support a field; never invent terms.

Return only this JSON object:
{{"domain": ["..."], "task": ["..."], "method": ["..."]}}

Paper:
{content}"""


def load_results(path: Path, overwrite: bool) -> dict[int, dict[str, Any]]:
    if overwrite or not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return {int(row["node_idx"]): row for row in json.load(file)}


async def extract_one(client: AsyncOpenAI, model: str, paper: dict[str, Any], max_chars: int) -> dict[str, Any]:
    started = time.perf_counter()
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for(paper, max_chars)},
        ],
        "response_format": {"type": "json_object"},
    }
    # GPT-5 Chat Completions uses max_completion_tokens; gpt-4o-mini keeps max_tokens.
    if model.startswith("gpt-5"):
        request["max_completion_tokens"] = 220
        # gpt-5-nano는 ``none`` 대신 ``minimal``부터 지원한다.
        request["reasoning_effort"] = "minimal" if model == "gpt-5-nano" else "none"
    else:
        request["temperature"] = 0
        request["max_tokens"] = 220
    try:
        response = await client.chat.completions.create(**request)
        raw = response.choices[0].message.content or ""
        knowledge = validate_dtm(json.loads(raw))
        usage = response.usage
        return {
            **paper,
            "model": model,
            "status": "success",
            "knowledge": knowledge,
            "raw_response": raw,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            },
        }
    except Exception as exc:
        return {**paper, "model": model, "status": "error", "error": str(exc), "latency_seconds": round(time.perf_counter() - started, 3)}


def save_model_results(path: Path, results: dict[int, dict[str, Any]]) -> None:
    save_json(path, [results[node] for node in sorted(results)])


async def run_model(client: AsyncOpenAI, model: str, papers: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.output_dir / f"{model}.json"
    results = load_results(path, args.overwrite)
    # 오류는 체크포인트에 남기되 다음 실행에서 자동 재시도한다.
    pending = [paper for paper in papers if results.get(paper["node_idx"], {}).get("status") != "success"]
    print(f"[{model}] 기존 {len(results)}/{len(papers)}, 남은 {len(pending)}")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(paper: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await extract_one(client, model, paper, args.max_chars)

    for offset in range(0, len(pending), args.concurrency):
        batch = pending[offset:offset + args.concurrency]
        for result in await asyncio.gather(*(guarded(paper) for paper in batch)):
            results[result["node_idx"]] = result
        save_model_results(path, results)
        print(f"[{model}] {min(offset + len(batch), len(pending))}/{len(pending)} 완료")
    return [results[paper["node_idx"]] for paper in papers]


def terms(row: dict[str, Any], key: str) -> set[str]:
    return {term.lower() for term in row.get("knowledge", {}).get(key, [])}


def build_reports(args: argparse.Namespace, papers: list[dict[str, Any]]) -> None:
    by_model = {model: load_results(args.output_dir / f"{model}.json", False) for model in args.models}
    summary: dict[str, Any] = {"sample_size": len(papers), "models": {}}
    for model, rows in by_model.items():
        values = list(rows.values())
        successful = [row for row in values if row.get("status") == "success"]
        summary["models"][model] = {
            "completed": len(values), "success": len(successful), "errors": len(values) - len(successful),
            "success_rate": round(len(successful) / len(papers), 4),
            "avg_latency_seconds": round(sum(row.get("latency_seconds", 0) for row in successful) / max(len(successful), 1), 3),
            "token_usage": {field: sum(row.get("usage", {}).get(field, 0) for row in successful) for field in ("input_tokens", "output_tokens", "total_tokens")},
            "non_empty_rate": {key: round(sum(bool(terms(row, key)) for row in successful) / max(len(successful), 1), 4) for key in ("domain", "task", "method")},
            "unique_terms": {key: len(set().union(*(terms(row, key) for row in successful))) for key in ("domain", "task", "method")},
        }
    save_json(args.output_dir / "summary.json", summary)

    columns = ["node_idx", "paper_id", "title"] + [f"{model}_{key}" for model in args.models for key in ("domain", "task", "method", "status")]
    with (args.output_dir / "review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for paper in papers:
            row = {key: paper[key] for key in ("node_idx", "paper_id", "title")}
            for model in args.models:
                result = by_model[model].get(paper["node_idx"], {})
                for key in ("domain", "task", "method"):
                    row[f"{model}_{key}"] = " | ".join(result.get("knowledge", {}).get(key, []))
                row[f"{model}_status"] = result.get("status", "missing")
            writer.writerow(row)
    print(f"비교 요약: {args.output_dir / 'summary.json'}")
    print(f"사람 검토 CSV: {args.output_dir / 'review.csv'}")


async def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency는 1 이상이어야 합니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = load_or_create_manifest(args)
    if args.prepare_only:
        print(f"동일 표본 {len(papers)}편을 준비했습니다: {args.output_dir / 'sample_manifest.json'}")
        return
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 필요합니다. --prepare-only는 키 없이 실행할 수 있습니다.")
    client = AsyncOpenAI()
    for model in args.models:
        await run_model(client, model, papers, args)
    build_reports(args, papers)


if __name__ == "__main__":
    asyncio.run(main())
