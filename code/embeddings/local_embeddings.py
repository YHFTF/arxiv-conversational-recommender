"""Local Qwen embedding inference; no OpenAI calls."""
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
IDENTITY = MODEL + ":" + REVISION + ":nf4-double:fp16:sdpa:512:v1"
DATA_DIR = ROOT / "output" / "nl_recommendation_qwen4b_nf4"
INSTRUCTION = "Given a research query, retrieve relevant academic papers that satisfy the research need"


class LocalEmbedder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen NF4 requires a CUDA GPU.")
        cache = str(ROOT / "output" / "hf_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=cache, padding_side="left")
        self.model = AutoModel.from_pretrained(
            MODEL, revision=REVISION, cache_dir=cache, device_map={"": 0}, dtype=torch.float16,
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        ).eval()
        if not getattr(self.model, "is_loaded_in_4bit", False):
            raise RuntimeError("Model did not load in 4-bit mode")

    @torch.inference_mode()
    def encode(self, texts, query=False):
        rows = []
        for text in texts:  # Batch size 1 keeps activation memory bounded on 8 GiB GPUs.
            if query:
                text = f"Instruct: {INSTRUCTION}\nQuery: {text}"
            batch = self.tokenizer([text], max_length=512, truncation=True,
                                   padding=True, return_tensors="pt").to("cuda:0")
            output = self.model(**batch, use_cache=False)
            vector = F.normalize(output.last_hidden_state[:, -1].float(), dim=1)
            if not torch.isfinite(vector).all():
                raise RuntimeError("Non-finite embedding produced.")
            rows.append(vector.cpu())
        return torch.cat(rows) if rows else torch.empty((0, 2560))


def main():
    import argparse
    import hashlib
    import json
    import sys
    sys.path.insert(0, str(ROOT / "code" / "data_analysis"))
    from build_nl_recommendation_data import load_papers, paper_text, paper_signature, DEFAULT_MASTER
    parser = argparse.ArgumentParser(description="Build isolated Qwen NF4 catalog, resumably")
    parser.add_argument("--limit", type=int, default=0, help="0=full corpus; positive=separate smoke catalog")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--dtm-path", type=Path, default=None, help="Optional separate D/T/M extraction; otherwise use master knowledge")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    papers = None
    if not args.smoke_test:
        if args.dtm_path is not None:
            papers = load_papers(args.master_path, args.dtm_path)
        else:
            master = json.loads(args.master_path.read_text(encoding="utf-8"))
            papers = []
            for row in master:
                if not row.get("title") or row["title"] == "Unknown" or not row.get("abstract"):
                    continue
                row = dict(row)
                row["node_idx"] = int(row["node_idx"])
                row["paper_id"] = str(row.get("paper_id", row["node_idx"]))
                knowledge = row.get("knowledge") or {}
                row["knowledge"] = {key: knowledge.get(key, []) for key in ("domain", "task", "method")}
                papers.append(row)
            papers.sort(key=lambda row: row["node_idx"])
        if not papers or len({p["node_idx"] for p in papers}) != len(papers):
            raise ValueError("Empty corpus or duplicate node indices")
        print(f"[corpus] {len(papers)} eligible papers; limit={args.limit}", flush=True)
    embedder = LocalEmbedder()
    if args.smoke_test:
        q = embedder.encode(["graph neural networks for paper recommendation",
                             "그래프 신경망을 활용한 학술 논문 추천"], query=True)
        d = embedder.encode(["Graph neural networks for recommending scientific papers.", "A study of marine biology."])
        long_vector = embedder.encode(["scientific paper recommendation with graph neural networks " * 150])
        if not torch.allclose(long_vector.norm(dim=1), torch.ones(1), atol=1e-5):
            raise RuntimeError("Long input normalization failed")
        scores = (q @ d.T).tolist()
        result = {"model": IDENTITY, "shape": list(d.shape), "scores": scores,
                  "loaded_in_4bit": bool(embedder.model.is_loaded_in_4bit),
                  "allocated_GiB": torch.cuda.memory_allocated() / 2**30,
                  "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}
        print(json.dumps(result), flush=True)
        if any(row[0] <= row[1] for row in scores):
            raise RuntimeError("Smoke retrieval ordering failed")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if args.limit:
        papers = papers[:args.limit]
    destination = DATA_DIR if not args.limit else DATA_DIR / f"smoke_{args.limit}"
    destination.mkdir(parents=True, exist_ok=True)
    texts = [paper_text(p, 1800) for p in papers]
    signature = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    checkpoint = destination / "local_checkpoint.pt"
    vectors = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if saved["model"] != IDENTITY or saved["signature"] != signature:
            raise ValueError("Checkpoint configuration/corpus mismatch")
        vectors = list(saved["vectors"].split(1))
    for i in range(len(vectors), len(texts)):
        vectors.append(embedder.encode([texts[i]]))
        if (i + 1) % 32 == 0 or i + 1 == len(texts):
            temp = checkpoint.with_suffix(".tmp")
            torch.save({"model": IDENTITY, "signature": signature, "vectors": torch.cat(vectors)}, temp)
            temp.replace(checkpoint)
            print(f"[local embeddings] {i + 1}/{len(texts)}", flush=True)
    torch.save({"model": IDENTITY, "node_indices": torch.tensor([p["node_idx"] for p in papers]),
                "paper_ids": [p["paper_id"] for p in papers], "embeddings": torch.cat(vectors),
                "corpus_signature": paper_signature(papers)}, destination / "paper_text_embeddings.pt")


if __name__ == "__main__":
    main()
