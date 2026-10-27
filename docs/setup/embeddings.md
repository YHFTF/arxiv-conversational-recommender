# Qwen3-Embedding-4B local NF4

Run these commands in the project folder, using the existing CUDA environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-local-embeddings.txt
.\venv\Scripts\python.exe code/embeddings/local_embeddings.py --smoke-test
.\venv\Scripts\python.exe code/embeddings/local_embeddings.py --limit 32
.\venv\Scripts\python.exe code/retrieval/run_nl_recommendation.py --data-dir output/nl_recommendation_qwen4b_nf4/smoke_32 --no-intent-analysis --query "graph neural networks for paper recommendation"
```

Build all papers after the smoke run:

```powershell
.\venv\Scripts\python.exe code/embeddings/local_embeddings.py
.\venv\Scripts\python.exe code/retrieval/run_nl_recommendation.py --data-dir output/nl_recommendation_qwen4b_nf4 --no-intent-analysis --query "graph neural networks for paper recommendation"
```

The official `Qwen/Qwen3-Embedding-4B` weights are downloaded to `output/hf_cache`
and quantized on loading with bitsandbytes NF4, double quantization and FP16
compute. This is not a separately published third-party quantized checkpoint.
The initial download is roughly 8 GB. Inference runs on CUDA device 0, using
SDPA attention, batch size 1 and a 512-token cap (including the query prompt).
Outputs are normalized float32 2560-dimensional vectors on CPU.

If the Hugging Face large-file download stalls, the fallback downloads the two
official shards directly and verifies their SHA256 hashes against Hub metadata:

```powershell
.\venv\Scripts\python.exe code/embeddings/download_qwen_weights.py
```

Queries and search facets receive an academic retrieval instruction; documents
do not. Both the corpus builder and retrieval use the same versioned embedding
identity. Changes to quantization, prompts, token limits or model revision must
use a new identity and rebuild both corpus and query caches.

The builder uses title, abstract (up to 1800 characters), and the existing
master's D/T/M fields. `--dtm-path PATH` optionally uses a separate successful
D/T/M extraction instead. Results and caches live under
`output/nl_recommendation_qwen4b_nf4`; limited runs have separate `smoke_N`
subfolders. The corpus checkpoints every 32 papers and resumes on rerun,
checking both embedding identity and a hash of all input texts.

`--no-intent-analysis` needs no OpenAI key for local retrieval. Default intent
analysis still uses OpenAI and requires its API key. The dashboard is not yet
configured to select this local catalog. Evaluation needs the same queries and
judgments as the OpenAI baseline copied into the local data directory; compare
quality before replacing the default catalog. Existing semantic thresholds were
tuned for the previous embedding and may need calibration.

## Verified 2026-10-27 on RTX 3060 Ti 8 GiB

- NF4 CUDA kernel and full 4-bit model inference passed.
- English and Korean queries ranked the relevant English sample above the unrelated sample.
- A 512-token input produced a finite, normalized vector.
- Peak PyTorch allocated CUDA memory in the smoke test: 2.887 GiB; this excludes driver/display overhead.
- Actual corpus smoke catalog: 32 x 2560, finite normalized vectors.
- End-to-end local retrieval returned "evasion attacks against machine learning at test time"
  first for "malware detection under adversarial evasion attacks" (cosine 0.706566).
- `pip check`, Python compilation and changed-file whitespace checks passed.

The full build, when launched in the background, records its process ID in
`output/nl_recommendation_qwen4b_nf4/build.pid` and progress in `build.log`
(model loading messages in `build-error.log`). Checkpoint progress is saved every
32 papers. Rerun the same build command after an interruption to resume; do not
run two builders against the same output directory at once.
