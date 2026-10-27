"""Direct, resumable download fallback for HF large-file transfer stalls."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from local_embeddings import ROOT, REVISION, MODEL


def download(filename):
    folder = ROOT / "output" / "hf_cache" / "models--Qwen--Qwen3-Embedding-4B" / "snapshots" / REVISION
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if target.exists():
        return
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}?download=true"
    metadata = requests.head(url, allow_redirects=False, timeout=30)
    metadata.raise_for_status()
    expected = metadata.headers.get("x-linked-etag", "").strip('"')
    temporary = target.with_suffix(".download")
    offset = temporary.stat().st_size if temporary.exists() else 0
    response = requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=(30, 60))
    response.raise_for_status()
    append = offset and response.status_code == 206
    if not append:
        offset = 0
    with temporary.open("ab" if append else "wb") as handle:
        last = offset
        for chunk in response.iter_content(8 * 1024 * 1024):
            handle.write(chunk)
            offset += len(chunk)
            if offset - last >= 256 * 1024 * 1024:
                print(filename, round(offset / 2**30, 2), "GiB", flush=True)
                last = offset
    response.close()
    digest = hashlib.sha256()
    with temporary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if len(expected) != 64 or digest.hexdigest() != expected:
        raise ValueError(f"SHA256 verification failed: {filename}")
    temporary.replace(target)
    print(filename, "SHA256 verified", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, [f"model-{i:05d}-of-00002.safetensors" for i in (1, 2)]))
