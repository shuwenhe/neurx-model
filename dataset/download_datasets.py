#!/usr/bin/env python3
"""Download training datasets from Hugging Face mirror and GitHub.

This version avoids the `datasets` Python package so it can run in minimal
environments where optional stdlib modules are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
TEXT_DIR = ROOT / "text"
HF_DIR = ROOT / "huggingface"
GH_DIR = ROOT / "github"
MANIFEST = ROOT / "datasets.manifest.json"


def ensure_dirs() -> None:
    for d in (TEXT_DIR, HF_DIR, GH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def download_file(name: str, url: str, out_file: Path, timeout: int = 180) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    out_file.write_bytes(data)
    return {
        "name": name,
        "url": url,
        "path": str(out_file.relative_to(ROOT)),
        "bytes": len(data),
    }


def merge_text_corpus(files: list[Path], out_file: Path) -> int:
    chunks = []
    for f in files:
        if f.exists():
            chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
    corpus = "\n\n".join(chunks)
    out_file.write_text(corpus, encoding="utf-8")
    return len(corpus)


def main() -> int:
    ensure_dirs()

    manifest: dict[str, object] = {
        "root": str(ROOT),
        "downloads": [],
        "notes": "Downloaded from Hugging Face mirror and GitHub raw files.",
    }

    hf_items = [
        (
            "tinystories_valid",
            "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
            HF_DIR / "TinyStories-valid.txt",
        ),
        (
            "wikitext2_train_parquet",
            "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
            HF_DIR / "wikitext2-train.parquet",
        ),
    ]

    for name, url, out_file in hf_items:
        try:
            item = download_file(name, url, out_file)
            item["source"] = "huggingface"
            manifest["downloads"].append(item)
            print(f"[ok] hf {name} -> {item['path']}")
        except Exception as exc:
            print(f"[warn] hf {name} failed: {exc}")

    github_items = [
        (
            "karpathy_tinyshakespeare",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            GH_DIR / "tinyshakespeare.txt",
        ),
        (
            "stanford_alpaca",
            "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
            GH_DIR / "alpaca_data.json",
        ),
    ]

    for name, url, out_file in github_items:
        try:
            item = download_file(name, url, out_file)
            item["source"] = "github"
            manifest["downloads"].append(item)
            print(f"[ok] github {name} -> {item['path']}")
        except Exception as exc:
            print(f"[warn] github {name} failed: {exc}")

    merged_inputs = [
        HF_DIR / "TinyStories-valid.txt",
        GH_DIR / "tinyshakespeare.txt",
    ]
    merged_out = TEXT_DIR / "neurx_train_mix_v1.txt"
    merged_chars = merge_text_corpus(merged_inputs, merged_out)
    manifest["merged_corpus"] = {
        "path": str(merged_out.relative_to(ROOT)),
        "chars": merged_chars,
        "inputs": [str(p.relative_to(ROOT)) for p in merged_inputs if p.exists()],
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] manifest -> {MANIFEST}")
    print(f"[ok] merged corpus -> {merged_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
