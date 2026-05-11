# neurx-model dataset workspace

This directory stores local training data downloaded from public sources.

## Structure

- `huggingface/`: sampled datasets downloaded via `datasets`
- `github/`: raw files downloaded from GitHub
- `text/neurx_train_mix_v1.txt`: merged text corpus for quick training
- `datasets.manifest.json`: download metadata

## Download datasets

Run from repo root:

```bash
cd /app/neurx-model
./venv/bin/python dataset/download_datasets.py
```

## Use downloaded corpus for training

Example (if your training entry accepts custom text file):

```bash
cd /app/neurx-model
python train_cli.py --preset standard --data-file dataset/text/neurx_train_mix_v1.txt
```

Or use the generated files under `dataset/huggingface/` and `dataset/github/` directly.

## GPT training data formats

For GPT-style training, this repository most commonly uses plain text corpora, but JSONL is a good fit when the data has structure.

### Plain text corpus

Use one sample per line when you only need raw text:

```text
The quick brown fox jumps over the lazy dog.
Another training sentence goes here.
```

### JSONL for instruction tuning

Use one JSON object per line when you need structured samples:

```json
{"text":"raw pretraining text"}
{"instruction":"Translate to English","input":"你好，世界","output":"Hello, world"}
{"messages":[{"role":"user","content":"Say hello"},{"role":"assistant","content":"Hello!"}]}
```

Recommended fields:

- `text`: raw pretraining text
- `instruction` / `input` / `output`: instruction tuning
- `messages`: chat or multi-turn dialogue

If you add a JSONL dataset under `dataset/text/`, keep each line self-contained so it can be streamed and shuffled efficiently.
