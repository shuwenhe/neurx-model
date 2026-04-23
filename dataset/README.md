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
