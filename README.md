# Industrial LLM Training System

An OpenAI-style, production-oriented large language model training system with local training, serving, and observability workflows.

## Quick Start

New here? Start with [docs/START_HERE.md](docs/START_HERE.md).

```bash
python train_cli.py --preset quick
```

## Core Features

- End-to-end training pipeline from data preparation to model export
- GPT-style autoregressive Transformer architecture
- Flexible training and model configuration
- Simple CLI/Makefile workflow for local development
- Production-friendly API endpoints and deployment scripts
- Built-in observability utilities

## Project Layout

```text
neurx-model/
├─ app/                    # API, training, inference, core modules
├─ checkpoints/            # Model artifacts (.pkl/.pt and s_arch outputs)
├─ data/                   # Local data cache
├─ deploy/                 # Systemd/nginx and local deployment assets
├─ docs/                   # Project documentation
├─ frontend/               # Next.js frontend
├─ logs/                   # Runtime logs
├─ reports/                # Training/build reports
├─ scripts/                # Utility scripts (including S-only bundle export)
├─ Makefile
└─ README.md
```

## Setup and Training

If installation fails, check [docs/INSTALL.md](docs/INSTALL.md).

### Recommended: Makefile workflow

```bash
# One-step setup: create virtualenv + install dependencies
make setup-all

# Activate virtualenv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# Validate baseline tests
make test

# Train (default entry)
make train

# Multimodal training
make train-multimodal

# Start API service
make serve

# Text generation
make generate
```

### Step-by-step setup

```bash
make setup
source venv/bin/activate
make install
make help
```

### Python-only manual flow (optional)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python generate.py
```

## Important Training Behavior

- `make train` calls `make train-chinese`.
- `make train-chinese` first checks whether the NeurX Python training runtime is complete.
- If required Python symbols are missing, it automatically falls back to the S-only export flow via `scripts/s_only_train_bundle.sh`.
- S-only artifacts are exported to `checkpoints/` as:
  - `s_arch_YYYYMMDDHHMMSS.bin`
  - `s_arch_YYYYMMDDHHMMSS.json`
  - `s_arch_latest.bin`
  - `s_arch_latest.json`

## Makefile Commands

### Environment

- `make setup-all`: create virtualenv and install dependencies
- `make setup`: create virtualenv only
- `make install`: install dependencies (requires activated virtualenv)
- `make install-force`: force install without virtualenv checks

### Training and Development

- `make test`: run baseline validation tests
- `make train`: default training entry
- `make train-basic`: basic text training script
- `make train-core`: custom core backend training
- `make train-chinese`: Chinese text training path with S-only fallback
- `make train-multimodal`: multimodal training
- `make train-neurx-s-multimodal`: multimodal training with S runtime precompile checks
- `make train-flow`: one-command training flow script

### Serving and Inference

- `make serve`: start API service
- `make serve-dev`: start API with reload
- `make serve-core`: start custom core API
- `make serve-core-dev`: start custom core API with reload
- `make gateway`: start gateway service
- `make generate`: interactive text generation
- `make inference-generate`: generation via service boundary
- `make quick-generate`: batch generation tests
- `make inference-quick`: quick generation via service boundary

### Frontend

- `make frontend-install`
- `make frontend-dev`
- `make frontend-build`
- `make frontend-start`

### Operations and Logs

- `make status-services`: check backend/frontend/nginx status
- `make restart-services`: restart systemd services
- `make logs`: show backend file log + systemd status + journal logs
- `make logs-follow`: follow backend file log in real time
- `make obs-up`: start observability stack
- `make obs-down`: stop observability stack

### Cleanup and Utilities

- `make clean`
- `make clean-checkpoints`
- `make clean-all`
- `make check-deps`
- `make info`
- `make init`

## API Endpoints

When the service is running, common endpoints include:

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `POST /v1/generate`
- `GET /v1/model-status`
- `GET /v1/s-arch`
- `GET /v1/s-arch/download`

Session APIs:

- `GET /v1/sessions/{session_id}`
- `DELETE /v1/sessions/{session_id}`

## Security and Rate Limiting

Recommended production environment variables:

- `LLM_API_KEYS`: comma-separated API keys
- `LLM_USERS`: OAuth2 users in `user:pass` pairs
- `LLM_JWT_SECRET`: JWT signing secret (must be changed in production)
- `LLM_JWT_EXPIRE_MINUTES`: token expiration in minutes
- `LLM_RATE_LIMIT_RPM`: rate limit per caller per minute (`0` disables)
- `LLM_LOG_LEVEL`: log level (`INFO` by default)
- `LLM_SESSION_DB`: SQLite file path for sessions

Example:

```bash
export LLM_API_KEYS="prod-key-1,prod-key-2"
export LLM_USERS="admin:admin123"
export LLM_JWT_SECRET="replace-with-strong-secret"
export LLM_RATE_LIMIT_RPM=60
make serve
```

## Container Deployment

```bash
docker build -t my-llm:latest .
docker run --rm -p 8000:8000 -e LLM_CHECKPOINT=checkpoints/best_model.pt my-llm:latest
```

## Observability

```bash
make obs-up
```

- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 (default: `admin/admin`)

Stop:

```bash
make obs-down
```

## Documentation Index

- [docs/START_HERE.md](docs/START_HERE.md)
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- [docs/CHEATSHEET.md](docs/CHEATSHEET.md)
- [docs/commands_reference.md](docs/commands_reference.md)
- [docs/checkpoint_system.md](docs/checkpoint_system.md)
- [docs/training_visualization.md](docs/training_visualization.md)
- [docs/openai_training_guide.md](docs/openai_training_guide.md)
- [docs/openai_vs_local_comparison.md](docs/openai_vs_local_comparison.md)
- [docs/README_DOCS.md](docs/README_DOCS.md)
- [docs/TRAINING_README.md](docs/TRAINING_README.md)

## FAQ

### Q: I see "externally-managed-environment" when installing packages on Linux. What should I do?

Use a virtual environment:

```bash
make setup-all
source venv/bin/activate
```

### Q: Training is too slow. How can I speed it up?

- Reduce model size or batch size
- Use GPU/CUDA instead of CPU
- Enable compiler/runtime acceleration where available

### Q: I hit memory limits. What should I tune first?

- Lower `batch_size`
- Lower `block_size`
- Reduce model dimensions (`n_layer`, `n_embd`)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT License
