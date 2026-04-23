"""Use local image-caption datasets to train a small NeurX vision-language model."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("Pillow is required for vision training. Install with: pip install Pillow") from exc

try:
    import neurx
    import neurx.nn as nn
    from neurx.optim import Adam
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("NeurX framework is required. Install the local neurx package first.") from exc

from app.core.models_neurx import NeurXTransformerBlock
from app.core.tokenizer import CharTokenizer


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                NeurXTransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, image_features, targets=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        token_embeddings = self.tok_emb(input_ids)
        pos_ids = neurx.arange(seq_len, dtype="int64").unsqueeze(0)
        hidden_states = token_embeddings + self.pos_emb(pos_ids)

        image_embeddings = self.image_proj(image_features).reshape(batch_size, 1, self.hidden_dim)
        hidden_states = neurx.cat([image_embeddings, hidden_states[:, 1:, :]], dim=1)
        hidden_states = self.dropout(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            logits_reshaped = logits.reshape(-1, self.vocab_size)
            targets_reshaped = targets.reshape(-1)
            loss = nn.functional.cross_entropy(logits_reshaped, targets_reshaped)

        return {
            "logits": logits,
            "loss": loss,
        }


def load_captions(data_path: Path) -> dict[str, str]:
    captions_path = data_path / "captions.json"
    if not captions_path.exists():
        raise FileNotFoundError(f"captions.json not found in {data_path}")
    with captions_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("captions.json must contain a non-empty object mapping image paths to text.")
    return {str(k): str(v) for k, v in payload.items()}


def load_image_features(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(-1)


def build_samples(data_path: Path, image_size: int) -> tuple[list[dict[str, object]], CharTokenizer]:
    captions = load_captions(data_path)
    texts = list(captions.values())
    tokenizer = CharTokenizer.from_texts(texts)

    samples = []
    for relative_image_path, text in captions.items():
        image_path = data_path / relative_image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        samples.append(
            {
                "image_path": str(image_path),
                "text": text,
                "image_features": load_image_features(image_path, image_size),
                "token_ids": tokenizer.encode(text),
            }
        )
    return samples, tokenizer


def make_batch(samples, batch_size: int, seq_len: int, tokenizer: CharTokenizer):
    batch = []
    if not samples:
        raise ValueError("No samples loaded for vision training.")

    while True:
        index = np.random.randint(0, len(samples))
        batch.append(samples[index])
        if len(batch) < batch_size:
            continue

        x_batch = []
        y_batch = []
        image_batch = []
        for sample in batch:
            token_ids = list(sample["token_ids"])
            if not token_ids:
                token_ids = [0]
            needed = seq_len + 1
            repeat = (needed // len(token_ids)) + 1
            padded = (token_ids * repeat)[:needed]
            x_batch.append(padded[:seq_len])
            y_batch.append(padded[1 : seq_len + 1])
            image_batch.append(sample["image_features"])

        yield (
            np.asarray(x_batch, dtype=np.int64),
            np.asarray(y_batch, dtype=np.int64),
            np.asarray(image_batch, dtype=np.float32),
        )
        batch = []


def clip_grad_norm(parameters, max_norm: float) -> float:
    total_norm = 0.0
    for parameter in parameters:
        if getattr(parameter, "grad", None) is None:
            continue
        grad = parameter.grad
        total_norm += float(np.sum(grad ** 2))
    total_norm = float(np.sqrt(total_norm))

    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        for parameter in parameters:
            if getattr(parameter, "grad", None) is not None:
                parameter.grad *= scale
    return total_norm


def train_vision_real(args) -> None:
    np.random.seed(args.seed)
    neurx.manual_seed(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    samples, tokenizer = build_samples(data_path, args.image_size)
    image_dim = args.image_size * args.image_size * 3

    model = VisionLanguageModel(
        vocab_size=tokenizer.vocab_size,
        image_dim=image_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = max(1, len(samples) * args.steps_multiplier)
    batches = make_batch(samples, args.batch_size, args.seq_len, tokenizer)
    losses = []

    print("=" * 70)
    print("NeurX Vision-Language Training")
    print("=" * 70)
    print(f"data_path={data_path}")
    print(f"samples={len(samples)}, vocab_size={tokenizer.vocab_size}, image_size={args.image_size}")

    for epoch in range(args.epochs):
        epoch_losses = []
        for step in range(steps_per_epoch):
            input_ids, targets, image_features = next(batches)
            optimizer.zero_grad()

            outputs = model(input_ids, image_features, targets=targets)
            loss = outputs["loss"]
            loss.backward()
            grad_norm = clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_value = float(loss) if hasattr(loss, "__float__") else float(loss.item())
            losses.append(loss_value)
            epoch_losses.append(loss_value)

            if (step + 1) % max(1, steps_per_epoch // 2) == 0 or step == 0:
                avg_loss = float(np.mean(epoch_losses[-5:]))
                print(
                    f"epoch {epoch + 1}/{args.epochs}, step {step + 1}/{steps_per_epoch}: "
                    f"loss={loss_value:.4f} avg={avg_loss:.4f} grad_norm={grad_norm:.4f}"
                )

    output_path = Path(args.output).expanduser().resolve()
    os.makedirs(output_path.parent, exist_ok=True)

    state_dict = {}
    for index, parameter in enumerate(model.parameters()):
        state_dict[f"param_{index}"] = (
            parameter.data.copy() if hasattr(parameter.data, "copy") else np.array(parameter.data)
        )

    payload = {
        "backend": "neurx",
        "task": "vision_language",
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "seq_len": args.seq_len,
            "image_size": args.image_size,
            "state_dict": state_dict,
        },
        "tokenizer": tokenizer.to_dict(),
        "captions": {sample["image_path"]: sample["text"] for sample in samples},
        "metrics": {
            "start_loss": float(losses[0]) if losses else 0.0,
            "end_loss": float(losses[-1]) if losses else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
        },
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print("✅ Vision-language training completed")
    print(f"checkpoint={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small local image-caption model with NeurX")
    parser.add_argument("--data-source", type=str, default="local", choices=["local"])
    parser.add_argument("--data-path", type=str, required=True, help="Path containing images/ and captions.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-multiplier", type=int, default=4, help="Training steps per sample per epoch")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/vision_trained_model.pkl",
        help="Output checkpoint path",
    )
    args = parser.parse_args()
    train_vision_real(args)


if __name__ == "__main__":
    main()