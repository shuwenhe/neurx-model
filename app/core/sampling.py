"""Sampling utilities for text generation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    def validate(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must satisfy 0 < top_p <= 1, got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}"
            )


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / (exp_logits.sum() + 1e-12)


def _apply_repetition_penalty(logits: np.ndarray, token_ids: list[int], penalty: float) -> np.ndarray:
    if penalty == 1.0 or not token_ids:
        return logits

    out = logits.copy()
    for token_id in set(token_ids):
        if token_id < 0 or token_id >= out.shape[0]:
            continue
        # Hugging Face style penalty.
        if out[token_id] > 0:
            out[token_id] /= penalty
        else:
            out[token_id] *= penalty
    return out


def _apply_top_k(logits: np.ndarray, top_k: int | None) -> np.ndarray:
    if top_k is None or top_k >= logits.shape[0]:
        return logits

    out = np.full_like(logits, -np.inf)
    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    out[top_idx] = logits[top_idx]
    return out


def _apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return logits

    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]
    sorted_probs = _softmax(sorted_logits)
    cum_probs = np.cumsum(sorted_probs)

    keep_mask = cum_probs <= top_p
    if not np.any(keep_mask):
        keep_mask[0] = True
    else:
        first_exceed = np.argmax(cum_probs > top_p)
        if cum_probs[first_exceed] > top_p:
            keep_mask[first_exceed] = True

    keep_idx = sorted_idx[keep_mask]
    out = np.full_like(logits, -np.inf)
    out[keep_idx] = logits[keep_idx]
    return out


def sample_next_token(
    raw_logits: np.ndarray,
    token_ids: list[int],
    cfg: SamplingConfig,
    rng: np.random.Generator,
) -> int:
    cfg.validate()
    logits = np.asarray(raw_logits, dtype=np.float64)

    top_k_value = -1 if cfg.top_k is None else int(cfg.top_k)
    token_arr = np.asarray(token_ids, dtype=np.int64)
    try:
        from neurx.compile.runtime import try_invoke_ops_function

        if cfg.temperature == 0:
            next_id = try_invoke_ops_function(
                "generation_step",
                logits,
                token_arr,
                float(cfg.temperature),
                top_k_value,
                float(cfg.top_p),
                float(cfg.repetition_penalty),
            )
            if next_id is not None:
                return int(next_id)
        filtered = try_invoke_ops_function(
            "sampling_top_k_top_p",
            logits,
            token_arr,
            float(cfg.temperature),
            top_k_value,
            float(cfg.top_p),
            float(cfg.repetition_penalty),
        )
    except Exception:
        filtered = None
    if filtered is not None:
        probs = _softmax(np.asarray(filtered, dtype=np.float64))
        return int(rng.choice(len(probs), p=probs))

    logits = _apply_repetition_penalty(logits, token_ids, cfg.repetition_penalty)
    logits = _apply_top_k(logits, cfg.top_k)
    logits = _apply_top_p(logits, cfg.top_p)

    if cfg.temperature == 0:
        return int(np.argmax(logits))

    scaled = logits / max(cfg.temperature, 1e-12)
    probs = _softmax(scaled)
    return int(rng.choice(len(probs), p=probs))
