"""自研后端 API 主链路（纯 numpy）"""

import logging
import os
import pickle
import time
import uuid
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from app.core.models import TinyLM, TransformerLM
    MODEL_BACKEND = "tensor"
except ImportError:
    from app.core.models_neurx import NeurXTinyLM as TinyLM
    from app.core.models_neurx import NeurXChatModel as TransformerLM
    MODEL_BACKEND = "neurx"

from app.core.sampling import SamplingConfig, sample_next_token
from app.core.tokenizer import CharTokenizer


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=40, ge=1, le=1024)
    top_p: float = Field(default=0.92, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=2.0)
    seed: int | None = Field(default=None, ge=0)


class GenerateResponse(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=8192)


class ChatCompletionsRequest(BaseModel):
    model: str = Field(default="core-transformer")
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.92, gt=0.0, le=1.0)
    top_k: int | None = Field(default=40, ge=1, le=1024)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=2.0)
    seed: int | None = Field(default=None, ge=0)
    stream: bool = Field(default=False)
    stop: str | list[str] | None = None


@dataclass
class State:
    model: TinyLM | None = None
    tokenizer: CharTokenizer | None = None


state = State()
app = FastAPI(title="LLM Core API", version="0.1.0")
logger = logging.getLogger(__name__)


def _to_numpy(value):
    if hasattr(value, "data"):
        value = value.data
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _extract_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    if isinstance(model_output, dict):
        return model_output.get("logits")
    return model_output


def _infer_transformer_config(model_cfg):
    state_dict = model_cfg["state_dict"]
    n_embd = int(model_cfg["n_embd"])

    # 兼容旧 checkpoint：没有 n_layers/n_heads/max_seq_len 时，按参数形状推断 Transformer
    has_transformer_meta = all(k in model_cfg for k in ("n_layers", "n_heads", "max_seq_len"))
    looks_like_transformer = (
        "param_1" in state_dict
        and getattr(state_dict["param_1"], "ndim", 0) == 2
        and state_dict["param_1"].shape[1] == n_embd
        and len(state_dict) > 3
    )
    if not has_transformer_meta and not looks_like_transformer:
        return None

    max_seq_len = int(model_cfg.get("max_seq_len", state_dict["param_1"].shape[0]))
    n_layers = model_cfg.get("n_layers")
    if n_layers is None:
        inferred = (len(state_dict) - 5) / 12
        if inferred < 1 or int(inferred) != inferred:
            raise ValueError("无法从 checkpoint 推断 n_layers，请在 model 中补充 n_layers")
        n_layers = int(inferred)

    n_heads = model_cfg.get("n_heads")
    if n_heads is None:
        if n_embd % 8 == 0:
            n_heads = 8
        elif n_embd % 4 == 0:
            n_heads = 4
        else:
            n_heads = 1

    return {
        "vocab_size": int(model_cfg["vocab_size"]),
        "n_embd": n_embd,
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "max_seq_len": max_seq_len,
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }


def _build_tiny_model(vocab_size: int, hidden_dim: int = 128):
    if MODEL_BACKEND == "neurx":
        return TinyLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    return TinyLM(vocab_size=vocab_size, n_embd=hidden_dim)


def _build_transformer_model(cfg: dict):
    if MODEL_BACKEND == "neurx":
        return TransformerLM(
            vocab_size=cfg["vocab_size"],
            hidden_dim=cfg["n_embd"],
            num_layers=cfg["n_layers"],
            num_heads=cfg["n_heads"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
        )
    return TransformerLM(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    )


def _init_fallback_model():
    tok = CharTokenizer.from_texts(["你好，世界", "自研后端服务"])
    model = _build_tiny_model(vocab_size=tok.vocab_size, hidden_dim=128)
    if hasattr(model, "eval"):
        model.eval()
    state.model = model
    state.tokenizer = tok


def _load_or_init():
    ckpt = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    if not os.path.exists(ckpt):
        _init_fallback_model()
        return

    try:
        with open(ckpt, "rb") as f:
            payload = pickle.load(f)
        tok = CharTokenizer.from_dict(payload["tokenizer"])
        model_cfg = payload["model"]

        transformer_cfg = _infer_transformer_config(model_cfg)
        if transformer_cfg is not None:
            model = _build_transformer_model(transformer_cfg)
        else:
            model = _build_tiny_model(
                vocab_size=int(model_cfg["vocab_size"]),
                hidden_dim=int(model_cfg.get("n_embd", 128)),
            )

        state_dict = model_cfg["state_dict"]
        for i, p in enumerate(model.parameters()):
            key = f"param_{i}"
            if key not in state_dict:
                raise ValueError(f"checkpoint 缺少参数: {key}")
            src = state_dict[key]
            dst = p.data if hasattr(p, "data") else p
            if dst.shape != src.shape:
                raise ValueError(
                    f"checkpoint 参数形状不匹配: {key}, src={src.shape}, dst={dst.shape}"
                )
            dst[...] = src

        if hasattr(model, "eval"):
            model.eval()
        state.model = model
        state.tokenizer = tok
    except Exception as exc:
        logger.warning("加载 checkpoint 失败，回退到随机初始化: %s", exc)
        _init_fallback_model()


@app.on_event("startup")
def startup_event():
    _load_or_init()


@app.get("/health")
def health():
    return {"status": "ok", "backend": "core"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "core-transformer",
                "object": "model",
                "created": 0,
                "owned_by": "self-hosted-core",
            }
        ],
    }


def _build_prompt(messages: list[ChatMessage]) -> str:
    lines = []
    for m in messages:
        if m.role == "system":
            lines.append(f"[System]\n{m.content}")
        elif m.role == "user":
            lines.append(f"[User]\n{m.content}")
        else:
            lines.append(f"[Assistant]\n{m.content}")
    lines.append("[Assistant]\n")
    return "\n\n".join(lines)


def _generate_ids(
    initial_ids: list[int],
    max_new_tokens: int,
    sampling_cfg: SamplingConfig,
    stop_sequences: list[str],
) -> tuple[list[int], str]:
    rng = np.random.default_rng(sampling_cfg.seed)
    ids = initial_ids[:]
    max_ctx = getattr(state.model, "max_seq_len", None)
    generated_text = ""

    for _ in range(max_new_tokens):
        ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
        x = np.array([ctx], dtype=np.int64)
        model_output = state.model(x, None)
        logits = _extract_logits(model_output)
        logits_np = _to_numpy(logits)
        next_id = sample_next_token(
            logits_np[0, -1],
            token_ids=ids,
            cfg=sampling_cfg,
            rng=rng,
        )
        ids.append(next_id)

        if stop_sequences:
            generated_text = state.tokenizer.decode(ids)
            for stop_seq in stop_sequences:
                if stop_seq and stop_seq in generated_text:
                    cut_idx = generated_text.index(stop_seq)
                    return ids, generated_text[:cut_idx]

    return ids, state.tokenizer.decode(ids)


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="model not ready")

    sampling_cfg = SamplingConfig(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        seed=req.seed,
    )
    sampling_cfg.validate()
    ids = state.tokenizer.encode(req.prompt)
    if not ids:
        ids = [0]

    ids, text = _generate_ids(
        initial_ids=ids,
        max_new_tokens=req.max_new_tokens,
        sampling_cfg=sampling_cfg,
        stop_sequences=[],
    )
    if text.startswith(req.prompt):
        text = text[len(req.prompt):]

    return GenerateResponse(text=text)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported yet")
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="model not ready")

    sampling_cfg = SamplingConfig(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        seed=req.seed,
    )
    sampling_cfg.validate()

    prompt = _build_prompt(req.messages)
    prompt_ids = state.tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    stop_sequences = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    ids, full_text = _generate_ids(
        initial_ids=prompt_ids,
        max_new_tokens=req.max_tokens,
        sampling_cfg=sampling_cfg,
        stop_sequences=stop_sequences,
    )
    completion_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text

    usage_prompt_tokens = len(prompt_ids)
    usage_total_tokens = len(ids)
    usage_completion_tokens = max(0, usage_total_tokens - usage_prompt_tokens)
    created = int(time.time())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage_prompt_tokens,
            "completion_tokens": usage_completion_tokens,
            "total_tokens": usage_total_tokens,
        },
    }
