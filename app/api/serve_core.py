"""自研后端 API 主链路（纯 numpy）"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.models import TinyLM, TransformerLM
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


@dataclass
class State:
    model: TinyLM | None = None
    tokenizer: CharTokenizer | None = None


state = State()
app = FastAPI(title="LLM Core API", version="0.1.0")


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


def _load_or_init():
    ckpt = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            payload = pickle.load(f)
        tok = CharTokenizer.from_dict(payload["tokenizer"])
        model_cfg = payload["model"]

        transformer_cfg = _infer_transformer_config(model_cfg)
        if transformer_cfg is not None:
            model = TransformerLM(
                vocab_size=transformer_cfg["vocab_size"],
                n_embd=transformer_cfg["n_embd"],
                n_layers=transformer_cfg["n_layers"],
                n_heads=transformer_cfg["n_heads"],
                max_seq_len=transformer_cfg["max_seq_len"],
                dropout=transformer_cfg["dropout"],
            )
        else:
            model = TinyLM(vocab_size=model_cfg["vocab_size"], n_embd=model_cfg["n_embd"])

        state_dict = model_cfg["state_dict"]
        for i, p in enumerate(model.parameters()):
            key = f"param_{i}"
            if key not in state_dict:
                raise ValueError(f"checkpoint 缺少参数: {key}")
            src = state_dict[key]
            if p.data.shape != src.shape:
                raise ValueError(
                    f"checkpoint 参数形状不匹配: {key}, src={src.shape}, dst={p.data.shape}"
                )
            p.data[...] = src

        model.eval()
        state.model = model
        state.tokenizer = tok
        return

    tok = CharTokenizer.from_texts(["你好，世界", "自研后端服务"]) 
    state.model = TinyLM(vocab_size=tok.vocab_size, n_embd=128)
    state.model.eval()
    state.tokenizer = tok


@app.on_event("startup")
def startup_event():
    _load_or_init()


@app.get("/health")
def health():
    return {"status": "ok", "backend": "core"}


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
    rng = np.random.default_rng(req.seed)

    ids = state.tokenizer.encode(req.prompt)
    if not ids:
        ids = [0]

    max_ctx = getattr(state.model, "max_seq_len", None)
    for _ in range(req.max_new_tokens):
        ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
        x = np.array([ctx], dtype=np.int64)
        logits, _ = state.model(x, None)
        next_id = sample_next_token(
            logits.data[0, -1],
            token_ids=ids,
            cfg=sampling_cfg,
            rng=rng,
        )
        ids.append(next_id)

    text = state.tokenizer.decode(ids)
    if text.startswith(req.prompt):
        text = text[len(req.prompt):]

    return GenerateResponse(text=text)
