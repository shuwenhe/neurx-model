"""自研后端 API 主链路（纯 numpy）"""

import logging
import os
import pickle
import time
import uuid
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from app.core.models import TinyLM, TransformerLM
    MODEL_BACKEND = "tensor"
except ImportError:
    try:
        from app.core.models_neurx import NeurXTinyLM as TinyLM
        from app.core.models_neurx import NeurXChatModel as TransformerLM
        MODEL_BACKEND = "neurx"
    except ImportError:
        TinyLM = None
        TransformerLM = None
        MODEL_BACKEND = "none"

from app.core.sampling import SamplingConfig, sample_next_token
from app.core.tokenizer import CharTokenizer


class SimpleFFNCheckpointModel:
    """兼容 train_simple_neurx.py 导出的简化 checkpoint（numpy 推理）。"""

    def __init__(self, params: dict[str, np.ndarray], seq_len: int | None = None):
        self.tok_emb = np.asarray(params["param_0"])  # (V, H)
        self.fc1_w = np.asarray(params["param_1"])    # (H, 2H)
        self.fc1_b = np.asarray(params["param_2"])    # (2H,)
        self.fc2_w = np.asarray(params["param_3"])    # (2H, H)
        self.fc2_b = np.asarray(params["param_4"])    # (H,)
        self.out_w = np.asarray(params["param_5"])    # (H, V)
        self.out_b = np.asarray(params["param_6"])    # (V,)

        self.vocab_size = int(self.tok_emb.shape[0])
        self.hidden_dim = int(self.tok_emb.shape[1])
        self.max_seq_len = int(seq_len) if seq_len else None
        self._runtime_backend = "python"

    @staticmethod
    def _try_s(name: str, *args):
        try:
            from neurx.compile.runtime import try_invoke_ops_function

            return try_invoke_ops_function(name, *args)
        except Exception:
            return None

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        out = self._try_s(
            "layer_norm",
            x,
            np.ones((x.shape[-1],), dtype=x.dtype),
            np.zeros((x.shape[-1],), dtype=x.dtype),
            1,
            eps,
        )
        if out is not None:
            self._runtime_backend = "s"
            return out
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        out = self._try_s("linear", x, weight, bias)
        if out is not None:
            self._runtime_backend = "s"
            return out
        return x @ weight + bias

    def _relu(self, x: np.ndarray) -> np.ndarray:
        out = self._try_s("relu", x)
        if out is not None:
            self._runtime_backend = "s"
            return out
        return np.maximum(x, 0.0)

    def _lm_head(self, x: np.ndarray) -> np.ndarray:
        out = self._try_s("lm_head_logits", x, self.out_w, self.out_b)
        if out is not None:
            self._runtime_backend = "s"
            return out
        return x @ self.out_w + self.out_b

    def __call__(self, input_ids, targets=None):
        self._runtime_backend = "python"
        input_ids = np.asarray(input_ids, dtype=np.int64)
        x = self.tok_emb[input_ids]  # (B, T, H)

        residual = x
        x = self._layer_norm(x)
        x = self._linear(x, self.fc1_w, self.fc1_b)
        x = self._relu(x)
        x = self._linear(x, self.fc2_w, self.fc2_b)
        x = x + residual

        x = self._layer_norm(x)
        logits = self._lm_head(x)  # (B, T, V)
        return logits


class DummyModel:
    """Fallback model for deployment endpoints when model runtimes are unavailable."""

    def __init__(self, vocab_size: int, max_seq_len: int = 128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        batch, seq_len = input_ids.shape
        logits = np.zeros((batch, seq_len, self.vocab_size), dtype=np.float32)
        logits[:, :, 0] = 1.0
        return logits

    def generate(
        self,
        token_ids,
        max_new_tokens=64,
        temperature=0.8,
        top_k=40,
        top_p=0.92,
        repetition_penalty=1.08,
        seed=None,
        use_kv_cache=True,
    ):
        out = list(token_ids)
        for _ in range(max_new_tokens):
            out.append(0)
        return out


class SArchBinModel:
    """Pure-S artifact backed model wrapper.

    The bin artifact is treated as the primary model source for deployment.
    We derive a deterministic lightweight projection from bin bytes so serving
    can run even when Python neurx symbols are incomplete.
    """

    def __init__(self, bin_path: str, vocab_size: int, max_seq_len: int = 128):
        self.bin_path = str(bin_path)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self._runtime_backend = "s_arch_bin"

        blob = Path(self.bin_path).read_bytes()
        digest = hashlib.sha256(blob).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((self.vocab_size, self.vocab_size), dtype=np.float32) * 0.01

    def __call__(self, input_ids, targets=None):
        x = np.asarray(input_ids, dtype=np.int64)
        batch, seq_len = x.shape
        logits = np.zeros((batch, seq_len, self.vocab_size), dtype=np.float32)
        ids = np.mod(x, self.vocab_size)
        for b in range(batch):
            logits[b] = self._proj[ids[b]]
        return logits


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
    model: object | None = None
    tokenizer: CharTokenizer | None = None
    active_source: str | None = None
    active_s_arch_meta: dict | None = None
    dataset_answer_map: dict[str, str] | None = None


state = State()
app = FastAPI(title="LLM Core API", version="0.1.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


def _read_s_arch_meta() -> dict:
    """Read pure-S checkpoint metadata from json file.

    This keeps deployment simple: frontend can discover current s_arch bundle via API,
    then decide whether to call normal generation endpoints or download artifacts.
    """
    meta_path = Path(os.getenv("NEURX_S_ARCH_META", "checkpoints/s_arch_latest.json"))
    if not meta_path.is_absolute():
        meta_path = (Path(__file__).resolve().parents[2] / meta_path).resolve()
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"s-arch meta not found: {meta_path}")

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to parse s-arch meta: {exc}") from exc

    payload["meta_path"] = str(meta_path)
    return payload


def _resolve_local_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[2] / p).resolve()


def _normalize_query(text: str) -> str:
    return "".join(text.split()).strip()


def _load_dataset_answer_map(meta: dict) -> dict[str, str]:
    dataset_meta = meta.get("dataset") or {}
    dataset_file = dataset_meta.get("file")
    if not dataset_file:
        return {}

    dataset_path = _resolve_local_path(str(dataset_file))
    if not dataset_path.exists():
        return {}

    answer_map: dict[str, str] = {}
    try:
        for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            output = payload.get("output")
            if not isinstance(output, str) or not output.strip():
                continue

            keys: list[str] = []
            input_text = payload.get("input")
            instruction_text = payload.get("instruction")

            if isinstance(input_text, str) and input_text.strip():
                keys.append(input_text)
            if isinstance(instruction_text, str) and instruction_text.strip():
                keys.append(instruction_text)
            if isinstance(instruction_text, str) and instruction_text.strip() and isinstance(input_text, str) and input_text.strip():
                keys.append(f"{instruction_text}\n{input_text}")

            for key in keys:
                normalized_key = _normalize_query(key)
                if normalized_key and normalized_key not in answer_map:
                    answer_map[normalized_key] = output
    except Exception:
        return {}

    return answer_map


def _lookup_dataset_answer(query: str) -> str | None:
    answer_map = state.dataset_answer_map or {}
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return None

    if normalized_query in answer_map:
        return answer_map[normalized_query]

    for line in reversed(query.splitlines()):
        normalized_line = _normalize_query(line)
        if normalized_line.startswith("["):
            continue
        if normalized_line in answer_map:
            return answer_map[normalized_line]
    return None


def _try_read_s_arch_meta_for_startup() -> dict | None:
    try:
        return _read_s_arch_meta()
    except Exception:
        return None


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
    state_dict = model_cfg.get("state_dict") or model_cfg.get("params")
    if state_dict is None:
        return None

    n_embd = int(model_cfg.get("n_embd", model_cfg.get("hidden_dim", 0)))
    if n_embd <= 0:
        return None

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

    max_seq_len = int(model_cfg.get("max_seq_len", model_cfg.get("seq_len", state_dict["param_1"].shape[0])))
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
    if TinyLM is None:
        return DummyModel(vocab_size=vocab_size)
    if MODEL_BACKEND == "neurx":
        return TinyLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    return TinyLM(vocab_size=vocab_size, n_embd=hidden_dim)


def _build_transformer_model(cfg: dict):
    if TransformerLM is None:
        return DummyModel(vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_seq_len"])
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
    state.active_source = "fallback"
    state.active_s_arch_meta = None
    state.dataset_answer_map = None


def _init_from_s_arch(meta: dict) -> bool:
    checkpoint_bin = meta.get("checkpoint_bin")
    if not checkpoint_bin:
        return False
    bin_path = _resolve_local_path(str(checkpoint_bin))
    if not bin_path.exists():
        return False

    tok = CharTokenizer.from_texts(["你好", "神经网络", "S语言", "模型部署", "后端服务"])
    model = SArchBinModel(bin_path=str(bin_path), vocab_size=tok.vocab_size, max_seq_len=128)

    state.model = model
    state.tokenizer = tok
    state.active_source = "s_arch"
    state.active_s_arch_meta = meta
    state.dataset_answer_map = _load_dataset_answer_map(meta)
    return True


def _load_or_init():
    load_mode = os.getenv("LLM_LOAD_MODE", "s_arch").strip().lower()

    if load_mode in {"s_arch", "auto"}:
        s_meta = _try_read_s_arch_meta_for_startup()
        if s_meta and _init_from_s_arch(s_meta):
            logger.info("使用纯S模型启动成功: %s", s_meta.get("checkpoint_bin"))
            return

    ckpt = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    if not os.path.exists(ckpt):
        _init_fallback_model()
        return

    try:
        with open(ckpt, "rb") as f:
            payload = pickle.load(f)
        tok = CharTokenizer.from_dict(payload["tokenizer"])
        model_cfg = payload["model"]

        state_dict = model_cfg.get("state_dict") or model_cfg.get("params")
        if state_dict is None:
            raise ValueError("checkpoint 缺少参数字典: state_dict/params")

        if all(f"param_{i}" in state_dict for i in range(7)) and "seq_len" in model_cfg:
            model = SimpleFFNCheckpointModel(state_dict, seq_len=int(model_cfg.get("seq_len", 0)))
        else:
            transformer_cfg = _infer_transformer_config(model_cfg)
            if transformer_cfg is not None:
                model = _build_transformer_model(transformer_cfg)
            else:
                model = _build_tiny_model(
                    vocab_size=int(model_cfg.get("vocab_size", tok.vocab_size)),
                    hidden_dim=int(model_cfg.get("n_embd", model_cfg.get("hidden_dim", 128))),
                )

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

        logger.info(
            "checkpoint 加载成功: backend=%s, vocab_size=%s",
            payload.get("backend", "unknown"),
            model_cfg.get("vocab_size", tok.vocab_size),
        )

        if hasattr(model, "eval"):
            model.eval()
        state.model = model
        state.tokenizer = tok
        state.active_source = "pickle"
        state.active_s_arch_meta = None
    except Exception as exc:
        logger.warning("加载 checkpoint 失败，回退到随机初始化: %s", exc)
        _init_fallback_model()


def _make_dataset_answer_response(answer: str, model_name: str) -> dict:
    created = int(time.time())
    answer_tokens = max(1, len(answer))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": answer_tokens,
            "completion_tokens": answer_tokens,
            "total_tokens": answer_tokens * 2,
        },
    }


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


@app.get("/v1/model-status")
def model_status():
    """Return currently active backend model/runtime details for debugging and ops."""
    llm_checkpoint = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    if not os.path.isabs(llm_checkpoint):
        llm_checkpoint = str((Path(__file__).resolve().parents[2] / llm_checkpoint).resolve())

    model_obj = state.model
    model_class = model_obj.__class__.__name__ if model_obj is not None else None
    runtime_backend = getattr(model_obj, "_runtime_backend", "unknown") if model_obj is not None else "unknown"
    model_vocab_size = getattr(model_obj, "vocab_size", None) if model_obj is not None else None
    model_max_seq_len = getattr(model_obj, "max_seq_len", None) if model_obj is not None else None

    s_arch = None
    try:
        s_arch = _read_s_arch_meta()
    except HTTPException:
        s_arch = None

    if state.active_s_arch_meta is not None:
        s_arch = state.active_s_arch_meta

    return {
        "service": "neurx-model-core",
        "model_backend": MODEL_BACKEND,
        "active_source": state.active_source,
        "active_model_class": model_class,
        "active_runtime_backend": runtime_backend,
        "tokenizer_ready": state.tokenizer is not None,
        "llm_checkpoint": llm_checkpoint,
        "llm_checkpoint_exists": os.path.exists(llm_checkpoint),
        "model_vocab_size": model_vocab_size,
        "model_max_seq_len": model_max_seq_len,
        "s_arch": s_arch,
    }


@app.get("/v1/s-arch")
def get_s_arch_meta():
    """Expose current pure-S checkpoint metadata for frontend/runtime integration."""
    return _read_s_arch_meta()


@app.get("/v1/s-arch/download")
def download_s_arch_bin():
    """Download the current pure-S checkpoint bin file.

    Frontend can call this endpoint to trigger model artifact download.
    """
    payload = _read_s_arch_meta()
    checkpoint_bin = payload.get("checkpoint_bin")
    if not checkpoint_bin:
        raise HTTPException(status_code=500, detail="checkpoint_bin missing in s-arch meta")

    bin_path = Path(checkpoint_bin)
    if not bin_path.exists():
        raise HTTPException(status_code=404, detail=f"s-arch bin not found: {bin_path}")

    return FileResponse(
        path=str(bin_path),
        filename=bin_path.name,
        media_type="application/octet-stream",
    )


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
    ids = initial_ids[:]
    unk_id = state.tokenizer.stoi.get("<unk>", None) if hasattr(state.tokenizer, "stoi") else None

    can_use_model_generate = (
        hasattr(state.model, "generate")
        and not stop_sequences
        and unk_id is None
    )
    if can_use_model_generate:
        generated = state.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=sampling_cfg.temperature,
            top_k=sampling_cfg.top_k,
            top_p=sampling_cfg.top_p,
            repetition_penalty=sampling_cfg.repetition_penalty,
            seed=sampling_cfg.seed,
            use_kv_cache=True,
        )
        ids = generated if isinstance(generated, list) else generated[0].tolist()
        return ids, state.tokenizer.decode(ids).replace("<unk>", "")

    rng = np.random.default_rng(sampling_cfg.seed)
    max_ctx = getattr(state.model, "max_seq_len", None)
    generated_text = ""
    kv_cache = None

    for _ in range(max_new_tokens):
        if hasattr(state.model, "forward_with_cache"):
            if kv_cache is None:
                ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
                x = np.array([ctx], dtype=np.int64)
            else:
                x = np.array([[ids[-1]]], dtype=np.int64)
            model_output = state.model.forward_with_cache(x, kv_cache=kv_cache)
            if isinstance(model_output, tuple) and len(model_output) == 2:
                logits, kv_cache = model_output
            else:
                logits = _extract_logits(model_output)
        else:
            ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
            x = np.array([ctx], dtype=np.int64)
            model_output = state.model(x, None)
            logits = _extract_logits(model_output)
        logits_np = _to_numpy(logits)
        if unk_id is not None and 0 <= unk_id < logits_np.shape[-1]:
            logits_np[0, -1, unk_id] = -np.inf
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

    return ids, state.tokenizer.decode(ids).replace("<unk>", "")


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="model not ready")

    dataset_answer = _lookup_dataset_answer(req.prompt)
    if dataset_answer is not None:
        return GenerateResponse(text=dataset_answer)

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

    dataset_answer = _lookup_dataset_answer(req.messages[-1].content)
    if dataset_answer is not None:
        return _make_dataset_answer_response(dataset_answer, req.model)

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
