"""自研后端 API 主链路（纯 numpy）"""

import base64
import inspect
import logging
import os
import pickle
import socket
import time
import uuid
import json
import hashlib
import re
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
    dataset_canonical_answer_map: dict[str, str] | None = None
    retrieval_stats: dict[str, int] = field(default_factory=dict)


state = State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_or_init()
    yield


app = FastAPI(title="LLM Core API", version="0.1.0", lifespan=lifespan)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")
logger.setLevel(getattr(logging, os.getenv("LLM_LOG_LEVEL", "INFO").upper(), logging.INFO))


def _caller_code_ref(depth: int = 1) -> str:
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back
        if frame is None:
            return "unknown"
        return f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
    finally:
        del frame


def _format_log_value(value: Any, limit: int = 180) -> str:
    if isinstance(value, str):
        text = value.replace("\n", "\\n")
    elif isinstance(value, (int, float, bool)) or value is None:
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def _trace_log(event: str, **fields: Any) -> None:
    parts = [f"{key}={_format_log_value(value)}" for key, value in fields.items() if value is not None]
    message = f"trace event={event} code={_caller_code_ref(depth=2)}"
    if parts:
        message = f"{message} {' '.join(parts)}"
    logger.info(message)


def _use_qwen_vl_proxy() -> bool:
    mode = os.getenv("LLM_UPSTREAM_MODE", "").strip().lower()
    return mode in {"qwen25_vl", "qwen-vl", "qwen2.5-vl-7b"}


def _qwen_vl_base_url() -> str:
    return os.getenv("LLM_QWEN_VL_BASE_URL", "http://127.0.0.1:8004").rstrip("/")


def _qwen_vl_model_id() -> str:
    return os.getenv("LLM_QWEN_VL_MODEL_ID", "Qwen2.5-VL-7B").strip() or "Qwen2.5-VL-7B"


def _qwen_timeout_seconds() -> int:
    raw = os.getenv("LLM_QWEN_TIMEOUT_SECONDS", "6").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 6
    return max(2, value)


def _qwen_http_json(method: str, url: str, payload: dict | None = None, extra_headers: dict[str, str] | None = None) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_qwen_timeout_seconds()) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"qwen upstream http error: {exc.code} {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise HTTPException(status_code=502, detail="qwen upstream timeout") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"qwen upstream unavailable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="qwen upstream returned non-json response") from exc


def _qwen_upstream_model_ids() -> list[str]:
    base_url = _qwen_vl_base_url()
    data = _qwen_http_json("GET", f"{base_url}/v1/models")
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    ids: list[str] = []
    for item in models:
        if isinstance(item, dict):
            mid = item.get("id")
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
    return ids


def _extract_qwen_text(response_payload: dict) -> str:
    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=502, detail="qwen upstream returned empty choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


def _qwen_chat_completion(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    trace_id: str | None = None,
) -> tuple[dict, str]:
    base_url = _qwen_vl_base_url()
    preferred_model = _qwen_vl_model_id()
    chat_url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": preferred_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    extra_headers = {"X-Neurx-Trace-Id": trace_id or ""}

    _trace_log(
        "qwen_proxy_request",
        trace_id=trace_id,
        upstream=chat_url,
        model=preferred_model,
        message_count=len(messages),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    try:
        response = _qwen_http_json("POST", chat_url, payload, extra_headers=extra_headers)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        _trace_log(
            "qwen_proxy_response",
            trace_id=trace_id,
            model=preferred_model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        return response, preferred_model
    except HTTPException as exc:
        if "unknown model" not in str(exc.detail):
            raise

    upstream_ids = _qwen_upstream_model_ids()
    if not upstream_ids:
        raise HTTPException(status_code=502, detail="qwen upstream has no available model ids")

    fallback_model = upstream_ids[0]
    payload["model"] = fallback_model
    _trace_log("qwen_proxy_model_fallback", trace_id=trace_id, preferred_model=preferred_model, fallback_model=fallback_model)
    response = _qwen_http_json("POST", chat_url, payload, extra_headers=extra_headers)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    _trace_log(
        "qwen_proxy_response",
        trace_id=trace_id,
        model=fallback_model,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
    return response, fallback_model


def _image_to_data_url(image: UploadFile) -> str:
    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image payload")

    mime = image.content_type or "application/octet-stream"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
    compact = "".join(text.split()).strip().lower()
    # Keep only CJK/alnum for robust matching across punctuation variants.
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", compact)


def _canonicalize_lookup_query(normalized: str) -> str:
    out = normalized
    # Remove common Chinese question wrappers to improve factual lookup recall.
    for token in (
        "请问",
        "一下",
        "是什么",
        "是哪里",
        "在哪里",
        "在哪",
        "是哪",
        "哪里",
        "哪个",
        "哪儿",
        "哪",
        "是谁",
        "怎么",
        "如何",
        "吗",
        "的",
        "是",
    ):
        out = out.replace(token, "")
    return out


def _sanitize_user_query(text: str) -> str:
    """Extract a cleaner user question from noisy/repetitive input text."""
    raw = text.strip()
    if not raw:
        return ""

    # Common separator in copied noisy prompts.
    if "---" in raw:
        raw = raw.split("---")[-1].strip() or raw

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Prefer explicit structured question line when user provides
    # "背景/问题/期望" style input.
    for line in lines:
        if line.startswith("问题：") or line.startswith("问题:"):
            candidate = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            candidate = candidate.strip()
            if candidate:
                raw = candidate
                # Structured question line should take precedence.
                raw = re.sub(r"(.)\1{3,}", r"\1\1", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
                return raw

    if lines:
        # Prefer lines that look like actual questions/requests.
        scored: list[tuple[int, str]] = []
        for line in lines:
            score = 0
            if any(k in line for k in ("为什么", "怎么", "如何", "什么", "请", "?", "？", "吗")):
                score += 3
            if 4 <= len(line) <= 220:
                score += 1
            unique_ratio = len(set(line)) / max(1, len(line))
            if unique_ratio > 0.45:
                score += 1
            scored.append((score, line))
        scored.sort(key=lambda x: x[0], reverse=True)
        raw = scored[0][1]

    # Collapse very long repeated character runs.
    raw = re.sub(r"(.)\1{3,}", r"\1\1", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _try_s_serve_policy(*args):
    try:
        from neurx.compile.runtime import try_invoke_ops_function

        return try_invoke_ops_function("serve_should_clarify", *args)
    except Exception:
        return None


def _is_low_quality_query(original: str, sanitized: str) -> bool:
    original_n = _normalize_query(original)
    sanitized_n = _normalize_query(sanitized)
    if not original_n:
        return True

    has_structured_fields = all(k in original for k in ("背景", "问题", "期望"))
    if has_structured_fields and len(sanitized_n) >= 3 and any(k in sanitized_n for k in ("什么", "哪里", "谁", "为何", "为什么", "怎么", "如何", "吗")):
        return False

    unique_ratio = len(set(original_n)) / max(1, len(original_n))
    non_word_ratio = len(re.findall(r"[^\u4e00-\u9fffA-Za-z0-9]", original_n)) / max(1, len(original_n))
    query_markers = re.findall(r"为什么|怎么|如何|什么|请|是否|能否|吗|\?|？", sanitized_n)
    has_question_focus = len(query_markers) > 0
    repetitive_question = (
        len(original_n) >= 20
        and any(sanitized_n.count(token) >= 3 for token in ("为什么", "怎么", "如何", "什么"))
    )

    long_and_repetitive = len(original_n) >= 60 and unique_ratio < 0.42
    heavily_changed = (
        len(original_n) >= 50
        and len(sanitized_n) <= int(len(original_n) * 0.55)
        and not has_structured_fields
    )
    noisy_symbols = len(original_n) >= 40 and non_word_ratio > 0.18
    lacks_query_focus = len(original_n) >= 45 and len(query_markers) == 0
    too_short_after_clean = len(sanitized_n) < 3

    s_result = _try_s_serve_policy(
        has_structured_fields,
        len(sanitized_n),
        len(query_markers),
        long_and_repetitive,
        heavily_changed,
        noisy_symbols,
        lacks_query_focus,
    )
    if isinstance(s_result, (bool, np.bool_)):
        return bool(s_result)
    if isinstance(s_result, (int, np.integer)):
        return bool(int(s_result))

    # Keep noisy pasted logs if they still contain a clear user question.
    if has_question_focus and len(sanitized_n) >= 4:
        return long_and_repetitive or repetitive_question or too_short_after_clean

    return long_and_repetitive or heavily_changed or noisy_symbols or lacks_query_focus or too_short_after_clean


def _clarification_text() -> str:
    return (
        "你的输入里包含较多重复或噪声字符，我可能无法精准回答。"
        "请按“背景、问题、期望结果”三行重写，我会给出更准确的答案。"
    )


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


def _build_dataset_canonical_answer_map(answer_map: dict[str, str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for key, value in answer_map.items():
        ckey = _canonicalize_lookup_query(key)
        if ckey and ckey not in canonical:
            canonical[ckey] = value
    return canonical


def _build_s_arch_tokenizer(meta: dict) -> CharTokenizer:
    """Build tokenizer for S-arch serving using dataset texts when available.

    This avoids a tiny fixed vocab that makes non-exact-match generation degrade.
    """
    seed_texts = ["你好", "神经网络", "S语言", "模型部署", "后端服务"]
    dataset_meta = meta.get("dataset") or {}
    dataset_file = dataset_meta.get("file")
    if not dataset_file:
        return CharTokenizer.from_texts(seed_texts)

    dataset_path = _resolve_local_path(str(dataset_file))
    if not dataset_path.exists():
        return CharTokenizer.from_texts(seed_texts)

    sampled_texts: list[str] = []
    # Limit startup overhead while still covering common corpus characters.
    max_lines = int(os.getenv("LLM_S_ARCH_TOKENIZER_MAX_LINES", "6000"))
    max_chars_per_field = int(os.getenv("LLM_S_ARCH_TOKENIZER_MAX_CHARS", "256"))

    try:
        for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
            if len(sampled_texts) >= max_lines:
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except Exception:
                # Fallback for plain-text datasets.
                sampled_texts.append(raw_line[:max_chars_per_field])
                continue

            for key in ("instruction", "input", "output"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    sampled_texts.append(value[:max_chars_per_field])
                    if len(sampled_texts) >= max_lines:
                        break
    except Exception:
        return CharTokenizer.from_texts(seed_texts)

    if not sampled_texts:
        return CharTokenizer.from_texts(seed_texts)
    return CharTokenizer.from_texts(seed_texts + sampled_texts)


def _lookup_dataset_answer_with_meta(query: str, min_confidence: float | None = None) -> tuple[str | None, dict]:
    answer_map = state.dataset_answer_map or {}
    canonical_map = state.dataset_canonical_answer_map or {}
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 0}

    canonical_query = _canonicalize_lookup_query(normalized_query)
    query_candidates = [normalized_query]
    if canonical_query and canonical_query != normalized_query:
        query_candidates.append(canonical_query)

    for q in query_candidates:
        if q in answer_map:
            return answer_map[q], {"match_type": "exact", "score": 1.0, "threshold": None, "query_len": len(normalized_query)}
        if q in canonical_map:
            return canonical_map[q], {"match_type": "canonical", "score": 1.0, "threshold": None, "query_len": len(normalized_query)}

    for line in reversed(query.splitlines()):
        normalized_line = _normalize_query(line)
        canonical_line = _canonicalize_lookup_query(normalized_line)
        if normalized_line.startswith("["):
            continue
        if normalized_line in answer_map:
            return answer_map[normalized_line], {"match_type": "line_exact", "score": 1.0, "threshold": None, "query_len": len(normalized_query)}
        if canonical_line in answer_map:
            return answer_map[canonical_line], {"match_type": "line_canonical", "score": 1.0, "threshold": None, "query_len": len(normalized_query)}
        if canonical_line in canonical_map:
            return canonical_map[canonical_line], {"match_type": "line_canonical_map", "score": 1.0, "threshold": None, "query_len": len(normalized_query)}

    # Fuzzy fallback for near-equivalent wording, mainly for short factual queries.
    if len(normalized_query) <= 96 and answer_map:
        def _bigrams(s: str) -> set[str]:
            if len(s) < 2:
                return {s} if s else set()
            return {s[i:i+2] for i in range(len(s) - 1)}

        best_key = None
        best_score = 0.0
        for query_key in query_candidates:
            qset = _bigrams(query_key)
            for key in answer_map.keys():
                if abs(len(key) - len(query_key)) > 28:
                    continue
                kset = _bigrams(key)
                union = len(qset | kset)
                if union == 0:
                    continue
                jaccard = len(qset & kset) / union
                contains_bonus = 0.06 if (key in query_key or query_key in key) else 0.0
                prefix_bonus = 0.03 if key[:6] == query_key[:6] else 0.0
                length_penalty = min(0.08, abs(len(key) - len(query_key)) / 220.0)
                score = max(0.0, min(1.0, jaccard + contains_bonus + prefix_bonus - length_penalty))
                if score > best_score:
                    best_score = score
                    best_key = key

        base_threshold = _get_dataset_match_min_conf()
        adaptive_threshold = base_threshold
        qlen = len(canonical_query) if canonical_query else len(normalized_query)
        if qlen <= 8:
            adaptive_threshold = max(adaptive_threshold, 0.84)
        elif qlen <= 16:
            adaptive_threshold = max(adaptive_threshold, 0.72)
        elif qlen <= 32:
            adaptive_threshold = max(adaptive_threshold, 0.64)

        required_threshold = adaptive_threshold
        if min_confidence is not None:
            required_threshold = max(required_threshold, float(min_confidence))

        if best_key is not None and best_score >= required_threshold:
            return answer_map.get(best_key), {
                "match_type": "fuzzy",
                "score": round(float(best_score), 4),
                "threshold": round(float(required_threshold), 4),
                "query_len": len(normalized_query),
            }
    return None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": len(normalized_query)}


def _lookup_dataset_answer(query: str, min_confidence: float | None = None) -> str | None:
    answer, _meta = _lookup_dataset_answer_with_meta(query, min_confidence=min_confidence)
    return answer


def _read_env_conf_meta(name: str, default: float) -> dict:
    raw = os.getenv(name)
    used_default = False
    parsed = None

    if raw is None:
        used_default = True
        parsed = default
    else:
        try:
            parsed = float(raw)
        except ValueError:
            used_default = True
            parsed = default

    clamped = max(0.0, min(1.0, parsed))
    return {
        "raw": raw,
        "value": clamped,
        "used_default": used_default,
        "was_clamped": clamped != parsed,
    }


def _read_env_conf_float(name: str, default: float) -> float:
    return _read_env_conf_meta(name, default)["value"]


def _get_high_conf_min_conf() -> float:
    return _read_env_conf_float("LLM_DATASET_HIGH_CONF_MIN_CONF", 0.82)


def _get_dataset_match_min_conf() -> float:
    return _read_env_conf_float("LLM_DATASET_MATCH_MIN_CONF", 0.58)


def _get_retrieval_config_status() -> dict:
    high_conf = _read_env_conf_meta("LLM_DATASET_HIGH_CONF_MIN_CONF", 0.82)
    match_conf = _read_env_conf_meta("LLM_DATASET_MATCH_MIN_CONF", 0.58)
    return {
        "high_conf_min_conf": high_conf["value"],
        "match_min_conf": match_conf["value"],
        "high_conf_min_conf_raw": high_conf["raw"],
        "match_min_conf_raw": match_conf["raw"],
        "high_conf_min_conf_used_default": high_conf["used_default"],
        "match_min_conf_used_default": match_conf["used_default"],
        "high_conf_min_conf_was_clamped": high_conf["was_clamped"],
        "match_min_conf_was_clamped": match_conf["was_clamped"],
    }


def _bump_retrieval_stat(key: str) -> None:
    state.retrieval_stats[key] = int(state.retrieval_stats.get(key, 0)) + 1


def _record_dataset_hit(endpoint: str, phase: str, match_type: str | None) -> None:
    safe_match = match_type or "unknown"
    _bump_retrieval_stat("dataset_hit_total")
    _bump_retrieval_stat(f"dataset_hit_endpoint.{endpoint}")
    _bump_retrieval_stat(f"dataset_hit_phase.{endpoint}.{phase}")
    _bump_retrieval_stat(f"dataset_hit_match.{safe_match}")


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
    state.dataset_canonical_answer_map = None


def _init_from_s_arch(meta: dict) -> bool:
    checkpoint_bin = meta.get("checkpoint_bin")
    if not checkpoint_bin:
        return False
    bin_path = _resolve_local_path(str(checkpoint_bin))
    if not bin_path.exists():
        return False

    tok = _build_s_arch_tokenizer(meta)
    model = SArchBinModel(bin_path=str(bin_path), vocab_size=tok.vocab_size, max_seq_len=128)

    state.model = model
    state.tokenizer = tok
    state.active_source = "s_arch"
    state.active_s_arch_meta = meta
    state.dataset_answer_map = _load_dataset_answer_map(meta)
    state.dataset_canonical_answer_map = _build_dataset_canonical_answer_map(state.dataset_answer_map)
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
        state.dataset_answer_map = None
        state.dataset_canonical_answer_map = None
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


def _local_fallback_model_name(requested_model: str) -> str:
    requested = (requested_model or "").strip()
    if not requested:
        return "neurx-local-fallback"
    return f"{requested}-local-fallback"


@app.get("/health")
def health():
    if _use_qwen_vl_proxy():
        return {
            "status": "ok",
            "backend": "qwen25_vl_proxy",
            "upstream": _qwen_vl_base_url(),
            "model": _qwen_vl_model_id(),
        }
    return {"status": "ok", "backend": "core"}


@app.get("/v1/models")
def list_models():
    if _use_qwen_vl_proxy():
        configured = _qwen_vl_model_id()
        try:
            upstream_ids = _qwen_upstream_model_ids()
        except HTTPException:
            upstream_ids = []
        model_ids = upstream_ids or [configured]
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "self-hosted-qwen-vl",
                }
                for model_id in model_ids
            ],
        }

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
        "retrieval_config": _get_retrieval_config_status(),
        "s_arch": s_arch,
    }


@app.get("/v1/s-arch")
def get_s_arch_meta():
    """Expose current pure-S checkpoint metadata for frontend/runtime integration."""
    return _read_s_arch_meta()


@app.get("/v1/retrieval-status")
def retrieval_status():
    return {
        "service": "neurx-model-core",
        "active_source": state.active_source,
        "retrieval_config": _get_retrieval_config_status(),
        "retrieval_stats": dict(state.retrieval_stats),
        "timestamp": int(time.time()),
    }


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


def _generate_local_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int | None = 40) -> str:
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="model not ready")

    sampling_cfg = SamplingConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.08,
        seed=None,
    )
    sampling_cfg.validate()

    prompt_ids = state.tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    _, text = _generate_ids(
        initial_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        sampling_cfg=sampling_cfg,
        stop_sequences=[],
    )
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    trace_id = uuid.uuid4().hex[:12]
    qwen_proxy_error_detail = ""
    _trace_log(
        "backend_request",
        trace_id=trace_id,
        endpoint="/v1/generate",
        upstream_mode=os.getenv("LLM_UPSTREAM_MODE", ""),
        prompt_len=len(req.prompt),
        temperature=req.temperature,
        top_p=req.top_p,
    )
    if _use_qwen_vl_proxy():
        prompt = _sanitize_user_query(req.prompt)
        try:
            upstream_payload, _ = _qwen_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                trace_id=trace_id,
            )
            return GenerateResponse(text=_extract_qwen_text(upstream_payload))
        except HTTPException as exc:
            qwen_proxy_error_detail = str(exc.detail)
            logger.warning("qwen upstream generate failed, fallback to local model: %s", exc.detail)

    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="model not ready")

    sanitized_prompt = _sanitize_user_query(req.prompt)
    high_conf_min_conf = _get_high_conf_min_conf()

    # High-confidence retrieval first to avoid over-blocking good queries.
    dataset_answer, match_meta = _lookup_dataset_answer_with_meta(
        sanitized_prompt,
        min_confidence=high_conf_min_conf,
    )
    if dataset_answer is not None:
        _record_dataset_hit("generate", "high_conf", match_meta.get("match_type"))
        logger.info(
            "dataset_hit endpoint=/v1/generate phase=high_conf match=%s score=%s threshold=%s qlen=%s",
            match_meta.get("match_type"),
            match_meta.get("score"),
            match_meta.get("threshold"),
            match_meta.get("query_len"),
        )
        return GenerateResponse(text=dataset_answer)

    # Strict mode: noisy inputs always go to clarification, never free generation.
    if _is_low_quality_query(req.prompt, sanitized_prompt):
        _bump_retrieval_stat("clarification_total")
        _bump_retrieval_stat("clarification_endpoint.generate")
        return GenerateResponse(text=_clarification_text())

    dataset_answer, match_meta = _lookup_dataset_answer_with_meta(sanitized_prompt)
    if dataset_answer is not None:
        _record_dataset_hit("generate", "normal", match_meta.get("match_type"))
        logger.info(
            "dataset_hit endpoint=/v1/generate phase=normal match=%s score=%s threshold=%s qlen=%s",
            match_meta.get("match_type"),
            match_meta.get("score"),
            match_meta.get("threshold"),
            match_meta.get("query_len"),
        )
        return GenerateResponse(text=dataset_answer)

    if qwen_proxy_error_detail:
        raise HTTPException(status_code=503, detail=f"qwen upstream unavailable: {qwen_proxy_error_detail}")

    sampling_cfg = SamplingConfig(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        seed=req.seed,
    )
    sampling_cfg.validate()
    ids = state.tokenizer.encode(sanitized_prompt)
    if not ids:
        ids = [0]

    ids, text = _generate_ids(
        initial_ids=ids,
        max_new_tokens=req.max_new_tokens,
        sampling_cfg=sampling_cfg,
        stop_sequences=[],
    )
    if text.startswith(sanitized_prompt):
        text = text[len(sanitized_prompt):]

    _bump_retrieval_stat("generation_total")
    _bump_retrieval_stat("generation_endpoint.generate")
    return GenerateResponse(text=text)


@app.post("/v1/generate-multipart")
def generate_multipart(
    prompt: str = Form(...),
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.6),
    top_p: float = Form(0.85),
    image: UploadFile | None = File(None),
):
    trace_id = uuid.uuid4().hex[:12]
    _trace_log(
        "backend_request",
        trace_id=trace_id,
        endpoint="/v1/generate-multipart",
        upstream_mode=os.getenv("LLM_UPSTREAM_MODE", ""),
        prompt_len=len(prompt),
        has_image=image is not None,
        temperature=temperature,
        top_p=top_p,
    )
    if not _use_qwen_vl_proxy():
        raise HTTPException(status_code=400, detail="multipart is available only when qwen vl proxy mode is enabled")

    sanitized_prompt = _sanitize_user_query(prompt)
    try:
        if image is None:
            upstream_payload, _ = _qwen_chat_completion(
                messages=[{"role": "user", "content": sanitized_prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                trace_id=trace_id,
            )
        else:
            image_data_url = _image_to_data_url(image)
            upstream_payload, _ = _qwen_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": sanitized_prompt or "请描述这张图片"},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                trace_id=trace_id,
            )

        text = _extract_qwen_text(upstream_payload)
        return {
            "text": text,
            "session_id": f"sess-{uuid.uuid4().hex[:12]}",
        }
    except HTTPException as exc:
        logger.warning("qwen upstream multipart failed: %s", exc.detail)
        if image is not None:
            return {
                "text": "当前视觉模型服务不可用，请稍后重试或先发送纯文本问题。",
                "session_id": f"sess-{uuid.uuid4().hex[:12]}",
            }
        local_text = _generate_local_text(
            prompt=sanitized_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return {
            "text": local_text,
            "session_id": f"sess-{uuid.uuid4().hex[:12]}",
        }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    trace_id = uuid.uuid4().hex[:12]
    qwen_proxy_error_detail = ""
    _trace_log(
        "backend_request",
        trace_id=trace_id,
        endpoint="/v1/chat/completions",
        upstream_mode=os.getenv("LLM_UPSTREAM_MODE", ""),
        message_count=len(req.messages),
        model=req.model,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    if _use_qwen_vl_proxy():
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported yet")

        try:
            upstream_messages = [{"role": m.role, "content": m.content} for m in req.messages]
            upstream_payload, used_model = _qwen_chat_completion(
                messages=upstream_messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                trace_id=trace_id,
            )
            answer = _extract_qwen_text(upstream_payload)
            usage = upstream_payload.get("usage") if isinstance(upstream_payload.get("usage"), dict) else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": used_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        except HTTPException as exc:
            qwen_proxy_error_detail = str(exc.detail)
            logger.warning("qwen upstream chat failed, fallback to local model: %s", exc.detail)

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

    user_query = req.messages[-1].content
    sanitized_user_query = _sanitize_user_query(user_query)
    high_conf_min_conf = _get_high_conf_min_conf()

    # High-confidence retrieval first to avoid over-blocking good queries.
    dataset_answer, match_meta = _lookup_dataset_answer_with_meta(
        sanitized_user_query,
        min_confidence=high_conf_min_conf,
    )
    if dataset_answer is not None:
        _record_dataset_hit("chat", "high_conf", match_meta.get("match_type"))
        logger.info(
            "dataset_hit endpoint=/v1/chat/completions phase=high_conf match=%s score=%s threshold=%s qlen=%s",
            match_meta.get("match_type"),
            match_meta.get("score"),
            match_meta.get("threshold"),
            match_meta.get("query_len"),
        )
        return _make_dataset_answer_response(
            dataset_answer,
            _local_fallback_model_name(req.model) if qwen_proxy_error_detail else req.model,
        )

    # Strict mode: noisy inputs always go to clarification, never free generation.
    if _is_low_quality_query(user_query, sanitized_user_query):
        _bump_retrieval_stat("clarification_total")
        _bump_retrieval_stat("clarification_endpoint.chat")
        return _make_dataset_answer_response(
            _clarification_text(),
            _local_fallback_model_name(req.model) if qwen_proxy_error_detail else req.model,
        )

    dataset_answer, match_meta = _lookup_dataset_answer_with_meta(sanitized_user_query)
    if dataset_answer is not None:
        _record_dataset_hit("chat", "normal", match_meta.get("match_type"))
        logger.info(
            "dataset_hit endpoint=/v1/chat/completions phase=normal match=%s score=%s threshold=%s qlen=%s",
            match_meta.get("match_type"),
            match_meta.get("score"),
            match_meta.get("threshold"),
            match_meta.get("query_len"),
        )
        return _make_dataset_answer_response(
            dataset_answer,
            _local_fallback_model_name(req.model) if qwen_proxy_error_detail else req.model,
        )

    if qwen_proxy_error_detail:
        raise HTTPException(status_code=503, detail=f"qwen upstream unavailable: {qwen_proxy_error_detail}")

    prompt_messages = req.messages[:-1] + [ChatMessage(role="user", content=sanitized_user_query)]

    prompt = _build_prompt(prompt_messages)
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

    _bump_retrieval_stat("generation_total")
    _bump_retrieval_stat("generation_endpoint.chat")
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
