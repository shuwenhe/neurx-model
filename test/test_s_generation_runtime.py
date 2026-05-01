import numpy as np

from app.core.gpt_model import GPT
from app.api import serve_core
from app.api.serve_core import SimpleFFNCheckpointModel
from app.core.models import TinyLM, TransformerLM
from app.core.sampling import SamplingConfig, sample_next_token
from app.inference.inference_neurx import ChatModelInference
from app.modeling.config import ModelConfig
from neurx.compile import supports_runtime_function


def test_tinylm_uses_s_lm_head_for_inference(monkeypatch):
    monkeypatch.setenv("NEURX_S_OPS_BACKEND", "auto")
    assert supports_runtime_function("ops", "lm_head_logits")
    model = TinyLM(vocab_size=7, n_embd=4)
    logits, loss = model(np.array([[1, 2, 3]], dtype=np.int64))
    assert loss is None
    assert logits.shape == (1, 3, 7)
    assert getattr(logits, "_runtime_backend", None) == "s"


def test_transformerlm_generation_uses_s_lm_head_and_sampling(monkeypatch):
    monkeypatch.setenv("NEURX_S_OPS_BACKEND", "auto")
    assert supports_runtime_function("ops", "lm_head_logits")
    assert supports_runtime_function("ops", "generation_step")
    model = TransformerLM(
        vocab_size=8,
        n_embd=4,
        n_layers=1,
        n_heads=1,
        max_seq_len=8,
        dropout=0.0,
    )
    logits, cache = model.forward_with_cache(np.array([[1, 2]], dtype=np.int64))
    assert logits.shape == (1, 2, 8)
    assert len(cache) == 1
    assert getattr(logits, "_runtime_backend", None) == "s"
    generated = model.generate([1, 2], max_new_tokens=2, temperature=0.0, top_k=3, seed=123)
    assert len(generated) == 4
    assert all(isinstance(token_id, int) for token_id in generated)


def test_gpt_generation_logits_use_s_lm_head(monkeypatch):
    monkeypatch.setenv("NEURX_S_OPS_BACKEND", "auto")
    config = ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=4, dropout=0.0, bias=False)
    model = GPT(config)
    logits, loss = model(np.array([[1, 2, 3]], dtype=np.int64))
    assert loss is None
    assert logits.shape == (1, 1, 11)
    assert getattr(logits, "_runtime_backend", None) == "s"
    cached_logits, cache = model.forward_with_cache(np.array([[1, 2]], dtype=np.int64))
    assert cached_logits.shape == (1, 2, 11)
    assert len(cache) == 1
    assert getattr(cached_logits, "_runtime_backend", None) == "s"


def test_sampling_uses_s_greedy_generation_step(monkeypatch):
    monkeypatch.setenv("NEURX_S_OPS_BACKEND", "auto")
    assert supports_runtime_function("ops", "generation_step")
    cfg = SamplingConfig(temperature=0.0, top_k=2, top_p=1.0, repetition_penalty=2.0)
    next_id = sample_next_token(
        np.array([0.1, 1.0, 0.2, 0.9], dtype=np.float64),
        token_ids=[1],
        cfg=cfg,
        rng=np.random.default_rng(123),
    )
    assert next_id == 3


class _TokenizerWithoutUnk:
    stoi = {"A": 0, "B": 1}

    def decode(self, ids):
        return "".join("AB"[int(i)] for i in ids)


class _GenerateModel:
    def __init__(self):
        self.generate_called = False

    def generate(self, ids, **kwargs):
        self.generate_called = True
        assert kwargs["use_kv_cache"] is True
        return ids + [1, 1]


class _LoopModel:
    max_seq_len = 8

    def __call__(self, input_ids, targets=None):
        logits = np.array([[[0.0, 10.0]]], dtype=np.float64)
        return logits, None


class _CacheLoopModel:
    max_seq_len = 8

    def __init__(self):
        self.cache_calls = []

    def forward_with_cache(self, input_ids, kv_cache=None):
        self.cache_calls.append((np.asarray(input_ids).copy(), kv_cache))
        logits = np.array([[[0.0, 10.0]]], dtype=np.float64)
        return logits, {"seen": len(self.cache_calls)}


def test_api_generate_ids_uses_model_generate_when_unconstrained(monkeypatch):
    model = _GenerateModel()
    monkeypatch.setattr(serve_core.state, "model", model)
    monkeypatch.setattr(serve_core.state, "tokenizer", _TokenizerWithoutUnk())
    cfg = SamplingConfig(temperature=0.0, top_k=None, top_p=1.0, repetition_penalty=1.0, seed=123)
    ids, text = serve_core._generate_ids([0], 2, cfg, stop_sequences=[])
    assert model.generate_called
    assert ids == [0, 1, 1]
    assert text == "ABB"


def test_api_generate_ids_preserves_manual_loop_for_stop_sequences(monkeypatch):
    monkeypatch.setattr(serve_core.state, "model", _LoopModel())
    monkeypatch.setattr(serve_core.state, "tokenizer", _TokenizerWithoutUnk())
    cfg = SamplingConfig(temperature=0.0, top_k=None, top_p=1.0, repetition_penalty=1.0, seed=123)
    ids, text = serve_core._generate_ids([0], 2, cfg, stop_sequences=["B"])
    assert ids == [0, 1]
    assert text == "A"


def test_api_manual_loop_uses_forward_with_cache_when_available(monkeypatch):
    model = _CacheLoopModel()
    monkeypatch.setattr(serve_core.state, "model", model)
    monkeypatch.setattr(serve_core.state, "tokenizer", _TokenizerWithoutUnk())
    cfg = SamplingConfig(temperature=0.0, top_k=None, top_p=1.0, repetition_penalty=1.0, seed=123)
    ids, text = serve_core._generate_ids([0], 2, cfg, stop_sequences=["Z"])
    assert ids == [0, 1, 1]
    assert text == "ABB"
    assert len(model.cache_calls) == 2
    assert model.cache_calls[0][0].tolist() == [[0]]
    assert model.cache_calls[0][1] is None
    assert model.cache_calls[1][0].tolist() == [[1]]
    assert model.cache_calls[1][1] == {"seen": 1}


class _LegacyInferenceModel:
    def eval(self):
        return self

    def __call__(self, input_ids):
        import neurx

        return {"logits": neurx.Tensor([[[0.0, 10.0]]])}


def test_legacy_neurx_inference_generate_uses_shared_sampling(monkeypatch):
    monkeypatch.setenv("NEURX_S_OPS_BACKEND", "auto")
    inference = ChatModelInference.__new__(ChatModelInference)
    inference.model = _LegacyInferenceModel()
    inference.char_to_id = {"A": 0, "B": 1}
    inference.id_to_char = {0: "A", 1: "B"}
    text = inference.generate("A", max_length=2, temperature=0.0, top_k=None, top_p=1.0)
    assert text == "ABB"


def test_simple_ffn_checkpoint_model_uses_s_runtime(monkeypatch):
    monkeypatch.setenv("NEURX_S_OPS_BACKEND", "auto")
    params = {
        "param_0": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        "param_1": np.array([[0.2, -0.1, 0.3, 0.4], [0.5, 0.6, -0.2, 0.1]], dtype=np.float64),
        "param_2": np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float64),
        "param_3": np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6], [0.7, 0.8]], dtype=np.float64),
        "param_4": np.array([0.05, -0.06], dtype=np.float64),
        "param_5": np.array([[0.3, -0.2, 0.1], [0.4, 0.5, -0.6]], dtype=np.float64),
        "param_6": np.array([0.01, -0.02, 0.03], dtype=np.float64),
    }
    model = SimpleFFNCheckpointModel(params)
    logits = model(np.array([[0, 1]], dtype=np.int64))
    assert logits.shape == (1, 2, 3)
    assert getattr(model, "_runtime_backend", None) == "s"