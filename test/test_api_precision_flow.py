import pytest

from app.api import serve_core


def _set_minimal_ready_state(monkeypatch):
    monkeypatch.setattr(serve_core.state, "model", object())
    monkeypatch.setattr(serve_core.state, "tokenizer", object())


def test_generate_high_conf_hit_short_circuits_low_quality(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    calls = []

    def fake_lookup(_query, min_confidence=None):
        calls.append(min_confidence)
        if min_confidence == 0.82:
            return (
                "命中高置信答案",
                {"match_type": "exact", "score": 1.0, "threshold": None, "query_len": 4},
            )
        return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 4})

    def should_not_call(*_args, **_kwargs):
        raise AssertionError("_is_low_quality_query should not run after high-confidence hit")

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)
    monkeypatch.setattr(serve_core, "_is_low_quality_query", should_not_call)

    res = serve_core.generate(serve_core.GenerateRequest(prompt="中国首都在哪里"))
    assert res.text == "命中高置信答案"
    assert calls == [0.82]


def test_generate_low_quality_short_circuits_normal_lookup(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    calls = []

    def fake_lookup(_query, min_confidence=None):
        calls.append(min_confidence)
        if min_confidence == 0.82:
            return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 3})
        raise AssertionError("normal lookup should not run when low quality is true")

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)
    monkeypatch.setattr(serve_core, "_is_low_quality_query", lambda *_args: True)

    res = serve_core.generate(serve_core.GenerateRequest(prompt="啊"))
    assert res.text == serve_core._clarification_text()
    assert calls == [0.82]


def test_low_quality_query_allows_noisy_question_with_clear_focus():
    prompt = "错误日志 ###@@@ /tmp/main.cpp:42 std::bad_alloc !!!!! 为什么总是回复这个，应该怎么处理？"
    sanitized = serve_core._sanitize_user_query(prompt)

    assert serve_core._is_low_quality_query(prompt, sanitized) is False


def test_low_quality_query_keeps_blocking_repetitive_noise():
    prompt = "为什么为什么为什么为什么为什么为什么为什么为什么为什么为什么????!!!!"
    sanitized = serve_core._sanitize_user_query(prompt)

    assert serve_core._is_low_quality_query(prompt, sanitized) is True


def test_chat_high_conf_hit_returns_dataset_answer(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    calls = []

    def fake_lookup(_query, min_confidence=None):
        calls.append(min_confidence)
        if min_confidence == 0.82:
            return (
                "巴黎",
                {"match_type": "canonical", "score": 1.0, "threshold": None, "query_len": 6},
            )
        return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 6})

    def should_not_call(*_args, **_kwargs):
        raise AssertionError("_is_low_quality_query should not run after high-confidence hit")

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)
    monkeypatch.setattr(serve_core, "_is_low_quality_query", should_not_call)

    req = serve_core.ChatCompletionsRequest(
        model="core-transformer",
        messages=[serve_core.ChatMessage(role="user", content="法国首都是哪里")],
    )
    res = serve_core.chat_completions(req)
    assert res["choices"][0]["message"]["content"] == "巴黎"
    assert calls == [0.82]


def test_generate_logs_structured_fields_on_high_conf_hit(monkeypatch, caplog):
    _set_minimal_ready_state(monkeypatch)

    def fake_lookup(_query, min_confidence=None):
        if min_confidence == 0.82:
            return (
                "命中",
                {"match_type": "exact", "score": 1.0, "threshold": None, "query_len": 5},
            )
        return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 5})

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)

    with caplog.at_level("INFO"):
        res = serve_core.generate(serve_core.GenerateRequest(prompt="中国首都在哪"))

    assert res.text == "命中"
    msg = "\n".join(record.getMessage() for record in caplog.records)
    assert "dataset_hit endpoint=/v1/generate phase=high_conf" in msg
    assert "match=exact" in msg
    assert "score=1.0" in msg
    assert "threshold=None" in msg
    assert "qlen=5" in msg


def test_chat_logs_structured_fields_on_normal_hit(monkeypatch, caplog):
    _set_minimal_ready_state(monkeypatch)

    def fake_lookup(_query, min_confidence=None):
        if min_confidence == 0.82:
            return (None, {"match_type": "none", "score": 0.0, "threshold": 0.82, "query_len": 7})
        return (
            "普通命中",
            {"match_type": "fuzzy", "score": 0.91, "threshold": 0.58, "query_len": 7},
        )

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)
    monkeypatch.setattr(serve_core, "_is_low_quality_query", lambda *_args: False)

    req = serve_core.ChatCompletionsRequest(
        model="core-transformer",
        messages=[serve_core.ChatMessage(role="user", content="法国首都是哪里")],
    )
    with caplog.at_level("INFO"):
        res = serve_core.chat_completions(req)

    assert res["choices"][0]["message"]["content"] == "普通命中"
    msg = "\n".join(record.getMessage() for record in caplog.records)
    assert "dataset_hit endpoint=/v1/chat/completions phase=normal" in msg
    assert "match=fuzzy" in msg
    assert "score=0.91" in msg
    assert "threshold=0.58" in msg
    assert "qlen=7" in msg


def test_generate_uses_configured_high_conf_threshold(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    calls = []
    monkeypatch.setenv("LLM_DATASET_HIGH_CONF_MIN_CONF", "0.91")

    def fake_lookup(_query, min_confidence=None):
        calls.append(min_confidence)
        if min_confidence == 0.91:
            return (
                "配置阈值命中",
                {"match_type": "exact", "score": 1.0, "threshold": None, "query_len": 6},
            )
        return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 6})

    def should_not_call(*_args, **_kwargs):
        raise AssertionError("_is_low_quality_query should not run after high-confidence hit")

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)
    monkeypatch.setattr(serve_core, "_is_low_quality_query", should_not_call)

    res = serve_core.generate(serve_core.GenerateRequest(prompt="中国首都在哪里"))
    assert res.text == "配置阈值命中"
    assert calls == [0.91]


def test_high_conf_threshold_parser_clamps_and_fallbacks(monkeypatch):
    monkeypatch.setenv("LLM_DATASET_HIGH_CONF_MIN_CONF", "abc")
    assert serve_core._get_high_conf_min_conf() == 0.82

    monkeypatch.setenv("LLM_DATASET_HIGH_CONF_MIN_CONF", "1.9")
    assert serve_core._get_high_conf_min_conf() == 1.0

    monkeypatch.setenv("LLM_DATASET_HIGH_CONF_MIN_CONF", "-0.2")
    assert serve_core._get_high_conf_min_conf() == 0.0


def test_model_status_exposes_retrieval_config(monkeypatch):
    monkeypatch.setattr(serve_core.state, "model", None)
    monkeypatch.setattr(serve_core.state, "tokenizer", None)
    monkeypatch.setattr(serve_core.state, "active_source", "pickle")
    monkeypatch.setattr(serve_core.state, "active_s_arch_meta", None)
    monkeypatch.setenv("LLM_DATASET_HIGH_CONF_MIN_CONF", "0.9")
    monkeypatch.setenv("LLM_DATASET_MATCH_MIN_CONF", "0.66")

    payload = serve_core.model_status()
    retrieval = payload["retrieval_config"]
    assert retrieval["high_conf_min_conf"] == 0.9
    assert retrieval["match_min_conf"] == 0.66
    assert retrieval["high_conf_min_conf_raw"] == "0.9"
    assert retrieval["match_min_conf_raw"] == "0.66"
    assert retrieval["high_conf_min_conf_used_default"] is False
    assert retrieval["match_min_conf_used_default"] is False
    assert retrieval["high_conf_min_conf_was_clamped"] is False
    assert retrieval["match_min_conf_was_clamped"] is False


def test_model_status_retrieval_config_reports_default_and_clamp(monkeypatch):
    monkeypatch.setattr(serve_core.state, "model", None)
    monkeypatch.setattr(serve_core.state, "tokenizer", None)
    monkeypatch.setattr(serve_core.state, "active_source", "pickle")
    monkeypatch.setattr(serve_core.state, "active_s_arch_meta", None)
    monkeypatch.setenv("LLM_DATASET_HIGH_CONF_MIN_CONF", "bad")
    monkeypatch.setenv("LLM_DATASET_MATCH_MIN_CONF", "1.5")

    payload = serve_core.model_status()
    retrieval = payload["retrieval_config"]
    assert retrieval["high_conf_min_conf"] == 0.82
    assert retrieval["high_conf_min_conf_raw"] == "bad"
    assert retrieval["high_conf_min_conf_used_default"] is True
    assert retrieval["high_conf_min_conf_was_clamped"] is False

    assert retrieval["match_min_conf"] == 1.0
    assert retrieval["match_min_conf_raw"] == "1.5"
    assert retrieval["match_min_conf_used_default"] is False
    assert retrieval["match_min_conf_was_clamped"] is True


def test_retrieval_status_counts_hits(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    monkeypatch.setattr(serve_core.state, "retrieval_stats", {})

    def fake_generate_lookup(_query, min_confidence=None):
        if min_confidence == 0.82:
            return (
                "高置信命中",
                {"match_type": "exact", "score": 1.0, "threshold": None, "query_len": 6},
            )
        return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 6})

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_generate_lookup)
    monkeypatch.setattr(serve_core, "_is_low_quality_query", lambda *_args: False)

    gen_res = serve_core.generate(serve_core.GenerateRequest(prompt="中国首都在哪里"))
    assert gen_res.text == "高置信命中"

    def fake_chat_lookup(_query, min_confidence=None):
        if min_confidence == 0.82:
            return (None, {"match_type": "none", "score": 0.0, "threshold": 0.82, "query_len": 7})
        return (
            "普通命中",
            {"match_type": "fuzzy", "score": 0.9, "threshold": 0.58, "query_len": 7},
        )

    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_chat_lookup)

    chat_req = serve_core.ChatCompletionsRequest(
        model="core-transformer",
        messages=[serve_core.ChatMessage(role="user", content="法国首都是哪里")],
    )
    chat_res = serve_core.chat_completions(chat_req)
    assert chat_res["choices"][0]["message"]["content"] == "普通命中"

    payload = serve_core.retrieval_status()
    stats = payload["retrieval_stats"]
    assert stats["dataset_hit_total"] == 2
    assert stats["dataset_hit_endpoint.generate"] == 1
    assert stats["dataset_hit_endpoint.chat"] == 1
    assert stats["dataset_hit_phase.generate.high_conf"] == 1
    assert stats["dataset_hit_phase.chat.normal"] == 1
    assert stats["dataset_hit_match.exact"] == 1
    assert stats["dataset_hit_match.fuzzy"] == 1


def test_chat_qwen_proxy_failure_returns_local_fallback_model_for_dataset_hit(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    monkeypatch.setattr(serve_core, "_use_qwen_vl_proxy", lambda: True)

    def fail_upstream(**_kwargs):
        raise serve_core.HTTPException(status_code=502, detail="qwen upstream timeout")

    def fake_lookup(_query, min_confidence=None):
        if min_confidence == 0.82:
            return (
                "中国的首都是北京。",
                {"match_type": "exact", "score": 1.0, "threshold": None, "query_len": 7},
            )
        return (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 7})

    monkeypatch.setattr(serve_core, "_qwen_chat_completion", fail_upstream)
    monkeypatch.setattr(serve_core, "_lookup_dataset_answer_with_meta", fake_lookup)

    req = serve_core.ChatCompletionsRequest(
        model="Qwen2.5-VL-7B",
        messages=[serve_core.ChatMessage(role="user", content="中国首都是哪里")],
    )
    res = serve_core.chat_completions(req)

    assert res["model"] == "Qwen2.5-VL-7B-local-fallback"
    assert res["choices"][0]["message"]["content"] == "中国的首都是北京。"


def test_chat_qwen_proxy_failure_does_not_free_generate(monkeypatch):
    _set_minimal_ready_state(monkeypatch)
    monkeypatch.setattr(serve_core, "_use_qwen_vl_proxy", lambda: True)
    monkeypatch.setattr(
        serve_core,
        "_qwen_chat_completion",
        lambda **_kwargs: (_ for _ in ()).throw(serve_core.HTTPException(status_code=502, detail="qwen upstream timeout")),
    )
    monkeypatch.setattr(
        serve_core,
        "_lookup_dataset_answer_with_meta",
        lambda *_args, **_kwargs: (None, {"match_type": "none", "score": 0.0, "threshold": None, "query_len": 10}),
    )
    monkeypatch.setattr(serve_core, "_is_low_quality_query", lambda *_args: False)

    def should_not_generate(*_args, **_kwargs):
        raise AssertionError("_generate_ids should not run when qwen proxy is unavailable")

    monkeypatch.setattr(serve_core, "_generate_ids", should_not_generate)

    req = serve_core.ChatCompletionsRequest(
        model="Qwen2.5-VL-7B",
        messages=[serve_core.ChatMessage(role="user", content="为什么总是回复这个")],
    )

    with pytest.raises(serve_core.HTTPException) as exc_info:
        serve_core.chat_completions(req)

    assert exc_info.value.status_code == 503
    assert "qwen upstream unavailable" in str(exc_info.value.detail)
