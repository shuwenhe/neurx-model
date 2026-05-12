from app.api import serve_core


def test_lookup_canonical_variants_hit_same_answer(monkeypatch):
    answer_map = {
        serve_core._normalize_query("中国的首都是什么"): "中国的首都是北京。",
    }
    monkeypatch.setattr(serve_core.state, "dataset_answer_map", answer_map)
    monkeypatch.setattr(
        serve_core.state,
        "dataset_canonical_answer_map",
        serve_core._build_dataset_canonical_answer_map(answer_map),
    )

    assert serve_core._lookup_dataset_answer("中国首都在哪") == "中国的首都是北京。"
    assert serve_core._lookup_dataset_answer("中国首都在哪里") == "中国的首都是北京。"


def test_lookup_with_meta_reports_match_type(monkeypatch):
    answer_map = {
        serve_core._normalize_query("法国首都"): "法国的首都是巴黎。",
    }
    monkeypatch.setattr(serve_core.state, "dataset_answer_map", answer_map)
    monkeypatch.setattr(
        serve_core.state,
        "dataset_canonical_answer_map",
        serve_core._build_dataset_canonical_answer_map(answer_map),
    )

    answer, meta = serve_core._lookup_dataset_answer_with_meta("法国首都是哪里", min_confidence=0.82)
    assert answer == "法国的首都是巴黎。"
    assert meta["match_type"] in {"canonical", "line_canonical", "line_canonical_map", "fuzzy", "exact", "line_exact"}
