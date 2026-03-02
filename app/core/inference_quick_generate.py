"""core 快速文本生成测试实现"""

import os
import pickle

import numpy as np

from app.core.models import TinyLM, TransformerLM
from app.core.sampling import SamplingConfig, sample_next_token
from app.core.tokenizer import CharTokenizer


def _infer_transformer_config(model_cfg):
    state_dict = model_cfg["state_dict"]
    n_embd = int(model_cfg["n_embd"])

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
        "use_rmsnorm": bool(model_cfg.get("use_rmsnorm", False)),
        "use_swiglu": bool(model_cfg.get("use_swiglu", False)),
        "use_rope": bool(model_cfg.get("use_rope", False)),
        "rope_theta": float(model_cfg.get("rope_theta", 10000.0)),
    }


def quick_test():
    checkpoint_path = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")

    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型检查点文件不存在: {checkpoint_path}\n")
        print("请先训练模型:")
        print("  make train-core")
        print("  make train-chinese")
        return

    print("🚀 快速生成测试")
    print("后端: core")
    print("=" * 60)

    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
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
            use_rmsnorm=transformer_cfg["use_rmsnorm"],
            use_swiglu=transformer_cfg["use_swiglu"],
            use_rope=transformer_cfg["use_rope"],
            rope_theta=transformer_cfg["rope_theta"],
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

    test_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In a world where",
    ]

    test_configs = [
        {"name": "保守模式", "temp": 0.7, "tokens": 80, "top_p": 0.85, "top_k": 20, "rp": 1.05},
        {"name": "平衡模式", "temp": 0.8, "tokens": 120, "top_p": 0.92, "top_k": 40, "rp": 1.08},
        {"name": "创意模式", "temp": 1.0, "tokens": 150, "top_p": 0.98, "top_k": 80, "rp": 1.10},
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"📝 提示词: \"{prompt}\"")
        print(f"{'='*60}")

        for cfg in test_configs:
            print(
                f"\n🔧 {cfg['name']} (temp={cfg['temp']}, top_p={cfg['top_p']}, "
                f"top_k={cfg['top_k']}, rp={cfg['rp']}, tokens={cfg['tokens']})"
            )
            print("-" * 60)
            sampling_cfg = SamplingConfig(
                temperature=cfg["temp"],
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
                repetition_penalty=cfg["rp"],
                seed=42,
            )
            rng = np.random.default_rng(sampling_cfg.seed)

            ids = tokenizer.encode(prompt)
            if not ids:
                ids = [0]

            if hasattr(model, "generate"):
                generated_ids = model.generate(
                    ids,
                    max_new_tokens=cfg["tokens"],
                    temperature=cfg["temp"],
                    top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                    repetition_penalty=cfg["rp"],
                    seed=42,
                )
                ids = generated_ids if isinstance(generated_ids, list) else generated_ids[0].tolist()
            else:
                max_ctx = getattr(model, "max_seq_len", None)
                for _ in range(cfg["tokens"]):
                    ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
                    x = np.array([ctx], dtype=np.int64)
                    logits, _ = model(x, None)
                    next_id = sample_next_token(
                        logits.data[0, -1],
                        token_ids=ids,
                        cfg=sampling_cfg,
                        rng=rng,
                    )
                    ids.append(next_id)

            generated_text = tokenizer.decode(ids)
            print(generated_text)

    print("\n" + "=" * 60)
    print("✅ 测试完成！选择效果最好的参数在 generate.py 中使用")


if __name__ == "__main__":
    quick_test()
