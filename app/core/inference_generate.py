"""core 文本生成实现"""

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


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ 模型检查点文件不存在: {checkpoint_path}\n\n"
            f"请先训练模型:\n"
            f"  make train-core\n"
            f"  make train-chinese\n"
        )

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
    return model, tokenizer


def generate_text(
    prompt,
    model,
    tokenizer,
    max_new_tokens=120,
    temperature=0.8,
    top_k=None,
    top_p=1.0,
    repetition_penalty=1.0,
    seed=None,
):
    if hasattr(model, "eval"):
        model.eval()

    sampling_cfg = SamplingConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    sampling_cfg.validate()
    rng = np.random.default_rng(seed)

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    if hasattr(model, "generate"):
        generated_ids = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        ids = generated_ids if isinstance(generated_ids, list) else generated_ids[0].tolist()
    else:
        max_ctx = getattr(model, "max_seq_len", None)
        for _ in range(max_new_tokens):
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
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    return generated_text.strip()


def main():
    checkpoint_path = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    print(f"加载 core 模型: {checkpoint_path}")

    model, tokenizer = load_model(checkpoint_path)
    print(f"模型参数量: {sum(p.data.size for p in model.parameters())/1e6:.2f}M")

    presets = {
        "1": {"name": "保守模式", "temp": 0.65, "tokens": 80, "top_p": 0.85, "top_k": 20, "rp": 1.05},
        "2": {"name": "平衡模式", "temp": 0.80, "tokens": 120, "top_p": 0.92, "top_k": 40, "rp": 1.08},
        "3": {"name": "创意模式", "temp": 1.00, "tokens": 160, "top_p": 0.98, "top_k": 80, "rp": 1.10},
    }

    print("\n" + "=" * 50)
    print("文本生成器(core) (输入 'quit' 退出)")
    print("=" * 50)
    current_preset = presets["2"]

    while True:
        prompt = input("\n请输入提示词 (或输入1/2/3切换模式): ")
        if prompt.lower() == "quit":
            break
        if prompt in presets:
            current_preset = presets[prompt]
            print(f"✓ 已切换到: {current_preset['name']}")
            continue
        if not prompt.strip():
            continue

        generated = generate_text(
            prompt,
            model,
            tokenizer,
            max_new_tokens=current_preset["tokens"],
            temperature=current_preset["temp"],
            top_k=current_preset["top_k"],
            top_p=current_preset["top_p"],
            repetition_penalty=current_preset["rp"],
        )
        print(f"\n生成结果 [{current_preset['name']}]:\n{generated}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
