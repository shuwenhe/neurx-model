"""自研后端训练主链路（使用 NeurX 框架）"""

import argparse
import os
import pickle

import numpy as np

try:
    # Try using NeurX models first
    from app.core.models_neurx import NeurXChatModel
    from neurx.optim import Adam as AdamW
    import neurx as nn
    BACKEND = "neurx"
except ImportError:
    # Fallback to tensor models if available
    try:
        from app.core.models import TransformerLM
        from tensor.core.optim import AdamW, clip_grad_norm
        BACKEND = "tensor"
    except ImportError:
        raise ImportError(
            "Neither 'neurx' nor 'tensor' framework found. "
            "Please install: pip install /home/shuwen/neurx"
        )

from app.core.tokenizer import CharTokenizer

# Helper for gradient clipping in NeurX
def clip_grad_norm(parameters, max_norm):
    """Clip gradient norm for stability"""
    total_norm = 0.0
    for p in parameters:
        if hasattr(p, 'grad') and p.grad is not None:
            param_norm = np.linalg.norm(p.grad.flatten())
            total_norm += param_norm ** 2
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad *= clip_coef
    
    return float(total_norm)


def _sample_corpus():
    return [
        "北京是中国的首都，位于华北平原中部。",
        "人工智能正在改变世界，推动社会进步。",
        "语言模型可以生成文本，理解语义信息。",
        "机器学习需要数据和算力，还需要算法。",
        "模型训练要关注损失函数下降，学习率很重要。",
        "深度学习在计算机视觉领域取得了巨大成就。",
        "自然语言处理技术在电商和搜索引擎中广泛应用。",
        "神经网络通过反向传播算法实现参数更新。",
        "数据预处理对模型训练的质量有重要影响。",
        "特征工程是机器学习中的关键环节。",
        "超参数调优需要在验证集上反复测试。",
        "正则化技术可以防止模型过拟合。",
        "批归一化加快了神经网络的训练速度。",
        "注意力机制大幅提高了序列模型的性能。",
        "transformers架构革命了自然语言处理。",
        "迁移学习让我们可以利用预训练模型。",
        "多任务学习能够提高模型的泛化能力。",
        "知识蒸馏可以将大模型压缩为小模型。",
        "梯度裁剪防止了训练过程中的梯度爆炸。",
        "学习率预热有助于模型的稳定训练。",
        "对数据进行增强可以扩大训练集的规模。",
        "交叉验证是评估模型性能的重要方法。",
        "集成学习通过多个模型的组合提高准确率。",
        "强化学习让机器可以通过奖励学习策略。",
        "生成对抗网络可以生成逼真的虚假数据。",
    ] * 20


def _make_batches(token_ids, batch_size, seq_len):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    max_start = len(token_ids) - seq_len - 1
    while True:
        starts = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([token_ids[s:s + seq_len] for s in starts], axis=0)
        y = np.stack([token_ids[s + 1:s + seq_len + 1] for s in starts], axis=0)
        yield x, y


def _cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def train_core(
    batch_size=8,
    epochs=3,
    learning_rate=3e-3,
    output="checkpoints/model_core.pkl",
    grad_clip=1.0,
    warmup_ratio=0.1,
    min_lr_ratio=0.1,
    use_rmsnorm=False,
    use_swiglu=False,
    use_rope=False,
    rope_theta=10000.0,
):
    np.random.seed(42)

    texts = _sample_corpus()
    tokenizer = CharTokenizer.from_texts(texts)
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer.encode(t))

    seq_len = 64
    
    # Create model using appropriate backend
    if BACKEND == "neurx":
        # Use NeurX model (parameter names: hidden_dim, num_layers, num_heads)
        model = NeurXChatModel(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            max_seq_len=seq_len,
            dropout=0.1,
        )
    else:
        # Use tensor model (parameter names: n_embd, n_layers, n_heads)
        model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            n_embd=512,
            n_layers=4,
            n_heads=8,
            max_seq_len=seq_len,
            dropout=0.1,
            use_rmsnorm=use_rmsnorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    steps_per_epoch = 10
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr = learning_rate * min_lr_ratio
    batches = _make_batches(all_tokens, batch_size=batch_size, seq_len=seq_len)

    losses = []
    for step in range(total_steps):
        x, y = next(batches)
        optimizer.zero_grad()
        
        if BACKEND == "neurx":
            # NeurX forward pass
            output = model(x, y)  # Returns dict with 'logits', 'loss', 'hidden_states'
            if isinstance(output, dict):
                logits = output['logits']
                loss = output.get('loss')
                if loss is None:
                    # Reshape for loss computation if not provided by model
                    logits_reshaped = logits.reshape(-1, tokenizer.vocab_size)
                    targets = y.reshape(-1)
                    loss = nn.losses.cross_entropy(logits_reshaped, targets)
            else:
                # If output is just a tensor (for simpler models)
                logits = output
                # Reshape for loss computation
                logits_reshaped = logits.reshape(-1, tokenizer.vocab_size)
                targets = y.reshape(-1)
                loss = nn.losses.cross_entropy(logits_reshaped, targets)
        else:
            # Tensor model forward pass
            _, loss = model(x, y)
        
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), grad_clip)
        optimizer.lr = _cosine_lr(step, total_steps, learning_rate, min_lr, warmup_steps)
        optimizer.step()
        
        # Extract loss value appropriately
        loss_val = float(loss) if hasattr(loss, '__float__') else float(loss.item()) if hasattr(loss, 'item') else loss
        losses.append(loss_val)
        
        if step % 5 == 0 or step == total_steps - 1:
            print(
                f"step {step+1}/{total_steps}: "
                f"loss={loss_val:.4f}, lr={optimizer.lr:.6f}, grad_norm={grad_norm:.4f}"
            )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    
    # Save checkpoint
    if BACKEND == "neurx":
        # Save NeurX model state
        state_dict = {}
        for i, p in enumerate(model.parameters()):
            state_dict[f"param_{i}"] = p.data.copy() if hasattr(p.data, 'copy') else np.array(p.data)
    else:
        # Save tensor model state
        state_dict = {f"param_{i}": p.data.copy() for i, p in enumerate(model.parameters())}
    
    payload = {
        "backend": BACKEND,
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": 512,
            "n_layers": 4,
            "n_heads": 8,
            "max_seq_len": seq_len,
            "dropout": 0.1,
            "use_rmsnorm": use_rmsnorm,
            "use_swiglu": use_swiglu,
            "use_rope": use_rope,
            "rope_theta": rope_theta,
            "state_dict": state_dict,
        },
        "tokenizer": tokenizer.to_dict(),
        "metrics": {
            "start_loss": losses[0],
            "end_loss": losses[-1],
        },
    }
    with open(output, "wb") as f:
        pickle.dump(payload, f)

    print("✅ core 训练完成")
    print(f"   backend={BACKEND}")
    print(f"   start_loss={losses[0]:.4f}, end_loss={losses[-1]:.4f}")
    print(f"   checkpoint={output}")


def main():
    parser = argparse.ArgumentParser(description="core backend training")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--output", type=str, default="checkpoints/model_core.pkl")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--use-rmsnorm", action="store_true")
    parser.add_argument("--use-swiglu", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--checkpoint", type=str, default="", help="保留参数兼容，当前未使用")
    args = parser.parse_args()

    train_core(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output=args.output,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        use_rmsnorm=args.use_rmsnorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
    )


if __name__ == "__main__":
    main()
