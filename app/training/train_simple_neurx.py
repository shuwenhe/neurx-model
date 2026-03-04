"""简化的 NeurX 训练脚本 - 直接使用 NeurX API"""

import argparse
import os
import pickle
import numpy as np

try:
    import neurx
    import neurx.nn as nn
    from neurx.optim import Adam
except ImportError:
    print("❌ NeurX 框架未安装")
    print("请运行: pip install -e /home/shuwen/neurx")
    exit(1)

from app.core.tokenizer import CharTokenizer


class SimpleTransformer(nn.Module):
    """极简 Transformer 模型"""
    
    def __init__(self, vocab_size, hidden_dim=256, num_layers=2, seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Simple feed-forward layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) 张量
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, hidden)
        x = self.embedding(x)
        
        # Feed-forward with residual
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + residual
        
        # Output projection
        x = self.norm2(x)
        logits = self.output(x)  # (batch, seq_len, vocab_size)
        
        return logits


def get_sample_corpus():
    """获取中文语料库样本"""
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
    ] * 20


def tokenize_corpus(texts, tokenizer):
    """分词语料库"""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    return all_tokens


def make_batches(token_ids, batch_size, seq_len):
    """生成训练批次"""
    token_ids = np.array(token_ids, dtype=np.int64)
    max_start = len(token_ids) - seq_len - 1
    
    if max_start <= 0:
        print(f"⚠️  语料库太小 (token数: {len(token_ids)}, 需要 > {seq_len + 1})")
        # 循环补充数据
        token_ids = np.tile(token_ids, (seq_len + 10) // len(token_ids) + 1)
        max_start = len(token_ids) - seq_len - 1
    
    while True:
        starts = np.random.randint(0, max(1, max_start), size=batch_size)
        x = np.stack([token_ids[s:s + seq_len] for s in starts], axis=0)
        y = np.stack([token_ids[s + 1:s + seq_len + 1] for s in starts], axis=0)
        
        yield neurx.Tensor(x), neurx.Tensor(y)


def train_simple_neurx(
    batch_size=4,
    epochs=1,
    learning_rate=1e-4,
    output="checkpoints/model_neurx_simple.pkl",
    seq_len=64,
    hidden_dim=256,
    num_layers=2,
):
    """训练模型"""
    
    np.random.seed(42)
    print(f"🚀 开始 NeurX 简化训练")
    print(f"   batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")
    
    # 准备数据
    print("📚 准备语料库...")
    texts = get_sample_corpus()
    tokenizer = CharTokenizer.from_texts(texts)
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    
    all_tokens = tokenize_corpus(texts, tokenizer)
    print(f"   总 token 数: {len(all_tokens)}")
    
    # 创建模型
    print("🔧 创建模型...")
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
    )
    
    # 创建优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    batches = make_batches(all_tokens, batch_size, seq_len)
    steps_per_epoch = max(1, len(all_tokens) // (batch_size * seq_len))
    total_steps = epochs * steps_per_epoch
    
    print(f"⏱️  每个 epoch {steps_per_epoch} 步，共 {epochs} 个 epoch")
    print()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            # 获取批次
            x, y = next(batches)
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(x)  # (batch, seq_len, vocab_size)
            
            # 计算损失
            batch_size_actual = x.shape[0]
            logits_flat = logits.reshape(-1, tokenizer.vocab_size)  # (batch*seq_len, vocab_size)
            y_flat = y.reshape(-1)  # (batch*seq_len,)
            
            loss = neurx.losses.cross_entropy(logits_flat, y_flat)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失
            loss_val = float(loss) if hasattr(loss, '__float__') else (
                float(loss.item()) if hasattr(loss, 'item') else loss
            )
            epoch_losses.append(loss_val)
            losses.append(loss_val)
            
            # 日志
            if (step + 1) % max(1, steps_per_epoch // 3) == 0 or step == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) > 0 else loss_val
                print(f"epoch {epoch+1}/{epochs}, step {step+1}/{steps_per_epoch}: loss={loss_val:.4f} (avg={avg_loss:.4f})")
    
    # 保存模型
    print()
    print("💾 保存模型...")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    
    # 收集参数
    params = {}
    for i, p in enumerate(model.parameters()):
        if hasattr(p, 'data'):
            params[f"param_{i}"] = np.array(p.data) if not isinstance(p.data, np.ndarray) else p.data
        else:
            params[f"param_{i}"] = np.array(p)
    
    # 保存检查点
    checkpoint = {
        "backend": "neurx",
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "params": params,
        },
        "tokenizer": tokenizer.to_dict(),
        "metrics": {
            "start_loss": float(losses[0]) if losses else 0.0,
            "end_loss": float(losses[-1]) if losses else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
        },
    }
    
    with open(output, "wb") as f:
        pickle.dump(checkpoint, f)
    
    # 结果
    print(f"✅ 训练完成")
    print(f"   开始损失: {losses[0]:.4f}")
    print(f"   最终损失: {losses[-1]:.4f}")
    print(f"   平均损失: {np.mean(losses):.4f}")
    print(f"   检查点: {output}")


def main():
    parser = argparse.ArgumentParser(description="NeurX 简化训练脚本")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--output", type=str, default="checkpoints/model_neurx_simple.pkl", help="输出路径")
    parser.add_argument("--seq-len", type=int, default=64, help="序列长度")
    parser.add_argument("--hidden-dim", type=int, default=256, help="隐层维度")
    parser.add_argument("--num-layers", type=int, default=2, help="层数")
    # 保留参数兼容性
    parser.add_argument("--checkpoint", type=str, default="", help="(保留，未使用)")
    
    args = parser.parse_args()
    
    train_simple_neurx(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output=args.output,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
