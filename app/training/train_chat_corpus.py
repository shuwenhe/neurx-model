#!/usr/bin/env python
"""基于对话语料的快速训练脚本 - 替代原始训练"""

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
    exit(1)

from app.core.tokenizer import CharTokenizer


class SimpleTransformer(nn.Module):
    """极简 Transformer 模型"""
    
    def __init__(self, vocab_size, hidden_dim=256, num_layers=2, seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + residual
        x = self.norm2(x)
        logits = self.output(x)
        return logits


def load_corpus(filepath):
    """加载文本语料库"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # 按行分割，每行一条
    return [line.strip() for line in text.split('\n') if line.strip()]


def tokenize_corpus(texts, tokenizer):
    """分词语料库 - 每行单独处理"""
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
        token_ids = np.tile(token_ids, (seq_len + 10) // max(1, len(token_ids)) + 1)
        max_start = len(token_ids) - seq_len - 1
    
    while True:
        starts = np.random.randint(0, max(1, max_start), size=batch_size)
        x = np.stack([token_ids[s:s + seq_len] for s in starts], axis=0)
        y = np.stack([token_ids[s + 1:s + seq_len + 1] for s in starts], axis=0)
        
        yield neurx.Tensor(x), neurx.Tensor(y)


def train_on_corpus(
    corpus_file="data/chat_corpus.txt",
    batch_size=4,
    epochs=3,
    learning_rate=1e-4,
    output="checkpoints/model_core.pkl",
    seq_len=64,
    hidden_dim=256,
):
    """基于语料库训练模型"""
    
    np.random.seed(42)
    print(f"🚀 开始基于对话语料的训练")
    print(f"   语料文件: {corpus_file}")
    print(f"   batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")
    print()
    
    # 加载语料
    print("📚 加载对话语料...")
    if not os.path.exists(corpus_file):
        print(f"❌ 找不到语料文件: {corpus_file}")
        return
    
    texts = load_corpus(corpus_file)
    print(f"   加载了 {len(texts)} 条句子")
    
    # 建立 tokenizer
    print("🔤 建立 tokenizer...")
    tokenizer = CharTokenizer.from_texts(texts)
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    
    all_tokens = tokenize_corpus(texts, tokenizer)
    print(f"   总 token 数: {len(all_tokens)}")
    
    # 创建模型
    print("🔧 创建模型...")
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        seq_len=seq_len,
    )
    
    # 训练
    optimizer = Adam(model.parameters(), lr=learning_rate)
    batches = make_batches(all_tokens, batch_size, seq_len)
    steps_per_epoch = max(1, len(all_tokens) // (batch_size * seq_len))
    
    print(f"⏱️  每个 epoch {steps_per_epoch} 步，共 {epochs} 个 epoch")
    print()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            x, y = next(batches)
            
            optimizer.zero_grad()
            logits = model(x)
            
            batch_size_actual = x.shape[0]
            logits_flat = logits.reshape(-1, tokenizer.vocab_size)
            y_flat = y.reshape(-1)
            
            loss = neurx.losses.cross_entropy(logits_flat, y_flat)
            
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss) if hasattr(loss, '__float__') else (
                float(loss.item()) if hasattr(loss, 'item') else loss
            )
            epoch_losses.append(loss_val)
            losses.append(loss_val)
            
            if (step + 1) % max(1, steps_per_epoch // 3) == 0 or step == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) > 0 else loss_val
                print(f"epoch {epoch+1}/{epochs}, step {step+1}/{steps_per_epoch}: loss={loss_val:.4f} (avg={avg_loss:.4f})")
    
    # 保存模型
    print()
    print("💾 保存模型...")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    
    params = {}
    for i, p in enumerate(model.parameters()):
        if hasattr(p, 'data'):
            params[f"param_{i}"] = np.array(p.data) if not isinstance(p.data, np.ndarray) else p.data
        else:
            params[f"param_{i}"] = np.array(p)
    
    checkpoint = {
        "backend": "neurx",
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": 2,
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
    
    print(f"✅ 训练完成")
    print(f"   开始损失: {losses[0]:.4f}")
    print(f"   最终损失: {losses[-1]:.4f}")
    print(f"   平均损失: {np.mean(losses):.4f}")
    print(f"   checkpoint: {output}")


def main():
    parser = argparse.ArgumentParser(description="基于对话语料的训练脚本")
    parser.add_argument("--corpus", type=str, default="data/chat_corpus.txt", help="对话语料文件")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--output", type=str, default="checkpoints/model_core.pkl", help="输出路径")
    parser.add_argument("--seq-len", type=int, default=64, help="序列长度")
    parser.add_argument("--hidden-dim", type=int, default=256, help="隐层维度")
    
    args = parser.parse_args()
    
    train_on_corpus(
        corpus_file=args.corpus,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output=args.output,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
