#!/usr/bin/env python
"""改进的训练脚本 - 使用更好的配置"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
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


class ImprovedTransformer(nn.Module):
    """改进的 Transformer 模型 - 更大更强"""
    
    def __init__(self, vocab_size, hidden_dim=512, num_layers=6, seq_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Embedding(seq_len, hidden_dim)
        
        # 多层Transformer
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'norm1': nn.LayerNorm(hidden_dim),
                'fc1': nn.Linear(hidden_dim, hidden_dim * 4),
                'relu': nn.ReLU(),
                'fc2': nn.Linear(hidden_dim * 4, hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout),
            })
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # Embedding + 位置编码
        x_emb = self.embedding(x)
        seq_len = x.shape[1]
        pos = np.arange(seq_len, dtype=np.int64)
        pos_emb = self.pos_encoding(pos)
        x = x_emb + pos_emb
        
        # 多层处理
        for layer in self.layers:
            # 残差连接 + LayerNorm + FFN
            residual = x
            x = layer['norm1'](x)
            x = layer['fc1'](x)
            x = layer['relu'](x)
            x = layer['fc2'](x)
            x = layer['dropout'](x)
            x = x + residual  # 残差连接
            x = layer['norm2'](x)
        
        logits = self.output(x)
        return logits


def load_corpus(filepath):
    """加载文本语料库"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return [line.strip() for line in text.split('\n') if line.strip()]


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
        # 数据太少，需要重复
        repeat_times = (seq_len + 10) // max(1, len(token_ids)) + 1
        token_ids = np.tile(token_ids, repeat_times)
        max_start = len(token_ids) - seq_len - 1
    
    while True:
        starts = np.random.randint(0, max(1, max_start), size=batch_size)
        x = np.stack([token_ids[s:s + seq_len] for s in starts], axis=0)
        y = np.stack([token_ids[s + 1:s + seq_len + 1] for s in starts], axis=0)
        
        yield neurx.Tensor(x), neurx.Tensor(y)


def train_improved(
    corpus_file="data/chat_corpus_expanded.txt",
    batch_size=32,
    epochs=30,
    learning_rate=3e-4,
    output="checkpoints/model_improved.pkl",
    seq_len=128,
    hidden_dim=512,
    num_layers=6,
    dropout=0.1,
    warmup_steps=1000,
    eval_interval=500,
    save_interval=2000,
):
    """使用改进配置训练模型"""
    
    np.random.seed(42)
    print("=" * 60)
    print("🚀 ChatNeurX 改进版训练")
    print("=" * 60)
    print(f"📊 配置:")
    print(f"   语料文件: {corpus_file}")
    print(f"   batch_size: {batch_size}")
    print(f"   epochs: {epochs}")
    print(f"   learning_rate: {learning_rate}")
    print(f"   hidden_dim: {hidden_dim}")
    print(f"   num_layers: {num_layers}")
    print(f"   seq_len: {seq_len}")
    print(f"   dropout: {dropout}")
    print()
    
    # 1. 加载数据
    print("📚 加载训练数据...")
    if not os.path.exists(corpus_file):
        print(f"❌ 找不到语料文件: {corpus_file}")
        print(f"💡 建议运行: python scripts/generate_more_data.py")
        return
    
    texts = load_corpus(corpus_file)
    print(f"   ✅ 加载了 {len(texts)} 条句子")
    
    # 2. 建立tokenizer
    print("🔤 建立 tokenizer...")
    tokenizer = CharTokenizer.from_texts(texts)
    vocab_size = tokenizer.vocab_size
    print(f"   ✅ 词汇表大小: {vocab_size}")
    
    all_tokens = tokenize_corpus(texts, tokenizer)
    print(f"   ✅ 总 token 数: {len(all_tokens):,}")
    
    # 数据分割（90% train, 10% val）
    split_idx = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    print(f"   ✅ 训练集: {len(train_tokens):,} tokens")
    print(f"   ✅ 验证集: {len(val_tokens):,} tokens")
    
    # 3. 创建模型
    print()
    print("🔧 创建改进模型...")
    model = ImprovedTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        dropout=dropout,
    )
    
    # 统计参数量
    total_params = sum(
        p.data.size if hasattr(p, 'data') else p.size 
        for p in model.parameters()
    )
    print(f"   ✅ 模型参数量: {total_params / 1e6:.2f}M")
    
    # 4. 训练准备
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_batches = make_batches(train_tokens, batch_size, seq_len)
    val_batches = make_batches(val_tokens, batch_size, seq_len)
    
    steps_per_epoch = max(100, len(train_tokens) // (batch_size * seq_len))
    total_steps = steps_per_epoch * epochs
    
    print()
    print("⏱️  训练配置:")
    print(f"   每个epoch: {steps_per_epoch} 步")
    print(f"   总步数: {total_steps}")
    print(f"   Warmup步数: {warmup_steps}")
    print(f"   评估间隔: {eval_interval} 步")
    print(f"   保存间隔: {save_interval} 步")
    print()
    print("=" * 60)
    
    # 5. 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\n📖 Epoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            global_step += 1
            
            # 学习率warmup
            if global_step <= warmup_steps:
                lr = learning_rate * (global_step / warmup_steps)
                optimizer.lr = lr
            
            # 训练步
            x, y = next(train_batches)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # 计算损失
            logits_flat = logits.reshape(-1, vocab_size)
            y_flat = y.reshape(-1)
            loss = neurx.losses.cross_entropy(logits_flat, y_flat)
            
            loss.backward()
            
            # 梯度裁剪
            # TODO: 实现梯度裁剪
            
            optimizer.step()
            
            # 记录
            loss_val = float(loss) if hasattr(loss, '__float__') else (
                float(loss.item()) if hasattr(loss, 'item') else loss
            )
            epoch_losses.append(loss_val)
            train_losses.append(loss_val)
            
            # 定期打印
            if (step + 1) % 50 == 0 or step == 0:
                avg_loss = np.mean(epoch_losses[-50:])
                print(f"   步骤 {step + 1:4d}/{steps_per_epoch} | "
                      f"loss: {loss_val:.4f} | "
                      f"avg: {avg_loss:.4f} | "
                      f"lr: {optimizer.lr:.6f}")
            
            # 验证
            if global_step % eval_interval == 0:
                print(f"\n   🔍 验证 (步骤 {global_step})...")
                val_loss_samples = []
                for _ in range(20):  # 20个验证批次
                    x_val, y_val = next(val_batches)
                    logits_val = model(x_val)
                    logits_val_flat = logits_val.reshape(-1, vocab_size)
                    y_val_flat = y_val.reshape(-1)
                    loss_val = neurx.losses.cross_entropy(logits_val_flat, y_val_flat)
                    loss_val_num = float(loss_val) if hasattr(loss_val, '__float__') else loss_val
                    val_loss_samples.append(loss_val_num)
                
                avg_val_loss = np.mean(val_loss_samples)
                val_losses.append(avg_val_loss)
                print(f"   📊 验证损失: {avg_val_loss:.4f}")
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"   ⭐ 新的最佳验证损失！保存模型...")
                    save_checkpoint(model, tokenizer, output.replace('.pkl', '_best.pkl'),
                                  train_losses, val_losses, hidden_dim, num_layers, seq_len)
            
            # 定期保存
            if global_step % save_interval == 0:
                checkpoint_path = output.replace('.pkl', f'_step{global_step}.pkl')
                save_checkpoint(model, tokenizer, checkpoint_path,
                              train_losses, val_losses, hidden_dim, num_layers, seq_len)
                print(f"   💾 保存检查点: {checkpoint_path}")
        
        # Epoch总结
        epoch_avg_loss = np.mean(epoch_losses)
        print(f"\n   📊 Epoch {epoch + 1} 平均损失: {epoch_avg_loss:.4f}")
    
    # 6. 保存最终模型
    print()
    print("=" * 60)
    print("💾 保存最终模型...")
    save_checkpoint(model, tokenizer, output, train_losses, val_losses,
                   hidden_dim, num_layers, seq_len)
    
    print()
    print("✅ 训练完成！")
    print(f"   开始损失: {train_losses[0]:.4f}")
    print(f"   最终损失: {train_losses[-1]:.4f}")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    print(f"   最终checkpoint: {output}")
    print(f"   最佳checkpoint: {output.replace('.pkl', '_best.pkl')}")
    print("=" * 60)


def save_checkpoint(model, tokenizer, path, train_losses, val_losses,
                   hidden_dim, num_layers, seq_len):
    """保存检查点"""
    params = {}
    for i, p in enumerate(model.parameters()):
        if hasattr(p, 'data'):
            params[f"param_{i}"] = np.array(p.data) if not isinstance(p.data, np.ndarray) else p.data
        else:
            params[f"param_{i}"] = np.array(p)
    
    checkpoint = {
        "backend": "neurx",
        "model_type": "improved_transformer",
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "params": params,
        },
        "tokenizer": tokenizer.to_dict(),
        "metrics": {
            "train_losses": train_losses[-100:],  # 只保存最后100个
            "val_losses": val_losses[-20:],  # 只保存最后20个
            "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
            "final_val_loss": float(val_losses[-1]) if val_losses else 0.0,
        },
    }
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    parser = argparse.ArgumentParser(description="改进的训练脚本")
    parser.add_argument("--corpus", type=str, default="data/chat_corpus_expanded.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="checkpoints/model_improved.pkl")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=2000)
    
    args = parser.parse_args()
    
    train_improved(
        corpus_file=args.corpus,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output=args.output,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
