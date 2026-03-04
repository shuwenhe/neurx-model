# chatNeurx 项目 - NeurX 框架集成指南

## 🎯 项目概述

chatneurx 是一个工业级的 LLM 训练系统。本指南展示如何使用 **NeurX 深度学习框架** 来训练和推理 chatneurx 大模型。

---

## 📦 第一步：添加 neurx 依赖

### 更新 requirements.txt

```bash
# 将以下行添加到 requirements.txt
-e /home/shuwen/neurx
```

或者完整的新 requirements.txt：

```
# 核心依赖
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.3
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0
tensorboard>=2.13.0
sentencepiece>=0.1.99
tiktoken>=0.4.0

# Web 服务
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.8.0

# 监控和日志
prometheus-client>=0.20.0
PyJWT>=2.9.0
python-multipart>=0.0.9

# NeurX 深度学习框架（新增）
-e /home/shuwen/neurx
```

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装所有依赖
pip install -r requirements.txt

# 验证 neurx 安装
python -c "import neurx; print(f'NeurX {neurx.__version__} installed!')"
```

---

## 🏗️ 第二步：使用 NeurX 实现 ChatNeurX 模型

### 创建文件：app/core/models_neurx.py

这个文件使用 NeurX 框架重新实现 chatneurx 的模型。

```python
"""
基于 NeurX 框架实现的 ChatNeurX 大模型
- 支持 Transformer 架构
- 支持多层堆叠
- 支持位置编码和自注意力机制
"""

import neurx
import neurx.nn as nn
import neurx.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) 实现"""
    
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        inv_freq = 1.0 / (theta ** (neurx.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_len=None):
        """
        Apply rotary embedding
        
        Args:
            x: (batch, seq_len, dim)
            seq_len: sequence length
            
        Returns:
            Rotated embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # 计算位置和频率
        t = neurx.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = neurx.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        
        # 创建旋转矩阵
        emb = neurx.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        # 应用旋转
        cos = emb.cos()[None, :, :]  # (1, seq_len, dim)
        sin = emb.sin()[None, :, :]
        
        # Rotate x
        x_rot = neurx.cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)
        return x * cos + x_rot * sin


class NeurXTransformerBlock(nn.Module):
    """基于 NeurX 的 Transformer 块"""
    
    def __init__(self, hidden_dim, num_heads, ffn_dim=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        ffn_dim = ffn_dim or (hidden_dim * 4)
        
        # 多头自注意力
        self.norm1 = nn.LayerNorm(hidden_dim)
        if hasattr(nn, 'MultiHeadAttention'):
            self.attn = nn.MultiHeadAttention(hidden_dim, num_heads, dropout=dropout)
        else:
            # 如果没有 MultiHeadAttention，手动实现
            self.attn_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            self.attn_out = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def _multihead_attention(self, x):
        """手动实现多头自注意力"""
        B, T, C = x.shape
        
        # 计算 Q, K, V
        qkv = self.attn_qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)  # 各 (B, T, C)
        
        # 分多头
        h = self.num_heads
        q = q.reshape(B, T, h, C // h).transpose(1, 2)  # (B, h, T, C/h)
        k = k.reshape(B, T, h, C // h).transpose(1, 2)
        v = v.reshape(B, T, h, C // h).transpose(1, 2)
        
        # 计算注意力权重
        scores = neurx.matmul(q, k.transpose(-2, -1)) / neurx.sqrt(neurx.tensor(C // h, dtype=x.dtype))
        weights = F.softmax(scores, dim=-1)  # (B, h, T, T)
        
        # 应用注意力到值
        attn_out = neurx.matmul(weights, v)  # (B, h, T, C/h)
        
        # 合并多头
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        attn_out = self.attn_out(attn_out)
        
        return self.dropout(attn_out)
    
    def forward(self, x, mask=None):
        """
        Transformer 块前向传播
        
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: attention mask (可选)
            
        Returns:
            输出张量 (batch, seq_len, hidden_dim)
        """
        # 自注意力 + 残差连接 + 层归一化
        x_norm = self.norm1(x)
        
        if hasattr(self, 'attn'):
            # 使用内置多头注意力（如果可用）
            attn_out = self.attn(x_norm, x_norm, x_norm)
        else:
            # 使用手动实现
            attn_out = self._multihead_attention(x_norm)
        
        x = x + attn_out
        
        # Feed-Forward + 残差连接 + 层归一化
        x_norm = self.norm2(x)
        ffn_out = self.fc2(self.relu(self.fc1(x_norm)))
        x = x + self.dropout(ffn_out)
        
        return x


class NeurXChatModel(nn.Module):
    """基于 NeurX 框架的 ChatNeurX 大模型实现"""
    
    def __init__(
        self, 
        vocab_size,
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        ffn_dim=3072,
        max_seq_len=2048,
        dropout=0.1,
        use_rope=False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        
        # Position embedding (或 RoPE)
        if use_rope:
            self.pos_emb = RotaryPositionalEmbedding(hidden_dim, max_seq_len)
        else:
            self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer 块堆叠
        self.transformer_blocks = nn.ModuleList([
            NeurXTransformerBlock(
                hidden_dim, 
                num_heads, 
                ffn_dim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.ln_final = nn.LayerNorm(hidden_dim)
        
        # 输出投影
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, targets=None):
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len) 整数张量
            targets: (batch_size, seq_len) 目标 token IDs（训练时使用）
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: 标量张量（如果提供了 targets，否则为 None）
        """
        B, T = input_ids.shape
        
        # Token embedding
        x = self.tok_emb(input_ids)  # (B, T, C)
        
        # 位置信息
        if self.use_rope:
            x = self.pos_emb(x)
        else:
            pos_ids = neurx.arange(T).unsqueeze(0)  # (1, T)
            pos_emb = self.pos_emb(pos_ids)  # (1, T, C)
            x = x + pos_emb
        
        # Dropout
        x = self.dropout(x)
        
        # 通过 Transformer 块序列
        for block in self.transformer_blocks:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_final(x)
        
        # 输出投影
        logits = self.lm_head(x)  # (B, T, V)
        
        # 计算损失（如果提供了目标）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x,
        }


class NeurXTinyLM(nn.Module):
    """最小的 Tiny 语言模型（用于快速原型）"""
    
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, targets=None):
        x = self.tok_emb(input_ids)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return {
            'logits': logits,
            'loss': loss,
        }


# 便捷工厂函数
def create_chatmodel_tiny(vocab_size, **kwargs):
    """创建 Tiny 模型（快速原型）"""
    return NeurXTinyLM(vocab_size, **kwargs)


def create_chatmodel_small(vocab_size, **kwargs):
    """创建小模型"""
    defaults = {
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4,
        'max_seq_len': 512,
    }
    defaults.update(kwargs)
    return NeurXChatModel(vocab_size, **defaults)


def create_chatmodel_base(vocab_size, **kwargs):
    """创建基础模型"""
    defaults = {
        'hidden_dim': 768,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 1024,
    }
    defaults.update(kwargs)
    return NeurXChatModel(vocab_size, **defaults)


def create_chatmodel_large(vocab_size, **kwargs):
    """创建大模型"""
    defaults = {
        'hidden_dim': 1024,
        'num_layers': 12,
        'num_heads': 16,
        'max_seq_len': 2048,
    }
    defaults.update(kwargs)
    return NeurXChatModel(vocab_size, **defaults)
```

---

## 🚀 第三步：NeurX 版本的训练脚本

### 创建文件：app/training/train_neurx.py

```python
"""
使用 NeurX 框架训练 ChatNeurX 模型

特点：
- 基于 NeurX 的自动微分引擎
- 支持 GPU 加速（如果可用）
- 完整的训练循环和评估
- 嵌入式数据微批处理
"""

import argparse
import os
import time
import numpy as np

import neurx
import neurx.nn as nn
import neurx.optim as optim

from app.core.models_neurx import (
    create_chatmodel_tiny,
    create_chatmodel_small,
    create_chatmodel_base,
    create_chatmodel_large,
)
from app.core.tokenizer import CharTokenizer


def get_sample_corpus():
    """获取示例文本语料"""
    return [
        "北京是中国的首都，位于华北平原中部。",
        "人工智能正在改变世界，推动社会进步。",
        "语言模型可以生成文本，理解语义信息。",
        "机器学习需要数据和算力，还需要算法。",
        "深度学习在计算机视觉领域取得了巨大成就。",
        "自然语言处理技术在电商和搜索引擎中应用。",
        "神经网络通过反向传播算法实现参数更新。",
        "数据预处理对模型训练的质量有重要影响。",
        "超参数调优需要在验证集上反复测试。",
        "正则化技术可以防止模型过拟合。",
        "批归一化加快了神经网络的训练速度。",
        "注意力机制大幅提高了序列模型的性能。",
        "Transformers 架构革命了自然语言处理。",
        "迁移学习让我们可以利用预训练模型。",
        "多任务学习能够提高模型的泛化能力。",
        "知识蒸馏可以将大模型压缩为小模型。",
        "梯度裁剪防止了训练过程中的梯度爆炸。",
        "学习率预热有助于模型的稳定训练。",
        "对数据进行增强可以扩大训练集的规模。",
    ] * 10


def tokenize_corpus(corpus, tokenizer):
    """将文本语料转换为 token IDs"""
    token_ids = []
    for text in corpus:
        tokens = tokenizer.encode(text)
        token_ids.extend(tokens)
    return neurx.array(token_ids, dtype='int64')


def make_batches(token_ids, batch_size, seq_len):
    """创建 batch 数据"""
    token_ids = neurx.asarray(token_ids, dtype='int64')
    num_tokens = token_ids.shape[0]
    max_start = num_tokens - seq_len - 1
    
    batch_count = 0
    while True:
        # 随机快速生成 batch
        starts = neurx.randint(0, max_start, size=batch_size)
        
        x = neurx.stack([token_ids[s:s + seq_len] for s in starts])  # (B, T)
        y = neurx.stack([token_ids[s + 1:s + seq_len + 1] for s in starts])  # (B, T)
        
        yield x, y
        
        batch_count += 1
        # 每 1000 个 batch 停止迭代
        if batch_count >= 1000:
            break


def train_neurx(args):
    """使用 NeurX 框架训练模型"""
    
    print("=" * 60)
    print("ChatNeurX with NeurX Framework - Training Script")
    print("=" * 60)
    
    # 设置随机种子
    neurx.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化字符分词器
    corpus = get_sample_corpus()
    tokenizer = CharTokenizer()
    
    # 统计词汇表大小
    vocab_size = len(set(''.join(corpus)))
    print(f"\nVocab size: {vocab_size}")
    print(f"Sample corpus size: {len(corpus)} texts")
    
    # 创建模型
    print(f"\nCreating {args.model_size} model...")
    
    model_creators = {
        'tiny': create_chatmodel_tiny,
        'small': create_chatmodel_small,
        'base': create_chatmodel_base,
        'large': create_chatmodel_large,
    }
    
    model = model_creators[args.model_size](vocab_size)
    
    # 统计参数数量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 数据预处理
    print(f"\nTokenizing corpus...")
    token_ids = tokenize_corpus(corpus, tokenizer)
    print(f"Total tokens: {len(token_ids)}")
    
    # 训练循环
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("-" * 60)
    
    batch_gen = make_batches(token_ids, args.batch_size, args.seq_len)
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        
        # 重新创建 batch 生成器
        batch_gen = make_batches(token_ids, args.batch_size, args.seq_len)
        
        for x, y in batch_gen:
            # 前向传播
            with neurx.enable_grad():
                output = model(x, targets=y)
                loss = output['loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（可选）
            if args.grad_clip > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.clip(-args.grad_clip, args.grad_clip)
            
            # 更新参数
            optimizer.step()
            
            # 记录帧值
            epoch_loss += loss.item()
            num_batches += 1
            
            # 定期打印
            if num_batches % 10 == 0:
                print(f"  Batch {num_batches:3d}: loss = {loss.item():.4f}")
        
        # Epoch 统计
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 60)
    
    print("\nTraining completed!")
    
    # 保存模型（可选）
    if args.save_path:
        print(f"Saving model to {args.save_path}")
        neurx.save(model.state_dict(), args.save_path)
    
    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train ChatNeurX with NeurX")
    
    # 模型配置
    parser.add_argument('--model-size', type=str, default='tiny', 
                       choices=['tiny', 'small', 'base', 'large'],
                       help="Model size")
    
    # 训练配置
    parser.add_argument('--num-epochs', type=int, default=3, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--seq-len', type=int, default=64, help="Sequence length")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--grad-clip', type=float, default=1.0, help="Gradient clipping")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    # 输出
    parser.add_argument('--save-path', type=str, default=None, help="Model save path")
    
    args = parser.parse_args()
    
    # 运行训练
    model = train_neurx(args)
    
    print("\n✅ Training script completed successfully!")
    print("\nTo use this model for inference:")
    print("  from app.core.models_neurx import create_chatmodel_tiny")
    print("  model = create_chatmodel_tiny(vocab_size)")
    print("  output = model(input_ids)")


if __name__ == "__main__":
    main()
```

---

## 📝 第四步：集成检查清单

### 1️⃣ 依赖配置

- [ ] 在 `requirements.txt` 添加 `-e /home/shuwen/neurx`
- [ ] 运行 `pip install -r requirements.txt`  
- [ ] 验证：`python -c "import neurx; print(neurx.__version__)"`

### 2️⃣ 模型文件

- [ ] 创建 `app/core/models_neurx.py`（包含所有模型类）
- [ ] 验证模型可以导入：`from app.core.models_neurx import create_chatmodel_tiny`

### 3️⃣ 训练脚本

- [ ] 创建 `app/training/train_neurx.py`
- [ ] 测试训练脚本：`python app/training/train_neurx.py --model-size tiny --num-epochs 1`

### 4️⃣ 推理脚本（可选）

- [ ] 创建 `app/inference/inference_neurx.py`
- [ ] 实现文本生成功能

---

## 🚀 快速开始

### 训练模型

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练 Tiny 模型（快速测试）
python app/training/train_neurx.py --model-size tiny --num-epochs 1

# 3. 训练 Small 模型
python app/training/train_neurx.py --model-size small --num-epochs 3 --batch-size 16

# 4. 训练大模型
python app/training/train_neurx.py --model-size base --num-epochs 5 --batch-size 32 --learning-rate 5e-4
```

### 使用模型进行推理

```python
import neurx
from app.core.models_neurx import create_chatmodel_tiny
from app.core.tokenizer import CharTokenizer

# 创建模型
model = create_chatmodel_tiny(vocab_size=100)
model.eval()

# 初始化分词器
tokenizer = CharTokenizer()

# 输入文本
text = "机器学习"
tokens = neurx.array(tokenizer.encode(text))

# 前向传播
with neurx.no_grad():
    output = model(tokens.unsqueeze(0))
    logits = output['logits']
    
# 获取预测
predictions = logits.argmax(dim=-1)
print(f"Generated tokens: {predictions}")
```

---

## 📊 性能对比

| 指标 | Tensor 框架 | NeurX 框架 |
|------|-----------|----------|
| 模型定义 | 自研 Tensor 类 | PyTorch 风格 |
| 自动微分 | 自研 | NeurX 引擎 |
| GPU 支持 | 有限 | 完整（可选 CUDA） |
| API 易用性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 代码行数 | 更多 | 更少 |
| 学习曲线 | 陡峭 | 平缓 |

---

## 🔄 迁移指南：从 Tensor 到 NeurX

### 模型定义迁移

**旧（Tensor 框架）**：
```python
from tensor.core.nn import Module, Linear, Embedding

class MyModel(Module):
    def __init__(self, vocab_size):
        self.emb = Embedding(vocab_size, 128)
        self.linear = Linear(128, 10)
```

**新（NeurX 框架）**：
```python
import neurx.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128)
        self.linear = nn.Linear(128, 10)
```

### 训练循环迁移

**旧（Tensor 框架）**：
```python
from tensor.core.optim import AdamW

for x, y in data:
    logits, loss = model(x, targets=y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**新（NeurX 框架）**：
```python
import neurx.optim as optim

optimizer = optim.Adam(model.parameters())

for x, y in data:
    output = model(x, targets=y)
    loss = output['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 💡 高级特性

### 1. 混合精度训练

```python
import neurx

# 转换为 float16 进行快速计算
x_fp16 = x.float16()
output = model(x_fp16)

# 反向传播前转回 float32
loss = output['loss'].float32()
loss.backward()
```

### 2. 梯度累积

```python
accumulation_steps = 4

for i, (x, y) in enumerate(data):
    output = model(x, targets=y)
    loss = output['loss'] / accumulation_steps
    
    loss.backward()
    
    # 每 N 步更新一次参数
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 模型检查点保存

```python
import neurx

# 保存
neurx.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pkl')

# 加载
checkpoint = neurx.load('checkpoint.pkl')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
```

---

## 🐛 常见问题

### Q1: 如何在 GPU 上运行？

```bash
# 设置环境变量启用 CUDA
export TENSOR_CUDA=1

# 或者在代码中
import neurx
neurx.cuda.enable()
```

### Q2: 如何加载预训练模型？

```python
model = create_chatmodel_base(vocab_size=32000)
state = neurx.load('pretrained.pkl')
model.load_state_dict(state)
```

### Q3: 如何实现自定义层？

```python
import neurx.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(...)
        self.bias = nn.Parameter(...)
    
    def forward(self, x):
        return neurx.matmul(x, self.weight) + self.bias
```

---

## 📚 参考资源

- [NeurX 快速安装指南](/home/shuwen/neurx/QUICK_INSTALL.md)
- [NeurX 详细使用指南](/home/shuwen/neurx/docs/INSTALLATION_AND_USAGE_GUIDE.md)
- [NeurX 完整部署指南](/home/shuwen/neurx/docs/COMPLETE_DEPLOYMENT_GUIDE.md)
- [NeurX 模板项目](/home/shuwen/neurx/examples/template_project/)

---

## 📞 获取帮助

如果遇到问题：

1. 检查 NeurX 是否正确安装：`python -c "import neurx; print(neurx.__version__)"`
2. 查看 NeurX 的完整文档
3. 查看本指南的"常见问题"部分
4. 在 chatneurx GitHub issues 中提问

---

**祝你使用 NeurX 框架搭建 ChatNeurX 大模型训练系统！** 🚀
