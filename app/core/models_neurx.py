"""
基于 NeurX 框架实现的 ChatNeurX 大模型

特点：
- 完整的 Transformer 架构
- 支持多层堆叠
- 支持位置编码和自注意力机制
- 与标准 PyTorch API 兼容
"""

import numpy as np
import neurx
import neurx.nn as nn


class NeurXTransformerBlock(nn.Module):
    """基于 NeurX 的 Transformer 块
    
    包含：
    - 多头自注意力
    - Feed-Forward Network
    - 残差连接
    - 层归一化
    """
    
    def __init__(self, hidden_dim, num_heads, ffn_dim=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        ffn_dim = ffn_dim or (hidden_dim * 4)
        
        # 多头自注意力
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _multihead_attention(self, x, mask=None):
        """多头自注意力实现
        
        Args:
            x: (B, T, C)
            mask: (B, T, T) 注意力掩码（可选）
            
        Returns:
            attention_output: (B, T, C)
        """
        B, T, C = x.shape
        
        # 计算 Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 分多头：(B, T, C) -> (B, T, num_heads, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        
        # 转置为 (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = neurx.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores + mask
        
        # 计算注意力权重
        weights = neurx.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # 应用注意力到值
        attn_output = neurx.matmul(weights, v)  # (B, num_heads, T, head_dim)
        
        # 合并多头：(B, num_heads, T, head_dim) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2)  # (B, T, num_heads, head_dim)
        attn_output = attn_output.reshape(B, T, C)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def forward(self, x, mask=None):
        """Transformer 块前向传播
        
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: attention mask (可选)
            
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # 自注意力 + 残差连接
        x_norm = self.norm1(x)
        attn_out = self._multihead_attention(x_norm, mask)
        x = x + self.dropout(attn_out)
        
        # Feed-Forward + 残差连接
        x_norm = self.norm2(x)
        ffn_out = self.fc2(self.relu(self.fc1(x_norm)))
        x = x + self.dropout(ffn_out)
        
        return x


class NeurXChatModel(nn.Module):
    """基于 NeurX 框架的 ChatNeurX 大模型
    
    架构：
    - Token Embedding
    - Position Embedding
    - Transformer 块堆叠
    - 层归一化
    - 输出投影
    """
    
    def __init__(
        self, 
        vocab_size,
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        ffn_dim=3072,
        max_seq_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        
        # Position embedding
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
        """前向传播
        
        Args:
            input_ids: (batch_size, seq_len) 整数张量
            targets: (batch_size, seq_len) 目标 token IDs（训练时使用）
            
        Returns:
            dict: {
                'logits': (batch_size, seq_len, vocab_size),
                'loss': 标量张量（如果提供了 targets，否则为 None），
                'hidden_states': (batch_size, seq_len, hidden_dim)
            }
        """
        B, T = input_ids.shape
        
        # Token embedding
        x = self.tok_emb(input_ids)  # (B, T, C)
        
        # Position embedding
        pos_ids = neurx.arange(T, dtype='int64')
        pos_ids = pos_ids.unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos_ids)  # (1, T, C)
        
        # 结合 token 和 position embedding
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
            # Reshape for cross_entropy: (B*T, V) and (B*T,)
            logits_reshaped = logits.reshape(-1, self.vocab_size)
            targets_reshaped = targets.reshape(-1)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x,
        }


class NeurXTinyLM(nn.Module):
    """最小的 Tiny 语言模型（用于快速原型和测试）"""
    
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 简单的两层模型
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, targets=None):
        """前向传播
        
        Args:
            input_ids: (batch_size, seq_len)
            targets: (batch_size, seq_len) 可选的目标
            
        Returns:
            dict: {'logits': logits, 'loss': loss}
        """
        # Embedding
        x = self.tok_emb(input_ids)  # (B, T, C)
        
        # 隐藏层
        x = self.relu(self.linear(x))  # (B, T, C)
        
        # 输出
        logits = self.lm_head(x)  # (B, T, V)
        
        # 计算损失
        loss = None
        if targets is not None:
            logits_reshaped = logits.reshape(-1, self.vocab_size)
            targets_reshaped = targets.reshape(-1)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        
        return {
            'logits': logits,
            'loss': loss,
        }


# 便捷工厂函数
def create_chatmodel_tiny(vocab_size, **kwargs):
    """创建 Tiny 模型（快速原型，仅需几秒训练）
    
    Args:
        vocab_size: 词汇表大小
        **kwargs: 其他参数（hidden_dim 等）
        
    Returns:
        NeurXTinyLM 模型实例
    """
    hidden_dim = kwargs.pop('hidden_dim', 128)
    return NeurXTinyLM(vocab_size, hidden_dim=hidden_dim)


def create_chatmodel_small(vocab_size, **kwargs):
    """创建小模型
    
    规格：
    - hidden_dim: 256
    - num_layers: 4
    - num_heads: 4
    - max_seq_len: 512
    """
    defaults = {
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4,
        'max_seq_len': 512,
    }
    defaults.update(kwargs)
    return NeurXChatModel(vocab_size, **defaults)


def create_chatmodel_base(vocab_size, **kwargs):
    """创建基础模型
    
    规格：
    - hidden_dim: 768
    - num_layers: 6
    - num_heads: 8
    - max_seq_len: 1024
    """
    defaults = {
        'hidden_dim': 768,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 1024,
    }
    defaults.update(kwargs)
    return NeurXChatModel(vocab_size, **defaults)


def create_chatmodel_large(vocab_size, **kwargs):
    """创建大模型
    
    规格：
    - hidden_dim: 1024
    - num_layers: 12
    - num_heads: 16
    - max_seq_len: 2048
    """
    defaults = {
        'hidden_dim': 1024,
        'num_layers': 12,
        'num_heads': 16,
        'max_seq_len': 2048,
    }
    defaults.update(kwargs)
    return NeurXChatModel(vocab_size, **defaults)


if __name__ == "__main__":
    # 简单测试
    print("Testing NeurX ChatModel implementations...")
    
    vocab_size = 1000
    
    # 测试 Tiny 模型
    print("\n1. Testing Tiny model...")
    model_tiny = create_chatmodel_tiny(vocab_size)
    params_tiny = sum(p.numel() for p in model_tiny.parameters())
    print(f"   Tiny model parameters: {params_tiny:,}")
    
    # 测试 Small 模型
    print("\n2. Testing Small model...")
    model_small = create_chatmodel_small(vocab_size)
    params_small = sum(p.numel() for p in model_small.parameters())
    print(f"   Small model parameters: {params_small:,}")
    
    # 测试 Base 模型
    print("\n3. Testing Base model...")
    model_base = create_chatmodel_base(vocab_size)
    params_base = sum(p.numel() for p in model_base.parameters())
    print(f"   Base model parameters: {params_base:,}")
    
    # 测试前向传播
    print("\n4. Testing forward pass...")
    model_tiny.eval()
    
    import neurx as nx
    input_ids = nx.randint(0, vocab_size, (2, 32))  # (batch_size=2, seq_len=32)
    
    with nx.no_grad():
        output = model_tiny(input_ids)
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Logits shape: {output['logits'].shape}")
        print(f"   Loss: {output['loss']}")
    
    print("\n✅ All tests passed!")
