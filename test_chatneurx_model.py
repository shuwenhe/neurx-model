#!/usr/bin/env python
"""ChatNeurX 模型测试脚本 - 使用自研 NeurX 框架"""

import os
import pickle
import sys

import numpy as np

# 设置路径
sys.path.insert(0, '/home/shuwen/chatneurx')

from app.core.tokenizer import CharTokenizer


def load_model_and_tokenizer(checkpoint_path="checkpoints/model_core.pkl"):
    """加载模型和tokenizer"""
    print(f"📦 加载 checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    # 加载tokenizer
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    
    # 加载模型配置
    model_cfg = ckpt["model"]
    params = model_cfg["params"]
    
    print(f"   模型参数:")
    print(f"   - hidden_dim: {model_cfg.get('hidden_dim', 256)}")
    print(f"   - num_layers: {model_cfg.get('num_layers', 2)}")
    print(f"   - vocab_size: {model_cfg.get('vocab_size')}")
    print(f"   - 损失: {ckpt['metrics']['end_loss']:.4f}")
    
    return params, model_cfg, tokenizer


def simple_forward(params, input_ids, vocab_size):
    """简单的前向推理（numpy实现）"""
    # Embedding
    tok_emb = params["param_0"]  # (V, H)
    x = tok_emb[input_ids]  # (T, H)
    
    # Layer Norm 1
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + 1e-5)
    
    # FC1
    residual = x
    x = x @ params["param_1"] + params["param_2"]
    x = np.maximum(x, 0.0)  # ReLU
    
    # FC2
    x = x @ params["param_3"] + params["param_4"]
    x = x + residual
    
    # Layer Norm 2
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + 1e-5)
    
    # Output projection
    logits = x @ params["param_5"] + params["param_6"]  # (T, V)
    
    return logits


def sample_token(logits, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.3, used_tokens=None):
    """采样下一个token"""
    logits = logits.copy()
    
    # 重复惩罚
    if used_tokens and repetition_penalty > 1.0:
        for tid in set(used_tokens[-20:]):  # 最近20个token
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty
    
    # Temperature
    logits = logits / max(temperature, 1e-8)
    
    # Top-k
    if top_k > 0 and top_k < len(logits):
        indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[indices] = logits[indices]
        logits = mask
    
    # Softmax
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / (exp_logits.sum() + 1e-12)
    
    # 采样
    token = np.random.choice(len(probs), p=probs)
    return int(token)


def generate_text(params, model_cfg, tokenizer, prompt, max_tokens=60, temperature=0.8, 
                  top_k=40, top_p=0.9, repetition_penalty=1.3):
    """生成文本"""
    # 编码prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]
    
    vocab_size = model_cfg["vocab_size"]
    generated_ids = input_ids[:]
    
    # 生成
    for _ in range(max_tokens):
        # 前向推理
        logits = simple_forward(params, np.array(generated_ids), vocab_size)
        next_logits = logits[-1]  # 取最后一个位置的logits
        
        # 屏蔽<unk>
        unk_id = tokenizer.stoi.get("<unk>")
        if unk_id is not None:
            next_logits[unk_id] = -np.inf
        
        # 采样
        next_id = sample_token(
            next_logits, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            used_tokens=generated_ids
        )
        
        generated_ids.append(next_id)
    
    # 解码
    full_text = tokenizer.decode(generated_ids)
    response = full_text[len(prompt):].replace("<unk>", "").strip()
    
    return response


def run_tests():
    """运行测试"""
    print("=" * 70)
    print("🚀 ChatNeurX 模型测试 (基于自研 NeurX 框架)")
    print("=" * 70)
    print()
    
    # 加载模型
    checkpoint_path = "/home/shuwen/chatneurx/checkpoints/model_core.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到 checkpoint: {checkpoint_path}")
        return
    
    params, model_cfg, tokenizer = load_model_and_tokenizer(checkpoint_path)
    print()
    
    # 测试用例
    test_cases = [
        ("自我介绍", {"temperature": 0.7, "top_k": 50, "repetition_penalty": 1.5}),
        ("你好", {"temperature": 0.8, "top_k": 40, "repetition_penalty": 1.4}),
        ("你能做什么", {"temperature": 0.75, "top_k": 45, "repetition_penalty": 1.3}),
        ("你是谁", {"temperature": 0.7, "top_k": 50, "repetition_penalty": 1.5}),
        ("介绍一下自己", {"temperature": 0.7, "top_k": 50, "repetition_penalty": 1.6}),
    ]
    
    print("📊 开始测试生成效果:")
    print("-" * 70)
    
    for i, (prompt, params_dict) in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】")
        print(f"提示词: {prompt}")
        print(f"参数: temp={params_dict['temperature']}, top_k={params_dict['top_k']}, rep_penalty={params_dict['repetition_penalty']}")
        
        response = generate_text(
            params, model_cfg, tokenizer, prompt,
            max_tokens=60,
            **params_dict
        )
        
        print(f"回答: {response[:200]}")  # 限制显示长度
        print()
    
    print("-" * 70)
    print("\n✅ 测试完成!")
    print()
    
    # 模型分析
    print("=" * 70)
    print("📈 模型架构分析:")
    print("=" * 70)
    print(f"""
当前架构: 简化的前馈神经网络
- Tokenizer: 字符级 (CharTokenizer)
- 层数: 2层前馈网络 + LayerNorm
- 参数量: ~{sum(p.size for p in params.values()) / 1e6:.2f}M
- 框架: 自研 NeurX

优势:
✓ 完全自研实现，掌握每个细节
✓ 轻量级，推理速度快
✓ 无外部依赖（不依赖 transformers 等库）

当前限制:
✗ 字符级 tokenizer 难度大（每步 {tokenizer.vocab_size} 个选项）
✗ 简单前馈网络缺少注意力机制
✗ 需要海量数据才能学好字符级模式

建议改进方向（基于 NeurX）:
1. 实现多头自注意力 (Multi-Head Attention)
2. 增加模型深度 (4-6 层)
3. 提高 hidden_dim (512-1024)
4. 或实现简单的词级/BPE tokenizer

要完全发挥 NeurX 框架的能力，建议使用完整的 Transformer 架构！
""")


if __name__ == "__main__":
    np.random.seed(42)
    run_tests()
