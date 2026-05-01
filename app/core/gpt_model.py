"""GPT模型的core实现（纯numpy后端）"""
import math
import os
import numpy as np
try:
    from neurx import Tensor
    from neurx.nn import (
        Module,
        Parameter,
        Embedding,
        Linear,
        LayerNorm as _NeurXLayerNorm,
        RMSNorm,
        Dropout,
        GELU,
        SiLU,
        ModuleList,
        ModuleDict,
        MoE,
    )
    from neurx.losses import cross_entropy_loss

    class LayerNorm(_NeurXLayerNorm):
        def __init__(self, normalized_shape, bias=True, eps=1e-5):
            super().__init__(normalized_shape, eps=eps, elementwise_affine=bool(bias))
except ImportError:
    from tensor.core.tensor import Tensor
    from tensor.core.nn import (
        Module,
        Parameter,
        Embedding,
        Linear,
        LayerNorm,
        RMSNorm,
        Dropout,
        GELU,
        SiLU,
        ModuleList,
        ModuleDict,
        MoE,
    )
    from tensor.core.losses import cross_entropy_loss


_S_RUNTIME = None
_S_RUNTIME_IMPORT_ERROR = None
_S_RUNTIME_MODULES = ("gpt_model_ops", "ops")


def _s_runtime_enabled():
    mode = os.environ.get("NEURX_S_OPS_BACKEND", "auto").strip().lower()
    return mode in {"1", "true", "on", "yes", "auto", "s"}


def _load_s_runtime_fn():
    global _S_RUNTIME, _S_RUNTIME_IMPORT_ERROR
    if _S_RUNTIME is not None:
        return _S_RUNTIME
    if _S_RUNTIME_IMPORT_ERROR is not None:
        return None
    try:
        import neurx.compile.runtime as runtime

        _S_RUNTIME = runtime
        return _S_RUNTIME
    except Exception as exc:  # pragma: no cover - optional runtime integration
        _S_RUNTIME_IMPORT_ERROR = exc
        return None


def _try_s_intrinsic(name, *args):
    if not _s_runtime_enabled():
        return None
    runtime = _load_s_runtime_fn()
    if runtime is None:
        return None
    for module_name in _S_RUNTIME_MODULES:
        try:
            if not runtime.supports_runtime_function(module_name, name):
                continue
            return runtime.invoke_runtime_function(module_name, name, *args)
        except Exception:
            continue
    return None


def _s_matmul(a, b):
    out = _try_s_intrinsic("matmul", a, b)
    if out is not None:
        return out
    return np.matmul(a, b)


def _s_softmax(x, dim=-1):
    out = _try_s_intrinsic("softmax", x, int(dim))
    if out is not None:
        return out
    x_max = np.max(x, axis=dim, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / (np.sum(x_exp, axis=dim, keepdims=True) + 1e-12)


def _s_silu(x):
    out = _try_s_intrinsic("silu", x)
    if out is not None:
        return out
    return None


def _s_gelu(x, approximate=False):
    out = _try_s_intrinsic("gelu", x, bool(approximate))
    if out is not None:
        return out
    return None


def _s_activation_tensor(tensor, intrinsic_name, fallback_fn, *intrinsic_args):
    if intrinsic_name == "silu":
        out = _s_silu(tensor.data)
    elif intrinsic_name == "gelu":
        approximate = bool(intrinsic_args[0]) if intrinsic_args else False
        out = _s_gelu(tensor.data, approximate=approximate)
    else:
        out = None

    if out is None:
        return fallback_fn(tensor)

    return Tensor(
        out,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op=f"{intrinsic_name}_s",
    )


def _s_lm_head_tensor(hidden, lm_head):
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return None
    weight_data = weight.data if hasattr(weight, "data") else weight
    bias = getattr(lm_head, "bias", None)
    if bias is None:
        bias_data = np.zeros((weight_data.shape[-1],), dtype=np.asarray(weight_data).dtype)
    else:
        bias_data = bias.data if hasattr(bias, "data") else bias
    out = _try_s_intrinsic("lm_head_logits", hidden.data, weight_data, bias_data)
    if out is None:
        return None
    result = Tensor(out, requires_grad=hidden.requires_grad, _children=(hidden,), _op="lm_head_logits_s")
    result._runtime_backend = "s"
    return result


def _rope_cache(seq_len, dim, theta=10000.0, start_pos=0):
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {dim}")
    positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float64)
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = positions[:, None] * freqs[None, :]
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


def _apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x_rot = np.empty_like(x)
    x_rot[..., ::2] = x1 * cos - x2 * sin
    x_rot[..., 1::2] = x1 * sin + x2 * cos
    return x_rot


class CausalSelfAttention(Module):
    """多头因果自注意力机制（简化版）"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V投影（合并为一个矩阵）
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = getattr(config, "rope_enabled", False)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        
        # 因果mask（下三角矩阵）- 作为numpy数组存储
        self.causal_mask = np.tril(np.ones((config.block_size, config.block_size)))

    def __call__(self, x):
        out, _ = self.forward_with_cache(x, kv_cache=None)
        return out

    def forward_with_cache(self, x, kv_cache=None):
        B, T, C = x.data.shape  # batch, sequence length, embedding dim
        past_k = None
        past_v = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None and past_v is not None:
                if past_k.shape[0] != B or past_v.shape[0] != B:
                    raise ValueError("kv_cache batch size mismatch")
        
        # 计算Q, K, V
        qkv = self.c_attn(x)
        # 将qkv分割为q, k, v
        qkv_data = qkv.data.reshape(B, T, 3, C)
        q_data = qkv_data[:, :, 0, :]  # (B, T, C)
        k_data = qkv_data[:, :, 1, :]
        v_data = qkv_data[:, :, 2, :]
        
        # 重塑为多头形式
        head_dim = C // self.n_head
        q_data = q_data.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k_data = k_data.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v_data = v_data.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        if self.use_rope:
            start_pos = 0 if past_k is None else past_k.shape[2]
            cos, sin = _rope_cache(T, head_dim, theta=self.rope_theta, start_pos=start_pos)
            q_data = _apply_rope(q_data, cos, sin)
            k_data = _apply_rope(k_data, cos, sin)

        if past_k is not None and past_v is not None:
            k_data = np.concatenate([past_k, k_data], axis=2)
            v_data = np.concatenate([past_v, v_data], axis=2)
        
        # 注意力计算（scaled dot-product attention）
        scale = 1.0 / math.sqrt(head_dim)
        att = _s_matmul(q_data, k_data.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)
        
        # 应用因果mask
        total_T = k_data.shape[2]
        mask = self.causal_mask[:total_T, :total_T]
        mask = mask[-T:, :]
        att = np.where(mask[None, None, :, :] == 0, -1e10, att)
        
        # Softmax
        att_probs = _s_softmax(att, dim=-1)
        
        # Dropout (简化：直接在numpy上操作)
        if self.training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, size=att_probs.shape) / (1 - self.dropout)
            att_probs = att_probs * dropout_mask
        
        # 应用注意力权重
        y_data = _s_matmul(att_probs, v_data)  # (B, nh, T, hs)
        
        # 重新组合所有头的输出
        y_data = y_data.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 创建Tensor并设置梯度传播（简化版）
        y = Tensor(y_data, requires_grad=x.requires_grad, _children=(x,), _op="attention")
        
        # 输出投影
        out = self.resid_dropout(self.c_proj(y))
        return out, (k_data, v_data)


class MLP(Module):
    """前馈神经网络"""
    def __init__(self, config):
        super().__init__()
        self.use_swiglu = getattr(config, "swiglu_enabled", False)
        hidden_dim = 4 * config.n_embd
        if self.use_swiglu:
            self.w1 = Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w2 = Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.act = SiLU()
            self.w3 = Linear(hidden_dim, config.n_embd, bias=config.bias)
        else:
            self.c_fc = Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.gelu = GELU()
            self.c_proj = Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def __call__(self, x):
        if self.use_swiglu:
            gate = self.w1(x)
            value = self.w2(x)
            gate_act = _s_activation_tensor(gate, "silu", self.act)
            x = gate_act * value
            x = self.w3(x)
        else:
            x = self.c_fc(x)
            x = _s_activation_tensor(x, "gelu", self.gelu, False)
            x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(Module):
    """Transformer块"""
    def __init__(self, config):
        super().__init__()
        norm_cls = RMSNorm if getattr(config, "rmsnorm_enabled", False) else LayerNorm
        self.ln_1 = norm_cls(config.n_embd, bias=getattr(config, "rmsnorm_bias", config.bias))
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd, bias=getattr(config, "rmsnorm_bias", config.bias))
        if getattr(config, "moe_enabled", False):
            self.mlp = MoE(
                config.n_embd,
                num_experts=getattr(config, "moe_num_experts", 4),
                top_k=getattr(config, "moe_top_k", 2),
                hidden_dim=getattr(config, "moe_hidden_dim", None),
                dropout=config.dropout,
                bias=config.bias,
                use_swiglu=getattr(config, "swiglu_enabled", False),
            )
        else:
            self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接
        x = x + self.mlp(self.ln_2(x))   # 残差连接
        return x

    def forward_with_cache(self, x, kv_cache=None):
        x_norm = self.ln_1(x)
        attn_out, new_cache = self.attn.forward_with_cache(x_norm, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(Module):
    """GPT语言模型（核心版本，不含多模态）"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token和位置嵌入
        self.wte = Embedding(config.vocab_size, config.n_embd)  # token embedding
        self.wpe = Embedding(config.block_size, config.n_embd)  # position embedding
        self.drop = Dropout(config.dropout)
        
        # Transformer块
        self.h = ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        norm_cls = RMSNorm if getattr(config, "rmsnorm_enabled", False) else LayerNorm
        self.ln_f = norm_cls(config.n_embd, bias=getattr(config, "rmsnorm_bias", config.bias))
        
        # 输出层（语言模型头）
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享：embedding和输出层共享权重
        # 注意：这里简化处理，不做严格的权重共享
        
        # 初始化权重（简化版）
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 对所有Linear层的权重进行特殊初始化
        for module in self.parameters():
            if isinstance(module, Parameter):
                # 简单的正态分布初始化
                pass  # 已在各层构造时初始化

    def __call__(self, idx, targets=None):
        """
        前向传播
        idx: (B, T) token索引
        targets: (B, T) 目标token (可选，用于训练)
        """
        if isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            idx_array = idx.data if isinstance(idx, Tensor) else np.array(idx)
        
        B, T = idx_array.shape
        assert T <= self.config.block_size, f"序列长度{T}超过最大长度{self.config.block_size}"
        
        # Token嵌入
        tok_emb = self.wte(idx_array)  # (B, T, n_embd)
        
        # 位置编码
        pos_emb = None
        if not getattr(self.config, "rope_enabled", False):
            pos = np.arange(0, T, dtype=np.int64)
            pos_emb = self.wpe(pos)  # (T, n_embd)
        
        # 相加并应用dropout
        x = self.drop(tok_emb if pos_emb is None else (tok_emb + pos_emb))
        
        # 通过所有Transformer块
        for block in self.h:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        if targets is not None:
            # 训练模式：计算所有位置的logits和损失
            logits = self.lm_head(x)
            # 计算交叉熵损失
            loss = cross_entropy_loss(logits, Tensor(targets))
        else:
            # 推理模式：只计算最后一个token的logits
            # 取最后一个时间步
            x_last = Tensor(x.data[:, -1:, :], requires_grad=x.requires_grad, _children=(x,), _op="slice")
            logits = _s_lm_head_tensor(x_last, self.lm_head)
            if logits is None:
                logits = self.lm_head(x_last)
            loss = None
        
        return logits, loss

    def get_num_params(self):
        """返回模型参数总数"""
        total = 0
        for p in self.parameters():
            total += p.data.size
        return total

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=None,
        use_kv_cache=True,
    ):
        return self.generate_advanced(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            use_kv_cache=use_kv_cache,
        )

    def generate_advanced(
        self,
        idx,
        max_new_tokens=120,
        temperature=0.8,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=None,
        use_kv_cache=True,
    ):
        from app.core.sampling import SamplingConfig, sample_next_token

        self.eval()
        rng = np.random.default_rng(seed)
        sampling_cfg = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        sampling_cfg.validate()

        if isinstance(idx, list):
            idx = np.array(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]  # 添加batch维度

        kv_cache = None
        for _ in range(max_new_tokens):
            if use_kv_cache and kv_cache is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            if use_kv_cache:
                logits, kv_cache = self.forward_with_cache(idx_cond, kv_cache=kv_cache)
            else:
                logits, _ = self(idx_cond, targets=None)

            next_token = sample_next_token(
                logits.data[0, -1],
                token_ids=idx[0].tolist(),
                cfg=sampling_cfg,
                rng=rng,
            )
            idx_next = np.array([[next_token]], dtype=np.int64)
            idx = np.concatenate([idx, idx_next], axis=1)

        return idx

    def forward_with_cache(self, idx, kv_cache=None):
        if isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            idx_array = idx.data if isinstance(idx, Tensor) else np.array(idx)

        B, T = idx_array.shape
        assert T <= self.config.block_size, f"序列长度{T}超过最大长度{self.config.block_size}"

        tok_emb = self.wte(idx_array)
        if getattr(self.config, "rope_enabled", False):
            x = self.drop(tok_emb)
        else:
            start_pos = 0
            if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
                cached_k, _ = kv_cache[0]
                if cached_k is not None:
                    start_pos = cached_k.shape[2]
            pos = np.arange(start_pos, start_pos + T, dtype=np.int64)
            pos_emb = self.wpe(pos)
            x = self.drop(tok_emb + pos_emb)

        new_cache = []
        for i, block in enumerate(self.h):
            block_cache = None
            if kv_cache is not None and i < len(kv_cache):
                block_cache = kv_cache[i]
            x, updated = block.forward_with_cache(x, kv_cache=block_cache)
            new_cache.append(updated)

        x = self.ln_f(x)
        logits = _s_lm_head_tensor(x, self.lm_head)
        if logits is None:
            logits = self.lm_head(x)
        return logits, new_cache


if __name__ == "__main__":
    # 测试模型
    from app.modeling.config import ModelConfig
    
    config = ModelConfig(
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        dropout=0.1,
        bias=True
    )
    
    model = GPT(config)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 测试前向传播
    x = np.random.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(x, x)
    print(f"输出形状: {logits.data.shape}")
    print(f"损失: {loss.data if loss else None}")
    
    # 测试生成
    start_ids = np.array([[1, 2, 3]], dtype=np.int64)
    generated = model.generate(start_ids, max_new_tokens=10, temperature=0.8)
    print(f"生成序列: {generated}")
