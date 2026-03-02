from tensor.core.nn import Module, Embedding, Linear, LayerNorm, RMSNorm, TransformerBlock
from tensor.core.losses import cross_entropy


class TinyLM(Module):
    """最小可用语言模型：Embedding + Linear(vocab projection)"""

    def __init__(self, vocab_size, n_embd):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.lm_head = Linear(n_embd, vocab_size, bias=True)

    def __call__(self, input_ids, targets=None):
        x = self.tok_emb(input_ids)          # (B, T, C)
        logits = self.lm_head(x)             # (B, T, V)
        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)
        return logits, loss


class TransformerLM(Module):
    """基于 Transformer 的语言模型（支持自注意力、位置编码、多层堆叠）"""

    def __init__(
        self,
        vocab_size,
        n_embd,
        n_layers=2,
        n_heads=4,
        max_seq_len=2048,
        dropout=0.1,
        use_moe=False,
        moe_num_experts=4,
        moe_top_k=2,
        moe_hidden_dim=None,
        use_rmsnorm=False,
        use_swiglu=False,
        use_rope=False,
        rope_theta=10000.0,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Token embedding + Position embedding
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(max_seq_len, n_embd)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                n_embd,
                n_heads,
                dropout,
                use_moe=use_moe,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_hidden_dim=moe_hidden_dim,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
                rope_theta=rope_theta,
            )
            for _ in range(n_layers)
        ]
        
        # 最后的 LayerNorm 和输出投影
        self.ln_f = (RMSNorm if use_rmsnorm else LayerNorm)(n_embd)
        self.lm_head = Linear(n_embd, vocab_size, bias=False)

    def __call__(self, input_ids, targets=None):
        B, T = input_ids.shape if hasattr(input_ids, 'shape') else (len(input_ids), len(input_ids[0]))
        
        # 位置索引
        import numpy as np
        pos = np.arange(0, T, dtype=np.int64)[None, :]  # (1, T)
        
        # Token + Position embeddings
        tok_emb = self.tok_emb(input_ids)  # (B, T, C)
        pos_emb = self.pos_emb(pos)        # (1, T, C)
        x = tok_emb if self.use_rope else (tok_emb + pos_emb)              # (B, T, C)
        
        # 通过 Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 最终归一化和输出
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        
        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)
        
        return logits, loss

    def forward_with_cache(self, input_ids, kv_cache=None):
        B, T = input_ids.shape if hasattr(input_ids, 'shape') else (len(input_ids), len(input_ids[0]))
        tok_emb = self.tok_emb(input_ids)  # (B, T, C)
        if self.use_rope:
            x = tok_emb
        else:
            import numpy as np
            start_pos = 0
            if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
                cached_k, _ = kv_cache[0]
                if cached_k is not None:
                    start_pos = cached_k.shape[2]
            pos = np.arange(start_pos, start_pos + T, dtype=np.int64)[None, :]  # (1, T)
            pos_emb = self.pos_emb(pos)
            x = tok_emb + pos_emb

        new_cache = []
        for i, block in enumerate(self.blocks):
            block_cache = None
            if kv_cache is not None and i < len(kv_cache):
                block_cache = kv_cache[i]
            x, updated = block.forward_with_cache(x, kv_cache=block_cache)
            new_cache.append(updated)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_cache

    def generate(
        self,
        input_ids,
        max_new_tokens=120,
        temperature=0.8,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=None,
        use_kv_cache=True,
    ):
        from app.core.sampling import SamplingConfig, sample_next_token
        import numpy as np

        if hasattr(self, "eval"):
            self.eval()

        sampling_cfg = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        sampling_cfg.validate()
        rng = np.random.default_rng(seed)

        if isinstance(input_ids, list):
            ids = input_ids
        else:
            ids = input_ids.tolist() if isinstance(input_ids, np.ndarray) else list(input_ids)
        if not ids:
            ids = [0]

        kv_cache = None
        for _ in range(max_new_tokens):
            if use_kv_cache and kv_cache is not None:
                x = np.array([[ids[-1]]], dtype=np.int64)
            else:
                ctx = ids[-self.max_seq_len:] if self.max_seq_len else ids
                x = np.array([ctx], dtype=np.int64)
            if use_kv_cache:
                logits, kv_cache = self.forward_with_cache(x, kv_cache=kv_cache)
            else:
                logits, _ = self(x, None)
            next_id = sample_next_token(
                logits.data[0, -1],
                token_ids=ids,
                cfg=sampling_cfg,
                rng=rng,
            )
            ids.append(next_id)
        return ids
