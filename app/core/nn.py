import numpy as np

from app.core.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        params = []
        seen = set()

        def add_param(param):
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)

        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                add_param(value)
            elif isinstance(value, Module):
                for p in value.parameters():
                    add_param(p)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        add_param(item)
                    elif isinstance(item, Module):
                        for p in item.parameters():
                            add_param(p)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        """切换到训练模式"""
        self.training = True
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train()

    def eval(self):
        """切换到评估模式"""
        self.training = False
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.eval()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.eval()


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def __call__(self, input_ids):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        out_data = self.weight.data[input_ids]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad, _children=(self.weight,), _op="embedding")

        def _backward():
            if self.weight.requires_grad:
                grad = np.zeros_like(self.weight.data)
                np.add.at(grad, input_ids, out.grad)
                self.weight.grad += grad

        out._backward = _backward
        return out


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        scale = (2.0 / max(1, in_features)) ** 0.5
        self.weight = Parameter(np.random.randn(in_features, out_features) * scale)
        self.bias = Parameter(np.zeros((out_features,))) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    """层归一化：对最后一个维度进行归一化"""
    def __init__(self, normalized_shape, eps=1e-5, bias=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape)) if bias else None

    def __call__(self, x):
        # x.shape = (..., normalized_shape)
        norm_dims = len(self.normalized_shape)
        norm_axes = tuple(range(x.data.ndim - norm_dims, x.data.ndim))
        mean = x.data.mean(axis=norm_axes, keepdims=True)
        var = x.data.var(axis=norm_axes, keepdims=True)
        x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = x_normalized * self.weight.data + (self.bias.data if self.bias else 0)

        requires_grad = x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        children = [c for c in [x, self.weight, self.bias] if c is not None and getattr(c, 'requires_grad', False)]
        out = Tensor(out_data, requires_grad=requires_grad, _children=tuple(children), _op="layernorm")

        def _backward():
            if not out.grad.any():
                return
            reduce_axes = tuple(range(out.grad.ndim - len(self.normalized_shape)))
            if self.weight.requires_grad:
                self.weight.grad += (out.grad * x_normalized).sum(axis=reduce_axes)
            if self.bias and self.bias.requires_grad:
                self.bias.grad += out.grad.sum(axis=reduce_axes)
            if x.requires_grad:
                dxhat = out.grad * self.weight.data
                inv_std = 1.0 / np.sqrt(var + self.eps)
                n = float(np.prod(self.normalized_shape))
                sum_dxhat = dxhat.sum(axis=norm_axes, keepdims=True)
                sum_dxhat_xhat = (dxhat * x_normalized).sum(axis=norm_axes, keepdims=True)
                x.grad += (inv_std / n) * (n * dxhat - sum_dxhat - x_normalized * sum_dxhat_xhat)

        out._backward = _backward
        return out


class Dropout(Module):
    """Dropout正则化"""
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout p must satisfy 0 <= p < 1, got {p}")
        self.p = p

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        # 训练时随机置零
        mask = np.random.binomial(1, 1 - self.p, size=x.data.shape) / (1 - self.p)
        out_data = x.data * mask
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="dropout")

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * mask

        out._backward = _backward
        return out

    def eval(self):
        """切换到评估模式"""
        self.training = False

    def train(self):
        """切换到训练模式"""
        self.training = True


class GELU(Module):
    """GELU激活函数（近似版本）"""
    def __call__(self, x):
        # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # 简化版本: GELU(x) ≈ x * sigmoid(1.702 * x)
        out_data = x.data * (1.0 / (1.0 + np.exp(-1.702 * x.data)))
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="gelu")

        def _backward():
            if x.requires_grad:
                # 近似梯度
                sigmoid = 1.0 / (1.0 + np.exp(-1.702 * x.data))
                grad_sigmoid = 1.702 * sigmoid * (1 - sigmoid)
                x.grad += out.grad * (sigmoid + x.data * grad_sigmoid)

        out._backward = _backward
        return out


class ModuleList(Module):
    """存储Module列表"""
    def __init__(self, modules=None):
        super().__init__()
        self._modules = []
        if modules:
            self._modules.extend(modules)

    def append(self, module):
        self._modules.append(module)

    def __iter__(self):
        return iter(self._modules)

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)

    def parameters(self):
        params = []
        seen = set()

        def add_param(param):
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)

        for module in self._modules:
            if isinstance(module, Module):
                for p in module.parameters():
                    add_param(p)
            elif isinstance(module, Parameter):
                add_param(module)
        return params


class ModuleDict(Module):
    """存储Module字典"""
    def __init__(self, modules=None):
        super().__init__()
        self._modules = {}
        if modules:
            self._modules.update(modules)

    def __setitem__(self, key, module):
        self._modules[key] = module

    def __getitem__(self, key):
        return self._modules[key]

    def __contains__(self, key):
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def parameters(self):
        params = []
        seen = set()

        def add_param(param):
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)

        for module in self._modules.values():
            if isinstance(module, Module):
                for p in module.parameters():
                    add_param(p)
            elif isinstance(module, Parameter):
                add_param(module)
        return params


class MultiHeadAttention(Module):
    """多头自注意力机制（带因果遮罩）"""
    def __init__(self, n_embd, n_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads
        
        # QKV 投影
        self.qkv = Linear(n_embd, 3 * n_embd, bias=True)
        self.out_proj = Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        
        # 因果遮罩（下三角矩阵）
        self.causal_mask = np.tril(np.ones((max_seq_len, max_seq_len)))
    
    def __call__(self, x):
        B, T, C = x.data.shape
        if T > self.causal_mask.shape[0]:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.causal_mask.shape[0]}; "
                "increase max_seq_len when constructing MultiHeadAttention"
            )
        
        # QKV 投影: (B, T, 3*C)
        qkv = self.qkv(x)
        qkv_data = qkv.data.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv_data = qkv_data.transpose(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv_data[0], qkv_data[1], qkv_data[2]  # 每个 (B, nh, T, hd)
        
        # 注意力分数: Q @ K^T / sqrt(d_k)
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # (B, nh, T, T)
        
        # 应用因果遮罩（将上三角设为很小的值）
        mask = self.causal_mask[:T, :T]
        att = np.where(mask == 1, att, -1e9)
        
        # Softmax
        att = self._softmax(att, axis=-1)
        
        # Dropout（训练时）
        dropout_mask = None
        if self.attn_dropout.training and self.attn_dropout.p > 0:
            dropout_mask = np.random.binomial(1, 1 - self.attn_dropout.p, size=att.shape) / (1 - self.attn_dropout.p)
            att = att * dropout_mask
        
        # 加权求和: (B, nh, T, T) @ (B, nh, T, hd) -> (B, nh, T, hd)
        y = att @ v
        
        # 重新组合多头: (B, nh, T, hd) -> (B, T, C)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y_tensor = Tensor(y, requires_grad=qkv.requires_grad, _children=(qkv,), _op="mha")

        def _backward():
            if not qkv.requires_grad:
                return

            dy = y_tensor.grad.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hd)

            # y = att @ v
            datt = dy @ v.transpose(0, 1, 3, 2)              # (B, nh, T, T)
            dv = att.transpose(0, 1, 3, 2) @ dy              # (B, nh, T, hd)

            # dropout backward on attention probabilities
            if dropout_mask is not None:
                datt = datt * dropout_mask

            # softmax backward: dscore = s * (g - sum(g*s))
            sum_gs = (datt * att).sum(axis=-1, keepdims=True)
            dscore = att * (datt - sum_gs)

            # masked positions are constants (-1e9), stop gradient
            dscore = np.where(mask[None, None, :, :] == 1, dscore, 0.0)

            scale = 1.0 / np.sqrt(self.head_dim)
            dscore = dscore * scale

            # score = q @ k^T
            dq = dscore @ k                                # (B, nh, T, hd)
            dk = dscore.transpose(0, 1, 3, 2) @ q         # (B, nh, T, hd)

            # merge back to qkv layout: (B, T, 3*C)
            dqkv_data = np.stack([dq, dk, dv], axis=0)    # (3, B, nh, T, hd)
            dqkv_data = dqkv_data.transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * C)
            qkv.grad += dqkv_data

        y_tensor._backward = _backward
        
        # 输出投影
        out = self.out_proj(y_tensor)
        out = self.resid_dropout(out)
        
        return out
    
    def _softmax(self, x, axis=-1):
        """数值稳定的 softmax"""
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=axis, keepdims=True)


class MLP(Module):
    """前馈神经网络（Transformer 的 FFN）"""
    def __init__(self, n_embd, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * n_embd
        self.fc1 = Linear(n_embd, hidden_dim)
        self.gelu = GELU()
        self.fc2 = Linear(hidden_dim, n_embd)
        self.dropout = Dropout(dropout)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(Module):
    """Transformer 块：LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual"""
    def __init__(self, n_embd, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout=dropout)
    
    def __call__(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
