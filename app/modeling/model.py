"""GPT模型实现（wrapper，指向core实现）"""
# 从 core 模块导入所有实现
from app.core.gpt_model import (
    GPT,
    Block,
    CausalSelfAttention,
    MLP,
)
try:
    from neurx.nn import LayerNorm, GELU, Dropout, MoE
except ImportError:
    from tensor.core.nn import LayerNorm, GELU, Dropout, MoE
from app.modeling.config import ModelConfig

# 导出给外部使用
__all__ = ['GPT', 'Block', 'CausalSelfAttention', 'MLP', 'MoE', 'LayerNorm', 'ModelConfig']
