import sys
import os
try:
    # Try to import from neurx framework (recommended)
    from neurx import Tensor
    from neurx.nn import Module, Parameter, Embedding, Linear
    from neurx.optim import AdamW
    from neurx.losses import cross_entropy
except ImportError:
    # Fallback: Try legacy tensor module if available
    try:
        from tensor.core.tensor import Tensor
        from tensor.core.nn import Module, Parameter, Embedding, Linear
        from tensor.core.optim import AdamW
        from tensor.core.losses import cross_entropy
    except ImportError:
        raise ImportError(
            "Neither 'neurx' nor 'tensor' module found. "
            "Please install NeurX: pip install /home/shuwen/neurx"
        )

try:
    from app.core.models import TinyLM
except ImportError:
    # If models.py has tensor dependency, try models_neurx as fallback
    try:
        from app.core.models_neurx import NeurXTinyLM as TinyLM
    except ImportError:
        TinyLM = None

__all__ = [
    "Tensor",
    "Module",
    "Parameter",
    "Embedding",
    "Linear",
    "AdamW",
    "cross_entropy",
    "TinyLM",
]
