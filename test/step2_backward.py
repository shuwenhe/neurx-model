"""Step 2: 单步反向传播验证（自研后端）"""

import numpy as np

from app.core.models import TinyLM
from tensor.core.optim import AdamW


def _pick_param_tensor(model: TinyLM):
    for param in model.parameters():
        if getattr(param, "requires_grad", False):
            return param
    raise RuntimeError("模型中没有可训练参数")


def run_step2_backward_check() -> None:
    np.random.seed(42)
    vocab_size = 1000
    model = TinyLM(vocab_size=vocab_size, n_embd=128)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    batch_size = 2
    seq_len = 32
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    y = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)

    tracked_param = _pick_param_tensor(model)
    param_before = tracked_param.data.copy()

    optimizer.zero_grad()
    logits, loss = model(x, y)

    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"logits 形状错误: {logits.shape}, 期望 {(batch_size, seq_len, vocab_size)}"
    )
    assert loss is not None, "loss 不应为 None"
    assert np.isfinite(loss.item()), f"loss 非有限值: {loss.item()}"

    loss.backward()

    grad_norm_sq = 0.0
    grad_found = False
    for param in model.parameters():
        if param.grad is not None:
            grad_found = True
            grad_norm_sq += float((param.grad ** 2).sum())
    assert grad_found, "未找到任何梯度，backward 可能未生效"
    grad_norm = grad_norm_sq ** 0.5
    assert grad_norm > 0, "梯度范数为 0，训练步无效"

    optimizer.step()

    param_after = tracked_param.data.copy()
    param_changed = not np.array_equal(param_before, param_after)
    assert param_changed, "参数未发生变化，optimizer.step 可能未生效"

    print("✅ Step 2 完成：单步反向传播打通")
    print(f"   输入形状: {tuple(x.shape)}")
    print(f"   logits形状: {tuple(logits.shape)}")
    print(f"   loss: {loss.item():.6f}")
    print(f"   grad_norm: {grad_norm:.6f}")
    print("   参数更新: 已发生")


if __name__ == "__main__":
    run_step2_backward_check()
