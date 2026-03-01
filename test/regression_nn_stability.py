"""回归测试: nn 数值稳定性与边界行为"""

import numpy as np

from app.core.nn import LayerNorm, MultiHeadAttention
from app.core.tensor import Tensor


def _finite_diff_grad(
    fn,
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[idx] += eps
        x_neg[idx] -= eps
        grad[idx] = (fn(x_pos) - fn(x_neg)) / (2.0 * eps)
    return grad


def run_layernorm_grad_regression() -> None:
    np.random.seed(123)
    x_data = np.random.randn(2, 3, 4)
    ln = LayerNorm(4)
    ln.weight.data = np.random.randn(4)
    ln.bias.data = np.random.randn(4)

    x = Tensor(x_data.copy(), requires_grad=True)
    y = ln(x)
    loss = y.mean()
    loss.backward()

    def f(inp: np.ndarray) -> float:
        out = ln(Tensor(inp, requires_grad=False))
        return float(out.data.mean())

    grad_num = _finite_diff_grad(f, x_data)
    grad_err = np.max(np.abs(x.grad - grad_num))
    assert grad_err < 1e-5, f"LayerNorm 输入梯度误差过大: max_err={grad_err:.3e}"
    print(f"✅ LayerNorm 梯度回归通过: max_err={grad_err:.3e}")


def run_mha_stability_regression() -> None:
    np.random.seed(7)
    mha = MultiHeadAttention(n_embd=8, n_heads=2, dropout=0.0, max_seq_len=8)

    x_data = (np.random.randn(2, 8, 8) * 50.0).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    out = mha(x)
    assert np.isfinite(out.data).all(), "MHA 前向出现 NaN/Inf"
    loss = out.mean()
    loss.backward()
    assert np.isfinite(x.grad).all(), "MHA 反向出现 NaN/Inf"

    x_long = Tensor(np.random.randn(1, 9, 8), requires_grad=False)
    raised = False
    try:
        _ = mha(x_long)
    except ValueError:
        raised = True
    assert raised, "超出 max_seq_len 时应抛出 ValueError"

    print("✅ MHA 稳定性与边界回归通过")


if __name__ == "__main__":
    run_layernorm_grad_regression()
    run_mha_stability_regression()
