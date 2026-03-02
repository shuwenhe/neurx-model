"""回归测试: batched matmul 的反向传播梯度形状与数值检查"""

import numpy as np

from tensor.core.tensor import Tensor


def run_regression_batched_matmul_grad() -> None:
    np.random.seed(7)

    b, t, c, v = 4, 6, 8, 5
    x_data = np.random.randn(b, t, c)
    w_data = np.random.randn(c, v)

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)

    out = x @ w
    loss = out.mean()
    loss.backward()

    assert w.grad is not None, "w.grad 不应为 None"
    assert w.grad.shape == w.data.shape, (
        f"梯度形状错误: got {w.grad.shape}, expected {w.data.shape}"
    )

    grad_out = np.ones_like(out.data) / out.data.size
    expected_w_grad = np.einsum("btc,btv->cv", x_data, grad_out)

    assert np.allclose(w.grad, expected_w_grad, atol=1e-8), (
        f"w.grad 数值不匹配: max_err={np.max(np.abs(w.grad - expected_w_grad)):.3e}"
    )

    print("✅ 回归测试通过: batched matmul 梯度形状与数值正确")
    print(f"   x.shape={x_data.shape}, w.shape={w_data.shape}, out.shape={out.data.shape}")


if __name__ == "__main__":
    run_regression_batched_matmul_grad()
