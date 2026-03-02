"""Step 3: 迷你训练 10 step 验证（自研后端）"""

import numpy as np

from app.core.models import TinyLM
from tensor.core.optim import AdamW


def run_step3_mini_train_check() -> None:
    np.random.seed(42)
    vocab_size = 1000
    model = TinyLM(vocab_size=vocab_size, n_embd=128)
    optimizer = AdamW(model.parameters(), lr=3e-3)

    batch_size = 4
    seq_len = 32

    # 固定小批次，验证是否能在短步数内过拟合
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    y = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)

    losses: list[float] = []

    for step in range(10):
        optimizer.zero_grad()
        logits, loss = model(x, y)

        assert logits.shape == (batch_size, seq_len, vocab_size), (
            f"logits 形状错误: {logits.shape}, 期望 {(batch_size, seq_len, vocab_size)}"
        )
        assert loss is not None and np.isfinite(loss.item()), f"step={step} loss 非法: {loss.item()}"

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    start_loss = losses[0]
    end_loss = losses[-1]

    # 采用宽松但明确的检查：末尾 loss 需要低于起始 loss
    assert end_loss < start_loss, (
        f"10 step 后 loss 未下降: start={start_loss:.6f}, end={end_loss:.6f}, losses={losses}"
    )

    print("✅ Step 3 完成：迷你训练 10 step 打通")
    print(f"   初始 loss: {start_loss:.6f}")
    print(f"   结束 loss: {end_loss:.6f}")
    print("   loss 序列:")
    print("   " + " -> ".join(f"{v:.4f}" for v in losses))


if __name__ == "__main__":
    run_step3_mini_train_check()
