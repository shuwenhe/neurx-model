"""
使用 NeurX 框架训练 ChatNeurX 大模型

特点：
- 基于 NeurX 的自动微分引擎
- 支持 GPU 加速（如果可用）
- 完整的训练循环和评估
- 实时性能监控
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
import numpy as np

import neurx
import neurx.nn as nn
import neurx.optim as optim

from app.core.models_neurx import (
    create_chatmodel_tiny,
    create_chatmodel_small,
    create_chatmodel_base,
    create_chatmodel_large,
)


def build_timestamped_save_path(save_path):
    """在输出文件名末尾追加 YYYYMMDDHHMMSS 时间戳。"""
    if not save_path:
        return save_path

    tz_utc8 = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz_utc8).strftime("%Y%m%d%H%M%S")
    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".pkl"
    return f"{base}{timestamp}{ext}"


def get_sample_corpus():
    """获取示例文本语料库"""
    return [
        "北京是中国的首都，位于华北平原中部。",
        "人工智能正在改变世界，推动社会进步。",
        "语言模型可以生成文本，理解语义信息。",
        "机器学习需要数据和算力，还需要算法。",
        "模型训练要关注损失函数下降，学习率很重要。",
        "深度学习在计算机视觉领域取得了巨大成就。",
        "自然语言处理技术在电商和搜索引擎中应用。",
        "神经网络通过反向传播算法实现参数更新。",
        "数据预处理对模型训练的质量有重要影响。",
        "特征工程是机器学习中的关键环节。",
        "超参数调优需要在验证集上反复测试。",
        "正则化技术可以防止模型过拟合。",
        "批归一化加快了神经网络的训练速度。",
        "注意力机制大幅提高了序列模型的性能。",
        "Transformers 架构革命了自然语言处理。",
        "迁移学习让我们可以利用预训练模型。",
        "多任务学习能够提高模型的泛化能力。",
        "知识蒸馏可以将大模型压缩为小模型。",
        "梯度裁剪防止了训练过程中的梯度爆炸。",
        "学习率预热有助于模型的稳定训练。",
    ] * 10


def load_corpus(dataset_file=None):
    """加载训练语料，优先使用 dataset 文件。"""
    if dataset_file:
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        with open(dataset_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"Dataset file is empty: {dataset_file}")

        return lines

    return get_sample_corpus()


def char_tokenize(text):
    """简单的字符级分词"""
    return [ord(c) for c in text]


def tokenize_corpus(corpus):
    """将文本语料转换为 token IDs
    
    Args:
        corpus: 文本列表
        
    Returns:
        (token_ids, vocab_size, char_to_id)
    """
    # 收集所有字符
    all_chars = set()
    token_ids = []
    
    for text in corpus:
        for char in text:
            all_chars.add(char)
    
    # 创建字符映射
    chars = sorted(list(all_chars))
    char_to_id = {c: i for i, c in enumerate(chars)}
    
    # 编码
    for text in corpus:
        ids = [char_to_id[c] for c in text]
        token_ids.extend(ids)
    
    return np.array(token_ids, dtype=np.int64), len(chars), char_to_id


def make_batches(token_ids, batch_size, seq_len, num_batches=100):
    """创建训练批次
    
    Args:
        token_ids: token ID 数组
        batch_size: 批大小
        seq_len: 序列长度
        num_batches: 每个 epoch 的批数
        
    Yields:
        (input_ids, targets) 元组
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    num_tokens = token_ids.shape[0]
    
    if num_tokens < seq_len + 1:
        raise ValueError(f"Corpus too small: {num_tokens} tokens, need at least {seq_len + 1}")
    
    max_start = num_tokens - seq_len - 1
    
    for _ in range(num_batches):
        # 随机选择起点
        starts = np.random.randint(0, max_start, size=batch_size)
        
        # 创建输入和目标
        x = np.stack([token_ids[s:s + seq_len] for s in starts], axis=0)
        y = np.stack([token_ids[s + 1:s + seq_len + 1] for s in starts], axis=0)
        
        yield neurx.Tensor(x), neurx.Tensor(y)


def train_one_epoch(model, optimizer, train_loader, epoch, args):
    """训练一个 epoch
    
    Args:
        model: 模型
        optimizer: 优化器
        train_loader: 数据加载器
        epoch: 当前 epoch 号
        args: 命令行参数
        
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # 前向传播
        output = model(x, targets=y)
        loss = output['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if args.grad_clip > 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq_sum = (p.grad ** 2).sum()
                    if hasattr(grad_sq_sum, "sqrt"):
                        param_norm = grad_sq_sum.sqrt()
                        total_norm += param_norm ** 2
                    else:
                        total_norm += float(grad_sq_sum)

            if not hasattr(total_norm, "sqrt"):
                total_norm = float(np.sqrt(total_norm))
            else:
                total_norm = total_norm.sqrt()
            
            if total_norm > args.grad_clip:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad * (args.grad_clip / (total_norm + 1e-6))
        
        # 更新参数
        optimizer.step()
        
        # 记录损失
        loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
        total_loss += loss_value
        num_batches += 1
        
        # 定期打印
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch} [{batch_idx + 1}/{args.num_batches_per_epoch}] "
                  f"Loss: {loss_value:.4f} (Avg: {avg_loss:.4f})")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, epoch_time


def train_neurx(args):
    """主训练函数
    
    Args:
        args: 命令行参数
        
    Returns:
        训练好的模型
    """
    print("=" * 70)
    print("ChatNeurX with NeurX Framework - Training Script")
    print("=" * 70)
    
    # 设置随机种子
    if hasattr(neurx, "manual_seed"):
        neurx.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取数据
    print("\n[1/4] Preparing data...")
    corpus = load_corpus(args.dataset_file)
    token_ids, vocab_size, char_to_id = tokenize_corpus(corpus)
    print(f"      Corpus size: {len(corpus)} texts")
    print(f"      Vocab size: {vocab_size}")
    print(f"      Total tokens: {len(token_ids)}")
    if args.dataset_file:
        print(f"      Dataset file: {args.dataset_file}")
    
    # 创建模型
    print(f"\n[2/4] Creating {args.model_size} model...")
    
    model_creators = {
        'tiny': create_chatmodel_tiny,
        'small': create_chatmodel_small,
        'base': create_chatmodel_base,
        'large': create_chatmodel_large,
    }
    
    model = model_creators[args.model_size](vocab_size)
    
    # 统计参数
    param_count = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {param_count:,}")
    
    # 创建优化器
    print(f"\n[3/4] Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"      Optimizer: Adam")
    print(f"      Learning rate: {args.learning_rate}")
    print(f"      Batch size: {args.batch_size}")
    print(f"      Sequence length: {args.seq_len}")
    
    # 训练循环
    print(f"\n[4/4] Starting training for {args.num_epochs} epochs...")
    print("-" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 创建数据加载器
        train_loader = make_batches(
            token_ids,
            args.batch_size,
            args.seq_len,
            num_batches=args.num_batches_per_epoch
        )
        
        # 训练一个 epoch
        avg_loss, epoch_time = train_one_epoch(
            model, optimizer, train_loader, epoch + 1, args
        )
        
        # 打印结果
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # 记录最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ✓ Best loss updated: {best_loss:.4f}")
        
        print("-" * 70)
    
    print("\n✅ Training completed!")
    print(f"\nFinal Results:")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Total parameters: {param_count:,}")
    
    # 保存模型
    if args.save_path:
        final_save_path = build_timestamped_save_path(args.save_path)
        print(f"\nSaving model to {final_save_path}...")
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        
        checkpoint = {
            'model_state': model.state_dict(),
            'vocab_size': vocab_size,
            'char_to_id': char_to_id,
            'config': {
                'model_size': args.model_size,
                'hidden_dim': getattr(model, 'hidden_dim', None),
                'num_layers': getattr(model, 'num_layers', None),
            }
        }
        
        try:
            import pickle
            with open(final_save_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"✓ Model saved successfully!")
        except Exception as e:
            print(f"⚠ Warning: Failed to save model: {e}")
    
    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Train ChatNeurX with NeurX Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with Tiny model
  python train_neurx.py --model-size tiny --num-epochs 1

  # Train small model
  python train_neurx.py --model-size small --num-epochs 3 --batch-size 16

  # Train base model
  python train_neurx.py --model-size base --num-epochs 5 --batch-size 32

  # Train with GPU
  TENSOR_CUDA=1 python train_neurx.py --model-size base
        """)
    
    # 模型配置
    parser.add_argument(
        '--model-size', 
        type=str, 
        default='tiny',
        choices=['tiny', 'small', 'base', 'large'],
        help='Model size (default: tiny)'
    )
    
    # 训练配置
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-batches-per-epoch', type=int, default=50, help='Batches per epoch')
    parser.add_argument('--log-interval', type=int, default=10, help='Print frequency')
    
    # 输出
    parser.add_argument('--save-path', type=str, default=None, help='Path to save model checkpoint')
    parser.add_argument(
        '--dataset-file',
        type=str,
        default='dataset/text/neurx_train_mix_v1.txt',
        help='Text dataset file path, one sample per line'
    )
    
    args = parser.parse_args()
    
    # 运行训练
    try:
        model = train_neurx(args)
        print("\n✅ Training script completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
