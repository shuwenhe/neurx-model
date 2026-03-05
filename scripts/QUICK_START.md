# ChatNeurX 模型准确率提升 - 快速开始指南

## 🎯 核心问题

当前模型准确率受限于：
1. **训练数据太少**：仅182行
2. **模型太小**：hidden_dim=256, 2层
3. **训练不足**：仅3个epoch

## 🚀 快速改进方案（3步走）

### 第一步：扩充数据 ⭐⭐⭐⭐⭐ （最重要）

```bash
cd /home/shuwen/chatneurx

# 方案A：快速生成10,000行数据（模板+增强）
python scripts/generate_more_data.py --target-lines 10000

# 生成的文件：data/chat_corpus_expanded.txt
```

**更好的方案B：使用真实数据集**
```bash
# 下载LCCC或其他中文对话数据集
# https://github.com/thu-coai/CDial-GPT
# 或使用GPT-4生成高质量数据
```

### 第二步：使用改进配置训练 ⭐⭐⭐⭐

```bash
# 使用改进的训练脚本（更大模型，更多epoch）
python scripts/train_improved.py \
    --corpus data/chat_corpus_expanded.txt \
    --batch-size 32 \
    --epochs 30 \
    --learning-rate 3e-4 \
    --hidden-dim 512 \
    --num-layers 6 \
    --seq-len 128

# 预计训练时间：根据硬件，可能需要数小时
```

**配置说明**：
- `hidden_dim`: 256→512 (模型容量翻倍)
- `num_layers`: 2→6 (深度增加3倍)
- `epochs`: 3→30 (训练充分)
- `batch_size`: 4→32 (更稳定的梯度)
- `seq_len`: 64→128 (处理更长上下文)

### 第三步：测试和评估 ⭐⭐⭐

```bash
# 测试改进后的模型
python test_chatneurx_model.py --checkpoint checkpoints/model_improved_best.pkl

# 对比原模型和改进模型
python test_chatneurx_model.py --checkpoint checkpoints/model_core.pkl
```

## 📊 预期效果

| 指标 | 原模型 | 改进模型 | 提升 |
|------|--------|----------|------|
| 数据量 | 182行 | 10,000行 | 55倍 |
| 模型参数 | ~0.3M | ~8M | 27倍 |
| 训练步数 | ~100 | ~9,000 | 90倍 |
| 对话连贯性 | ⭐⭐ | ⭐⭐⭐⭐ | 明显提升 |
| 回答准确性 | ⭐⭐ | ⭐⭐⭐⭐ | 明显提升 |

## 💡 进阶优化（可选）

### 1. 使用完整GPT架构
```bash
# 修改 app/modeling/config.py
python app/training/train_full_gpt.py  # 如果存在
```

### 2. 获取开源中文数据集
```bash
# 推荐数据集：
# - LCCC: 千万级中文对话
# - DuConv: 百度对话数据集
# - Chinese-Chitchat-Corpus
# 
# 下载后放到 data/ 目录
```

### 3. 启用高级特性
编辑 [app/modeling/config.py](../app/modeling/config.py):
```python
config = ModelConfig(
    n_layer=12,           # 更深
    n_head=12,
    n_embd=768,           # 更宽
    block_size=256,       # 更长上下文
    
    rmsnorm_enabled=True,   # 更稳定
    rope_enabled=True,      # 更好的位置编码
    swiglu_enabled=True,    # 更好的激活
)
```

### 4. 超参数调优
```bash
# 实验不同配置
python scripts/train_improved.py --learning-rate 1e-4
python scripts/train_improved.py --batch-size 64
python scripts/train_improved.py --dropout 0.2
```

## 📝 监控训练

训练时关注这些指标：

✅ **好的信号**：
- Loss持续下降
- 验证loss不上升（无过拟合）
- 生成文本越来越连贯

❌ **坏的信号**：
- Loss不下降或震荡 → 降低学习率
- 验证loss上升 → 过拟合，增加dropout或更多数据
- Loss变成NaN → 梯度爆炸，降低学习率

## 🔧 故障排除

### 问题1：内存不足
```bash
# 减小batch size
--batch-size 16

# 减小模型
--hidden-dim 256 --num-layers 4
```

### 问题2：训练太慢
```bash
# 减少数据量先快速验证
--corpus data/chat_corpus.txt --epochs 5

# 或使用GPU（如果可用）
```

### 问题3：效果仍不好
```bash
# 检查数据质量
head -100 data/chat_corpus_expanded.txt

# 增加更多真实数据（最重要！）
# 增加训练轮数
--epochs 50
```

## 📚 参考资料

详细说明：[scripts/improve_accuracy.md](improve_accuracy.md)

核心建议：
1. **数据第一**：10,000+条 > 1,000条 > 182条
2. **模型适配**：数据多→模型大，数据少→模型小
3. **充分训练**：至少20-30个epoch
4. **验证集监控**：避免过拟合
5. **实验对比**：记录不同配置的效果

## 🎉 预期时间线

- **Day 1**: 生成扩充数据 (10分钟)
- **Day 2-3**: 训练改进模型 (几小时到1天)
- **Day 4**: 测试评估 (1小时)
- **Week 2**: 获取真实数据集，重新训练 (关键提升)
- **Week 3+**: 持续优化超参数

---

**最重要的一点**：真实的高质量对话数据是提升准确率的关键！模型架构和训练技巧是其次的。优先投入精力获取10,000+真实对话数据。
