# 使用开源中文对话数据集 - 快速指南

## 🎯 概述

我已经为你准备好了完整的开源中文数据集下载和处理工具，可以将训练数据从182行提升到10万+条，极大提高模型准确率！

## 📦 已创建的工具

1. **[scripts/process_lccc.py](../scripts/process_lccc.py)** - LCCC数据集处理脚本（推荐）
2. **[scripts/download_chinese_datasets.py](../scripts/download_chinese_datasets.py)** - 通用数据集下载工具
3. **[scripts/quick_setup_lccc.sh](../scripts/quick_setup_lccc.sh)** - 一键设置脚本
4. **[docs/DATASET_GUIDE.md](../docs/DATASET_GUIDE.md)** - 详细使用指南

## 🚀 三种使用方式

### 方式1️⃣: 一键脚本（最简单，推荐新手）

```bash
cd /home/shuwen/chatneurx

# 运行一键脚本（会自动安装依赖、下载数据、询问是否训练）
bash scripts/quick_setup_lccc.sh
```

这个脚本会：
- ✅ 自动检查和安装依赖
- ✅ 下载 LCCC 数据集（10万条对话）
- ✅ 自动处理和清洗数据
- ✅ 询问是否立即开始训练
- ✅ 全程有进度提示

**预计时间**: 20-40分钟（取决于网络）

---

### 方式2️⃣: 分步执行（推荐进阶用户）

```bash
cd /home/shuwen/chatneurx

# 步骤1: 安装依赖
pip install datasets tqdm

# 步骤2: 下载并处理 LCCC 数据集（10万条对话）
python scripts/process_lccc.py \
    --output data/chat_corpus_lccc.txt \
    --max-samples 100000 \
    --min-length 5 \
    --max-length 150

# 步骤3: (可选) 与现有数据合并
python scripts/process_lccc.py \
    --merge \
    --existing data/chat_corpus.txt \
    --merge-output data/chat_corpus_final.txt

# 步骤4: 使用新数据训练
python scripts/train_improved.py \
    --corpus data/chat_corpus_lccc.txt \
    --batch-size 32 \
    --epochs 30 \
    --hidden-dim 512 \
    --num-layers 6 \
    --output checkpoints/model_lccc.pkl
```

**优势**: 可以自定义每一步的参数

---

### 方式3️⃣: 使用 Python API（推荐开发者）

```python
from datasets import load_dataset

# 直接使用 Hugging Face datasets
dataset = load_dataset("silver/lccc", "base", split="train")

# 提取数据
dialogues = []
for example in dataset[:100000]:
    if 'dialog' in example:
        dialogues.extend(example['dialog'])

# 保存
with open("data/my_corpus.txt", "w", encoding="utf-8") as f:
    for line in dialogues:
        f.write(line.strip() + "\n")
```

**优势**: 最灵活，可以自定义所有处理逻辑

## 📊 推荐参数配置

### 快速测试（开发阶段）
```bash
--max-samples 10000      # 1万条对话
--epochs 10              # 10轮训练
--batch-size 16          # 小batch
# 训练时间: ~30分钟
```

### 标准配置（推荐）
```bash
--max-samples 100000     # 10万条对话
--epochs 30              # 30轮训练
--batch-size 32          # 标准batch
# 训练时间: ~3-5小时
```

### 高质量配置（生产环境）
```bash
--max-samples 500000     # 50万条对话
--epochs 50              # 50轮训练
--batch-size 64          # 大batch（需要更多内存）
# 训练时间: ~12-24小时
```

## 🎯 关键数据集介绍

### LCCC (推荐⭐⭐⭐⭐⭐)
- **规模**: 700万+对话
- **质量**: 高质量日常对话
- **来源**: 微博、贴吧
- **适用**: 闲聊机器人、日常对话
- **获取**: `python scripts/process_lccc.py`

### Chinese-Chitchat-Corpus (可选⭐⭐⭐⭐)
- **规模**: 50万+对话
- **质量**: 中等
- **来源**: 豆瓣、小黄鸡
- **适用**: 多样化对话
- **获取**: `git clone https://github.com/codemayq/chinese_chatbot_corpus`

### DuConv (可选⭐⭐⭐)
- **规模**: 3万对话
- **质量**: 高质量知识对话
- **来源**: 百度知识图谱
- **适用**: 知识问答
- **获取**: 手动下载

## 💡 使用建议

### 1. 首次使用
```bash
# 先用小数据集验证流程
python scripts/process_lccc.py --max-samples 10000
python scripts/train_improved.py --corpus data/chat_corpus_lccc.txt --epochs 5

# 确认无误后再用完整数据
```

### 2. 网络问题
```bash
# 如果下载慢或失败，使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 重新运行
python scripts/process_lccc.py
```

### 3. 内存不足
```bash
# 减少 max-samples
python scripts/process_lccc.py --max-samples 50000

# 或减小 batch-size
python scripts/train_improved.py --batch-size 16
```

### 4. 数据质量优化
```bash
# 调整过滤参数
python scripts/process_lccc.py \
    --min-length 10 \      # 过滤太短的句子
    --max-length 100 \     # 过滤太长的句子
    --max-samples 100000
```

## 📈 预期效果对比

| 指标 | 原始(182行) | LCCC(10万) | 提升倍数 |
|------|-------------|------------|----------|
| 训练数据量 | 182 | 100,000+ | **550x** |
| 词汇丰富度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 极大提升 |
| 对话连贯性 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 极大提升 |
| 回答准确性 | ⭐⭐ | ⭐⭐⭐⭐ | 显著提升 |
| 泛化能力 | ⭐ | ⭐⭐⭐⭐ | 极大提升 |

## 🔧 故障排除

### 问题1: ImportError: No module named 'datasets'
```bash
# 解决方案
pip install datasets tqdm
```

### 问题2: 下载太慢
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/process_lccc.py
```

### 问题3: 磁盘空间不足
```bash
# 检查空间
df -h

# 减少数据量
python scripts/process_lccc.py --max-samples 50000
```

### 问题4: 内存溢出
```bash
# 分批处理
python scripts/process_lccc.py --max-samples 50000 --output batch1.txt
python scripts/process_lccc.py --max-samples 50000 --output batch2.txt

# 合并
cat batch1.txt batch2.txt > final.txt
```

## 📚 相关文档

- **详细指南**: [docs/DATASET_GUIDE.md](../docs/DATASET_GUIDE.md)
- **快速开始**: [scripts/QUICK_START.md](../scripts/QUICK_START.md)
- **改进方案**: [scripts/improve_accuracy.md](../scripts/improve_accuracy.md)

## ✅ 检查清单

开始之前确认：
- [ ] 已激活虚拟环境: `source ~/.venv/bin/activate`
- [ ] 已安装依赖: `pip install datasets tqdm`
- [ ] 有足够磁盘空间: 至少 5GB
- [ ] 网络连接正常

开始训练之前确认：
- [ ] 数据已下载: `ls -lh data/chat_corpus_lccc.txt`
- [ ] 数据行数合理: `wc -l data/chat_corpus_lccc.txt` (应该>10000)
- [ ] 有足够时间: 训练可能需要数小时

## 🚀 立即开始

**推荐命令（直接复制粘贴）**:

```bash
cd /home/shuwen/chatneurx

# 方法1: 一键运行（最简单）
bash scripts/quick_setup_lccc.sh

# 或方法2: 分步运行
pip install datasets tqdm
python scripts/process_lccc.py --max-samples 100000
python scripts/train_improved.py --corpus data/chat_corpus_lccc.txt --epochs 30
```

## 💬 预期训练时间

| 配置 | 数据量 | 训练时间 | 模型效果 |
|------|--------|----------|----------|
| 测试 | 10,000 | 30分钟 | ⭐⭐⭐ |
| 标准 | 100,000 | 3-5小时 | ⭐⭐⭐⭐ |
| 高质量 | 500,000 | 12-24小时 | ⭐⭐⭐⭐⭐ |

---

**最重要的建议**: 先用10,000条数据快速验证，确认整个流程无误后，再使用100,000+条数据进行完整训练！

有问题请查看 [docs/DATASET_GUIDE.md](../docs/DATASET_GUIDE.md) 获取更多帮助。
