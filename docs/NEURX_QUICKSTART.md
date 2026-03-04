# NeurX 集成快速开始指南

## 📦 第一步：添加 NeurX 依赖

编辑 `requirements.txt`，添加：

```
-e /home/shuwen/neurx
```

完整的 requirements.txt：

```
# 核心依赖
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.3
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0
tensorboard>=2.13.0
sentencepiece>=0.1.99
tiktoken>=0.4.0

# Web 服务
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.8.0

# 监控和日志
prometheus-client>=0.20.0
PyJWT>=2.9.0
python-multipart>=0.0.9

# NeurX 深度学习框架
-e /home/shuwen/neurx
```

## 🚀 第二步：安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装所有依赖
pip install -r requirements.txt
```

## 💻 第三步：训练模型

### 快速测试（30 秒）

```bash
python app/training/train_neurx.py --model-size tiny --num-epochs 1
```

### 训练小模型（2-3 分钟）

```bash
python app/training/train_neurx.py \
    --model-size small \
    --num-epochs 3 \
    --batch-size 16 \
    --num-batches-per-epoch 20
```

### 训练大模型（5-10 分钟）

```bash
python app/training/train_neurx.py \
    --model-size base \
    --num-epochs 5 \
    --batch-size 32 \
    --learning-rate 5e-4 \
    --num-batches-per-epoch 50 \
    --save-path checkpoints/chatmodel_base.pkl
```

## 📝 使用已训练模型进行推理

```python
import neurx as nx
import pickle
from app.core.models_neurx import create_chatmodel_base

# 加载检查点
with open('checkpoints/chatmodel_base.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# 创建模型
vocab_size = checkpoint['vocab_size']
model = create_chatmodel_base(vocab_size)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 推理
char_to_id = checkpoint['char_to_id']
text = "人工智能"
token_ids = [char_to_id[c] for c in text]
input_ids = nx.array([token_ids], dtype='int64')

with nx.no_grad():
    output = model(input_ids)
    logits = output['logits']
    print(f"Output shape: {logits.shape}")
```

## 🔧 命令行选项

```bash
python app/training/train_neurx.py --help
```

主要选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--model-size` | tiny | 模型大小：tiny, small, base, large |
| `--num-epochs` | 3 | 训练 epoch 数 |
| `--batch-size` | 16 | 批大小 |
| `--seq-len` | 64 | 序列长度 |
| `--learning-rate` | 1e-3 | 学习率 |
| `--grad-clip` | 1.0 | 梯度裁剪值 |
| `--save-path` | None | 模型保存路径 |

## 📊 模型规格

| 模型 | Hidden Dim | Layers | Heads | 参数数量 | 训练时间（1 epoch） |
|------|-----------|--------|-------|---------|-------------------|
| Tiny | 128 | 2 | 2 | ~50K | 5 秒 |
| Small | 256 | 4 | 4 | ~500K | 20 秒 |
| Base | 768 | 6 | 8 | ~85M | 2-3 分钟 |
| Large | 1024 | 12 | 16 | ~300M | 5-10 分钟 |

## 🐛 故障排除

### 问题 1: 找不到 neurx 模块

```bash
# 检查安装
python -c "import neurx; print(neurx.__version__)"

# 重新安装
pip install -e /home/shuwen/neurx
```

### 问题 2: CUDA 相关错误

```bash
# 禁用 CUDA，使用 CPU
export TENSOR_CUDA=0
python app/training/train_neurx.py --model-size tiny
```

### 问题 3: 内存不足

```bash
# 减小批大小
python app/training/train_neurx.py \
    --model-size small \
    --batch-size 8 \
    --seq-len 32
```

## 📚 完整文档

详见 `docs/NEURX_INTEGRATION_GUIDE.md`

## ✅ 下一步

1. ✓ 安装依赖
2. ✓ 运行训练脚本
3. [ ] 实现推理 API
4. [ ] 集成到 Web 服务
5. [ ] 部署到生产环境

---

**祝你使用 NeurX 框架训练 ChatNeurX 大模型！** 🚀
