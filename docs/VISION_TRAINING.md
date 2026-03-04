# 视觉编码器训练指南

让你的 LLM 模型能够理解图片内容。

## 快速开始

### 方法 1: 使用 Hugging Face 数据集（推荐）

使用 Flickr30k 数据集（约 3万张图片带描述）：

```bash
# 1. 安装依赖
pip install datasets pillow torchvision

# 2. 开始训练（会自动下载数据集）
python train_vision_real.py \
    --data-source huggingface \
    --dataset-name nlphuji/flickr30k \
    --batch-size 8 \
    --epochs 10 \
    --lr 1e-4

# 训练时间: 约2-4小时 (GPU)
```

### 方法 2: 使用本地图片数据

准备你自己的图片和描述：

```bash
# 1. 创建数据目录
mkdir -p data/vision_train/images

# 2. 放入图片
cp your_images/*.jpg data/vision_train/images/

# 3. 创建描述文件 captions.json
# 格式: {"图片名.jpg": "图片描述", ...}
cat > data/vision_train/captions.json << 'EOF'
{
  "cat.jpg": "一只橙色的猫坐在窗台上看着外面",
  "mountain.jpg": "雪山在蓝天白云下显得格外壮观",
  "city.jpg": "繁华的城市街道上车水马龙"
}
EOF

# 4. 开始训练
python train_vision_real.py \
    --data-source local \
    --data-path data/vision_train \
    --batch-size 4 \
    --epochs 20
```

### 方法 3: 快速演示（合成数据）

如果只想快速测试流程：

```bash
# 使用合成数据快速训练
python train_vision.py

# 这会生成随机图片，不能真正理解图片，仅用于测试
```

## 使用训练后的模型

训练完成后，模型保存在 `checkpoints/vision_trained_model.pt`

```bash
# 1. 设置环境变量使用新模型
export LLM_CHECKPOINT=checkpoints/vision_trained_model.pt

# 2. 重启服务
make serve-dev

# 3. 上传图片测试
# 打开浏览器 http://localhost:3000
# 上传图片并提问
```

## 推荐的数据集

### 小规模（快速训练）
- **Flickr8k**: 8千张图片，每张5个描述
- **Flickr30k**: 3万张图片，每张5个描述 ⭐ 推荐
- 训练时间: 2-4小时 (GPU)

### 中等规模（更好效果）
- **COCO Captions**: 12万张图片
- 训练时间: 8-12小时 (GPU)

### 大规模（最佳效果）
- **Conceptual Captions**: 300万张图片
- 训练时间: 1-2天 (GPU)

## 训练参数说明

```bash
python train_vision_real.py --help

参数:
  --data-source     数据源: local 或 huggingface
  --data-path       本地数据路径
  --dataset-name    HuggingFace数据集名称
  --batch-size      批次大小 (默认: 8)
  --epochs          训练轮数 (默认: 10)
  --lr              学习率 (默认: 1e-4)
  --checkpoint      基础模型路径
  --output          输出模型路径
```

## 训练监控

训练过程中会显示：
- Loss（损失值）: 越低越好，通常会从 8-10 降到 3-5
- LR（学习率）: 使用余弦退火调度
- 进度条显示当前epoch和批次

示例输出：
```
Epoch 5/10: 100%|████| 3750/3750 [15:23<00:00, loss=3.24, avg=3.45, lr=5.2e-05]
Epoch 5 完成, 平均损失: 3.45
→ 新的最佳损失! 保存模型...
```

## 验证效果

训练后测试图片理解能力：

```bash
# 1. 启动服务
export LLM_CHECKPOINT=checkpoints/vision_trained_model.pt
make serve-dev

# 2. 测试API
curl -X POST http://localhost:8000/v1/generate-multipart \
  -F "image=@test_image.jpg" \
  -F "prompt=描述这张图片" \
  -F "max_new_tokens=100"
```

## 常见问题

### Q: 训练需要多少显存？
A: 
- batch_size=4: 约6GB
- batch_size=8: 约10GB
- batch_size=16: 约16GB

如果显存不足，减小 batch_size

### Q: 可以用CPU训练吗？
A: 可以，但会很慢（GPU的10-20倍时间）

### Q: 训练多久能看到效果？
A: 
- 合成数据: 5分钟（无实际效果）
- Flickr30k: 2-4小时（基本效果）
- COCO: 8-12小时（较好效果）

### Q: Loss降不下来怎么办？
A: 
1. 检查数据质量
2. 降低学习率: `--lr 5e-5`
3. 增加训练轮数: `--epochs 20`
4. 确保图片和文本匹配

### Q: 如何使用自己的图片？
A: 按照"方法2"准备数据，确保：
- 图片格式: JPG/PNG
- 描述清晰准确
- 数量建议 >1000张

## 进阶技巧

### 使用预训练的视觉编码器

可以使用 CLIP 等预训练模型初始化视觉编码器：

```python
# 在 train_vision_real.py 中添加
from transformers import CLIPVisionModel

clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
# 将 CLIP 权重迁移到你的 visual_encoder
```

### 混合数据集训练

```python
# 同时使用多个数据集
datasets = [
    HuggingFaceImageCaptionDataset("nlphuji/flickr30k"),
    HuggingFaceImageCaptionDataset("nlphuji/flickr8k"),
    LocalImageCaptionDataset("data/my_images")
]
combined_dataset = torch.utils.data.ConcatDataset(datasets)
```

## 下一步

训练完成后，你的模型就能：
- 识别图片中的物体
- 描述场景和动作
- 回答关于图片的问题
- 理解图片上下文

建议阅读：
- `model.py` - 查看模型架构
- `serve.py` - 查看图片处理流程
- `docs/` - 更多文档
