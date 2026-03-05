# 数据集获取总结

## ✅ 当前已有数据

### 1. data/chat_corpus_expanded.txt（推荐使用）
- **行数**: 10,000 行
- **大小**: 676 KB
- **来源**: 自动生成（基于原始数据扩展）
- **质量**: 中等，包含10个类别的对话
- **优点**: 
  - 立即可用，无需下载
  - 比原始数据多55倍
  - 涵盖多种对话场景
  - 足够验证训练流程

### 2. data/chat_corpus.txt（原始数据）
- **行数**: 182 行
- **大小**: 14 KB
- **来源**: 项目原始数据
- **质量**: 高，但数量太少

## 📥 获取更大规模真实数据的方法

### 方案1：Chinese Chatbot Corpus（推荐）

已克隆项目到: `data/chinese_chatbot_corpus/`

**包含8个语料库**:
1. chatterbot - 560条
2. 豆瓣多轮 - 352万条 ⭐
3. PTT八卦语料 - 77万条
4. 青云语料 - 10万条
5. 电视剧对白语料 - 274万条 ⭐
6. 贴吧论坛回帖语料 - 232万条 ⭐
7. 微博语料 - 443万条 ⭐
8. 小黄鸡语料 - 45万条

**使用步骤**:
```bash
# 从以下链接下载原始语料
# 阿里云盘: https://www.aliyundrive.com/s/qXBdAYtz5j5 (提取码: 81ao)
# 或 Google Drive: https://drive.google.com/file/d/1So-m83NdUHexfjJ912rQ4GItdLvnmJMD/view?usp=sharing

# 将解压后的 raw_chat_corpus 文件夹放到:
# /home/shuwen/chatneurx/data/chinese_chatbot_corpus/raw_chat_corpus

# 修改 config.py 中的路径，然后运行:
cd /home/shuwen/chatneurx/data/chinese_chatbot_corpus
python main.py

# 处理后的数据将在 clean_chat_corpus/ 目录下
# 格式为: query \t answer
```

### 方案2：LCCC数据集（需要手动下载）

由于Hugging Face API变更，需要手动下载:

```bash
# 方法1: 使用Git LFS
sudo apt install git-lfs
git lfs install
cd /home/shuwen/chatneurx/data
git clone https://huggingface.co/datasets/silver/lccc

# 方法2: 手动下载Parquet文件
# 访问: https://huggingface.co/datasets/silver/lccc/tree/main
# 下载 train.parquet, 使用pandas处理:
# import pandas as pd
# df = pd.read_parquet('train.parquet')
```

### 方案3：其他开源中文对话数据集

```bash
# CDial-GPT (清华大学)
git clone https://github.com/thu-coai/CDial-GPT.git

# STC (Short Text Conversation)
# Microsoft Research: http://www.nlpir.org/download/STC-corpus.zip

# LCQMC (Large-scale Chinese Question Matching Corpus)
# 哈工大: http://icrc.hitsz.edu.cn/Article/show/171.html
```

## 🚀 推荐训练流程

### 第一步：使用现有10K数据验证训练流程（立即执行）

```bash
cd /home/shuwen/chatneurx

# 使用改进的训练脚本
python scripts/train_improved.py \
    --corpus data/chat_corpus_expanded.txt \
    --batch-size 32 \
    --epochs 20 \
    --hidden-dim 512 \
    --num-layers 6

# 训练时间: 约2-4小时（取决于硬件）
```

### 第二步：获取更大规模数据

在等待第一步训练时，可以：
1. 从阿里云盘或Google Drive下载Chinese Chatbot Corpus
2. 或尝试手动下载LCCC数据集
3. 选择其中一个质量最好的语料（推荐豆瓣或微博）

### 第三步：使用大规模数据重新训练

```bash
# 假设已获取豆瓣语料（352万条）
python scripts/train_improved.py \
    --corpus data/chinese_chatbot_corpus/clean_chat_corpus/douban.tsv \
    --batch-size 64 \
    --epochs 10 \
    --hidden-dim 512 \
    --num-layers 6 \
    --max-samples 100000  # 先用10万条测试

# 如果效果好，可以使用全部数据
```

## 📊 数据对比

| 数据集 | 行数 | 质量 | 可用性 | 推荐度 |
|--------|------|------|--------|--------|
| chat_corpus.txt | 182 | ⭐⭐⭐⭐⭐ | ✅ 立即可用 | ⭐ |
| chat_corpus_expanded.txt | 10,000 | ⭐⭐⭐ | ✅ 立即可用 | ⭐⭐⭐⭐ |
| 豆瓣多轮 | 3,520,000 | ⭐⭐⭐⭐ | 需下载 | ⭐⭐⭐⭐⭐ |
| 微博语料 | 4,430,000 | ⭐⭐⭐ | 需下载 | ⭐⭐⭐⭐ |
| 电视剧对白 | 2,740,000 | ⭐⭐⭐⭐ | 需下载 | ⭐⭐⭐⭐ |
| LCCC | 7,000,000+ | ⭐⭐⭐⭐ | 需手动下载 | ⭐⭐⭐⭐⭐ |

## ⚡ 快速开始（推荐）

**立即开始训练，不要等待下载**：

```bash
# 1. 开始训练（后台运行）
cd /home/shuwen/chatneurx
nohup python scripts/train_improved.py --corpus data/chat_corpus_expanded.txt > train.log 2>&1 &

# 2. 查看训练进度
tail -f train.log

# 3. 训练完成后测试模型
python test_chatneurx_model.py --checkpoint checkpoints/model_improved_best.pkl
```

## 💡 总结

- ✅ **现在就有可用的10,000行数据** - 足够验证整个训练流程
- ⏳ **更大数据集需要下载** - 但不影响立即开始训练
  - **策略**: 先用10K数据训练看效果，同时下载大数据集
- 🎯 **最终目标**: 使用100万+真实对话数据训练生产级模型

## 文件位置一览

```
/home/shuwen/chatneurx/data/
├── chat_corpus.txt                    # 原始数据（182行）
├── chat_corpus_expanded.txt           # 扩展数据（10,000行）⭐ 推荐
├── chinese_chatbot_corpus/            # 中文聊天语料库项目
│   ├── readme.md                      # 数据下载说明
│   ├── main.py                        # 数据处理脚本
│   └── raw_chat_corpus/               # 需要下载到这里
└── [未来] chat_corpus_douban.txt      # 豆瓣语料（需下载）
```
