# 开源中文对话数据集使用指南

## 🌟 推荐数据集

### 1. LCCC (Large-scale Chinese Conversation Corpus) ⭐⭐⭐⭐⭐

**最推荐！质量高，规模大**

- **规模**: 700万+对话，1200万+句子
- **来源**: 微博、贴吧等社交媒体
- **特点**: 日常闲聊为主，口语化
- **GitHub**: https://github.com/thu-coai/CDial-GPT
- **Hugging Face**: https://huggingface.co/datasets/silver/lccc

**快速使用**：
```bash
# 方法1: 一键脚本（推荐）
bash scripts/quick_setup_lccc.sh

# 方法2: 手动处理
pip install datasets tqdm
python scripts/process_lccc.py --max-samples 100000
```

### 2. Chinese-Chitchat-Corpus ⭐⭐⭐⭐

**中等规模，多样化来源**

- **规模**: 50万+对话
- **来源**: 豆瓣、微博、小黄鸡、PTT等
- **特点**: 多领域覆盖
- **GitHub**: https://github.com/codemayq/chinese_chatbot_corpus

**下载方式**：
```bash
# 克隆仓库
git clone https://github.com/codemayq/chinese_chatbot_corpus.git

# 处理数据
cd chinese_chatbot_corpus
# 合并所有数据文件
cat chit-chat/*.tsv > ../data/chitchat_raw.txt
```

### 3. DuConv (百度对话数据集) ⭐⭐⭐

**知识驱动对话**

- **规模**: 3万对话，9万轮次
- **来源**: 百度知识图谱
- **特点**: 基于知识的对话
- **链接**: https://dataset-bj.cdn.bcebos.com/duconv/

**下载方式**：
```bash
# 下载数据文件
wget https://dataset-bj.cdn.bcebos.com/duconv/train.txt
wget https://dataset-bj.cdn.bcebos.com/duconv/dev.txt
wget https://dataset-bj.cdn.bcebos.com/duconv/test.txt

# 移动到数据目录
mv *.txt data/duconv/
```

### 4. KdConv (知识驱动对话) ⭐⭐⭐

**电影、音乐等领域对话**

- **规模**: 4.5万对话
- **来源**: 电影、音乐、旅游
- **特点**: 多轮对话，知识密集
- **GitHub**: https://github.com/thu-coai/KdConv

### 5. CLUE对话数据 ⭐⭐⭐

**多任务对话数据**

- **规模**: 10万+
- **来源**: CLUE基准测试
- **特点**: 任务导向
- **链接**: https://github.com/CLUEbenchmark/CLUE

## 📥 快速开始（推荐流程）

### 选项A: 使用一键脚本（最简单）

```bash
cd /home/shuwen/chatneurx

# 运行一键脚本
bash scripts/quick_setup_lccc.sh

# 这个脚本会：
# 1. 自动安装依赖
# 2. 下载 LCCC 数据集
# 3. 处理成训练格式
# 4. 询问是否立即开始训练
```

### 选项B: 分步执行

```bash
# 1. 安装依赖
pip install datasets tqdm

# 2. 下载并处理 LCCC（推荐10万条对话）
python scripts/process_lccc.py \
    --output data/chat_corpus_lccc.txt \
    --max-samples 100000 \
    --min-length 5 \
    --max-length 150

# 3. 可选：与现有数据合并
python scripts/process_lccc.py \
    --merge \
    --existing data/chat_corpus.txt \
    --merge-output data/chat_corpus_final.txt

# 4. 开始训练
python scripts/train_improved.py \
    --corpus data/chat_corpus_lccc.txt \
    --batch-size 32 \
    --epochs 30 \
    --hidden-dim 512 \
    --num-layers 6
```

### 选项C: 使用 Hugging Face 界面

```python
# quick_download.py
from datasets import load_dataset

# 下载数据集
dataset = load_dataset("silver/lccc", "base", split="train")

# 处理并保存
with open("data/lccc_raw.txt", "w", encoding="utf-8") as f:
    for example in dataset[:100000]:  # 取前10万条
        if 'dialog' in example:
            for sentence in example['dialog']:
                f.write(sentence.strip() + "\n")
```

## 🔧 数据处理技巧

### 1. 数据清洗

```python
# 去除低质量数据
- 过短（<5字）或过长（>200字）
- 包含URL、广告关键词
- 重复内容
- 乱码
```

### 2. 数据增强

```python
# 在 process_lccc.py 中已实现
- 去重
- 长度过滤
- 关键词过滤
- 格式统一
```

### 3. 数据采样

```python
# 根据硬件能力选择数据量
- 个人电脑: 10,000 - 50,000 条
- 工作站: 50,000 - 200,000 条
- 服务器: 200,000+ 条
```

## 📊 数据集对比

| 数据集 | 规模 | 质量 | 下载难度 | 推荐度 |
|--------|------|------|----------|--------|
| LCCC | 700万+ | ⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐⭐ |
| Chitchat | 50万+ | ⭐⭐⭐ | 简单 | ⭐⭐⭐⭐ |
| DuConv | 3万+ | ⭐⭐⭐⭐⭐ | 中等 | ⭐⭐⭐ |
| KdConv | 4.5万+ | ⭐⭐⭐⭐ | 中等 | ⭐⭐⭐ |

## 🎯 针对不同场景的推荐

### 场景1: 日常闲聊机器人
**推荐**: LCCC + Chitchat
```bash
python scripts/process_lccc.py --max-samples 100000
# 数据特点: 口语化、日常对话
```

### 场景2: 知识问答
**推荐**: DuConv + KdConv
```bash
# 下载 DuConv
# 数据特点: 基于知识图谱
```

### 场景3: 客服机器人
**推荐**: 自建数据 + LCCC
```bash
# 使用公司内部客服对话数据
# 补充 LCCC 增加语言多样性
```

### 场景4: 教育/陪伴
**推荐**: LCCC + 定制数据
```bash
# LCCC 提供基础对话能力
# 添加教育相关对话数据
```

## 🚀 性能优化建议

### 1. 数据量选择

```bash
# 开发测试（快速验证）
--max-samples 10000

# 正常训练
--max-samples 50000-100000

# 高质量模型
--max-samples 200000+
```

### 2. 内存优化

如果内存不足：
```bash
# 分批处理
python scripts/process_lccc.py --max-samples 50000 --output batch1.txt
python scripts/process_lccc.py --max-samples 50000 --output batch2.txt

# 合并
cat batch1.txt batch2.txt > final.txt
```

### 3. 网络问题

如果下载失败：
```bash
# 使用 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 重新运行
python scripts/process_lccc.py
```

## 📝 常见问题

### Q1: 下载太慢怎么办？
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://127.0.0.1:7890
```

### Q2: 内存不够怎么办？
```bash
# 减少 max-samples
python scripts/process_lccc.py --max-samples 50000

# 或分批处理
```

### Q3: 数据质量不满意？
```bash
# 调整过滤参数
--min-length 10       # 增加最小长度
--max-length 100      # 减小最大长度

# 添加自定义过滤规则（修改 process_lccc.py）
```

### Q4: 需要多少数据才够？
```bash
# 最小可用: 10,000 条
# 推荐量: 50,000 - 100,000 条
# 高质量: 200,000+ 条

# 经验法则: 数据越多越好，但收益递减
```

## 🎉 预期效果

使用 LCCC 100,000 条数据训练后：

| 指标 | 原模型(182行) | LCCC模型(10万条) | 提升 |
|------|---------------|------------------|------|
| 对话连贯性 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 极大 |
| 词汇丰富度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 极大 |
| 回答准确性 | ⭐⭐ | ⭐⭐⭐⭐ | 显著 |
| 泛化能力 | ⭐⭐ | ⭐⭐⭐⭐ | 显著 |

## 📚 更多资源

- **数据集列表**: https://github.com/CLUEbenchmark/CLUEDatasetSearch
- **对话系统资源**: https://github.com/AImissq/Chinese-Dialogue-Corpus
- **NLP数据集**: https://github.com/InsaneLife/ChineseNLPCorpus

## 🆘 获取帮助

遇到问题？
1. 检查 [QUICK_START.md](../scripts/QUICK_START.md)
2. 查看脚本输出的错误信息
3. 确认依赖已正确安装: `pip list | grep datasets`

---

**最重要的建议**: 先用小数据集（10,000条）快速验证整个流程，确认无误后再下载完整数据集训练！
