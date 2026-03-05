# 数据集存储位置说明

## 📂 数据存储目录

所有数据集默认保存在项目根目录的 `data/` 文件夹下。

### 目录结构

```
chatneurx/
├── data/                          # 主数据目录
│   ├── chat_corpus.txt            # 原始对话语料（182行）
│   ├── chat_corpus_lccc.txt       # LCCC数据集（下载后）
│   ├── chat_corpus_expanded.txt   # 扩展语料（generate_more_data.py生成）
│   ├── chat_corpus_final.txt      # 合并后的最终语料
│   └── raw/                       # 原始数据存储目录
│       └── lccc_train.txt         # LCCC原始数据
```

## 📥 各脚本的输出位置

### 1. process_lccc.py（推荐使用）
```bash
python scripts/process_lccc.py
```
**默认输出**: `data/chat_corpus_lccc.txt`

**自定义输出**:
```bash
python scripts/process_lccc.py --output data/my_corpus.txt
```

**合并输出**:
```bash
python scripts/process_lccc.py --merge --merge-output data/chat_corpus_final.txt
```

### 2. download_chinese_datasets.py
```bash
python scripts/download_chinese_datasets.py --action download
```
**默认输出**: `data/raw/lccc_train.txt`

**自定义输出目录**:
```bash
python scripts/download_chinese_datasets.py --output-dir data/my_raw
```

### 3. generate_more_data.py
```bash
python scripts/generate_more_data.py
```
**默认输出**: `data/chat_corpus_expanded.txt`

**自定义输出**:
```bash
python scripts/generate_more_data.py --output data/my_expanded.txt
```

## 🔍 查看已下载的数据

### 检查数据目录
```bash
# 查看 data 目录内容
ls -lh data/

# 查看各文件行数
wc -l data/*.txt

# 查看文件大小
du -h data/*.txt
```

### 查看数据内容
```bash
# 查看前10行
head -10 data/chat_corpus_lccc.txt

# 查看后10行
tail -10 data/chat_corpus_lccc.txt

# 统计总行数
wc -l data/chat_corpus_lccc.txt
```

## 📊 典型数据集大小

| 文件名 | 数据量 | 文件大小 | 来源 |
|--------|--------|----------|------|
| chat_corpus.txt | 182行 | ~20KB | 原始数据 |
| chat_corpus_expanded.txt | 10,000行 | ~1MB | 生成数据 |
| chat_corpus_lccc.txt | 100,000+行 | ~10-50MB | LCCC数据集 |
| chat_corpus_final.txt | 100,000+行 | ~10-50MB | 合并数据 |

## 🎯 推荐使用流程

### 方案1: 使用LCCC数据集（推荐）

```bash
# 1. 下载LCCC数据（10万条）
python scripts/process_lccc.py --max-samples 100000

# 2. 数据会保存到
#    → data/chat_corpus_lccc.txt

# 3. 使用这个数据训练
python scripts/train_improved.py --corpus data/chat_corpus_lccc.txt
```

### 方案2: 合并多个数据源

```bash
# 1. 生成扩展数据
python scripts/generate_more_data.py

# 2. 下载LCCC并合并
python scripts/process_lccc.py --merge --existing data/chat_corpus.txt

# 3. 最终数据保存到
#    → data/chat_corpus_final.txt

# 4. 训练
python scripts/train_improved.py --corpus data/chat_corpus_final.txt
```

### 方案3: 使用一键脚本

```bash
# 一键下载和设置
bash scripts/quick_setup_lccc.sh

# 数据会自动保存到 data/ 目录
# 脚本会显示具体位置
```

## 🔧 自定义数据位置

如果你想使用自定义位置：

```bash
# 下载到自定义位置
python scripts/process_lccc.py \
    --output /path/to/my/data/corpus.txt

# 训练时指定自定义位置
python scripts/train_improved.py \
    --corpus /path/to/my/data/corpus.txt
```

## 💾 Hugging Face 缓存位置

LCCC数据集从Hugging Face下载时，原始数据会缓存到：

**默认位置**:
```bash
~/.cache/huggingface/datasets/
```

**查看缓存**:
```bash
ls -lh ~/.cache/huggingface/datasets/
du -sh ~/.cache/huggingface/datasets/
```

**自定义缓存位置**:
```bash
export HF_HOME=/path/to/cache
python scripts/process_lccc.py
```

## 🗑️ 清理数据

### 清理处理后的数据
```bash
# 删除处理后的txt文件（保留原始缓存）
rm data/chat_corpus_lccc.txt
rm data/chat_corpus_expanded.txt
rm data/chat_corpus_final.txt
```

### 清理Hugging Face缓存
```bash
# 清理所有HF缓存（会删除所有下载的数据集）
rm -rf ~/.cache/huggingface/datasets/

# 只清理LCCC缓存
rm -rf ~/.cache/huggingface/datasets/silver___lccc/
```

## 📝 验证数据完整性

```bash
# 检查文件是否存在
test -f data/chat_corpus_lccc.txt && echo "✅ 文件存在" || echo "❌ 文件不存在"

# 检查文件是否为空
if [ -s data/chat_corpus_lccc.txt ]; then
    echo "✅ 文件不为空"
    wc -l data/chat_corpus_lccc.txt
else
    echo "❌ 文件为空或不存在"
fi

# 查看文件前几行确认格式
head -5 data/chat_corpus_lccc.txt
```

## 🚀 快速命令参考

```bash
# 查看所有数据文件
find data/ -name "*.txt" -type f

# 统计所有txt文件行数
find data/ -name "*.txt" -exec wc -l {} \;

# 查看数据目录占用空间
du -sh data/

# 创建数据目录（如果不存在）
mkdir -p data/raw
```

## ❓ 常见问题

### Q: 数据下载后找不到？
**A**: 检查以下位置：
1. `data/chat_corpus_lccc.txt` - process_lccc.py的默认输出
2. `data/raw/lccc_train.txt` - download_chinese_datasets.py的输出
3. `~/.cache/huggingface/datasets/` - HF原始缓存

### Q: 如何知道数据下载成功？
**A**: 
```bash
# 查看文件大小（应该>1MB）
ls -lh data/chat_corpus_lccc.txt

# 查看行数（应该>10000）
wc -l data/chat_corpus_lccc.txt
```

### Q: 可以移动数据文件吗？
**A**: 可以，训练时用 `--corpus` 参数指定新位置：
```bash
python scripts/train_improved.py --corpus /new/path/corpus.txt
```

---

**总结**: 所有数据默认在 `data/` 目录，使用 `ls -lh data/` 查看具体文件。
