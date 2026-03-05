#!/bin/bash
# 数据集位置查看脚本

echo "========================================"
echo "📂 ChatNeurX 数据集存储位置"
echo "========================================"
echo ""

# 检查data目录
if [ -d "data" ]; then
    echo "✅ data/ 目录存在"
    echo ""
    echo "📁 当前文件列表："
    echo "----------------------------------------"
    ls -lh data/*.txt 2>/dev/null | awk '{printf "  %-30s %8s  %s\n", $9, $5, ""}'
    
    echo ""
    echo "📊 文件行数统计："
    echo "----------------------------------------"
    for file in data/*.txt; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file" 2>/dev/null)
            printf "  %-30s %10s 行\n" "$(basename $file)" "$lines"
        fi
    done
else
    echo "❌ data/ 目录不存在"
    echo "运行: mkdir -p data"
fi

echo ""
echo "========================================"
echo "📥 数据下载后的存储位置"
echo "========================================"
echo ""
echo "各脚本的默认输出位置："
echo ""
echo "1️⃣  process_lccc.py (推荐)"
echo "   → data/chat_corpus_lccc.txt"
echo "   用法: python scripts/process_lccc.py"
echo ""
echo "2️⃣  generate_more_data.py"
echo "   → data/chat_corpus_expanded.txt"
echo "   用法: python scripts/generate_more_data.py"
echo ""
echo "3️⃣  合并后的数据"
echo "   → data/chat_corpus_final.txt"
echo "   用法: python scripts/process_lccc.py --merge"
echo ""
echo "4️⃣  原始LCCC下载"
echo "   → data/raw/lccc_train.txt"
echo "   用法: python scripts/download_chinese_datasets.py"
echo ""

echo "========================================"
echo "💡 快速操作"
echo "========================================"
echo ""
echo "查看所有数据文件:"
echo "  find data/ -name '*.txt' -type f"
echo ""
echo "统计所有文件行数:"
echo "  wc -l data/*.txt"
echo ""
echo "查看文件内容:"
echo "  head -20 data/chat_corpus_lccc.txt"
echo ""
echo "检查文件大小:"
echo "  du -h data/*.txt"
echo ""

echo "========================================"
echo "🚀 下载数据集"
echo "========================================"
echo ""
echo "方法1 - 使用LCCC数据集 (推荐):"
echo "  python scripts/process_lccc.py --max-samples 100000"
echo ""
echo "方法2 - 一键脚本:"
echo "  bash scripts/quick_setup_lccc.sh"
echo ""
echo "方法3 - 生成扩展数据:"
echo "  python scripts/generate_more_data.py --target-lines 10000"
echo ""

# 检查HF缓存
if [ -d "$HOME/.cache/huggingface/datasets" ]; then
    echo "========================================"
    echo "💾 Hugging Face 缓存位置"
    echo "========================================"
    echo ""
    echo "位置: $HOME/.cache/huggingface/datasets/"
    echo "大小: $(du -sh ~/.cache/huggingface/datasets/ 2>/dev/null | cut -f1)"
    echo ""
fi

echo "========================================"
echo "📖 查看详细文档"
echo "========================================"
echo ""
echo "  cat docs/DATA_LOCATION.md"
echo "  cat README_DATASET.md"
echo ""
