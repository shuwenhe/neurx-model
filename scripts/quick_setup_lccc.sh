#!/bin/bash
# 快速设置脚本：下载LCCC数据集并开始训练

set -e  # 遇到错误立即退出

echo "======================================"
echo "🚀 ChatNeurX - LCCC 数据集快速设置"
echo "======================================"
echo ""

# 切换到项目目录
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
echo "📁 项目目录: $PROJECT_DIR"
echo ""

# 1. 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  未检测到虚拟环境，尝试激活..."
    if [ -f "$HOME/.venv/bin/activate" ]; then
        source "$HOME/.venv/bin/activate"
        echo "✅ 激活虚拟环境: $HOME/.venv"
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ 激活虚拟环境: venv"
    else
        echo "❌ 找不到虚拟环境"
        echo "请先创建虚拟环境: python -m venv venv"
        exit 1
    fi
else
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
fi
echo ""

# 2. 安装必要的依赖
echo "📦 检查依赖..."
pip list | grep -q datasets || {
    echo "安装 datasets 库..."
    pip install datasets tqdm -q
}
echo "✅ 依赖已安装"
echo ""

# 3. 下载并处理 LCCC 数据集
echo "======================================"
echo "📥 下载 LCCC 数据集"
echo "======================================"
echo ""
echo "这将下载约2-3GB的数据，提取100,000条对话"
echo "预计时间: 10-30分钟（取决于网络速度）"
echo ""
read -p "是否继续？[Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "取消操作"
    exit 0
fi

python scripts/process_lccc.py \
    --output data/chat_corpus_lccc.txt \
    --max-samples 100000 \
    --min-length 5 \
    --max-length 150

if [ $? -ne 0 ]; then
    echo "❌ 数据处理失败"
    echo ""
    echo "可能的原因："
    echo "1. 网络问题 - 尝试使用 HF 镜像: export HF_ENDPOINT=https://hf-mirror.com"
    echo "2. 依赖问题 - 确保安装了 datasets: pip install datasets"
    echo "3. 磁盘空间不足 - 至少需要5GB空间"
    exit 1
fi

echo ""

# 4. 可选：与现有数据合并
if [ -f "data/chat_corpus.txt" ]; then
    echo "📁 检测到现有数据: data/chat_corpus.txt"
    read -p "是否与现有数据合并？[Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        python scripts/process_lccc.py \
            --output data/chat_corpus_lccc.txt \
            --merge \
            --existing data/chat_corpus.txt \
            --merge-output data/chat_corpus_final.txt
        
        CORPUS_FILE="data/chat_corpus_final.txt"
    else
        CORPUS_FILE="data/chat_corpus_lccc.txt"
    fi
else
    CORPUS_FILE="data/chat_corpus_lccc.txt"
fi

echo ""
echo "======================================"
echo "✅ 数据准备完成！"
echo "======================================"
echo ""
echo "📊 使用的语料文件: $CORPUS_FILE"
echo "📦 文件大小: $(du -h $CORPUS_FILE | cut -f1)"
echo "📝 行数: $(wc -l < $CORPUS_FILE)"
echo ""

# 5. 询问是否立即开始训练
read -p "是否立即开始训练？[Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo ""
    echo "======================================"
    echo "🚀 开始训练"
    echo "======================================"
    echo ""
    echo "使用配置："
    echo "  - 语料: $CORPUS_FILE"
    echo "  - Batch Size: 32"
    echo "  - Epochs: 30"
    echo "  - Hidden Dim: 512"
    echo "  - Layers: 6"
    echo ""
    echo "预计训练时间: 2-6小时（取决于硬件）"
    echo ""
    
    python scripts/train_improved.py \
        --corpus "$CORPUS_FILE" \
        --batch-size 32 \
        --epochs 30 \
        --learning-rate 3e-4 \
        --hidden-dim 512 \
        --num-layers 6 \
        --seq-len 128 \
        --output checkpoints/model_lccc.pkl
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "🎉 训练完成！"
        echo "======================================"
        echo ""
        echo "模型已保存到:"
        echo "  - checkpoints/model_lccc.pkl"
        echo "  - checkpoints/model_lccc_best.pkl (最佳模型)"
        echo ""
        echo "测试模型:"
        echo "  python test_chatneurx_model.py --checkpoint checkpoints/model_lccc_best.pkl"
        echo ""
    else
        echo "❌ 训练失败"
    fi
else
    echo ""
    echo "跳过训练。手动训练命令："
    echo ""
    echo "python scripts/train_improved.py \\"
    echo "    --corpus $CORPUS_FILE \\"
    echo "    --batch-size 32 \\"
    echo "    --epochs 30 \\"
    echo "    --hidden-dim 512 \\"
    echo "    --num-layers 6"
    echo ""
fi

echo "======================================"
echo "✅ 设置完成！"
echo "======================================"
