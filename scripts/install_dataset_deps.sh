#!/bin/bash
# 安装依赖脚本

echo "======================================"
echo "📦 安装开源数据集所需依赖"
echo "======================================"
echo ""

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  建议在虚拟环境中安装"
    echo ""
    read -p "是否继续在全局环境安装？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消安装"
        echo ""
        echo "激活虚拟环境后重新运行:"
        echo "  source ~/.venv/bin/activate"
        echo "  bash scripts/install_dataset_deps.sh"
        exit 1
    fi
else
    echo "✅ 虚拟环境: $VIRTUAL_ENV"
fi

echo ""
echo "安装以下依赖包:"
echo "  - datasets (Hugging Face 数据集库)"
echo "  - tqdm (进度条)"
echo ""

# 安装
pip install datasets tqdm

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ 依赖安装成功！"
    echo "======================================"
    echo ""
    echo "验证安装:"
    python scripts/test_environment.py
    echo ""
    echo "下一步:"
    echo "  bash scripts/quick_setup_lccc.sh"
else
    echo ""
    echo "❌ 安装失败"
    echo ""
    echo "尝试手动安装:"
    echo "  pip install --upgrade pip"
    echo "  pip install datasets tqdm"
fi
