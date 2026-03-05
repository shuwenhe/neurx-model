#!/usr/bin/env python
"""快速测试脚本 - 验证环境和下载小样本数据"""

import sys
import os

print("=" * 60)
print("🔍 环境检查")
print("=" * 60)
print()

# 检查Python版本
print(f"Python 版本: {sys.version.split()[0]}")

# 检查必要的库
required_packages = {
    'datasets': '用于下载 Hugging Face 数据集',
    'tqdm': '用于显示进度条',
    'numpy': '数值计算',
}

missing_packages = []

for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"✅ {package:20s} - {description}")
    except ImportError:
        print(f"❌ {package:20s} - {description} (未安装)")
        missing_packages.append(package)

print()

if missing_packages:
    print("⚠️  缺少以下依赖包:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print()
    print("安装命令:")
    print(f"   pip install {' '.join(missing_packages)}")
    print()
    sys.exit(1)

print("=" * 60)
print("✅ 环境检查通过！")
print("=" * 60)
print()

# 测试下载小样本
print("=" * 60)
print("🧪 测试下载（仅10条样本）")
print("=" * 60)
print()

try:
    from datasets import load_dataset
    
    print("正在连接 Hugging Face...")
    dataset = load_dataset("silver/lccc", "base", split="train", streaming=True)
    
    print("✅ 连接成功！")
    print()
    print("获取前10条样本...")
    
    count = 0
    for i, example in enumerate(dataset):
        if i >= 10:
            break
        if 'dialog' in example:
            print(f"\n对话 {i+1}:")
            for j, sentence in enumerate(example['dialog'][:3]):  # 只显示前3句
                print(f"  {j+1}. {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            count += 1
    
    print()
    print(f"✅ 成功获取 {count} 条样本！")
    print()
    print("=" * 60)
    print("🎉 测试完成！环境配置正常")
    print("=" * 60)
    print()
    print("下一步:")
    print("  1. 下载完整数据: python scripts/process_lccc.py --max-samples 100000")
    print("  2. 或运行一键脚本: bash scripts/quick_setup_lccc.sh")
    print()

except Exception as e:
    print(f"❌ 测试失败: {e}")
    print()
    print("可能的原因:")
    print("  1. 网络连接问题 - 尝试使用镜像: export HF_ENDPOINT=https://hf-mirror.com")
    print("  2. 防火墙限制")
    print("  3. 依赖包版本问题 - 尝试: pip install --upgrade datasets")
    print()
    sys.exit(1)
