#!/usr/bin/env python
"""
处理 LCCC 数据集的简化脚本
直接从 Hugging Face 下载并转换为训练格式
"""

import os
import argparse
from tqdm import tqdm


def process_lccc_dataset(
    output_file="data/chat_corpus_lccc.txt",
    max_samples=100000,
    min_length=2,
    max_length=200,
):
    """
    下载并处理 LCCC 数据集
    
    参数:
        output_file: 输出文件路径
        max_samples: 最大样本数
        min_length: 最小句子长度（字符）
        max_length: 最大句子长度（字符）
    """
    
    print("=" * 60)
    print("📥 处理 LCCC 数据集")
    print("=" * 60)
    print()
    
    # 1. 检查依赖
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 未安装 datasets 库")
        print()
        print("请运行以下命令安装:")
        print("  pip install datasets")
        print()
        return False
    
    # 2. 下载数据集
    print("📥 从 Hugging Face 下载 LCCC-base...")
    print("   (首次下载可能需要较长时间，约2-3GB)")
    print()
    
    try:
        # LCCC-base 是较小的版本，LCCC-large 更大但更慢
        dataset = load_dataset("silver/lccc", "base", split="train", trust_remote_code=True)
        print(f"✅ 下载成功！总对话数: {len(dataset):,}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
        print("备选方案:")
        print("1. 检查网络连接")
        print("2. 使用代理: export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 或手动下载数据集")
        return False
    
    # 3. 处理数据
    print()
    print("🔄 处理数据中...")
    print(f"   输出文件: {output_file}")
    print(f"   最大样本: {max_samples:,}")
    print(f"   句子长度: {min_length}-{max_length} 字符")
    print()
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    processed_count = 0
    total_sentences = 0
    seen_sentences = set()  # 用于去重
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(tqdm(dataset, desc="处理进度", total=min(len(dataset), max_samples))):
            if processed_count >= max_samples:
                break
            
            # LCCC 格式: 每个 example 有一个 dialog 字段，包含对话列表
            if 'dialog' not in example:
                continue
            
            dialog = example['dialog']
            
            # 提取对话中的每一句
            for sentence in dialog:
                sentence = sentence.strip()
                
                # 过滤条件
                if not sentence:
                    continue
                if len(sentence) < min_length or len(sentence) > max_length:
                    continue
                if sentence in seen_sentences:  # 去重
                    continue
                
                # 基本清洗
                # 去除URL
                if 'http' in sentence or 'www.' in sentence:
                    continue
                # 去除过多标点
                if sentence.count('。') > 5 or sentence.count('，') > 10:
                    continue
                # 去除广告关键词
                spam_keywords = ['微信', '加我', '广告', '链接', '网址', 'VX']
                if any(kw in sentence for kw in spam_keywords):
                    continue
                
                # 保存
                f.write(sentence + '\n')
                seen_sentences.add(sentence)
                total_sentences += 1
            
            processed_count += 1
            
            # 定期刷新
            if processed_count % 1000 == 0:
                f.flush()
    
    # 4. 统计信息
    print()
    print("=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)
    print(f"📊 统计信息:")
    print(f"   处理的对话数: {processed_count:,}")
    print(f"   提取的句子数: {total_sentences:,}")
    print(f"   输出文件: {output_file}")
    print(f"   文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print()
    print("💡 下一步:")
    print(f"   python scripts/train_improved.py --corpus {output_file}")
    print()
    
    return True


def merge_with_existing(lccc_file, existing_file, output_file):
    """合并 LCCC 数据和现有数据"""
    print("🔗 合并数据集...")
    print(f"   LCCC: {lccc_file}")
    print(f"   现有: {existing_file}")
    print(f"   输出: {output_file}")
    print()
    
    all_lines = []
    
    # 读取 LCCC
    if os.path.exists(lccc_file):
        with open(lccc_file, 'r', encoding='utf-8') as f:
            lccc_lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(lccc_lines)
            print(f"   LCCC: {len(lccc_lines):,} 行")
    
    # 读取现有数据
    if os.path.exists(existing_file):
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(existing_lines)
            print(f"   现有: {len(existing_lines):,} 行")
    
    # 去重
    print("   去重中...")
    all_lines = list(dict.fromkeys(all_lines))
    
    # 打乱
    import random
    random.shuffle(all_lines)
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    
    print()
    print(f"✅ 合并完成！")
    print(f"   总行数: {len(all_lines):,}")
    print(f"   输出: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="处理 LCCC 数据集")
    parser.add_argument("--output", type=str, default="data/chat_corpus_lccc.txt",
                       help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=100000,
                       help="最大样本数（对话数）")
    parser.add_argument("--min-length", type=int, default=2,
                       help="最小句子长度")
    parser.add_argument("--max-length", type=int, default=200,
                       help="最大句子长度")
    parser.add_argument("--merge", action="store_true",
                       help="与现有数据合并")
    parser.add_argument("--existing", type=str, default="data/chat_corpus.txt",
                       help="现有数据文件")
    parser.add_argument("--merge-output", type=str, default="data/chat_corpus_final.txt",
                       help="合并后的输出文件")
    
    args = parser.parse_args()
    
    # 处理 LCCC
    success = process_lccc_dataset(
        output_file=args.output,
        max_samples=args.max_samples,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    
    # 可选：合并
    if success and args.merge:
        print()
        merge_with_existing(args.output, args.existing, args.merge_output)


if __name__ == "__main__":
    main()
