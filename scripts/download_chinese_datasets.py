#!/usr/bin/env python
"""下载和处理开源中文对话数据集"""

import os
import json
import argparse
from pathlib import Path


def download_lccc_base():
    """
    下载 LCCC-base (Large-scale Chinese Conversation Corpus)
    - 最大的开源中文对话数据集
    - 约700万对话，1200万句子
    - 来源：https://github.com/thu-coai/CDial-GPT
    """
    print("=" * 60)
    print("📥 LCCC-base (Large-scale Chinese Conversation Corpus)")
    print("=" * 60)
    print("数据集信息：")
    print("  - 约700万对话对")
    print("  - 来自微博、贴吧等社交媒体")
    print("  - GitHub: https://github.com/thu-coai/CDial-GPT")
    print()
    
    print("下载方式：")
    print()
    print("方法1: 使用 Hugging Face (推荐)")
    print("-" * 60)
    print("pip install datasets")
    print()
    print("from datasets import load_dataset")
    print("dataset = load_dataset('silver/lccc', 'base')")
    print("# 或者")
    print("dataset = load_dataset('thu-coai/lccc', 'base')")
    print()
    
    print("方法2: 直接下载")
    print("-" * 60)
    print("# LCCC-base 数据文件")
    print("wget https://cloud.tsinghua.edu.cn/f/1b60e36d8c9a438fa1cd/?dl=1 -O LCCC-base.json")
    print()
    
    print("方法3: Git LFS (需要较大空间)")
    print("-" * 60)
    print("git lfs install")
    print("git clone https://huggingface.co/datasets/silver/lccc")
    print()


def download_chinese_chitchat():
    """
    下载 Chinese-Chitchat-Corpus
    - 中等规模的中文闲聊数据集
    - 约50万对话
    """
    print("=" * 60)
    print("📥 Chinese-Chitchat-Corpus")
    print("=" * 60)
    print("数据集信息：")
    print("  - 约50万对话对")
    print("  - 来自豆瓣、微博、小黄鸡等")
    print("  - GitHub: https://github.com/codemayq/chinese_chatbot_corpus")
    print()
    
    print("下载命令：")
    print("-" * 60)
    print("git clone https://github.com/codemayq/chinese_chatbot_corpus.git")
    print("cd chinese_chatbot_corpus")
    print()


def download_duconv():
    """
    下载 DuConv (百度对话数据集)
    - 基于知识图谱的对话数据集
    - 约3万对话
    """
    print("=" * 60)
    print("📥 DuConv (百度对话数据集)")
    print("=" * 60)
    print("数据集信息：")
    print("  - 约3万对话，约9万轮次")
    print("  - 基于知识图谱的对话")
    print("  - GitHub: https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO")
    print()
    
    print("下载方式：")
    print("-" * 60)
    print("访问: https://dataset-bj.cdn.bcebos.com/duconv/")
    print("下载文件:")
    print("  - train.txt")
    print("  - dev.txt")
    print("  - test.txt")
    print()


def download_with_huggingface(dataset_name, output_dir, max_samples=100000):
    """使用 Hugging Face datasets 下载数据"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 未安装 datasets 库")
        print("安装命令: pip install datasets")
        return False
    
    print(f"📥 正在从 Hugging Face 下载 {dataset_name}...")
    
    try:
        if dataset_name == "lccc":
            # LCCC 数据集
            dataset = load_dataset("silver/lccc", "base", split="train")
            output_file = os.path.join(output_dir, "lccc_train.txt")
            
            print(f"   总样本数: {len(dataset)}")
            print(f"   提取前 {max_samples} 条...")
            
            os.makedirs(output_dir, exist_ok=True)
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    # LCCC 格式: 每个example包含对话列表
                    if 'dialog' in example:
                        dialog = example['dialog']
                        # 提取问答对
                        for j in range(len(dialog) - 1):
                            question = dialog[j].strip()
                            answer = dialog[j + 1].strip()
                            if question and answer:
                                f.write(f"Q: {question}\n")
                                f.write(f"A: {answer}\n")
                                count += 1
                    
                    if (i + 1) % 10000 == 0:
                        print(f"   已处理: {i + 1} 条对话...")
            
            print(f"✅ 成功下载并处理 {count} 对问答")
            print(f"   保存到: {output_file}")
            return True
            
        else:
            print(f"❌ 未知数据集: {dataset_name}")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def convert_to_corpus_format(input_file, output_file, format_type="qa"):
    """
    转换数据集格式为训练语料格式
    
    format_type:
    - "qa": 问答格式 (Q: ... A: ...)
    - "line": 每行一句
    - "json": JSON格式
    """
    print(f"🔄 转换格式: {input_file} -> {output_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return False
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                
                if format_type == "qa":
                    # Q: ... A: ... 格式
                    lines = f_in.readlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            f_out.write(line + '\n')
                
                elif format_type == "line":
                    # 每行一句
                    for line in f_in:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # 移除 Q: 和 A: 前缀
                            if line.startswith('Q: ') or line.startswith('A: '):
                                line = line[3:]
                            f_out.write(line + '\n')
                
                elif format_type == "json":
                    # JSON格式
                    data = json.load(f_in)
                    for item in data:
                        if isinstance(item, dict):
                            if 'question' in item and 'answer' in item:
                                f_out.write(item['question'].strip() + '\n')
                                f_out.write(item['answer'].strip() + '\n')
                        elif isinstance(item, list):
                            for text in item:
                                f_out.write(text.strip() + '\n')
        
        print(f"✅ 转换完成: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


def merge_corpus_files(input_files, output_file, max_lines=None, deduplicate=True):
    """合并多个语料文件"""
    print(f"🔗 合并语料文件到: {output_file}")
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    all_lines = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"⚠️  跳过不存在的文件: {input_file}")
            continue
        
        print(f"   读取: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(lines)
            print(f"      +{len(lines)} 行")
    
    print(f"   总行数: {len(all_lines)}")
    
    # 去重
    if deduplicate:
        print("   去重中...")
        original_count = len(all_lines)
        all_lines = list(dict.fromkeys(all_lines))  # 保持顺序去重
        print(f"   去重后: {len(all_lines)} 行 (删除 {original_count - len(all_lines)} 条重复)")
    
    # 限制行数
    if max_lines and len(all_lines) > max_lines:
        print(f"   限制到 {max_lines} 行")
        import random
        random.shuffle(all_lines)
        all_lines = all_lines[:max_lines]
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    
    print(f"✅ 合并完成: {output_file}")
    print(f"   最终行数: {len(all_lines)}")


def main():
    parser = argparse.ArgumentParser(description="下载和处理开源中文对话数据集")
    parser.add_argument("--action", type=str, default="info",
                       choices=["info", "download", "convert", "merge"],
                       help="操作类型")
    parser.add_argument("--dataset", type=str, default="lccc",
                       choices=["lccc", "chitchat", "duconv", "all"],
                       help="数据集名称")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="输出目录")
    parser.add_argument("--max-samples", type=int, default=100000,
                       help="最大样本数")
    parser.add_argument("--merge-output", type=str, default="data/chat_corpus_large.txt",
                       help="合并后的输出文件")
    
    args = parser.parse_args()
    
    if args.action == "info":
        print("\n")
        print("🌟" * 30)
        print("开源中文对话数据集资源")
        print("🌟" * 30)
        print()
        
        download_lccc_base()
        print()
        download_chinese_chitchat()
        print()
        download_duconv()
        print()
        
        print("=" * 60)
        print("💡 推荐使用流程")
        print("=" * 60)
        print()
        print("1. 安装依赖:")
        print("   pip install datasets")
        print()
        print("2. 下载 LCCC 数据集 (推荐，最大):")
        print("   python scripts/download_chinese_datasets.py --action download --dataset lccc --max-samples 100000")
        print()
        print("3. 转换格式:")
        print("   python scripts/download_chinese_datasets.py --action convert")
        print()
        print("4. 或使用更简单的方式:")
        print("   python scripts/process_lccc.py  # 见下一个脚本")
        print()
        
    elif args.action == "download":
        if args.dataset == "lccc":
            download_with_huggingface("lccc", args.output_dir, args.max_samples)
        elif args.dataset == "all":
            download_with_huggingface("lccc", args.output_dir, args.max_samples)
        else:
            print(f"数据集 {args.dataset} 需要手动下载，请参考 info 信息")
    
    elif args.action == "convert":
        # 转换示例
        raw_files = [
            "data/raw/lccc_train.txt",
            "data/chat_corpus.txt",
        ]
        convert_to_corpus_format(raw_files[0], "data/lccc_converted.txt", format_type="line")
        
    elif args.action == "merge":
        # 合并所有语料
        files_to_merge = [
            "data/chat_corpus.txt",
            "data/chat_corpus_expanded.txt",
            "data/raw/lccc_train.txt",
            "data/lccc_converted.txt",
        ]
        merge_corpus_files(files_to_merge, args.merge_output, max_lines=None, deduplicate=True)


if __name__ == "__main__":
    main()
