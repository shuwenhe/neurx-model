#!/usr/bin/env python
"""
下载并处理 Chinese-Chitchat-Corpus
这是一个可用的开源中文对话数据集
"""

import os
import urllib.request
import gzip
import shutil

def download_duconv():
    """下载DuConv数据集（百度对话数据）"""
    
    print("=" * 60)
    print("📥 下载 DuConv 中文对话数据集")
    print("=" * 60)
    print()
    
    output_dir = "data/duconv"
    os.makedirs(output_dir, exist_ok=True)
    
    urls = [
        "https://dataset-bj.cdn.bcebos.com/duconv/train.txt",
        "https://dataset-bj.cdn.bcebos.com/duconv/dev.txt",
        "https://dataset-bj.cdn.bcebos.com/duconv/test.txt",
    ]
    
    print("正在下载 DuConv 数据集...")
    print()
    
    downloaded_files = []
    
    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)
        
        try:
            print(f"下载: {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ 保存到: {filepath}")
            print(f"   大小: {os.path.getsize(filepath) / 1024:.2f} KB")
            print()
            downloaded_files.append(filepath)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print()
    
    return downloaded_files


def process_duconv_to_chat_corpus(input_files, output_file):
    """将DuConv格式转换为chat_corpus格式"""
    
    print("=" * 60)
    print("🔄 处理 DuConv 数据")
    print("=" * 60)
    print()
    
    processed_lines = []
    
    for input_file in input_files:
        print(f"处理: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        import json
                        data = json.loads(line)
                        
                        # DuConv 格式：包含多轮对话
                        if 'conversation' in data:
                            conversation = data['conversation']
                            # 提取对话对
                            for i in range(0, len(conversation) - 1, 2):
                                if i + 1 < len(conversation):
                                    q = conversation[i].strip()
                                    a = conversation[i + 1].strip()
                                    if q and a:
                                        processed_lines.append(f"{q}\t{a}\n")
                        
                    except json.JSONDecodeError:
                        # 如果不是JSON格式，尝试按制表符分割
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            q = parts[0].strip()
                            a = parts[1].strip()
                            if q and a:
                                processed_lines.append(f"{q}\t{a}\n")
            
            print(f"✅ 提取了 {len(processed_lines)} 条对话")
            print()
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            print()
    
    # 保存处理后的数据
    if processed_lines:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)
        
        print(f"✅ 保存到: {output_file}")
        print(f"   总行数: {len(processed_lines)}")
        print(f"   大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        print()
        
        return True
    
    return False


def download_xiaohuangji():
    """下载小黄鸡对话数据（另一个流行的中文对话数据集）"""
    
    print("=" * 60)
    print("📥 尝试下载小黄鸡对话数据")
    print("=" * 60)
    print()
    
    output_dir = "data/xiaohuangji"
    os.makedirs(output_dir, exist_ok=True)
    
    # 小黄鸡数据的GitHub raw链接
    url = "https://raw.githubusercontent.com/candlewill/Dialog_Corpus/master/xiaohuangji50w_nofenci.conv"
    filepath = os.path.join(output_dir, "xiaohuangji.conv")
    
    try:
        print(f"下载小黄鸡数据...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ 保存到: {filepath}")
        print(f"   大小: {os.path.getsize(filepath) / 1024:.2f} KB")
        print()
        
        # 处理数据
        output_file = "data/chat_corpus_xiaohuangji.txt"
        process_xiaohuangji(filepath, output_file)
        
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
        return False


def process_xiaohuangji(input_file, output_file):
    """处理小黄鸡对话数据"""
    
    print("🔄 处理小黄鸡数据...")
    print()
    
    processed_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 小黄鸡格式：E开头是问题，M开头是回答
            current_q = None
            for line in lines:
                line = line.strip()
                if line.startswith('E'):
                    current_q = line[2:].strip()
                elif line.startswith('M') and current_q:
                    current_a = line[2:].strip()
                    if current_q and current_a:
                        processed_lines.append(f"{current_q}\t{current_a}\n")
                    current_q = None
        
        if processed_lines:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)
            
            print(f"✅ 保存到: {output_file}")
            print(f"   总行数: {len(processed_lines)}")
            print(f"   大小: {os.path.getsize(output_file) / 1024:.2f} KB")
            print()
            
            return True
    
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        print()
    
    return False


if __name__ == "__main__":
    print()
    print("🚀 开始下载真实中文对话数据集")
    print()
    
    success = False
    
    # 方案1: DuConv（百度对话数据）
    print("【方案1】尝试下载 DuConv 数据集...")
    print()
    try:
        files = download_duconv()
        if files:
            if process_duconv_to_chat_corpus(files, "data/chat_corpus_duconv.txt"):
                success = True
                print("✅ DuConv 数据集下载并处理成功！")
                print()
    except Exception as e:
        print(f"❌ DuConv 下载失败: {e}")
        print()
    
    # 方案2: 小黄鸡数据
    if not success:
        print("【方案2】尝试下载小黄鸡对话数据...")
        print()
        try:
            if download_xiaohuangji():
                success = True
                print("✅ 小黄鸡数据集下载并处理成功！")
                print()
        except Exception as e:
            print(f"❌ 小黄鸡数据下载失败: {e}")
            print()
    
    print("=" * 60)
    print("📊 下载结果")
    print("=" * 60)
    print()
    
    # 列出所有可用的数据文件
    data_files = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith('.txt') and 'chat_corpus' in file:
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                data_files.append((filepath, size, lines))
    
    if data_files:
        print("可用的训练数据：")
        print()
        for filepath, size, lines in sorted(data_files, key=lambda x: x[2], reverse=True):
            print(f"  📄 {filepath}")
            print(f"     行数: {lines:,}")
            print(f"     大小: {size / 1024:.2f} KB")
            print()
        
        # 推荐使用最大的数据集
        largest = max(data_files, key=lambda x: x[2])
        print(f"💡 推荐使用: {largest[0]} ({largest[2]:,} 行)")
        print()
        print("开始训练：")
        print(f"  python scripts/train_improved.py --corpus {largest[0]}")
        print()
    else:
        print("未找到可用的训练数据。")
        print()
        print("请尝试以下手动下载方式：")
        print("  1. GitHub: git clone https://github.com/codemayq/chinese_chatbot_corpus")
        print("  2. 或使用已生成的数据: data/chat_corpus_expanded.txt")
        print()
