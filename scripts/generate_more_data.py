#!/usr/bin/env python
"""数据扩充脚本 - 生成更多训练对话数据"""

import random
import os

# 扩展的对话模板和变体
templates = {
    "greeting": [
        ("你好", "你好！我是ChatNeurX，很高兴认识你。"),
        ("早上好", "早上好！希望你有美好的一天。"),
        ("晚上好", "晚上好！有什么我可以帮助你的吗？"),
        ("嗨", "嗨！我在这里随时为你服务。"),
        ("您好", "您好！我是AI助手ChatNeurX，请问有什么可以帮你的？"),
    ],
    
    "capability": [
        ("你能做什么", "我可以回答问题、提供建议、进行对话、帮助学习等。"),
        ("你有什么功能", "我具备对话交流、信息查询、问题解答、创意协作等功能。"),
        ("你可以帮我什么", "我可以帮你解答疑问、提供信息、进行头脑风暴等。"),
        ("你的能力有哪些", "我能够理解中文、生成文本、回答各类问题。"),
        ("你会做什么", "我会对话聊天、回答问题、提供帮助和建议。"),
    ],
    
    "identity": [
        ("你是谁", "我是ChatNeurX，一个基于神经网络的AI对话助手。"),
        ("介绍下你自己", "我是ChatNeurX，使用自研NeurX框架构建的语言模型。"),
        ("你叫什么", "我叫ChatNeurX，一个智能对话系统。"),
        ("你的名字", "我的名字是ChatNeurX。"),
        ("告诉我你是什么", "我是一个人工智能助手，专注于中文对话。"),
    ],
    
    "knowledge": [
        ("什么是AI", "AI是人工智能（Artificial Intelligence）的缩写，是计算机科学的一个分支。"),
        ("什么是机器学习", "机器学习是AI的子领域，让计算机通过数据学习而非显式编程。"),
        ("什么是深度学习", "深度学习是机器学习的一种，使用多层神经网络处理复杂数据。"),
        ("什么是神经网络", "神经网络是受大脑启发的计算模型，由相互连接的神经元组成。"),
        ("解释一下Transformer", "Transformer是一种基于注意力机制的神经网络架构，广泛用于NLP任务。"),
    ],
    
    "help": [
        ("帮帮我", "当然可以！请告诉我你需要什么帮助。"),
        ("我需要帮助", "我在这里！请说说你遇到的问题。"),
        ("你能帮我吗", "当然可以帮你，请详细说明你的需求。"),
        ("求助", "别担心，我会尽力帮助你解决问题。"),
        ("我有问题", "请说出你的问题，我会认真回答。"),
    ],
    
    "thanks": [
        ("谢谢", "不客气！很高兴能帮到你。"),
        ("感谢你", "不用谢！这是我的荣幸。"),
        ("多谢", "不用客气，有其他问题随时问我。"),
        ("太感谢了", "能帮到你我很开心！"),
        ("谢谢你的帮助", "不客气！期待下次为你服务。"),
    ],
    
    "technical": [
        ("Python是什么", "Python是一种高级编程语言，以简洁易读著称，广泛用于Web开发、数据科学、AI等领域。"),
        ("如何学习编程", "建议从基础语法开始，多写代码实践，做项目巩固，参与开源社区学习。"),
        ("什么是API", "API是应用程序接口，允许不同软件之间相互通信和交换数据。"),
        ("数据库是什么", "数据库是有组织地存储和管理数据的系统，如MySQL、PostgreSQL等。"),
        ("什么是卷积神经网络", "CNN是专门处理图像的神经网络，通过卷积层提取空间特征。"),
    ],
    
    "conversation": [
        ("今天天气怎么样", "抱歉，我无法获取实时天气信息，建议你查看天气应用。"),
        ("你喜欢什么", "作为AI，我对学习新知识和帮助他人充满兴趣。"),
        ("你会累吗", "我不会疲劳，可以7×24小时为你服务。"),
        ("你有感情吗", "我可以理解和模拟情感，但我不具备真正的人类情感。"),
        ("聊聊天吧", "好啊！你想聊什么话题呢？"),
    ],
    
    "math": [
        ("1加1等于多少", "1加1等于2。"),
        ("什么是质数", "质数是只能被1和自身整除的大于1的自然数，如2、3、5、7等。"),
        ("圆的面积公式", "圆的面积公式是 A = πr²，其中r是半径。"),
        ("什么是微积分", "微积分是研究变化的数学分支，包括微分（研究变化率）和积分（研究累积）。"),
        ("解释概率", "概率是衡量事件发生可能性的数值，范围在0到1之间。"),
    ],
    
    "life_advice": [
        ("如何保持健康", "建议均衡饮食、规律运动、充足睡眠、保持良好心态。"),
        ("怎样提高效率", "可以使用番茄工作法、制定任务清单、减少干扰、保持专注。"),
        ("如何学习新技能", "设定明确目标、制定学习计划、持续练习、寻求反馈、保持耐心。"),
        ("怎么缓解压力", "尝试深呼吸、适度运动、与朋友交流、培养兴趣爱好、保证休息。"),
        ("时间管理建议", "优先处理重要任务、避免拖延、合理分配时间、定期回顾调整。"),
    ],
}

# 更多变体词汇
synonyms = {
    "你好": ["你好", "您好", "嗨", "hello", "hi"],
    "谢谢": ["谢谢", "感谢", "多谢", "太感谢了"],
    "是": ["是", "对", "没错", "确实", "是的"],
    "不": ["不", "不是", "并非", "并不", "否"],
}

def augment_data(original_corpus, output_file, target_lines=10000):
    """数据增强：生成更多对话数据"""
    
    print(f"🚀 开始数据增强，目标行数: {target_lines}")
    
    all_conversations = []
    
    # 1. 保留原始数据
    if os.path.exists(original_corpus):
        with open(original_corpus, 'r', encoding='utf-8') as f:
            original_lines = [line.strip() for line in f if line.strip()]
            all_conversations.extend(original_lines)
            print(f"✅ 加载原始数据: {len(original_lines)} 条")
    
    # 2. 从模板生成新数据
    template_conversations = []
    for category, pairs in templates.items():
        for q, a in pairs:
            template_conversations.append(q)
            template_conversations.append(a)
    
    print(f"✅ 生成模板数据: {len(template_conversations)} 条")
    all_conversations.extend(template_conversations)
    
    # 3. 数据增强：变体生成
    augmented = []
    for conv in all_conversations[:]:  # 复制避免修改原列表
        # 添加原始版本
        augmented.append(conv)
        
        # 生成变体（简单的同义词替换示例）
        if random.random() < 0.3:  # 30%概率生成变体
            modified = conv
            for word, syns in synonyms.items():
                if word in modified:
                    modified = modified.replace(word, random.choice(syns))
            if modified != conv:
                augmented.append(modified)
    
    all_conversations = augmented
    print(f"✅ 增强后数据量: {len(all_conversations)} 条")
    
    # 4. 如果还不够，重复并打乱
    while len(all_conversations) < target_lines:
        all_conversations.extend(all_conversations[:target_lines - len(all_conversations)])
    
    # 打乱顺序
    random.shuffle(all_conversations)
    all_conversations = all_conversations[:target_lines]
    
    # 5. 保存
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_conversations:
            f.write(line + '\n')
    
    print(f"✅ 数据增强完成！保存到: {output_file}")
    print(f"   总行数: {len(all_conversations)}")
    print(f"   建议：进一步从开源数据集（LCCC等）获取更多真实对话数据")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据增强脚本")
    parser.add_argument("--original", type=str, default="data/chat_corpus.txt", help="原始语料文件")
    parser.add_argument("--output", type=str, default="data/chat_corpus_expanded.txt", help="输出文件")
    parser.add_argument("--target-lines", type=int, default=10000, help="目标行数")
    
    args = parser.parse_args()
    
    augment_data(args.original, args.output, args.target_lines)
