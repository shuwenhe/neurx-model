"""
NeurX ChatModel 推理脚本

用于加载已训练的模型并进行文本生成推理
"""

import argparse
import pickle
import numpy as np

import neurx
import neurx.nn as nn

from app.core.models_neurx import (
    create_chatmodel_tiny,
    create_chatmodel_small,
    create_chatmodel_base,
    create_chatmodel_large,
)


class ChatModelInference:
    """模型推理类"""
    
    def __init__(self, model_path=None, model_size='tiny', vocab_size=None):
        """
        初始化推理器
        
        Args:
            model_path: 预训练模型路径（.pkl 文件）
            model_size: 模型大小 (tiny, small, base, large)
            vocab_size: 词汇表大小（如果不加载检查点）
        """
        self.model_size = model_size
        self.device = 'cuda' if neurx.cuda.is_available() else 'cpu'
        
        if model_path:
            self._load_from_checkpoint(model_path)
        else:
            if vocab_size is None:
                raise ValueError("vocab_size required when not loading checkpoint")
            self._create_fresh_model(vocab_size)
    
    def _load_from_checkpoint(self, model_path):
        """从检查点加载模型"""
        print(f"Loading checkpoint from {model_path}...")
        
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.vocab_size = checkpoint['vocab_size']
        self.char_to_id = checkpoint['char_to_id']
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # 创建模型
        model_creators = {
            'tiny': create_chatmodel_tiny,
            'small': create_chatmodel_small,
            'base': create_chatmodel_base,
            'large': create_chatmodel_large,
        }
        
        model_size = checkpoint.get('config', {}).get('model_size', self.model_size)
        self.model = model_creators[model_size](self.vocab_size)
        
        # 加载参数
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Model size: {model_size}")
    
    def _create_fresh_model(self, vocab_size):
        """创建新模型"""
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}
        
        model_creators = {
            'tiny': create_chatmodel_tiny,
            'small': create_chatmodel_small,
            'base': create_chatmodel_base,
            'large': create_chatmodel_large,
        }
        
        self.model = model_creators[self.model_size](vocab_size)
        self.model.eval()
        
        print(f"Created fresh {self.model_size} model")
    
    def encode_text(self, text):
        """将文本编码为 token IDs
        
        Args:
            text: 输入文本字符串
            
        Returns:
            Token IDs 数组
        """
        token_ids = []
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # 未知字符映射到特殊 token
                token_ids.append(0)
        return neurx.array(token_ids, dtype='int64')
    
    def decode_tokens(self, token_ids):
        """将 token IDs 解码为文本
        
        Args:
            token_ids: Token IDs 数组或列表
            
        Returns:
            解码后的文本字符串
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        text = ''
        for token_id in token_ids:
            if isinstance(token_id, (float, np.floating)):
                token_id = int(token_id)
            
            if token_id in self.id_to_char:
                text += self.id_to_char[token_id]
        
        return text
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=None, top_p=0.9):
        """生成文本
        
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 采样温度（越高越随机）
            top_k: Top-k 采样的 k 值
            top_p: Top-p (nucleus) 采样的 p 值
            
        Returns:
            生成的文本
        """
        # 编码 prompt
        token_ids = self.encode_text(prompt)
        
        print(f"\n生成过程:")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        with neurx.no_grad():
            for i in range(max_length):
                # 获取当前序列的最后 1024 个 token（模型的 max_seq_len）
                current_tokens = token_ids[-1024:] if len(token_ids) > 1024 else token_ids
                
                # 前向传播
                input_ids = current_tokens.unsqueeze(0)  # (1, T)
                output = self.model(input_ids)
                logits = output['logits']  # (1, T, V)
                
                # 获取最后一个 token 的 logits
                next_logits = logits[0, -1, :]  # (V,)
                
                # 应用温度
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                
                # 计算概率
                probs = neurx.softmax(next_logits, dim=-1)
                
                # Top-k 采样
                if top_k is not None:
                    top_k_probs, top_k_indices = neurx.topk(probs, k=top_k)
                    next_token_id = neurx.multinomial(top_k_probs, 1).item()
                    next_token_id = top_k_indices[next_token_id].item()
                
                # Top-p 采样
                elif top_p < 1.0:
                    sorted_probs, sorted_indices = neurx.sort(probs, descending=True)
                    cumsum_probs = neurx.cumsum(sorted_probs, dim=0)
                    
                    # 找到 cumulative probability > top_p 的位置
                    mask = cumsum_probs <= top_p
                    valid_probs = sorted_probs[mask]
                    valid_indices = sorted_indices[mask]
                    
                    # 正则化概率
                    valid_probs = valid_probs / valid_probs.sum()
                    
                    next_token_id = neurx.multinomial(valid_probs, 1).item()
                    next_token_id = valid_indices[next_token_id].item()
                
                # 贪心采样（概率最高的 token）
                else:
                    next_token_id = probs.argmax(dim=-1).item()
                
                # 添加到序列
                token_ids = neurx.cat([token_ids, neurx.array([next_token_id], dtype='int64')])
                
                # 打印生成进度
                next_char = self.id_to_char.get(next_token_id, '?')
                print(next_char, end='', flush=True)
        
        print("\n" + "-" * 50 + "\n")
        
        # 解码完整输出
        full_text = self.decode_tokens(token_ids)
        return full_text
    
    def predict_next_tokens(self, prompt, num_predictions=5):
        """预测下一个 token 的最可能的几个选择
        
        Args:
            prompt: 输入提示
            num_predictions: 预测数量
            
        Returns:
            列表，包含 (token, probability) 元组
        """
        token_ids = self.encode_text(prompt)
        
        with neurx.no_grad():
            input_ids = token_ids.unsqueeze(0)
            output = self.model(input_ids)
            logits = output['logits'][0, -1, :]  # 最后一个位置的 logits
            
            probs = neurx.softmax(logits, dim=-1)
            
            # 获取 top-k
            if hasattr(neurx, 'topk'):
                top_probs, top_indices = neurx.topk(probs, k=num_predictions)
            else:
                # 备选方案
                top_indices = neurx.argsort(probs, descending=True)[:num_predictions]
                top_probs = probs[top_indices]
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                char = self.id_to_char.get(idx.item() if hasattr(idx, 'item') else idx, '?')
                results.append((char, prob.item() if hasattr(prob, 'item') else float(prob)))
            
            return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Inference with NeurX ChatModel")
    
    parser.add_argument('--model-path', type=str, default=None, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-size', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size (used if no checkpoint)')
    parser.add_argument('--vocab-size', type=int, default=None,
                       help='Vocabulary size (used if no checkpoint)')
    
    # 生成参数
    parser.add_argument('--prompt', type=str, default='人工智能',
                       help='Input prompt for generation')
    parser.add_argument('--max-length', type=int, default=50,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling parameter')
    
    # 推理模式
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'predict'],
                       help='Inference mode')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # 初始化推理器
    print("=" * 70)
    print("NeurX ChatModel - Inference Script")
    print("=" * 70 + "\n")
    
    if args.model_path:
        inference = ChatModelInference(model_path=args.model_path)
    else:
        if args.vocab_size is None:
            args.vocab_size = 100  # 默认词汇表大小
        inference = ChatModelInference(
            model_size=args.model_size,
            vocab_size=args.vocab_size
        )
    
    # 单次推理
    if args.mode == 'generate':
        result = inference.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"Final output:\n{result}\n")
    
    elif args.mode == 'predict':
        results = inference.predict_next_tokens(args.prompt, num_predictions=5)
        print(f"Top-5 next tokens after '{args.prompt}':")
        for char, prob in results:
            print(f"  '{char}': {prob:.4f}")
        print()
    
    # 交互式推理
    if args.interactive:
        print("\n进入交互模式 (输入 'quit' 退出):\n")
        
        while True:
            prompt = input("输入提示文本: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not prompt:
                continue
            
            print()
            result = inference.generate(
                prompt,
                max_length=50,
                temperature=0.8
            )


if __name__ == "__main__":
    main()
