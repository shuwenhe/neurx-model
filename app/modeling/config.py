"""模型和训练配置"""

class ModelConfig:
    """GPT模型配置"""
    # 模型架构
    vocab_size = 50257  # GPT-2 tokenizer的词表大小
    n_layer = 12  # Transformer层数
    n_head = 12   # 注意力头数
    n_embd = 768  # 嵌入维度
    
    # 序列和训练
    block_size = 512  # 最大序列长度
    dropout = 0.1
    bias = True  # 是否在Linear和LayerNorm中使用bias

    # MoE（Mixture of Experts）
    moe_enabled = False
    moe_num_experts = 4
    moe_top_k = 2
    moe_hidden_dim = None  # None -> 4 * n_embd

    # 架构改进
    rmsnorm_enabled = False
    rmsnorm_bias = False
    swiglu_enabled = False
    rope_enabled = False
    rope_theta = 10000.0

    # 多模态（可选）
    multimodal_enabled = False
    modality_dropout = 0.0

    # 视觉编码器配置
    vision_input_channels = 3
    vision_patch_size = 16

    # 语音编码器配置（输入通常是mel特征）
    audio_input_dim = 80
    audio_kernel_size = 3
    audio_stride = 2
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainConfig:
    """训练配置"""
    # 数据
    dataset_name = "wikitext"  # 或使用自己的数据
    dataset_config = "wikitext-2-raw-v1"
    clean_data = True  # 是否清洗WikiText格式标记
    train_multimodal = False  # 是否启用完整多模态训练（文本+图像+语音）
    multimodal_image_size = 64
    multimodal_audio_len = 50
    multimodal_batch_size = 4
    multimodal_block_size = 256
    
    # 训练参数
    batch_size = 16
    max_iters = 20000
    eval_interval = 500
    eval_iters = 100
    log_interval = 10
    
    # 优化器
    learning_rate = 3e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    grad_norm_warn = 5.0      # 梯度范数告警阈值
    grad_norm_warn_start = 100  # 从第几步开始检查告警（默认warmup后）
    grad_norm_warn_interval = 50  # 告警最小间隔步数（防止刷屏）
    
    # 学习率调度
    warmup_iters = 100
    lr_decay_iters = 20000
    min_lr = 3e-5
    
    # 系统
    device = "cuda"  # cuda, mps, cpu
    compile = False  # 自研后端不使用框架编译选项
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 1000
    log_to_wandb = False
    wandb_project = "my-llm"
    wandb_run_name = "gpt-small"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
