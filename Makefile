.PHONY: help install test step1 step2 step3 bootstrap-check train train-basic train-core train-multimodal train-chinese serve serve-dev serve-core serve-core-dev obs-up obs-down generate quick-generate quick-test-multimodal demo gateway inference-generate inference-quick deploy-local-up deploy-local-down clean clean-checkpoints clean-all frontend-install frontend-dev frontend-build frontend-start kill-frontend kill-backend dev-all

# Python解释器（优先使用项目内虚拟环境）
PYTHON := $(shell if [ -x ./venv/bin/python ]; then echo ./venv/bin/python; else echo python3; fi)

# Python路径（以app为唯一源码根）
export PYTHONPATH:=.

# 核心模型 checkpoint (所有模态共用)
CORE_MODEL_CHECKPOINT ?= checkpoints/model_core.pkl

# 视觉训练参数（可在命令行覆盖）
VISION_DATA_SOURCE ?= cifar10
VISION_BATCH_SIZE ?= 8
VISION_EPOCHS ?= 1
VISION_MAX_STEPS ?= 0
VISION_LR ?= 1e-4
VISION_DATASET_NAME ?= nlphuji/flickr30k
VISION_CHECKPOINT ?= $(CORE_MODEL_CHECKPOINT)
VISION_OUTPUT ?= $(CORE_MODEL_CHECKPOINT)

# 中文文本训练参数（可在命令行覆盖）
CHINESE_DATA_SOURCE ?= wikitext_zh
CHINESE_BATCH_SIZE ?= 4
CHINESE_EPOCHS ?= 3
CHINESE_MAX_STEPS ?= 0
CHINESE_LR ?= 1e-4
CHINESE_CHECKPOINT ?= $(CORE_MODEL_CHECKPOINT)
CHINESE_OUTPUT ?= $(CORE_MODEL_CHECKPOINT)

# 颜色输出
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# 默认目标
help:
	@echo ""
	@echo "${GREEN}LLM项目 - 可用命令:${NC}"
	@echo ""
	@echo "${YELLOW}环境设置:${NC}"
	@echo "  make setup            - 创建虚拟环境"
	@echo "  make setup-all        - 创建虚拟环境并安装依赖(推荐)"
	@echo "  make install          - 安装依赖(需要先激活虚拟环境)"
	@echo "  make install-force    - 强制安装(不推荐，跳过虚拟环境检查)"
	@echo ""
	@echo "${YELLOW}开发与训练:${NC}"
	@echo "  make test             - 运行模型测试"
	@echo "  make step1            - 第一步: 仅验证模型前向传播"
	@echo "  make step2            - 第二步: 验证单步反向传播"
	@echo "  make step3            - 第三步: 迷你训练10步验证"
	@echo "  make bootstrap-check  - 一次跑完 step1/step2/step3"
	@echo "  make train            - 开始训练模型(默认:中文文本)"
	@echo "  make train-core       - 使用自研后端训练"
	@echo "  make train-chinese    - 训练中文文本能力"
	@echo "  make train-multimodal - 完整多模态训练"
	@echo "  make train-basic      - 基础文本训练"
	@echo "  make serve            - 启动推理API服务(使用统一模型)"
	@echo "  make serve-dev        - 启动推理API服务(开发热更新)"
	@echo "  make serve-core       - 启动自研后端API服务"
	@echo "  make serve-core-dev   - 启动自研后端API服务(开发热更新)"
	@echo ""
	@echo "${YELLOW}前端开发 (Next.js):${NC}"
	@echo "  make frontend-install - 安装前端依赖"
	@echo "  make frontend-dev     - 启动前端(开发模式) - ${RED}Ctrl+C可正确关闭${NC}"
	@echo "  make frontend-build   - 构建前端(生产构建)"
	@echo "  make frontend-start   - 启动前端(生产模式)"
	@echo ""
	@echo "${YELLOW}前端/后端合并操作:${NC}"
	@echo "  make dev-all          - 同时启动后端+前端(开发模式,需要2个终端)"
	@echo "  make kill-frontend    - 关闭前端(端口3000)"
	@echo "  make kill-backend     - 关闭后端(端口8000)"
	@echo ""
	@echo "${YELLOW}可观测性:${NC}"
	@echo "  make obs-up           - 启动可观测性栈(LLM+Prometheus+Grafana)"
	@echo "  make obs-down         - 停止可观测性栈"
	@echo "  make deploy-local-up  - 启动本地标准部署编排(deploy/local)"
	@echo "  make deploy-local-down - 停止本地标准部署编排(deploy/local)"
	@echo ""
	@echo "${YELLOW}工具与测试:${NC}"
	@echo "  make generate         - 运行交互式文本生成"
	@echo "  make quick-generate   - 批量测试生成参数"
	@echo "  make gateway          - 启动网关服务(services/gateway)"
	@echo "  make inference-generate - 通过services边界运行生成"
	@echo "  make inference-quick  - 通过services边界运行快速生成"
	@echo "  make demo             - 创建演示模型(无需训练，快速测试)"
	@echo "  make quick-test       - 快速测试(验证模型可用)"
	@echo "  make quick-test-multimodal - 快速测试多模态前向"
	@echo "  make info             - 查看模型配置信息"
	@echo "  make check-deps       - 检查依赖安装情况"
	@echo "  make init             - 创建必要的项目目录"
	@echo ""
	@echo "${YELLOW}清理:${NC}"
	@echo "  make clean            - 清理Python缓存文件"
	@echo "  make clean-checkpoints - 删除所有checkpoint文件"
	@echo "  make clean-all        - 清理所有生成文件"
	@echo ""

# 检查是否在虚拟环境中
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ 错误：未检测到虚拟环境！"; \
		echo ""; \
		echo "请先创建并激活虚拟环境："; \
		echo "  方式1: make setup && source venv/bin/activate"; \
		echo "  方式2: python3 -m venv venv && source venv/bin/activate"; \
		echo ""; \
		echo "然后再运行: make install"; \
		exit 1; \
	fi

# 安装依赖（需要在虚拟环境中）
install: check-venv
	@echo "安装项目依赖..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "${GREEN}✓ 依赖安装完成${NC}"

# 强制安装（不检查虚拟环境，不推荐）
install-force:
	@echo "${YELLOW}⚠️  强制安装依赖（不推荐，可能影响系统Python）...${NC}"
	$(PYTHON) -m pip install --upgrade pip --break-system-packages
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages
	@echo "${GREEN}✓ 依赖安装完成${NC}"

# 创建虚拟环境并安装依赖
setup:
	@echo "创建虚拟环境..."
	$(PYTHON) -m venv venv
	@echo "${GREEN}✓ 虚拟环境创建完成: venv/${NC}"
	@echo ""
	@echo "激活虚拟环境："
	@echo "  Linux/Mac: source venv/bin/activate"
	@echo "  Windows:   venv\\Scripts\\activate"
	@echo ""
	@echo "然后运行: make install"

# 一键安装（创建venv并安装依赖）
setup-all:
	@echo "创建虚拟环境并安装依赖..."
	@if [ ! -d "venv" ]; then \
		echo "创建虚拟环境..."; \
		$(PYTHON) -m venv venv; \
	fi
	@echo "安装依赖到虚拟环境..."
	@./venv/bin/pip install --upgrade pip
	@./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "${GREEN}✓ 设置完成！${NC}"
	@echo ""
	@echo "激活虚拟环境："
	@echo "  source venv/bin/activate"
	@echo ""
	@echo "然后可以运行："
	@echo "  make test    # 测试模型"
	@echo "  make train   # 训练模型"

# 运行模型测试
test:
	@echo "运行基础验收链路..."
	$(MAKE) step1
	$(MAKE) step2
	$(PYTHON) test/regression_batched_matmul_grad.py
	$(MAKE) step3
	$(PYTHON) test/test_model.py
	@echo "✓ 基础验收链路全部通过"

# 第一步：模型前向传播最小验证
step1:
	@echo "第一步：验证模型前向传播..."
	$(PYTHON) test/step1_forward.py

# 第二步：单步反向传播最小验证
step2:
	@echo "第二步：验证单步反向传播..."
	$(PYTHON) test/step2_backward.py

# 第三步：迷你训练10步验证
step3:
	@echo "第三步：迷你训练10步验证..."
	$(PYTHON) test/step3_mini_train.py

# 串行执行 Step1/Step2/Step3 启动验证
bootstrap-check:
	@echo "启动引导检查: step1 -> step2 -> step3"
	$(MAKE) step1
	$(MAKE) step2
	$(MAKE) step3
	@echo "✓ bootstrap-check 全部通过"

# 训练模型（默认：中文文本训练）
train:
	@echo "开始训练模型..."
	@echo "使用中文文本训练（推荐）"
	@echo "其他选项: make train-multimodal, make train-basic"
	@echo ""
	$(MAKE) train-chinese

# 基础文本训练（原始训练脚本）
train-basic:
	@echo "开始基础文本训练..."
	$(PYTHON) -m app.training.train

train-core:
	@echo "开始自研后端训练..."
	$(PYTHON) -m app.training.train_core

# 多模态训练
train-multimodal:
	@echo "开始多模态训练模型..."
	LLM_MULTIMODAL=1 $(PYTHON) -m app.training.train

# 中文文本训练（支持参数覆盖）
# 示例:
#   make train-chinese
#   make train-chinese CHINESE_DATA_SOURCE=zhwiki CHINESE_EPOCHS=5
#   make train-chinese CHINESE_DATA_SOURCE=baidubaike CHINESE_BATCH_SIZE=16
train-chinese:
	@echo "开始训练中文文本能力..."
	@echo "batch_size=$(CHINESE_BATCH_SIZE), epochs=$(CHINESE_EPOCHS), lr=$(CHINESE_LR)"
	$(PYTHON) -m app.training.train_simple_neurx \
		--batch-size $(CHINESE_BATCH_SIZE) \
		--epochs $(CHINESE_EPOCHS) \
		--learning-rate $(CHINESE_LR) \
		--checkpoint $(CHINESE_CHECKPOINT) \
		--output $(CHINESE_OUTPUT)

# NeurX 框架训练（推荐）
train-neurx:
	@echo "开始使用 NeurX 框架训练..."
	@echo "batch_size=4, epochs=3, lr=1e-4"
	$(PYTHON) -m app.training.train_simple_neurx \
		--batch-size 4 \
		--epochs 3 \
		--learning-rate 1e-4 \
		--hidden-dim 256 \
		--num-layers 2

# 推理服务（开发）
serve-dev:
	@echo "启动推理API服务(开发模式)..."
	$(PYTHON) -m uvicorn app.api.serve:app --host 0.0.0.0 --port 8000 --reload

serve-core-dev:
	@echo "启动自研后端API服务(开发模式)..."
	$(PYTHON) -m uvicorn app.api.serve_core:app --host 0.0.0.0 --port 8000 --reload

gateway:
	@echo "启动网关服务(服务边界入口)..."
	$(PYTHON) -m uvicorn services.gateway.main:app --host 0.0.0.0 --port 8000 --reload

# 推理服务（统一模型）
serve:
	@echo "启动推理API服务..."
	@echo "使用 core 模型: checkpoints/model_core.pkl"
	LLM_CHECKPOINT=checkpoints/model_core.pkl $(PYTHON) -m uvicorn app.api.serve:app --host 0.0.0.0 --port 8000 --reload

serve-core:
	@echo "启动自研后端API服务..."
	LLM_CHECKPOINT=checkpoints/model_core.pkl $(PYTHON) -m uvicorn app.api.serve_core:app --host 0.0.0.0 --port 8000 --reload

# 前端（Next.js）
frontend-install:
	@echo "安装前端依赖..."
	cd frontend && npm install
	@echo "${GREEN}✓ 前端依赖安装完成${NC}"

frontend-dev:
	@echo "启动前端(开发模式 - 端口3000)..."
	@echo "提示: 按 Ctrl+C 可以正确关闭服务"
	@cd frontend && npm run dev

frontend-build:
	@echo "构建前端(生产构建)..."
	cd frontend && npm run build
	@echo "${GREEN}✓ 前端构建完成${NC}"

frontend-start:
	@echo "启动前端(生产模式 - 端口3000)..."
	cd frontend && npm run start

# 可观测性栈（服务 + Prometheus + Grafana）
obs-up:
	@echo "启动可观测性栈..."
	docker compose -f docker-compose.observability.yml up -d --build
	@echo "${GREEN}✓ 可观测性栈启动完成${NC}"

obs-down:
	@echo "停止可观测性栈..."
	docker compose -f docker-compose.observability.yml down
	@echo "${GREEN}✓ 可观测性栈已停止${NC}"

# 杀死前端进程
kill-frontend:
	@echo "关闭前端 (端口3000)..."
	@if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		kill -9 $$(lsof -t -i :3000) 2>/dev/null || pkill -9 -f "next dev" 2>/dev/null || true; \
		echo "${GREEN}✓ 前端进程已关闭${NC}"; \
	else \
		echo "○ 前端未运行"; \
	fi

# 杀死后端进程
kill-backend:
	@echo "关闭后端 (端口8000)..."
	@if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		kill -9 $$(lsof -t -i :8000) 2>/dev/null || pkill -9 -f "uvicorn" 2>/dev/null || true; \
		echo "${GREEN}✓ 后端进程已关闭${NC}"; \
	else \
		echo "○ 后端未运行"; \
	fi

# 同时启动前后端（用于开发，需要在2个终端中分别运行）
dev-all:
	@echo "${YELLOW}开发模式 - 启动后端+前端${NC}"
	@echo ""
	@echo "需要在2个终端中分别运行:"
	@echo ""
	@echo "  ${GREEN}终端1 (后端):${NC}  make serve-dev"
	@echo "  ${GREEN}终端2 (前端):${NC}  make frontend-dev"
	@echo ""
	@echo "或者如果要在后台运行："
	@echo "  make serve-dev &"
	@echo "  make frontend-dev"
	@echo ""
	@echo "停止服务时使用:"
	@echo "  make kill-backend"
	@echo "  make kill-frontend"
	@echo ""

# 文本生成
generate:
	@echo "启动文本生成..."
	$(PYTHON) -m app.inference.generate

inference-generate:
	@echo "通过services边界启动文本生成..."
	$(PYTHON) -m services.inference.generate

# 快速生成测试（批量测试不同参数）
quick-generate:
	@echo "批量测试生成参数..."
	$(PYTHON) -m app.inference.quick_generate

inference-quick:
	@echo "通过services边界批量测试生成参数..."
	$(PYTHON) -m services.inference.quick_generate

deploy-local-up:
	@echo "启动本地标准部署编排..."
	docker compose -f deploy/local/docker-compose.yml up -d --build

deploy-local-down:
	@echo "停止本地标准部署编排..."
	docker compose -f deploy/local/docker-compose.yml down

# 创建演示模型（用于快速测试，无需训练）
demo:
	@echo "创建演示模型..."
	$(PYTHON) -m app.inference.create_demo_model

# 快速测试（用于验证代码）
quick-test:
	@echo "快速测试模式..."
	$(PYTHON) -c "from app.modeling.model import GPT; from app.modeling.config import ModelConfig; \
		config = ModelConfig(n_layer=2, n_head=2, n_embd=128); \
		model = GPT(config); \
		print(f'模型参数: {model.get_num_params()/1e6:.2f}M'); \
		print('✓ 模型创建成功')"

# 多模态快速测试（随机输入）
quick-test-multimodal:
	@echo "多模态快速测试模式..."
	$(PYTHON) -c "import numpy as np; from app.modeling.model import GPT; from app.modeling.config import ModelConfig; \
		config = ModelConfig(multimodal_enabled=False, n_layer=2, n_head=2, n_embd=128, block_size=64); \
		model = GPT(config); \
		idx = np.random.randint(0, config.vocab_size, (2, 32)); \
		logits, loss = model(idx, idx); \
		print(f'logits形状: {tuple(logits.shape)}'); \
		print(f'loss: {float(loss):.4f}' if loss is not None else 'loss: None'); \
		print('✓ 前向测试成功')"

# 清理Python缓存
clean:
	@echo "清理Python缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "${GREEN}✓ 缓存清理完成${NC}"

# 清理checkpoint文件
clean-checkpoints:
	@echo "删除checkpoint文件..."
	rm -rf checkpoints/*.pt checkpoints/*.pkl
	@echo "${GREEN}✓ Checkpoint清理完成${NC}"

# 清理所有生成文件
clean-all: clean clean-checkpoints
	@echo "清理所有生成文件..."
	rm -rf logs/
	rm -rf wandb/
	rm -rf runs/
	rm -rf data/
	@echo "${GREEN}✓ 完全清理完成${NC}"

# 查看模型信息
info:
	@echo "模型信息:"
	@$(PYTHON) -c "from app.modeling.model import GPT; from app.modeling.config import ModelConfig; \
		config = ModelConfig(); \
		model = GPT(config); \
		print(f'参数量: {model.get_num_params()/1e6:.2f}M'); \
		print(f'层数: {config.n_layer}'); \
		print(f'嵌入维度: {config.n_embd}'); \
		print(f'注意力头数: {config.n_head}'); \
		print(f'序列长度: {config.block_size}')"

# 检查依赖
check-deps:
	@echo "检查依赖安装情况..."
	@$(PYTHON) -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "✗ Transformers未安装"
	@$(PYTHON) -c "import datasets; print(f'✓ Datasets {datasets.__version__}')" || echo "✗ Datasets未安装"

# 创建必要的目录
init:
	@echo "创建项目目录..."
	mkdir -p checkpoints
	mkdir -p logs
	mkdir -p data
	@echo "✓ 目录创建完成"
