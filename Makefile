.PHONY: help install test step1 step2 step3 bootstrap-check train train-s-dataset serve serve-dev serve-core serve-core-dev obs-up obs-down generate quick-generate quick-test-multimodal demo gateway inference-generate inference-quick deploy-local-up deploy-local-down deploy-model deploy-model-s-dataset precision-test clean clean-checkpoints clean-all frontend-install frontend-dev frontend-build frontend-start kill-frontend kill-backend dev-all install-systemd-nginx restart-services status-services logs logs-follow

.DEFAULT_GOAL := train

# 核心模型 checkpoint (所有模态共用)
CORE_MODEL_CHECKPOINT ?= checkpoints/model_core.pkl

# S runtime 参数
S_COMPILER ?= /usr/local/bin/s
S_RUNTIME_MODE ?= compile
S_SOURCE_ROOT ?= /app/neurx/s
S_MODEL_SOURCE_ROOT ?= /app/neurx-model/s
S_IR_DIR ?= reports/s_ir

# 默认使用仓库内已清洗训练集作为纯 S 数据集输入
DATASET_FILE ?= dataset/text/Neurx-SFT-Text-v1.finalclean.train.jsonl

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
	@echo "  make train            - 纯S训练/部署默认入口"
	@echo "  make train-s-dataset DATASET_FILE=... - 纯S链路绑定真实数据集导出"
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
	@echo "  make deploy-model     - 重新编译最新模型并重启后端加载"
	@echo "  make deploy-model-s-dataset DATASET_FILE=... - 纯S链路使用真实数据集并上线"
	@echo "  make logs             - 查看后端日志(文件+systemd+journal)"
	@echo "  make logs-follow      - 实时跟踪后端文件日志"
	@echo "  make deploy-local-up  - 启动本地标准部署编排(deploy/local)"
	@echo "  make deploy-local-down - 停止本地标准部署编排(deploy/local)"
	@echo ""
	@echo "${YELLOW}工具与测试:${NC}"
	@echo "  make generate         - 运行交互式文本生成"
	@echo "  make quick-generate   - 批量测试生成参数"
	@echo "  make precision-test   - 运行精准度回归测试(lookup + api flow + structured logs)"
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

train-s-dataset:
	@if [ -z "$(DATASET_FILE)" ]; then \
		echo "❌ 请指定 DATASET_FILE，例如: make train-s-dataset DATASET_FILE=dataset/text/train.txt"; \
		exit 1; \
	fi
	@echo "使用纯S链路导出并绑定数据集: $(DATASET_FILE)"
	@DATASET_FILE="$(DATASET_FILE)" bash scripts/s_only_train_bundle.sh --dataset-file "$(DATASET_FILE)"

train: train-s-dataset

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

# 启动前端(生产模式)并绑定端口8080，用于外部访问 /neurx
frontend-start-8080:
	@echo "启动前端(生产模式 - 端口8080, basePath=/neurx)..."
	cd frontend && HOST=0.0.0.0 npm run start -- -p 8080

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

precision-test:
	@echo "运行精准度回归测试..."
	./venv/bin/python -m pytest -q test/test_precision_lookup.py test/test_api_precision_flow.py

deploy-local-up:
	@echo "启动本地标准部署编排..."
	docker compose -f deploy/local/docker-compose.yml up -d --build

deploy-local-down:
	@echo "停止本地标准部署编排..."
	docker compose -f deploy/local/docker-compose.yml down

deploy-model:
	@echo "重新编译最新模型并部署到后端..."
	$(MAKE) train-s-dataset DATASET_FILE="$(DATASET_FILE)"
	@echo "重启后端服务以加载 checkpoints/s_arch_latest.json ..."
	systemctl restart neurx-model-backend.service
	@echo "等待后端服务就绪..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -fsS http://127.0.0.1:8000/v1/model-status >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@echo "当前后端模型状态:"
	@curl -fsS http://127.0.0.1:8000/v1/model-status

deploy-model-s-dataset:
	@if [ -z "$(DATASET_FILE)" ]; then \
		echo "❌ 请指定 DATASET_FILE，例如: make deploy-model-s-dataset DATASET_FILE=dataset/text/train.txt"; \
		exit 1; \
	fi
	@echo "使用纯S链路+真实数据集重新导出并部署..."
	$(MAKE) train-s-dataset DATASET_FILE="$(DATASET_FILE)"
	@echo "重启后端服务以加载 checkpoints/s_arch_latest.json ..."
	systemctl restart neurx-model-backend.service
	@echo "等待后端服务就绪..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -fsS http://127.0.0.1:8000/v1/model-status >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@echo "当前后端模型状态:"
	@curl -fsS http://127.0.0.1:8000/v1/model-status

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

# 安装 systemd + nginx（需 root）
install-systemd-nginx:
	@echo "安装 systemd 和 nginx 反向代理配置..."
	bash scripts/install_systemd_nginx.sh
	@echo "${GREEN}✓ 部署配置完成${NC}"

# 重启 systemd 服务
restart-services:
	@echo "重启后端和前端服务..."
	systemctl restart neurx-model-backend.service neurx-model-frontend.service
	@echo "${GREEN}✓ 服务已重启${NC}"

# 查看服务状态
status-services:
	@echo "后端服务状态:"
	@systemctl --no-pager --lines=30 status neurx-model-backend.service || true
	@echo ""
	@echo "前端服务状态:"
	@systemctl --no-pager --lines=30 status neurx-model-frontend.service || true
	@echo ""
	@echo "Nginx 状态:"
	@systemctl --no-pager --lines=30 status nginx || true

# 查看后端日志（文件 + systemd + journal）
logs:
	@echo "== backend.out (last 120 lines) =="
	@if [ -f logs/backend.out ]; then \
		tail -n 120 logs/backend.out; \
	else \
		echo "logs/backend.out 不存在"; \
	fi
	@echo ""
	@echo "== backend recent request lines (from file) =="
	@if [ -f logs/backend.out ]; then \
		grep -E '"(GET|POST|PUT|DELETE) /(v1|healthz|readyz|metrics)' logs/backend.out | tail -n 60 || true; \
	else \
		echo "logs/backend.out 不存在"; \
	fi
	@echo ""
	@echo "== systemd status (neurx-model-backend.service) =="
	@if command -v systemctl >/dev/null 2>&1; then \
		systemctl --no-pager --lines=40 status neurx-model-backend.service || true; \
	else \
		echo "systemctl 不可用"; \
	fi
	@echo ""
	@echo "== journal (last 80 lines) =="
	@if command -v journalctl >/dev/null 2>&1; then \
		journalctl -u neurx-model-backend.service -n 80 --no-pager || true; \
	else \
		echo "journalctl 不可用"; \
	fi
	@echo ""
	@echo "== backend recent request lines (from journal) =="
	@if command -v journalctl >/dev/null 2>&1; then \
		journalctl -u neurx-model-backend.service -n 300 --no-pager | grep -E '"(GET|POST|PUT|DELETE) /(v1|healthz|readyz|metrics)' | tail -n 60 || true; \
	else \
		echo "journalctl 不可用"; \
	fi
	@echo ""
	@echo "== nginx access (frontend + api via :8080, last 80 lines) =="
	@if [ -f /var/log/nginx/access.log ]; then \
		tail -n 80 /var/log/nginx/access.log; \
	else \
		echo "/var/log/nginx/access.log 不存在或不可读"; \
	fi
	@echo ""
	@echo "== nginx error (last 40 lines) =="
	@if [ -f /var/log/nginx/error.log ]; then \
		tail -n 40 /var/log/nginx/error.log; \
	else \
		echo "/var/log/nginx/error.log 不存在或不可读"; \
	fi

# 实时跟踪后端文件日志
logs-follow:
	@echo "等待前端请求中（Ctrl+C 退出）..."
	@if command -v journalctl >/dev/null 2>&1; then \
		journalctl -u neurx-model-backend.service -f --no-pager | grep --line-buffered -E '"(GET|POST|PUT|DELETE) /'; \
	elif [ -f logs/backend.out ]; then \
		tail -f logs/backend.out | grep --line-buffered -E '"(GET|POST|PUT|DELETE) /'; \
	else \
		echo "journalctl 不可用且 logs/backend.out 不存在"; \
		exit 1; \
	fi
