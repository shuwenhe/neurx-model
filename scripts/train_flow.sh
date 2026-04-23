#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="./venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

DATASET_FILE="${FLOW_DATASET_FILE:-dataset/text/neurx_train_mix_v1.txt}"
MODEL_SIZE="${FLOW_MODEL_SIZE:-tiny}"
EPOCHS="${FLOW_EPOCHS:-1}"
BATCH_SIZE="${FLOW_BATCH_SIZE:-8}"
SEQ_LEN="${FLOW_SEQ_LEN:-64}"
LR="${FLOW_LR:-1e-4}"
STEPS_PER_EPOCH="${FLOW_STEPS_PER_EPOCH:-50}"
SAVE_PATH="${FLOW_SAVE_PATH:-checkpoints/model_neurx_dataset.pkl}"

if [[ ! -f "$DATASET_FILE" ]]; then
  echo "[train-flow] dataset file not found: $DATASET_FILE" >&2
  exit 1
fi

mkdir -p checkpoints reports
RUN_TS="$(TZ=Asia/Shanghai date +%Y%m%d%H%M%S)"

printf "[train-flow] start run at %s (UTC+8)\n" "$RUN_TS"
printf "[train-flow] dataset: %s\n" "$DATASET_FILE"
printf "[train-flow] model=%s epochs=%s batch=%s seq=%s lr=%s steps=%s\n" \
  "$MODEL_SIZE" "$EPOCHS" "$BATCH_SIZE" "$SEQ_LEN" "$LR" "$STEPS_PER_EPOCH"

"$PYTHON_BIN" -m app.training.train_neurx \
  --model-size "$MODEL_SIZE" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --learning-rate "$LR" \
  --num-batches-per-epoch "$STEPS_PER_EPOCH" \
  --dataset-file "$DATASET_FILE" \
  --save-path "$SAVE_PATH"

LATEST_CKPT="$(ls -1t checkpoints/model_neurx_dataset*.pkl | head -n 1 || true)"
if [[ -z "$LATEST_CKPT" ]]; then
  echo "[train-flow] no checkpoint produced" >&2
  exit 1
fi

cp -f "$LATEST_CKPT" checkpoints/model_neurx_dataset_latest.pkl
REPORT_PATH="reports/training_flow_${RUN_TS}.json"

cat > "$REPORT_PATH" <<EOF
{
  "run_ts_utc8": "$RUN_TS",
  "dataset_file": "$DATASET_FILE",
  "model_size": "$MODEL_SIZE",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "seq_len": $SEQ_LEN,
  "learning_rate": "$LR",
  "steps_per_epoch": $STEPS_PER_EPOCH,
  "checkpoint": "$LATEST_CKPT",
  "latest_alias": "checkpoints/model_neurx_dataset_latest.pkl"
}
EOF

printf "[train-flow] done\n"
printf "[train-flow] checkpoint: %s\n" "$LATEST_CKPT"
printf "[train-flow] latest alias: %s\n" "checkpoints/model_neurx_dataset_latest.pkl"
printf "[train-flow] report: %s\n" "$REPORT_PATH"
