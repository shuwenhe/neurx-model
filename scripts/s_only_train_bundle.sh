#!/usr/bin/env bash
set -euo pipefail

S_COMPILER="${S_COMPILER:-/usr/local/bin/s}"
NEURX_ROOT="${NEURX_ROOT:-/app/neurx}"
MODEL_ROOT="${MODEL_ROOT:-/app/neurx-model}"
RUN_TS="$(TZ=Asia/Shanghai date +%Y%m%d%H%M%S)"
OUT_DIR_DEFAULT="$MODEL_ROOT/reports/s_train_${RUN_TS}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
DATASET_FILE="${DATASET_FILE:-}"

MODE="s_only_compile_bundle"
CHECKPOINT_MODE="s_only_checkpoint_export"
DATASET_FILE_ABS=""
DATASET_SHA256=""
DATASET_LINE_COUNT="0"
DATASET_BYTE_COUNT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      shift
      OUT_DIR="$1"
      ;;
    --out-dir=*)
      OUT_DIR="${1#*=}"
      ;;
    --dataset-file)
      shift
      DATASET_FILE="$1"
      ;;
    --dataset-file=*)
      DATASET_FILE="${1#*=}"
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--out-dir PATH] [--dataset-file PATH]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -n "$DATASET_FILE" ]]; then
  if [[ ! -f "$DATASET_FILE" ]]; then
    echo "dataset file not found: $DATASET_FILE" >&2
    exit 1
  fi
  DATASET_FILE_ABS="$(readlink -f "$DATASET_FILE")"
  if [[ ! -s "$DATASET_FILE_ABS" ]]; then
    echo "dataset file is empty: $DATASET_FILE_ABS" >&2
    exit 1
  fi
  DATASET_SHA256="$(sha256sum "$DATASET_FILE_ABS" | awk '{print $1}')"
  DATASET_LINE_COUNT="$(wc -l < "$DATASET_FILE_ABS" | tr -d ' ')"
  DATASET_BYTE_COUNT="$(wc -c < "$DATASET_FILE_ABS" | tr -d ' ')"
  MODE="s_only_dataset_compile_bundle"
  CHECKPOINT_MODE="s_only_dataset_checkpoint_export"
fi

IR_DIR="$OUT_DIR/ir"
BIN_DIR="$OUT_DIR/bin"
CHECKPOINT_DIR="$MODEL_ROOT/checkpoints"
CHECKPOINT_BIN="$CHECKPOINT_DIR/s_arch_${RUN_TS}.bin"
CHECKPOINT_META="$CHECKPOINT_DIR/s_arch_${RUN_TS}.json"

mkdir -p "$IR_DIR" "$BIN_DIR"
mkdir -p "$CHECKPOINT_DIR"

# --emit-bin expects runtime/memory.h in the S source root.
ln -sfn "$NEURX_ROOT/../s/src/cmd/compile/seed/runtime" "$NEURX_ROOT/../s/runtime"

cd "$NEURX_ROOT"
for src in $(printf '%s\n' s/*.s ops/*.s tensor/*.s ad/*.s engine/*.s nn/*.s opt/*.s lf/*.s train/*.s runtime/*.s distributed/*.s platform/*.s compile/*.s); do
  [ -e "$src" ] || continue
  base="$(basename "$src" .s)"
  dir="$(dirname "$src")"
  case "$dir" in
    s|ops) sub="" ;;
    *) sub="/$dir" ;;
  esac
  mkdir -p "$IR_DIR$sub"
  "$S_COMPILER" "$src" "$IR_DIR$sub/$base.ir"
done

mkdir -p "$IR_DIR/model"
"$S_COMPILER" "$MODEL_ROOT/s/gpt_model_ops.s" "$IR_DIR/model/gpt_model_ops.ir"

cd "$NEURX_ROOT/../s"
"$S_COMPILER" --emit-bin "$IR_DIR/train/loop.ir" "$BIN_DIR/train_loop.bin"
"$S_COMPILER" --emit-bin "$IR_DIR/ops.ir" "$BIN_DIR/ops.bin"
"$S_COMPILER" --emit-bin "$IR_DIR/model/gpt_model_ops.ir" "$BIN_DIR/gpt_model_ops.bin"

cat > "$OUT_DIR/manifest.txt" << EOF
run_ts=$RUN_TS
mode=$MODE
compiler=$S_COMPILER
ir_root=$IR_DIR
bin_root=$BIN_DIR
entry_train_ir=$IR_DIR/train/loop.ir
entry_train_bin=$BIN_DIR/train_loop.bin
entry_ops_ir=$IR_DIR/ops.ir
entry_ops_bin=$BIN_DIR/ops.bin
entry_model_ir=$IR_DIR/model/gpt_model_ops.ir
entry_model_bin=$BIN_DIR/gpt_model_ops.bin
EOF

if [[ -n "$DATASET_FILE_ABS" ]]; then
cat >> "$OUT_DIR/manifest.txt" << EOF
dataset_file=$DATASET_FILE_ABS
dataset_sha256=$DATASET_SHA256
dataset_line_count=$DATASET_LINE_COUNT
dataset_byte_count=$DATASET_BYTE_COUNT
EOF
fi

cat > "$OUT_DIR/s_runtime_entry.env" << EOF
NEURX_S_ENTRY_TRAIN_BIN=$BIN_DIR/train_loop.bin
NEURX_S_ENTRY_OPS_BIN=$BIN_DIR/ops.bin
NEURX_S_ENTRY_MODEL_BIN=$BIN_DIR/gpt_model_ops.bin
NEURX_S_ENTRY_TRAIN_IR=$IR_DIR/train/loop.ir
NEURX_S_ENTRY_OPS_IR=$IR_DIR/ops.ir
NEURX_S_ENTRY_MODEL_IR=$IR_DIR/model/gpt_model_ops.ir
NEURX_S_BUNDLE_ROOT=$OUT_DIR
NEURX_S_BUNDLE_RUN_TS=$RUN_TS
EOF

if [[ -n "$DATASET_FILE_ABS" ]]; then
cat >> "$OUT_DIR/s_runtime_entry.env" << EOF
NEURX_S_DATASET_FILE=$DATASET_FILE_ABS
NEURX_S_DATASET_SHA256=$DATASET_SHA256
NEURX_S_DATASET_LINE_COUNT=$DATASET_LINE_COUNT
NEURX_S_DATASET_BYTE_COUNT=$DATASET_BYTE_COUNT
EOF
fi

if [[ -n "$DATASET_FILE_ABS" ]]; then
cat > "$OUT_DIR/s_runtime_entry.json" << EOF
{
  "bundle_root": "$OUT_DIR",
  "run_ts": "$RUN_TS",
  "dataset": {
    "file": "$DATASET_FILE_ABS",
    "sha256": "$DATASET_SHA256",
    "line_count": $DATASET_LINE_COUNT,
    "byte_count": $DATASET_BYTE_COUNT
  },
  "entries": {
    "train_bin": "$BIN_DIR/train_loop.bin",
    "ops_bin": "$BIN_DIR/ops.bin",
    "model_bin": "$BIN_DIR/gpt_model_ops.bin",
    "train_ir": "$IR_DIR/train/loop.ir",
    "ops_ir": "$IR_DIR/ops.ir",
    "model_ir": "$IR_DIR/model/gpt_model_ops.ir"
  }
}
EOF
else
cat > "$OUT_DIR/s_runtime_entry.json" << EOF
{
  "bundle_root": "$OUT_DIR",
  "run_ts": "$RUN_TS",
  "entries": {
    "train_bin": "$BIN_DIR/train_loop.bin",
    "ops_bin": "$BIN_DIR/ops.bin",
    "model_bin": "$BIN_DIR/gpt_model_ops.bin",
    "train_ir": "$IR_DIR/train/loop.ir",
    "ops_ir": "$IR_DIR/ops.ir",
    "model_ir": "$IR_DIR/model/gpt_model_ops.ir"
  }
}
EOF
fi

# Export a deployable pure-S checkpoint into neurx-model/checkpoints.
cp -f "$BIN_DIR/gpt_model_ops.bin" "$CHECKPOINT_BIN"

if [[ -n "$DATASET_FILE_ABS" ]]; then
cat > "$CHECKPOINT_META" << EOF
{
  "run_ts": "$RUN_TS",
  "mode": "$CHECKPOINT_MODE",
  "checkpoint_bin": "$CHECKPOINT_BIN",
  "bundle_root": "$OUT_DIR",
  "dataset": {
    "file": "$DATASET_FILE_ABS",
    "sha256": "$DATASET_SHA256",
    "line_count": $DATASET_LINE_COUNT,
    "byte_count": $DATASET_BYTE_COUNT
  },
  "entries": {
    "train_bin": "$BIN_DIR/train_loop.bin",
    "ops_bin": "$BIN_DIR/ops.bin",
    "model_bin": "$BIN_DIR/gpt_model_ops.bin"
  }
}
EOF
else
cat > "$CHECKPOINT_META" << EOF
{
  "run_ts": "$RUN_TS",
  "mode": "$CHECKPOINT_MODE",
  "checkpoint_bin": "$CHECKPOINT_BIN",
  "bundle_root": "$OUT_DIR",
  "entries": {
    "train_bin": "$BIN_DIR/train_loop.bin",
    "ops_bin": "$BIN_DIR/ops.bin",
    "model_bin": "$BIN_DIR/gpt_model_ops.bin"
  }
}
EOF
fi

ln -sfn "$CHECKPOINT_BIN" "$CHECKPOINT_DIR/s_arch_latest.bin"
ln -sfn "$CHECKPOINT_META" "$CHECKPOINT_DIR/s_arch_latest.json"

ln -sfn "$OUT_DIR" "$MODEL_ROOT/reports/s_train_latest"

echo "S-only training bundle ready: $OUT_DIR"
echo "Checkpoint bin: $CHECKPOINT_BIN"
echo "Checkpoint meta: $CHECKPOINT_META"
echo "Checkpoint latest: $CHECKPOINT_DIR/s_arch_latest.bin"
echo "Runtime env entry: $OUT_DIR/s_runtime_entry.env"
echo "Runtime json entry: $OUT_DIR/s_runtime_entry.json"
echo "Latest link: $MODEL_ROOT/reports/s_train_latest"
if [[ -n "$DATASET_FILE_ABS" ]]; then
  echo "Dataset file: $DATASET_FILE_ABS"
  echo "Dataset sha256: $DATASET_SHA256"
  echo "Dataset lines: $DATASET_LINE_COUNT"
  echo "Dataset bytes: $DATASET_BYTE_COUNT"
fi
ls -1 "$BIN_DIR"
