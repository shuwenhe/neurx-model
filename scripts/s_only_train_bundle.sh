#!/usr/bin/env bash
set -euo pipefail

S_COMPILER="${S_COMPILER:-/usr/local/bin/s}"
NEURX_ROOT="${NEURX_ROOT:-/app/neurx}"
MODEL_ROOT="${MODEL_ROOT:-/app/neurx-model}"
RUN_TS="$(TZ=Asia/Shanghai date +%Y%m%d%H%M%S)"
OUT_DIR_DEFAULT="$MODEL_ROOT/reports/s_train_${RUN_TS}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      shift
      OUT_DIR="$1"
      ;;
    --out-dir=*)
      OUT_DIR="${1#*=}"
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--out-dir PATH]" >&2
      exit 2
      ;;
  esac
  shift
done

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
mode=s_only_compile_bundle
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

# Export a deployable pure-S checkpoint into neurx-model/checkpoints.
cp -f "$BIN_DIR/gpt_model_ops.bin" "$CHECKPOINT_BIN"

cat > "$CHECKPOINT_META" << EOF
{
  "run_ts": "$RUN_TS",
  "mode": "s_only_checkpoint_export",
  "checkpoint_bin": "$CHECKPOINT_BIN",
  "bundle_root": "$OUT_DIR",
  "entries": {
    "train_bin": "$BIN_DIR/train_loop.bin",
    "ops_bin": "$BIN_DIR/ops.bin",
    "model_bin": "$BIN_DIR/gpt_model_ops.bin"
  }
}
EOF

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
ls -1 "$BIN_DIR"
