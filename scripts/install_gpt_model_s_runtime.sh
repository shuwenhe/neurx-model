#!/usr/bin/env bash
set -euo pipefail

S_COMPILER="${S_COMPILER:-/usr/local/bin/s}"
SRC="/app/neurx-model/s/gpt_model_ops.s"
RUNTIME_ROOT="/app/neurx/python/neurx/compile/_s_runtime"
OUT_IR="$RUNTIME_ROOT/gpt_model_ops.ir"

if [[ ! -x "$S_COMPILER" ]]; then
  echo "error: s compiler not executable: $S_COMPILER" >&2
  exit 1
fi

if [[ ! -f "$SRC" ]]; then
  echo "error: source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT"

"$S_COMPILER" ir "$SRC" -o "$OUT_IR"

python3 - <<'PY'
from pathlib import Path
import json

root = Path('/app/neurx/python/neurx/compile/_s_runtime')
manifest_path = root / 'manifest.json'
ir_files = sorted(p.name for p in root.glob('*.ir'))

if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        manifest = {}
else:
    manifest = {}

manifest['artifact_root'] = str(root.resolve())
manifest['ir_files'] = ir_files
manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
print(f'updated runtime manifest: {manifest_path} ({len(ir_files)} ir files)')
PY

echo "installed: $OUT_IR"
