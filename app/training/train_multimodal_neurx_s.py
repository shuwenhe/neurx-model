"""Multimodal training entrypoint using NeurX with S-runtime preflight."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path

from app.training.train_vision_real import train_vision_real


S_SOURCE_FILES = [
    "/app/neurx/s/tensor.s",
    "/app/neurx/s/ops.s",
    "/app/neurx/s/autograd.s",
    "/app/neurx/s/schedule.s",
]


def resolve_s_compiler(explicit_path: str | None) -> str:
    if explicit_path:
        compiler = Path(explicit_path).expanduser().resolve()
        if compiler.exists() and os.access(compiler, os.X_OK):
            return str(compiler)
        raise FileNotFoundError(f"S compiler not executable: {compiler}")

    candidates = sorted(glob.glob("/app/s/bin/s_*"))
    if not candidates:
        raise FileNotFoundError("No S compiler found under /app/s/bin")

    compiler = candidates[-1]
    if not os.access(compiler, os.X_OK):
        raise PermissionError(f"S compiler not executable: {compiler}")
    return compiler


def compile_s_runtime(compiler: str, out_dir: str) -> list[str]:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    compiled = []
    for src in S_SOURCE_FILES:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing S runtime source: {src}")
        ir_path = out_path / f"{src_path.stem}.ir"
        cmd = [compiler, str(src_path), str(ir_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "S runtime compile failed for "
                f"{src_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        compiled.append(str(ir_path))
    return compiled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train neurx-model multimodal model with latest S NeurX runtime preflight"
    )
    parser.add_argument("--data-source", type=str, default="local", choices=["local"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-multiplier", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="checkpoints/vision_trained_model_s.pkl")

    parser.add_argument("--s-compiler", type=str, default="", help="Path to S compiler binary")
    parser.add_argument(
        "--s-ir-dir",
        type=str,
        default="reports/s_ir",
        help="Directory for generated IR files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    compiler = resolve_s_compiler(args.s_compiler or None)
    compiled_ir = compile_s_runtime(compiler, args.s_ir_dir)

    print("=" * 70)
    print("NeurX Multimodal Training (S runtime preflight)")
    print("=" * 70)
    print(f"s_compiler={compiler}")
    print(f"compiled_ir={len(compiled_ir)} files")
    for path in compiled_ir:
        print(f"  - {path}")

    args.backend = "neurx_s_latest"
    args.s_compiler = compiler
    args.compiled_s_ir = compiled_ir

    train_vision_real(args)


if __name__ == "__main__":
    main()
