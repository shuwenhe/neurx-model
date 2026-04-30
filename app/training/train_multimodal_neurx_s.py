"""Multimodal training entrypoint using NeurX with S-runtime preflight."""

from __future__ import annotations

import argparse
import os

from app.training.train_vision_real import train_vision_real
from app.training.s_runtime import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SYSTEM_RUNTIME_ROOT,
    prepare_s_runtime,
)


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
    parser.add_argument(
        "--s-source-root",
        type=str,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory of neurx S runtime sources",
    )
    parser.add_argument(
        "--s-runtime-mode",
        type=str,
        default="auto",
        choices=["auto", "system", "compile"],
        help="Use system runtime, compile runtime, or auto-detect",
    )
    parser.add_argument(
        "--s-runtime-root",
        type=str,
        default=os.environ.get("NEURX_S_OPS_RUNTIME", DEFAULT_SYSTEM_RUNTIME_ROOT),
        help="System S runtime root (contains *.ir and manifest.json)",
    )
    parser.add_argument(
        "--allow-s-compile-fail",
        action="store_true",
        help="Continue training when S runtime compile fails",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runtime_result = prepare_s_runtime(
        mode=args.s_runtime_mode,
        system_runtime_root=args.s_runtime_root,
        source_root=args.s_source_root,
        compile_out_dir=args.s_ir_dir,
        compiler_path=args.s_compiler,
        allow_compile_fail=args.allow_s_compile_fail,
    )

    print("=" * 70)
    print("NeurX Multimodal Training (S runtime preflight)")
    print("=" * 70)
    print(f"s_runtime_mode={runtime_result.mode}")
    print(f"s_runtime_source={runtime_result.runtime_source}")
    print(f"s_runtime_destination={runtime_result.destination_root}")
    print(f"s_compiler={runtime_result.compiler}")
    if runtime_result.compile_error:
        print("s_runtime_status=failed")
        print(runtime_result.compile_error)
    else:
        print("s_runtime_status=ok")
        print(f"runtime_ir={len(runtime_result.ir_files)} files")
        for path in runtime_result.ir_files:
            print(f"  - {path}")

    args.backend = "neurx_s_latest"
    args.s_compiler = runtime_result.compiler
    args.s_runtime_mode = runtime_result.mode
    args.s_runtime_source = runtime_result.runtime_source
    args.compiled_s_ir = runtime_result.ir_files
    args.s_compile_error = runtime_result.compile_error

    train_vision_real(args)


if __name__ == "__main__":
    main()
