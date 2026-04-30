"""Multimodal training entrypoint using NeurX with S-runtime preflight."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

from app.training.train_vision_real import train_vision_real


S_SOURCE_ROOT = "/app/neurx/s"


def list_s_sources(root_dir: str) -> list[str]:
    pattern = str(Path(root_dir).expanduser().resolve() / "*.s")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No S sources found in: {root_dir}")
    return files


def resolve_s_compiler(explicit_path: str | None) -> str:
    if explicit_path:
        compiler = Path(explicit_path).expanduser().resolve()
        if compiler.exists() and os.access(compiler, os.X_OK):
            return str(compiler)
        raise FileNotFoundError(f"S compiler not executable: {compiler}")

    candidates = []
    for candidate in ("/usr/local/bin/s", "/app/s/bin/s"):
        path = Path(candidate)
        if path.exists() and os.access(path, os.X_OK):
            candidates.append(str(path))
    candidates.extend(sorted(glob.glob("/app/s/bin/s_*")))
    if not candidates:
        raise FileNotFoundError("No S compiler found under /app/s/bin or /usr/local/bin/s")

    compiler = candidates[0]
    if not os.access(compiler, os.X_OK):
        raise PermissionError(f"S compiler not executable: {compiler}")
    return compiler


def compile_s_runtime(compiler: str, out_dir: str, source_files: list[str]) -> list[str]:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    compiled = []
    for src in source_files:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing S runtime source: {src}")
        ir_path = out_path / f"{src_path.stem}.ir"
        cmd = [compiler, "ir", str(src_path), "-o", str(ir_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "S runtime compile failed for "
                f"{src_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        compiled.append(str(ir_path))

    manifest_path = out_path / "manifest.json"
    manifest = {
        "compiler": compiler,
        "source_root": str(Path(S_SOURCE_ROOT).resolve()),
        "sources": source_files,
        "ir_files": compiled,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

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
    parser.add_argument(
        "--s-source-root",
        type=str,
        default=S_SOURCE_ROOT,
        help="Root directory of neurx S runtime sources",
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

    compiler = resolve_s_compiler(args.s_compiler or None)
    source_files = list_s_sources(args.s_source_root)

    compiled_ir = []
    compile_error = ""
    try:
        compiled_ir = compile_s_runtime(compiler, args.s_ir_dir, source_files)
    except Exception as exc:
        compile_error = str(exc)
        if not args.allow_s_compile_fail:
            raise

    print("=" * 70)
    print("NeurX Multimodal Training (S runtime preflight)")
    print("=" * 70)
    print(f"s_compiler={compiler}")
    if compile_error:
        print("s_compile_status=failed")
        print(compile_error)
    else:
        print("s_compile_status=ok")
        print(f"compiled_ir={len(compiled_ir)} files")
        for path in compiled_ir:
            print(f"  - {path}")

    args.backend = "neurx_s_latest"
    args.s_compiler = compiler
    args.compiled_s_ir = compiled_ir
    args.s_compile_error = compile_error

    train_vision_real(args)


if __name__ == "__main__":
    main()
