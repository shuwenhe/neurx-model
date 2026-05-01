"""Utilities for preparing NeurX S runtime artifacts."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SYSTEM_RUNTIME_ROOT = "/usr/local/share/neurx_s_runtime"
DEFAULT_SOURCE_ROOT = "/app/neurx/s"
DEFAULT_MODEL_SOURCE_ROOT = "/app/neurx-model/s"


@dataclass
class RuntimePreparationResult:
    mode: str
    destination_root: str
    compiler: str
    runtime_source: str
    ir_files: list[str]
    compile_error: str


def discover_neurx_runtime_root() -> Path:
    import neurx.compile.runtime as runtime

    return Path(runtime.__file__).resolve().parent / "_s_runtime"


def list_s_sources(root_dir: str) -> list[str]:
    pattern = str(Path(root_dir).expanduser().resolve() / "*.s")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No S sources found in: {root_dir}")
    return files


def list_s_sources_from_roots(root_dirs: list[str]) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for root_dir in root_dirs:
        root_dir = (root_dir or "").strip()
        if not root_dir:
            continue
        for src in list_s_sources(root_dir):
            if src not in seen:
                files.append(src)
                seen.add(src)
    if not files:
        raise FileNotFoundError(f"No S sources found in: {root_dirs}")
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


def _runtime_root_valid(runtime_root: Path) -> bool:
    if not runtime_root.exists() or not runtime_root.is_dir():
        return False
    if not (runtime_root / "manifest.json").exists():
        return False
    return any(runtime_root.glob("*.ir"))


def _compile_runtime(compiler: str, out_dir: Path, source_files: list[str], source_root: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled: list[str] = []
    for src in source_files:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing S runtime source: {src}")
        ir_path = out_dir / f"{src_path.stem}.ir"
        cmd = [compiler, str(src_path), str(ir_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "S runtime compile failed for "
                f"{src_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        compiled.append(str(ir_path))

    manifest = {
        "compiler": compiler,
        "source_root": str(Path(source_root).resolve()),
        "sources": source_files,
        "ir_files": compiled,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return compiled


def _sync_runtime(src_root: Path, dst_root: Path) -> list[str]:
    dst_root.mkdir(parents=True, exist_ok=True)

    # Remove stale IRs before syncing to avoid mixed runtime versions.
    for stale in dst_root.glob("*.ir"):
        stale.unlink()

    copied: list[str] = []
    for ir_file in sorted(src_root.glob("*.ir")):
        target = dst_root / ir_file.name
        shutil.copy2(ir_file, target)
        copied.append(str(target))

    manifest = src_root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.json not found in runtime source: {src_root}")
    shutil.copy2(manifest, dst_root / "manifest.json")

    return copied


def prepare_s_runtime(
    mode: str,
    system_runtime_root: str,
    source_root: str,
    model_source_root: str,
    compile_out_dir: str,
    compiler_path: str,
    allow_compile_fail: bool,
) -> RuntimePreparationResult:
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "system", "compile"}:
        raise ValueError(f"unsupported s-runtime mode: {mode}")

    destination_root = discover_neurx_runtime_root()
    system_root = Path(system_runtime_root).expanduser().resolve()
    compile_out_root = Path(compile_out_dir).expanduser().resolve()

    compile_error = ""
    compiler = ""
    runtime_source = ""

    if mode in {"auto", "system"} and _runtime_root_valid(system_root):
        copied = _sync_runtime(system_root, destination_root)
        return RuntimePreparationResult(
            mode="system",
            destination_root=str(destination_root),
            compiler=compiler,
            runtime_source=str(system_root),
            ir_files=copied,
            compile_error=compile_error,
        )

    if mode == "system":
        raise FileNotFoundError(f"system runtime not found or invalid: {system_root}")

    compiler = resolve_s_compiler(compiler_path or None)
    try:
        source_files = list_s_sources_from_roots([source_root, model_source_root])
        _compile_runtime(compiler, compile_out_root, source_files, source_root)
        copied = _sync_runtime(compile_out_root, destination_root)
        runtime_source = str(compile_out_root)
    except Exception as exc:
        compile_error = str(exc)
        if not allow_compile_fail:
            raise
        copied = []

    return RuntimePreparationResult(
        mode="compile",
        destination_root=str(destination_root),
        compiler=compiler,
        runtime_source=runtime_source,
        ir_files=copied,
        compile_error=compile_error,
    )
