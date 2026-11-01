"""Environment snapshot written next to every run (provenance, not metrics)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_info(root: Path = REPO_ROOT) -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {
        "sha": sha,  # None until the repo has a commit
        "dirty": bool(status) if status is not None else None,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
    }


def pip_freeze(python: str | None = None) -> list[str]:
    """Installed distributions. For the current interpreter use importlib.metadata
    (works in uv venvs without pip); for another venv shell out to it."""
    # Compare venv prefixes, not resolved paths: every uv venv's python is a
    # symlink to the same base interpreter, so resolve() made them all look equal.
    if python and Path(python).absolute().parent.parent != Path(sys.prefix).absolute():
        out = _run([python, "-m", "pip", "freeze"]) or _run(["uv", "pip", "freeze", "--python", python])
        return out.splitlines() if out else []
    pkgs = {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata.get("Name")}
    return sorted(pkgs, key=str.lower)


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    if shutil.which("nvidia-smi"):
        info["nvidia_smi"] = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,pci.bus_id,clocks.max.sm,power.limit",
                "--format=csv",
            ]
        )
        info["nvidia_smi_full"] = _run(["nvidia-smi"])
    if shutil.which("nvcc"):
        info["nvcc"] = _run(["nvcc", "--version"])
    try:
        import torch  # optional

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        info["torch"] = None
    return info


def env_snapshot(engine_desc: dict[str, Any] | None = None, engine_python: str | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "git": git_info(),
        "packages": pip_freeze(),
        "gpu": gpu_info(),
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA", "NCCL", "VLLM", "SGLANG", "HF_", "TORCH"))
            and not any(s in k for s in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
        },
        "engine": engine_desc or {},
    }
    # Server engines run in their own venv: record its packages too.
    if engine_python:
        snap["engine_packages"] = pip_freeze(engine_python)
        snap["engine_python"] = engine_python
    return snap
