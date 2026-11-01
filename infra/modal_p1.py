"""Run the P1 GPU sessions (scripts/run_all.sh) on Modal.

    modal run infra/modal_p1.py --sessions "1"            # official baseline
    modal run infra/modal_p1.py --sessions "2 3" --limit 5
    modal run infra/modal_p1.py --cmd "nvidia-smi && ls /root/p1/venvs"
    modal volume get p1-runs runs ./modal-runs            # pull results

The image runs scripts/setup_gpu_box.sh once per engine step (each step is its
own cached layer). Only setup_gpu_box.sh/_common.sh and a package stub are baked
in; the live repo is mounted at /root/repo-src and synced over /root/repo at
start, so editing code does not rebuild the engine venvs. runs/ lives on the
`p1-runs` volume.
"""
from __future__ import annotations

import pathlib
import subprocess

import modal

REPO_LOCAL = pathlib.Path(__file__).resolve().parents[1]
REPO = "/root/repo"
WORK = "/root/p1"
GPU = "H100"

setup_env = {
    "WORK": WORK,
    "TORCH_BACKEND": "cu128",  # no driver at build time, so "auto" would pick CPU wheels
    "UV_LINK_MODE": "copy",
    "MAX_JOBS": "16",
}


FA_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)


def step(name: str, env: str = "") -> str:
    return f"cd {REPO} && {env} bash scripts/setup_gpu_box.sh {name}"


image = (
    # CUDA 13.0 toolkit: SGLang's DeepGEMM JIT requires nvcc >= 12.9, and the
    # vLLM/SGLang venvs run torch cu130 (official venv keeps torch 2.8 cu128).
    modal.Image.from_registry("nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "sox", "libsox-fmt-all", "libsndfile1", "ffmpeg", "build-essential", "rsync")
    .run_commands(
        # Nsight Systems CLI from the CUDA apt repo already configured in this image.
        "apt-get update && (apt-get install -y nsight-systems-cli || apt-get install -y cuda-nsight-systems-13-0) && nsys --version",
        "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
    )
    .env(setup_env)
    .add_local_file(REPO_LOCAL / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(REPO_LOCAL / "README.md", f"{REPO}/README.md", copy=True)
    .add_local_file(REPO_LOCAL / "scripts/_common.sh", f"{REPO}/scripts/_common.sh", copy=True)
    .add_local_file(REPO_LOCAL / "scripts/setup_gpu_box.sh", f"{REPO}/scripts/setup_gpu_box.sh", copy=True)
    .run_commands(f"mkdir -p {REPO}/src/tts_bench && touch {REPO}/src/tts_bench/__init__.py")
    .run_commands(step("model"))
    .run_commands(step("client"))
    .run_commands(step("eval"))
    .run_commands(step("official", f"OFFICIAL_TORCH=2.8.0 FLASH_ATTN_WHEEL={FA_WHEEL}"))
    .run_commands(step("vllm", "TORCH_BACKEND=cu130"))  # vllm 0.31.0 pins torch 2.13.0, not published for cu128
    .run_commands(step("sglang", "CC=gcc CXX=g++"))  # Modal python sysconfig defaults CC to clang
    # FlashInfer JIT (vLLM sampler, SGLang kernels) shells out to ninja.
    .run_commands(f"for v in vllm sglang; do uv pip install --python {WORK}/venvs/$v/bin/python ninja; done")
    # UTMOSv2's git install pulls PyPI (cu130) torchvision/torchaudio over the eval venv's
    # torch cu128; reinstall the matching build. Done as a late layer so the engine
    # venvs that produced the session 1-3 measurements are not rebuilt.
    .run_commands(f"uv pip install --python {WORK}/venvs/eval/bin/python --torch-backend=cu128 --reinstall-package torchvision --reinstall-package torchaudio torchvision torchaudio && {WORK}/venvs/eval/bin/python -c 'import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)'")
    .apt_install("wget")  # UTMOSv2 downloads its pretrained weights with wget
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        REPO_LOCAL,
        "/root/repo-src",
        ignore=[".venv", ".git", "runs", "reports", "__pycache__", ".pytest_cache", "modal-runs"],
    )
)

app = modal.App("p1-tts-engine-bench", image=image)
vol = modal.Volume.from_name("p1-runs", create_if_missing=True)


def _prepare() -> None:
    subprocess.run(
        f"rsync -a /root/repo-src/ {REPO}/ && mkdir -p /vol/runs /vol/reports /vol/hf "
        f"&& rm -rf {REPO}/runs {REPO}/reports && ln -s /vol/runs {REPO}/runs && ln -s /vol/reports {REPO}/reports",
        shell=True,
        check=True,
    )


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600, cpu=8, memory=65536)
def run(sessions: str = "", limit: int = 0, cmd: str = "", env: str = "") -> int:
    _prepare()
    if not cmd:
        cmd = f"bash scripts/run_all.sh {sessions}"
    prefix = f"export {env} && " if env else ""
    if limit:
        prefix += f"export LIMIT={limit} && "
    print(f"$ {prefix}{cmd}", flush=True)
    rc = subprocess.run(f"cd {REPO} && {prefix}{cmd}", shell=True).returncode
    vol.commit()
    print(f"exit code {rc}")
    return rc


@app.local_entrypoint()
def main(sessions: str = "", limit: int = 0, cmd: str = "", env: str = "") -> None:
    rc = run.remote(sessions=sessions, limit=limit, cmd=cmd, env=env)
    if rc != 0:
        raise SystemExit(rc)
