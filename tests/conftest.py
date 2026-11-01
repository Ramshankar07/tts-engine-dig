from __future__ import annotations

from pathlib import Path

import pytest

from tts_bench.config import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def fake_cfg(tmp_path: Path):
    """Fast fake-engine config writing runs into a temp dir."""
    return load_config(
        REPO / "configs" / "fake_cpu.toml",
        overrides={
            "run": {"output_dir": str(tmp_path / "runs"), "limit": 6, "warmup_requests": 1},
            "quality": {"bootstrap_samples": 200},
            "engines": {"fake": {"prefill_ms": 10, "rtf": 0.01, "chunk_overhead_ms": 1, "jitter_ms": 0}},
        },
    )
