"""Engine registry: build any engine from the pinned BenchConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AudioChunk, Engine, EngineError, SynthesisRequest, SynthesisResult
from .fake import FakeEngine

if TYPE_CHECKING:
    from ..config import BenchConfig

ENGINE_NAMES = ("official", "vllm_omni", "sglang_omni", "fake")

__all__ = [
    "AudioChunk",
    "Engine",
    "EngineError",
    "SynthesisRequest",
    "SynthesisResult",
    "ENGINE_NAMES",
    "build_engine",
]


def build_engine(name: str, cfg: "BenchConfig") -> Engine:
    s = cfg.engine_settings(name)
    sr = cfg.model.sample_rate
    sampling = cfg.sampling.model_dump()
    if name == "fake":
        return FakeEngine(sample_rate=sr, frame_rate=cfg.model.codec_frame_rate_hz, **s)
    if name == "official":
        from .official import OfficialQwenEngine  # lazy: imports torch on start()

        return OfficialQwenEngine(
            model_path=cfg.model.path_or_id,
            sampling=sampling,
            precision=cfg.model.precision,
            revision=cfg.model.revision,
            sample_rate=sr,
            **s,
        )
    if name == "vllm_omni":
        from .vllm_omni import VllmOmniEngine

        return VllmOmniEngine(
            max_new_tokens=cfg.sampling.max_new_tokens,
            sample_rate=sr,
            timeout_s=cfg.run.request_timeout_s,
            **s,
        )
    if name == "sglang_omni":
        from .sglang_omni import SglangOmniEngine

        s.setdefault("suppress_bootstrap_silence", cfg.streaming.suppress_bootstrap_silence)
        return SglangOmniEngine(
            sampling=sampling,
            max_new_tokens=cfg.sampling.max_new_tokens,
            sample_rate=sr,
            timeout_s=cfg.run.request_timeout_s,
            **s,
        )
    raise ValueError(f"unknown engine {name!r}; choose from {ENGINE_NAMES}")
