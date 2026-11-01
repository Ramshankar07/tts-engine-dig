"""Pinned run configuration (TOML -> pydantic).

One config file fixes everything that must be matched across engines: checkpoint
+ revision, voice, language, precision, sampling parameters, seed, concurrency
levels and streaming chunk settings. Engine-specific connection details live
under [engines.<name>].
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Sentinel used in TOML for "do not send initial_codec_chunk_frames; let the
# engine use its own default schedule" (TOML arrays cannot mix ints and strings).
ENGINE_DEFAULT_CHUNK = -1


class RunSection(BaseModel):
    name: str = "p1"
    seed: int = 1234
    prompts: str = "prompts/p1_prompts.jsonl"
    output_dir: str = "runs"
    concurrency: list[int] = Field(default_factory=lambda: [1, 4, 8])
    modes: list[Literal["full", "stream"]] = Field(default_factory=lambda: ["full", "stream"])
    warmup_requests: int = 3
    repeats: int = 1
    save_audio: bool = True
    request_timeout_s: float = 300.0
    limit: int | None = None  # use only the first N prompts (smoke tests)

    @field_validator("concurrency")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if not v or any(c < 1 for c in v):
            raise ValueError("concurrency levels must be >= 1")
        return v


class ModelSection(BaseModel):
    id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    revision: str = "main"
    local_path: str = ""  # if set, engines load from here instead of the hub id
    voice: str = "Ryan"
    language: str = "English"
    instructions: str = ""
    precision: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    sample_rate: int = 24000
    codec_frame_rate_hz: float = 12.5

    @property
    def path_or_id(self) -> str:
        return self.local_path or self.id


class SamplingSection(BaseModel):
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    max_new_tokens: int = 2048


class VerdictThresholds(BaseModel):
    """Rules used by streaming.py to label a chunk setting. Stated, not tuned."""

    min_ttfa_reduction: float = 0.5  # p50 TTFA must drop by >= 50% vs full
    max_underrun_request_frac: float = 0.05  # <= 5% of requests with a stall > threshold
    max_total_time_overhead: float = 0.2  # p50 total gen time may grow by <= 20%


class StreamingSection(BaseModel):
    # Values for `initial_codec_chunk_frames`; -1 = omit the field (engine default).
    initial_codec_chunk_frames: list[int] = Field(default_factory=lambda: [ENGINE_DEFAULT_CHUNK, 1, 4, 12])
    prebuffer_ms: float = 0.0  # client playback buffer before starting playback
    underrun_threshold_ms: float = 50.0  # stalls longer than this are "audible"
    silence_threshold: float = 1e-3  # |x| above this counts as non-silent audio
    suppress_bootstrap_silence: bool = False  # SGLang-only knob; False keeps outputs comparable
    verdict: VerdictThresholds = Field(default_factory=VerdictThresholds)


class GpuSection(BaseModel):
    device_index: int = 0
    poll_interval_s: float = 0.2


class QualitySection(BaseModel):
    asr_backend: Literal["whisper", "fake", "none"] = "whisper"
    asr_model: str = "openai/whisper-large-v3"
    asr_revision: str = "main"
    asr_language: str = "english"
    asr_device: str = "cuda:0"
    asr_batch_size: int = 8
    mos_backend: Literal["utmosv2", "none"] = "utmosv2"
    bootstrap_samples: int = 2000
    fake_error_every: int = 0  # fake ASR: drop every k-th word (0 = perfect)


class EngineSettings(BaseModel):
    """Free-form per-engine settings; each adapter validates what it needs."""

    model_config = {"extra": "allow"}


class BenchConfig(BaseModel):
    run: RunSection = Field(default_factory=RunSection)
    model: ModelSection = Field(default_factory=ModelSection)
    sampling: SamplingSection = Field(default_factory=SamplingSection)
    streaming: StreamingSection = Field(default_factory=StreamingSection)
    gpu: GpuSection = Field(default_factory=GpuSection)
    quality: QualitySection = Field(default_factory=QualitySection)
    engines: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Directory the config file was loaded from; relative paths resolve against
    # the repository root (parent of configs/) so commands work from anywhere.
    base_dir: str = "."

    def engine_settings(self, name: str) -> dict[str, Any]:
        return dict(self.engines.get(name, {}))

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else Path(self.base_dir) / p


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> BenchConfig:
    path = Path(path).resolve()
    with path.open("rb") as f:
        data = tomllib.load(f)
    if overrides:
        data = _deep_merge(data, overrides)
    # configs/<file>.toml -> repo root
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    data.setdefault("base_dir", str(base))
    return BenchConfig.model_validate(data)


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
