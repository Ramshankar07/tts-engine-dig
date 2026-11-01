"""Official Qwen3-TTS (PyTorch / HF transformers) in-process adapter.

API verified against QwenLM/Qwen3-TTS @ 022e286b (qwen_tts 0.1.1):

    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0",
                                          dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2")
    wavs, sr = model.generate_custom_voice(text=..., speaker="Ryan", language="English",
                                           instruct=None, do_sample=..., top_k=..., ...)

``generate_custom_voice`` accepts a list of texts (static batch) and returns
``(List[np.ndarray], sample_rate)``. Internally it calls
``self.model.generate(...)`` (talker LM + code predictor) and then
``self.model.speech_tokenizer.decode(...)`` (12 Hz codec decoder); we wrap
those two calls to time LM vs codec separately and to label profiler ranges.

Streaming: the package exposes no streaming generation API. Its
``non_streaming_mode=False`` flag only *simulates streaming text input* (see the
docstring at 022e286b) and still returns complete audio. So this engine has
``supports_streaming = False``; ``stream()`` yields a single final chunk and the
streaming comparison skips it.

Concurrency: there is no request scheduler; concurrency N is run as static
batches of N prompts (``batch_concurrency = True``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np

from .base import Engine, EngineError, SynthesisRequest


class OfficialQwenEngine(Engine):
    name = "official"
    supports_streaming = False
    batch_concurrency = True

    def __init__(
        self,
        model_path: str,
        sampling: dict[str, Any] | None = None,
        precision: str = "bfloat16",
        device: str = "cuda:0",
        attn_implementation: str = "flash_attention_2",
        revision: str | None = None,
        sample_rate: int = 24000,
        **_ignored: Any,
    ) -> None:
        super().__init__(sample_rate)
        self.model_path = model_path
        self.sampling = dict(sampling or {})
        self.precision = precision
        self.device = device
        self.attn_implementation = attn_implementation
        self.revision = revision
        self.model = None
        self._lock = threading.Lock()  # one GPU, one model: serialize calls
        self._timing = threading.local()
        self.profile_ranges = False  # set by profile.py to emit record_function ranges

    # -- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        if self.model is None:
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:  # pragma: no cover - GPU box only
            raise EngineError("official engine needs the [official] extra: uv pip install -e '.[official]'") from e
        attn = self.attn_implementation
        if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            attn = "sdpa"  # README: FA2 recommended but optional
        self.attn_used = attn
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.precision]
        kwargs: dict[str, Any] = dict(device_map=self.device, dtype=dtype, attn_implementation=attn)
        # from_pretrained forwards kwargs to AutoModel only (the processor is
        # loaded without them), so pin the revision by pre-downloading to a
        # local dir (scripts/setup_gpu_box.sh) and passing that path.
        self.model = Qwen3TTSModel.from_pretrained(self.model_path, **kwargs)
        self._instrument()

    def _instrument(self) -> None:
        """Wrap LM generate and codec decode for timing + profiler ranges."""
        import torch

        inner = self.model.model  # Qwen3TTSForConditionalGeneration
        orig_generate = inner.generate
        tok = inner.speech_tokenizer
        orig_decode = tok.decode
        engine = self

        @contextmanager
        def timed(label: str):
            rf = torch.profiler.record_function(label) if engine.profile_ranges else nullcontext()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            with rf:
                yield
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            setattr(engine._timing, label, getattr(engine._timing, label, 0.0) + time.perf_counter() - t)

        def generate(*a, **k):
            with timed("lm_generate"):
                return orig_generate(*a, **k)

        def decode(*a, **k):
            with timed("codec_decode"):
                return orig_decode(*a, **k)

        # Instance attributes shadow the bound methods; nn.Module.__setattr__
        # stores plain callables normally.
        inner.generate = generate
        tok.decode = decode

    async def close(self) -> None:
        return None  # keep the model resident across runs

    def describe(self) -> dict[str, Any]:
        d = {**super().describe(), "model_path": self.model_path, "precision": self.precision, "device": self.device}
        d["attn_implementation"] = getattr(self, "attn_used", self.attn_implementation)
        try:
            import qwen_tts  # noqa: F401
            from importlib.metadata import version

            d["qwen_tts_version"] = version("qwen-tts")
        except Exception:  # noqa: BLE001
            pass
        return d

    # -- generation --------------------------------------------------------------
    def _gen_kwargs(self) -> dict[str, Any]:
        keys = (
            "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature",
            "max_new_tokens",
        )
        return {k: self.sampling[k] for k in keys if k in self.sampling}

    def _generate_blocking(self, reqs: list[SynthesisRequest]) -> list[tuple[np.ndarray, int, dict[str, Any]]]:
        import torch

        if self.model is None:
            raise EngineError("engine not started")
        with self._lock:
            seed = reqs[0].seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self._timing.__dict__.clear()
            texts = [r.text for r in reqs]
            instr = [r.instructions or "" for r in reqs]
            wavs, sr = self.model.generate_custom_voice(
                text=texts if len(texts) > 1 else texts[0],
                speaker=[r.voice for r in reqs] if len(reqs) > 1 else reqs[0].voice,
                language=[r.language for r in reqs] if len(reqs) > 1 else reqs[0].language,
                instruct=(instr if len(reqs) > 1 else instr[0]) if any(instr) else None,
                **self._gen_kwargs(),
            )
            meta: dict[str, Any] = {
                "lm_generate_s": getattr(self._timing, "lm_generate", None),
                "codec_decode_s": getattr(self._timing, "codec_decode", None),
            }
            if torch.cuda.is_available():
                meta["torch_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
                meta["torch_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        self.sample_rate = int(sr)
        return [(np.asarray(w, dtype=np.float32).reshape(-1), int(sr), dict(meta)) for w in wavs]

    async def _synthesize(self, req: SynthesisRequest) -> tuple[np.ndarray, int, dict[str, Any]]:
        (out,) = await asyncio.to_thread(self._generate_blocking, [req])
        return out

    async def _synthesize_batch(self, reqs: list[SynthesisRequest]) -> list[tuple[np.ndarray, int, dict[str, Any]]]:
        return await asyncio.to_thread(self._generate_blocking, reqs)
