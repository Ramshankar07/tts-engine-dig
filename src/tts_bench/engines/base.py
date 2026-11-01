"""Common engine interface.

Every engine exposes two generation paths for the same request:

* ``synthesize(req)``  - full (non-streaming) generation: returns once all audio exists.
* ``stream(req)``      - async iterator of ``AudioChunk`` (PCM float32 + arrival time).

``run(req, mode)`` drives either path and returns a ``SynthesisResult`` with a
uniform timeline, so the benchmark never needs engine-specific timing code.
All timestamps are ``time.perf_counter()`` seconds relative to request submit.

The interface is async because two of the three engines are HTTP clients;
in-process engines (official, fake) run blocking work in a worker thread.
``synthesize_sync`` / ``stream_sync`` are thin wrappers for scripts and demos.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Mode = Literal["full", "stream"]


@dataclass(frozen=True)
class SynthesisRequest:
    request_id: str
    text: str
    voice: str
    language: str = "English"
    instructions: str = ""
    seed: int | None = None
    # Streaming knob shared by vLLM-Omni and SGLang-Omni. None = engine default.
    initial_codec_chunk_frames: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioChunk:
    pcm: np.ndarray  # float32 mono
    t_arrival: float  # seconds since submit
    index: int


@dataclass
class SynthesisResult:
    request_id: str
    mode: Mode
    ok: bool
    sample_rate: int
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    t_first_audio: float | None = None  # first non-empty chunk (full mode: == t_end)
    t_end: float | None = None
    chunk_timeline: list[tuple[float, int]] = field(default_factory=list)  # (t_arrival, n_samples)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def audio_duration_s(self) -> float:
        return float(self.audio.size) / self.sample_rate if self.sample_rate else 0.0


class EngineError(RuntimeError):
    pass


class Engine(ABC):
    """Base class. Subclasses implement ``_synthesize`` and optionally ``_stream``."""

    name: str = "base"
    #: True only if audio is produced incrementally *by the engine*. Engines
    #: without it fall back to one chunk at the end (and the bench skips them
    #: for streaming comparisons unless explicitly asked).
    supports_streaming: bool = False
    #: True if concurrency is realised by static batching in-process rather
    #: than independent concurrent requests.
    batch_concurrency: bool = False

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:  # load model / open HTTP client
        return None

    async def close(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        """Engine/version info recorded in the run's env snapshot."""
        return {"engine": self.name, "supports_streaming": self.supports_streaming}

    # -- to implement ------------------------------------------------------
    @abstractmethod
    async def _synthesize(self, req: SynthesisRequest) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Return (audio float32 mono, sample_rate, meta)."""

    async def _stream(self, req: SynthesisRequest) -> AsyncIterator[np.ndarray]:
        """Yield PCM float32 chunks as they become available. Default: one chunk."""
        audio, sr, _meta = await self._synthesize(req)
        self.sample_rate = sr
        yield audio

    async def _synthesize_batch(self, reqs: list[SynthesisRequest]) -> list[tuple[np.ndarray, int, dict[str, Any]]]:
        """Only used when ``batch_concurrency`` is True."""
        raise NotImplementedError

    # -- public API ----------------------------------------------------------
    async def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        return await self.run(req, "full")

    async def stream(self, req: SynthesisRequest) -> AsyncIterator[AudioChunk]:
        t0 = time.perf_counter()
        i = 0
        async for pcm in self._stream(req):
            yield AudioChunk(pcm=pcm, t_arrival=time.perf_counter() - t0, index=i)
            i += 1

    async def run(self, req: SynthesisRequest, mode: Mode) -> SynthesisResult:
        """Execute one request and build a uniform timeline. Never raises for
        generation errors: failures come back as ``ok=False`` so they are
        counted, not dropped."""
        t0 = time.perf_counter()
        try:
            if mode == "full":
                audio, sr, meta = await self._synthesize(req)
                t_end = time.perf_counter() - t0
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                ok = audio.size > 0 and bool(np.all(np.isfinite(audio)))
                return SynthesisResult(
                    request_id=req.request_id,
                    mode=mode,
                    ok=ok,
                    sample_rate=sr,
                    audio=audio,
                    t_first_audio=t_end if audio.size else None,
                    t_end=t_end,
                    chunk_timeline=[(t_end, int(audio.size))] if audio.size else [],
                    error=None if ok else "empty or non-finite audio",
                    meta=meta,
                )
            parts: list[np.ndarray] = []
            timeline: list[tuple[float, int]] = []
            t_first = None
            async for pcm in self._stream(req):
                t = time.perf_counter() - t0
                pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
                if pcm.size == 0:
                    continue
                if t_first is None:
                    t_first = t
                parts.append(pcm)
                timeline.append((t, int(pcm.size)))
            t_end = time.perf_counter() - t0
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            ok = audio.size > 0 and bool(np.all(np.isfinite(audio)))
            return SynthesisResult(
                request_id=req.request_id,
                mode=mode,
                ok=ok,
                sample_rate=self.sample_rate,
                audio=audio,
                t_first_audio=t_first,
                t_end=t_end,
                chunk_timeline=timeline,
                error=None if ok else "empty or non-finite audio",
                meta={"true_streaming": self.supports_streaming},
            )
        except Exception as e:  # noqa: BLE001 - failures are data
            return SynthesisResult(
                request_id=req.request_id,
                mode=mode,
                ok=False,
                sample_rate=self.sample_rate,
                t_end=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    async def run_batch(self, reqs: list[SynthesisRequest]) -> list[SynthesisResult]:
        """Static batch (full mode only). Every item shares the batch's end time."""
        t0 = time.perf_counter()
        try:
            outs = await self._synthesize_batch(reqs)
        except Exception as e:  # noqa: BLE001
            t = time.perf_counter() - t0
            return [
                SynthesisResult(r.request_id, "full", False, self.sample_rate, t_end=t, error=f"{type(e).__name__}: {e}")
                for r in reqs
            ]
        t_end = time.perf_counter() - t0
        results = []
        for r, (audio, sr, meta) in zip(reqs, outs, strict=True):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            ok = audio.size > 0 and bool(np.all(np.isfinite(audio)))
            results.append(
                SynthesisResult(
                    request_id=r.request_id,
                    mode="full",
                    ok=ok,
                    sample_rate=sr,
                    audio=audio,
                    t_first_audio=t_end if audio.size else None,
                    t_end=t_end,
                    chunk_timeline=[(t_end, int(audio.size))] if audio.size else [],
                    error=None if ok else "empty or non-finite audio",
                    meta={**meta, "batch_size": len(reqs)},
                )
            )
        return results

    # -- sync conveniences -----------------------------------------------------
    def synthesize_sync(self, req: SynthesisRequest) -> SynthesisResult:
        async def _go() -> SynthesisResult:
            await self.start()
            try:
                return await self.synthesize(req)
            finally:
                await self.close()

        return asyncio.run(_go())

    def stream_sync(self, req: SynthesisRequest) -> Iterator[AudioChunk]:
        """Collects the async stream in a background loop; yields as chunks arrive."""
        import queue
        import threading

        q: queue.Queue = queue.Queue()
        sentinel = object()

        def worker() -> None:
            async def _go() -> None:
                await self.start()
                try:
                    async for c in self.stream(req):
                        q.put(c)
                except Exception as e:  # noqa: BLE001
                    q.put(e)
                finally:
                    await self.close()
                    q.put(sentinel)

            asyncio.run(_go())

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        th.join()
