"""Deterministic fake engine so the full pipeline runs on CPU.

Models the *shape* of a codec-LM TTS server, not its numbers:

* audio is a deterministic function of (text, seed): one tone burst per word,
  with a silent first codec frame (mimicking the bootstrap frame) and pauses at
  punctuation;
* latency = prefill + frames x frame_time, where frame_time = frame_period x rtf;
* streaming emits chunks on a codec-frame schedule (default ramp 1 -> 2 -> 4,
  then ``chunk_frames``), each chunk paying a fixed decode overhead, so small
  chunks lower TTFA but cost total time;
* ``contention`` slows every active request by (1 + contention x (active-1)),
  so concurrency has a visible throughput/latency tradeoff.

Numbers produced by this engine mean nothing about real engines.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from .base import Engine, EngineError, SynthesisRequest

DEFAULT_RAMP = (1, 2, 4)


def _stable_int(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def fake_audio(text: str, seed: int, sr: int = 24000, frame_rate: float = 12.5) -> np.ndarray:
    """Deterministic speech-like signal; length rounded up to whole codec frames."""
    rng = np.random.default_rng(_stable_int(text, seed))
    frame = int(round(sr / frame_rate))
    words = text.split()
    pieces = [np.zeros(frame, dtype=np.float32)]  # silent bootstrap frame
    for w in words:
        dur = 0.18 + 0.045 * min(len(w), 12)
        n = int(dur * sr)
        t = np.arange(n, dtype=np.float32) / sr
        f0 = 110 + (_stable_int(w.lower()) % 120) + rng.uniform(-5, 5)
        env = np.sin(np.pi * np.linspace(0, 1, n, dtype=np.float32)) ** 2
        sig = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.08 * np.sin(2 * np.pi * 2.01 * f0 * t)
        pieces.append((env * sig).astype(np.float32))
        gap = 0.06 if w[-1:] not in ",;:.!?" else 0.22
        pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
    audio = np.concatenate(pieces)
    n_frames = math.ceil(audio.size / frame)
    out = np.zeros(n_frames * frame, dtype=np.float32)
    out[: audio.size] = audio
    return out


class FakeEngine(Engine):
    name = "fake"
    supports_streaming = True

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_rate: float = 12.5,
        prefill_ms: float = 120.0,
        rtf: float = 0.25,
        chunk_frames: int = 12,
        chunk_overhead_ms: float = 15.0,
        contention: float = 0.35,
        jitter_ms: float = 5.0,
        fail_request_ids: list[str] | None = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(sample_rate)
        self.frame_rate = frame_rate
        self.prefill_s = prefill_ms / 1000
        self.rtf = rtf
        self.chunk_frames = max(1, int(chunk_frames))
        self.chunk_overhead_s = chunk_overhead_ms / 1000
        self.contention = contention
        self.jitter_s = jitter_ms / 1000
        self.fail_request_ids = set(fail_request_ids or [])
        self._active = 0
        self.frame_samples = int(round(sample_rate / frame_rate))

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "prefill_s": self.prefill_s,
            "rtf": self.rtf,
            "chunk_frames": self.chunk_frames,
            "chunk_overhead_s": self.chunk_overhead_s,
            "contention": self.contention,
        }

    # -- helpers ---------------------------------------------------------------
    def _slowdown(self) -> float:
        return 1.0 + self.contention * max(0, self._active - 1)

    def _frame_time(self) -> float:
        return (1.0 / self.frame_rate) * self.rtf * self._slowdown()

    def _jitter(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0, self.jitter_s)) if self.jitter_s > 0 else 0.0

    def schedule(self, n_frames: int, initial: int | None) -> list[int]:
        """Chunk sizes in frames. initial=None -> default ramp; 0 -> steady from start."""
        if initial is None:
            head = list(DEFAULT_RAMP)
        elif initial <= 0:
            head = []
        else:
            head = [initial]
        sizes, left = [], n_frames
        for s in head:
            if left <= 0:
                break
            s = min(s, left)
            sizes.append(s)
            left -= s
        while left > 0:
            s = min(self.chunk_frames, left)
            sizes.append(s)
            left -= s
        return sizes

    def _check_fail(self, req: SynthesisRequest) -> None:
        if req.request_id in self.fail_request_ids or req.extra.get("prompt_id") in self.fail_request_ids:
            raise EngineError(f"injected failure for {req.request_id}")

    # -- engine API --------------------------------------------------------------
    async def _synthesize(self, req: SynthesisRequest) -> tuple[np.ndarray, int, dict[str, Any]]:
        self._active += 1
        try:
            self._check_fail(req)
            rng = np.random.default_rng(_stable_int(req.request_id, req.seed))
            audio = fake_audio(req.text, req.seed or 0, self.sample_rate, self.frame_rate)
            n_frames = audio.size // self.frame_samples
            await asyncio.sleep(self.prefill_s * self._slowdown() + self._jitter(rng))
            # Generate frames in slices so concurrent requests interleave.
            step = 8
            for i in range(0, n_frames, step):
                await asyncio.sleep(self._frame_time() * min(step, n_frames - i))
            await asyncio.sleep(self.chunk_overhead_s)  # one whole-utterance decode
            return audio, self.sample_rate, {"n_frames": n_frames}
        finally:
            self._active -= 1

    async def _stream(self, req: SynthesisRequest) -> AsyncIterator[np.ndarray]:
        self._active += 1
        try:
            self._check_fail(req)
            rng = np.random.default_rng(_stable_int(req.request_id, req.seed))
            audio = fake_audio(req.text, req.seed or 0, self.sample_rate, self.frame_rate)
            n_frames = audio.size // self.frame_samples
            await asyncio.sleep(self.prefill_s * self._slowdown() + self._jitter(rng))
            pos = 0
            for size in self.schedule(n_frames, req.initial_codec_chunk_frames):
                await asyncio.sleep(self._frame_time() * size + self.chunk_overhead_s + self._jitter(rng))
                end = pos + size * self.frame_samples
                yield audio[pos:end]
                pos = end
        finally:
            self._active -= 1

    async def _synthesize_batch(self, reqs: list[SynthesisRequest]) -> list[tuple[np.ndarray, int, dict[str, Any]]]:
        return list(await asyncio.gather(*(self._synthesize(r) for r in reqs)))
