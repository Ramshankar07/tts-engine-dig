"""Shared HTTP client for OpenAI-compatible ``POST /v1/audio/speech`` servers.

Both vLLM-Omni and SGLang-Omni expose this endpoint (docs read at the SHAs in
docs/SOURCES.md). Full mode requests a WAV body; stream mode requests raw
16-bit little-endian mono PCM bytes and records the arrival time of every
network read, which is exactly what a real client's playback buffer sees.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np

from ..audio import Pcm16Assembler, decode_audio_bytes
from .base import Engine, EngineError, SynthesisRequest

SPEECH_PATH = "/v1/audio/speech"


class OpenAISpeechEngine(Engine):
    supports_streaming = True
    #: response headers that may carry the stream sample rate (case-insensitive)
    sample_rate_headers: tuple[str, ...] = ("x-audio-sample-rate", "x-sample-rate")

    def __init__(
        self,
        base_url: str,
        sample_rate: int = 24000,
        timeout_s: float = 300.0,
        max_connections: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject httpx.MockTransport
        **settings: Any,
    ) -> None:
        super().__init__(sample_rate)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self.settings = settings
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._server_info: dict[str, Any] = {}

    # -- payloads (engine-specific) ----------------------------------------------
    def build_payload(self, req: SynthesisRequest, stream: bool) -> dict[str, Any]:
        raise NotImplementedError

    # -- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        if self._client is None:
            limits = httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=self.max_connections)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
                limits=limits,
                transport=self._transport,
            )
            self._server_info = await self._probe()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _probe(self) -> dict[str, Any]:
        """Best-effort server metadata for the env snapshot. Never fails the run."""
        info: dict[str, Any] = {"base_url": self.base_url}
        assert self._client is not None
        for path in ("/health", "/version", "/v1/models", "/v1/audio/voices"):
            try:
                r = await self._client.get(path, timeout=5.0)
                ctype = r.headers.get("content-type", "")
                body: Any = r.json() if "json" in ctype else r.text[:500]
                info[path] = {"status": r.status_code, "body": body}
            except Exception as e:  # noqa: BLE001
                info[path] = {"error": f"{type(e).__name__}: {e}"}
        return info

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "server": self._server_info, "settings": self.settings}

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            raise EngineError("engine not started; call await engine.start()")
        return self._client

    def _sr_from_headers(self, headers: httpx.Headers) -> int:
        for h in self.sample_rate_headers:
            v = headers.get(h)
            if v:
                try:
                    return int(v)
                except ValueError:
                    pass
        return self.sample_rate

    # -- generation ----------------------------------------------------------------
    async def _synthesize(self, req: SynthesisRequest) -> tuple[np.ndarray, int, dict[str, Any]]:
        client = self._client_or_raise()
        payload = self.build_payload(req, stream=False)
        r = await client.post(SPEECH_PATH, json=payload)
        if r.status_code != 200:
            raise EngineError(f"HTTP {r.status_code}: {r.text[:300]}")
        audio, sr = decode_audio_bytes(r.content)
        meta = {k: v for k, v in r.headers.items() if k.lower().startswith(("x-vllm-omni", "x-sglang-omni", "x-audio"))}
        return audio, sr, meta

    async def _stream(self, req: SynthesisRequest) -> AsyncIterator[np.ndarray]:
        client = self._client_or_raise()
        payload = self.build_payload(req, stream=True)
        asm = Pcm16Assembler()
        async with client.stream("POST", SPEECH_PATH, json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")
                raise EngineError(f"HTTP {r.status_code}: {body[:300]}")
            self.sample_rate = self._sr_from_headers(r.headers)
            async for data in r.aiter_bytes():  # yields each network read (decoded if gzip)
                pcm = asm.push(data)
                if pcm.size:
                    yield pcm
        if asm.leftover:
            raise EngineError("stream ended mid-sample (odd byte count)")
