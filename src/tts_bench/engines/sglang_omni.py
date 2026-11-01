"""SGLang-Omni adapter (HTTP client to ``sgl-omni serve``).

Request fields from docs/basic_usage/tts.md and docs/cookbook/qwen3_tts.md at
sglang-omni b880acf8 (2026-10-13):

* full:   {"input", "voice", "language", "task_type": "CustomVoice", "response_format": "wav",
           "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed"}
* stream: + "stream": true, "response_format": "pcm" (stream_format defaults to "audio")
          -> raw 16-bit mono PCM; sample rate in the ``X-Sample-Rate`` header
          (verified in sglang_omni/serve/openai_api.py at b880acf8).
* ``initial_codec_chunk_frames``: omitted -> 1 -> 2 -> 4 ramp; 0 -> steady stride.
* ``suppress_bootstrap_silence`` (default true server-side) withholds ~80 ms of
  leading silence for Ryan/English streams. We send false by default so streamed
  and full outputs from all engines stay sample-comparable; flip it in the config
  to measure the server's default behaviour.
"""

from __future__ import annotations

from typing import Any

from .base import SynthesisRequest
from .http_speech import OpenAISpeechEngine


class SglangOmniEngine(OpenAISpeechEngine):
    name = "sglang_omni"
    sample_rate_headers = ("x-sample-rate",)

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        sampling: dict[str, Any] | None = None,
        max_new_tokens: int = 2048,
        send_seed: bool = True,
        send_sampling: bool = True,
        suppress_bootstrap_silence: bool = False,
        **kw: Any,
    ) -> None:
        super().__init__(base_url=base_url, **kw)
        self.sampling = sampling or {}
        self.max_new_tokens = max_new_tokens
        self.send_seed = send_seed
        self.send_sampling = send_sampling
        self.suppress_bootstrap_silence = suppress_bootstrap_silence

    def build_payload(self, req: SynthesisRequest, stream: bool) -> dict[str, Any]:
        p: dict[str, Any] = {
            "input": req.text,
            "voice": req.voice,  # documented as case-insensitive
            "language": req.language,
            "task_type": "CustomVoice",
            "max_new_tokens": self.max_new_tokens,
        }
        if req.instructions:
            p["instructions"] = req.instructions
        if self.send_sampling:
            for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
                if k in self.sampling:
                    p[k] = self.sampling[k]
            # Subtalker (code predictor) sampling is not a request field; the server
            # takes it from the checkpoint generation_config (0.9 / top_k 50 / top_p 1.0,
            # same as ours) with identical fallbacks (request_builders.py @ b880acf8).
            # Note: request_builders treats top_p=1.0 as an "implicit default" and
            # falls back to the checkpoint value, which is also 1.0.
        if self.send_seed and req.seed is not None:
            # tts.md only says "Qwen3-TTS Base accepts request-scoped seed"; the
            # request builder reads `seed` for every Qwen3-TTS task and splits it
            # into semantic/subtalker seeds (request_builders.py @ b880acf8).
            # Confirm on the box that two identical seeded requests match.
            p["seed"] = req.seed
        if stream:
            p.update(stream=True, response_format="pcm", suppress_bootstrap_silence=self.suppress_bootstrap_silence)
            if req.initial_codec_chunk_frames is not None:
                p["initial_codec_chunk_frames"] = req.initial_codec_chunk_frames
        else:
            p["response_format"] = "wav"
        return p
