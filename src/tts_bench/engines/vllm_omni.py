"""vLLM-Omni adapter (HTTP client to ``vllm serve <model> --omni``).

Request fields come from docs/serving/speech_api.md and
recipes/Qwen/Qwen3-TTS.md at vllm-omni fa8b4199 (2026-10-13):

* full:   {"input", "voice", "language", "task_type", "response_format": "wav", "max_new_tokens", "seed"}
* stream: + "stream": true, "stream_format": "audio", "response_format": "pcm"
          -> raw 16-bit LE PCM, 24 kHz mono, sample rate in ``x-audio-sample-rate``.
* ``initial_codec_chunk_frames`` is a documented per-request field (first chunk
  size). The steady chunk (``codec_chunk_frames: 25``) is a deploy-YAML setting
  documented as "must match the decoder sliding attention window", so it is not
  swept here.

Sampling: the speech API table documents no per-request temperature/top_k for
Qwen3-TTS; the server uses ``default_sampling_params`` from the deploy YAML
(temperature 0.9, top_k 50, repetition_penalty 1.05, subtalker 0.9/50/1.0),
which equal configs/p1_run.toml. Runs record the server's reported config.
"""

from __future__ import annotations

from typing import Any

from .base import SynthesisRequest
from .http_speech import OpenAISpeechEngine


class VllmOmniEngine(OpenAISpeechEngine):
    name = "vllm_omni"
    sample_rate_headers = ("x-audio-sample-rate",)

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8091",
        max_new_tokens: int = 2048,
        send_seed: bool = True,
        lowercase_voice: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(base_url=base_url, **kw)
        self.max_new_tokens = max_new_tokens
        self.send_seed = send_seed
        # Docs use lowercase preset names ("vivian", "ryan"); the server also
        # lowercases `voice` (tts_adapters/qwen3_tts.py @ fa8b4199), so this is cosmetic.
        self.lowercase_voice = lowercase_voice

    def build_payload(self, req: SynthesisRequest, stream: bool) -> dict[str, Any]:
        p: dict[str, Any] = {
            "input": req.text,
            "voice": req.voice.lower() if self.lowercase_voice else req.voice,
            "language": req.language,
            "task_type": "CustomVoice",
            # Explicit max_new_tokens also disables the server's silent retry on
            # runaway generation (recipe "Codec termination" note) - we want
            # failures visible, so this is intended.
            "max_new_tokens": self.max_new_tokens,
        }
        if req.instructions:
            p["instructions"] = req.instructions
        if self.send_seed and req.seed is not None:
            # Not in the speech_api.md table, but a field of the speech request
            # model (entrypoints/openai/protocol/audio.py @ fa8b4199) and applied by
            # the serving layer. Seeds make runs repeatable per engine; they do not
            # make different engines produce identical samples.
            p["seed"] = req.seed
        if stream:
            p.update(stream=True, stream_format="audio", response_format="pcm")
            if req.initial_codec_chunk_frames is not None:
                p["initial_codec_chunk_frames"] = req.initial_codec_chunk_frames
        else:
            p["response_format"] = "wav"
        return p
