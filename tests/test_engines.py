"""Fake engine behaviour + HTTP adapters against a mocked server."""

import asyncio
import io
import json

import httpx
import numpy as np
import soundfile as sf

from tts_bench.audio import float_to_pcm16_bytes
from tts_bench.engines import SynthesisRequest, build_engine
from tts_bench.engines.fake import FakeEngine, fake_audio
from tts_bench.engines.sglang_omni import SglangOmniEngine
from tts_bench.engines.vllm_omni import VllmOmniEngine


def req(**kw):
    base = dict(request_id="p1-001-r0", text="Hello there, general.", voice="Ryan", language="English", seed=7)
    base.update(kw)
    return SynthesisRequest(**base)


def test_fake_audio_deterministic_and_frame_aligned():
    a = fake_audio("hello world", 1)
    b = fake_audio("hello world", 1)
    np.testing.assert_array_equal(a, b)
    assert a.size % 1920 == 0
    assert np.all(a[:1920] == 0)  # silent bootstrap frame
    assert not np.array_equal(a, fake_audio("hello world", 2))


def test_fake_schedule():
    e = FakeEngine(chunk_frames=10)
    assert e.schedule(30, None) == [1, 2, 4, 10, 10, 3]
    assert e.schedule(30, 0) == [10, 10, 10]
    assert e.schedule(30, 5) == [5, 10, 10, 5]
    assert e.schedule(2, None) == [1, 1]


def test_fake_stream_equals_full_and_streams_earlier():
    e = FakeEngine(prefill_ms=5, rtf=0.05, chunk_overhead_ms=1, jitter_ms=0)

    async def go():
        full = await e.run(req(), "full")
        st = await e.run(req(), "stream")
        return full, st

    full, st = asyncio.run(go())
    assert full.ok and st.ok
    np.testing.assert_array_equal(full.audio, st.audio)
    assert st.t_first_audio < full.t_first_audio
    assert len(st.chunk_timeline) > 1 and len(full.chunk_timeline) == 1


def test_failure_is_returned_not_raised():
    e = FakeEngine(prefill_ms=1, rtf=0.01, fail_request_ids=["p1-001-r0"])
    res = asyncio.run(e.run(req(), "stream"))
    assert not res.ok and "injected failure" in res.error


def test_build_engine_registry(fake_cfg):
    for name in ("fake", "vllm_omni", "sglang_omni", "official"):
        eng = build_engine(name, fake_cfg)  # official must not import torch at build time
        assert eng.name == name
    assert build_engine("official", fake_cfg).supports_streaming is False


# ---------------------------------------------------------------- HTTP adapters
def _mock_server(seen: list):
    audio = (0.1 * np.sin(np.linspace(0, 200, 4800))).astype(np.float32)
    pcm = float_to_pcm16_bytes(audio)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        seen.append(body)
        if body.get("stream"):
            return httpx.Response(200, content=pcm, headers={"x-audio-sample-rate": "24000", "X-Sample-Rate": "24000"})
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
        return httpx.Response(200, content=buf.getvalue(), headers={"content-type": "audio/wav"})

    return httpx.MockTransport(handler), audio


def test_vllm_payloads_and_roundtrip():
    seen: list = []
    transport, audio = _mock_server(seen)
    e = VllmOmniEngine(base_url="http://x", transport=transport, max_new_tokens=2048)

    async def go():
        await e.start()
        try:
            return await e.run(req(initial_codec_chunk_frames=4), "full"), await e.run(req(initial_codec_chunk_frames=4), "stream")
        finally:
            await e.close()

    full, st = asyncio.run(go())
    assert full.ok and st.ok
    assert full.audio.size == audio.size == st.audio.size
    f, s = seen
    assert f == {"input": "Hello there, general.", "voice": "ryan", "language": "English", "task_type": "CustomVoice",
                 "max_new_tokens": 2048, "seed": 7, "response_format": "wav"}
    assert s["stream"] is True and s["stream_format"] == "audio" and s["response_format"] == "pcm"
    assert s["initial_codec_chunk_frames"] == 4
    assert "initial_codec_chunk_frames" not in f


def test_sglang_payloads():
    e = SglangOmniEngine(sampling={"temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05})
    f = e.build_payload(req(), stream=False)
    s = e.build_payload(req(initial_codec_chunk_frames=0), stream=True)
    assert f["voice"] == "Ryan" and f["response_format"] == "wav" and f["temperature"] == 0.9 and f["top_k"] == 50
    assert s["stream"] is True and s["response_format"] == "pcm" and "stream_format" not in s
    assert s["initial_codec_chunk_frames"] == 0 and s["suppress_bootstrap_silence"] is False
    assert "initial_codec_chunk_frames" not in e.build_payload(req(), stream=True)


def test_http_error_becomes_failed_result():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(400, json={"error": {"message": "bad voice"}})

    e = SglangOmniEngine(base_url="http://x", transport=httpx.MockTransport(handler))

    async def go():
        await e.start()
        try:
            return await e.run(req(), "stream")
        finally:
            await e.close()

    res = asyncio.run(go())
    assert not res.ok and "HTTP 400" in res.error
