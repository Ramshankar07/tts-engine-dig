"""Official engine wrapper logic against stub `torch` / `qwen_tts` modules.

The real package needs a GPU; this checks our adapter's call shape (kwargs
forwarded to generate_custom_voice, batching, LM/codec timing hooks, sdpa
fallback) using stubs that mirror qwen_tts @ 022e286b's structure:
generate_custom_voice -> self.model.generate(...) -> self.model.speech_tokenizer.decode(...).
"""

import asyncio
import contextlib
import sys
import types

import numpy as np
import pytest

from tts_bench.engines import SynthesisRequest, build_engine


@pytest.fixture
def stubs(monkeypatch):
    calls: dict = {}

    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
    torch.manual_seed = lambda s: calls.setdefault("seeds", []).append(s)
    torch.profiler = types.SimpleNamespace(record_function=lambda name: contextlib.nullcontext())

    class Tokenizer:
        def decode(self, items):
            return [np.full(1920 * len(c["audio_codes"]), 0.1, dtype=np.float32) for c in items], 24000

    class Inner:
        def __init__(self):
            self.speech_tokenizer = Tokenizer()

        def generate(self, **kw):
            calls["generate"] = kw
            return [[0] * (3 + i) for i in range(len(kw["input_ids"]))], None

    class Qwen3TTSModel:
        def __init__(self):
            self.model = Inner()

        @classmethod
        def from_pretrained(cls, path, **kw):
            calls["from_pretrained"] = (path, kw)
            return cls()

        def generate_custom_voice(self, text, speaker, language=None, instruct=None, non_streaming_mode=True, **kw):
            texts = text if isinstance(text, list) else [text]
            calls["gcv"] = dict(text=text, speaker=speaker, language=language, instruct=instruct, kw=kw)
            codes, _ = self.model.generate(input_ids=texts, **kw)
            return self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes])

    qwen_tts = types.ModuleType("qwen_tts")
    qwen_tts.Qwen3TTSModel = Qwen3TTSModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts)
    return calls


def _req(i, text="Hello there."):
    return SynthesisRequest(request_id=f"p{i}", text=text, voice="Ryan", language="English", seed=100 + i)


def test_official_engine_single_and_batch(stubs, fake_cfg):
    eng = build_engine("official", fake_cfg)

    async def go():
        await eng.start()
        one = await eng.run(_req(0), "full")
        batch = await eng.run_batch([_req(1), _req(2, "Second one.")])
        return one, batch

    one, batch = asyncio.run(go())
    path, kw = stubs["from_pretrained"]
    assert path == fake_cfg.model.path_or_id
    assert kw == {"device_map": "cuda:0", "dtype": "bf16", "attn_implementation": "sdpa"}  # no flash_attn locally
    assert one.ok and one.sample_rate == 24000 and one.audio.size == 3 * 1920
    assert one.meta["lm_generate_s"] is not None and one.meta["codec_decode_s"] is not None
    gcv = stubs["gcv"]
    assert gcv["text"] == ["Hello there.", "Second one."] and gcv["speaker"] == ["Ryan", "Ryan"]
    assert gcv["instruct"] is None
    s = fake_cfg.sampling
    assert gcv["kw"] == {
        "do_sample": s.do_sample, "top_k": s.top_k, "top_p": s.top_p, "temperature": s.temperature,
        "repetition_penalty": s.repetition_penalty, "subtalker_dosample": s.subtalker_dosample,
        "subtalker_top_k": s.subtalker_top_k, "subtalker_top_p": s.subtalker_top_p,
        "subtalker_temperature": s.subtalker_temperature, "max_new_tokens": s.max_new_tokens,
    }
    assert [r.ok for r in batch] == [True, True]
    assert batch[0].t_end == batch[1].t_end and batch[0].meta["batch_size"] == 2
    assert stubs["seeds"] == [100, 101]


def test_official_engine_falls_back_for_stream(stubs, fake_cfg):
    eng = build_engine("official", fake_cfg)
    asyncio.run(eng.start())
    res = asyncio.run(eng.run(_req(0), "stream"))
    assert res.ok and len(res.chunk_timeline) == 1 and res.meta["true_streaming"] is False
