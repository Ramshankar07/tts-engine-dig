"""End-to-end benchmark with the fake engine at concurrency 1 and 4 (CPU only)."""

import asyncio
import json

import pytest

from tts_bench.bench import load_rows, run_bench
from tts_bench.engines.fake import FakeEngine
from tts_bench.quality import evaluate_run, load_quality

ROW_KEYS = {
    "run_id", "engine", "cell", "mode", "concurrency", "chunk", "prompt_id", "text", "length_bucket", "category",
    "seed", "ok", "error", "audio_duration_s", "ttfa_s", "ttfa_audible_s", "gen_time_s", "rtf", "n_chunks",
    "chunk_timeline", "underrun_count", "stall_total_ms", "stall_max_ms", "stalls_over_threshold",
    "peak_gpu_mem_mb", "audio_path",
}


@pytest.fixture
def run_dir(fake_cfg):
    return asyncio.run(run_bench(fake_cfg, "fake", modes=["full", "stream"], concurrency=[1, 4], log=lambda *_: None))


def test_run_directory_layout(run_dir):
    for f in ("config.json", "env.json", "results.jsonl", "cells.json"):
        assert (run_dir / f).exists(), f
    env = json.loads((run_dir / "env.json").read_text())
    assert {"git", "packages", "gpu", "python", "engine"} <= set(env)
    assert any(p.lower().startswith("numpy==") for p in env["packages"])
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["model"]["id"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert len(cfg["prompts_sha256"]) == 64
    assert run_dir.name.endswith("-fake")


def test_rows_cover_every_cell(run_dir):
    rows = load_rows(run_dir)
    cells = {r["cell"] for r in rows}
    assert cells == {"full-c1", "full-c4", "stream-c1-kdefault", "stream-c4-kdefault"}
    assert len(rows) == 4 * 6
    for r in rows:
        assert ROW_KEYS <= set(r)
        assert r["ok"], r["error"]
        assert r["audio_duration_s"] > 0
        assert r["rtf"] == pytest.approx(r["gen_time_s"] / r["audio_duration_s"])
        assert (run_dir / r["audio_path"]).exists()
    # same seed for the same prompt in every cell (paired comparisons)
    seeds = {}
    for r in rows:
        seeds.setdefault(r["prompt_id"], set()).add(r["seed"])
    assert all(len(s) == 1 for s in seeds.values())


def test_streaming_reaches_first_audio_earlier(run_dir):
    rows = load_rows(run_dir)
    for c in (1, 4):
        full = {r["prompt_id"]: r for r in rows if r["cell"] == f"full-c{c}"}
        st = {r["prompt_id"]: r for r in rows if r["cell"] == f"stream-c{c}-kdefault"}
        assert all(st[p]["ttfa_s"] < full[p]["ttfa_s"] for p in full)
        assert all(full[p]["n_chunks"] == 1 for p in full)
        assert all(st[p]["n_chunks"] > 1 for p in st)
        # audible TTFA skips the silent bootstrap chunk
        assert all(st[p]["ttfa_audible_s"] >= st[p]["ttfa_s"] for p in st)


def test_concurrency_summary(run_dir):
    cells = {c["cell"]: c for c in json.loads((run_dir / "cells.json").read_text())["cells"]}
    for k in ("full-c1", "full-c4"):
        assert cells[k]["n_requests"] == 6 and cells[k]["n_failed"] == 0
        assert cells[k]["throughput_audio_s_per_s"] > 0
    # 4 workers finish the same work faster than 1 (fake contention < 1)
    assert cells["full-c4"]["wall_time_s"] < cells["full-c1"]["wall_time_s"]


def test_failures_counted_not_dropped(fake_cfg):
    eng = FakeEngine(prefill_ms=5, rtf=0.01, chunk_overhead_ms=1, jitter_ms=0, fail_request_ids=["p1-002"])
    d = asyncio.run(run_bench(fake_cfg, "fake", modes=["full"], concurrency=[1], engine=eng, log=lambda *_: None))
    rows = load_rows(d)
    assert len(rows) == 6
    failed = [r for r in rows if not r["ok"]]
    assert [r["prompt_id"] for r in failed] == ["p1-002"]
    assert failed[0]["audio_path"] is None and "injected failure" in failed[0]["error"]
    cells = json.loads((d / "cells.json").read_text())["cells"]
    assert cells[0]["n_failed"] == 1

    # quality: the failure is scored as all-deletions and kept in the corpus WER
    summary = evaluate_run(d, fake_cfg, asr_backend="fake", mos_backend="none", log=lambda *_: None)
    q = {x["prompt_id"]: x for x in load_quality(d)}
    assert q["p1-002"]["failed_synthesis"] and q["p1-002"]["D"] == q["p1-002"]["N"]
    total_err = sum(x["S"] + x["D"] + x["I"] for x in q.values())
    total_ref = sum(x["N"] for x in q.values())
    assert summary["cells"]["full-c1"]["corpus_wer"]["value"] == pytest.approx(total_err / total_ref)
    assert summary["cells"]["full-c1"]["n_failed_synthesis"] == 1
    assert summary["mos"] == {"skipped": "disabled"}


def test_quality_skips_gracefully_when_whisper_missing(fake_cfg, run_dir, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name in ("torch", "transformers", "utmosv2"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    s = evaluate_run(run_dir, fake_cfg, asr_backend="whisper", mos_backend="utmosv2", log=lambda *_: None)
    assert "skipped" in s["asr"] and "skipped" in s["mos"]
    assert (run_dir / "quality.jsonl").exists()
