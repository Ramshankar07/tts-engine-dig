"""Report generation, bootstrap CIs and the streaming comparison (fake engine)."""

import asyncio
import json

import numpy as np
import pytest

from tts_bench.bench import run_bench
from tts_bench.engines.fake import FakeEngine
from tts_bench.quality import evaluate_run
from tts_bench.report import build_report, write_report
from tts_bench.stats import bootstrap_ci, bootstrap_ratio_ci, p50, paired_bootstrap_ratio_diff
from tts_bench.streaming import compare_streaming, run_streaming_comparison

quiet = dict(log=lambda *_: None)


def test_bootstrap_is_deterministic_and_brackets_point():
    x = np.arange(1, 101, dtype=float)
    a = bootstrap_ci(x, p50, n_boot=500, seed=3)
    b = bootstrap_ci(x, p50, n_boot=500, seed=3)
    assert a == b
    assert a.lo <= a.value <= a.hi and a.value == pytest.approx(50.5)
    r = bootstrap_ratio_ci([1, 2, 0], [6, 3, 5], n_boot=300)
    assert r.value == pytest.approx(3 / 14)
    d = paired_bootstrap_ratio_diff([1, 1], [1, 1], [5, 5], n_boot=100)
    assert d.value == 0 and d.lo == 0 and d.hi == 0
    assert bootstrap_ci([], p50).n == 0


def test_report_tables(fake_cfg, tmp_path):
    d1 = asyncio.run(run_bench(fake_cfg, "fake", modes=["full", "stream"], concurrency=[1, 4], **quiet))
    other = FakeEngine(prefill_ms=15, rtf=0.02, chunk_overhead_ms=1, jitter_ms=0)
    other.name = "fake_b"
    d2 = asyncio.run(run_bench(fake_cfg, "fake_b", modes=["full"], concurrency=[1], engine=other, **quiet))
    for d in (d1, d2):
        evaluate_run(d, fake_cfg, asr_backend="fake", mos_backend="none", **quiet)

    md, df = build_report([d1, d2], n_boot=200, baseline_engine="fake")
    assert "## Latency, throughput, memory" in md and "## Quality" in md
    assert "tts-bench-norm-v1" in md
    assert "Paired WER difference vs `fake`" in md
    assert f"| fake_b | {d2.name[:15]} | full-c1 |" in md
    assert set(df["cell"]) >= {"full-c1", "full-c4", "stream-c1-kdefault"}
    assert {"ttfa_p50", "ttfa_p50_lo", "ttfa_p50_hi", "rtf_p50", "wer"} <= set(df.columns)
    # same fake ASR error pattern on the same prompts -> zero paired difference
    assert "0.00 [0.00, 0.00]" in md

    out = write_report([d1, d2], tmp_path / "rep" / "p1.md", n_boot=100, baseline_engine="fake")
    assert out.exists() and out.with_suffix(".csv").exists()


def test_streaming_comparison_end_to_end(fake_cfg):
    d = asyncio.run(run_streaming_comparison(fake_cfg, "fake", concurrency=[1], **quiet))
    md = (d / "streaming_comparison.md").read_text()
    res = json.loads((d / "streaming_comparison.json").read_text())
    entries = res["by_concurrency"]["1"]["entries"]
    settings = [e["setting"] for e in entries]
    assert settings[0] == "full"
    assert "stream (engine default)" in settings and "stream (initial 1 frames)" in settings and "stream (initial 4 frames)" in settings
    assert all(e["verdict"] in {"baseline", "helps", "mixed", "no gain"} for e in entries)
    assert all(e["ttfa_p50_s"] < entries[0]["ttfa_p50_s"] for e in entries[1:])
    assert "**Conclusion:**" in md


def _row(cell, mode, pid, ttfa, gen, stalls=0, chunk=None):
    return {"run_id": "x", "engine": "e", "cell": cell, "mode": mode, "concurrency": 1, "chunk": chunk,
            "prompt_id": pid, "repeat": 0, "ok": True, "ttfa_s": ttfa, "ttfa_audible_s": ttfa, "gen_time_s": gen,
            "stall_total_ms": 100.0 * stalls, "stall_max_ms": 100.0 * stalls, "stalls_over_threshold": stalls}


@pytest.mark.parametrize(
    "s_ttfa,s_gen,stalls,expected",
    [
        (0.2, 1.1, 0, "helps"),     # 80% TTFA cut, 10% overhead, no stalls
        (0.2, 1.5, 0, "mixed"),     # overhead 50% > 20%
        (0.2, 1.0, 1, "mixed"),     # every request stalls
        (0.8, 1.0, 0, "no gain"),   # only 20% TTFA cut
    ],
)
def test_streaming_verdict_rules(fake_cfg, tmp_path, s_ttfa, s_gen, stalls, expected):
    d = tmp_path / "run"
    d.mkdir()
    rows = []
    for i in range(10):
        rows.append(_row("full-c1", "full", f"p{i}", 1.0, 1.0))
        rows.append(_row("stream-c1-k1", "stream", f"p{i}", s_ttfa, s_gen, stalls, chunk=1))
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    res = compare_streaming(d, fake_cfg)
    entry = res["by_concurrency"]["1"]["entries"][1]
    assert entry["verdict"] == expected
    concl = res["by_concurrency"]["1"]["conclusion"]
    assert ("helps" in concl) == (expected == "helps")
