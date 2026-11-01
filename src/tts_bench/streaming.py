"""Streaming vs full-output comparison for the same texts, across chunk sizes.

For each engine that supports true streaming:
  1. run the prompt set in full mode and in stream mode for every
     ``initial_codec_chunk_frames`` setting (same text, same seed per prompt);
  2. compare each stream setting against full mode on paired prompts:
     TTFA, audible TTFA, total generation time overhead, playback stalls,
     and (if quality.jsonl exists) corpus WER and predicted MOS;
  3. label each setting with stated thresholds (configs: [streaming.verdict]):
       helps     - TTFA p50 drops by >= min_ttfa_reduction AND requests with an
                   audible stall <= max_underrun_request_frac AND paired median
                   total-time overhead <= max_total_time_overhead
       mixed     - TTFA criterion met but stalls or overhead exceed limits
       no gain   - TTFA criterion not met
A negative result ("no setting helps") is a valid outcome and is reported as such.
Quality deltas are reported, not folded into the verdict: with one sample per
prompt they are noisy and need listening checks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .bench import chunk_value, load_rows, run_bench
from .config import BenchConfig
from .engines import Engine
from .quality import load_quality
from .stats import bootstrap_ratio_ci


def _p50(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def compare_streaming(run_dir: str | Path, cfg: BenchConfig) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    quality = {(q["cell"], q["prompt_id"], q["repeat"]): q for q in load_quality(run_dir)}
    v = cfg.streaming.verdict
    by_conc: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_conc[r["concurrency"]][r["cell"]].append(r)

    results: dict[str, Any] = {"run_id": run_dir.name, "thresholds": v.model_dump(), "by_concurrency": {}}
    for conc, cells in sorted(by_conc.items()):
        full_key = f"full-c{conc}"
        if full_key not in cells:
            continue
        full = {(r["prompt_id"], r["repeat"]): r for r in cells[full_key]}
        full_ok = [r for r in full.values() if r["ok"]]
        full_ttfa = _p50([r["ttfa_s"] for r in full_ok])
        full_gen = _p50([r["gen_time_s"] for r in full_ok])
        entries = [_quality_entry({"setting": "full", "cell": full_key, "n": len(full), "failed": len(full) - len(full_ok),
                                   "ttfa_p50_s": full_ttfa, "ttfa_audible_p50_s": _p50([r.get("ttfa_audible_s") for r in full_ok]),
                                   "gen_p50_s": full_gen, "overhead_p50": 0.0, "underrun_req_frac": 0.0,
                                   "stall_ms_mean": 0.0, "stall_ms_max": 0.0, "verdict": "baseline"},
                                  cells[full_key], quality)]
        for key, rs in cells.items():
            if not key.startswith("stream"):
                continue
            ok = [r for r in rs if r["ok"]]
            pairs = [(r, full[(r["prompt_id"], r["repeat"])]) for r in ok if (r["prompt_id"], r["repeat"]) in full and full[(r["prompt_id"], r["repeat"])]["ok"]]
            overhead = _p50([s["gen_time_s"] / f["gen_time_s"] - 1.0 for s, f in pairs if f["gen_time_s"]])
            ttfa = _p50([r["ttfa_s"] for r in ok])
            under_frac = (sum(1 for r in ok if r["stalls_over_threshold"] > 0) / len(ok)) if ok else float("nan")
            reduction = 1.0 - ttfa / full_ttfa if full_ttfa and full_ttfa == full_ttfa else float("nan")
            if not (reduction >= v.min_ttfa_reduction):
                verdict = "no gain"
            elif under_frac <= v.max_underrun_request_frac and overhead <= v.max_total_time_overhead:
                verdict = "helps"
            else:
                verdict = "mixed"
            chunk = rs[0]["chunk"]
            e = {
                "setting": "stream (engine default)" if chunk is None else f"stream (initial {chunk} frames)",
                "cell": key,
                "n": len(rs),
                "failed": len(rs) - len(ok),
                "ttfa_p50_s": ttfa,
                "ttfa_reduction": reduction,
                "ttfa_audible_p50_s": _p50([r.get("ttfa_audible_s") for r in ok]),
                "gen_p50_s": _p50([r["gen_time_s"] for r in ok]),
                "overhead_p50": overhead,
                "underrun_req_frac": under_frac,
                "stall_ms_mean": float(np.mean([r["stall_total_ms"] for r in ok])) if ok else float("nan"),
                "stall_ms_max": float(max((r["stall_max_ms"] for r in ok), default=float("nan"))),
                "verdict": verdict,
            }
            entries.append(_quality_entry(e, rs, quality))
        helps = [e for e in entries if e["verdict"] == "helps"]
        if len(entries) == 1:
            concl = f"No stream cells at concurrency {conc} (engine lacks true streaming); not compared."
        elif helps:
            # Rank by audible TTFA: an engine may send a silent bootstrap frame first
            # (vLLM-Omni does), which makes payload TTFA look better than what is heard.
            best = min(helps, key=lambda e: (e["ttfa_audible_p50_s"] if e["ttfa_audible_p50_s"] is not None else e["ttfa_p50_s"]))
            concl = (f"Streaming helps at concurrency {conc}: best setting '{best['setting']}' (lowest audible TTFA) cuts p50 "
                     f"audible TTFA {full_ttfa:.3f}s -> {best['ttfa_audible_p50_s']:.3f}s (first payload {best['ttfa_p50_s']:.3f}s) "
                     f"with {best['underrun_req_frac'] * 100:.1f}% of requests "
                     f"stalling >{cfg.streaming.underrun_threshold_ms:.0f} ms and {best['overhead_p50'] * 100:+.1f}% median total-time overhead.")
        elif any(e["verdict"] == "mixed" for e in entries):
            concl = (f"Mixed at concurrency {conc}: streaming lowers TTFA but every setting exceeds the stall or "
                     f"total-time-overhead limits.")
        else:
            concl = f"No streaming setting met the TTFA-reduction criterion at concurrency {conc} (negative result)."
        results["by_concurrency"][str(conc)] = {"entries": entries, "conclusion": concl}
    return results


def _quality_entry(e: dict[str, Any], rs: list[dict[str, Any]], quality: dict) -> dict[str, Any]:
    qs = [quality[(r["cell"], r["prompt_id"], r["repeat"])] for r in rs if (r["cell"], r["prompt_id"], r["repeat"]) in quality]
    if qs and "N" in qs[0]:
        est = bootstrap_ratio_ci([q["S"] + q["D"] + q["I"] for q in qs], [q["N"] for q in qs], n_boot=500)
        e["wer"] = est.value
    if qs and "pred_mos" in qs[0]:
        vals = [q["pred_mos"] for q in qs if q["pred_mos"] is not None]
        e["pred_mos"] = float(np.mean(vals)) if vals else None
    return e


def _f(x: Any, d: int = 3, scale: float = 1.0, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x * scale:.{d}f}{suffix}"


def render_markdown(res: dict[str, Any], engine: str) -> str:
    th = res["thresholds"]
    out = [f"# Streaming vs full - {engine}", "", f"Run `{res['run_id']}`.", ""]
    out.append(
        f"Verdict rules: *helps* = p50 TTFA reduced >= {th['min_ttfa_reduction'] * 100:.0f}%, <= {th['max_underrun_request_frac'] * 100:.0f}% "
        f"of requests with an audible stall, paired median total-time overhead <= {th['max_total_time_overhead'] * 100:.0f}%. "
        "Quality columns are informational (one sample per prompt; confirm by listening)."
    )
    for conc, block in res["by_concurrency"].items():
        out += ["", f"## Concurrency {conc}", ""]
        out.append("| setting | n | failed | TTFA p50 s | TTFA reduction | audible TTFA p50 s | total p50 s | overhead p50 | reqs stalling | stall ms mean | stall ms max | WER % | pred. MOS | verdict |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in block["entries"]:
            out.append(
                f"| {e['setting']} | {e['n']} | {e['failed']} | {_f(e['ttfa_p50_s'])} | {_f(e.get('ttfa_reduction'), 1, 100, '%')} | "
                f"{_f(e['ttfa_audible_p50_s'])} | {_f(e['gen_p50_s'])} | {_f(e['overhead_p50'], 1, 100, '%')} | "
                f"{_f(e['underrun_req_frac'], 1, 100, '%')} | {_f(e['stall_ms_mean'], 1)} | {_f(e['stall_ms_max'], 1)} | "
                f"{_f(e.get('wer'), 2, 100)} | {_f(e.get('pred_mos'), 2)} | {e['verdict']} |"
            )
        out += ["", f"**Conclusion:** {block['conclusion']}"]
    return "\n".join(out) + "\n"


def write_comparison(run_dir: str | Path, cfg: BenchConfig) -> Path:
    run_dir = Path(run_dir)
    res = compare_streaming(run_dir, cfg)
    rows = load_rows(run_dir)
    engine = rows[0]["engine"] if rows else "?"
    (run_dir / "streaming_comparison.json").write_text(json.dumps(res, indent=2))
    md = run_dir / "streaming_comparison.md"
    md.write_text(render_markdown(res, engine))
    return md


async def run_streaming_comparison(
    cfg: BenchConfig,
    engine_name: str,
    *,
    concurrency: list[int] | None = None,
    out_dir: str | Path | None = None,
    engine: Engine | None = None,
    log: Any = print,
) -> Path:
    chunks = [chunk_value(c) for c in cfg.streaming.initial_codec_chunk_frames]
    run_dir = await run_bench(
        cfg,
        engine_name,
        modes=["full", "stream"],
        concurrency=concurrency or [1],
        chunk_settings=chunks,
        out_dir=out_dir,
        tag="streaming",
        engine=engine,
        log=log,
    )
    md = write_comparison(run_dir, cfg)
    log(f"wrote {md}")
    return run_dir
