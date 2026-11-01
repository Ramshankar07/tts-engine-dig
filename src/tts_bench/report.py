"""Aggregate run directories into markdown comparison tables.

Every point estimate carries a 95% percentile-bootstrap CI over requests.
Quality differences between engines use a *paired* bootstrap over prompts
that both runs share (same text, same seed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .bench import load_rows
from .quality import load_quality
from .stats import Estimate, bootstrap_ci, bootstrap_ratio_ci, mean, p50, p90, paired_bootstrap_ratio_diff



def _gpu_names(gpu: dict[str, Any]) -> list[str]:
    """torch device names, else the nvidia-smi CSV names (HTTP-client venvs have no torch)."""
    if gpu.get("devices"):
        return list(gpu["devices"])
    csv = gpu.get("nvidia_smi") or ""
    names = [ln.split(",")[1].strip() for ln in csv.splitlines()[1:] if ln.count(",") >= 2]
    return names or ["none detected"]

def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text()) if p.exists() else {}


def load_run(run_dir: str | Path) -> dict[str, Any]:
    d = Path(run_dir)
    return {
        "dir": d,
        "rows": load_rows(d),
        "quality": load_quality(d),
        "cells": _read_json(d / "cells.json"),
        "config": _read_json(d / "config.json"),
        "env": _read_json(d / "env.json"),
        "quality_summary": _read_json(d / "quality_summary.json"),
    }


def _cell_table(run: dict[str, Any], n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows = run["rows"]
    cell_meta = {c["cell"]: c for c in run["cells"].get("cells", [])}
    q_index = {(q["cell"], q["prompt_id"], q["repeat"]): q for q in run["quality"]}
    out = []
    for cell in dict.fromkeys(r["cell"] for r in rows):  # preserve order
        rs = [r for r in rows if r["cell"] == cell]
        ok = [r for r in rs if r["ok"]]
        ttfa = [r["ttfa_s"] for r in ok]
        aud = [r["ttfa_audible_s"] for r in ok if r.get("ttfa_audible_s") is not None]
        rtf = [r["rtf"] for r in ok if r.get("rtf") is not None]
        stall = [r["stall_total_ms"] for r in ok]
        meta = cell_meta.get(cell, {})
        rec: dict[str, Any] = {
            "engine": rs[0]["engine"],
            "run": run["dir"].name,
            "cell": cell,
            "mode": rs[0]["mode"],
            "concurrency": rs[0]["concurrency"],
            "chunk": rs[0]["chunk"],
            "n": len(rs),
            "failed": len(rs) - len(ok),
            "ttfa_p50": bootstrap_ci(ttfa, p50, n_boot, seed=seed),
            "ttfa_p90": bootstrap_ci(ttfa, p90, n_boot, seed=seed),
            "ttfa_audible_p50": bootstrap_ci(aud, p50, n_boot, seed=seed),
            "rtf_p50": bootstrap_ci(rtf, p50, n_boot, seed=seed),
            "stall_ms_mean": bootstrap_ci(stall, mean, n_boot, seed=seed),
            "underrun_req_frac": (sum(1 for r in ok if r["stalls_over_threshold"] > 0) / len(ok)) if ok else float("nan"),
            "throughput": meta.get("throughput_audio_s_per_s"),
            "req_per_s": meta.get("requests_per_s"),
            "peak_mem_mb": meta.get("peak_gpu_mem_mb"),
            "torch_peak_mb": max((r["torch_peak_alloc_mb"] for r in rs if r.get("torch_peak_alloc_mb")), default=None),
        }
        qs = [q_index[(r["cell"], r["prompt_id"], r["repeat"])] for r in rs if (r["cell"], r["prompt_id"], r["repeat"]) in q_index]
        if qs and "N" in qs[0]:
            rec["wer"] = bootstrap_ratio_ci([q["S"] + q["D"] + q["I"] for q in qs], [q["N"] for q in qs], n_boot, seed=seed)
        if qs and "pred_mos" in qs[0]:
            rec["pred_mos"] = bootstrap_ci([q["pred_mos"] for q in qs if q["pred_mos"] is not None], mean, n_boot, seed=seed)
        out.append(rec)
    return out


def _fmt(v: Any, digits: int = 3, scale: float = 1.0) -> str:
    if isinstance(v, Estimate):
        return v.fmt(digits, scale)
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if isinstance(v, float):
        return f"{v * scale:.{digits}f}"
    return str(v)


def paired_wer_diffs(runs: list[dict[str, Any]], baseline_engine: str, n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Corpus-WER(engine) - corpus-WER(baseline) for matching cells, paired by prompt."""
    def qmap(run: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
        return {(q["cell"], q["prompt_id"], q["repeat"]): q for q in run["quality"] if "N" in q}

    base_runs = [r for r in runs if r["rows"] and r["rows"][0]["engine"] == baseline_engine]
    if not base_runs:
        return []
    base = qmap(base_runs[0])
    out = []
    for run in runs:
        if not run["rows"] or run["rows"][0]["engine"] == baseline_engine:
            continue
        other = qmap(run)
        for cell in sorted({k[0] for k in other}):
            # Compare against the baseline's full-mode cell at the same concurrency
            # if the exact cell doesn't exist there (official engine has no stream cells).
            conc = cell.split("-")[1]
            bcell = cell if any(k[0] == cell for k in base) else f"full-{conc}"
            keys = [(cell, p, r) for (c, p, r) in other if c == cell and (bcell, p, r) in base]
            if not keys:
                continue
            a = [other[k]["S"] + other[k]["D"] + other[k]["I"] for k in keys]
            b = [base[(bcell, k[1], k[2])]["S"] + base[(bcell, k[1], k[2])]["D"] + base[(bcell, k[1], k[2])]["I"] for k in keys]
            n = [other[k]["N"] for k in keys]
            out.append(
                {
                    "engine": run["rows"][0]["engine"],
                    "cell": cell,
                    "baseline_cell": f"{baseline_engine}:{bcell}",
                    "n_pairs": len(keys),
                    "delta_wer": paired_bootstrap_ratio_diff(a, b, n, n_boot, seed=seed),
                }
            )
    return out


def build_report(run_dirs: list[str | Path], n_boot: int = 2000, seed: int = 0, baseline_engine: str = "official") -> tuple[str, pd.DataFrame]:
    runs = [load_run(d) for d in run_dirs]
    runs = [r for r in runs if r["rows"]]
    if not runs:
        raise ValueError("no runs with results.jsonl found")
    table: list[dict[str, Any]] = []
    for run in runs:
        table += _cell_table(run, n_boot, seed)

    lines = ["# P1 engine comparison", ""]
    lines.append("Generated by `tts-bench report`. Values: point estimate [95% bootstrap CI]. Times in seconds.")
    lines.append("TTFA = time to first audio payload; audible TTFA = first chunk containing non-silent audio.")
    lines.append("RTF = generation time / audio duration (lower is faster). Throughput = audio seconds produced per wall second for the cell.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| run | engine | model revision | prompts sha256 | git sha | GPU |")
    lines.append("|---|---|---|---|---|---|")
    for run in runs:
        cfg, env = run["config"], run["env"]
        gpu = _gpu_names(env.get("gpu", {}) or {})
        lines.append(
            f"| {run['dir'].name} | {run['rows'][0]['engine']} | {cfg.get('model', {}).get('revision', '?')[:12]} | "
            f"{str(cfg.get('prompts_sha256', '?'))[:12]} | {str((env.get('git') or {}).get('sha'))[:12]} | {', '.join(gpu)} |"
        )
    lines.append("")
    lines.append("## Latency, throughput, memory")
    lines.append("")
    lines.append("| engine | run | cell | n | failed | TTFA p50 | TTFA p90 | audible TTFA p50 | RTF p50 | throughput (audio s/s) | req/s | peak GPU MB | torch peak MB | reqs w/ stall>thr | stall ms mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        lines.append(
            f"| {t['engine']} | {t['run'][:15]} | {t['cell']} | {t['n']} | {t['failed']} | {_fmt(t['ttfa_p50'])} | {_fmt(t['ttfa_p90'])} | "
            f"{_fmt(t['ttfa_audible_p50'])} | {_fmt(t['rtf_p50'])} | {_fmt(t['throughput'], 2)} | {_fmt(t['req_per_s'], 2)} | "
            f"{_fmt(t['peak_mem_mb'], 0)} | {_fmt(t['torch_peak_mb'], 0)} | {_fmt(t['underrun_req_frac'], 1, 100)}% | {_fmt(t['stall_ms_mean'], 1)} |"
        )
    if any("wer" in t or "pred_mos" in t for t in table):
        norm = next((r["quality_summary"].get("normalizer") for r in runs if r["quality_summary"]), "?")
        asr = next((r["quality_summary"].get("asr") for r in runs if r["quality_summary"]), {})
        lines += ["", "## Quality", ""]
        lines.append(f"Corpus WER = sum(S+D+I)/sum(ref words); failed syntheses count as full deletions. Normalizer `{norm}`. ASR: `{asr}`.")
        lines.append("Predicted MOS is UTMOSv2 output, a proxy - not a human MOS.")
        lines.append("")
        lines.append("| engine | run | cell | n | failed | corpus WER % | predicted MOS |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in table:
            lines.append(f"| {t['engine']} | {t['run'][:15]} | {t['cell']} | {t['n']} | {t['failed']} | {_fmt(t.get('wer'), 2, 100)} | {_fmt(t.get('pred_mos'), 2)} |")
        diffs = paired_wer_diffs(runs, baseline_engine, n_boot, seed)
        if diffs:
            lines += ["", f"### Paired WER difference vs `{baseline_engine}` (percentage points)", ""]
            lines.append("| engine | cell | baseline | pairs | delta WER pp |")
            lines.append("|---|---|---|---|---|")
            for d in diffs:
                lines.append(f"| {d['engine']} | {d['cell']} | {d['baseline_cell']} | {d['n_pairs']} | {_fmt(d['delta_wer'], 2, 100)} |")
    lines.append("")

    flat = []
    for t in table:
        f = {}
        for k, v in t.items():
            if isinstance(v, Estimate):
                f[k], f[k + "_lo"], f[k + "_hi"] = v.value, v.lo, v.hi
            else:
                f[k] = v
        flat.append(f)
    return "\n".join(lines), pd.DataFrame(flat)


def write_report(run_dirs: list[str | Path], out_md: str | Path, **kw: Any) -> Path:
    md, df = build_report(run_dirs, **kw)
    out = Path(out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    df.to_csv(out.with_suffix(".csv"), index=False)
    return out
