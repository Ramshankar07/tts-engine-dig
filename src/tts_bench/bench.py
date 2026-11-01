"""Benchmark runner.

For one engine, runs the fixed prompt set for every (mode x concurrency x
chunk-setting) "cell" and writes, into ``runs/<timestamp>-<engine>[-tag]/``:

* ``config.json``   resolved pinned config (+ prompt-file sha256)
* ``env.json``      git sha, packages, GPU/driver, engine/server info
* ``results.jsonl`` one row per request (failures included, never dropped)
* ``cells.json``    per-cell aggregates (wall time, throughput, peak memory)
* ``audio/``        generated wavs (for WER / predicted MOS)

Per-request metrics
  ttfa_s            time to first audio payload (full mode: == gen_time_s)
  ttfa_audible_s    time until the chunk containing the first non-silent sample arrived
  gen_time_s        submit -> last byte
  audio_duration_s  samples / sample_rate
  rtf               gen_time_s / audio_duration_s  (< 1 is faster than real time)
  playback          simulated real-time playback from chunk arrivals: underruns,
                    total/max stall ms, stalls above the audibility threshold
  peak_gpu_mem_mb   max device memory sampled while the request was in flight

Concurrency = number of closed-loop client workers with independent requests
(HTTP engines) or static batch size (in-process official engine).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .audio import first_nonsilent_sample, write_wav
from .config import ENGINE_DEFAULT_CHUNK, BenchConfig
from .engines import Engine, SynthesisRequest, SynthesisResult, build_engine
from .envinfo import env_snapshot
from .gpu import GpuMemoryPoller
from .playback import simulate_playback
from .prompts import Prompt, load_prompts, prompt_set_hash


@dataclass(frozen=True)
class Cell:
    mode: str
    concurrency: int
    chunk: int | None  # initial_codec_chunk_frames; None = engine default

    @property
    def key(self) -> str:
        k = "default" if self.chunk is None else str(self.chunk)
        return f"{self.mode}-c{self.concurrency}-k{k}" if self.mode == "stream" else f"{self.mode}-c{self.concurrency}"


def chunk_value(v: int) -> int | None:
    return None if v == ENGINE_DEFAULT_CHUNK else v


def plan_cells(
    engine: Engine,
    modes: list[str],
    concurrency: list[int],
    chunk_settings: list[int | None],
    allow_fallback_stream: bool = False,
) -> tuple[list[Cell], list[str]]:
    cells, skipped = [], []
    for c in concurrency:
        for m in modes:
            if m == "stream":
                if not engine.supports_streaming and not allow_fallback_stream:
                    skipped.append(f"stream-c{c}: {engine.name} has no true streaming")
                    continue
                if engine.batch_concurrency:
                    skipped.append(f"stream-c{c}: batched in-process engine cannot stream")
                    continue
                for ch in chunk_settings:
                    cells.append(Cell("stream", c, ch))
            else:
                cells.append(Cell("full", c, None))
    return cells, skipped


def make_request(cfg: BenchConfig, prompt: Prompt, index: int, repeat: int, chunk: int | None) -> SynthesisRequest:
    return SynthesisRequest(
        request_id=f"{prompt.id}-r{repeat}",
        text=prompt.text,
        voice=cfg.model.voice,
        language=cfg.model.language,
        instructions=cfg.model.instructions,
        # Same seed for the same prompt across engines/modes -> paired comparisons.
        seed=cfg.run.seed + index,
        initial_codec_chunk_frames=chunk,
        extra={"prompt_id": prompt.id},
    )


def request_metrics(res: SynthesisResult, cfg: BenchConfig) -> dict[str, Any]:
    """Derived per-request metrics from a SynthesisResult timeline."""
    sr = res.sample_rate or cfg.model.sample_rate
    dur = res.audio.size / sr if sr else 0.0
    out: dict[str, Any] = {
        "ok": res.ok,
        "error": res.error,
        "sample_rate": sr,
        "audio_duration_s": dur,
        "ttfa_s": res.t_first_audio,
        "gen_time_s": res.t_end,
        "rtf": (res.t_end / dur) if (res.ok and dur > 0 and res.t_end is not None) else None,
        "n_chunks": len(res.chunk_timeline),
        "chunk_timeline": [[round(t, 6), n] for t, n in res.chunk_timeline],
    }
    # Time to first *audible* audio.
    idx = first_nonsilent_sample(res.audio, cfg.streaming.silence_threshold) if res.audio.size else None
    out["leading_silence_s"] = (idx / sr) if idx is not None else None
    t_aud = None
    if idx is not None:
        cum = 0
        for t, n in res.chunk_timeline:
            cum += n
            if cum > idx:
                t_aud = t
                break
    out["ttfa_audible_s"] = t_aud
    pb = simulate_playback(
        [(t, n / sr) for t, n in res.chunk_timeline],
        prebuffer_ms=cfg.streaming.prebuffer_ms,
        threshold_ms=cfg.streaming.underrun_threshold_ms,
    )
    out.update(
        playback_start_s=pb.playback_start_s,
        playback_end_s=pb.playback_end_s,
        underrun_count=pb.underrun_count,
        stall_total_ms=pb.stall_total_ms,
        stall_max_ms=pb.stall_max_ms,
        stalls_over_threshold=pb.stalls_over_threshold,
    )
    return out


class BenchRunner:
    def __init__(
        self,
        cfg: BenchConfig,
        engine: Engine,
        run_dir: Path,
        prompts: list[Prompt],
        poller: GpuMemoryPoller | None = None,
        log: Any = print,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.run_dir = run_dir
        self.prompts = prompts
        self.poller = poller
        self.log = log
        self.rows_path = run_dir / "results.jsonl"

    async def _timed(self, req: SynthesisRequest, mode: str) -> tuple[SynthesisResult, float, float]:
        t0 = time.perf_counter()
        res = await self.engine.run(req, mode)  # type: ignore[arg-type]
        return res, t0, time.perf_counter()

    async def run_cell(self, cell: Cell) -> dict[str, Any]:
        cfg = self.cfg
        # Warmup (not recorded): CUDA graphs, allocator, HTTP keep-alive.
        for i, p in enumerate(self.prompts[: cfg.run.warmup_requests]):
            await self.engine.run(make_request(cfg, p, i, -1, cell.chunk), cell.mode)  # type: ignore[arg-type]

        jobs = [(i, p, r) for r in range(cfg.run.repeats) for i, p in enumerate(self.prompts)]
        records: list[tuple[int, Prompt, int, SynthesisResult, float, float]] = []
        t_cell0 = time.perf_counter()

        if self.engine.batch_concurrency and cell.concurrency > 1:
            for b in range(0, len(jobs), cell.concurrency):
                batch = jobs[b : b + cell.concurrency]
                reqs = [make_request(cfg, p, i, r, cell.chunk) for i, p, r in batch]
                t0 = time.perf_counter()
                results = await self.engine.run_batch(reqs)
                t1 = time.perf_counter()
                records += [(i, p, r, res, t0, t1) for (i, p, r), res in zip(batch, results, strict=True)]
        else:
            queue: asyncio.Queue = asyncio.Queue()
            for j in jobs:
                queue.put_nowait(j)

            async def worker() -> None:
                while True:
                    try:
                        i, p, r = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    res, t0, t1 = await self._timed(make_request(cfg, p, i, r, cell.chunk), cell.mode)
                    records.append((i, p, r, res, t0, t1))

            await asyncio.gather(*(worker() for _ in range(cell.concurrency)))
        wall = time.perf_counter() - t_cell0

        rows = []
        for i, p, r, res, t0, t1 in sorted(records, key=lambda x: (x[2], x[0])):
            row: dict[str, Any] = {
                "run_id": self.run_dir.name,
                "engine": self.engine.name,
                "cell": cell.key,
                "mode": cell.mode,
                "concurrency": cell.concurrency,
                "chunk": cell.chunk,
                "prompt_id": p.id,
                "prompt_index": i,
                "repeat": r,
                "text": p.text,
                "length_bucket": p.length_bucket,
                "category": p.category,
                "n_words": p.n_words,
                "seed": cfg.run.seed + i,
                "t_submit_rel_s": t0 - t_cell0,
            }
            row.update(request_metrics(res, cfg))
            row["peak_gpu_mem_mb"] = self.poller.max_between(t0, t1) if self.poller else None
            for k in ("torch_peak_alloc_mb", "torch_peak_reserved_mb", "lm_generate_s", "codec_decode_s", "batch_size"):
                if k in res.meta:
                    row[k] = res.meta[k]
            row["audio_path"] = None
            if cfg.run.save_audio and res.audio.size:
                rel = Path("audio") / cell.key / f"{p.id}-r{r}.wav"
                write_wav(self.run_dir / rel, res.audio, res.sample_rate)
                row["audio_path"] = str(rel)
            rows.append(row)

        with self.rows_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        ok = [x for x in rows if x["ok"]]
        audio_s = sum(x["audio_duration_s"] for x in ok)
        summary = {
            **asdict(cell),
            "cell": cell.key,
            "n_requests": len(rows),
            "n_failed": len(rows) - len(ok),
            "wall_time_s": wall,
            "audio_seconds": audio_s,
            "throughput_audio_s_per_s": audio_s / wall if wall > 0 else None,
            "requests_per_s": len(ok) / wall if wall > 0 else None,
            "peak_gpu_mem_mb": max((x["peak_gpu_mem_mb"] for x in rows if x["peak_gpu_mem_mb"] is not None), default=None),
        }
        ttfa = [x["ttfa_s"] for x in ok if x["ttfa_s"] is not None]
        if ttfa:
            summary["ttfa_p50_s"] = float(np.percentile(ttfa, 50))
        self.log(
            f"[{self.engine.name}] {cell.key}: {len(ok)}/{len(rows)} ok, wall {wall:.2f}s, "
            f"throughput {summary['throughput_audio_s_per_s'] or 0:.2f} audio-s/s"
        )
        return summary


def new_run_dir(cfg: BenchConfig, engine_name: str, tag: str | None = None, out_dir: str | Path | None = None) -> Path:
    base = Path(out_dir) if out_dir else cfg.resolve(cfg.run.output_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{engine_name}" + (f"-{tag}" if tag else "")
    d = base / name
    n = 1
    while d.exists():  # two runs in the same second (tests)
        n += 1
        d = base / f"{name}-{n}"
    d.mkdir(parents=True)
    return d


async def run_bench(
    cfg: BenchConfig,
    engine_name: str,
    *,
    modes: list[str] | None = None,
    concurrency: list[int] | None = None,
    chunk_settings: list[int | None] | None = None,
    out_dir: str | Path | None = None,
    tag: str | None = None,
    engine: Engine | None = None,
    engine_python: str | None = None,
    allow_fallback_stream: bool = False,
    log: Any = print,
) -> Path:
    """Run all cells for one engine; returns the run directory."""
    engine = engine or build_engine(engine_name, cfg)
    prompts_path = cfg.resolve(cfg.run.prompts)
    prompts = load_prompts(prompts_path, cfg.run.limit)
    modes = modes or list(cfg.run.modes)
    concurrency = concurrency or list(cfg.run.concurrency)
    if chunk_settings is None:
        chunk_settings = [None]  # bench: engine-default streaming; streaming.py sweeps
    run_dir = new_run_dir(cfg, engine.name, tag, out_dir)

    cells, skipped = plan_cells(engine, modes, concurrency, chunk_settings, allow_fallback_stream)
    for s in skipped:
        log(f"skip {s}")

    resolved = cfg.model_dump()
    resolved["prompts_sha256"] = prompt_set_hash(prompts_path)
    resolved["n_prompts"] = len(prompts)
    resolved["cells"] = [c.key for c in cells]
    resolved["skipped_cells"] = skipped
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    poller = GpuMemoryPoller(cfg.gpu.device_index, cfg.gpu.poll_interval_s)
    await engine.start()
    try:
        (run_dir / "env.json").write_text(
            json.dumps(env_snapshot(engine.describe(), engine_python) | {"gpu_poller": poller.backend}, indent=2, default=str)
        )
        poller.start()
        summaries = []
        runner = BenchRunner(cfg, engine, run_dir, prompts, poller, log)
        for cell in cells:
            summaries.append(await runner.run_cell(cell))
    finally:
        poller.stop()
        await engine.close()
    (run_dir / "cells.json").write_text(json.dumps({"engine": engine.name, "cells": summaries, "gpu_peak_mb": poller.peak_mb()}, indent=2))
    log(f"wrote {run_dir}")
    return run_dir


def load_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
