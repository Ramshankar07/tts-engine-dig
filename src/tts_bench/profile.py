"""Kernel profiling helpers.

* ``profile_official``: torch.profiler around the official in-process engine.
  Exports a chrome trace (open in chrome://tracing or https://ui.perfetto.dev),
  the key_averages table, and a top-kernel table grouped into
  attention / GEMM / codec-conv / other. LM generation and codec decoding are
  labelled with ``record_function`` ranges ("lm_generate", "codec_decode").
* ``kernels_from_nsys_csv``: parse ``nsys stats --report cuda_gpu_kern_sum
  --format csv`` output (server engines are profiled with scripts/profile_nsys.sh)
  into the same grouped table, so all three engines are compared on one scale.

Grouping is by kernel-name regex and is necessarily approximate; always
inspect the top kernels by hand before writing KERNEL_NOTES.md.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Order matters: first match wins. Attention first (fused attention kernels can
# contain "gemm" in their names), then convolution (cuDNN conv kernels contain
# "implicit_gemm"), then GEMM.
GROUP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("attention", re.compile(r"flash|fmha|attention|attn|sdpa|xformers|flashinfer|paged_?kv|mha_|decode_attention", re.I)),
    ("codec-conv", re.compile(r"conv(?!ert)|cudnn|winograd|implicit_?gemm|dgrad|wgrad|im2col|col2im|snake|fft|depthwise", re.I)),
    ("gemm", re.compile(r"gemm|gemv|cutlass|cublas|matmul|addmm|bmm|xmma|nvjet|splitk|s\d+_.*_(tn|nt|nn|tt)\b|sm\d+_xmma", re.I)),
]
GROUPS = ["attention", "gemm", "codec-conv", "other"]


def classify_kernel(name: str) -> str:
    for group, pat in GROUP_PATTERNS:
        if pat.search(name):
            return group
    return "other"


@dataclass
class KernelStat:
    name: str
    total_us: float
    count: int

    @property
    def group(self) -> str:
        return classify_kernel(self.name)


def group_totals(stats: list[KernelStat]) -> dict[str, dict[str, float]]:
    total = sum(s.total_us for s in stats) or 1.0
    out = {g: {"total_us": 0.0, "count": 0, "pct": 0.0} for g in GROUPS}
    for s in stats:
        g = out[s.group]
        g["total_us"] += s.total_us
        g["count"] += s.count
    for g in out.values():
        g["pct"] = 100.0 * g["total_us"] / total
    return out


def render_kernel_markdown(stats: list[KernelStat], title: str, top_n: int = 25) -> str:
    stats = sorted(stats, key=lambda s: s.total_us, reverse=True)
    totals = group_totals(stats)
    grand = sum(s.total_us for s in stats)
    lines = [f"# {title}", "", f"Total GPU kernel time: {grand / 1000:.2f} ms over {sum(s.count for s in stats)} launches.", ""]
    lines += ["| group | time ms | % | launches |", "|---|---|---|---|"]
    for g in GROUPS:
        t = totals[g]
        lines.append(f"| {g} | {t['total_us'] / 1000:.2f} | {t['pct']:.1f} | {int(t['count'])} |")
    lines += ["", f"## Top {top_n} kernels", "", "| # | group | time ms | % | launches | avg us | kernel |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(stats[:top_n], 1):
        name = s.name if len(s.name) <= 140 else s.name[:137] + "..."
        name = name.replace("|", "\\|")
        lines.append(
            f"| {i} | {s.group} | {s.total_us / 1000:.3f} | {100 * s.total_us / (grand or 1):.1f} | {s.count} | "
            f"{s.total_us / max(s.count, 1):.1f} | `{name}` |"
        )
    return "\n".join(lines) + "\n"


def write_kernel_outputs(stats: list[KernelStat], out_dir: Path, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "kernels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "total_us", "count", "name"])
        for s in sorted(stats, key=lambda s: s.total_us, reverse=True):
            w.writerow([s.group, f"{s.total_us:.3f}", s.count, s.name])
    (out_dir / "kernel_groups.json").write_text(json.dumps(group_totals(stats), indent=2))
    md = out_dir / "kernels_grouped.md"
    md.write_text(render_kernel_markdown(stats, title))
    return md


# ----------------------------------------------------------------------------- nsys
def kernels_from_nsys_csv(text: str) -> list[KernelStat]:
    """Parse `nsys stats --report cuda_gpu_kern_sum --format csv`.

    Verified on Nsight Systems 2025.3.2 ("Time (%)","Total Time (ns)","Instances",
    ...,"Name"). Older versions used "Time(%)"; columns are matched loosely.
    """
    # nsys may print notices before the CSV header; start at the header line.
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "Total Time" in l and "Name" in l), None)
    if start is None:
        raise ValueError("no cuda_gpu_kern_sum CSV header found")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    tcol = next(c for c in reader.fieldnames or [] if c.replace(" ", "").lower().startswith("totaltime"))
    ncol = next(c for c in reader.fieldnames or [] if c.strip().lower() in ("instances", "count", "calls"))
    stats = []
    for row in reader:
        name = row.get("Name") or ""
        if not name:
            continue
        stats.append(KernelStat(name=name, total_us=float(row[tcol]) / 1000.0, count=int(float(row[ncol]))))
    return stats


# ----------------------------------------------------------------------------- torch.profiler
ANNOTATIONS = ("lm_generate", "codec_decode")


def is_annotation(name: str) -> bool:
    """record_function range names used by the official engine / profile_official."""
    return name in ANNOTATIONS or name.startswith("request:")


def stats_from_kernels_csv(path: str | Path) -> list[KernelStat]:
    """Re-read kernels.csv written by write_kernel_outputs, dropping annotation rows."""
    with Path(path).open() as f:
        return [
            KernelStat(name=r["name"], total_us=float(r["total_us"]), count=int(r["count"]))
            for r in csv.DictReader(f)
            if not is_annotation(r["name"])
        ]

def kernels_from_torch_profiler(prof: Any, averages: Any = None) -> list[KernelStat]:
    """CUDA kernel rows of ``prof.key_averages()``.

    Verified on torch 2.8 (``self_device_time_total``/``device_type``); older
    releases use ``self_cuda_time_total``, handled by the fallback.
    """
    from torch.autograd import DeviceType

    stats = []
    for evt in averages if averages is not None else prof.key_averages():
        if getattr(evt, "device_type", None) != DeviceType.CUDA:
            continue
        # record_function ranges also appear as device-side rows (gpu_user_annotation);
        # they span many kernels and must not be counted as kernels themselves.
        if getattr(evt, "is_user_annotation", False) or is_annotation(evt.key):
            continue
        t = getattr(evt, "self_device_time_total", None)
        if t is None:
            t = getattr(evt, "self_cuda_time_total", 0.0)
        if t:
            stats.append(KernelStat(name=evt.key, total_us=float(t), count=int(evt.count)))
    return stats


def _param_dtypes(engine: Any) -> dict[str, int]:
    """Parameter count per dtype across the loaded model's torch modules."""
    import torch

    counts: dict[str, int] = {}
    seen: set[int] = set()
    for obj in vars(getattr(engine, "model", None) or object()).values():
        if isinstance(obj, torch.nn.Module):
            for prm in obj.parameters():
                if id(prm) not in seen:
                    seen.add(id(prm))
                    counts[str(prm.dtype)] = counts.get(str(prm.dtype), 0) + prm.numel()
    return counts


def profile_official(cfg: Any, out_dir: str | Path, n_prompts: int = 5, warmup: int = 2, log: Any = print) -> Path:
    """Profile the official engine on the first ``n_prompts`` prompts (full mode, batch 1)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from .bench import make_request
    from .engines import build_engine
    from .prompts import load_prompts

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    engine = build_engine("official", cfg)
    engine._load()  # synchronous load in this thread
    engine.profile_ranges = True
    prompts = load_prompts(cfg.resolve(cfg.run.prompts))
    for i, p in enumerate(prompts[:warmup]):
        engine._generate_blocking([make_request(cfg, p, i, -1, None)])
    sel = prompts[:n_prompts]
    # record_shapes=False: with shapes, HF generate over 5 prompts produced a ~6 GB
    # trace and key_averages() took >15 min of CPU. Keep n_prompts small (1-2).
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for i, p in enumerate(sel):
            with torch.profiler.record_function(f"request:{p.id}"):
                engine._generate_blocking([make_request(cfg, p, i, 0, None)])
        torch.cuda.synchronize()
    averages = prof.key_averages()  # computed once; reused for the table and the grouped kernels
    try:
        table = averages.table(sort_by="self_device_time_total", row_limit=60)
    except Exception:  # noqa: BLE001 - older torch
        table = averages.table(sort_by="self_cuda_time_total", row_limit=60)
    (out / "key_averages.txt").write_text(table)
    stats = kernels_from_torch_profiler(prof, averages)
    md = write_kernel_outputs(stats, out, f"Official Qwen3-TTS (PyTorch) - {len(sel)} prompts, batch 1")
    prof.export_chrome_trace(str(out / "trace_official.json"))
    (out / "profile_meta.json").write_text(
        json.dumps(
            {
                "engine": engine.describe(),
                "prompts": [p.id for p in sel],
                "torch": torch.__version__,
                "param_dtypes": _param_dtypes(engine),
            },
            indent=2,
            default=str,
        )
    )
    log(f"wrote {out}")
    return md


def summarize_nsys(csv_path: str | Path, out_dir: str | Path, title: str) -> Path:
    stats = kernels_from_nsys_csv(Path(csv_path).read_text())
    return write_kernel_outputs(stats, Path(out_dir), title)


__all__ = [
    "KernelStat",
    "classify_kernel",
    "group_totals",
    "render_kernel_markdown",
    "kernels_from_nsys_csv",
    "kernels_from_torch_profiler",
    "is_annotation",
    "stats_from_kernels_csv",
    "profile_official",
    "summarize_nsys",
]
