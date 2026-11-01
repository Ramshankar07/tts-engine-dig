"""`tts-bench` command line.

    tts-bench validate-prompts [prompts.jsonl]
    tts-bench bench      --config C --engine E [--concurrency 1 4] [--modes full stream] [--limit N]
    tts-bench quality    RUN_DIR --config C [--asr whisper|fake|none] [--mos utmosv2|none]
    tts-bench streaming  --config C --engine E [--concurrency 1] [--limit N]
    tts-bench report     RUN_DIR [RUN_DIR ...] --out reports/p1.md
    tts-bench profile    --config C [--n-prompts 5] --out runs/profile-official
    tts-bench nsys-summary CSV --title T --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import load_config
from .engines import ENGINE_NAMES

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "p1_run.toml"


def _cfg(args: argparse.Namespace):
    overrides: dict = {}
    if getattr(args, "limit", None):
        overrides.setdefault("run", {})["limit"] = args.limit
    if getattr(args, "model_path", None):
        overrides.setdefault("model", {})["local_path"] = args.model_path
    if getattr(args, "base_url", None) and getattr(args, "engine", None):
        overrides.setdefault("engines", {}).setdefault(args.engine, {})["base_url"] = args.base_url
    return load_config(args.config, overrides)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tts-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate-prompts")
    p.add_argument("path", nargs="?", default=str(REPO / "prompts" / "p1_prompts.jsonl"))

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
        p.add_argument("--model-path", default=None, help="local checkpoint dir (overrides model.local_path)")

    p = sub.add_parser("bench")
    common(p)
    p.add_argument("--engine", required=True, choices=ENGINE_NAMES)
    p.add_argument("--concurrency", type=int, nargs="+")
    p.add_argument("--modes", nargs="+", choices=["full", "stream"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--out", default=None, help="runs directory (default: run.output_dir)")
    p.add_argument("--tag", default=None)
    p.add_argument("--engine-python", default=None, help="python of the server venv, for env.json package list")

    p = sub.add_parser("quality")
    p.add_argument("run_dir")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--asr", choices=["whisper", "fake", "none"], default=None)
    p.add_argument("--mos", choices=["utmosv2", "none"], default=None)

    p = sub.add_parser("streaming")
    common(p)
    p.add_argument("--engine", choices=ENGINE_NAMES, help="required unless --from-run")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1])
    p.add_argument("--base-url", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--from-run", default=None, help="only (re)build the comparison for an existing run dir")

    p = sub.add_parser("report")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default=str(REPO / "reports" / "p1_report.md"))
    p.add_argument("--baseline", default="official")
    p.add_argument("--bootstrap", type=int, default=2000)

    p = sub.add_parser("profile")
    common(p)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", default=str(REPO / "runs" / "profile-official"))

    p = sub.add_parser("nsys-summary")
    p.add_argument("csv")
    p.add_argument("--title", required=True)
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "validate-prompts":
        from .prompts import load_prompts, validate_prompt_file

        errs = validate_prompt_file(args.path)
        for e in errs:
            print("ERROR", e)
        if not errs:
            ps = load_prompts(args.path)
            buckets = {b: sum(p.length_bucket == b for p in ps) for b in ("short", "medium", "long")}
            print(f"OK: {len(ps)} prompts {buckets}")
        return 1 if errs else 0

    if args.cmd == "bench":
        from .bench import run_bench

        cfg = _cfg(args)
        asyncio.run(
            run_bench(cfg, args.engine, modes=args.modes, concurrency=args.concurrency, out_dir=args.out, tag=args.tag, engine_python=args.engine_python)
        )
        return 0

    if args.cmd == "quality":
        from .quality import evaluate_run

        evaluate_run(args.run_dir, load_config(args.config), asr_backend=args.asr, mos_backend=args.mos)
        return 0

    if args.cmd == "streaming":
        from .streaming import run_streaming_comparison, write_comparison

        cfg = _cfg(args)
        if args.from_run:
            print(f"wrote {write_comparison(args.from_run, cfg)}")
        elif not args.engine:
            ap.error("streaming: --engine is required unless --from-run is given")
        else:
            asyncio.run(run_streaming_comparison(cfg, args.engine, concurrency=args.concurrency, out_dir=args.out))
        return 0

    if args.cmd == "report":
        from .report import write_report

        out = write_report(args.run_dirs, args.out, n_boot=args.bootstrap, baseline_engine=args.baseline)
        print(f"wrote {out} (+ {out.with_suffix('.csv').name})")
        return 0

    if args.cmd == "profile":
        from .profile import profile_official

        profile_official(_cfg(args), args.out, n_prompts=args.n_prompts, warmup=args.warmup)
        return 0

    if args.cmd == "nsys-summary":
        from .profile import summarize_nsys

        print(f"wrote {summarize_nsys(args.csv, args.out, args.title)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
