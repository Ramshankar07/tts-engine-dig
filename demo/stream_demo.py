#!/usr/bin/env python3
"""Stream one utterance to a WAV file while (optionally) playing it live.

    # CPU, no server: fake engine (a synthetic tone signal, not speech)
    uv run python demo/stream_demo.py --engine fake --text "Hello from the fake engine." --play

    # GPU box, against a running server
    python demo/stream_demo.py --engine vllm_omni --text "Hello, how are you?" --play
    python demo/stream_demo.py --engine sglang_omni --initial-chunk 4 --out demo/out.wav

Live playback pipes raw PCM into the first available player:
sox `play`, ALSA `aplay`, or `ffplay`. Without one, the WAV is played after the
stream ends with `afplay` (macOS) if present. Prints per-chunk arrival times and
the simulated playback-stall summary used by the benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # allow running without installing

from tts_bench.audio import float_to_pcm16_bytes  # noqa: E402
from tts_bench.config import load_config  # noqa: E402
from tts_bench.engines import SynthesisRequest, build_engine  # noqa: E402
from tts_bench.playback import simulate_playback  # noqa: E402


def live_player(sr: int) -> subprocess.Popen | None:
    cmds = [
        ["play", "-q", "-t", "raw", "-r", str(sr), "-e", "signed", "-b", "16", "-c", "1", "-"],
        ["aplay", "-q", "-f", "S16_LE", "-r", str(sr), "-c", "1"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-"],
    ]
    for cmd in cmds:
        if shutil.which(cmd[0]):
            return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return None


async def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.base_url:
        cfg.engines.setdefault(args.engine, {})["base_url"] = args.base_url
    engine = build_engine(args.engine, cfg)
    req = SynthesisRequest(
        request_id="demo",
        text=args.text,
        voice=cfg.model.voice,
        language=cfg.model.language,
        seed=cfg.run.seed,
        initial_codec_chunk_frames=args.initial_chunk,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    await engine.start()
    player = None
    timeline: list[tuple[float, float]] = []
    try:
        if not engine.supports_streaming:
            print(f"note: {engine.name} has no true streaming; audio arrives as one chunk at the end")
        t0 = time.perf_counter()
        wav = None
        async for chunk in engine.stream(req):
            sr = engine.sample_rate
            if wav is None:
                wav = sf.SoundFile(str(out), "w", samplerate=sr, channels=1, subtype="PCM_16")
                if args.play:
                    player = live_player(sr)
            wav.write(chunk.pcm)
            if player and player.stdin:
                player.stdin.write(float_to_pcm16_bytes(chunk.pcm))
                player.stdin.flush()
            dur = chunk.pcm.size / sr
            timeline.append((chunk.t_arrival, dur))
            print(f"chunk {chunk.index:3d}  t={chunk.t_arrival * 1000:8.1f} ms  +{dur * 1000:6.1f} ms audio")
        total = time.perf_counter() - t0
        if wav is None:
            print("no audio received")
            return 1
        wav.close()
    finally:
        await engine.close()
        if player and player.stdin:
            player.stdin.close()
            player.wait()

    audio_s = sum(d for _t, d in timeline)
    pb = simulate_playback(timeline, threshold_ms=cfg.streaming.underrun_threshold_ms)
    print(f"\nTTFA {timeline[0][0] * 1000:.1f} ms | total {total:.2f} s | audio {audio_s:.2f} s | RTF {total / audio_s:.3f}")
    print(f"simulated playback: {pb.underrun_count} underruns, {pb.stall_total_ms:.1f} ms total stall, max {pb.stall_max_ms:.1f} ms")
    print(f"wrote {out}")
    if args.play and player is None:
        if shutil.which("afplay"):
            subprocess.run(["afplay", str(out)], check=False)
        else:
            print("no audio player found (install sox for live playback)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="fake", choices=["fake", "vllm_omni", "sglang_omni", "official"])
    ap.add_argument("--text", default="Streaming lets playback start before the whole sentence has been generated.")
    ap.add_argument("--config", default=str(REPO / "configs" / "p1_run.toml"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--initial-chunk", type=int, default=None, help="initial_codec_chunk_frames (default: engine default)")
    ap.add_argument("--out", default=str(REPO / "demo" / "out" / "demo.wav"))
    ap.add_argument("--play", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
