"""Quality evaluation for a run directory: WER via ASR + UTMOSv2 predicted MOS.

Protocol (frozen, see p1-p2-research-plan.md "P2 evaluation foundation"):

* Text normalizer: ``textnorm.normalize`` @ ``NORMALIZER_VERSION`` applied to
  both reference and hypothesis.
* Corpus WER = sum(S + D + I) / sum(reference words) over ALL requests in a cell.
  A failed synthesis (error / empty audio) is scored as an empty hypothesis,
  i.e. every reference word is a deletion. Failures are counted, never dropped.
* CI: percentile bootstrap over utterances of the corpus ratio.
* UTMOSv2 output is labelled "predicted MOS" everywhere; it is a proxy, not a
  listening test. Failed syntheses have no predicted MOS and are reported as a
  separate count.
* Missing optional deps (transformers / utmosv2) -> that metric is skipped with
  a recorded reason; the run is not aborted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jiwer

from .bench import load_rows
from .config import BenchConfig
from .stats import bootstrap_ci, bootstrap_ratio_ci, mean
from .textnorm import NORMALIZER_VERSION, normalize


# ----------------------------------------------------------------------------- WER math
@dataclass(frozen=True)
class WordErrors:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float | None:
        return self.errors / self.ref_words if self.ref_words else None


def word_errors(reference: str, hypothesis: str, already_normalized: bool = False) -> WordErrors:
    """Levenshtein alignment counts on normalized words."""
    ref = reference if already_normalized else normalize(reference)
    hyp = hypothesis if already_normalized else normalize(hypothesis)
    r, h = ref.split(), hyp.split()
    if not r:
        return WordErrors(0, 0, len(h), 0)
    if not h:
        return WordErrors(0, len(r), 0, len(r))
    out = jiwer.process_words(ref, hyp)
    return WordErrors(out.substitutions, out.deletions, out.insertions, len(r))


def corpus_wer(items: list[WordErrors]) -> float | None:
    n = sum(e.ref_words for e in items)
    return sum(e.errors for e in items) / n if n else None


# ----------------------------------------------------------------------------- backends
class ASRBackend(Protocol):
    name: str

    def transcribe(self, wav_paths: list[Path], references: list[str]) -> list[str]: ...

    def describe(self) -> dict[str, Any]: ...


class FakeASR:
    """Returns the reference text, dropping every k-th word (k=0: perfect).
    For CPU pipeline tests only - not a measurement."""

    name = "fake"

    def __init__(self, error_every: int = 0) -> None:
        self.error_every = error_every

    def transcribe(self, wav_paths: list[Path], references: list[str]) -> list[str]:
        out = []
        for ref in references:
            words = ref.split()
            if self.error_every > 0:
                words = [w for i, w in enumerate(words, 1) if i % self.error_every]
            out.append(" ".join(words))
        return out

    def describe(self) -> dict[str, Any]:
        return {"backend": "fake", "error_every": self.error_every}


class WhisperASR:
    """Whisper large-v3 via the transformers ASR pipeline (the [eval] extra)."""

    name = "whisper"

    def __init__(self, model: str, revision: str, language: str, device: str, batch_size: int = 8) -> None:
        import torch
        from transformers import pipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model_id, self.revision, self.language = model, revision, language
        self.batch_size = batch_size
        # Verified on transformers 5.19 with Whisper large-v3 (H100, 2026-10-13).
        # Older releases spell torch_dtype; long-form uses chunk_length_s + batch_size.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            revision=revision,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(self, wav_paths: list[Path], references: list[str]) -> list[str]:
        import librosa

        audios = [librosa.load(str(p), sr=16000, mono=True)[0] for p in wav_paths]
        outs = self.pipe(
            [{"raw": a, "sampling_rate": 16000} for a in audios],
            batch_size=self.batch_size,
            # Long prompts can exceed 30 s; the pipeline's chunking handles that.
            chunk_length_s=30,
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        return [o["text"].strip() for o in outs]

    def describe(self) -> dict[str, Any]:
        import transformers

        return {"backend": "whisper", "model": self.model_id, "revision": self.revision, "transformers": transformers.__version__}


class UtmosV2:
    """UTMOSv2 predicted MOS (sarulab-speech/UTMOSv2 README API)."""

    name = "utmosv2"

    def __init__(self) -> None:
        import utmosv2

        self.model = utmosv2.create_model(pretrained=True)

    def predict(self, wav_path: Path) -> float:
        # README @ cc2700db: predict(input_path=...) "Returns a float value".
        return float(self.model.predict(input_path=str(wav_path)))

    def describe(self) -> dict[str, Any]:
        return {"backend": "utmosv2", "label": "predicted MOS"}


def make_asr(cfg: BenchConfig, backend: str | None = None) -> tuple[ASRBackend | None, str | None]:
    backend = backend or cfg.quality.asr_backend
    q = cfg.quality
    if backend == "none":
        return None, "disabled"
    if backend == "fake":
        return FakeASR(q.fake_error_every), None
    try:
        return WhisperASR(q.asr_model, q.asr_revision, q.asr_language, q.asr_device, q.asr_batch_size), None
    except ImportError as e:
        return None, f"whisper unavailable ({e}); install the [eval] extra"


def make_mos(cfg: BenchConfig, backend: str | None = None) -> tuple[UtmosV2 | None, str | None]:
    backend = backend or cfg.quality.mos_backend
    if backend == "none":
        return None, "disabled"
    try:
        return UtmosV2(), None
    except ImportError as e:
        return None, f"utmosv2 unavailable ({e}); see scripts/setup_gpu_box.sh"


# ----------------------------------------------------------------------------- evaluation
def evaluate_run(
    run_dir: str | Path,
    cfg: BenchConfig,
    asr_backend: str | None = None,
    mos_backend: str | None = None,
    asr: ASRBackend | None = None,
    log: Any = print,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    if not rows:
        raise FileNotFoundError(f"no results.jsonl rows in {run_dir}")
    asr_reason = None
    if asr is None:
        asr, asr_reason = make_asr(cfg, asr_backend)
    mos, mos_reason = make_mos(cfg, mos_backend)
    if asr_reason:
        log(f"WER skipped: {asr_reason}")
    if mos_reason:
        log(f"predicted MOS skipped: {mos_reason}")

    ok_rows = [r for r in rows if r["ok"] and r.get("audio_path")]
    hyps: dict[int, str] = {}
    if asr is not None and ok_rows:
        paths = [run_dir / r["audio_path"] for r in ok_rows]
        texts = asr.transcribe(paths, [r["text"] for r in ok_rows])
        hyps = {id(r): t for r, t in zip(ok_rows, texts, strict=True)}

    qrows = []
    for r in rows:
        synth_ok = bool(r["ok"] and r.get("audio_path"))
        q: dict[str, Any] = {
            "run_id": r["run_id"],
            "engine": r["engine"],
            "cell": r["cell"],
            "prompt_id": r["prompt_id"],
            "repeat": r["repeat"],
            "failed_synthesis": not synth_ok,
            "normalizer": NORMALIZER_VERSION,
        }
        if asr is not None:
            hyp = hyps.get(id(r), "") if synth_ok else ""
            e = word_errors(r["text"], hyp)
            q.update(
                hypothesis=hyp,
                ref_norm=normalize(r["text"]),
                hyp_norm=normalize(hyp),
                S=e.substitutions,
                D=e.deletions,
                I=e.insertions,
                N=e.ref_words,
                wer=e.wer,
            )
        if mos is not None:
            q["pred_mos"] = mos.predict(run_dir / r["audio_path"]) if synth_ok else None
        qrows.append(q)

    with (run_dir / "quality.jsonl").open("w", encoding="utf-8") as f:
        for q in qrows:
            f.write(json.dumps(q) + "\n")

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in qrows:
        by_cell[q["cell"]].append(q)
    nb = cfg.quality.bootstrap_samples
    cells = {}
    for cell, qs in by_cell.items():
        c: dict[str, Any] = {"n": len(qs), "n_failed_synthesis": sum(q["failed_synthesis"] for q in qs)}
        if asr is not None:
            est = bootstrap_ratio_ci([q["S"] + q["D"] + q["I"] for q in qs], [q["N"] for q in qs], n_boot=nb, seed=cfg.run.seed)
            c["corpus_wer"] = {"value": est.value, "lo": est.lo, "hi": est.hi}
        if mos is not None:
            vals = [q["pred_mos"] for q in qs if q.get("pred_mos") is not None]
            est = bootstrap_ci(vals, mean, n_boot=nb, seed=cfg.run.seed)
            c["pred_mos"] = {"value": est.value, "lo": est.lo, "hi": est.hi, "n": est.n}
        cells[cell] = c
    summary = {
        "run_id": run_dir.name,
        "normalizer": NORMALIZER_VERSION,
        "asr": asr.describe() if asr is not None else {"skipped": asr_reason},
        "mos": mos.describe() if mos is not None else {"skipped": mos_reason},
        "cells": cells,
    }
    (run_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {run_dir / 'quality_summary.json'}")
    return summary


def load_quality(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "quality.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
