"""Percentiles and bootstrap confidence intervals (seeded, so reports are reproducible)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n: int

    def fmt(self, digits: int = 3, scale: float = 1.0) -> str:
        if self.n == 0 or not np.isfinite(self.value):
            return "n/a"
        v, lo, hi = self.value * scale, self.lo * scale, self.hi * scale
        return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def percentile(x: Sequence[float], q: float) -> float:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return float(np.percentile(a, q)) if a.size else float("nan")


def bootstrap_ci(
    x: Sequence[float],
    stat: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI of ``stat`` over i.i.d. resamples of ``x``."""
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    point = float(stat(a))
    if a.size == 1:
        return Estimate(point, point, point, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.array([stat(a[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), int(a.size))


def bootstrap_ratio_ci(
    num: Sequence[float],
    den: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """CI for sum(num)/sum(den) resampling paired items (e.g. corpus WER:
    errors / reference words, resampled by utterance)."""
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    if n.size == 0 or d.sum() == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    point = float(n.sum() / d.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(n_boot, n.size))
    sd = d[idx].sum(axis=1)
    boots = np.where(sd > 0, n[idx].sum(axis=1) / np.maximum(sd, 1e-12), np.nan)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), int(n.size))


def paired_bootstrap_ratio_diff(
    num_a: Sequence[float],
    num_b: Sequence[float],
    den: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """CI for (sum num_a - sum num_b) / sum den over the *same* items
    (paired by prompt), e.g. corpus-WER difference between two engines."""
    a, b, d = (np.asarray(v, dtype=float) for v in (num_a, num_b, den))
    if a.size == 0 or a.size != b.size or a.size != d.size or d.sum() == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    point = float((a.sum() - b.sum()) / d.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    sd = np.maximum(d[idx].sum(axis=1), 1e-12)
    boots = (a[idx].sum(axis=1) - b[idx].sum(axis=1)) / sd
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), int(a.size))


def p50(a: np.ndarray) -> float:
    return float(np.percentile(a, 50))


def p90(a: np.ndarray) -> float:
    return float(np.percentile(a, 90))


def mean(a: np.ndarray) -> float:
    return float(np.mean(a))
