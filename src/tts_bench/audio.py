"""Small audio helpers: PCM16 <-> float32, WAV IO, silence detection."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf


def pcm16_bytes_to_float(buf: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono PCM -> float32 in [-1, 1]."""
    if len(buf) % 2:
        raise ValueError("PCM16 buffer must have an even number of bytes")
    return (np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0).copy()


def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).round().astype("<i2").tobytes()


class Pcm16Assembler:
    """Accumulates arbitrary byte slices from a raw PCM16 HTTP stream.

    Network reads can split a 2-byte sample; the odd byte is carried over to
    the next read so every emitted array is sample-aligned.
    """

    def __init__(self) -> None:
        self._carry = b""

    def push(self, data: bytes) -> np.ndarray:
        data = self._carry + data
        cut = len(data) - (len(data) % 2)
        self._carry = data[cut:]
        return pcm16_bytes_to_float(data[:cut]) if cut else np.zeros(0, dtype=np.float32)

    @property
    def leftover(self) -> int:
        return len(self._carry)


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV/FLAC response body -> (mono float32, sample_rate)."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(sr)


def first_nonsilent_sample(audio: np.ndarray, threshold: float = 1e-3, win: int = 240) -> int | None:
    """Index of the first `win`-sample window whose peak exceeds `threshold`.

    Used for "time to first *audible* audio": SGLang-Omni's docs note that the
    first PCM payload can be a silent bootstrap frame (~80 ms), so TTFA of the
    first byte and of the first audible sample are both recorded.
    """
    a = np.abs(np.asarray(audio, dtype=np.float32))
    if a.size == 0:
        return None
    n = (a.size + win - 1) // win
    padded = np.zeros(n * win, dtype=np.float32)
    padded[: a.size] = a
    peaks = padded.reshape(n, win).max(axis=1)
    hits = np.nonzero(peaks > threshold)[0]
    if hits.size == 0:
        return None
    w = int(hits[0])
    seg = a[w * win : (w + 1) * win]
    return w * win + int(np.argmax(seg > threshold))
