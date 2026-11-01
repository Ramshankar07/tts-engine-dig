"""Real-time playback buffer simulator.

Given when each audio chunk arrived (seconds since request submit) and how long
it plays, simulate a client that starts playing once `prebuffer_ms` of audio is
buffered (or the stream ends), plays at real-time speed, and stalls whenever the
buffer runs dry. A stall ends when the next chunk arrives; playback resumes
immediately (no re-buffering). This is the standard "underrun" model used for
streaming TTS continuity scoring.

Outputs: playback start time, number of underruns, total/max stall, the number
of stalls longer than an audibility threshold, and when playback finished.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PlaybackStats:
    playback_start_s: float | None
    playback_end_s: float | None
    underrun_count: int
    stall_total_ms: float
    stall_max_ms: float
    stalls_over_threshold: int
    stalls_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stalls_ms"] = [round(s, 3) for s in self.stalls_ms]
        return d


def simulate_playback(
    chunks: list[tuple[float, float]],
    prebuffer_ms: float = 0.0,
    threshold_ms: float = 50.0,
    eps_ms: float = 1e-6,
) -> PlaybackStats:
    """Simulate playback.

    Args:
        chunks: list of (arrival_time_s, duration_s), in arrival order.
            Zero-duration chunks are ignored.
        prebuffer_ms: audio that must be buffered before playback starts. If the
            stream ends before that, playback starts at the last arrival.
        threshold_ms: stalls strictly longer than this count as audible.
        eps_ms: stalls shorter than this are treated as float noise, not stalls.
    """
    chunks = [(float(t), float(d)) for t, d in chunks if d > 0]
    if not chunks:
        return PlaybackStats(None, None, 0, 0.0, 0.0, 0, [])
    if any(chunks[i][0] < chunks[i - 1][0] for i in range(1, len(chunks))):
        raise ValueError("chunk arrival times must be non-decreasing")

    prebuffer_s = prebuffer_ms / 1000.0
    # 1) Find the playback start: first arrival at which buffered audio >= prebuffer.
    buffered = 0.0
    start_idx = len(chunks) - 1
    for i, (_t, d) in enumerate(chunks):
        buffered += d
        if buffered + 1e-12 >= prebuffer_s:
            start_idx = i
            break
    clock = chunks[start_idx][0]  # wall time at which playback starts
    start = clock
    # Everything up to start_idx is already in the buffer at `start`.
    play_until = clock + sum(d for _t, d in chunks[: start_idx + 1])

    stalls: list[float] = []
    for t, d in chunks[start_idx + 1 :]:
        if t > play_until:  # buffer ran dry before this chunk arrived
            gap_ms = (t - play_until) * 1000.0
            if gap_ms > eps_ms:
                stalls.append(gap_ms)
            play_until = t + d
        else:
            play_until += d

    over = sum(1 for s in stalls if s > threshold_ms)
    return PlaybackStats(
        playback_start_s=start,
        playback_end_s=play_until,
        underrun_count=len(stalls),
        stall_total_ms=float(sum(stalls)),
        stall_max_ms=float(max(stalls) if stalls else 0.0),
        stalls_over_threshold=over,
        stalls_ms=stalls,
    )
