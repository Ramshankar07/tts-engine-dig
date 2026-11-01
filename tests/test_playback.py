import numpy as np
import pytest

from tts_bench.audio import Pcm16Assembler, first_nonsilent_sample, float_to_pcm16_bytes, pcm16_bytes_to_float
from tts_bench.playback import simulate_playback


def test_empty_stream():
    s = simulate_playback([])
    assert s.playback_start_s is None and s.underrun_count == 0


def test_single_chunk_no_stall():
    s = simulate_playback([(1.0, 3.0)])
    assert s.playback_start_s == 1.0
    assert s.playback_end_s == pytest.approx(4.0)
    assert s.underrun_count == 0


def test_chunk_arrives_before_buffer_runs_dry():
    s = simulate_playback([(0.1, 0.5), (0.5, 0.5)])
    assert s.underrun_count == 0
    assert s.playback_end_s == pytest.approx(1.1)


def test_one_stall_hand_computed():
    # plays 0.1-0.3, buffer dry, next chunk at 0.5 -> 200 ms stall, then plays to 0.7
    s = simulate_playback([(0.1, 0.2), (0.5, 0.2)], threshold_ms=50)
    assert s.underrun_count == 1
    assert s.stall_total_ms == pytest.approx(200.0)
    assert s.stall_max_ms == pytest.approx(200.0)
    assert s.stalls_over_threshold == 1
    assert s.playback_end_s == pytest.approx(0.7)


def test_multiple_stalls_and_threshold():
    # 0.0-0.1 play; chunk at 0.13 (30 ms stall) plays to 0.23; chunk at 0.33 (100 ms stall) plays to 0.43;
    # chunk at 0.40 arrives in time -> plays to 0.53
    chunks = [(0.0, 0.1), (0.13, 0.1), (0.33, 0.1), (0.40, 0.1)]
    s = simulate_playback(chunks, threshold_ms=50)
    assert s.underrun_count == 2
    assert s.stalls_ms == pytest.approx([30.0, 100.0])
    assert s.stall_total_ms == pytest.approx(130.0)
    assert s.stalls_over_threshold == 1
    assert s.playback_end_s == pytest.approx(0.53)


def test_prebuffer_absorbs_jitter():
    chunks = [(0.1, 0.2), (0.5, 0.2), (0.6, 0.2)]
    assert simulate_playback(chunks).underrun_count == 1  # without prebuffer: stall at 0.3-0.5
    s = simulate_playback(chunks, prebuffer_ms=300)
    assert s.playback_start_s == pytest.approx(0.5)  # 0.4 s buffered at 0.5
    assert s.underrun_count == 0
    assert s.playback_end_s == pytest.approx(1.1)


def test_prebuffer_never_reached_starts_at_last_arrival():
    s = simulate_playback([(0.1, 0.05), (0.2, 0.05)], prebuffer_ms=1000)
    assert s.playback_start_s == pytest.approx(0.2)
    assert s.underrun_count == 0


def test_rejects_out_of_order():
    with pytest.raises(ValueError):
        simulate_playback([(0.5, 0.1), (0.2, 0.1)])


def test_pcm_roundtrip_and_odd_byte_carry():
    x = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    b = float_to_pcm16_bytes(x)
    asm = Pcm16Assembler()
    a1 = asm.push(b[:3])  # 1.5 samples
    assert a1.size == 1 and asm.leftover == 1
    a2 = asm.push(b[3:])
    y = np.concatenate([a1, a2])
    np.testing.assert_allclose(y, x, atol=1e-4)
    np.testing.assert_allclose(pcm16_bytes_to_float(b), x, atol=1e-4)


def test_first_nonsilent_sample():
    a = np.zeros(1000, dtype=np.float32)
    a[537] = 0.2
    assert first_nonsilent_sample(a, 1e-3, win=100) == 537
    assert first_nonsilent_sample(np.zeros(10, dtype=np.float32)) is None
