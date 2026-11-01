# Streaming vs full - vllm_omni

Run `20261013-232410-vllm_omni-streaming`.

Verdict rules: *helps* = p50 TTFA reduced >= 50%, <= 5% of requests with an audible stall, paired median total-time overhead <= 20%. Quality columns are informational (one sample per prompt; confirm by listening).

## Concurrency 1

| setting | n | failed | TTFA p50 s | TTFA reduction | audible TTFA p50 s | total p50 s | overhead p50 | reqs stalling | stall ms mean | stall ms max | WER % | pred. MOS | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 70 | 0 | 0.676 | n/a | 0.676 | 0.676 | 0.0% | 0.0% | 0.0 | 0.0 | 1.50 | 3.21 | baseline |
| stream (engine default) | 70 | 0 | 0.020 | 97.0% | 0.179 | 0.671 | -0.6% | 97.1% | 77.7 | 82.2 | 1.50 | 3.19 | mixed |
| stream (initial 1 frames) | 70 | 0 | 0.020 | 97.1% | 0.178 | 0.669 | -0.7% | 97.1% | 77.7 | 81.5 | 1.50 | 3.24 | mixed |
| stream (initial 4 frames) | 70 | 0 | 0.018 | 97.3% | 0.066 | 0.673 | 0.9% | 0.0% | 0.0 | 0.0 | 1.50 | 3.29 | helps |
| stream (initial 12 frames) | 70 | 0 | 0.018 | 97.3% | 0.115 | 0.675 | 0.8% | 0.0% | 16.3 | 19.0 | 1.50 | 3.25 | helps |

**Conclusion:** Streaming helps at concurrency 1: best setting 'stream (initial 4 frames)' (lowest audible TTFA) cuts p50 audible TTFA 0.676s -> 0.066s (first payload 0.018s) with 0.0% of requests stalling >50 ms and +0.9% median total-time overhead.

## Concurrency 4

| setting | n | failed | TTFA p50 s | TTFA reduction | audible TTFA p50 s | total p50 s | overhead p50 | reqs stalling | stall ms mean | stall ms max | WER % | pred. MOS | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 70 | 0 | 0.881 | n/a | 0.881 | 0.881 | 0.0% | 0.0% | 0.0 | 0.0 | 2.22 | 3.25 | baseline |
| stream (engine default) | 70 | 0 | 0.026 | 97.1% | 0.219 | 0.829 | -0.1% | 98.6% | 111.7 | 130.4 | 2.46 | 3.31 | mixed |
| stream (initial 1 frames) | 70 | 0 | 0.026 | 97.0% | 0.219 | 0.804 | -0.6% | 98.6% | 110.6 | 130.0 | 2.10 | 3.33 | mixed |
| stream (initial 4 frames) | 70 | 0 | 0.026 | 97.0% | 0.088 | 0.794 | 4.1% | 1.4% | 3.5 | 56.6 | 2.28 | 3.27 | helps |
| stream (initial 12 frames) | 70 | 0 | 0.026 | 97.1% | 0.148 | 0.809 | 3.7% | 30.0% | 46.5 | 112.3 | 2.52 | 3.30 | mixed |

**Conclusion:** Streaming helps at concurrency 4: best setting 'stream (initial 4 frames)' (lowest audible TTFA) cuts p50 audible TTFA 0.881s -> 0.088s (first payload 0.026s) with 1.4% of requests stalling >50 ms and +4.1% median total-time overhead.
