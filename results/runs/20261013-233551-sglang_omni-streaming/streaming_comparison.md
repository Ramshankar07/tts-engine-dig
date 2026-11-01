# Streaming vs full - sglang_omni

Run `20261013-233551-sglang_omni-streaming`.

Verdict rules: *helps* = p50 TTFA reduced >= 50%, <= 5% of requests with an audible stall, paired median total-time overhead <= 20%. Quality columns are informational (one sample per prompt; confirm by listening).

## Concurrency 1

| setting | n | failed | TTFA p50 s | TTFA reduction | audible TTFA p50 s | total p50 s | overhead p50 | reqs stalling | stall ms mean | stall ms max | WER % | pred. MOS | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 70 | 0 | 0.609 | n/a | 0.609 | 0.609 | 0.0% | 0.0% | 0.0 | 0.0 | 1.56 | 3.23 | baseline |
| stream (engine default) | 70 | 0 | 0.015 | 97.6% | 0.027 | 0.536 | -10.5% | 0.0% | 0.0 | 0.0 | 1.56 | 3.27 | helps |
| stream (initial 1 frames) | 70 | 0 | 0.015 | 97.6% | 0.027 | 0.534 | -10.3% | 0.0% | 0.0 | 0.0 | 1.56 | 3.29 | helps |
| stream (initial 4 frames) | 70 | 0 | 0.030 | 95.1% | 0.030 | 0.534 | -10.7% | 0.0% | 0.0 | 0.0 | 1.56 | 3.23 | helps |
| stream (initial 12 frames) | 70 | 0 | 0.070 | 88.6% | 0.070 | 0.533 | -10.7% | 0.0% | 0.0 | 0.0 | 1.56 | 3.20 | helps |

**Conclusion:** Streaming helps at concurrency 1: best setting 'stream (initial 1 frames)' (lowest audible TTFA) cuts p50 audible TTFA 0.609s -> 0.027s (first payload 0.015s) with 0.0% of requests stalling >50 ms and -10.3% median total-time overhead.

## Concurrency 4

| setting | n | failed | TTFA p50 s | TTFA reduction | audible TTFA p50 s | total p50 s | overhead p50 | reqs stalling | stall ms mean | stall ms max | WER % | pred. MOS | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 70 | 0 | 0.705 | n/a | 0.705 | 0.705 | 0.0% | 0.0% | 0.0 | 0.0 | 1.56 | 3.25 | baseline |
| stream (engine default) | 70 | 0 | 0.018 | 97.4% | 0.034 | 0.606 | -7.3% | 0.0% | 0.0 | 0.0 | 1.56 | 3.29 | helps |
| stream (initial 1 frames) | 70 | 0 | 0.018 | 97.5% | 0.032 | 0.598 | -8.8% | 0.0% | 0.0 | 0.0 | 1.56 | 3.29 | helps |
| stream (initial 4 frames) | 70 | 0 | 0.035 | 95.1% | 0.035 | 0.601 | -8.2% | 0.0% | 0.0 | 0.0 | 1.56 | 3.24 | helps |
| stream (initial 12 frames) | 70 | 0 | 0.080 | 88.6% | 0.080 | 0.603 | -8.9% | 0.0% | 0.0 | 0.0 | 1.56 | 3.29 | helps |

**Conclusion:** Streaming helps at concurrency 4: best setting 'stream (initial 1 frames)' (lowest audible TTFA) cuts p50 audible TTFA 0.705s -> 0.032s (first payload 0.018s) with 0.0% of requests stalling >50 ms and -8.8% median total-time overhead.
