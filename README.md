# tts-engine-dig

Benchmark harness: serve one TTS checkpoint,
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, on three inference engines and compare
them under matched settings:

| engine | how it runs | full output | true streaming |
|---|---|---|---|
| official Qwen/PyTorch (`qwen-tts` 0.1.1) | in-process, HF transformers | yes | **no**: the package has no streaming generation API (see docs/SOURCES.md) |
| vLLM-Omni | HTTP client -> `vllm serve ... --omni` | yes | yes (raw PCM) |
| SGLang-Omni | HTTP client -> `sgl-omni serve` | yes | yes (raw PCM) |
| fake | in-process, deterministic synthetic audio | yes | yes; CPU-only test double |

It measures time to first audio (TTFA), total generation time, real-time factor,
throughput and GPU memory at concurrency 1/4/8. It also simulates playback gaps
from chunk arrival times, scores quality (WER via Whisper large-v3, UTMOSv2
predicted MOS), compares streaming with full output across chunk sizes, and
collects kernel profiles (torch.profiler + Nsight Systems).

## Status

- **Measured on 2026-10-13/14** on one NVIDIA H100 80GB (Modal). All five plan
  sessions ran on the full 70-prompt set. Results are in
  [`results/`](results/) and summarized under [Results](#results) below.
- Kernel analysis is in [`docs/KERNEL_NOTES.md`](docs/KERNEL_NOTES.md).
  The demo audio is in [`demo/samples/`](demo/samples/).
- All numbers apply to this GPU, these pinned engine versions, one voice
  (Ryan, English), batch-1 to concurrency-8 load, and the sampling settings in
  `configs/p1_run.toml`. They are not general claims about the engines.
- Not done: human listening tests (WER and predicted MOS are proxies), and
  repeated runs on different hosts or GPUs.
- The CPU test suite still covers everything with the fake engine: `uv run pytest -q`.

## Layout

```
configs/p1_run.toml        pinned run config (model+revision, voice Ryan/English, bf16, sampling, seed, concurrency 1/4/8, chunk sweep)
configs/fake_cpu.toml      fast CPU smoke config (fake engine, fake ASR)
prompts/p1_prompts.jsonl   70 original English prompts (32 short / 28 medium / 10 long; 10 categories)
src/tts_bench/
  engines/                 base.py (interface), official.py, vllm_omni.py, sglang_omni.py, http_speech.py, fake.py
  bench.py                 runner -> runs/<ts>-<engine>/{config,env,cells}.json, results.jsonl, audio/
  playback.py              real-time playback-buffer simulator (underruns, stall ms)
  quality.py, textnorm.py  WER (corpus, frozen normalizer tts-bench-norm-v1), UTMOSv2 predicted MOS
  streaming.py             streaming vs full across initial_codec_chunk_frames, verdict table
  profile.py               torch.profiler wrapper + nsys CSV parser, kernel grouping
  report.py, stats.py      markdown tables, bootstrap CIs, paired WER differences
  gpu.py, envinfo.py       NVML/nvidia-smi memory polling, env snapshot
scripts/                   setup_gpu_box.sh, launch_*.sh, run_all.sh (sessions 1-6), profile_nsys.sh, debug_server.sh, record_demo.sh
demo/stream_demo.py        stream one utterance to WAV with live CLI playback
docs/SOURCES.md            commits/dates read, facts relied on, open VERIFY items
docs/KERNEL_NOTES.md       per-engine kernel analysis from the measured traces
infra/modal_p1.py          Modal app that runs the GPU sessions (image = setup_gpu_box.sh, runs on a volume)
results/                   measured outputs: p1_report.md/csv, per-run results/quality/streaming files, kernel tables
demo/samples/              recorded streaming demo (WAV + chunk-arrival logs)
tests/                     CPU-only pytest suite (fake engine)
```

## Quickstart on CPU (fake engine)

```bash
uv sync                      # core + dev deps only (no torch)
uv run pytest -q             # full suite, about 20 s

uv run tts-bench validate-prompts
uv run tts-bench bench     --config configs/fake_cpu.toml --engine fake --out /tmp/p1runs
uv run tts-bench quality   /tmp/p1runs/<ts>-fake --config configs/fake_cpu.toml
uv run tts-bench streaming --config configs/fake_cpu.toml --engine fake --out /tmp/p1runs
uv run tts-bench report    /tmp/p1runs/* --out /tmp/p1runs/report.md --baseline fake
uv run python demo/stream_demo.py --engine fake --play     # tones, not speech
```

## What is measured, and how

- **Matched settings** (configs/p1_run.toml): the same checkpoint revision
  (each engine loads the same local snapshot), speaker `Ryan`, language
  `English`, no instruction, bf16, sampling equal to the checkpoint's
  `generation_config.json` (temperature 0.9, top_k 50, top_p 1.0, repetition
  penalty 1.05, sub-talker 0.9/50/1.0), and `max_new_tokens=2048`. vLLM-Omni has
  no per-request sampling fields for Qwen3-TTS, so its deploy-YAML defaults
  (which match these) apply. Each prompt gets the seed `seed + index` on every
  engine. Seeds make each engine repeatable. They do not make different engines
  produce identical audio.
- **Concurrency**: for HTTP engines, N closed-loop clients send independent
  requests. For the official engine, N is a **static batch** (`generate_custom_voice`
  takes a list). There is no request scheduler in the official package. All of
  a batch's items share its completion time, so its TTFA is the batch latency.
  Report the official c=4/8 cells with this caveat.
- **TTFA** is the arrival of the first PCM payload. **Audible TTFA** is the
  arrival of the chunk holding the first non-silent sample. They differ because
  the first payload can be a silent bootstrap frame. In full mode, TTFA equals
  the total time.
- **Playback gaps**: the simulator plays chunks in real time from their arrival
  times. With `prebuffer_ms=0` it starts at the first chunk. It counts
  underruns, total and maximum stall, and stalls longer than 50 ms.
- **Memory**: the device memory poller (NVML, falling back to nvidia-smi) gives
  each request the maximum sampled while it was in flight, and each cell its
  overall peak. vLLM-Omni pre-allocates KV cache by `gpu_memory_utilization`, so
  its device memory reflects a reservation, not live demand. The official engine
  also records `torch.cuda.max_memory_allocated`.
- **Quality**: corpus WER is the sum of S+D+I over the summed reference words,
  after the frozen normalizer `tts-bench-norm-v1`. A failed synthesis counts as
  all deletions. UTMOSv2 output is always labelled *predicted MOS*. Every
  estimate gets a 95% bootstrap CI, and engine-vs-engine WER uses a paired
  bootstrap over shared prompts.
- **Streaming verdict** (configs `[streaming.verdict]`): a setting *helps* if
  p50 TTFA drops by at least 50%, at most 5% of requests stall for more than
  50 ms, and the paired median total-time overhead is at most 20%. A negative
  result is a valid outcome.

## GPU-box runbook

You need one Linux CUDA GPU (24 GB or more for the 1.7B model plus a server's
KV reservation), [Nsight Systems](https://developer.nvidia.com/nsight-systems)
for session 4, and outbound access to GitHub and Hugging Face.

```bash
git clone https://github.com/Ramshankar07/tts-engine-dig.git && cd tts-engine-dig   # or rsync it
bash scripts/setup_gpu_box.sh          # venvs: client, official, eval, vllm, sglang; model snapshot -> ~/p1/models
# settings used for the measured runs (see "Running on Modal"):
#   OFFICIAL_TORCH=2.8.0 FLASH_ATTN_WHEEL=<flash_attn-2.8.3+cu12torch2.8 wheel URL> TORCH_BACKEND=cu128 ... official
#   TORCH_BACKEND=cu130 ... vllm      (CUDA 13.0 toolkit on PATH for SGLang's DeepGEMM JIT)
source ~/p1/p1.env
```

The steps below follow the plan's **First five work sessions**. Each session is
`bash scripts/run_all.sh <n>`. Prefix it with `LIMIT=5` for a smoke pass first.

| # | plan | command | output |
|---|---|---|---|
| 1 | Load the checkpoint, generate the fixed prompt set | `bash scripts/run_all.sh 1` | `runs/<ts>-official/` (full mode, c=1/4/8 static batches) |
| 2 | Official baseline + vLLM-Omni on the same GPU | `bash scripts/run_all.sh 2` | `runs/<ts>-vllm_omni/` (full + stream, c=1/4/8) |
| 3 | Add SGLang-Omni, comparable timing/memory | `bash scripts/run_all.sh 3` | `runs/<ts>-sglang_omni/` |
| 4 | Profile all engines, document dominant kernels | `bash scripts/run_all.sh 4`, then fill `docs/KERNEL_NOTES.md` | `runs/profile-official-*/`, `runs/nsys-*/` |
| 5 | Streaming on/off, finish table, report, demo | `bash scripts/run_all.sh 5`, then `python demo/stream_demo.py --engine vllm_omni --play` | `runs/*-streaming/streaming_comparison.md`, `reports/p1_report.md` |

Manual equivalents, if you want to drive things by hand:

```bash
# official (in-process)
$VENVS/official/bin/tts-bench bench --engine official --model-path $MODEL_DIR
# vLLM-Omni
bash scripts/launch_vllm_omni.sh &          # waits on :8091/health; first start captures CUDA graphs
$VENVS/client/bin/tts-bench bench --engine vllm_omni --engine-python $VENVS/vllm/bin/python
$VENVS/client/bin/tts-bench streaming --engine vllm_omni --concurrency 1 4
# SGLang-Omni
bash scripts/launch_sglang_omni.sh &        # :8000
$VENVS/client/bin/tts-bench bench --engine sglang_omni --engine-python $VENVS/sglang/bin/python
# quality (GPU free of servers), report
$VENVS/eval/bin/tts-bench quality runs/<run>
$VENVS/client/bin/tts-bench report runs/<run1> runs/<run2> ... --out reports/p1_report.md
```

Before trusting the numbers, check that:
1. Every smoke WAV (`runs/smoke-*.wav`) sounds like the same speaker (Ryan) saying the same text.
2. The `# VERIFY:` items in docs/SOURCES.md hold on the box. For example, two identical seeded requests should return identical audio on each server.
3. `env.json` in each run shows the pinned commits and versions. Engine venv packages are recorded through `--engine-python`.
4. Run engine variants (vLLM stage-1 `max_num_seqs`, SGLang `stream_chunk_ramp`, `suppress_bootstrap_silence`) as separate, labelled runs. Do not change them inside one series.

## Running on Modal

`infra/modal_p1.py` builds an image by running `scripts/setup_gpu_box.sh` one
step per layer, mounts this repo, and keeps `runs/` and `reports/` on the
`p1-runs` volume.

```bash
modal run infra/modal_p1.py --sessions "1 2 3"           # official, vLLM-Omni, SGLang-Omni
modal run infra/modal_p1.py --sessions "4"               # profiling (parallel with 5 is fine)
modal run infra/modal_p1.py --sessions "5"               # streaming sweeps -> session 6
modal run infra/modal_p1.py --sessions "6"               # WER + predicted MOS for runs/INDEX.txt, report
modal run infra/modal_p1.py --cmd "bash scripts/debug_server.sh vllm_omni"
modal run infra/modal_p1.py --cmd "bash scripts/record_demo.sh"
modal volume get p1-runs runs ./modal-runs               # pull results
```

Image fixes that were needed (all recorded in `infra/modal_p1.py`):
- **Base image:** CUDA 13.0 devel. SGLang's DeepGEMM JIT requires nvcc >= 12.9.
- **vLLM venv:** torch cu130. vLLM 0.31.0 pins torch 2.13.0, which has no cu128 build.
- **Official venv:**
  - torch 2.8.0 with the prebuilt flash-attn 2.8.3 wheel. flash-attn has no wheel for torch 2.11, and building it from source failed.
  - `CC=gcc` for the SGLang build, because Modal's Python reports clang.
- **ninja** in the server venvs, for the FlashInfer JIT.
- **wget**, for the UTMOSv2 weight download.
- **Eval venv:** cu128 torchvision and torchaudio. The UTMOSv2 install pulled cu130 builds.

Compute used: about 3.7 H100-hours, including debugging. That is roughly $18
at Modal list prices; this is an estimate from container wall time, and
Modal's billing page has the exact figure.

## Results

H100 80GB HBM3, bf16, voice Ryan, 70 prompts, seed 1234. Values are medians
over requests unless noted; brackets are 95% bootstrap CIs. Full tables:
[`results/p1_report.md`](results/p1_report.md).

### Engine comparison (full output)

| engine | concurrency | throughput (audio s / s) | RTF p50 | time to complete audio p50 | corpus WER % | peak GPU mem |
|---|---|---|---|---|---|---|
| official (PyTorch, FA2) | 1 | 0.72 | 1.38 | 11.61 s | 2.16 [0.70, 4.08] | 6.7 GB |
| official | 4 | 1.92 | 1.81 | 14.94 s | 1.68 [0.60, 3.12] | 11.3 GB |
| official | 8 | 2.98 | 2.36 | 19.92 s | 3.18 [0.71, 7.75] | 18.9 GB |
| vLLM-Omni | 1 | 12.35 | 0.082 | 0.68 s | 1.50 [0.57, 2.68] | 27.0 GB* |
| vLLM-Omni | 4 | 37.21 | 0.100 | 0.86 s | 2.34 [1.08, 3.93] | 27.0 GB* |
| vLLM-Omni | 8 | 61.59 | 0.119 | 1.04 s | 1.74 [0.78, 2.95] | 27.1 GB* |
| SGLang-Omni | 1 | **14.00** | **0.071** | **0.63 s** | 1.56 [0.70, 2.61] | 71.5 GB* |
| SGLang-Omni | 4 | **46.57** | **0.076** | **0.70 s** | 1.56 [0.70, 2.61] | 71.7 GB* |
| SGLang-Omni | 8 | **78.18** | **0.095** | **0.76 s** | 1.50 [0.66, 2.54] | 75.1 GB* |

\* Whole-GPU memory from nvidia-smi, mostly preallocated KV/cache pools set
by each server's memory fraction. It is not the memory the workload needs.

- **Speed:**
  - Both serving engines are 17-26x faster than the official package, which runs slower than real time even at batch 1.
  - SGLang-Omni is 13% ahead of vLLM-Omni at concurrency 1 and 27% ahead at 8.
  - Rerunning the same cells gave the same numbers within 1-2%: 12.37 vs 12.35, 37.38 vs 37.21, 14.27 vs 14.00.
- **Intelligibility:** no engine is measurably worse; every WER CI overlaps.
- **Predicted MOS (UTMOSv2) cannot separate the engines:**
  - All cells fall within 3.19-3.33.
  - Seeded batch-1 runs produce bit-identical WAVs across repeated runs (md5 checked), yet UTMOSv2 scored one file five times with SD 0.16. So differences of about 0.1 between cells are predictor noise.
  - A human listening test is still needed for naturalness.
- **Concurrency changes vLLM's output:** at concurrency 4 and above, vLLM-Omni's total audio length varies between runs (685-703 s), so batching changes its output. SGLang's output stayed at exactly 708.4 s across every streaming setting.

### Does streaming help?

Yes on both servers, but vLLM-Omni needs a non-default first chunk.
Streaming vs full at concurrency 1 (per-engine
`results/runs/*-streaming/streaming_comparison.md`):

| engine / first chunk | first payload p50 | audible audio p50 | requests with a >50 ms gap | total-time change |
|---|---|---|---|---|
| vLLM-Omni full output | 676 ms | 676 ms | n/a | baseline |
| vLLM-Omni stream, default (1 frame) | 20 ms | 178 ms | **97%** (mean 78 ms) | -0.6% |
| vLLM-Omni stream, **4 frames** | 18 ms | **66 ms** | **0%** | +0.9% |
| vLLM-Omni stream, 12 frames | 18 ms | 115 ms | 0% | +0.8% |
| SGLang-Omni full output | 609 ms | 609 ms | n/a | baseline |
| SGLang-Omni stream, default (1->2->4 ramp) | 15 ms | **27 ms** | **0%** | **-10.5%** |
| SGLang-Omni stream, 12 frames | 70 ms | 70 ms | 0% | -10.7% |

- **vLLM-Omni's default stalls.** Its first payload is one silent 80 ms bootstrap frame. The next 25-frame chunk arrives at about 178 ms, so 97% of requests stall for about 78 ms right after starting. With `initial_codec_chunk_frames=4` the stalls disappear and audible audio starts 2.7x sooner. At concurrency 4 the same setting still holds: 1.4% of requests stall, audible audio starts at 88 ms, and total time rises 4%.
- **SGLang-Omni's default is already right.** Its 1->2->4 ramp gives audible audio in 27 ms with no stalls. Streaming also finishes about 10% sooner than full output. (inference) The incremental codec decoder overlaps with generation.
- **Streaming did not change intelligibility.** WER was the same as full output for every chunk setting on SGLang, and within CI on vLLM.

### Why (kernel profiles)

Details and per-kernel tables: [`docs/KERNEL_NOTES.md`](docs/KERNEL_NOTES.md).

- **The official engine is launch-bound:**
  - About 8,000 kernel launches per 80 ms audio frame, averaging 2.1 us each.
  - Summed kernel time is only about 16% of generation time.
  - 70% of kernel time goes to small unfused ops. RMSNorm alone takes 5 kernels per call.
  - Codec decoding is 0.4% of request time; the cost is entirely in the two autoregressive LMs.
- **vLLM-Omni:**
  - Cuts launches 3.9x (2,071 per frame) with Inductor-fused Triton kernels and CUDA graphs.
  - GEMM becomes 52% of kernel time.
  - It runs paged FlashAttention-3 for the talker and torch-SDPA FlashAttention-2 for the 5-layer code predictor. The launch ratio between the two attention paths, 2.69, matches the 75-vs-28 layer-steps per frame from the model config.
- **SGLang-Omni** halves launches again (1,076 per frame) and uses 30% less kernel time per audio second than vLLM. It does this with:
  - hand-fused code-predictor layer kernels (`norm_qkv_rope_store_kernel`, `norm_gate_up_silu_kernel`)
  - a custom `gemv_add_kernel`
  - a seeded top-k sampler, costing 2.0% of kernel time vs 5.3% for vLLM's sort-based sampling
  - an incremental codec decoder: 27 vs 252 large conv launches
- **Attention is 3-21% of kernel time everywhere.** For this model, per-step small-op overhead in the code predictor is what separates the engines.

### Demo

[`demo/samples/`](demo/samples/) has one 7-8 s sentence streamed from each
server:
- `vllm_omni-default.wav`
- `vllm_omni-k4.wav`
- `sglang_omni-default.wav`

Each comes with a chunk-arrival log and the simulated playback result.

The single-sample demo needs a caveat. The first vLLM `k4` request after
startup stalled once (325 ms) even though none of the 70 sweep requests did.
(inference) This looks like a one-time warmup cost at a new chunk size; the
sweep is the measurement, not the demo.
