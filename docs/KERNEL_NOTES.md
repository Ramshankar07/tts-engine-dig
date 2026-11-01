# Kernel notes

Measured on Modal, 2026-10-13/14. Every number below comes from a trace or
results file under `runs/` (pulled locally to `modal-runs/`); file names are
given per section. Lines marked **(inference)** are interpretations of those
numbers, not separate measurements.

Groups are assigned by kernel-name regex (`src/tts_bench/profile.py`). They
are approximate, and SGLang's hand-fused kernels make its group shares not
comparable to the other two engines (see "Classification caveats").

## Model structure (config.json @ `0c0e3051`)

| component | layers | hidden | heads (q/kv) | runs |
|---|---|---|---|---|
| Talker LM | 28 | 2048 | 16 / 8 | once per 12.5 Hz codec frame (codebook 0) |
| Code predictor (sub-talker) | 5 | 1024 | 16 / 8 | 15 AR steps per frame (codebooks 1-15) |
| Codec decoder | transformer (sliding window 72) + causal conv / transposed-conv upsampling, SnakeBeta | | | per chunk or per utterance |

Per generated frame the code predictor runs 5 x 15 = 75 layer-steps against
the talker's 28, a ratio of **2.68**. This ratio shows up directly in the
kernel launch counts of both serving engines and is used below to attribute
kernels to the two LMs.

## Common setup

| item | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (Modal), driver 580.95.05 for the benchmark/profiling containers |
| CUDA toolkit (image) | 13.0.1 (`nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04`); nsys 2025.3.2 |
| official venv | torch 2.8.0+cu128, flash-attn 2.8.3 (prebuilt wheel), transformers 4.57.3, qwen-tts 0.1.1, cuDNN 9.10 |
| vLLM venv | vllm 0.31.0, vllm-omni @ `fa8b4199`, torch 2.13.0+cu130, flashinfer 0.7.0.post1, triton 3.7.1, cuDNN 9.20 |
| SGLang venv | sglang 0.5.21, sglang-omni 0.1.8 @ `b880acf8`, torch 2.13.0 (cu13), flashinfer 0.6.18, triton 3.7.1, cuDNN 9.20 |
| model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` @ `0c0e3051`, bf16 (all 1,916,676,352 params bf16, `profile_meta.json`) |
| voice / sampling | Ryan, English; temperature 0.9, top_k 50, top_p 1.0, repetition penalty 1.05 |

Workloads:
- **Official, torch.profiler:** prompts `p1-001`, `p1-002`, batch 1 (`runs/profile-official-20261013-234313/`).
- **Official, nsys:** same 2 prompts, but the capture also includes the 3 warmup requests because the engine is in-process (`runs/nsys-official-20261013-234836/`). Its totals are ~2x the torch.profiler totals for that reason; use torch.profiler numbers for the official engine.
- **vLLM-Omni / SGLang-Omni, nsys:** prompts `p1-001`..`p1-008`, full + stream at concurrency 1 (16 requests). Server startup, weight loading, CUDA-graph capture and 4 warm requests are outside the capture window (`nsys start/stop`). Runs: `runs/nsys-vllm_omni-20261013-234944/`, `runs/nsys-sglang_omni-20261013-235611/`.

## Cross-engine summary

| engine | kernel launches / request | launches / audio frame | GPU kernel ms per audio s | kernel time / wall time | attention % | GEMM % | codec-conv % | other % |
|---|---|---|---|---|---|---|---|---|
| official (PyTorch) | 236,587 | ~8,160 | ~219 | ~0.16 | 3.1 | 26.1 | 0.5 | 70.3 |
| vLLM-Omni | 173,697 | 2,071 | 100.6 | 1.07 | 21.3 | 52.4 | 3.2 | 23.0 |
| SGLang-Omni | 95,922 | 1,076 | 70.1 | 0.86 | 13.0 | 48.0 | 1.5 | 37.5 (mostly fused norm+proj kernels) |

How the derived columns are computed:
- Audio frames = audio seconds x 12.5.
- For the official engine, launches per frame and kernel ms per audio s use the same two prompts' audio from the unprofiled bench run (`20261013-223813-official`, `full-c1`: 4.64 s total). Profiled generations are seeded identically, but their audio length was not recorded separately, hence "~".
- Kernel time / wall time: summed kernel durations divided by wall time.
  - **Official:** divided by the unprofiled generation time of those prompts (6.52 s). torch.profiler itself slowed generation by about 2.3x (15.25 s inside `lm_generate`), so the profiled wall time is not used.
  - **Servers:** divided by the benchmark wall time inside the capture window. vLLM exceeds 1.0 because its talker and Code2Wav stages run in separate processes whose kernels overlap in time (inference). So this column is not an idle fraction for the servers.

## Official Qwen/PyTorch

**Time split** (`results.jsonl` of `20261013-223813-official`, full-c1, n=70):

| phase | ms per request (p50) | share of summed time |
|---|---|---|
| LM generate (talker + code predictor) | 11,550 | 99.6% |
| codec decode | 62 | 0.4% |

**Group table** (torch.profiler, record_function ranges excluded):

| group | time ms | % | launches |
|---|---|---|---|
| attention | 31.29 | 3.1 | 6,046 |
| gemm | 265.37 | 26.1 | 51,144 |
| codec-conv | 5.57 | 0.5 | 151 |
| other | 714.84 | 70.3 | 415,833 |

**Dominant kernels:**
1. **GEMM:** `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` (cuBLAS nvjet; 203 ms, 36,884 launches, 5.5 us avg). This is the generic (non-`sm90`) nvjet variant at a 64x8 tile, consistent with M=1 decode GEMMs.
2. **Unfused RMSNorm:** `reduce_kernel<MeanOps>` (25,084 launches), `pow_tensor_scalar`, `rsqrt`, `CUDAFunctorOnSelf_add`, binary mul, with identical launch counts of 25,084. That is the HF RMSNorm written as separate pow, mean, add-eps, rsqrt and mul ops: about 5 launches per norm.
3. **Rotary embedding:** `neg_kernel` plus `CatArrayBatchedCopy` (rotate_half), each about 12-25k launches.
4. **Attention:** `flash::flash_fwd_kernel` (FA2 2.8.3 via `attn_implementation="flash_attention_2"`, confirmed in `profile_meta.json`). 5.0 us avg; only 3.1% of kernel time.
5. **Sampling:** `sbtopk::gatherTopK` and `softmax_warp_forward`, under 0.2% each.
6. **Codec decode:** 151 conv launches, 0.5%.

**Observations:**
- Measured: about 8,000 kernel launches per 80 ms audio frame, averaging 2.1 us each. Summed kernel time is about 16% of generation wall time.
- (inference) The GPU is idle about 84% of the time, waiting for Python and HF `generate` to launch the next tiny kernel. The official engine is launch-bound. Faster kernels would barely help it; it needs fewer launches (fusion) and no per-launch CPU cost (CUDA graphs).

## vLLM-Omni

**Group table** (`runs/nsys-vllm_omni-20261013-234944/kernels_grouped.md`):

| group | time ms | % | launches |
|---|---|---|---|
| attention | 2,300.47 | 21.3 | 644,488 |
| gemm | 5,663.16 | 52.4 | 923,150 |
| codec-conv | 350.57 | 3.2 | 7,638 |
| other | 2,486.62 | 23.0 | 1,203,872 |

**Dominant kernels:**
1. **GEMM:** `nvjet_sm90_tst_64x8_*`, `nvjet_sm90_tst_64x24_*`, `..._splitK_*` and `cublasLt::splitKreduce_kernel`. These are Hopper-specific (`sm90`) nvjet variants; together they make up over 50% of kernel time.
2. **Two attention paths:**
   - `pytorch_flash::flash_fwd_kernel` (FA2 inside torch SDPA): 125,400 launches, 7.9 us avg.
   - `flash::FlashAttnFwdSm90` (FA3) plus `FlashAttnFwdCombine`, with `vllm::reshape_and_cache_flash_kernel` writing the paged KV cache: about 46,650 launches.
   - The launch ratio is 125,400 / 46,648 = **2.69**, matching the model's 2.68. (inference) The code predictor runs through torch SDPA (FA2), and the talker runs through vLLM's paged FA3 backend.
3. **Inductor-fused elementwise:** `triton_per_fused_*` / `triton_poi_fused_*` / `triton_red_fused_fused_add_rms_norm_*`. These fuse RMSNorm, rotary, SDPA input prep and SiLU-mul, produced by torch.compile (Inductor), and replace the official engine's 5-launch RMSNorm. The ones with 125,400 launches are the code predictor; those with 46,816 are the talker.
4. **Sampling:** `sbtopk::gatherTopK<bf16>` (306 ms), `warpMergeSortKVInPlace` (149 ms) and `ArgMaxOps` reduce (120 ms), together **5.3%** of kernel time. Top-k is done with torch sort-based kernels, about 25k launches (one per code-predictor step) (inference).
5. **Codec:** cuDNN `implicit_convolve_sgemm` (252 launches, 823 us avg) plus other convs: 3.2%. Large per-launch convolution work, consistent with chunked decoding that re-runs 72 frames of left context per 25-frame chunk (inference from the documented chunk config, not separately traced).

**Measured differences from the official engine:**
- Per audio frame: 2,071 launches vs about 8,160 (3.9x fewer). Kernel time per audio second: 100.6 ms vs about 219 ms (2.2x less).
- The engine is no longer launch-bound: summed kernel time exceeds wall time (1.07), because the two stages run concurrently (inference).
- GEMM share rises from 26% to 52%, and the elementwise "other" share falls from 70% to 23%. GPU time moves to the matmuls, as expected once the elementwise ops are fused.

## SGLang-Omni

**Group table** (`runs/nsys-sglang_omni-20261013-235611/kernels_grouped.md`):

| group | time ms | % | launches |
|---|---|---|---|
| attention | 1,038.52 | 13.0 | 342,586 |
| gemm | 3,838.08 | 48.0 | 528,830 |
| codec-conv | 118.52 | 1.5 | 12,780 |
| other | 2,997.20 | 37.5 | 650,550 |

**Dominant kernels:**
1. **`nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`** (21.5%, 174,040 launches): cuBLAS GEMM.
2. **`gemv_add_kernel`** (18.5%, 251,400 launches = 2 x 125,700): a custom GEMV with a fused residual add. With 125,700 per-step launches (inference: code predictor, 2 GEMVs per layer-step), it takes the M=1 projections off cuBLAS.
3. **`norm_qkv_rope_store_kernel`** (14.8%, 125,700) and **`norm_gate_up_silu_kernel`** (12.8%, 125,700): hand-fused code-predictor layer kernels. Judging by name, each fuses RMSNorm with the QKV projection, rotary and KV-cache store, or with the gate/up projection and SiLU. One launch replaces 5-10 separate ones.
4. **Attention:** `cudnn_generated_fort_native_sdpa_*` (cuDNN SDPA, 117,320 launches, 3.1 us avg; inference: code predictor) and `FlashAttnFwdSm90` (FA3, 46,872; talker). The ratio is again 2.68 when the two cuDNN SDPA variants are combined (117,320 + 8,380 = 125,700).
5. **Sampling:** `seeded_top_k_chunk_kernel` + `seeded_top_k_merge_sample_kernel`, about 160 ms (**2.0%**) for the same number of sampling steps. That is less than half of vLLM's sort-based path.
6. **Codec:** only 27 `implicit_convolve_sgemm` launches (vs 252 in vLLM), plus a custom `snake_beta_channels_last_kernel`; the codec group is 1.5%. (inference) Consistent with the stateful incremental decoder that the SGLang-Omni docs describe: it does not re-decode left context per chunk.

**Measured differences:**
- Compared with vLLM-Omni on the identical workload, SGLang-Omni uses 45% fewer launches per request (95,922 vs 173,697), 48% fewer per frame, and 30% less kernel time per audio second (70.1 vs 100.6 ms).
- The biggest single difference is in the code predictor: per layer-step, SGLang issues a few custom fused kernels, where vLLM issues several Inductor-generated Triton kernels plus cuBLAS GEMMs and torch SDPA.

## How the profiles explain the benchmark

From `reports/p1_report.md`, concurrency 1, full mode: the official engine has RTF 1.38 (slower than real time), vLLM-Omni 0.082 and SGLang-Omni 0.071.

1. **Official vs servers (17-19x throughput):** almost entirely launch overhead. Of each 80 ms frame, about 16% is spent in kernels; the rest is CPU-side launch and Python time for about 8k tiny kernels. Both servers replay CUDA graphs and fuse elementwise work, cutting launches 4-8x and removing per-launch CPU cost (inference for the CUDA graph part: graph replay is the documented default in both engines and is consistent with kernel time ≥ 86% of wall time).
2. **SGLang vs vLLM (1.13x throughput at c=1, 1.27x at c=8):** the code predictor dominates launch counts (75 layer-steps per frame vs the talker's 28). SGLang's hand-fused predictor kernels, GEMV and seeded top-k sampler do the same per-frame work in about half the launches and 30% less kernel time. The incremental codec decoder also avoids vLLM's left-context re-decoding (3.2% vs 1.5% codec share).
3. **Attention is not the bottleneck anywhere:** 3-21% of kernel time. For this model, short sequences (prompt + up to a few hundred frames) and batch-1 decode make the per-step projections and small-op overhead matter more than attention.

## Classification caveats

- SGLang's `norm_qkv_rope_store_kernel` and `norm_gate_up_silu_kernel` land in "other" by regex, although by name they include projection work. Its 37.5% "other" is mostly these fused layer kernels, not overhead.
- vLLM's Triton kernels whose names contain `scaled_dot_product_flash_attention` are classified as attention, but they are fused prep ops around SDPA, not the attention kernel itself.
- The official engine's nsys capture includes 3 warmup requests, so use the torch.profiler numbers for it.
- Attributing kernels to talker vs code predictor rests on launch-count ratios, not on NVTX ranges. Adding NVTX ranges per stage would make it direct.
