# Sources read for the engine adapters

All fetched on 2026-10-13 from raw.githubusercontent.com / the GitHub API at the
exact commits below. Adapter request fields and launch commands come from these
files; anything not confirmed here is marked `# VERIFY:` in code.

| Repository | Commit | Commit date (UTC) | Files read |
|---|---|---|---|
| [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` | 2026-03-17 06:38 | `README.md`, `pyproject.toml` (qwen-tts 0.1.1), `qwen_tts/inference/qwen3_tts_model.py`, `qwen_tts/core/models/modeling_qwen3_tts.py` (class list), `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` (class list, `chunked_decode`), `examples/test_model_12hz_custom_voice.py` |
| [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) | `fa8b4199c6cece6f61dc5d8d990817d3555c70bc` | 2026-10-13 19:06 | `recipes/Qwen/Qwen3-TTS.md`, `docs/user_guide/examples/online_serving/text_to_speech.md`, `docs/serving/speech_api.md`, `vllm_omni/deploy/qwen3_tts.yaml`, `docs/getting_started/installation/gpu/cuda.inc.md`, `vllm_omni/entrypoints/openai/protocol/audio.py` (request model), `vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py` (voice lowercasing), `vllm_omni/entrypoints/openai/api_server.py` (`/health`) |
| [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) | `b880acf8f04fc782115c323cf2818b92241b063a` | 2026-10-13 18:36 | `docs/basic_usage/tts.md`, `docs/cookbook/qwen3_tts.md`, `docs/get_started/installation.md`, `examples/configs/qwen3_tts_1_7b_customvoice.yaml`, `sglang_omni/serve/openai_api.py` (routes, `X-Sample-Rate` header), `sglang_omni/models/qwen3_tts/request_builders.py` (seed + sampling resolution), `sglang_omni/cli/config.py` (`--model-path` precedence) |
| [sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | `cc2700db57bb83ee13dc31ebe1b868c254e15d09` | 2026-04-02 21:37 | `README.md` (`create_model(pretrained=True)`, `predict(input_path=...)`) |

Hugging Face (API, 2026-10-13):

| Model | Revision (sha) | lastModified | Files read |
|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `0c0e3051f131929182e2c023b9537f8b1c68adfe` | 2026-01-29 | `generation_config.json` (sampling defaults), `speech_tokenizer/config.json` (24 kHz, decode_upsample_rate 1920 -> 12.5 Hz, 16 codebooks, decoder sliding_window 72) |
| `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | 2024-08-12 | - (revision pin only) |

Release context: vllm-omni latest releases v0.31.0rc1 (2026-10-06) and v0.30.0
(2026-09-25); sglang-omni v0.1.8 (2026-10-12); qwen-tts on PyPI 0.1.1.

## Facts the adapters rely on

### Official `qwen_tts` (0.1.1)
- `Qwen3TTSModel.from_pretrained(path, device_map=..., dtype=..., attn_implementation=...)`; kwargs go to `AutoModel.from_pretrained` only (the processor is loaded without them), hence the pre-downloaded local snapshot to pin the revision.
- `generate_custom_voice(text, speaker, language=None, instruct=None, non_streaming_mode=True, **kwargs) -> (List[np.ndarray], sr)`; accepts lists (static batch). Generation kwargs: `do_sample, top_k, top_p, temperature, repetition_penalty, subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens` merged with `generate_config` from the checkpoint.
- Internally: `self.model.generate(...)` then `self.model.speech_tokenizer.decode([{"audio_codes": c} ...])`.
- **No streaming generation API.** `non_streaming_mode=False` "only simulates streaming text input ... rather than enabling true streaming input or streaming generation" (docstring).
- Speakers (README table): Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee. English-native: Ryan, Aiden. Validation is case-insensitive.
- FlashAttention 2 recommended, optional; only with fp16/bf16.

### vLLM-Omni
- Launch: `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --deploy-config vllm_omni/deploy/qwen3_tts.yaml --omni --port 8091`.
- `POST /v1/audio/speech` fields used: `input, voice, language, task_type, response_format, max_new_tokens, seed, stream, stream_format, initial_codec_chunk_frames`. No per-request temperature/top_k for Qwen3-TTS; sampling comes from the deploy YAML `default_sampling_params` (temperature 0.9, top_k 50, max_tokens 4096, min_tokens 2, repetition_penalty 1.05) and `subtalker_sampling_params` (0.9 / 50 / 1.0).
- Raw streaming: `stream=true, stream_format="audio", response_format="pcm"` -> 16-bit LE PCM, native 24 kHz mono; headers `x-audio-sample-rate`, `x-audio-channels`. SSE (`stream_format="sse"`) also exists; not used.
- Chunking (deploy YAML): `codec_chunk_frames: 25` ("must match the decoder sliding attention window"), `codec_left_context_frames: 72`, `initial_codec_chunk_frames: 1`; optional `codec_chunk_ramp` and adaptive ramp. Stage 1 (Code2Wav) `max_num_seqs: 64` in the YAML; docs give conflicting tuning advice for CustomVoice (`max_num_seqs: 1` "TTFA-optimal" in one doc, "avoid forcing ... back to 1" in another).
- Explicit `max_new_tokens` or `seed` disables the server's automatic retry on runaway generation; raw PCM streams terminate with a connection error on failure.

### SGLang-Omni
- Launch: `sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --config examples/configs/qwen3_tts_1_7b_customvoice.yaml --port 8000`.
- Install: `uv pip install --prerelease=allow "sglang-omni==0.1.8"` or from source; then `uv pip install --no-deps sox einops` and `uv pip install --no-deps qwen-tts==0.1.1` (both `--no-deps` required). Docker recommended for CUDA.
- `POST /v1/audio/speech` fields used: `input, voice, language, task_type, response_format, max_new_tokens, temperature, top_p, top_k, repetition_penalty, seed, stream, initial_codec_chunk_frames, suppress_bootstrap_silence`.
- Streaming: `stream=true, response_format="pcm"` -> raw 16-bit mono PCM; `X-Sample-Rate` header. No in-band end marker.
- Default first-chunk ramp 1 -> 2 -> 4 frames, then steady stride; `initial_codec_chunk_frames=0` = steady stride from the start; server-side `stream_chunk_ramp` in the vocoder stage config.
- `suppress_bootstrap_silence` (default true) withholds ~80 ms of silent first-frame audio for validated pairs (currently Ryan/English) - affects the first-payload content.
- Docs report that the 1.7B CustomVoice streaming run on H200 at concurrency 16 did not guarantee uninterrupted playback; these are the maintainers' numbers, not ours.

## Verified on the GPU (H100, Modal, 2026-10-13/14)
These were open `# VERIFY:` items before the GPU runs; all held:
- `profile.py`: `nsys stats` CSV columns (Nsight Systems 2025.3.2) and torch.profiler `key_averages()` attributes (torch 2.8).
- `scripts/profile_nsys.sh`: `nsys launch/start/stop` flags.
- `quality.py`: transformers 5.19 ASR pipeline kwargs with Whisper large-v3.
- flash-attn: no wheel exists for torch 2.11, and the source build failed. The official venv therefore pins torch 2.8.0 + the prebuilt `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE` wheel (`OFFICIAL_TORCH`, `FLASH_ATTN_WHEEL`).
- vllm 0.31.0 needs torch 2.13.0, which has no cu128 build; the vLLM venv uses cu130 and runs on driver 580.95.
- Seeded requests at concurrency 1 are bit-identical across runs on both servers (md5 of WAVs from two separate runs). At concurrency >= 4, vLLM-Omni output length varies between runs, while SGLang-Omni's does not.

Still open:
- The official engine's concurrency 4/8 cells are static batches, not a request scheduler (by design; see README).
- Kernels are attributed to talker vs code predictor by launch-count ratio, not NVTX ranges (docs/KERNEL_NOTES.md).
