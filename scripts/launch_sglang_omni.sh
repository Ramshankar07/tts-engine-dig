#!/usr/bin/env bash
# Launch SGLang-Omni serving Qwen3-TTS CustomVoice (docs/cookbook/qwen3_tts.md @ b880acf8):
#   sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#     --config examples/configs/qwen3_tts_1_7b_customvoice.yaml --port 8000
# First startup can take several minutes while CUDA graphs are captured and the
# streaming decode step is torch.compile'd.
#
# Env:
#   PORT=8000
#   SGL_CONFIG=<yaml>   default: the pinned checkout's qwen3_tts_1_7b_customvoice.yaml
#                       (stream_chunk_ramp variants go in a copy of this file, see cookbook
#                        "First-audio chunk ramp")
#   PREFIX="nsys launch ..."   command prefix (profile_nsys.sh)
#   EXTRA_ARGS="..."    e.g. "--tts_engine.engine.max_running_requests 64"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# JIT kernel builds (FlashInfer etc.) look up ninja/nvcc on PATH, so expose the venv first.
export PATH="$VENVS/sglang/bin:$PATH"

PORT="${PORT:-8000}"
MODEL="${MODEL:-$MODEL_DIR}"
SGL_CONFIG="${SGL_CONFIG:-$SRC/sglang-omni/examples/configs/qwen3_tts_1_7b_customvoice.yaml}"
SGL="$VENVS/sglang/bin/sgl-omni"
[[ -x "$SGL" ]] || { log "missing $SGL - run scripts/setup_gpu_box.sh sglang"; exit 1; }

# --model-path outranks the config file's model_path when both are given
# (sglang_omni/cli/config.py @ b880acf8), so the pinned local snapshot is served.
ARGS=(serve --model-path "$MODEL" --config "$SGL_CONFIG" --port "$PORT")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

log "sglang-omni @ $(git -C "$SRC/sglang-omni" rev-parse --short HEAD 2>/dev/null || echo '?'): $SGL ${ARGS[*]}"
# shellcheck disable=SC2086
exec ${PREFIX:-} "$SGL" "${ARGS[@]}"
