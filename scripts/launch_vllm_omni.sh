#!/usr/bin/env bash
# Launch vLLM-Omni serving Qwen3-TTS CustomVoice (recipes/Qwen/Qwen3-TTS.md @ fa8b4199):
#   vllm serve <model> --deploy-config vllm_omni/deploy/qwen3_tts.yaml --omni --port 8091
#
# Env:
#   PORT=8091
#   DEPLOY_CONFIG=<path>        default: bundled qwen3_tts.yaml in the pinned checkout
#   STAGE_OVERRIDES='<json>'    e.g. '{"1": {"max_num_seqs": 1}}' (docs disagree on the
#                               best stage-1 max_num_seqs for CustomVoice; run the default
#                               first, treat this as a separate, labelled variant)
#   NO_ASYNC_CHUNK=1            disable chunked streaming (pipeline uses the end-to-end codec path)
#   PREFIX="nsys launch ..."    command prefix (used by profile_nsys.sh)
#   EXTRA_ARGS="..."            appended verbatim
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# JIT kernel builds (FlashInfer etc.) look up ninja/nvcc on PATH, so expose the venv first.
export PATH="$VENVS/vllm/bin:$PATH"

PORT="${PORT:-8091}"
MODEL="${MODEL:-$MODEL_DIR}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$SRC/vllm-omni/vllm_omni/deploy/qwen3_tts.yaml}"
VLLM="$VENVS/vllm/bin/vllm"
[[ -x "$VLLM" ]] || { log "missing $VLLM - run scripts/setup_gpu_box.sh vllm"; exit 1; }

ARGS=(serve "$MODEL" --deploy-config "$DEPLOY_CONFIG" --omni --port "$PORT" --trust-remote-code)
[[ -n "${STAGE_OVERRIDES:-}" ]] && ARGS+=(--stage-overrides "$STAGE_OVERRIDES")
[[ -n "${NO_ASYNC_CHUNK:-}" ]] && ARGS+=(--no-async-chunk)
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

log "vllm-omni @ $(git -C "$SRC/vllm-omni" rev-parse --short HEAD 2>/dev/null || echo '?'): $VLLM ${ARGS[*]}"
# shellcheck disable=SC2086
exec ${PREFIX:-} "$VLLM" "${ARGS[@]}"
