#!/usr/bin/env bash
# Nsight Systems capture of a warmed engine under a short, fixed load.
#
#   bash scripts/profile_nsys.sh vllm_omni   [N_PROMPTS] [CONCURRENCY]
#   bash scripts/profile_nsys.sh sglang_omni [N_PROMPTS] [CONCURRENCY]
#   bash scripts/profile_nsys.sh official    [N_PROMPTS]
#
# Server engines: the server is started under `nsys launch` (collection idle),
# warmed with a few requests, then `nsys start` / client load / `nsys stop`
# brackets ONLY the measured window, so startup, weight loading and CUDA-graph
# capture are excluded. --trace-fork-before-exec follows the per-stage worker
# processes; --cuda-graph-trace=node shows kernels inside CUDA graphs (both
# engines replay graphs for the talker / decoder).
#
# Output: runs/nsys-<engine>-<ts>/{report.nsys-rep, kern_sum.csv, kernels_grouped.md, ...}
# Verified with Nsight Systems 2025.3.2 on vLLM-Omni and SGLang-Omni (2026-10-13).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ENGINE="${1:?engine: vllm_omni | sglang_omni | official}"
N="${2:-8}"
C="${3:-1}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="$REPO/runs/nsys-$ENGINE-$TS"
mkdir -p "$OUT"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
command -v nsys >/dev/null || { log "nsys not found"; exit 1; }

summarize() {
  nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output - "$OUT/report.nsys-rep" > "$OUT/kern_sum.csv"
  "$VENVS/client/bin/tts-bench" nsys-summary "$OUT/kern_sum.csv" --title "$ENGINE - nsys, $N prompts, concurrency $C" --out "$OUT"
  nsys --version > "$OUT/nsys_version.txt"
  log "kernel table: $OUT/kernels_grouped.md"
}

if [[ "$ENGINE" == "official" ]]; then
  # In-process: profile the bench command itself (full mode, batch 1). The
  # torch.profiler path (`tts-bench profile`) gives per-op attribution; this
  # gives the raw kernel timeline for cross-engine comparison.
  nsys profile --trace="$NSYS_TRACE" --cuda-graph-trace=node --force-overwrite=true -o "$OUT/report" \
    "$VENVS/official/bin/tts-bench" bench --config "$CONFIG" --engine official --model-path "$MODEL_DIR" \
      --modes full --concurrency 1 --limit "$N" --out "$OUT/bench" --tag nsys
  summarize
  exit 0
fi

case "$ENGINE" in
  vllm_omni)   LAUNCH="$REPO/scripts/launch_vllm_omni.sh";   PORT="${PORT:-8091}"; VOICE=ryan ;;
  sglang_omni) LAUNCH="$REPO/scripts/launch_sglang_omni.sh"; PORT="${PORT:-8000}"; VOICE=Ryan ;;
  *) log "unknown engine $ENGINE"; exit 1 ;;
esac
SESSION="p1-$ENGINE-$TS"
URL="http://127.0.0.1:$PORT"

PREFIX="nsys launch --session-new=$SESSION --trace=$NSYS_TRACE --cuda-graph-trace=node --trace-fork-before-exec=true" \
  PORT="$PORT" bash "$LAUNCH" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'stop_pid $SERVER_PID' EXIT
wait_http "$URL/health" 1800 "$SERVER_PID"
smoke_tts "$URL" "$VOICE" "$OUT/smoke.wav"

# warm (not captured)
"$VENVS/client/bin/tts-bench" bench --config "$CONFIG" --engine "$ENGINE" --base-url "$URL" \
  --modes full stream --concurrency "$C" --limit 4 --out "$OUT/warm" --tag warm > /dev/null

nsys start --session="$SESSION" --output="$OUT/report" --force-overwrite=true
"$VENVS/client/bin/tts-bench" bench --config "$CONFIG" --engine "$ENGINE" --base-url "$URL" \
  --modes full stream --concurrency "$C" --limit "$N" --out "$OUT/bench" --tag nsys
nsys stop --session="$SESSION"

stop_pid "$SERVER_PID"; trap - EXIT
summarize
