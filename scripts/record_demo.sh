#!/usr/bin/env bash
# Record the streaming demo on each server engine: one sentence per setting,
# WAV + chunk-arrival log under runs/demo/. vLLM-Omni is recorded with its
# default first chunk and with initial_codec_chunk_frames=4 (the setting that
# removed playback stalls in the streaming sweep).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="$REPO/runs/demo"; mkdir -p "$OUT"
PY="$VENVS/client/bin/python"
TEXT="${TEXT:-This sentence was streamed from a Qwen three text to speech model, and playback started before the rest of it had been generated.}"

demo() {  # demo ENGINE LAUNCH PORT VOICE TAG [INITIAL_CHUNK]
  local engine="$1" launch="$2" port="$3" voice="$4" tag="$5" chunk="${6:-}"
  "$PY" "$REPO/demo/stream_demo.py" --engine "$engine" --base-url "http://127.0.0.1:$port" \
    --text "$TEXT" --out "$OUT/$tag.wav" ${chunk:+--initial-chunk "$chunk"} | tee "$OUT/$tag.log"
}

run_engine() {  # run_engine ENGINE LAUNCH PORT VOICE
  local engine="$1" launch="$2" port="$3" voice="$4"
  PORT=$port bash "$REPO/scripts/$launch" > "$OUT/server-$engine.log" 2>&1 &
  local pid=$!
  wait_http "http://127.0.0.1:$port/health" 1500 "$pid" || { tail -40 "$OUT/server-$engine.log"; return 1; }
  "$PY" "$REPO/demo/stream_demo.py" --engine "$engine" --base-url "http://127.0.0.1:$port" --text "Warm up." --out "$OUT/warm-$engine.wav" > /dev/null
  shift 4
  for spec in "$@"; do demo "$engine" "$launch" "$port" "$voice" "$engine-$spec" "$( [[ $spec == k* ]] && echo "${spec#k}" )"; done
  stop_pid "$pid"
}

run_engine vllm_omni launch_vllm_omni.sh 8091 ryan default k4
run_engine sglang_omni launch_sglang_omni.sh 8000 Ryan default
ls -la "$OUT"
