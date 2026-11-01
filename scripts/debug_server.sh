#!/usr/bin/env bash
# Start one server engine, wait for health (fail fast on exit), send one smoke
# request, print the log tail, stop. Usage: bash scripts/debug_server.sh vllm_omni|sglang_omni
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
case "$1" in
  vllm_omni)   launch=launch_vllm_omni.sh;   port=8091; voice=ryan ;;
  sglang_omni) launch=launch_sglang_omni.sh; port=8000; voice=Ryan ;;
esac
logf="$REPO/runs/debug-$1-$(date +%Y%m%d-%H%M%S).log"
PORT=$port bash "$REPO/scripts/$launch" > "$logf" 2>&1 &
pid=$!
t0=$(date +%s)
if wait_http "http://127.0.0.1:$port/health" 1500 "$pid"; then
  log "healthy after $(( $(date +%s) - t0 ))s"
  smoke_tts "http://127.0.0.1:$port" "$voice" "$REPO/runs/debug-$1.wav" || log "smoke request FAILED"
  rc=0
else
  rc=1
fi
sed -E 's/\x1b\[[0-9;]*m//g' "$logf" | grep -E "Error|error|Traceback|WARNING" | tail -30
stop_pid "$pid"
exit $rc
