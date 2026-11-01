#!/usr/bin/env bash
# P1 runbook, one function per "work session" of p1-p2-research-plan.md:
#   1  official baseline: load checkpoint, generate the fixed prompt set (c=1,4,8)
#   2  vLLM-Omni on the same GPU (full + stream, c=1,4,8)
#   3  SGLang-Omni (full + stream, c=1,4,8)
#   4  profile: torch.profiler (official) + nsys (all three)
#   5  streaming sweeps, then session 6
#   6  quality (WER + predicted MOS) for every run in runs/INDEX.txt, report
#
#   bash scripts/run_all.sh          # all sessions in order
#   bash scripts/run_all.sh 2 3      # selected sessions
#   LIMIT=10 bash scripts/run_all.sh 1   # quick smoke on 10 prompts
#
# Only one engine holds the GPU at a time. Run directories are listed in
# runs/INDEX.txt so session 5 knows what to evaluate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SESSIONS=("$@")
[[ ${#SESSIONS[@]} -eq 0 ]] && SESSIONS=(1 2 3 4 5)
INDEX="$REPO/runs/INDEX.txt"
LIMIT_ARGS=(); [[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$LIMIT")
CLIENT="$VENVS/client/bin/tts-bench"
OFFICIAL="$VENVS/official/bin/tts-bench"
EVAL="$VENVS/eval/bin/tts-bench"

# run dirs are runs/<YYYYmmdd-HHMMSS>-<engine>[-<tag>]; the leading digit excludes profile-*/nsys-* dirs
newest_run() { ls -dt "$REPO"/runs/[0-9]*-"$1"${2:+-$2} 2>/dev/null | head -1; }
record() { echo "$1 $2" >> "$INDEX"; log "recorded $1 -> $2"; }

with_server() {  # with_server ENGINE CMD...
  local engine="$1"; shift
  local launch port voice
  case "$engine" in
    vllm_omni)   launch="$REPO/scripts/launch_vllm_omni.sh";   port=8091; voice=ryan ;;
    sglang_omni) launch="$REPO/scripts/launch_sglang_omni.sh"; port=8000; voice=Ryan ;;
  esac
  local logf="$REPO/runs/server-$engine-$(date +%Y%m%d-%H%M%S).log"
  PORT=$port bash "$launch" > "$logf" 2>&1 &
  local pid=$!
  trap "stop_pid $pid" EXIT
  wait_http "http://127.0.0.1:$port/health" 1800 "$pid" || { tail -50 "$logf"; exit 1; }
  smoke_tts "http://127.0.0.1:$port" "$voice" "$REPO/runs/smoke-$engine.wav"
  "$@"
  stop_pid "$pid"; trap - EXIT
  log "server log: $logf"
}

session1() {
  log "session 1: official Qwen3-TTS baseline"
  "$OFFICIAL" validate-prompts
  "$OFFICIAL" bench --config "$CONFIG" --engine official --model-path "$MODEL_DIR" "${LIMIT_ARGS[@]}"
  record official "$(newest_run official)"
}

session2() {
  log "session 2: vLLM-Omni"
  with_server vllm_omni "$CLIENT" bench --config "$CONFIG" --engine vllm_omni "${LIMIT_ARGS[@]}" \
    --engine-python "$VENVS/vllm/bin/python"
  record vllm_omni "$(newest_run vllm_omni)"
}

session3() {
  log "session 3: SGLang-Omni"
  with_server sglang_omni "$CLIENT" bench --config "$CONFIG" --engine sglang_omni "${LIMIT_ARGS[@]}" \
    --engine-python "$VENVS/sglang/bin/python"
  record sglang_omni "$(newest_run sglang_omni)"
}

session4() {
  log "session 4: profiling"
  # 2 prompts: torch.profiler over HF generate grows ~1 GB of trace per prompt.
  "$OFFICIAL" profile --config "$CONFIG" --model-path "$MODEL_DIR" --n-prompts 2 \
    --out "$REPO/runs/profile-official-$(date +%Y%m%d-%H%M%S)"
  bash "$REPO/scripts/profile_nsys.sh" official 2
  bash "$REPO/scripts/profile_nsys.sh" vllm_omni 8 1
  bash "$REPO/scripts/profile_nsys.sh" sglang_omni 8 1
  log "now fill docs/KERNEL_NOTES.md from runs/profile-official-*/ and runs/nsys-*/kernels_grouped.md"
}

session5() {
  log "session 5: streaming sweeps, quality, report"
  with_server vllm_omni "$CLIENT" streaming --config "$CONFIG" --engine vllm_omni --concurrency 1 4 "${LIMIT_ARGS[@]}"
  record vllm_omni_streaming "$(newest_run vllm_omni streaming)"
  with_server sglang_omni "$CLIENT" streaming --config "$CONFIG" --engine sglang_omni --concurrency 1 4 "${LIMIT_ARGS[@]}"
  record sglang_omni_streaming "$(newest_run sglang_omni streaming)"

  session6
}

session6() {
  log "session 6: quality (WER + predicted MOS) for every recorded run, report"
  # Quality for every recorded run (GPU is free now: no server running).
  local dirs=()
  while read -r _name dir; do [[ -d "$dir" ]] && dirs+=("$dir"); done < <(sort -u "$INDEX")
  for d in "${dirs[@]}"; do
    "$EVAL" quality "$d" --config "$CONFIG"
    if [[ "$d" == *streaming* ]]; then "$CLIENT" streaming --config "$CONFIG" --from-run "$d"; fi
  done
  "$CLIENT" report "${dirs[@]}" --out "$REPO/reports/p1_report.md" --baseline official
  log "report: $REPO/reports/p1_report.md"
}

for s in "${SESSIONS[@]}"; do "session$s"; done
