# Shared helpers for the GPU-box scripts. Source, don't execute.
# shellcheck shell=bash

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/p1}"
ENV_FILE="$WORK/p1.env"
if [[ -f "$ENV_FILE" ]]; then
  # written by setup_gpu_box.sh: VENVS, SRC, MODEL_DIR, pins
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
VENVS="${VENVS:-$WORK/venvs}"
SRC="${SRC:-$WORK/src}"
MODEL_DIR="${MODEL_DIR:-$WORK/models/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
CONFIG="${CONFIG:-$REPO/configs/p1_run.toml}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# wait_http URL [timeout_s] [pid] - poll until GET URL returns 200; fail fast if pid exits
wait_http() {
  local url="$1" timeout="${2:-1200}" pid="${3:-}" t=0
  log "waiting for $url (timeout ${timeout}s)"
  until curl -sf -o /dev/null "$url"; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then log "server process $pid exited before $url came up"; return 1; fi
    sleep 5; t=$((t + 5))
    if (( t >= timeout )); then log "timeout waiting for $url"; return 1; fi
  done
  log "$url is up"
}

# smoke_tts BASE_URL VOICE OUT.wav - one request so failures show up before a long run
smoke_tts() {
  curl -sf -X POST "$1/v1/audio/speech" -H "Content-Type: application/json" \
    -d "{\"input\": \"Hello, this is a smoke test.\", \"voice\": \"$2\", \"language\": \"English\"}" \
    --output "$3" && log "smoke test wrote $3 ($(stat -c %s "$3" 2>/dev/null || stat -f %z "$3") bytes)"
}

stop_pid() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
