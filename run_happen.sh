#!/usr/bin/env bash
set -euo pipefail

IMAGE="${HAPPEN_IMAGE:-happen:gpu}"

normalize_abs_path() {
  local p="$1"
  p="${p/#\~/$HOME}"

  if [[ -d "$p" ]]; then
    (cd "$p" && pwd -P)
    return
  fi

  if [[ -e "$p" ]]; then
    local dir base
    dir="$(dirname "$p")"
    base="$(basename "$p")"
    (cd "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
    return
  fi

  local dir base
  dir="$(dirname "$p")"
  base="$(basename "$p")"
  (cd "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
}

parse_config() {
  local config_path="$1"
  local config_dir="$2"

  docker run --rm -i \
    --entrypoint python \
    -e HOST_CONFIG_DIR="$config_dir" \
    -e HOST_CONFIG_ABS="$config_path" \
    -v "$config_path:/tmp/happen_config.toml:ro" \
    "$IMAGE" - /tmp/happen_config.toml <<'PY'
import os
import sys
import tomllib

cfg_path = sys.argv[1]
host_cfg_dir = os.environ["HOST_CONFIG_DIR"]
host_cfg_abs = os.environ["HOST_CONFIG_ABS"]

with open(cfg_path, "rb") as f:
    cfg = tomllib.load(f)

def get_path(*keys):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return ""
        cur = cur[k]
    if cur is None:
        return ""
    s = str(cur).strip()
    if not s:
        return ""
    s = os.path.expanduser(s)
    if os.path.isabs(s):
        return os.path.abspath(s)
    return os.path.abspath(os.path.join(host_cfg_dir, s))

def get_str(*keys, default=""):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if cur is None:
        return default
    return str(cur).strip()

print(f"CONFIG_ABS={host_cfg_abs}")
print(f"CONFIG_DIR={host_cfg_dir}")
print(f"RUN_OUT={get_path('run', 'out')}")
print(f"EXACT_ROOT={get_path('exact', 'root')}")
print(f"EXACT_CANDIDATES_CSV={get_path('exact', 'candidates_csv')}")
print(f"EXACT_VALID_CSV={get_path('exact', 'valid_csv')}")
print(f"REVIEW_PORT={get_str('review', 'port', default='5291') or '5291'}")
print(f"REVIEW_HOST={get_str('review', 'host', default='')}")
PY
}

is_same_or_within() {
  local child="$1"
  local parent="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

declare -a EXTRA_BINDS
declare -a DOCKER_ARGS
declare -a MOUNTS

MODE="$1"
CONFIG_INPUT="$2"
shift 2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bind)
      shift
      EXTRA_BINDS+=("$(normalize_abs_path "$1")")
      ;;
    --image)
      shift
      IMAGE="$1"
      ;;
  esac
  shift
done

CONFIG_INPUT="$(normalize_abs_path "$CONFIG_INPUT")"
CONFIG_DIR="$(dirname "$CONFIG_INPUT")"

while IFS='=' read -r key value; do
  case "$key" in
    CONFIG_ABS) CONFIG_ABS="$value" ;;
    CONFIG_DIR) CONFIG_DIR="$value" ;;
    RUN_OUT) RUN_OUT="$value" ;;
    EXACT_ROOT) EXACT_ROOT="$value" ;;
    EXACT_CANDIDATES_CSV) EXACT_CANDIDATES_CSV="$value" ;;
    EXACT_VALID_CSV) EXACT_VALID_CSV="$value" ;;
    REVIEW_PORT) REVIEW_PORT="$value" ;;
    REVIEW_HOST) REVIEW_HOST="$value" ;;
  esac
done < <(parse_config "$CONFIG_INPUT" "$CONFIG_DIR")

if [[ "$MODE" == "pipeline" ]]; then
  mkdir -p "$RUN_OUT"
fi

add_mount() {
  local src="$1"
  local mode="$2"
  [[ -n "$src" ]] || return 0

  local item="${src}:${src}:${mode}"
  for x in "${MOUNTS[@]:-}"; do
    [[ "$x" == "$item" ]] && return 0
  done

  MOUNTS+=("$item")
  DOCKER_ARGS+=(-v "$item")
}

DOCKER_ARGS=(docker run --rm --gpus all)

if ! is_same_or_within "$CONFIG_DIR" "$RUN_OUT"; then
  add_mount "$CONFIG_DIR" ro
fi

add_mount "$RUN_OUT" rw

if [[ "$MODE" == "pipeline" ]]; then
  [[ -z "${EXACT_ROOT:-}" ]] || add_mount "$EXACT_ROOT" ro
  [[ -z "${EXACT_CANDIDATES_CSV:-}" ]] || add_mount "$EXACT_CANDIDATES_CSV" ro
  [[ -z "${EXACT_VALID_CSV:-}" ]] || add_mount "$EXACT_VALID_CSV" ro
fi

for b in "${EXTRA_BINDS[@]:-}"; do
  add_mount "$b" ro
done

DOCKER_ARGS+=(-w "$CONFIG_DIR")

if [[ "$MODE" == "review" ]]; then
  DOCKER_ARGS+=(-p "${REVIEW_PORT}:${REVIEW_PORT}")
fi

echo "[run_happen] image -> $IMAGE"
echo "[run_happen] mode -> $MODE"
echo "[run_happen] config -> $CONFIG_INPUT"
echo "[run_happen] out -> $RUN_OUT"

if [[ "$MODE" == "pipeline" ]]; then
  [[ -z "${EXACT_ROOT:-}" ]] || echo "[run_happen] exact.root  -> $EXACT_ROOT"
  [[ -z "${EXACT_CANDIDATES_CSV:-}" ]] || echo "[run_happen] candidates  -> $EXACT_CANDIDATES_CSV"
  [[ -z "${EXACT_VALID_CSV:-}" ]] || echo "[run_happen] valid_csv   -> $EXACT_VALID_CSV"
else
  echo "[run_happen] review port -> $REVIEW_PORT"
fi

for b in "${EXTRA_BINDS[@]:-}"; do
  echo "[run_happen] extra bind  -> $b"
done

DOCKER_ARGS+=("$IMAGE" "$MODE" "$CONFIG_INPUT")
exec "${DOCKER_ARGS[@]}"