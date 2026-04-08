#!/usr/bin/env bash
# Author: Jiheng Li
# Email: jiheng.li.1@vanderbilt.edu

set -euo pipefail

IMAGE="${HAPPEN_IMAGE:-happen:latest}"

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
PY
}

is_same_or_within() {
  local child="$1"
  local parent="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

usage() {
  cat <<EOF
Usage:
  $0 pipeline <config.toml> [--bind <path>]...
  $0 review <config.toml> [finalize] [--bind <path>]...
EOF
}

declare -a EXTRA_BINDS
declare -a DOCKER_ARGS
declare -a MOUNT_PATHS

declare -A MOUNT_MODE
declare -A MOUNT_REASON

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

MODE="$1"
CONFIG_INPUT="$2"
shift 2

REVIEW_SUBCOMMAND=""

if [[ "$MODE" != "pipeline" && "$MODE" != "review" ]]; then
  echo "[run_happen][ERROR] mode must be 'pipeline' or 'review'"
  usage
  exit 2
fi

if [[ "$MODE" == "review" && $# -gt 0 ]]; then
  case "$1" in
    finalize)
      REVIEW_SUBCOMMAND="$1"
      shift
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bind)
      shift
      [[ $# -gt 0 ]] || { echo "[run_happen][ERROR] --bind requires a path"; exit 2; }
      EXTRA_BINDS+=("$(normalize_abs_path "$1")")
      ;;
    *)
      echo "[run_happen][ERROR] unknown argument: $1"
      usage
      exit 2
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
  esac
done < <(parse_config "$CONFIG_INPUT" "$CONFIG_DIR")

if [[ -z "${RUN_OUT:-}" ]]; then
  echo "[run_happen][ERROR] [run].out could not be resolved from config"
  exit 2
fi

if [[ "$MODE" == "pipeline" ]]; then
  mkdir -p "$RUN_OUT"
fi

request_mount() {
  local src="$1"
  local mode="$2"
  local reason="$3"

  [[ -n "$src" ]] || return 0
  src="$(normalize_abs_path "$src")"

  if is_same_or_within "$src" "$RUN_OUT"; then
    mode="rw"
  fi

  if [[ -v MOUNT_MODE["$src"] ]]; then
    if [[ "${MOUNT_MODE[$src]}" == "ro" && "$mode" == "rw" ]]; then
      MOUNT_MODE["$src"]="rw"
      MOUNT_REASON["$src"]="${MOUNT_REASON[$src]}; upgraded to rw by ${reason}"
    else
      MOUNT_REASON["$src"]="${MOUNT_REASON[$src]}; ${reason}"
    fi
    return 0
  fi

  MOUNT_PATHS+=("$src")
  MOUNT_MODE["$src"]="$mode"
  MOUNT_REASON["$src"]="$reason"
}

build_mounts() {
  if ! is_same_or_within "$CONFIG_DIR" "$RUN_OUT"; then
    request_mount "$CONFIG_DIR" "ro" "config_dir"
  fi

  if [[ "$MODE" == "pipeline" ]]; then
    [[ -z "${EXACT_ROOT:-}" ]] || request_mount "$EXACT_ROOT" "ro" "exact.root"
    [[ -z "${EXACT_CANDIDATES_CSV:-}" ]] || request_mount "$EXACT_CANDIDATES_CSV" "ro" "exact.candidates_csv"
    [[ -z "${EXACT_VALID_CSV:-}" ]] || request_mount "$EXACT_VALID_CSV" "ro" "exact.valid_csv"
  fi

  for b in "${EXTRA_BINDS[@]:-}"; do
    request_mount "$b" "ro" "extra_bind"
  done

  request_mount "$RUN_OUT" "rw" "run.out"
}

emit_mount_args() {
  local p
  for p in "${MOUNT_PATHS[@]}"; do
    DOCKER_ARGS+=(-v "${p}:${p}:${MOUNT_MODE[$p]}")
  done
}

if [[ "$MODE" == "pipeline" ]]; then
  DOCKER_ARGS=(docker run --rm --gpus all)
else
  DOCKER_ARGS=(docker run --rm)
fi

build_mounts
emit_mount_args

DOCKER_ARGS+=(-w "$CONFIG_DIR")

if [[ "$MODE" == "review" && "$REVIEW_SUBCOMMAND" != "finalize" ]]; then
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
  if [[ "$REVIEW_SUBCOMMAND" == "finalize" ]]; then
    echo "[run_happen] review cmd  -> finalize"
  else
    echo "[run_happen] review port -> $REVIEW_PORT"
  fi
fi

for p in "${MOUNT_PATHS[@]}"; do
  echo "[run_happen] mount -> ${p} (${MOUNT_MODE[$p]})"
done

DOCKER_ARGS+=("$IMAGE" "$MODE" "$CONFIG_INPUT")
if [[ -n "$REVIEW_SUBCOMMAND" ]]; then
  DOCKER_ARGS+=("$REVIEW_SUBCOMMAND")
fi

exec "${DOCKER_ARGS[@]}"