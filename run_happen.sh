#!/usr/bin/env bash
# Author: Jiheng Li
# Email: jiheng.li.1@vanderbilt.edu

set -euo pipefail

IMAGE="${HAPPEN_IMAGE:-happen:latest}"
RUNTIME="${HAPPEN_RUNTIME:-auto}"

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

detect_runtime() {
  local image="$1"
  local requested="$2"

  case "$requested" in
    docker|apptainer)
      printf '%s\n' "$requested"
      return
      ;;
    auto)
      ;;
    *)
      echo "[run_happen][ERROR] HAPPEN_RUNTIME must be one of: auto, docker, apptainer"
      exit 2
      ;;
  esac

  if [[ "$image" == *.sif ]]; then
    printf 'apptainer\n'
    return
  fi

  if [[ "$image" == docker://* || "$image" == oras://* || "$image" == library://* || "$image" == shub://* ]]; then
    printf 'apptainer\n'
    return
  fi

  if [[ -d "$image" ]]; then
    printf 'apptainer\n'
    return
  fi

  printf 'docker\n'
}

resolve_apptainer_cmd() {
  if command -v apptainer >/dev/null 2>&1; then
    printf 'apptainer\n'
    return
  fi

  if command -v singularity >/dev/null 2>&1; then
    printf 'singularity\n'
    return
  fi

  echo "[run_happen][ERROR] runtime resolved to apptainer, but neither 'apptainer' nor 'singularity' is available in PATH"
  exit 127
}

RUNTIME="$(detect_runtime "$IMAGE" "$RUNTIME")"
APPTAINER_CMD=""

if [[ "$RUNTIME" == "apptainer" ]]; then
  APPTAINER_CMD="$(resolve_apptainer_cmd)"
fi

is_same_or_within() {
  local child="$1"
  local parent="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

parse_config() {
  local config_path="$1"
  local config_dir="$2"

  if [[ "$RUNTIME" == "docker" ]]; then
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
  else
    APPTAINERENV_HOST_CONFIG_DIR="$config_dir" \
    APPTAINERENV_HOST_CONFIG_ABS="$config_path" \
    SINGULARITYENV_HOST_CONFIG_DIR="$config_dir" \
    SINGULARITYENV_HOST_CONFIG_ABS="$config_path" \
    "$APPTAINER_CMD" exec --cleanenv \
      --bind "$config_path:/tmp/happen_config.toml:ro" \
      "$IMAGE" python - /tmp/happen_config.toml <<'PY'
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
  fi
}

declare -a EXTRA_BINDS
declare -a ENGINE_ARGS
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
  if [[ "$RUNTIME" == "docker" ]]; then
    for p in "${MOUNT_PATHS[@]}"; do
      ENGINE_ARGS+=(-v "${p}:${p}:${MOUNT_MODE[$p]}")
    done
  else
    for p in "${MOUNT_PATHS[@]}"; do
      ENGINE_ARGS+=(--bind "${p}:${p}:${MOUNT_MODE[$p]}")
    done
  fi
}

build_engine_args() {
  if [[ "$RUNTIME" == "docker" ]]; then
    if [[ "$MODE" == "pipeline" ]]; then
      ENGINE_ARGS=(docker run --rm --gpus all)
    else
      ENGINE_ARGS=(docker run --rm)
    fi
  else
    if [[ "$MODE" == "pipeline" ]]; then
      ENGINE_ARGS=("$APPTAINER_CMD" run --cleanenv --nv)
    else
      ENGINE_ARGS=("$APPTAINER_CMD" run --cleanenv)
    fi
  fi
}

append_cwd_arg() {
  if [[ "$RUNTIME" == "docker" ]]; then
    ENGINE_ARGS+=(-w "$CONFIG_DIR")
  else
    ENGINE_ARGS+=(--cwd "$CONFIG_DIR")
  fi
}

append_review_network_arg() {
  if [[ "$MODE" != "review" || "$REVIEW_SUBCOMMAND" == "finalize" ]]; then
    return 0
  fi

  if [[ "$RUNTIME" == "docker" ]]; then
    ENGINE_ARGS+=(-p "${REVIEW_PORT}:${REVIEW_PORT}")
  else
    :
  fi
}

build_engine_args
build_mounts
emit_mount_args
append_cwd_arg
append_review_network_arg

echo "[run_happen] runtime -> $RUNTIME"
if [[ "$RUNTIME" == "apptainer" ]]; then
  echo "[run_happen] apptainer_cmd -> $APPTAINER_CMD"
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
    if [[ "$RUNTIME" == "apptainer" ]]; then
      echo "[run_happen] note -> apptainer uses host networking by default; no Docker-style -p mapping added"
    fi
  fi
fi

for p in "${MOUNT_PATHS[@]}"; do
  echo "[run_happen] mount -> ${p} (${MOUNT_MODE[$p]})"
done

ENGINE_ARGS+=("$IMAGE" "$MODE" "$CONFIG_INPUT")
if [[ -n "$REVIEW_SUBCOMMAND" ]]; then
  ENGINE_ARGS+=("$REVIEW_SUBCOMMAND")
fi

if [[ "$RUNTIME" == "apptainer" ]]; then
  exec env \
    APPTAINERENV_TINI_SUBREAPER=1 \
    SINGULARITYENV_TINI_SUBREAPER=1 \
    "${ENGINE_ARGS[@]}"
else
  exec "${ENGINE_ARGS[@]}"
fi