#!/usr/bin/env bash
# Author: Jiheng Li
# Email: jiheng.li.1@vanderbilt.edu

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  pipeline <config.toml>"
  echo "  review   <config.toml>"
  echo "  bash"
  exit 2
fi

mode="$1"
shift

case "${mode}" in
  pipeline)
    if [[ $# -lt 1 ]]; then
      echo "Usage: pipeline <config.toml>"
      exit 2
    fi
    exec python -m happen.pipeline "$@"
    ;;
  review)
    if [[ $# -lt 1 ]]; then
      echo "Usage: review <config.toml>"
      exit 2
    fi
    exec python -m happen.review "$@"
    ;;
  bash)
    exec /bin/bash "$@"
    ;;
  *)
    echo "Unknown command: ${mode}"
    echo "Expected one of: pipeline, review, bash"
    exit 2
    ;;
esac