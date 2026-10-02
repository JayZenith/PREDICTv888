#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
if [[ "$arm" != "a" && "$arm" != "b" ]]; then
  echo "usage: bash scripts/train_rl.sh a|b [extra PRIME-RL arguments]" >&2
  exit 2
fi
shift

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --project .vendor/prime-rl \
  --extra flash-attn \
  rl @ "$PWD/configs/arm_${arm}_rl.toml" "$@"
