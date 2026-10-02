#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
if [[ "$arm" != "a" && "$arm" != "b" ]]; then
  echo "usage: bash scripts/train_sft.sh a|b [extra PRIME-RL arguments]" >&2
  exit 2
fi
shift

test -f "data/sft/arm_${arm}/train.jsonl" || {
  echo "dataset missing" >&2
  exit 1
}

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --project .vendor/prime-rl \
  --extra flash-attn \
  sft @ "$PWD/configs/arm_${arm}_sft.toml" "$@"
