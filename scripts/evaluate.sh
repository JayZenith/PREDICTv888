#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
model="${2:-}"
split="${3:-}"
if [[ ( "$arm" != "a" && "$arm" != "b" ) || -z "$model" ]] \
  || [[ "$split" != "validation" && "$split" != "test" ]]; then
  echo "usage: bash scripts/evaluate.sh a|b MODEL validation|test [extra eval arguments]" >&2
  exit 2
fi
shift 3

count=40
[[ "$split" == "test" ]] && count=500

exec uv run eval glyph \
  --harness.id glyph \
  --harness.arm "$arm" \
  --taskset.data-path "data/arm_${arm}_${split}.jsonl" \
  --sampling.temperature 0 \
  --sampling.max-tokens 512 \
  --max-total-tokens 4096 \
  -m "$model" -n "$count" -r 1 --no-push "$@"
