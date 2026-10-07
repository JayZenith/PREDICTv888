#!/usr/bin/env bash
# usage:
#   bash scripts/run.sh sft  ARM [PRIME-RL args]   1 GPU
#   bash scripts/run.sh rl   ARM [PRIME-RL args]   2 GPUs (train + inference)
#   bash scripts/run.sh eval ARM MODEL             1 GPU; greedy pass@1 on the 500 test tasks
# ARM is a, b, or b_restate (the restate control, which uses Arm B's harness)
set -euo pipefail
cd "$(dirname "$0")/.."

cmd="${1:-}"; arm="${2:-}"
[[ "$arm" == a || "$arm" == b || "$arm" == b_restate ]] || { sed -n 2,6p "$0" >&2; exit 2; }
shift 2

export HF_HOME="${HF_HOME:-$PWD/.cache/hf}" WANDB_MODE="${WANDB_MODE:-offline}" PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
prime=(uv run --project .vendor/prime-rl --extra flash-attn)

case "$cmd" in
  sft) exec "${prime[@]}" sft @ "$PWD/configs/arm_${arm}_sft.toml" "$@" ;;
  rl)  exec "${prime[@]}" rl @ "$PWD/configs/arm_${arm}_rl.toml" "$@" ;;
  eval)
    model="${1:?MODEL}"; shift
    mkdir -p outputs; port="${PORT:-8020}"
    "${prime[@]}" vllm serve "$model" --served-model-name policy --port "$port" \
      --gpu-memory-utilization 0.85 > "outputs/vllm_${port}.log" 2>&1 &
    pid=$!; trap 'kill $pid 2>/dev/null' EXIT
    until curl -sf "localhost:$port/v1/models" >/dev/null; do
      kill -0 $pid 2>/dev/null || { tail -20 "outputs/vllm_${port}.log" >&2; exit 1; }
      sleep 5
    done
    uv run eval glyph \
      --harness.id glyph --harness.arm "${arm:0:1}" \
      --taskset.data-path "data/arm_${arm}_test.jsonl" \
      --sampling.temperature 0 --sampling.max-tokens 512 --max-total-tokens 4096 \
      --client.base-url "http://localhost:$port/v1" --client.api-key-var HOME \
      -m policy -n 500 -r 1 --no-push "$@"
    ;;
  *) sed -n 2,6p "$0" >&2; exit 2 ;;
esac
