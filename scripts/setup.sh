#!/usr/bin/env bash
# Reproducible setup: pinned PRIME-RL + prebuilt wheels (torch 2.11+cu128,
# vllm 0.24.0+cu129, flash-attn 2.8.3 cu128/torch2.11). Nothing compiles.
# Needs Linux x86_64, NVIDIA driver >= 570 (CUDA 12.9). Tested target: 2x RTX 3090.
set -euo pipefail
cd "$(dirname "$0")/.."

readonly UV_VERSION="0.11.29"
readonly PRIME_RL_COMMIT="d334ea52940b47f426293a7d146239e3fbf91caa"
readonly VERIFIERS_COMMIT="6c64ce6a3a01e8edde7c3c0e8e5315fb236e9faa"
readonly PRIME_DIR=".vendor/prime-rl"

retry() { for i in 1 2 3 4 5; do "$@" && return; echo "retry $i: $*" >&2; sleep 10; done; return 1; }

if [[ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$PRIME_DIR/.git" ]]; then
  git init -q "$PRIME_DIR"
  git -C "$PRIME_DIR" remote add origin https://github.com/PrimeIntellect-ai/prime-rl.git
fi
git -C "$PRIME_DIR" cat-file -e "${PRIME_RL_COMMIT}^{commit}" 2>/dev/null \
  || retry git -C "$PRIME_DIR" fetch -q --depth=1 origin "$PRIME_RL_COMMIT"
git -C "$PRIME_DIR" checkout -q --detach "$PRIME_RL_COMMIT"
for sub in renderers research-environments verifiers; do
  git -C "$PRIME_DIR" config "submodule.$sub.url" "https://github.com/PrimeIntellect-ai/$sub.git"
done
retry git -C "$PRIME_DIR" submodule update -q --init --depth=1 \
  deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
test "$(git -C "$PRIME_DIR/deps/verifiers" rev-parse HEAD)" = "$VERIFIERS_COMMIT"

patch="$PWD/patches/prime-rl.patch"
git -C "$PRIME_DIR" apply --reverse --check "$patch" 2>/dev/null \
  || git -C "$PRIME_DIR" apply "$patch"

uv sync --locked
uv sync --locked --project "$PRIME_DIR" --extra flash-attn --no-build-package flash-attn
uv run --project "$PRIME_DIR" --extra flash-attn python - <<'PY'
import flash_attn, torch, vllm
assert torch.cuda.is_available()
print("torch", torch.__version__, "vllm", vllm.__version__, "flash-attn", flash_attn.__version__)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
echo "ready"
