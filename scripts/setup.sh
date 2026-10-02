#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

readonly PRIME_RL_COMMIT="d334ea52940b47f426293a7d146239e3fbf91caa"
readonly VERIFIERS_COMMIT="6c64ce6a3a01e8edde7c3c0e8e5315fb236e9faa"
readonly PRIME_DIR=".vendor/prime-rl"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

if [[ ! -d "$PRIME_DIR/.git" ]]; then
  mkdir -p "$PRIME_DIR"
  git -C "$PRIME_DIR" init
  git -C "$PRIME_DIR" remote add origin https://github.com/PrimeIntellect-ai/prime-rl.git
fi
if ! git -C "$PRIME_DIR" cat-file -e "${PRIME_RL_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$PRIME_DIR" fetch --depth=1 origin "$PRIME_RL_COMMIT"
fi
git -C "$PRIME_DIR" checkout --detach "$PRIME_RL_COMMIT"
# Upstream pins public submodules with SSH URLs. Fresh training instances do not
# need GitHub SSH credentials, so override those URLs locally with HTTPS.
git -C "$PRIME_DIR" config submodule.renderers.url \
  https://github.com/PrimeIntellect-ai/renderers.git
git -C "$PRIME_DIR" config submodule.research-environments.url \
  https://github.com/PrimeIntellect-ai/research-environments.git
git -C "$PRIME_DIR" config submodule.verifiers.url \
  https://github.com/PrimeIntellect-ai/verifiers.git
git -C "$PRIME_DIR" submodule update --init --depth=1 \
  deps/verifiers deps/renderers deps/pydantic-config deps/research-environments

for patch in \
  "$PWD/patches/prime-rl-warn-on-truncation.patch" \
  "$PWD/patches/prime-rl-eos-token.patch" \
  "$PWD/patches/prime-rl-predict.patch"
do
  if ! git -C "$PRIME_DIR" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$PRIME_DIR" apply --check "$patch"
    git -C "$PRIME_DIR" apply "$patch"
  fi
done

test "$(git -C "$PRIME_DIR/deps/verifiers" rev-parse HEAD)" = "$VERIFIERS_COMMIT"
uv sync --locked

uv sync --project "$PRIME_DIR" --extra flash-attn --no-build-package flash-attn
uv run --project "$PRIME_DIR" --extra flash-attn python - <<'PY'
import flash_attn

print("flash-attn", flash_attn.__version__)
PY

echo "PREDICT is ready. PRIME-RL training dependencies are synced with the prebuilt flash-attn wheel."
