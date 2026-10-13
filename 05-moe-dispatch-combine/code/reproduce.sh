#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COMMIT=653e519cf9556b0cc8d9e7966aa4551eb65143ae
SOURCE_DIR=${SOURCE_DIR:-"$SCRIPT_DIR/work/sgl-kernel-npu"}
MODE=${1:-verify}
if [[ "$MODE" == --help || "$MODE" == help ]]; then
  cat <<'HELP_TEXT'
Usage: bash code/reproduce.sh <verify|checkout|build|test-int8|test-bf16>
  verify      Check the local upstream snapshot hashes (offline).
  checkout    Fetch the fixed complete upstream source (explicit network action).
  build       Build DeepEP; set HARDWARE=A2 or HARDWARE=A3.
  test-int8   Run the fixed low-latency operator test with INT8 Dispatch.
  test-bf16   Run the same test with BF16 Dispatch.
Environment: SOURCE_DIR, HARDWARE, NUM_PROCESSES (default 8).
Build/tests require a prepared matching CANN/NPU environment; no packages are installed.
HELP_TEXT
  exit 0
fi
if [[ "$MODE" == verify ]]; then
  python3 - "$SCRIPT_DIR" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'source-manifest.json').read_text())
for item in manifest['files']:
    data = (root / 'upstream' / item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
print(f"Verified {len(manifest['files'])} upstream files at {manifest['commit']}")
PY
  exit 0
fi
if [[ "$MODE" == checkout ]]; then
  if [[ -e "$SOURCE_DIR" ]]; then
    [[ -d "$SOURCE_DIR/.git" ]] || { echo "SOURCE_DIR exists but is not a Git checkout" >&2; exit 1; }
  else
    mkdir -p "$SOURCE_DIR"
    git -C "$SOURCE_DIR" init
    git -C "$SOURCE_DIR" remote add origin https://github.com/sgl-project/sgl-kernel-npu.git
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$COMMIT"
    git -C "$SOURCE_DIR" checkout --detach "$COMMIT"
  fi
  [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$COMMIT" ]] || { echo "Wrong source commit; use a separate SOURCE_DIR" >&2; exit 1; }
  echo "Pinned source ready. Prepare matching submodules/dependencies explicitly before building; see docs/reproduce.md."
  exit 0
fi
[[ -d "$SOURCE_DIR/.git" ]] || { echo "Run checkout first or set SOURCE_DIR to the full pinned checkout" >&2; exit 1; }
[[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$COMMIT" ]] || { echo "Source commit mismatch" >&2; exit 1; }
case "${HARDWARE:-}" in
  A2) BUILD_TARGET=deepep2; SOC_VERSION=Ascend910B1 ;;
  A3) BUILD_TARGET=deepep; SOC_VERSION=Ascend910_9382 ;;
  *) echo "Set HARDWARE=A2 or HARDWARE=A3; A5/950 is excluded" >&2; exit 1 ;;
esac
cd "$SOURCE_DIR"
case "$MODE" in
  build)
    [[ -n "${ASCEND_HOME_PATH:-}" ]] || { echo "Source the installed CANN set_env.sh first" >&2; exit 1; }
    bash build.sh -a "$BUILD_TARGET" "$SOC_VERSION"
    ;;
  test-int8|test-bf16)
    command -v npu-smi >/dev/null || { echo "NPU environment required" >&2; exit 1; }
    python3 -c 'import torch, torch_npu, deep_ep; assert torch.npu.is_available()'
    unset DEEP_NORMAL_MODE_USE_INT8_QUANT SGLANG_DEEPEP_BF16_DISPATCH
    TEST_ARGS=(--num-processes "${NUM_PROCESSES:-8}" --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --low-latency-strategy default)
    if [[ "$MODE" == test-int8 ]]; then TEST_ARGS+=(--use-fp8); fi
    python3 tests/python/deepep/test_low_latency.py "${TEST_ARGS[@]}"
    ;;
  *) echo "Modes: verify, checkout, build, test-int8, test-bf16" >&2; exit 1 ;;
esac
