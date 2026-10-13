#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-verify}
if [[ "$MODE" == --help || "$MODE" == help ]]; then
  cat <<'HELP_TEXT'
Usage: bash code/reproduce.sh <verify|cpu-check|checkout|benchmark> [benchmark arguments]
  verify      Check source hashes and adapted Python syntax (offline).
  cpu-check   Run scalar golden and A2/A3 device-name checks (offline).
  checkout    Fetch the fixed complete Triton-Ascend source (explicit network action).
  benchmark   Run the four Part 2 index-mapping variants on verified A2/A3 hardware.
For benchmark arguments: python3 code/examples/benchmark_part2.py --help
Environment: SOURCE_DIR. The benchmark requires prepared Triton-Ascend/CANN; no packages are installed.
HELP_TEXT
  exit 0
fi
if (( $# > 0 )); then shift; fi
COMMIT=2badfc89e70a9b7a5e88463a116c2feddce4b101
SOURCE_DIR=${SOURCE_DIR:-"$SCRIPT_DIR/work/triton-ascend"}
case "$MODE" in
  verify)
    python3 - "$SCRIPT_DIR" <<'PY'
import ast, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'source-manifest.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((root / 'upstream' / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
for item in (root / 'examples').glob('*.py'):
    ast.parse(item.read_text(), filename=str(item))
print(f"Verified {len(manifest['files'])} original files and all adapted Python ASTs")
PY
    ;;
  cpu-check)
    PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/examples/test_reference.py"
    PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/examples/test_device_guard.py"
    ;;
  checkout)
    if [[ -e "$SOURCE_DIR" ]]; then
      [[ -d "$SOURCE_DIR/.git" ]] || { echo "SOURCE_DIR exists but is not a Git checkout" >&2; exit 1; }
    else
      mkdir -p "$SOURCE_DIR"
      git -C "$SOURCE_DIR" init
      git -C "$SOURCE_DIR" remote add origin https://github.com/triton-lang/triton-ascend.git
      git -C "$SOURCE_DIR" fetch --depth 1 origin "$COMMIT"
      git -C "$SOURCE_DIR" checkout --detach "$COMMIT"
    fi
    [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$COMMIT" ]] || { echo "Wrong source commit; use a separate SOURCE_DIR" >&2; exit 1; }
    echo "Pinned source ready; no build or installation performed. See docs/reproduce.md."
    ;;
  benchmark)
    command -v npu-smi >/dev/null || { echo "NPU environment required; CPU checks are available via cpu-check" >&2; exit 1; }
    PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/examples/benchmark_part2.py" "$@"
    ;;
  *) echo "Modes: verify, cpu-check, checkout, benchmark [benchmark arguments]" >&2; exit 1 ;;
esac
