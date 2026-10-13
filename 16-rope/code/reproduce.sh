#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-verify}
if [[ "$MODE" == --help || "$MODE" == help ]]; then
  cat <<'HELP_TEXT'
Usage: bash code/reproduce.sh <verify|checkout|build|run>
  verify      Check local source and release-evidence hashes (offline).
  checkout    Fetch the fixed complete upstream source (explicit network action).
  build       Build the rotary_position_embedding custom package.
  run         Build/run its ACLNN example using the prepared custom package.
Environment: SOURCE_DIR, SOC_VERSION (ascend910b or ascend910_93; default ascend910b).
Build/run require matching CANN dependencies; package installation is manual.
HELP_TEXT
  exit 0
fi
COMMIT=afe72144f9f2ac8441929035795db88a111b30c5
SOURCE_DIR=${SOURCE_DIR:-"$SCRIPT_DIR/work/ops-transformer"}
if [[ "$MODE" == verify ]]; then
  python3 - "$SCRIPT_DIR" <<'PYVERIFY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'source-manifest.json').read_text())
for item in manifest['files']:
    data = (root / 'upstream' / item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
for item in manifest.get('evidence_sources', []):
    assert hashlib.sha256((root / item['local_path']).read_bytes()).hexdigest() == item['sha256']
print(f"Verified {len(manifest['files'])} source files and release evidence at {manifest['commit']}")
PYVERIFY
  exit 0
fi
if [[ "$MODE" == checkout ]]; then
  if [[ -e "$SOURCE_DIR" ]]; then
    [[ -d "$SOURCE_DIR/.git" ]] || { echo "SOURCE_DIR is not a Git checkout" >&2; exit 1; }
  else
    mkdir -p "$SOURCE_DIR"
    git -C "$SOURCE_DIR" init
    git -C "$SOURCE_DIR" remote add origin https://gitcode.com/cann/ops-transformer.git
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$COMMIT"
    git -C "$SOURCE_DIR" checkout --detach "$COMMIT"
  fi
  [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$COMMIT" ]] || { echo "Wrong commit; use a separate SOURCE_DIR" >&2; exit 1; }
  echo "Pinned source ready. No dependency installation performed."
  exit 0
fi
[[ -d "$SOURCE_DIR/.git" ]] || { echo "Run checkout first or set SOURCE_DIR" >&2; exit 1; }
[[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$COMMIT" ]] || { echo "Wrong source commit" >&2; exit 1; }
[[ -n "${ASCEND_HOME_PATH:-}" ]] || { echo "Source your installed CANN environment first" >&2; exit 1; }
SOC_VERSION=${SOC_VERSION:-ascend910b}
case "$SOC_VERSION" in ascend910b|ascend910_93) ;; *) echo "Select A2/A3 only: ascend910b or ascend910_93" >&2; exit 1;; esac
cd "$SOURCE_DIR"
case "$MODE" in
  build) bash build.sh --pkg --soc="$SOC_VERSION" --ops=rotary_position_embedding --vendor_name=blog_rope -j8 ;;
  run) command -v npu-smi >/dev/null; bash build.sh --run_example rotary_position_embedding eager cust --vendor_name=blog_rope --soc="$SOC_VERSION" ;;
  *) echo "Modes: verify, checkout, build, run" >&2; exit 1 ;;
esac
