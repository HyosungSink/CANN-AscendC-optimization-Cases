#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-verify}
if [[ "$MODE" == --help || "$MODE" == help ]]; then
  cat <<'HELP_TEXT'
Usage: bash code/reproduce.sh <verify|checkout|run|profile>
  verify      Check local source and release-evidence hashes (offline).
  checkout    Fetch the fixed complete PTO source (explicit network action).
  run         Generate/build/run the fixed noncausal Flash Attention case.
  profile     Perform run, then use the installed msprof on the same case.
Environment: SOURCE_DIR, SOC_VERSION (A2/A3; default Ascend910B1), JOBS,
             QK_PRELOAD (1,2,4; default 4), NPU_ID (default 0), SYS_CNT_MULTIPLE (default 20.0).
NPU modes require prepared CANN/NumPy/CMake/compiler dependencies; no packages are installed.
HELP_TEXT
  exit 0
fi
COMMIT=377526952e95c39f9149cb94d261eba9e394f0d4
SOURCE_DIR=${SOURCE_DIR:-"$SCRIPT_DIR/work/pto-isa"}
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
    git -C "$SOURCE_DIR" remote add origin https://gitcode.com/cann/pto-isa.git
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
SOC_VERSION=${SOC_VERSION:-Ascend910B1}
case "$SOC_VERSION" in Ascend910B*|Ascend910_9382|Ascend910_9391) ;; *) echo "Only A2/A3 SoCs are accepted; confirm the exact SoC with npu-smi" >&2; exit 1;; esac
[[ "$MODE" == run || "$MODE" == profile ]] || { echo "Modes: verify, checkout, run, profile" >&2; exit 1; }
command -v npu-smi >/dev/null || { echo "A2/A3 NPU environment required" >&2; exit 1; }
CASE_DIR="$SOURCE_DIR/kernels/manual/common/flash_atten"
# The upstream data generator uses CASE_DIR/build regardless of its cwd.
BUILD_DIR="$CASE_DIR/build"
mkdir -p "$BUILD_DIR"
QK_PRELOAD=${QK_PRELOAD:-4}
NPU_ID=${NPU_ID:-0}
SYS_CNT_MULTIPLE=${SYS_CNT_MULTIPLE:-20.0}
case "$QK_PRELOAD" in 1|2|4) ;; *) echo "Use QK_PRELOAD=1,2,4" >&2; exit 1;; esac
[[ "$NPU_ID" =~ ^[0-9]+$ ]] || { echo "NPU_ID must be numeric" >&2; exit 1; }
[[ "$SYS_CNT_MULTIPLE" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "SYS_CNT_MULTIPLE must be numeric" >&2; exit 1; }
FA_CASE=128,128,1024,128,128
python3 "$CASE_DIR/scripts/generate_cases.py" --qk-preload "$QK_PRELOAD" --cases "$FA_CASE" --output-header "$BUILD_DIR/generated_cases.h" --output-json "$BUILD_DIR/generated_cases.json" --causal-mask 0
cd "$BUILD_DIR"
python3 "$CASE_DIR/scripts/gen_data.py" --cases "$FA_CASE" --causal-mask 0
cmake -S "$CASE_DIR" -B "$BUILD_DIR" -DRUN_MODE=npu -DSOC_VERSION="$SOC_VERSION"
cmake --build "$BUILD_DIR" -j "${JOBS:-8}"
cd "$BUILD_DIR"
./fa_performance --npu="$NPU_ID" --cases="$FA_CASE" --sys_cnt_multiple="$SYS_CNT_MULTIPLE"
if [[ "$MODE" == profile ]]; then
  command -v msprof >/dev/null || { echo "msprof must already be installed" >&2; exit 1; }
  msprof --application="./fa_performance --npu=${NPU_ID} --cases=${FA_CASE} --sys_cnt_multiple=${SYS_CNT_MULTIPLE}"
fi
