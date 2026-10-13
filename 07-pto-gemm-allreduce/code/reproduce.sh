#!/usr/bin/env bash
set -euo pipefail

CASE_UPSTREAM=https://github.com/hw-native-sys/pto-isa.git
CASE_COMMIT=15a9e0a0845955f5d7a409a7f4d1609a263b5d25
CASE_MODE=${1:---help}
CASE_WORKDIR=${CASE_WORKDIR:-"$PWD/work/pto-gemm-ar"}
CASE_REPO="$CASE_WORKDIR/upstream"
CASE_BUILD="$CASE_WORKDIR/build"

case "$CASE_MODE" in
  --help|help)
    cat <<'USAGE'
Usage: bash reproduce.sh {fetch|build|run}
fetch: get the pinned upstream source in CASE_WORKDIR/upstream.
build: build Cube, Vector and Host targets on Linux with CANN 9.0.0+.
run:   execute the existing binary under MPICH (default 8 ranks).
Set ASCEND_CANN_PATH to the selected CANN set_env.sh for build/run.
Optional: CASE_M=5416 CASE_K=6144 CASE_N=1408 CASE_COMPUTE_BLOCKS=24
          CASE_COMM_BLOCKS=24 CASE_NRANKS=8 CASE_FIRST_DEVICE=0
No package installation or system-wide IPC cleanup is performed.
USAGE
    exit 0 ;;
  fetch|build|run) ;;
  *) printf 'Unknown mode: %s\n' "$CASE_MODE" >&2; exit 2 ;;
esac

if [[ "$CASE_MODE" == fetch ]]; then
  command -v git >/dev/null
  if [[ -e "$CASE_REPO" ]]; then
    [[ -d "$CASE_REPO/.git" ]] || { echo 'Existing source directory is not a Git checkout.' >&2; exit 2; }
    [[ "$(git -C "$CASE_REPO" remote get-url origin)" == "$CASE_UPSTREAM" ]] || { echo 'Existing checkout has a different origin.' >&2; exit 2; }
    [[ "$(git -C "$CASE_REPO" rev-parse HEAD)" == "$CASE_COMMIT" ]] || { echo 'Existing checkout is not at the pinned commit; use another CASE_WORKDIR.' >&2; exit 2; }
    [[ -z "$(git -C "$CASE_REPO" status --porcelain)" ]] || { echo 'Existing checkout has local changes; use another CASE_WORKDIR.' >&2; exit 2; }
    echo 'Pinned checkout already present.'
    exit 0
  fi
  mkdir -p "$CASE_WORKDIR"
  git init "$CASE_REPO"
  git -C "$CASE_REPO" remote add origin "$CASE_UPSTREAM"
  git -C "$CASE_REPO" sparse-checkout init --cone
  git -C "$CASE_REPO" sparse-checkout set include kernels/manual/a2a3/gemm_ar
  git -C "$CASE_REPO" fetch --depth 1 origin "$CASE_COMMIT"
  git -C "$CASE_REPO" checkout --detach FETCH_HEAD
  exit 0
fi

[[ "$(uname -s)" == Linux ]] || { echo 'NPU build/run requires Linux and A2/A3 hardware.' >&2; exit 2; }
[[ -d "$CASE_REPO/.git" ]] || { echo 'Run fetch first.' >&2; exit 2; }
[[ "$(git -C "$CASE_REPO" rev-parse HEAD)" == "$CASE_COMMIT" ]] || { echo 'Unexpected source revision.' >&2; exit 2; }
[[ -n "${ASCEND_CANN_PATH:-}" && -f "$ASCEND_CANN_PATH" ]] || { echo 'Set ASCEND_CANN_PATH to CANN 9.0.0+ set_env.sh.' >&2; exit 2; }
source "$ASCEND_CANN_PATH"
: "${ASCEND_HOME_PATH:?CANN environment did not set ASCEND_HOME_PATH}"

if [[ "$CASE_MODE" == build ]]; then
  command -v bisheng >/dev/null
  command -v cmake >/dev/null
  for CASE_NUMBER in "${CASE_M:-5416}" "${CASE_K:-6144}" "${CASE_N:-1408}" "${CASE_COMPUTE_BLOCKS:-24}" "${CASE_COMM_BLOCKS:-24}"; do
    [[ "$CASE_NUMBER" =~ ^[1-9][0-9]*$ ]] || { echo 'Shape and block counts must be positive integers.' >&2; exit 2; }
  done
  cmake -S "$CASE_REPO/kernels/manual/a2a3/gemm_ar" -B "$CASE_BUILD" \
    -DRUN_MODE=npu -DSOC_VERSION="${CASE_SOC_VERSION:-Ascend910B1}" \
    -DCONFIG_G_M="${CASE_M:-5416}" -DCONFIG_G_K="${CASE_K:-6144}" -DCONFIG_G_N="${CASE_N:-1408}" \
    -DCOMPUTE_BLOCKS="${CASE_COMPUTE_BLOCKS:-24}" -DCOMM_BLOCKS="${CASE_COMM_BLOCKS:-24}"
  cmake --build "$CASE_BUILD" --parallel "${CASE_BUILD_JOBS:-8}"
  exit 0
fi

command -v mpirun >/dev/null
[[ -x "$CASE_BUILD/gemm_allreduce" ]] || { echo 'Run build first.' >&2; exit 2; }
for CASE_NUMBER in "${CASE_NRANKS:-8}" "${CASE_FIRST_DEVICE:-0}"; do
  [[ "$CASE_NUMBER" =~ ^[0-9]+$ ]] || { echo 'Rank count and first device must be integers.' >&2; exit 2; }
done
[[ "${CASE_NRANKS:-8}" -gt 0 ]] || { echo 'Rank count must be positive.' >&2; exit 2; }
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export GEMM_AR_DIR="$CASE_REPO/kernels/manual/a2a3/gemm_ar"
mpirun -n "${CASE_NRANKS:-8}" "$CASE_BUILD/gemm_allreduce" --first-device "${CASE_FIRST_DEVICE:-0}"
