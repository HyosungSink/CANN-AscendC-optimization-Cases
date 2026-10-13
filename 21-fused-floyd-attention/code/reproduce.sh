#!/usr/bin/env bash
set -euo pipefail
CASE_MODE=${1:---help}
CASE_REPO_URL=https://gitcode.com/cann/ops-transformer.git
CASE_COMMIT=afe72144f9f2ac8441929035795db88a111b30c5
CASE_OPERATOR=fused_floyd_attention
CASE_WORKDIR=${CASE_WORKDIR:-"$PWD/work/$CASE_OPERATOR"}
CASE_REPO="$CASE_WORKDIR/upstream"
case "$CASE_MODE" in
 --help|help)
  echo "Usage: bash reproduce.sh {fetch|build}"
  echo "fetch obtains the pinned repository under CASE_WORKDIR; build requires Linux and sourced CANN 9.0.0."
  echo "Set CASE_SOC=ascend910b for A2 or ascend910_93 for supported A3. No package installation is performed."
  exit 0 ;;
 fetch)
  if [[ -e "$CASE_REPO" ]]; then
   [[ -d "$CASE_REPO/.git" ]] || { echo 'Existing directory is not a Git checkout.' >&2; exit 2; }
   [[ "$(git -C "$CASE_REPO" remote get-url origin)" == "$CASE_REPO_URL" ]] || { echo 'Different origin; use another CASE_WORKDIR.' >&2; exit 2; }
   [[ "$(git -C "$CASE_REPO" rev-parse HEAD)" == "$CASE_COMMIT" ]] || { echo 'Different commit; use another CASE_WORKDIR.' >&2; exit 2; }
   [[ -z "$(git -C "$CASE_REPO" status --porcelain)" ]] || { echo 'Local changes present; use another CASE_WORKDIR.' >&2; exit 2; }
   echo 'Pinned source already present.'
   exit 0
  fi
  mkdir -p "$CASE_WORKDIR"
  git init "$CASE_REPO"
  git -C "$CASE_REPO" remote add origin "$CASE_REPO_URL"
  git -C "$CASE_REPO" fetch --depth 1 origin "$CASE_COMMIT"
  git -C "$CASE_REPO" checkout --detach FETCH_HEAD
  exit 0 ;;
 build) ;;
 *) echo 'Unknown mode.' >&2; exit 2 ;;
esac
[[ "$(uname -s)" == Linux ]] || { echo 'NPU compilation requires Linux and CANN.' >&2; exit 2; }
[[ -d "$CASE_REPO/.git" ]] || { echo 'Run fetch first.' >&2; exit 2; }
[[ "$(git -C "$CASE_REPO" rev-parse HEAD)" == "$CASE_COMMIT" ]] || { echo 'Unexpected source revision.' >&2; exit 2; }
[[ -z "$(git -C "$CASE_REPO" status --porcelain)" ]] || { echo 'Source has local changes; record an experimental diff and build manually.' >&2; exit 2; }
: "${ASCEND_HOME_PATH:?Source the matching CANN Toolkit environment first}"
command -v bisheng >/dev/null
CASE_SOC=${CASE_SOC:-ascend910b}
case "$CASE_SOC" in ascend910b|ascend910_93) ;; *) echo 'Only A2/A3 targets are accepted.' >&2; exit 2;; esac
case_guard_opbase() {
 local dependency_dir="$1"
 [[ -e "$dependency_dir" ]] || return 0
 local canonical_dir git_root
 canonical_dir="$(cd "$dependency_dir" && pwd -P)"
 git_root="$(git -C "$dependency_dir" rev-parse --show-toplevel 2>/dev/null)" || { echo 'Unverified opbase directory.' >&2; exit 2; }
 [[ "$(cd "$git_root" && pwd -P)" == "$canonical_dir" ]] || { echo 'opbase is not an independent Git checkout.' >&2; exit 2; }
 [[ "$(git -C "$dependency_dir" rev-parse HEAD)" == "929902e577bb809a377a6ab17cb7b00a4dfddc1a" ]] || { echo 'opbase overrides the pinned dependency; use an isolated CASE_WORKDIR.' >&2; exit 2; }
 [[ -z "$(git -C "$dependency_dir" status --porcelain)" ]] || { echo 'opbase has local changes.' >&2; exit 2; }
}
CASE_PARENT_OPBASE="$(cd "$CASE_REPO/../.." && pwd)/ops-base"
case_guard_opbase "$CASE_PARENT_OPBASE"
case_guard_opbase "$CASE_REPO/third_party/opbase"
cd "$CASE_REPO"
bash build.sh --pkg --ops="$CASE_OPERATOR" --soc="$CASE_SOC" -j "${CASE_BUILD_JOBS:-8}"
