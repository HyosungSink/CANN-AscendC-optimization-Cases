#!/usr/bin/env bash
# Authored wrapper. Upstream files retain their original CANN license.
set -euo pipefail
CASE_NAME="22-deformable-conv2d"
OPS_COMMIT="0413e303e6cd1986b592c3e039f2579ac0f10b62"
OPBASE_COMMIT="ec88904b611b04f9b23b529aee66c78bb7cf6140"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_WORK_DIR="${CASE_WORK_DIR:-$PWD/work/$CASE_NAME}"
REPO_DIR="$CASE_WORK_DIR/ops-nn"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$CASE_WORK_DIR/third_party}"
ACTION="${1:-help}"
SOC="${SOC:-ascend910b}"
fail() { printf '%s\n' "$*" >&2; exit 1; }
check_soc() {
  case "$SOC" in ascend910b|ascend910_93) ;; *) fail "Only A2 ascend910b / A3 ascend910_93 are accepted." ;; esac
}
check_linux_cann() {
  [[ "$(uname -s)" == Linux ]] || fail "Build/run requires Linux and a matching CANN installation; this host has not built or timed the NPU kernel."
  [[ "${CANN_VERSION:-}" == 9.0.0 ]] || fail "Set CANN_VERSION=9.0.0 after checking the installed Toolkit/ops versions. This wrapper pins the 9.0.0 release sources."
  [[ -n "${ASCEND_HOME_PATH:-}" ]] || fail "Source your installed CANN set_env.sh/setenv.bash first."
  command -v bisheng >/dev/null || fail "bisheng is not in PATH."
  check_soc
}
check_repo() {
  [[ -d "$REPO_DIR/.git" ]] || fail "Run fetch first or set CASE_WORK_DIR to the prepared full checkout."
  [[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$OPS_COMMIT" ]] || fail "ops-nn commit mismatch."
  [[ -z "$(git -C "$REPO_DIR" status --porcelain)" ]] || fail "The fixed source checkout has local changes; record experimental edits in a separate commit and update the pins. Source snapshots are unchanged."
}
clone_pin() {
  local url="$1" target="$2" commit="$3"
  if [[ -e "$target" ]]; then
    [[ -d "$target/.git" ]] || fail "Existing target is not a Git checkout: $target"
    [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || fail "Existing checkout has a different commit: $target"
    return
  fi
  git init -q "$target"
  git -C "$target" remote add origin "$url"
  git -C "$target" fetch --depth=1 origin "$commit"
  git -C "$target" checkout --detach FETCH_HEAD
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || fail "Fetched commit mismatch."
}
case "$ACTION" in
  verify)
    python3 - "$SCRIPT_DIR" <<'PYVERIFY'
import hashlib, json, pathlib, sys
p=pathlib.Path(sys.argv[1])
m=json.loads((p/'source-manifest.json').read_text())
for item in m['files']:
    actual=hashlib.sha256((p/'upstream'/item['path']).read_bytes()).hexdigest()
    if actual != item['sha256']: raise SystemExit('hash mismatch: '+item['path'])
print('Snapshot verified:', len(m['files']), 'files; commit', m['commit'])
PYVERIFY
    ;;
  fetch)
    mkdir -p "$CASE_WORK_DIR" "$THIRD_PARTY_DIR"
    clone_pin https://gitcode.com/cann/ops-nn.git "$REPO_DIR" "$OPS_COMMIT"
    clone_pin https://gitcode.com/cann/opbase.git "$THIRD_PARTY_DIR/opbase" "$OPBASE_COMMIT"
    printf 'Sources pinned. Prepare offline archives in %s/pkg before build.\n' "$THIRD_PARTY_DIR"
    ;;
  build)
    check_linux_cann
    check_repo
    [[ -d "$THIRD_PARTY_DIR/opbase/.git" ]] || fail "Prepare the pinned opbase checkout by running fetch."
    [[ "$(git -C "$THIRD_PARTY_DIR/opbase" rev-parse HEAD)" == "$OPBASE_COMMIT" ]] || fail "opbase commit mismatch."
    [[ -z "$(git -C "$THIRD_PARTY_DIR/opbase" status --porcelain)" ]] || fail "The pinned opbase checkout has local changes."
    # Upstream CMake prefers ../../ops-base over the supplied dependency cache.
    if [[ -e "$REPO_DIR/../../ops-base" ]]; then
      [[ -d "$REPO_DIR/../../ops-base/.git" ]] || fail "CMake would select an unverified ../../ops-base source tree."
      [[ "$(git -C "$REPO_DIR/../../ops-base" rev-parse HEAD)" == "$OPBASE_COMMIT" ]] || fail "CMake would select a different ../../ops-base commit; use an isolated work directory."
      [[ -z "$(git -C "$REPO_DIR/../../ops-base" status --porcelain)" ]] || fail "CMake would select a dirty ../../ops-base checkout."
    fi
    # Refuse to invoke upstream build until its known package dependencies are local.
    # The wrapper does not install CANN, download binary packages, or run sudo.
    for archive in include.zip eigen-5.0.0.tar.gz makeself-release-2.5.0-patch1.tar.gz protobuf-25.1.tar.gz abseil-cpp-20230802.1.tar.gz; do
      [[ -f "$THIRD_PARTY_DIR/pkg/$archive" ]] || fail "Offline dependency missing: $THIRD_PARTY_DIR/pkg/$archive (see docs/reproduce.md)."
    done
    cd "$REPO_DIR"
    bash build.sh --pkg --soc="$SOC" --ops=deformable_conv2d --vendor_name=blog22 --cann_3rd_lib_path="$THIRD_PARTY_DIR" -j"${BUILD_JOBS:-8}"
    printf 'Package built under %s/build_out. Installation is a separate documented step.\n' "$REPO_DIR"
    ;;
  run-example)
    check_linux_cann
    check_repo
    : "${CUSTOM_OPAPI_ROOT:?Set CUSTOM_OPAPI_ROOT to the installed custom vendor op_api directory.}"
    [[ -d "$CUSTOM_OPAPI_ROOT/include" && -d "$CUSTOM_OPAPI_ROOT/lib" ]] || fail "CUSTOM_OPAPI_ROOT must contain include/ and lib/."
    mkdir -p "$CASE_WORK_DIR/bin"
    g++ "$REPO_DIR/conv/deformable_conv2d/examples/test_aclnn_deformable_conv2d.cpp" -I"$CUSTOM_OPAPI_ROOT/include" -I"$ASCEND_HOME_PATH/include" \
      -I"$ASCEND_HOME_PATH/include/aclnnop" -L"$CUSTOM_OPAPI_ROOT/lib" -L"$ASCEND_HOME_PATH/lib64" \
      -Wl,-rpath,"$CUSTOM_OPAPI_ROOT/lib" -lcust_opapi -lopapi_math -lascendcl -lnnopbase \
      -o "$CASE_WORK_DIR/bin/deformable_conv2d_smoke"
    "$CASE_WORK_DIR/bin/deformable_conv2d_smoke"
    ;;
  help|*)
    printf 'Usage: bash %s {verify|fetch|build|run-example}\n' "${BASH_SOURCE[0]}"
    printf 'fetch retrieves only pinned source repositories. build requires Linux, CANN_VERSION=9.0.0, sourced CANN, and prepared offline dependencies.\n'
    printf 'SOC=ascend910b (A2) or ascend910_93 (A3). No package installation or timing is performed automatically.\n'
    ;;
esac
