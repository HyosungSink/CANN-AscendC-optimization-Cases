#!/usr/bin/env bash
set -euo pipefail

OP="add_rms_norm_quant"
VENDOR="blog_add_rms_norm_quant"
COMMIT="0413e303e6cd1986b592c3e039f2579ac0f10b62"
OPBASE_COMMIT="ec88904b611b04f9b23b529aee66c78bb7cf6140"
REPO="https://gitcode.com/cann/ops-nn.git"
ACTION="${1:-fetch}"
SOC="${2:-ascend910b}"
BLOG_WORK_DIR="${BLOG_WORK_DIR:-${TMPDIR:-/tmp}/cann-blog-${OP}}"
CHECKOUT="${BLOG_WORK_DIR}/ops-nn"

if [[ "$ACTION" == "--help" || "$ACTION" == "help" ]]; then
    cat <<'HELP'
Usage: bash reproduce.sh [fetch|build|example|ut-host|ut-api] [ascend910b | ascend910_93]
Default: fetch only. Build/example/UT require Linux with matched CANN 9.0.0.
Set BLOG_WORK_DIR to choose the scratch checkout parent directory.
Set BLOG_THIRD_PARTY_DIR to use a prepared third-party source directory.
No CANN package install, no sudo, no generated .run package install.
HELP
    exit 0
fi
case "$ACTION" in
    fetch|build|example|ut-host|ut-api) ;;
    *) printf 'Unknown action: %s\n' "$ACTION" >&2; exit 2 ;;
esac
case "$SOC" in
    ascend910b|ascend910_93) ;;
    *) printf 'Unsupported target for this snapshot: %s\n' "$SOC" >&2; exit 2 ;;
esac

mkdir -p "$BLOG_WORK_DIR"
if [[ ! -d "$CHECKOUT/.git" ]]; then
    git clone --depth 1 --branch v9.0.0 "$REPO" "$CHECKOUT"
fi
ACTUAL_COMMIT="$(git -C "$CHECKOUT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$COMMIT" ]]; then
    printf 'Checkout mismatch: expected %s, found %s. Use a fresh BLOG_WORK_DIR.\n' "$COMMIT" "$ACTUAL_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git -C "$CHECKOUT" status --porcelain)" ]]; then
    printf 'Checkout has local changes; refusing to label it as the pristine pinned reproduction.\n' >&2
    exit 1
fi
printf 'Pinned checkout: %s (%s)\n' "$CHECKOUT" "$ACTUAL_COMMIT"
if [[ "$ACTION" == "fetch" ]]; then
    exit 0
fi
# Upstream CMake prefers ../../ops-base over any third-party cache.
# Verify every preexisting selectable source before a compile/run action.
verify_existing_opbase() {
    local candidate="$1"
    if [[ ! -e "$candidate" ]]; then
        return 0
    fi
    local source_root repo_root source_commit
    source_root="$(cd "$candidate" && pwd -P)"
    repo_root="$(git -C "$candidate" rev-parse --show-toplevel 2>/dev/null)" || {
        printf 'Unverified opbase source selected by CMake: %s. Use an isolated BLOG_WORK_DIR.\n' "$candidate" >&2
        exit 1
    }
    repo_root="$(cd "$repo_root" && pwd -P)"
    if [[ "$repo_root" != "$source_root" ]]; then
        printf 'opbase path is not its own Git checkout: %s\n' "$candidate" >&2
        exit 1
    fi
    source_commit="$(git -C "$candidate" rev-parse HEAD)"
    if [[ "$source_commit" != "$OPBASE_COMMIT" ]]; then
        printf 'opbase commit mismatch at %s: expected %s, found %s. Use an isolated BLOG_WORK_DIR.\n' "$candidate" "$OPBASE_COMMIT" "$source_commit" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$candidate" status --porcelain)" ]]; then
        printf 'opbase checkout has local changes at %s; refusing an unverified dependency.\n' "$candidate" >&2
        exit 1
    fi
}
verify_existing_opbase "$CHECKOUT/../../ops-base"
verify_existing_opbase "${BLOG_THIRD_PARTY_DIR:-$CHECKOUT/third_party}/opbase"
if [[ "$(uname -s)" != "Linux" ]]; then
    printf 'NPU build/run requires Linux. Fetch completed; see docs/reproduce.md.\n' >&2
    exit 1
fi
: "${ASCEND_HOME_PATH:?Source the matched CANN 9.0.0 environment first and set ASCEND_HOME_PATH}"
cd "$CHECKOUT"
THIRD_PARTY_ARGS=()
if [[ -n "${BLOG_THIRD_PARTY_DIR:-}" ]]; then
    THIRD_PARTY_ARGS+=("--cann_3rd_lib_path=${BLOG_THIRD_PARTY_DIR}")
fi
case "$ACTION" in
    build)
        bash build.sh --pkg "--soc=$SOC" "--ops=$OP" "--vendor_name=$VENDOR" "${THIRD_PARTY_ARGS[@]}"
        printf 'Build invocation finished. Review the generated build_out/*.run before manual installation.\n'
        ;;
    example)
        bash build.sh --run_example "$OP" eager cust "--vendor_name=$VENDOR" "--soc=$SOC"
        ;;
    ut-host)
        bash build.sh -u --ophost "--ops=$OP" "--soc=$SOC" "${THIRD_PARTY_ARGS[@]}"
        ;;
    ut-api)
        bash build.sh -u --opapi "--ops=$OP" "--soc=$SOC" "${THIRD_PARTY_ARGS[@]}"
        ;;
esac
