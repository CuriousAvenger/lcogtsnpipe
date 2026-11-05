#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

# Default to 2023ixf-2m0, or accept a specific case (e.g. basic-reduction/2023ixf-2m0)
case_input=${1:-basic-reduction/2023ixf-2m0}
case_file="$script_dir/cases/${case_input%.sh}.sh"

if [[ ! -f "$case_file" && -f "$script_dir/cases/basic-reduction/${case_input%.sh}.sh" ]]; then
    case_file="$script_dir/cases/basic-reduction/${case_input%.sh}.sh"
fi

[[ -f "$case_file" ]] || { echo "Unknown reduction case: $case_input" >&2; exit 2; }

# shellcheck source=/dev/null
source "$case_file"

# Check for LCO archive API key
if [[ -z "${LCO_API_KEY:-}" ]]; then
    echo "Error: LCO_API_KEY is not set. Please export LCO_API_KEY='<token>'." >&2
    exit 1
fi

case_name=$(basename "$case_file" .sh)
run_id="${case_name}-$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="${REDUCTION_RESULTS_DIR:-$repo_root/artifacts/reduction-runs}/$run_id"
data_dir="$run_dir/data"
log_dir="$run_dir/logs"

mkdir -p "$data_dir/standard/cat/apass" "$data_dir/standard/cat/landolt" "$log_dir"
export REDUCTION_WORKDIR="$data_dir"

compose=(docker compose -p "lcogt-$run_id" -f "$script_dir/compose.yml")

cleanup() {
    status=$?
    "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
    if ((status != 0)); then
        echo "Reduction failed. Logs and data are in $run_dir" >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Building and starting pipeline for $case_name..."
"${compose[@]}" build pipeline
"${compose[@]}" up -d db pipeline

run_pipeline() {
    local command=$1
    step_number=$((step_number + 1))
    local log_file="$log_dir/$(printf '%02d' "$step_number").log"
    echo "+ $command"
    if ! "${compose[@]}" exec -T pipeline bash -lc \
        'source /opt/conda/etc/profile.d/conda.sh && conda activate lcogtsnpipe && export MPLBACKEND=Agg TERM=xterm && eval "$1"' \
        reduction-command "$command" \
        >"$log_file" 2>&1; then
        echo "Failed: $command" >&2
        tail -n 80 "$log_file" >&2
        return 1
    fi
}

step_number=0
run_pipeline 'bash /lcogtsnpipe/init-db.sh'
run_case
echo "Reduction passed. Logs and data are in $run_dir"
