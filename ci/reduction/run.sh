#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
    echo "Usage: $0 METHOD/CASE (for example: basic-reduction/2017cbv)" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
case_name=$1
[[ "$case_name" =~ ^[a-z0-9_-]+/[a-z0-9_-]+$ ]] || {
    echo "Invalid reduction case name: $case_name" >&2
    exit 2
}
case_file="$script_dir/cases/$case_name.sh"
[[ -f "$case_file" ]] || { echo "Unknown reduction case: $case_name" >&2; exit 2; }
# Case files define DATASETS (space-separated fixture IDs) and run_case().
# shellcheck source=/dev/null
DATASETS=""
source "$case_file"

dataset_archive() {
    awk -v id="$1" '$1 == id {print $2}' "$script_dir/datasets.tsv"
}

dataset_checksum() {
    awk -v id="$1" '$1 == id {print $3}' "$script_dir/datasets.tsv"
}

dataset_local_archive() {
    awk -v id="$1" '$1 == id {print ($4 ? $4 : $2)}' "$script_dir/datasets.tsv"
}

checksum() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

valid_archive() {
    [[ -f "$1" && $(checksum "$1") == "$2" ]]
}

cache_dir=${REDUCTION_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lcogtsnpipe/datasets}
results_dir=${REDUCTION_RESULTS_DIR:-$repo_root/artifacts/reduction-runs}
run_id="${case_name//\//-}-$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="$results_dir/$run_id"
data_dir="$run_dir/data"
log_dir="$run_dir/logs"
mkdir -p "$cache_dir" "$data_dir/standard/cat/apass" \
    "$data_dir/standard/cat/landolt" "$log_dir"
export REDUCTION_WORKDIR="$data_dir"
export REDUCTION_DATASET_DIR="$cache_dir"

for dataset in $DATASETS; do
    archive=$(dataset_archive "$dataset")
    expected_sha=$(dataset_checksum "$dataset")
    [[ -n "$archive" && -n "$expected_sha" ]] || {
        echo "Dataset '$dataset' is missing from datasets.tsv" >&2
        exit 2
    }

    destination="$cache_dir/$archive"
    if valid_archive "$destination" "$expected_sha"; then
        continue
    fi

    local_archive="$repo_root/artifacts/$(dataset_local_archive "$dataset")"
    if valid_archive "$local_archive" "$expected_sha"; then
        cp "$local_archive" "$destination"
        continue
    fi

    url_variable="REDUCTION_DATASET_URL_$(printf '%s' "$dataset" | tr '[:lower:]' '[:upper:]')"
    dataset_url=$(printenv "$url_variable" 2>/dev/null || true)
    dataset_url=${dataset_url:-${REDUCTION_DATASET_URL:-}}
    if [[ -z "$dataset_url" ]]; then
        echo "Dataset $archive is unavailable. Set $url_variable or place it in artifacts/." >&2
        exit 1
    fi

    curl --fail --location --retry 3 --output "$destination.part" "$dataset_url"
    if ! valid_archive "$destination.part" "$expected_sha"; then
        rm -f "$destination.part"
        echo "Checksum mismatch for $archive" >&2
        exit 1
    fi
    mv "$destination.part" "$destination"
done

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

echo "Building the pipeline image for $case_name..."
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
