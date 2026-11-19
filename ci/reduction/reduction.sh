#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

TAG=${1:-local}
WORKSPACE=${2:-$repo_root}
CASE_INPUT=${3:-basic-reduction/2023ixf-2m0}

case_file="$script_dir/cases/${CASE_INPUT%.sh}.sh"
[[ -f "$case_file" ]] || case_file="$script_dir/cases/basic-reduction/${CASE_INPUT%.sh}.sh"
[[ -f "$case_file" ]] || { echo "Unknown reduction case: $CASE_INPUT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$case_file"

[[ -n "${LCO_API_KEY:-}" ]] || { echo "Error: LCO_API_KEY is not set." >&2; exit 1; }

run_id="${TAG}-$(basename "$case_file" .sh)-$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="$repo_root/artifacts/reduction-runs/$run_id"
data_dir="$run_dir/data"
dump_dir="$repo_root/artifacts/dumps"

mkdir -p "$data_dir/standard/cat/apass" "$data_dir/standard/cat/landolt" "$run_dir/logs" "$dump_dir"
chmod -R 777 "$run_dir"

export REDUCTION_WORKDIR="$data_dir"
export PIPELINE_IMAGE="lcogtsnpipe:${TAG}"
export PIPELINE_CONTEXT="$(cd "$WORKSPACE" && pwd)"
export PIPELINE_DOCKERFILE="$repo_root/Dockerfile"

compose=(docker compose -p "lcogt-$run_id" -f "$script_dir/compose.yml")

cleanup() {
    status=$?
    "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=== Building & starting $TAG pipeline from $WORKSPACE ==="
"${compose[@]}" build pipeline
"${compose[@]}" up -d db pipeline
"${compose[@]}" exec -u 0:0 -T pipeline chown -R supernova:supernova /supernova

run_pipeline() {
    local cmd=$1
    step_number=$((step_number + 1))
    local log_file="$run_dir/logs/$(printf '%02d' "$step_number").log"
    echo "+ $cmd"
    "${compose[@]}" exec -T pipeline bash -lc \
        'source /opt/conda/etc/profile.d/conda.sh && conda activate lcogtsnpipe && export MPLBACKEND=Agg TERM=xterm && eval "$1"' \
        reduction-command "$cmd" 2>&1 | tee "$log_file"
}

step_number=0
run_pipeline 'bash /lcogtsnpipe/init-db.sh'
run_case

echo "=== Dumping photlco for $TAG ==="
run_pipeline "python - <<'PY'
import json
import MySQLdb
import MySQLdb.cursors

connection = MySQLdb.connect(host='db', user='supernova', passwd='supernova', db='supernova')
cursor = connection.cursor(MySQLdb.cursors.DictCursor)
cursor.execute('SELECT * FROM photlco')
with open('/supernova/${TAG}_dump.json', 'w') as dump:
    json.dump(cursor.fetchall(), dump, default=str, indent=2)
PY"
cp "$data_dir/${TAG}_dump.json" "$dump_dir/"

echo "=== Reduction complete for $TAG: dump saved to $dump_dir ==="
