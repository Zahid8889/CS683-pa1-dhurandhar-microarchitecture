#!/usr/bin/env bash
# Runs ./bin/matmul separately for 'naive' and 'prefetch' at several matrix sizes,
# under perf stat, and saves L1D + L2 miss counts for each to a CSV.
#
# Usage: ./run_matmul_cache_sweep.sh <output_csv_filename>
#
# NOTE: the L2 event below (l2_cache_req_stat.ls_rd_blk_c) was found via
# `perf list` on this specific AMD CPU (HP Victus, Zen-based). If you're running
# this on a different machine, check `perf list | grep -i l2` first and update
# L2_EVENT if the name differs - see prior conversation for how we found this one.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_csv_filename>"
    exit 1
fi

output_csv="$1"
sizes=(128 256 512 1024 2048)

L1_EVENT="L1-dcache-load-misses"
L2_EVENT="l2_cache_req_stat.ls_rd_blk_c"
PERF_EVENTS="instructions,${L1_EVENT},${L2_EVENT}"

# Extracts the value for a given perf event name from perf stat's stderr output.
# Returns 0 (with a warning to stderr) if the event is "<not supported>" or the
# line is missing entirely - same graceful-degradation approach as the Python
# scripts, since this same AMD PMU reported LLC-load-misses as unsupported earlier.
extract_perf_metric() {
    local output="$1"
    local event="$2"
    local line
    line=$(echo "$output" | grep -F "$event" || true)
    if [ -z "$line" ]; then
        echo "0"
        return
    fi
    local val
    val=$(echo "$line" | awk '{print $1}')
    if [ "$val" = "<not" ]; then
        echo "0"
    else
        echo "${val//,/}"
    fi
}

run_and_extract() {
    local stage="$1"
    local size="$2"
    local perf_output
    perf_output=$(perf stat -e "$PERF_EVENTS" ./bin/matmul "$stage" "$size" "$size" "$size" 2>&1 >/dev/null)

    local l1 l2
    l1=$(extract_perf_metric "$perf_output" "$L1_EVENT")
    l2=$(extract_perf_metric "$perf_output" "$L2_EVENT")

    if [ "$l1" = "0" ]; then
        echo "  [Warning] ${L1_EVENT} unsupported/missing for stage=${stage} size=${size}" >&2
    fi
    if [ "$l2" = "0" ]; then
        echo "  [Warning] ${L2_EVENT} unsupported/missing for stage=${stage} size=${size}" >&2
    fi

    echo "${l1},${l2}"
}

echo "Matrix_Size,Naive_L1D_Misses,Naive_L2_Misses,Prefetch_L1D_Misses,Prefetch_L2_Misses" > "$output_csv"

for size in "${sizes[@]}"; do
    echo "Running size ${size}x${size}x${size}..."

    naive_result=$(run_and_extract "naive" "$size")
    prefetch_result=$(run_and_extract "prefetch" "$size")

    naive_l1=$(echo "$naive_result" | cut -d',' -f1)
    naive_l2=$(echo "$naive_result" | cut -d',' -f2)
    prefetch_l1=$(echo "$prefetch_result" | cut -d',' -f1)
    prefetch_l2=$(echo "$prefetch_result" | cut -d',' -f2)

    echo "${size},${naive_l1},${naive_l2},${prefetch_l1},${prefetch_l2}" >> "$output_csv"
done

echo "Done. Results saved to ${output_csv}"