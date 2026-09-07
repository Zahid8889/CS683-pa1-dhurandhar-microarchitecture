#!/usr/bin/env bash
# Runs ./bin/conv separately for 'naive' and a chosen stage, at K=3 and several
# H=W sizes, under perf stat, saving instructions + L1D + L2 miss counts to a CSV.
#
# Usage: ./run_conv_cache_sweep.sh <stage> <output_csv_filename>
# Example: ./run_conv_cache_sweep.sh optimized conv_cache_results.csv
#
# NOTE: L2_EVENT was found via `perf list` on this specific AMD CPU (HP Victus,
# Zen-based) - same event used in run_matmul_cache_sweep.sh. If running on a
# different machine, check `perf list | grep -i l2` first and update L2_EVENT.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <stage> <output_csv_filename>"
    echo "Example: $0 optimized conv_cache_results.csv"
    exit 1
fi

stage_arg="$1"
output_csv="$2"
sizes=(128 256 512 1024 2048)
K=3

L1_EVENT="L1-dcache-load-misses"
L2_EVENT="l2_cache_req_stat.ls_rd_blk_c"
PERF_EVENTS="instructions,${L1_EVENT},${L2_EVENT}"

# Extracts the value for a given perf event name from perf stat's stderr output.
# Returns 0 (with a warning) if the event is "<not supported>" or the line is
# missing entirely, rather than treating either as a fatal parse error.
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
    perf_output=$(perf stat -e "$PERF_EVENTS" ./bin/conv "$stage" "$size" "$size" "$K" 2>&1 >/dev/null)

    local inst l1 l2
    inst=$(extract_perf_metric "$perf_output" "instructions")
    l1=$(extract_perf_metric "$perf_output" "$L1_EVENT")
    l2=$(extract_perf_metric "$perf_output" "$L2_EVENT")

    if [ "$inst" = "0" ]; then
        echo "  [Warning] instructions unsupported/missing for stage=${stage} size=${size}" >&2
    fi
    if [ "$l1" = "0" ]; then
        echo "  [Warning] ${L1_EVENT} unsupported/missing for stage=${stage} size=${size}" >&2
    fi
    if [ "$l2" = "0" ]; then
        echo "  [Warning] ${L2_EVENT} unsupported/missing for stage=${stage} size=${size}" >&2
    fi

    echo "${inst},${l1},${l2}"
}

echo "Matrix_Size,Naive_Instructions,Naive_L1D_Misses,Naive_L2_Misses,${stage_arg}_Instructions,${stage_arg}_L1D_Misses,${stage_arg}_L2_Misses" > "$output_csv"

for size in "${sizes[@]}"; do
    echo "Running size ${size}x${size} K=${K}..."

    naive_result=$(run_and_extract "naive" "$size")
    stage_result=$(run_and_extract "$stage_arg" "$size")

    naive_inst=$(echo "$naive_result" | cut -d',' -f1)
    naive_l1=$(echo "$naive_result" | cut -d',' -f2)
    naive_l2=$(echo "$naive_result" | cut -d',' -f3)
    stage_inst=$(echo "$stage_result" | cut -d',' -f1)
    stage_l1=$(echo "$stage_result" | cut -d',' -f2)
    stage_l2=$(echo "$stage_result" | cut -d',' -f3)

    echo "${size},${naive_inst},${naive_l1},${naive_l2},${stage_inst},${stage_l1},${stage_l2}" >> "$output_csv"
done

echo "Done. Results saved to ${output_csv}"