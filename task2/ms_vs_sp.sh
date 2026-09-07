#!/usr/bin/env bash
# Runs ./bin/matmul prefetch at several matrix sizes and saves
# Matrix_Size, Naive_time, Prefetch_Time, Speedup to a CSV.
#
# Usage: ./run_matmul_sweep.sh <output_csv_filename>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_csv_filename>"
    exit 1
fi

output_csv="$1"
sizes=(128 256 512 1024 2048)

echo "Matrix_Size,Naive_time,Prefetch_Time,Speedup" > "$output_csv"

for size in "${sizes[@]}"; do
    echo "Running size ${size}x${size}x${size}..."
    output=$(./bin/matmul prefetch "$size" "$size" "$size")

    # Use fields counted from the end (NF-2, NF), since "naive (ref)" splits
    # into two whitespace-separated fields while "prefetch" is only one -
    # fixed column numbers would grab the wrong field for one of the two rows.
    naive_time=$(echo "$output"    | awk '/^naive/    {print $(NF-2)}')
    prefetch_time=$(echo "$output" | awk '/^prefetch/ {print $(NF-2)}')
    speedup=$(echo "$output"       | awk '/^prefetch/ {print $NF}' | tr -d 'x')

    if [ -z "$naive_time" ] || [ -z "$prefetch_time" ] || [ -z "$speedup" ]; then
        echo "  [Warning] Could not parse output for size ${size}; skipping row." >&2
        echo "  --- raw output ---" >&2
        echo "$output" >&2
        continue
    fi

    echo "${size},${naive_time},${prefetch_time},${speedup}" >> "$output_csv"
done

echo "Done. Results saved to ${output_csv}"