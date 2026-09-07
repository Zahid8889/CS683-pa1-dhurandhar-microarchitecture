#!/usr/bin/env python3
import sys
import subprocess
import re

# Found via `perf list` on this specific AMD CPU: l2_cache_req_stat.ls_rd_blk_c is
# "Data cache request miss in L2 (all types)" - the AMD equivalent of Intel's
# l2_rqsts.miss. perf's own description notes a newer alias
# 'l2_cache_misses_from_dc_misses' may exist instead - run
#   perf list | grep -i l2_cache_misses
# to check, and swap L2_EVENT below if it's present on your system.

L2_EVENT = "l2_cache_req_stat.ls_rd_blk_c"
EVENTS = f"instructions,L1-dcache-load-misses,LLC-load-misses,{L2_EVENT}"


def extract_metric(output, event_name):
    """Returns (value, status): 'ok', 'unsupported' (<not supported>), or 'missing'
    (line absent entirely)."""
    pattern = rf'([\d,]+|<not supported>)\s+{re.escape(event_name)}'
    m = re.search(pattern, output)
    if not m:
        return 0, "missing"
    val = m.group(1)
    if val == "<not supported>":
        return 0, "unsupported"
    return int(val.replace(',', '')), "ok"


def benchmark(stage, dimension, runs=100, warned=set()):
    cmd = [
        "perf", "stat",
        "-e", EVENTS,
        "./bin/matmul",
        stage,
        str(dimension), str(dimension), str(dimension)
    ]

    total_instructions = 0
    total_l1_misses = 0
    total_l2_misses = 0
    total_llc_misses = 0

    print(f"Running perf stat for stage '{stage}' (Dim: {dimension}) {runs} times...")
    for i in range(1, runs + 1):
        print(f"\rRun {i}/{runs}", end="", flush=True)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout

        instructions, inst_status = extract_metric(output, "instructions")
        l1_misses, l1_status = extract_metric(output, "L1-dcache-load-misses")
        llc_misses, llc_status = extract_metric(output, "LLC-load-misses")
        l2_misses, l2_status = extract_metric(output, L2_EVENT)

        if inst_status != "ok":
            print(f"\n\n[Error] Could not get 'instructions' count on run {i}.")
            print("--- Captured Output From Terminal ---")
            print(output)
            print("-------------------------------------")
            sys.exit(1)

        for name, status in [("L1-dcache-load-misses", l1_status),
                              ("LLC-load-misses", llc_status),
                              (L2_EVENT, l2_status)]:
            if status != "ok" and name not in warned:
                warned.add(name)
                print(f"\n[Warning] '{name}' is {status} on this CPU's PMU — reporting as 0.")

        total_instructions += instructions
        total_l1_misses += l1_misses
        total_l2_misses += l2_misses
        total_llc_misses += llc_misses
    print("\n")

    avg_instructions = total_instructions / runs
    avg_l1_misses = total_l1_misses / runs
    avg_l2_misses = total_l2_misses / runs
    avg_llc_misses = total_llc_misses / runs

    kilo_instructions = total_instructions / 1000.0
    l1_mpki = (total_l1_misses / kilo_instructions) if kilo_instructions > 0 else 0.0
    l2_mpki = (total_l2_misses / kilo_instructions) if kilo_instructions > 0 else 0.0
    llc_mpki = (total_llc_misses / kilo_instructions) if kilo_instructions > 0 else 0.0

    print(f"--- Performance Summary over {runs} Runs --- Dim:{dimension}")
    print(f"Average Instructions     : {avg_instructions:,.0f}")
    print(f"Average L1-D Cache Misses: {avg_l1_misses:,.0f}")
    print(f"Average L2 Cache Misses  : {avg_l2_misses:,.0f}")
    print(f"Average LLC Misses       : {avg_llc_misses:,.0f}")
    print("------------------------------------------")
    print(f"L1-dcache MPKI           : {l1_mpki:.4f} misses/KI")
    print(f"L2 MPKI                  : {l2_mpki:.4f} misses/KI")
    print(f"LLC MPKI                 : {llc_mpki:.4f} misses/KI")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <stage>")
        print(f"Example: python3 {sys.argv[0]} prefetch")
        sys.exit(1)
    stage_arg = sys.argv[1]

    print(f"[Info] Using L2 event: {L2_EVENT}")
    warned_metrics = set()
    for dim in [128, 256, 512, 1024, 2048]:
        benchmark(stage_arg, dim, runs=5, warned=warned_metrics)