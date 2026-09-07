#!/usr/bin/env python3
import sys
import subprocess
import re

def benchmark(stage, x=1024, y=1024, k=3, runs=100):
    cmd = ["sudo", "perf", "stat", "-d", "./bin/conv", stage, str(x), str(y), str(k)]
    total_instructions = 0
    total_misses = 0
    print(f"Running perf stat for '{stage}' {runs} times...")

    for i in range(1, runs + 1):
        print(f"\rRun {i}/{runs}", end="", flush=True)

        # Run perf stat and capture stderr/stdout combined
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout

        # Extract instructions and L1-dcache-load-misses using regex
        inst_match = re.search(r'([\d,]+)\s+instructions', output)
        misses_match = re.search(r'([\d,]+)\s+L1-dcache-load-misses', output)

        if not inst_match or not misses_match:
            print(f"\nError: Could not parse required counters on run {i}.")
            sys.exit(1)

        instructions = int(inst_match.group(1).replace(',', ''))
        misses = int(misses_match.group(1).replace(',', ''))

        total_instructions += instructions
        total_misses += misses

    print("\n")
    # Calculate statistical metrics across runs
    avg_instructions = total_instructions / runs
    avg_misses = total_misses / runs
    # MPKI calculation: (Total Misses / Total Instructions) * 1000
    mpki = (total_misses / (total_instructions / 1000)) if total_instructions > 0 else 0.0

    print(f"--- Performance Summary over 100 Runs --- X:{x}, Y: {y}, K:{k}")
    print(f"Average Instructions     : {avg_instructions:,.0f}")
    print(f"Average L1-dcache Misses : {avg_misses:,.0f}")
    print(f"L1-dcache MPKI           : {mpki:.4f} misses/KI")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <stage>")
        sys.exit(1)

    stage_arg = sys.argv[1]
    for u, v, w in [(128, 128, 3), (256, 256, 3),
                     (512, 512, 3), (1024, 1024, 3), (2048, 2048, 3)]:
        benchmark(stage_arg, x=u, y=v, k=w, runs=20)