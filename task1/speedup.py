#!/usr/bin/env python3
"""
avg_speedup.py

Runs a benchmark binary (e.g. ./bin/conv simd 512 512 3) N times, parses its
stdout table, and reports the average (and stddev) of the 'speedup' column
for a chosen stage (default: 'simd').

Why average over multiple runs at all: a single run's timing is a noisy
sample -- OS scheduling jitter, thermal throttling, cache state left over
from other processes, and frequency scaling (turbo boost ramp-up) all add
variance. Reporting mean +/- stddev over N=20 runs gives you a much more
defensible number for your report than a single perf snapshot, and the
stddev itself is worth reporting -- large variance is itself a finding
(e.g. it can indicate the CPU is thermal-throttling mid-sweep, or another
process is contending for the last-level cache).

Usage:
    IMPORTANT: --cmd captures EVERYTHING after it (including further flags),
    so put --stage/--runs/--warmup BEFORE --cmd, not after:

        python3 avg_speedup.py --stage simd --runs 20 --cmd ./bin/conv simd 512 512 3

    If you'd rather not worry about ordering, use --cmd-str with a quoted
    string instead -- flags can go in any order then:

        python3 avg_speedup.py --cmd-str "./bin/conv simd 512 512 3" --stage simd --runs 20
"""

import argparse
import re
import shlex
import statistics
import subprocess
import sys

# Matches a table row like:
# simd            yes           0.527        8.95      4.92x
# Captures: stage name, correct (yes/no), time(ms), GFLOP/s, speedup (as float, 'x' stripped)
ROW_RE = re.compile(
    r"^\s*(?P<stage>\S+)\s+(?P<correct>yes|no)\s+"
    r"(?P<time_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<gflops>[-+]?\d*\.?\d+)\s+"
    r"(?P<speedup>[-+]?\d*\.?\d+)x\s*$",
    re.IGNORECASE,
)


def parse_stage_metrics(stdout_text: str, stage: str):
    """
    Scan every line of a run's stdout for a row matching the target stage
    name (case-insensitive, exact token match e.g. 'simd' won't accidentally
    match 'simd128' unless you pass that exact stage name).
    Returns a dict with time_ms, gflops, speedup, correct -- or None if not found.
    """
    for line in stdout_text.splitlines():
        m = ROW_RE.match(line)
        if m and m.group("stage").lower() == stage.lower():
            return {
                "correct": m.group("correct").lower() == "yes",
                "time_ms": float(m.group("time_ms")),
                "gflops": float(m.group("gflops")),
                "speedup": float(m.group("speedup")),
            }
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--cmd", nargs=argparse.REMAINDER,
                        help="Command and args as separate tokens, e.g. --cmd ./bin/conv simd 512 512 3")
    group.add_argument("--cmd-str",
                        help="Command as a single shell-style string, e.g. --cmd-str \"./bin/conv simd 512 512 3\"")
    ap.add_argument("--stage", default="simd",
                     help="Which stage row to extract speedup from (default: simd). "
                          "Must match the first column exactly, e.g. 'naive', 'simd', 'tiled'.")
    ap.add_argument("--runs", type=int, default=20, help="Number of repetitions (default: 20)")
    ap.add_argument("--warmup", type=int, default=0,
                     help="Number of untimed warmup runs before the measured runs (default: 0). "
                          "Useful to let CPU frequency scaling settle before measuring.")
    ap.add_argument("--fail-fast", action="store_true",
                     help="Abort immediately if any run's correctness check fails ('correct' != yes), "
                          "instead of just warning and excluding it from the average.")
    args = ap.parse_args()

    if args.cmd_str is not None:
        cmd = shlex.split(args.cmd_str)
    else:
        cmd = args.cmd
    if not cmd:
        ap.error("No command provided.")

    print(f"Command:   {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"Stage:     {args.stage}")
    print(f"Runs:      {args.runs}  (+{args.warmup} warmup)")
    print("-" * 70)

    for w in range(args.warmup):
        subprocess.run(cmd, capture_output=True, text=True)
        print(f"  warmup {w + 1}/{args.warmup} done")

    speedups = []
    times_ms = []
    gflops_list = []
    failures = 0

    for i in range(1, args.runs + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  run {i:2d}/{args.runs}: TIMED OUT -- skipped")
            continue
        except FileNotFoundError:
            print(f"ERROR: could not find/execute '{cmd[0]}'. Check the path and that it's executable "
                  f"(chmod +x) and that you're running from the right working directory.")
            sys.exit(1)

        if result.returncode != 0:
            print(f"  run {i:2d}/{args.runs}: process exited with code {result.returncode}, stderr:\n"
                  f"    {result.stderr.strip()[:300]}")
            failures += 1
            continue

        metrics = parse_stage_metrics(result.stdout, args.stage)
        if metrics is None:
            print(f"  run {i:2d}/{args.runs}: could not find a row for stage '{args.stage}' in output. "
                  f"Raw stdout was:\n{result.stdout}")
            failures += 1
            continue

        if not metrics["correct"]:
            msg = f"  run {i:2d}/{args.runs}: stage '{args.stage}' reported correct=no (speedup={metrics['speedup']:.2f}x)"
            if args.fail_fast:
                print(msg + " -- aborting due to --fail-fast")
                sys.exit(1)
            print(msg + " -- EXCLUDED from average")
            failures += 1
            continue

        speedups.append(metrics["speedup"])
        times_ms.append(metrics["time_ms"])
        gflops_list.append(metrics["gflops"])
        # print(f"  run {i:2d}/{args.runs}: time={metrics['time_ms']:.3f} ms  "
            #   f"GFLOP/s={metrics['gflops']:.2f}  speedup={metrics['speedup']:.2f}x")

    print("-" * 70)
    if not speedups:
        print("No successful runs to average. Nothing to report.")
        sys.exit(1)

    n = len(speedups)
    mean_speedup = statistics.mean(speedups)
    mean_time = statistics.mean(times_ms)
    mean_gflops = statistics.mean(gflops_list)

    print(f"Successful runs used: {n}/{args.runs}  (excluded: {failures})")
    print(f"Mean time(ms):        {mean_time:.4f}")
    print(f"Mean GFLOP/s:         {mean_gflops:.4f}")
    print(f"Mean speedup:         {mean_speedup:.4f}x")

    if n > 1:
        stdev_speedup = statistics.stdev(speedups)
        cv = (stdev_speedup / mean_speedup) * 100 if mean_speedup else float("nan")
        print(f"Stddev speedup:       {stdev_speedup:.4f}x")
        print(f"Coeff. of variation:  {cv:.2f}%  "
              f"({'looks stable' if cv < 5 else 'noisy run-to-run -- consider more warmup, '
                                                  'pinning CPU freq, or checking for background load'})")
        print(f"Min / Max speedup:    {min(speedups):.4f}x / {max(speedups):.4f}x")

    print("-" * 70)
    print(f"REPORT LINE: {args.stage} stage, N={n} runs -> "
          f"avg speedup = {mean_speedup:.2f}x "
          f"(avg time = {mean_time:.3f} ms, avg GFLOP/s = {mean_gflops:.2f})")


if __name__ == "__main__":
    main()