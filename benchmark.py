#!/usr/bin/env python3
"""
MPCD collision-strategy benchmark.
Compares trivial / sorting / optimized kernel implementations
for both SRD and extended MPCD algorithms.

Usage:
    python benchmark_strategies.py
"""

import ctypes
import timeit

import pympcd

# ── timing parameters ──────────────────────────────────────────────────────────
STEPS_PER_CALL = 10  # steps per timed call — keep large enough to amortise Python overhead
N_LOOPS = 20  # -n
N_REPEATS = 2  # -r  (best-of is reported)
WARMUP_CALLS = 3  # calls before timing starts

# ── system sizes ───────────────────────────────────────────────────────────────
SIZES = [
    (1000, 1000, 20),
]

ALGORITHMS = ["srd", "extended"]
KERNELS = ["trivial", "sorting", "optimized"]

ALGO_PARAMS = {
    "srd": {"n": 10, "delta_t": 0.02},
    "extended": {"n": 20, "delta_t": 0.005},
}


def make_sim(algorithm: str, kernel: str, size: tuple) -> pympcd.Simulation:
    params = pympcd.Params()
    params.n = ALGO_PARAMS[algorithm]["n"]
    params.temperature = 1.0
    params.volume_size = size
    params.periodicity = (1, 1, 0)
    params.drag = 0.0
    params.delta_t = ALGO_PARAMS[algorithm]["delta_t"]
    params.experiment = "standard"
    params.algorithm = algorithm
    params.collision_kernel = kernel
    return pympcd.Simulation(params, "cuda")


libcuda = ctypes.CDLL("libcudart.so")


def run_bench(sim: pympcd.Simulation) -> float:
    """Returns best ms-per-step over N_REPEATS runs of N_LOOPS calls."""
    for _ in range(WARMUP_CALLS):
        sim.step(STEPS_PER_CALL)

    libcuda.cudaDeviceSynchronize()
    raw = timeit.repeat(
        lambda: (sim.step(STEPS_PER_CALL), libcuda.cudaDeviceSynchronize()),
        number=N_LOOPS,
        repeat=N_REPEATS,
    )

    # mean across repeats, divided by total steps — gives ms per simulation step
    return sum(raw) / (N_REPEATS * N_LOOPS * STEPS_PER_CALL) * 1e3


# ── run ────────────────────────────────────────────────────────────────────────
for size in SIZES:
    print(f"\n{'═' * 66}")
    print(
        f"  {size[0]}×{size[1]}×{size[2]}   "
        f"srd n=10 ({size[0] * size[1] * size[2] * 10:,} p)  "
        f"ext n=20 ({size[0] * size[1] * size[2] * 20:,} p)"
    )
    print(f"  timing: best of {N_REPEATS}×{N_LOOPS} calls, {STEPS_PER_CALL} steps/call")
    print(f"{'═' * 66}")
    print(f"  {'algorithm':10s}  {'kernel':10s}  {'ms/step':>9s}  {'speedup vs trivial':>18s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 9}  {'-' * 18}")

    for algorithm in ALGORITHMS:
        baseline_ms = None
        for kernel in KERNELS:
            label = f"  {algorithm:10s}  {kernel:10s}"
            print(f"{label}  ", end="", flush=True)

            sim = make_sim(algorithm, kernel, size)
            ms = run_bench(sim)

            if kernel == "trivial":
                baseline_ms = ms

            speedup = f"{baseline_ms / ms:.2f}×" if baseline_ms else "—"
            print(f"{ms:9.3f}  {speedup:>18s}")
        print()
