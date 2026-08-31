# Solver benchmark scripts

This folder contains optional scripts for collecting computational-cost data for
the LPS-H plate homogenization workflow. The scripts reuse the same geometry
generation and finite-element assembly building blocks as the main examples.

## CPU/GPU LPS-H benchmark

`benchmark_plate_solver.py` compares CPU and GPU conjugate-gradient solves for
the same LPS-H system under increasing voxel resolutions. It records geometry
generation time, assembly time, solve time, active elements, degrees of freedom,
sparse matrix memory, GPU memory use, and CPU/GPU speedup.

Example:

```bash
python benchmarks/benchmark_plate_solver.py ^
  --resolutions 32 40 48 56 64 ^
  --topology Primitive ^
  --density 0.15 ^
  --cells 1 1 1 ^
  --output benchmarks/benchmark_plate_solver_results.csv ^
  --json-output benchmarks/benchmark_plate_solver_results.json
```

For large cases where the CPU solve is impractical, keep the GPU solve and skip
CPU runs above a selected resolution:

```bash
python benchmarks/benchmark_plate_solver.py ^
  --resolutions 40 50 60 70 80 96 ^
  --skip-cpu-above 70
```

The benchmark intentionally does not run as part of the normal examples because
high-resolution cases can be expensive.

## Repeated end-to-end benchmark

`benchmark_end_to_end.py` leaves the core solver unchanged and records both
solver-only and complete workflow costs. The default protocol uses one
unrecorded warm-up and five recorded runs at `N = 20, 35, 50, 65`, with one CPU
thread and Jacobi-preconditioned CG on both backends:

```bash
python benchmarks/benchmark_end_to_end.py
```

The `benchmarks/r2_m5/` output contains per-run timings, mean and standard
deviation summaries, and environment metadata. End-to-end timing includes
geometry generation, assembly, backend setup and transfer, initialization,
solution and synchronization, return transfer, stress recovery, and ABD
extraction.
