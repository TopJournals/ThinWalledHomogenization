"""Benchmark CPU/GPU costs for the LPS-H plate homogenization workflow.

This script reuses the same geometry generation and finite-element assembly
building blocks used by ``core.plate_homogenizer``. It does not run by default
during examples because the larger resolutions can be expensive.

Example
-------
python benchmarks/benchmark_plate_solver.py --resolutions 32 40 48 56 64 --output benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg as scipy_cg

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
except Exception:  # pragma: no cover - exercised only in CPU-only setups.
    cp = None
    cpsp = None
    cpspla = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plate_homogenizer import (  # noqa: E402
    build_tensor_dof_mapping,
    compute_element_stiffness,
    get_isotropic_elasticity,
)
from utils.tpms_generator import generate_tpms_voxel_grid  # noqa: E402


@dataclass
class BenchmarkResult:
    resolution: int
    topology: str
    nx: int
    ny: int
    nz: int
    relative_density: float
    active_elements: int
    total_dofs: int
    active_dofs: int
    nnz: int
    geometry_time_s: float
    assembly_time_s: float
    cpu_solve_time_s: float | None
    gpu_solve_time_s: float | None
    cpu_iterations_max: int | None
    gpu_iterations_max: int | None
    cpu_matrix_memory_mb: float
    gpu_memory_delta_mb: float | None
    gpu_memory_pool_mb: float | None
    gpu_name: str
    speedup_cpu_over_gpu: float | None
    solver_rtol: float
    solver_maxiter: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU/GPU solve time and memory for LPS-H ABD extraction.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 40, 48, 56, 64],
        help="Voxel resolutions per unit cell to test.",
    )
    parser.add_argument("--topology", default="Primitive", help="TPMS topology name.")
    parser.add_argument("--density", type=float, default=0.15, help="Target relative density.")
    parser.add_argument("--thickness", type=float, default=10.0, help="Plate thickness in mm.")
    parser.add_argument("--cells", type=int, nargs=3, default=[1, 1, 1], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--youngs-modulus", type=float, default=1215.0, help="Base Young's modulus.")
    parser.add_argument("--poisson", type=float, default=0.35, help="Base Poisson's ratio.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="CG relative tolerance.")
    parser.add_argument("--maxiter", type=int, default=5000, help="Maximum CG iterations per load case.")
    parser.add_argument(
        "--skip-cpu-above",
        type=int,
        default=None,
        help="Skip CPU solves above this resolution while still reporting GPU results.",
    )
    parser.add_argument(
        "--no-cpu",
        action="store_true",
        help="Skip CPU solves and report GPU-only timings.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU solves and report CPU-only timings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "benchmark_plate_solver_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def monotonic() -> float:
    return time.perf_counter()


def matrix_memory_mb(matrix) -> float:
    bytes_used = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return bytes_used / 1024**2


def gpu_memory_used_bytes() -> int | None:
    if cp is None:
        return None
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return int(total_bytes - free_bytes)


def gpu_pool_bytes() -> int | None:
    if cp is None:
        return None
    return int(cp.get_default_memory_pool().total_bytes())


def gpu_name() -> str:
    if cp is None:
        return "not available"
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return str(name)


def assemble_lpsh_system(
    voxel: np.ndarray,
    youngs_modulus: float,
    poisson: float,
    thickness: float,
    cells: tuple[int, int, int],
):
    nx, ny, nz = voxel.shape
    nx_cells, ny_cells, nz_cells = cells
    cell_size = thickness / nz_cells
    lx = nx_cells * cell_size
    ly = ny_cells * cell_size
    lz = thickness
    dx, dy, dz = lx / nx, ly / ny, lz / nz

    constitutive = get_isotropic_elasticity(youngs_modulus, poisson)
    element_stiffness, b_matrices, det_j = compute_element_stiffness(constitutive, dx, dy, dz)
    edof_mat, z_active, total_dofs = build_tensor_dof_mapping(voxel, dx, dy, dz, thickness)

    ik = np.repeat(edof_mat, 24, axis=1).ravel()
    jk = np.tile(edof_mat, (1, 24)).ravel()
    sk = np.tile(element_stiffness.ravel(), edof_mat.shape[0])
    stiffness = coo_matrix((sk, (ik, jk)), shape=(total_dofs, total_dofs)).tocsr()

    e_macro = np.zeros((len(z_active), 6, 6))
    e_macro[:, 0, 0], e_macro[:, 1, 1], e_macro[:, 5, 2] = 1.0, 1.0, 1.0
    e_macro[:, 0, 3], e_macro[:, 1, 4], e_macro[:, 5, 5] = z_active, z_active, z_active

    f_ele = sum(
        np.einsum(
            "ji,kjl->kil",
            b_matrices[i],
            np.einsum("ij,kjl->kil", constitutive, e_macro),
        )
        * det_j
        for i in range(8)
    )
    loads = np.column_stack(
        [
            np.bincount(edof_mat.ravel(), weights=f_ele[:, :, case].ravel(), minlength=total_dofs)
            for case in range(6)
        ]
    )
    active_dofs = np.setdiff1d(np.unique(edof_mat), np.unique(edof_mat)[:3])
    return stiffness, loads, active_dofs, edof_mat.shape[0], total_dofs


def solve_cpu(stiffness, loads, active_dofs, rtol: float, maxiter: int):
    matrix = stiffness[active_dofs, :][:, active_dofs]
    rhs = loads[active_dofs, :]
    iteration_counts: list[int] = []
    start = monotonic()

    for case in range(rhs.shape[1]):
        count = 0

        def callback(_):
            nonlocal count
            count += 1

        try:
            _, info = scipy_cg(matrix, rhs[:, case], rtol=rtol, atol=0.0, maxiter=maxiter, callback=callback)
        except TypeError:
            _, info = scipy_cg(matrix, rhs[:, case], tol=rtol, maxiter=maxiter, callback=callback)
        if info != 0:
            raise RuntimeError(f"CPU CG did not converge for load case {case}; scipy info={info}")
        iteration_counts.append(count)

    return monotonic() - start, max(iteration_counts) if iteration_counts else 0


def solve_gpu(stiffness, loads, active_dofs, rtol: float, maxiter: int):
    if cp is None or cpsp is None or cpspla is None:
        raise RuntimeError("CuPy is not available; install cupy-cuda12x or run with --no-gpu.")

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    before_used = gpu_memory_used_bytes()

    matrix_gpu = cpsp.csr_matrix(stiffness[active_dofs, :][:, active_dofs])
    rhs_gpu = cp.asarray(loads[active_dofs, :])
    solution_gpu = cp.zeros((len(active_dofs), 6))
    preconditioner = cpsp.diags(1.0 / matrix_gpu.diagonal())
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(6)]
    iteration_counts = [0 for _ in range(6)]

    cp.cuda.Stream.null.synchronize()
    start = monotonic()
    for case in range(6):
        with streams[case]:

            def callback(_, case_index=case):
                iteration_counts[case_index] += 1

            solution, info = cpspla.cg(
                matrix_gpu,
                rhs_gpu[:, case],
                M=preconditioner,
                tol=rtol,
                maxiter=maxiter,
                callback=callback,
            )
            if info != 0:
                raise RuntimeError(f"GPU CG did not converge for load case {case}; cupy info={info}")
            solution_gpu[:, case] = solution

    for stream in streams:
        stream.synchronize()
    elapsed = monotonic() - start

    after_used = gpu_memory_used_bytes()
    pool_used = gpu_pool_bytes()
    memory_delta_mb = None if before_used is None or after_used is None else (after_used - before_used) / 1024**2
    pool_mb = None if pool_used is None else pool_used / 1024**2
    return elapsed, max(iteration_counts) if iteration_counts else 0, memory_delta_mb, pool_mb


def run_single_case(args: argparse.Namespace, resolution: int) -> BenchmarkResult:
    nx_cells, ny_cells, nz_cells = tuple(args.cells)
    note_parts: list[str] = []

    t0 = monotonic()
    voxel = generate_tpms_voxel_grid(
        tpms_type=args.topology,
        Nx=nx_cells,
        Ny=ny_cells,
        Nz=nz_cells,
        resolution=resolution,
        relative_density=args.density,
        is_sheet=True,
    )
    geometry_time = monotonic() - t0

    t1 = monotonic()
    stiffness, loads, active_dofs, active_elements, total_dofs = assemble_lpsh_system(
        voxel=voxel,
        youngs_modulus=args.youngs_modulus,
        poisson=args.poisson,
        thickness=args.thickness,
        cells=(nx_cells, ny_cells, nz_cells),
    )
    assembly_time = monotonic() - t1

    cpu_time = None
    cpu_iterations = None
    gpu_time = None
    gpu_iterations = None
    gpu_memory_delta_mb = None
    gpu_memory_pool_mb = None

    skip_cpu = args.no_cpu or (args.skip_cpu_above is not None and resolution > args.skip_cpu_above)
    if skip_cpu:
        note_parts.append("CPU solve skipped")
    else:
        cpu_time, cpu_iterations = solve_cpu(stiffness, loads, active_dofs, args.rtol, args.maxiter)

    if args.no_gpu:
        note_parts.append("GPU solve skipped")
    else:
        gpu_time, gpu_iterations, gpu_memory_delta_mb, gpu_memory_pool_mb = solve_gpu(
            stiffness,
            loads,
            active_dofs,
            args.rtol,
            args.maxiter,
        )

    speedup = None
    if cpu_time is not None and gpu_time is not None and gpu_time > 0:
        speedup = cpu_time / gpu_time

    return BenchmarkResult(
        resolution=resolution,
        topology=args.topology,
        nx=nx_cells,
        ny=ny_cells,
        nz=nz_cells,
        relative_density=args.density,
        active_elements=active_elements,
        total_dofs=total_dofs,
        active_dofs=len(active_dofs),
        nnz=stiffness.nnz,
        geometry_time_s=geometry_time,
        assembly_time_s=assembly_time,
        cpu_solve_time_s=cpu_time,
        gpu_solve_time_s=gpu_time,
        cpu_iterations_max=cpu_iterations,
        gpu_iterations_max=gpu_iterations,
        cpu_matrix_memory_mb=matrix_memory_mb(stiffness),
        gpu_memory_delta_mb=gpu_memory_delta_mb,
        gpu_memory_pool_mb=gpu_memory_pool_mb,
        gpu_name=gpu_name(),
        speedup_cpu_over_gpu=speedup,
        solver_rtol=args.rtol,
        solver_maxiter=args.maxiter,
        note="; ".join(note_parts),
    )


def write_csv(results: Iterable[BenchmarkResult], output: Path) -> None:
    rows = [asdict(result) for result in results]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(result: BenchmarkResult, output: Path) -> None:
    row = asdict(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(results: Iterable[BenchmarkResult], output: Path) -> None:
    payload = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": gpu_name(),
        },
        "results": [asdict(result) for result in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results: list[BenchmarkResult] = []
    if args.output.exists():
        args.output.unlink()

    for resolution in args.resolutions:
        print(f"[benchmark] resolution={resolution}", flush=True)
        result = run_single_case(args, resolution)
        results.append(result)
        append_csv(result, args.output)
        print(json.dumps(asdict(result), indent=2), flush=True)

    print(f"[benchmark] wrote CSV: {args.output}")
    if args.json_output is not None:
        write_json(results, args.json_output)
        print(f"[benchmark] wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
