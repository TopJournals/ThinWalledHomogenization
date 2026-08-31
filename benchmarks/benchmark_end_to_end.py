"""Repeated solver-only and end-to-end CPU/GPU benchmark for LPS-H.

The benchmark keeps the core solver unchanged and instruments the same geometry,
assembly, PCG, stress-recovery, and ABD-extraction operations used by the project.
One unrecorded warm-up is followed by five recorded runs by default.
"""

from __future__ import annotations

import os

# Fix the CPU-side numerical libraries to one thread before importing NumPy/SciPy.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg as scipy_cg

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
except Exception as exc:  # pragma: no cover - depends on CUDA availability.
    raise RuntimeError("CuPy with a working CUDA backend is required.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plate_homogenizer import (  # noqa: E402
    build_tensor_dof_mapping,
    compute_element_stiffness,
    get_isotropic_elasticity,
)
from utils.tpms_generator import generate_tpms_voxel_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[20, 35, 50, 65])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--topology", default="Primitive")
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--thickness", type=float, default=10.0)
    parser.add_argument("--youngs-modulus", type=float, default=1215.0)
    parser.add_argument("--poisson", type=float, default=0.35)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "r2_m5",
    )
    return parser.parse_args()


def clock() -> float:
    return time.perf_counter()


def cuda_version(value: int) -> str:
    return f"{value // 1000}.{(value % 1000) // 10}"


def gpu_name() -> str:
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def build_system(args: argparse.Namespace, resolution: int) -> dict:
    start = clock()
    voxel = generate_tpms_voxel_grid(
        tpms_type=args.topology,
        Nx=1,
        Ny=1,
        Nz=1,
        resolution=resolution,
        relative_density=args.density,
        is_sheet=True,
    )
    geometry_time = clock() - start

    start = clock()
    nx, ny, nz = voxel.shape
    dx = dy = dz = args.thickness / resolution
    constitutive = get_isotropic_elasticity(args.youngs_modulus, args.poisson)
    element_stiffness, b_matrices, det_j = compute_element_stiffness(
        constitutive, dx, dy, dz
    )
    edof_mat, z_active, total_dofs = build_tensor_dof_mapping(
        voxel, dx, dy, dz, args.thickness
    )

    ik = np.repeat(edof_mat, 24, axis=1).ravel()
    jk = np.tile(edof_mat, (1, 24)).ravel()
    sk = np.tile(element_stiffness.ravel(), edof_mat.shape[0])
    stiffness = coo_matrix(
        (sk, (ik, jk)), shape=(total_dofs, total_dofs)
    ).tocsr()

    e_macro = np.zeros((len(z_active), 6, 6))
    e_macro[:, 0, 0], e_macro[:, 1, 1], e_macro[:, 5, 2] = 1.0, 1.0, 1.0
    e_macro[:, 0, 3], e_macro[:, 1, 4], e_macro[:, 5, 5] = (
        z_active,
        z_active,
        z_active,
    )
    f_ele = sum(
        np.einsum(
            "ji,kjl->kil",
            b_matrices[point],
            np.einsum("ij,kjl->kil", constitutive, e_macro),
        )
        * det_j
        for point in range(8)
    )
    loads = np.column_stack(
        [
            np.bincount(
                edof_mat.ravel(),
                weights=f_ele[:, :, case].ravel(),
                minlength=total_dofs,
            )
            for case in range(6)
        ]
    )
    active_dofs = np.setdiff1d(np.unique(edof_mat), np.unique(edof_mat)[:3])
    assembly_time = clock() - start
    return {
        "resolution": resolution,
        "plate_area": args.thickness**2,
        "active_elements": int(edof_mat.shape[0]),
        "active_dofs": int(len(active_dofs)),
        "total_dofs": int(total_dofs),
        "geometry_time_s": geometry_time,
        "assembly_time_s": assembly_time,
        "stiffness": stiffness,
        "loads": loads,
        "active": active_dofs,
        "constitutive": constitutive,
        "b_matrices": b_matrices,
        "det_j": det_j,
        "edof_mat": edof_mat,
        "z_active": z_active,
        "e_macro": e_macro,
    }


def recover_abd(state: dict, displacement: np.ndarray) -> tuple[np.ndarray, float, float]:
    start = clock()
    stresses = [
        np.einsum(
            "ij,kjl->kil",
            state["constitutive"],
            state["e_macro"]
            - np.einsum(
                "ij,kjl->kil",
                state["b_matrices"][point],
                displacement[state["edof_mat"], :],
            ),
        )
        for point in range(8)
    ]
    recovery_time = clock() - start

    start = clock()
    abd = np.zeros((6, 6))
    for stress in stresses:
        in_plane = stress[:, [0, 1, 5], :]
        abd[0:3, :] += (
            np.sum(in_plane, axis=0) * state["det_j"] / state["plate_area"]
        )
        abd[3:6, :] += (
            np.sum(in_plane * state["z_active"][:, None, None], axis=0)
            * state["det_j"]
            / state["plate_area"]
        )
    abd_time = clock() - start
    return (abd + abd.T) / 2.0, recovery_time, abd_time


def run_cpu(args: argparse.Namespace, state: dict) -> tuple[dict, np.ndarray]:
    start = clock()
    matrix = state["stiffness"][state["active"], :][:, state["active"]]
    rhs = state["loads"][state["active"], :]
    preconditioner = diags(1.0 / matrix.diagonal())
    setup_time = clock() - start

    start = clock()
    active_solution = np.zeros((len(state["active"]), 6))
    for case in range(6):
        solution, info = scipy_cg(
            matrix,
            rhs[:, case],
            M=preconditioner,
            rtol=args.rtol,
            atol=0.0,
            maxiter=args.maxiter,
        )
        if info != 0:
            raise RuntimeError(f"CPU CG failed for load case {case}: info={info}")
        active_solution[:, case] = solution
    solve_time = clock() - start

    displacement = np.zeros((state["total_dofs"], 6))
    displacement[state["active"], :] = active_solution
    abd, recovery_time, abd_time = recover_abd(state, displacement)
    end_to_end = (
        state["geometry_time_s"]
        + state["assembly_time_s"]
        + setup_time
        + solve_time
        + recovery_time
        + abd_time
    )
    return {
        "cpu_setup_time_s": setup_time,
        "cpu_solve_time_s": solve_time,
        "cpu_stress_recovery_time_s": recovery_time,
        "cpu_abd_extraction_time_s": abd_time,
        "cpu_end_to_end_time_s": end_to_end,
    }, abd


def run_gpu(args: argparse.Namespace, state: dict) -> tuple[dict, np.ndarray]:
    start = clock()
    matrix_host = state["stiffness"][state["active"], :][:, state["active"]]
    rhs_host = state["loads"][state["active"], :]
    matrix = cpsp.csr_matrix(matrix_host)
    rhs = cp.asarray(rhs_host)
    cp.cuda.Stream.null.synchronize()
    transfer_time = clock() - start

    start = clock()
    active_solution = cp.zeros((len(state["active"]), 6))
    preconditioner = cpsp.diags(1.0 / matrix.diagonal())
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(6)]
    cp.cuda.Stream.null.synchronize()
    initialization_time = clock() - start

    start = clock()
    for case in range(6):
        with streams[case]:
            solution, info = cpspla.cg(
                matrix,
                rhs[:, case],
                M=preconditioner,
                tol=args.rtol,
                maxiter=args.maxiter,
            )
            if info != 0:
                raise RuntimeError(f"GPU CG failed for load case {case}: info={info}")
            active_solution[:, case] = solution
    launch_time = clock() - start

    start = clock()
    for stream in streams:
        stream.synchronize()
    synchronization_time = clock() - start
    solve_time = launch_time + synchronization_time

    start = clock()
    active_solution_host = active_solution.get()
    cp.cuda.Stream.null.synchronize()
    return_time = clock() - start

    displacement = np.zeros((state["total_dofs"], 6))
    displacement[state["active"], :] = active_solution_host
    abd, recovery_time, abd_time = recover_abd(state, displacement)
    end_to_end = (
        state["geometry_time_s"]
        + state["assembly_time_s"]
        + transfer_time
        + initialization_time
        + solve_time
        + return_time
        + recovery_time
        + abd_time
    )
    return {
        "gpu_transfer_time_s": transfer_time,
        "gpu_initialization_time_s": initialization_time,
        "gpu_solve_launch_time_s": launch_time,
        "gpu_synchronization_time_s": synchronization_time,
        "gpu_solve_time_s": solve_time,
        "gpu_return_time_s": return_time,
        "gpu_stress_recovery_time_s": recovery_time,
        "gpu_abd_extraction_time_s": abd_time,
        "gpu_end_to_end_time_s": end_to_end,
    }, abd


def run_once(args: argparse.Namespace, resolution: int, repeat: int) -> dict:
    state = build_system(args, resolution)
    cpu, abd_cpu = run_cpu(args, state)
    gpu, abd_gpu = run_gpu(args, state)
    relative_difference = float(
        np.linalg.norm(abd_cpu - abd_gpu) / np.linalg.norm(abd_cpu)
    )
    if relative_difference > 5e-4:
        raise RuntimeError(f"CPU/GPU ABD mismatch: {relative_difference:.3e}")
    return {
        "resolution": resolution,
        "repeat": repeat,
        "active_elements": state["active_elements"],
        "active_dofs": state["active_dofs"],
        "total_dofs": state["total_dofs"],
        "geometry_time_s": state["geometry_time_s"],
        "assembly_time_s": state["assembly_time_s"],
        **cpu,
        **gpu,
        "solver_speedup": cpu["cpu_solve_time_s"] / gpu["gpu_solve_time_s"],
        "end_to_end_speedup": cpu["cpu_end_to_end_time_s"]
        / gpu["gpu_end_to_end_time_s"],
        "abd_relative_difference": relative_difference,
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for resolution in sorted({row["resolution"] for row in rows}):
        group = [row for row in rows if row["resolution"] == resolution]
        summary = {
            "resolution": resolution,
            "repeats": len(group),
            "active_elements": group[0]["active_elements"],
            "active_dofs": group[0]["active_dofs"],
            "total_dofs": group[0]["total_dofs"],
        }
        for key, value in group[0].items():
            if key in summary or key == "repeat":
                continue
            if isinstance(value, (int, float)):
                values = np.asarray([row[key] for row in group], dtype=float)
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(
                    np.std(values, ddof=1 if len(values) > 1 else 0)
                )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cp.cuda.Device().use()
    cp.get_default_memory_pool().free_all_blocks()

    rows = []
    for resolution in args.resolutions:
        for warmup in range(args.warmups):
            print(f"[warm-up] N={resolution}, run={warmup + 1}", flush=True)
            run_once(args, resolution, repeat=0)
        for repeat in range(1, args.repeats + 1):
            print(f"[recorded] N={resolution}, run={repeat}/{args.repeats}", flush=True)
            row = run_once(args, resolution, repeat)
            rows.append(row)
            print(
                f"  solve speedup={row['solver_speedup']:.3f}, "
                f"end-to-end speedup={row['end_to_end_speedup']:.3f}",
                flush=True,
            )

    summaries = summarize(rows)
    write_csv(args.output_dir / "benchmark_end_to_end_runs.csv", rows)
    write_csv(args.output_dir / "benchmark_end_to_end_summary.csv", summaries)

    metadata = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "gpu": gpu_name(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": cuda_version(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": cuda_version(cp.cuda.runtime.driverGetVersion()),
            "cpu_threads": {
                variable: os.environ[variable]
                for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "protocol": {
            "warmups": args.warmups,
            "recorded_repeats": args.repeats,
            "preconditioner": "Jacobi on both CPU and GPU",
            "rtol": args.rtol,
            "maxiter": args.maxiter,
            "gpu_solver_time": "CG launch plus explicit stream synchronization",
            "end_to_end_time": "geometry, assembly, backend setup/transfer, solve, synchronization, return, stress recovery, and ABD extraction",
        },
        "summary": summaries,
    }
    (args.output_dir / "benchmark_end_to_end_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[done] outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
