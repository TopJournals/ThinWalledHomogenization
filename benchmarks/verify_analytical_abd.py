"""Verify the plate solvers against analytical ABD matrices at 64^3 voxels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cupy as cp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "benchmarks" / "r2_m2" / "analytical_abd_benchmarks.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plate_homogenizer import homogenization_plate as homogenize_single  # noqa: E402
from core.plate_multimaterial_homogenizer import homogenization_plate as homogenize_bilayer  # noqa: E402


def reduced_isotropic_stiffness(E: float, nu: float) -> np.ndarray:
    factor = E / (1.0 - nu**2)
    return np.array(
        [[factor, nu * factor, 0.0], [nu * factor, factor, 0.0], [0.0, 0.0, E / (2.0 * (1.0 + nu))]]
    )


def analytical_abd(layers: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Return ABD for (z_bottom, z_top, E, nu) layers using geometric z=0."""
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for z_bottom, z_top, E, nu in layers:
        Q = reduced_isotropic_stiffness(E, nu)
        A += Q * (z_top - z_bottom)
        B += 0.5 * Q * (z_top**2 - z_bottom**2)
        D += (1.0 / 3.0) * Q * (z_top**3 - z_bottom**3)
    return np.block([[A, B], [B, D]])


def error_summary(numerical: np.ndarray, analytical: np.ndarray) -> dict[str, float | None]:
    scale = float(np.max(np.abs(analytical)))
    nonzero = np.abs(analytical) > scale * 1.0e-12
    zero = ~nonzero
    relative = np.abs(numerical[nonzero] - analytical[nonzero]) / np.abs(analytical[nonzero])
    zero_residual = np.abs(numerical[zero] - analytical[zero]) / scale
    return {
        "max_nonzero_relative_error_percent": float(np.max(relative) * 100.0),
        "max_zero_normalized_residual": float(np.max(zero_residual)) if zero_residual.size else None,
        "frobenius_relative_error_percent": float(
            np.linalg.norm(numerical - analytical) / np.linalg.norm(analytical) * 100.0
        ),
    }


def block_summary(numerical: np.ndarray, analytical: np.ndarray) -> dict[str, dict[str, float | None]]:
    blocks = {
        "A": (slice(0, 3), slice(0, 3)),
        "B": (slice(0, 3), slice(3, 6)),
        "D": (slice(3, 6), slice(3, 6)),
    }
    global_scale = float(np.max(np.abs(analytical)))
    result: dict[str, dict[str, float | None]] = {}
    for name, index in blocks.items():
        num = numerical[index]
        ana = analytical[index]
        nonzero = np.abs(ana) > global_scale * 1.0e-12
        zero = ~nonzero
        result[name] = {
            "max_nonzero_relative_error_percent": (
                float(np.max(np.abs(num[nonzero] - ana[nonzero]) / np.abs(ana[nonzero])) * 100.0)
                if np.any(nonzero)
                else None
            ),
            "max_zero_normalized_residual": (
                float(np.max(np.abs(num[zero] - ana[zero])) / global_scale) if np.any(zero) else None
            ),
        }
    return result


def run() -> None:
    resolution = 64
    thickness = 10.0
    nu = 0.35
    voxel = np.ones((resolution, resolution, resolution), dtype=np.uint8)

    homogeneous_analytical = analytical_abd([(-5.0, 5.0, 1215.0, nu)])
    homogeneous_numerical = homogenize_single(
        voxel, E=1215.0, nu=nu, thickness=thickness, Nx=1, Ny=1, Nz=1
    )

    cp.get_default_memory_pool().free_all_blocks()
    bilayer_analytical = analytical_abd([(-5.0, 0.0, 1215.0, nu), (0.0, 5.0, 500.0, nu)])
    bilayer_numerical = homogenize_bilayer(voxel, nu=nu, thickness=thickness, Nx=1, Ny=1, Nz=1)

    cases = {
        "homogeneous_isotropic_plate": {
            "parameters": {"resolution": resolution, "E_MPa": 1215.0, "nu": nu, "thickness_mm": thickness},
            "analytical_ABD": homogeneous_analytical.tolist(),
            "numerical_ABD": homogeneous_numerical.tolist(),
            "error": error_summary(homogeneous_numerical, homogeneous_analytical),
            "block_error": block_summary(homogeneous_numerical, homogeneous_analytical),
        },
        "asymmetric_two_layer_plate": {
            "parameters": {
                "resolution": resolution,
                "lower_layer": {"z_mm": [-5.0, 0.0], "E_MPa": 1215.0, "nu": nu},
                "upper_layer": {"z_mm": [0.0, 5.0], "E_MPa": 500.0, "nu": nu},
            },
            "analytical_ABD": bilayer_analytical.tolist(),
            "numerical_ABD": bilayer_numerical.tolist(),
            "error": error_summary(bilayer_numerical, bilayer_analytical),
            "block_error": block_summary(bilayer_numerical, bilayer_analytical),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
