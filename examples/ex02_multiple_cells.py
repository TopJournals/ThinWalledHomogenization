"""
examples/ex02_multiple_cells.py

Demonstrates how to compute the homogenization of a TPMS plate with multiple
unit cell arrays along the Z-axis (e.g., 1x1x2).
As Nz increases, you can observe the variation in the bending stiffness (D matrix)
and the gradual attenuation of the extension-bending coupling (B matrix).
"""

import time
import numpy as np

# Note: The details of the TPMS generation module are provided in Appendix A.
from utils.tpms_generator import generate_tpms_voxel_grid
from core.plate_homogenizer import homogenization_plate


def run_simulation_ex02():
    # =====================================================================
    # 1. Define Physical and Geometric Parameters
    # =====================================================================
    E_base = 1215.0  # Young's modulus of the base material (MPa)
    nu_base = 0.35  # Poisson's ratio of the base material

    # Define the number of unit cells in X, Y, and Z directions.
    Nx, Ny, Nz = 1, 1, 3

    plate_thickness = 10.0
    unit_cell_size = plate_thickness / Nz

    res = 64  # Voxel resolution per unit cell (Total grid: 50x50x100 for 1x1x2)
    density = 0.15  # Target relative density (volume fraction)
    tpms_type = 'Gyroid'  # TPMS topology type

    # =====================================================================
    # 2. Generate Multi-Cell TPMS Voxel Model
    # =====================================================================
    print(f"\n[1/3] Generating Sheet {tpms_type} voxel grid ({Nx}x{Ny}x{Nz}) at res={res}...")
    t0 = time.time()

    voxel_grid = generate_tpms_voxel_grid(
        tpms_type=tpms_type,
        Nx=Nx, Ny=Ny, Nz=Nz,
        resolution=res,
        relative_density=density,
        is_sheet=True
    )

    t1 = time.time()
    print(f"      Grid generated! True shape (Nz*res, Ny*res, Nx*res): {voxel_grid.shape}")
    print(f"      Elapsed time: {t1 - t0:.2f} seconds")

    # =====================================================================
    # 3. Execute GPU-Accelerated 2D Plate Homogenization
    # =====================================================================
    print(f"\n[2/3] Feeding {voxel_grid.size} voxels into CuPy GPU solver...")
    t2 = time.time()

    ABD_matrix = homogenization_plate(
        voxel=voxel_grid,
        E=E_base,
        nu=nu_base,
        thickness=plate_thickness,
        Nx=Nx, Ny=Ny, Nz=Nz  # Passed to ensure correct macro-area (Lx*Ly) calculation
    )

    t3 = time.time()
    print(f"      Solving and stress recovery completed! Elapsed time: {t3 - t2:.2f} seconds")

    # =====================================================================
    # 4. Format and Output Results
    # =====================================================================
    print("\n[3/3] Macroscopic Plate Stiffness Matrix (ABD) Extraction Results:\n")

    # Limit output to 2 decimal places and suppress scientific notation for clarity
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"--- [ABD] Full Matrix (6x6) for {tpms_type} (Nz={Nz}) ---")
    print(ABD_matrix)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_simulation_ex02()