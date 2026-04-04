import time
import numpy as np

from core.plate_thermal_homogenizer import homogenization_plate_thermal
from utils.tpms_generator import generate_tpms_voxel_grid


def run_simulation():
    # =====================================================================
    # 1. Define Physical and Geometric Parameters
    # =====================================================================
    k_base = 60.5
    thickness = 10.0  # Absolute thickness of the macroscopic plate (mm)

    # Define the number of unit cells in X, Y, and Z directions.
    # Nx=Ny=Nz=1 represents a single truncated unit cell cube.
    Nx, Ny, Nz = 1, 1, 1

    res = 96  # Voxel resolution per unit cell (increase for higher accuracy)
    density = 0.15  # Target relative density (volume fraction)
    tpms_type = 'Primitive'  # TPMS topology type

    # =====================================================================
    # 2. Generate TPMS Voxel Model
    # =====================================================================
    print(f"Generating Sheet {tpms_type} ({Nx}x{Ny}x{Nz}) at res={res}...")
    t0 = time.time()

    voxel_grid = generate_tpms_voxel_grid(
        tpms_type=tpms_type,
        Nx=Nx, Ny=Ny, Nz=Nz,
        resolution=res,
        relative_density=density,
        is_sheet=True
    )

    # =====================================================================
    # 3. Execute GPU-Accelerated 2D Plate Homogenization
    # =====================================================================
    print("Solving ABD Matrix via CuPy GPU Solver...")
    ABD_matrix = homogenization_plate_thermal(
        voxel=voxel_grid,
        k_base=k_base,
        thickness=thickness,
        Nx=Nx, Ny=Ny, Nz=Nz
    )

    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds.\n")

    # =====================================================================
    # 4. Format and Output Results
    # =====================================================================
    # Limit output to 2 decimal places and suppress scientific notation for clarity
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"--- [ABD] Full Matrix (6x6) for {tpms_type} ---")
    print(ABD_matrix)


if __name__ == "__main__":
    run_simulation()