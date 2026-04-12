import time
import numpy as np

from utils.lattice_generator import generate_lattice_voxel_grid
from core.plate_homogenizer import homogenization_plate


def run_lattice_simulation():
    # =====================================================================
    # 1. Define Physical and Geometric Parameters
    # =====================================================================
    E_base = 1215.0  # Young's modulus of the base material (MPa)
    nu_base = 0.35   # Poisson's ratio of the base material
    thickness = 10.0 # Absolute thickness of the macroscopic plate (mm)

    # Define the number of unit cells in X, Y, and Z directions.
    Nx, Ny, Nz = 1, 1, 1

    res = 96         # Voxel resolution per unit cell (桁架较细时可适当提高)
    density = 0.15   # Target relative density (volume fraction)
    lattice_type = 'BCC'  # Lattice topology type

    # =====================================================================
    # 2. Generate Lattice Voxel Model
    # =====================================================================
    print(f"Generating Truss {lattice_type} ({Nx}x{Ny}x{Nz}) at res={res}...")
    t0 = time.time()

    # 调用桁架生成器，去除了 is_sheet 参数
    voxel_grid = generate_lattice_voxel_grid(
        lattice_type=lattice_type,
        Nx=Nx, Ny=Ny, Nz=Nz,
        resolution=res,
        relative_density=density
    )

    voxel_grid = np.pad(
        voxel_grid,
        pad_width=((0, 0), (0, 0), (2, 2)),
        mode='constant',
        constant_values=1
    )

    # =====================================================================
    # 3. Execute GPU-Accelerated 2D Plate Homogenization
    # =====================================================================
    print("Solving ABD Matrix via CuPy GPU Solver...")
    ABD_matrix = homogenization_plate(
        voxel=voxel_grid,
        E=E_base,
        nu=nu_base,
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
    print(f"--- [ABD] Full Matrix (6x6) for {lattice_type} Truss ---")
    print(ABD_matrix)


if __name__ == "__main__":
    run_lattice_simulation()