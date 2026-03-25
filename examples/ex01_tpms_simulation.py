import time
import numpy as np

from utils.tpms_generator import generate_tpms_voxel_grid
from core.plate_homogenizer import homogenization_plate

def run_simulation():
    # 物理参数 (反推适配你的正确结果)
    E_base = 1215.0       # MPa
    nu_base = 0.35
    thickness = 10.0      # 10mm
    Nx, Ny, Nz = 1, 1, 1  # 确保是 10x10x10 的立方体单元截断
    res = 128              # 网格分辨率 (可根据显存适当提高)
    density = 0.15        # 相对密度
    tpms_type = 'I-WP'
    print(f"Generating Sheet {tpms_type} ({Nx}x{Ny}x{Nz}) at res={res}...")
    t0 = time.time()
    voxel_grid = generate_tpms_voxel_grid(
        tpms_type=tpms_type, Nx=Nx, Ny=Ny, Nz=Nz,
        resolution=res, relative_density=density, is_sheet=True
    )

    print(f"Solving ABD Matrix via CuPy GPU Solver...")
    ABD_matrix = homogenization_plate(
        voxel=voxel_grid,
        E=E_base, nu=nu_base,
        thickness=thickness,
        Nx=Nx, Ny=Ny, Nz=Nz
    )
    t1 = time.time()
    print(f"Done in {t1-t0:.2f} seconds.\n")

    # 按照要求：仅输出整体 6x6 矩阵，保留两位小数
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print("--- [ABD] Full Matrix (6x6) ---")
    print(ABD_matrix)

if __name__ == "__main__":
    run_simulation()