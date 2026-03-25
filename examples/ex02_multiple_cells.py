"""
examples/ex02_multiple_cells.py
演示如何计算 Z 轴方向具有多个晶胞阵列 (例如 1x1x2) 的 TPMS 薄板均匀化。
随着 Nz 的增加，你可以观察到 D 矩阵的变化以及 B 矩阵（拉弯耦合）的逐渐衰减。
"""

import time
import numpy as np

from utils.tpms_generator import generate_tpms_voxel_grid
from core.plate_homogenizer import homogenization_plate


def run_simulation_ex02():
    print("=" * 70)
    print(" 启动多胞阵列 TPMS 薄板均匀化分析 (1x1x2) - GPU 加速")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. 物理参数与阵列设置
    # ---------------------------------------------------------
    E_base = 1215.0  # 基体材料杨氏模量 (MPa)
    nu_base = 0.35  # 基体材料泊松比

    # 晶胞阵列数设置
    Nx, Ny, Nz = 1, 1, 1.25

    # 假设单胞尺寸为 10mm x 10mm x 10mm
    # 当 Nz=2 时，宏观薄板的真实物理总厚度必须是 20.0 mm

    plate_thickness = 10.0
    unit_cell_size = plate_thickness / Nz

    res = 50  # 每个单胞的网格分辨率 (对于 1x1x2，总网格数为 50 x 50 x 100)
    density = 0.15  # 相对密度

    # ---------------------------------------------------------
    # 2. 生成多胞 TPMS 体素网格
    # ---------------------------------------------------------
    print(f"\n[1/3] 正在生成 Sheet Gyroid 体素网格 ({Nx}x{Ny}x{Nz})...")
    t0 = time.time()
    voxel_grid = generate_tpms_voxel_grid(
        tpms_type='Gyroid',
        Nx=Nx, Ny=Ny, Nz=Nz,
        resolution=res,
        relative_density=density,
        is_sheet=True
    )
    t1 = time.time()
    print(f"      网格生成完毕！实际形状 (Nz*res, Ny*res, Nx*res): {voxel_grid.shape}")
    print(f"      耗时: {t1 - t0:.2f} 秒")

    # 根据 Benchmark 坐标系定义，交换 X 与 Y 轴
    voxel_grid = np.swapaxes(voxel_grid, 1, 2)

    # ---------------------------------------------------------
    # 3. 运行 GPU 均匀化求解
    # ---------------------------------------------------------
    print(f"\n[2/3] 正在将 {voxel_grid.size} 个体素送入 CuPy 求解器计算...")
    t2 = time.time()
    ABD_matrix = homogenization_plate(
        voxel=voxel_grid,
        E=E_base,
        nu=nu_base,
        thickness=plate_thickness,  # 这里的厚度是 20.0
        Nx=Nx, Ny=Ny, Nz=Nz  # 传入阵列参数以确保面积 (Lx*Ly) 计算正确
    )
    t3 = time.time()
    print(f"      求解及应力恢复完毕！耗时: {t3 - t2:.2f} 秒")

    # ---------------------------------------------------------
    # 4. 格式化输出 6x6 ABD 矩阵
    # ---------------------------------------------------------
    print("\n[3/3] 宏观板壳刚度矩阵提取结果:\n")

    # 保留 2 位小数，不使用科学计数法
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print("--- [ABD] Full Matrix (6x6) ---")
    print(ABD_matrix)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_simulation_ex02()