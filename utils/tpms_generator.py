"""
utils/tpms_generator.py
A lightweight utility to generate voxel grids of Triply Periodic Minimal Surfaces (TPMS).
"""

import numpy as np


def _gyroid_equation(x, y, z):
    """Gyroid implicit equation (2*pi is embedded, cell domain is [0, 1])"""
    return (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + np.sin(2 * np.pi * z) * np.cos(2 * np.pi * x) + np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z))


def _primitive_equation(x, y, z):
    """Primitive implicit equation (2*pi is embedded, cell domain is [0, 1])"""
    return np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)


def _diamond_equation(x, y, z):
    """Diamond implicit equation (2*pi is embedded, cell domain is [0, 1])"""
    return (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) + np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z) + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z))


def generate_tpms_voxel_grid(tpms_type='Gyroid', Nx=1, Ny=1, Nz=1, resolution=60, relative_density=0.3, is_sheet=True):
    """
    Generate a 3D binary numpy array representing a TPMS structure.
    The output array uses (z, y, x) indexing for memory continuity in FEA.

    Parameters:
    -----------
    tpms_type : str
        'Gyroid', 'Primitive', or 'Diamond'.
    Nx, Ny, Nz : int
        Number of unit cells in X, Y, and Z directions.
    resolution : int
        Number of voxels per unit cell along one axis.
    relative_density : float
        Target volume fraction (0.0 to 1.0).
    is_sheet : bool
        If True, generates Sheet-based TPMS. If False, generates Strut-based.

    Returns:
    --------
    voxel_grid : numpy.ndarray (3D)
        A 3D array of shape (Nz*res, Ny*res, Nx*res) containing 0s (void) and 1s (solid).
    """

    equations = {'Gyroid': _gyroid_equation, 'Primitive': _primitive_equation, 'Diamond': _diamond_equation}
    if tpms_type not in equations:
        raise ValueError(f"TPMS type '{tpms_type}' not supported.")
    func = equations[tpms_type]

    # 1. Generate normalized coordinate grid using voxel center points
    # Calculate the pitch (size of one voxel)
    dx = Nx / (Nx * resolution)
    dy = Ny / (Ny * resolution)
    dz = Nz / (Nz * resolution)

    # Shift the coordinates by half a pitch to evaluate at the center of each voxel
    z = (np.arange(Nz * resolution) + 0.5) * dz
    y = (np.arange(Ny * resolution) + 0.5) * dy
    x = (np.arange(Nx * resolution) + 0.5) * dx

    # 2. Create 3D meshgrid in (z, y, x) order
    # indexing='ij' ensures output shape is strictly (len(z), len(y), len(x))
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # 3. Evaluate the implicit field
    field = func(X, Y, Z)

    # 4. Apply threshold based on the target relative density
    if is_sheet:
        eval_field = np.abs(field)
        threshold = np.percentile(eval_field, relative_density * 100)
        voxel_grid = (eval_field <= threshold).astype(np.int8)
    else:
        eval_field = field
        threshold = np.percentile(eval_field, (1 - relative_density) * 100)
        voxel_grid = (eval_field > threshold).astype(np.int8)

    return voxel_grid


if __name__ == "__main__":
    print("Generating Sheet Gyroid Voxel Grid with ZYX Indexing...")

    grid = generate_tpms_voxel_grid(tpms_type='Gyroid', Nx=1, Ny=1, Nz=1, resolution=64, relative_density=0.15, is_sheet=True)

    print(f"Grid shape (Nz, Ny, Nx): {grid.shape}")
    print(f"Actual volume fraction: {grid.mean():.4f}")
