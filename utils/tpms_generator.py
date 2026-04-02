"""
utils/tpms_generator.py

A lightweight utility to generate voxel grids of Triply Periodic Minimal Surfaces (TPMS).
The output array strictly follows the (x, y, z) indexing convention to ensure
seamless memory alignment with the 2D-PH finite element solver.
"""

import numpy as np


def _primitive_equation(x, y, z):
    """Primitive (P) / Sheet P"""
    return np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)


def _diamond_equation(x, y, z):
    """Diamond (D) / Sheet D"""
    return (np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) -
            np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z))


def _gyroid_equation(x, y, z):
    """Gyroid (G) / Sheet G"""
    return (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) +
            np.sin(2 * np.pi * z) * np.cos(2 * np.pi * x) +
            np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z))


def _iwp_equation(x, y, z):
    """I-WP (W) / Sheet I-WP"""
    return (2 * (np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) +
                 np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) +
                 np.cos(2 * np.pi * z) * np.cos(2 * np.pi * x)) -
            (np.cos(4 * np.pi * x) + np.cos(4 * np.pi * y) + np.cos(4 * np.pi * z)))


def _frd_equation(x, y, z):
    """F-RD / Sheet F-RD"""
    return (4 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) -
            (np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y) +
             np.cos(4 * np.pi * y) * np.cos(4 * np.pi * z) +
             np.cos(4 * np.pi * z) * np.cos(4 * np.pi * x)))


def _l_equation(x, y, z):
    """L / Sheet L"""
    return (0.5 * (np.sin(4 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z) +
                   np.sin(4 * np.pi * y) * np.cos(2 * np.pi * z) * np.sin(2 * np.pi * x) +
                   np.sin(4 * np.pi * z) * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)) -
            0.5 * (np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y) +
                   np.cos(4 * np.pi * y) * np.cos(4 * np.pi * z) +
                   np.cos(4 * np.pi * z) * np.cos(4 * np.pi * x)) + 0.15)


def _tubular_p_equation(x, y, z):
    """Tubular P / Sheet Tubular P"""
    return (10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)) -
            5.1 * (np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) +
                   np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) +
                   np.cos(2 * np.pi * z) * np.cos(2 * np.pi * x)) - 14.6)


def _tubular_g_equation(x, y, z):
    """Tubular G / Sheet Tubular G"""
    return (10 * (np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) +
                  np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z) +
                  np.cos(2 * np.pi * z) * np.sin(2 * np.pi * x)) -
            0.5 * (np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y) +
                   np.cos(4 * np.pi * y) * np.cos(4 * np.pi * z) +
                   np.cos(4 * np.pi * z) * np.cos(4 * np.pi * x)) - 14)


def _i2_y_equation(x, y, z):
    """I2-Y / Sheet I2-Y"""
    return (-2 * (np.sin(4 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z) +
                  np.sin(2 * np.pi * x) * np.sin(4 * np.pi * y) * np.cos(2 * np.pi * z) +
                  np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(4 * np.pi * z)) +
            np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y) +
            np.cos(4 * np.pi * y) * np.cos(4 * np.pi * z) +
            np.cos(4 * np.pi * x) * np.cos(4 * np.pi * z))


def generate_tpms_voxel_grid(tpms_type='Gyroid', Nx=1, Ny=1, Nz=1, resolution=60, relative_density=0.3, is_sheet=True):
    """
    Generate a 3D binary numpy array representing a TPMS structure.
    The output array uses (x, y, z) indexing for Cartesian geometric alignment.

    Parameters:
    -----------
    tpms_type : str
        'Gyroid', 'Primitive', 'Diamond', etc.
    Nx, Ny, Nz : int
        Number of unit cells in X, Y, and Z directions.
    resolution : int
        Number of voxels per unit cell along one axis.
    relative_density : float
        Target volume fraction (0.0 to 1.0).
    is_sheet : bool
        If True, generates Sheet-based TPMS. If False, generates Strut-based (Network).

    Returns:
    --------
    voxel_grid : numpy.ndarray (3D)
        A 3D array of shape (Nx*res, Ny*res, Nz*res) containing 0s (void) and 1s (solid).
    """

    equations = {
        'Primitive': _primitive_equation,
        'Diamond': _diamond_equation,
        'Gyroid': _gyroid_equation,
        'I-WP': _iwp_equation,
        'F-RD': _frd_equation,
        'L': _l_equation,
        'Tubular P': _tubular_p_equation,
        'Tubular G': _tubular_g_equation,
        'I2-Y': _i2_y_equation
    }

    if tpms_type not in equations:
        raise ValueError(f"TPMS type '{tpms_type}' not supported.")
    func = equations[tpms_type]

    # 1. Generate normalized coordinate grid using voxel center points
    dx = Nx / (Nx * resolution)
    dy = Ny / (Ny * resolution)
    dz = Nz / (Nz * resolution)

    # Shift the coordinates by half a pitch to evaluate at the center of each voxel
    x = (np.arange(Nx * resolution) + 0.5) * dx
    y = (np.arange(Ny * resolution) + 0.5) * dy
    z = (np.arange(Nz * resolution) + 0.5) * dz

    # 2. Create 3D meshgrid strictly in (x, y, z) order
    # indexing='ij' ensures output shape is (len(x), len(y), len(z))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

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
    print("Generating Sheet Gyroid Voxel Grid with strict XYZ Indexing...")

    grid = generate_tpms_voxel_grid(tpms_type='Gyroid', Nx=1, Ny=1, Nz=2, resolution=64, relative_density=0.15, is_sheet=True)

    print(f"Grid shape (Nx, Ny, Nz): {grid.shape}")
    print(f"Actual volume fraction: {grid.mean():.4f}")