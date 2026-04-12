"""
utils/lattice_generator.py

A lightweight utility to generate voxel grids of Truss Lattice structures.
The output array strictly follows the (x, y, z) indexing convention to ensure
seamless memory alignment with the 2D-PH finite element solver.
"""

import numpy as np


def _bcc_distance_field(X, Y, Z):
    """
    BCC (Body-Centered Cubic) Truss Distance Field.
    Calculates the minimum distance from any point in space to the BCC strut skeleton.
    """
    # Map global coordinates to the local unit cell [0.0, 1.0) to enforce periodicity
    x = X % 1.0
    y = Y % 1.0
    z = Z % 1.0

    # BCC contains 4 main diagonals. Calculate the squared distance to each line.
    # Line 1: (0,0,0) -> (1,1,1)
    d1_sq = ((y - z)**2 + (z - x)**2 + (x - y)**2) / 3.0

    # Line 2: (1,0,0) -> (0,1,1)
    d2_sq = ((y - z)**2 + (1.0 - x - z)**2 + (x + y - 1.0)**2) / 3.0

    # Line 3: (0,1,0) -> (1,0,1)
    d3_sq = ((y + z - 1.0)**2 + (z - x)**2 + (1.0 - x - y)**2) / 3.0

    # Line 4: (0,0,1) -> (1,1,0)
    d4_sq = ((1.0 - y - z)**2 + (x + z - 1.0)**2 + (x - y)**2) / 3.0

    # The field value is the distance to the closest strut
    return np.sqrt(np.minimum.reduce([d1_sq, d2_sq, d3_sq, d4_sq]))


def generate_lattice_voxel_grid(lattice_type='BCC', Nx=1, Ny=1, Nz=1, resolution=60, relative_density=0.15):
    """
    Generate a 3D binary numpy array representing a Truss Lattice structure.
    The output array uses (x, y, z) indexing for Cartesian geometric alignment.

    Parameters:
    -----------
    lattice_type : str
        'BCC', etc.
    Nx, Ny, Nz : int
        Number of unit cells in X, Y, and Z directions.
    resolution : int
        Number of voxels per unit cell along one axis.
    relative_density : float
        Target volume fraction (0.0 to 1.0).

    Returns:
    --------
    voxel_grid : numpy.ndarray (3D)
        A 3D array of shape (Nx*res, Ny*res, Nz*res) containing 0s (void) and 1s (solid).
    """

    fields = {
        'BCC': _bcc_distance_field
    }

    if lattice_type not in fields:
        raise ValueError(f"Lattice type '{lattice_type}' not supported.")

    func = fields[lattice_type]

    # 1. Generate normalized coordinate grid using voxel center points
    dx = 1.0 / resolution
    dy = 1.0 / resolution
    dz = 1.0 / resolution

    # Coordinate units are scaled to "Unit Cells"
    x = (np.arange(Nx * resolution) + 0.5) * dx
    y = (np.arange(Ny * resolution) + 0.5) * dy
    z = (np.arange(Nz * resolution) + 0.5) * dz

    # 2. Create 3D meshgrid strictly in (x, y, z) order (indexing='ij')
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 3. Evaluate the distance field
    distance_field = func(X, Y, Z)

    # 4. Apply threshold based on the target relative density
    # Smaller distance values correspond to regions closer to the strut skeleton.
    # Voxels with a distance less than or equal to the computed threshold become solid.
    threshold = np.percentile(distance_field, relative_density * 100)
    voxel_grid = (distance_field <= threshold).astype(np.int8)

    return voxel_grid


if __name__ == "__main__":
    grid = generate_lattice_voxel_grid(lattice_type='BCC', Nx=2, Ny=2, Nz=2, resolution=50, relative_density=0.20)