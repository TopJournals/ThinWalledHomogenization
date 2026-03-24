import numpy as np
from scipy.sparse import coo_matrix
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla


def get_isotropic_elasticity(E=1.0, nu=0.3):
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[0:3, 0:3] = lam
    np.fill_diagonal(C[0:3, 0:3], lam + 2 * mu)
    np.fill_diagonal(C[3:6, 3:6], mu)
    return C


def compute_element_stiffness(C, dx, dy, dz):
    Ke, Bs = np.zeros((24, 24)), []
    pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    detJ = dx * dy * dz / 8.0
    for z in pts:
        for y in pts:
            for x in pts:
                q = np.array([x, y, z])
                dN = 0.125 * np.array([(1 + q[1] * nodes[:, 1]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 0],
                                       (1 + q[0] * nodes[:, 0]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 1],
                                       (1 + q[0] * nodes[:, 0]) * (1 + q[1] * nodes[:, 1]) * nodes[:, 2]])
                B = np.zeros((6, 24))
                B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[0] * 2 / dx, dN[1] * 2 / dy, dN[2] * 2 / dz
                B[3, 1::3], B[3, 2::3] = dN[2] * 2 / dz, dN[1] * 2 / dy
                B[4, 0::3], B[4, 2::3] = dN[2] * 2 / dz, dN[0] * 2 / dx
                B[5, 0::3], B[5, 1::3] = dN[1] * 2 / dy, dN[0] * 2 / dx
                Bs.append(B)
                Ke += B.T @ C @ B * detJ
    return Ke, Bs, detJ


def build_tensor_dof_mapping(voxel, dx, dy, dz, thickness):
    nz, ny, nx = voxel.shape
    node_indices = np.arange((nz + 1) * (ny + 1) * (nx + 1)).reshape(nz + 1, ny + 1, nx + 1)
    dof_tensor = np.zeros((nz + 1, ny + 1, nx + 1, 3), dtype=int)
    dof_tensor[..., 0], dof_tensor[..., 1], dof_tensor[..., 2] = node_indices * 3, node_indices * 3 + 1, node_indices * 3 + 2

    # Apply 2D-PBC
    dof_tensor[:, :, nx, :] = dof_tensor[:, :, 0, :]
    dof_tensor[:, ny, :, :] = dof_tensor[:, 0, :, :]

    n1, n2 = dof_tensor[:-1, :-1, :-1, :], dof_tensor[:-1, :-1, 1:, :]
    n3, n4 = dof_tensor[:-1, 1:, 1:, :], dof_tensor[:-1, 1:, :-1, :]
    n5, n6 = dof_tensor[1:, :-1, :-1, :], dof_tensor[1:, :-1, 1:, :]
    n7, n8 = dof_tensor[1:, 1:, 1:, :], dof_tensor[1:, 1:, :-1, :]
    edof_tensor = np.concatenate([n1, n2, n3, n4, n5, n6, n7, n8], axis=-1)

    active_mask = voxel > 0
    edofMat = edof_tensor[active_mask]

    z_coords = np.linspace(dz/2, thickness - dz/2, nz)
    z_grid = np.broadcast_to(z_coords[:, None, None], (nz, ny, nx))
    z_active = z_grid[active_mask]
    return edofMat, z_active, node_indices.size * 3


def homogenization_plate(voxel, E=2000.0, nu=0.3, thickness=10.0, Nx=1, Ny=1, Nz=1):
    nz, ny, nx = voxel.shape

    # Calculate exact physical dimensions assuming cubic sub-cells
    cell_size = thickness / Nz
    Lx = Nx * cell_size
    Ly = Ny * cell_size
    Lz = thickness
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    plate_area = Lx * Ly  # Critical for ABD normalization

    C = get_isotropic_elasticity(E, nu)
    Ke, Bs, detJ = compute_element_stiffness(C, dx, dy, dz)
    edofMat, z_active, total_dofs = build_tensor_dof_mapping(voxel, dx, dy, dz, thickness)

    iK, jK = np.repeat(edofMat, 24, axis=1).flatten(), np.tile(edofMat, (1, 24)).flatten()
    sK = np.tile(Ke.flatten(), edofMat.shape[0])
    K = coo_matrix((sK, (iK, jK)), shape=(total_dofs, total_dofs)).tocsr()

    E_macro = np.zeros((len(z_active), 6, 6))
    E_macro[:, 0, 0], E_macro[:, 1, 1], E_macro[:, 5, 2] = 1.0, 1.0, 1.0
    E_macro[:, 0, 3], E_macro[:, 1, 4], E_macro[:, 5, 5] = z_active, z_active, z_active

    F_ele = sum([np.einsum('ji,kjl->kil', Bs[i], np.einsum('ij,kjl->kil', C, E_macro)) * detJ for i in range(8)])
    F = np.column_stack([np.bincount(edofMat.flatten(), weights=F_ele[:, :, c].flatten(), minlength=total_dofs) for c in range(6)])

    active_dofs = np.setdiff1d(np.unique(edofMat), np.unique(edofMat)[:3])
    U = np.zeros((total_dofs, 6))

    # CuPy GPU Solver
    K_active_gpu = cpsp.csr_matrix(K[active_dofs, :][:, active_dofs])
    for c in range(6):
        F_active_gpu = cp.asarray(F[active_dofs, c])
        U_gpu, _ = cpspla.cg(K_active_gpu, F_active_gpu, tol=1e-6)
        U[active_dofs, c] = U_gpu.get()

    ABD = np.zeros((6, 6))
    for i_gp in range(8):
        Sigma = np.einsum('ij,kjl->kil', C, E_macro - np.einsum('ij,kjl->kil', Bs[i_gp], U[edofMat, :]))
        # Normalize by plate_area!
        ABD[0:3, :] += np.sum(Sigma[:, [0, 1, 5], :], axis=0) * detJ / plate_area
        ABD[3:6, :] += np.sum(Sigma[:, [0, 1, 5], :] * z_active[:, None, None], axis=0) * detJ / plate_area

    return (ABD + ABD.T) / 2.0