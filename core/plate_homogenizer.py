import numpy as np
from scipy.sparse import coo_matrix
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla


def get_isotropic_elasticity(E=1.0, nu=0.3):
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))  # Calculate Lame parameters
    C = np.zeros((6, 6))  # Initialize 6x6 constitutive matrix
    C[0:3, 0:3] = lam
    np.fill_diagonal(C[0:3, 0:3], lam + 2 * mu)
    np.fill_diagonal(C[3:6, 3:6], mu)
    return C


def compute_element_stiffness(C, dx, dy, dz):
    Ke, Bs = np.zeros((24, 24)), []
    nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])  # 8-node local coordinates
    detJ = dx * dy * dz / 8.0
    for q in nodes / np.sqrt(3):  # Loop over 8 Gauss integration points
        dN = 0.125 * np.array([(1 + q[1] * nodes[:, 1]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 0] * (2 / dx), (1 + q[0] * nodes[:, 0]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 1] * (2 / dy), (1 + q[0] * nodes[:, 0]) * (1 + q[1] * nodes[:, 1]) * nodes[:, 2] * (2 / dz)])  # Cartesian derivatives
        B = np.zeros((6, 24))  # Strain-displacement matrix
        B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[0], dN[1], dN[2]
        B[3, 1::3], B[3, 2::3] = dN[2], dN[1]
        B[4, 0::3], B[4, 2::3] = dN[2], dN[0]
        B[5, 0::3], B[5, 1::3] = dN[1], dN[0]
        Bs.append(B)  # Store B matrices for stress recovery
        Ke += B.T @ C @ B * detJ  # Accumulate local stiffness
    return Ke, Bs, detJ


def build_tensor_dof_mapping(voxel, dx, dy, dz, thickness):
    nx, ny, nz = voxel.shape  # Enforce XYZ indexing convention
    node_indices = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
    dof_tensor = np.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=int)
    dof_tensor[..., 0], dof_tensor[..., 1], dof_tensor[..., 2] = node_indices * 3, node_indices * 3 + 1, node_indices * 3 + 2
    dof_tensor[nx, :, :, :] = dof_tensor[0, :, :, :]
    dof_tensor[:, ny, :, :] = dof_tensor[:, 0, :, :]

    n1, n2 = dof_tensor[:-1, :-1, :-1, :], dof_tensor[1:, :-1, :-1, :]
    n3, n4 = dof_tensor[1:, 1:, :-1, :], dof_tensor[:-1, 1:, :-1, :]
    n5, n6 = dof_tensor[:-1, :-1, 1:, :], dof_tensor[1:, :-1, 1:, :]
    n7, n8 = dof_tensor[1:, 1:, 1:, :], dof_tensor[:-1, 1:, 1:, :]
    edof_tensor = np.concatenate([n1, n2, n3, n4, n5, n6, n7, n8], axis=-1)

    active_mask = voxel > 0  # Filter out void regions
    edofMat = edof_tensor[active_mask]

    z_coords = np.linspace(dz / 2, thickness - dz / 2, nz) - thickness / 2.0
    z_grid = np.broadcast_to(z_coords[None, None, :], (nx, ny, nz))  # Z is the 3rd axis
    z_active = z_grid[active_mask]
    return edofMat, z_active, node_indices.size * 3

def homogenization_plate(voxel, E=2000.0, nu=0.3, thickness=10.0, Nx=1, Ny=1, Nz=1):
    nx, ny, nz = voxel.shape
    cell_size = thickness / Nz
    Lx = Nx * cell_size
    Ly = Ny * cell_size
    Lz = thickness
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    plate_area = Lx * Ly  # Normalization factor for LPS-H

    C = get_isotropic_elasticity(E, nu)
    Ke, Bs, detJ = compute_element_stiffness(C, dx, dy, dz)
    edofMat, z_active, total_dofs = build_tensor_dof_mapping(voxel, dx, dy, dz, thickness)

    iK, jK = np.repeat(edofMat, 24, axis=1).flatten(), np.tile(edofMat, (1, 24)).flatten()
    sK = np.tile(Ke.flatten(), edofMat.shape[0])
    K = coo_matrix((sK, (iK, jK)), shape=(total_dofs, total_dofs)).tocsr()  # Loop-free assembly of global stiffness

    E_macro = np.zeros((len(z_active), 6, 6))
    E_macro[:, 0, 0], E_macro[:, 1, 1], E_macro[:, 5, 2] = 1.0, 1.0, 1.0  # Apply unit membrane strains
    E_macro[:, 0, 3], E_macro[:, 1, 4], E_macro[:, 5, 5] = z_active, z_active, z_active  # Apply unit bending curvatures

    F_ele = sum([np.einsum('ji,kjl->kil', Bs[i], np.einsum('ij,kjl->kil', C, E_macro)) * detJ for i in range(8)])  # Local load via tensor contraction
    F = np.column_stack([np.bincount(edofMat.flatten(), weights=F_ele[:, :, c].flatten(), minlength=total_dofs) for c in range(6)])  # Global load assembly
    active_dofs = np.setdiff1d(np.unique(edofMat), np.unique(edofMat)[:3])  # Eliminate rigid body motions

    U = np.zeros((total_dofs, 6))
    K_active_gpu = cpsp.csr_matrix(K[active_dofs, :][:, active_dofs])
    F_active_gpu = cp.asarray(F[active_dofs, :])
    U_active_gpu = cp.zeros((len(active_dofs), 6))
    M_gpu = cpsp.diags(1.0 / K_active_gpu.diagonal())
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(6)]  # Initialize concurrent CUDA streams
    for c in range(6):
        with streams[c]:  # Launch asynchronous GPU solving for 6 load cases
            U_col_gpu, _ = cpspla.cg(K_active_gpu, F_active_gpu[:, c], M=M_gpu, tol=1e-6, maxiter=5000)  # Jacobi Preconditioner Conjugate gradient solver on GPU
            U_active_gpu[:, c] = U_col_gpu
    for stream in streams:
        stream.synchronize()  # Barrier synchronization
    U[active_dofs, :] = U_active_gpu.get()

    ABD = np.zeros((6, 6))
    for i_gp in range(8):
        Sigma = np.einsum('ij,kjl->kil', C, E_macro - np.einsum('ij,kjl->kil', Bs[i_gp], U[edofMat, :]))  # Recover true physical stress
        ABD[0:3, :] += np.sum(Sigma[:, [0, 1, 5], :], axis=0) * detJ / plate_area  # Extract A and B matrices (0th moment)
        ABD[3:6, :] += np.sum(Sigma[:, [0, 1, 5], :] * z_active[:, None, None], axis=0) * detJ / plate_area  # Extract D and B* matrices (1st moment)
    return (ABD + ABD.T) / 2.0  # Symmetrization for Maxwell-Betti reciprocity
