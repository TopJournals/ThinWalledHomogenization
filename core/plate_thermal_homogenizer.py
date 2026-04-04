import numpy as np
from scipy.sparse import coo_matrix
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla


def get_thermal_conductivity(k_base=200.0):
    k_tensor = np.zeros((3, 3))  # Initialize 3x3 conductivity matrix
    np.fill_diagonal(k_tensor, k_base)
    return k_tensor


def compute_element_thermal_conductivity(k_tensor, dx, dy, dz):
    Kt_e, gradNs = np.zeros((8, 8)), []
    nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])  # 8-node local coordinates
    detJ = dx * dy * dz / 8.0
    for q in nodes / np.sqrt(3):  # Loop over 8 Gauss integration points
        dN = 0.125 * np.array([(1 + q[1] * nodes[:, 1]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 0] * (2 / dx), (1 + q[0] * nodes[:, 0]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 1] * (2 / dy), (1 + q[0] * nodes[:, 0]) * (1 + q[1] * nodes[:, 1]) * nodes[:, 2] * (2 / dz)])
        gradNs.append(dN)
        Kt_e += dN.T @ k_tensor @ dN * detJ  # Accumulate local thermal conductivity
    return Kt_e, gradNs, detJ


def build_tensor_dof_mapping(voxel, dx, dy, dz, thickness):
    nx, ny, nz = voxel.shape  # Enforce XYZ indexing convention
    node_indices = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)

    dof_tensor = node_indices.copy()

    dof_tensor[nx, :, :] = dof_tensor[0, :, :]
    dof_tensor[:, ny, :] = dof_tensor[:, 0, :]

    n1, n2 = dof_tensor[:-1, :-1, :-1], dof_tensor[1:, :-1, :-1]
    n3, n4 = dof_tensor[1:, 1:, :-1], dof_tensor[:-1, 1:, :-1]
    n5, n6 = dof_tensor[:-1, :-1, 1:], dof_tensor[1:, :-1, 1:]
    n7, n8 = dof_tensor[1:, 1:, 1:], dof_tensor[:-1, 1:, 1:]
    edof_tensor = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=-1)

    active_mask = voxel > 0  # Filter out void regions
    edofMat = edof_tensor[active_mask]
    z_coords = np.linspace(dz / 2, thickness - dz / 2, nz) - thickness / 2.0
    z_grid = np.broadcast_to(z_coords[None, None, :], (nx, ny, nz))  # Z is the 3rd axis
    z_active = z_grid[active_mask]

    return edofMat, z_active, node_indices.size


def homogenization_plate_thermal(voxel, k_base=200.0, thickness=10.0, Nx=1, Ny=1, Nz=1):
    nz, ny, nx = voxel.shape
    cell_size = thickness / Nz
    Lx = Nx * cell_size
    Ly = Ny * cell_size
    Lz = thickness
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    plate_area = Lx * Ly  # Critical normalization factor for 2D-PH

    k_tensor = get_thermal_conductivity(k_base)
    Kt_e, gradNs, detJ = compute_element_thermal_conductivity(k_tensor, dx, dy, dz)
    edofMat, z_active, total_dofs = build_tensor_dof_mapping(voxel, dx, dy, dz, thickness)

    iK, jK = np.repeat(edofMat, 8, axis=1).flatten(), np.tile(edofMat, (1, 8)).flatten()
    sK = np.tile(Kt_e.flatten(), edofMat.shape[0])
    K = coo_matrix((sK, (iK, jK)), shape=(total_dofs, total_dofs)).tocsr()  # Loop-free assembly of global matrix

    Grad_macro = np.zeros((len(z_active), 3, 2))
    Grad_macro[:, 0, 0] = 1.0  # Apply unit temperature gradient along X
    Grad_macro[:, 1, 1] = 1.0  # Apply unit temperature gradient along Y

    F_ele = sum([np.einsum('ji,ejl->eil', gradNs[i], np.einsum('ij,ejl->eil', k_tensor, Grad_macro)) * detJ for i in range(8)])  # Local load via tensor contraction
    F = np.column_stack([np.bincount(edofMat.flatten(), weights=F_ele[:, :, c].flatten(), minlength=total_dofs) for c in range(2)])  # Global load assembly

    active_dofs = np.setdiff1d(np.unique(edofMat), np.unique(edofMat)[:1])  # Eliminate rigid body motions (1 reference temp)
    T_fluc = np.zeros((total_dofs, 2))

    K_active_gpu = cpsp.csr_matrix(K[active_dofs, :][:, active_dofs])
    F_active_gpu = cp.asarray(F[active_dofs, :])
    T_active_gpu = cp.zeros((len(active_dofs), 2))
    M_gpu = cpsp.diags(1.0 / K_active_gpu.diagonal())
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]  # Initialize concurrent CUDA streams

    for c in range(2):
        with streams[c]:  # Launch asynchronous GPU solving for 2 load cases
            T_col_gpu, _ = cpspla.cg(K_active_gpu, F_active_gpu[:, c], M=M_gpu, tol=1e-6, maxiter=5000)  # Jacobi Preconditioner Conjugate gradient solver on GPU
            T_active_gpu[:, c] = T_col_gpu

    for stream in streams:
        stream.synchronize()  # Barrier synchronization

    T_fluc[active_dofs, :] = T_active_gpu.get()

    k_hom = np.zeros((2, 2))
    for i_gp in range(8):
        Flux = np.einsum('ij,ejl->eil', k_tensor, Grad_macro - np.einsum('ij,ejl->eil', gradNs[i_gp], T_fluc[edofMat, :]))  # Recover true heat flux
        k_hom += np.sum(Flux[:, 0:2, :], axis=0) * detJ / plate_area  # Extract in-plane effective thermal conductivity

    return (k_hom + k_hom.T) / 2.0  # Symmetrization