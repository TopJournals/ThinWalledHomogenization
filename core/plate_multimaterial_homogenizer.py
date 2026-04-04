import numpy as np
from scipy.sparse import coo_matrix
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla
from utils.tpms_generator import generate_tpms_voxel_grid

def get_multi_material_elasticity(E_array, nu=0.3):
    lam = E_array * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_array / (2 * (1 + nu))
    N = len(E_array)
    C_active = np.zeros((N, 6, 6))
    C_active[:, 0, 1] = C_active[:, 0, 2] = C_active[:, 1, 0] = lam
    C_active[:, 1, 2] = C_active[:, 2, 0] = C_active[:, 2, 1] = lam
    C_active[:, 0, 0] = C_active[:, 1, 1] = C_active[:, 2, 2] = lam + 2 * mu
    C_active[:, 3, 3] = C_active[:, 4, 4] = C_active[:, 5, 5] = mu
    return C_active


def compute_kinematics(dx, dy, dz):
    Bs = []
    nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    detJ = dx * dy * dz / 8.0
    for q in nodes / np.sqrt(3):
        dN = 0.125 * np.array([(1 + q[1] * nodes[:, 1]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 0] * (2 / dx), (1 + q[0] * nodes[:, 0]) * (1 + q[2] * nodes[:, 2]) * nodes[:, 1] * (2 / dy), (1 + q[0] * nodes[:, 0]) * (1 + q[1] * nodes[:, 1]) * nodes[:, 2] * (2 / dz)])
        B = np.zeros((6, 24))
        B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[0], dN[1], dN[2]
        B[3, 1::3], B[3, 2::3] = dN[2], dN[1]
        B[4, 0::3], B[4, 2::3] = dN[2], dN[0]
        B[5, 0::3], B[5, 1::3] = dN[1], dN[0]
        Bs.append(B)
    return Bs, detJ


def build_tensor_dof_mapping(voxel, dx, dy, dz, thickness):
    nx, ny, nz = voxel.shape
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

    active_mask = voxel > 0
    edofMat = edof_tensor[active_mask]
    z_coords = np.linspace(dz / 2, thickness - dz / 2, nz) - thickness / 2.0
    z_grid = np.broadcast_to(z_coords[None, None, :], (nx, ny, nz))
    z_active = z_grid[active_mask]
    return edofMat, z_active, node_indices.size * 3


def homogenization_plate(voxel, nu=0.3, thickness=10.0, Nx=1, Ny=1, Nz=1):
    nz, ny, nx = voxel.shape
    cell_size = thickness / Nz
    Lx, Ly, Lz = Nx * cell_size, Ny * cell_size, thickness
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    plate_area = Lx * Ly
    edofMat, z_active, total_dofs = build_tensor_dof_mapping(voxel, dx, dy, dz, thickness)
    Bs, detJ = compute_kinematics(dx, dy, dz)

    E_active = np.where(z_active > 0, 500.0, 1215.0)  # New add
    C_active = get_multi_material_elasticity(E_active, nu)  # Shape: (N_active, 6, 6)
    Ke_active = np.zeros((len(edofMat), 24, 24))
    for i in range(8):
        Ke_active += np.einsum('ji,ekj,kl->eil', Bs[i], C_active, Bs[i]) * detJ  # 'ji' (B.T) * 'ekj' (C_active) * 'kl' (B) -> 'eil'
    iK = np.repeat(edofMat, 24, axis=1).flatten()
    jK = np.tile(edofMat, (1, 24)).flatten()
    sK = Ke_active.flatten()
    K = coo_matrix((sK, (iK, jK)), shape=(total_dofs, total_dofs)).tocsr()

    E_macro = np.zeros((len(z_active), 6, 6))
    E_macro[:, 0, 0] = E_macro[:, 1, 1] = E_macro[:, 5, 2] = 1.0
    E_macro[:, 0, 3] = E_macro[:, 1, 4] = E_macro[:, 5, 5] = z_active
    F_ele = sum([np.einsum('ji,ejl->eil', Bs[i], np.einsum('eij,ejl->eil', C_active, E_macro)) * detJ for i in range(8)])
    F = np.column_stack([np.bincount(edofMat.flatten(), weights=F_ele[:, :, c].flatten(), minlength=total_dofs) for c in range(6)])
    active_dofs = np.setdiff1d(np.unique(edofMat), np.unique(edofMat)[:3])
    U = np.zeros((total_dofs, 6))
    K_active_gpu = cpsp.csr_matrix(K[active_dofs, :][:, active_dofs])
    F_active_gpu = cp.asarray(F[active_dofs, :])
    U_active_gpu = cp.zeros((len(active_dofs), 6))
    M_gpu = cpsp.diags(1.0 / K_active_gpu.diagonal())
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(6)]
    for c in range(6):
        with streams[c]:
            U_col_gpu, info = cpspla.cg(K_active_gpu, F_active_gpu[:, c], M=M_gpu, tol=1e-6, maxiter=5000)
            if info != 0:
                print(f"      [Warning] PCG un-converged for load case {c}. Info: {info}")
            U_active_gpu[:, c] = U_col_gpu
    for stream in streams:
        stream.synchronize()
    U[active_dofs, :] = U_active_gpu.get()

    ABD = np.zeros((6, 6))
    for i_gp in range(8):
        # 使用局部 C_active 恢复应力
        Sigma = np.einsum('eij,ejl->eil', C_active, E_macro - np.einsum('ij,ejl->eil', Bs[i_gp], U[edofMat, :]))

        ABD[0:3, :] += np.sum(Sigma[:, [0, 1, 5], :], axis=0) * detJ / plate_area
        ABD[3:6, :] += np.sum(Sigma[:, [0, 1, 5], :] * z_active[:, None, None], axis=0) * detJ / plate_area

    return (ABD + ABD.T) / 2.0


