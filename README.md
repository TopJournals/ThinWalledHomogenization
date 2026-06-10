# ThinWalledHomogenization

Open-source Python implementation for the manuscript **"A 99-Line Homogenization Code for Lattice-skin Plate Structures"** ([arXiv:2604.23181](https://arxiv.org/abs/2604.23181)).

This repository implements a GPU-accelerated **LPS-H** framework, namely homogenization for finite-thickness **Lattice-skin Plate Structures**. It extracts equivalent plate/shell stiffness matrices directly from voxelized TPMS and lattice microstructures while retaining the free-surface effect in the thickness direction.

The core motivation is simple: conventional lattice-filled volume homogenization (**LVS-H**) imposes three-dimensional periodic boundary conditions on the representative cell. This is suitable for infinitely periodic bulk materials, but it can bias the stiffness prediction of thin plates because the upper and lower surfaces are physically free. LPS-H instead applies periodicity only in the in-plane directions and leaves the thickness direction free, making it more appropriate for finite-thickness metamaterial plates/shells.

## Paper Context

The manuscript studies the dimensional-reduction analysis of Lattice-skin Plate Structures used in lightweight aerospace and automotive applications. The proposed open-source framework:

- computes the equivalent plate/shell `ABD` stiffness matrix for thin-walled lattice/TPMS plates;
- captures thickness size effects caused by finite numbers of unit cells through the thickness;
- preserves extension-bending and extension-twist coupling terms that may be missed by volume homogenization;
- uses loop-free tensorized finite-element assembly with `NumPy`/`SciPy`;
- accelerates the repeated microscopic cell problems with `CuPy` and CUDA streams;
- extends naturally to multimaterial plates and steady-state thermal conduction.

In the benchmark reported in the manuscript, a full-scale finite element model gives a support reaction of `490.794 N`; LPS-H predicts `541.296 N` with `10.29%` relative deviation, while LVS-H predicts `610.729 N` with `24.43%` deviation. This illustrates why retaining the free surfaces through the thickness matters for finite-thickness Lattice-skin Plate Structures.

## Method Overview

LPS-H uses a mixed boundary condition:

- **2D-PBC** in the in-plane `x` and `y` directions;
- **1D-FBC** in the out-of-plane `z` thickness direction.

For each voxelized representative plate cell, the solver applies six macroscopic generalized unit deformation modes:

```text
[eps_11^0, eps_22^0, gamma_12^0, kappa_11, kappa_22, kappa_12]
```

The microscopic fluctuation displacement fields are solved by finite elements. The local stress field is then integrated over the projected plate area to obtain the generalized force/moment response:

```text
[N_11, N_22, N_12, M_11, M_22, M_12]
```

Column-by-column probing of these six modes produces the equivalent `6 x 6` plate stiffness matrix:

```text
[ N ]   [ A  B ] [ eps^0 ]
[ M ] = [ B  D ] [ kappa ]
```

## Features

- Voxel-based finite-element homogenization for finite-thickness plate/shell metamaterials.
- Direct extraction of the mechanical `ABD` matrix.
- TPMS voxel generation from analytical level-set equations.
- BCC lattice voxel generation from distance fields.
- Dense-skin modeling through simple voxel padding.
- Multimaterial extension with element-wise constitutive tensors.
- Thermal extension for in-plane effective conductivity.
- Loop-free local-to-global DOF mapping based on multidimensional tensor slicing.
- Loop-free sparse matrix/load assembly using `np.repeat`, `np.tile`, `np.bincount`, and `np.einsum`.
- GPU PCG solving with CuPy, Jacobi preconditioning, and non-blocking CUDA streams.

## Repository Layout

```text
ThinWalledHomogenization/
+-- core/
|   +-- plate_homogenizer.py                 # Main LPS-H mechanical ABD solver
|   +-- plate_multimaterial_homogenizer.py   # Multimaterial ABD extension
|   +-- plate_thermal_homogenizer.py         # In-plane thermal homogenization
+-- utils/
|   +-- tpms_generator.py                    # TPMS voxel generator
|   +-- lattice_generator.py                 # BCC lattice voxel generator
+-- examples/
|   +-- ex01_tpms_simulation.py              # Primitive TPMS base case
|   +-- ex01_2_lattice_simulation.py         # BCC lattice base case
|   +-- ex02_multiple_cells.py               # Thickness size-effect example
|   +-- ex03_multimaterial_simulation.py     # Bimaterial extension
|   +-- ex04_tpms_thermal_simulation.py      # Thermal conduction extension
+-- benchmarks/
|   +-- benchmark_plate_solver.py            # Optional CPU/GPU timing and memory benchmark
+-- Paper/                                   # Figure notebooks, heatmaps, VTU outputs
+-- Test/                                    # Experimental scripts and generated files
+-- requirements.txt                         # Dependency list
```

## Requirements

The current implementation uses CuPy and therefore expects an NVIDIA GPU with a CUDA runtime compatible with `cupy-cuda12x`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

The dependency file currently contains:

```text
cupy-cuda12x
scipy
```

`numpy` is also used throughout the project and is normally installed as a dependency of SciPy/CuPy.

If only a CPU environment is available, the GPU solver section in `core/plate_homogenizer.py` can be replaced by `scipy.sparse.linalg.cg`, as described in the manuscript. The repository version is currently GPU-oriented.

## Quick Start

Run the base TPMS example:

```bash
python examples/ex01_tpms_simulation.py
```

This example uses the paper's representative parameters:

```text
Topology:          sheet-type Primitive TPMS
Base material:     E = 1215 MPa, nu = 0.35
Plate thickness:   h = 10 mm
Relative density:  0.15
Resolution:        96 voxels per unit cell
Cell array:        1 x 1 x 1
```

Minimal Python usage:

```python
import numpy as np

from utils.tpms_generator import generate_tpms_voxel_grid
from core.plate_homogenizer import homogenization_plate

voxel_grid = generate_tpms_voxel_grid(
    tpms_type="Primitive",
    Nx=1,
    Ny=1,
    Nz=1,
    resolution=96,
    relative_density=0.15,
    is_sheet=True,
)

ABD = homogenization_plate(
    voxel=voxel_grid,
    E=1215.0,
    nu=0.35,
    thickness=10.0,
    Nx=1,
    Ny=1,
    Nz=1,
)

np.set_printoptions(precision=2, suppress=True, linewidth=120)
print(ABD)
```

## Optional CPU/GPU Benchmark

The repository includes an optional benchmark script for collecting the CPU/GPU
timing and memory data used to assess the practical cost of the LPS-H solver.
The script reuses the same TPMS generator and finite-element assembly routines
as the main homogenization example.

```bash
python benchmarks/benchmark_plate_solver.py --resolutions 32 40 48 56 64 --output benchmarks/benchmark_plate_solver_results.csv
```

The CSV output reports the resolution, active elements, total and active DOFs,
sparse matrix nonzeros, geometry and assembly time, CPU solve time, GPU solve
time, GPU memory use, and CPU/GPU speedup. Larger cases can skip CPU solves:

```bash
python benchmarks/benchmark_plate_solver.py --resolutions 40 50 60 70 80 96 --skip-cpu-above 70
```

## ABD Matrix

The mechanical solver returns:

```text
[ A11 A12 A16 B11 B12 B16 ]
[ A12 A22 A26 B12 B22 B26 ]
[ A16 A26 A66 B16 B26 B66 ]
[ B11 B12 B16 D11 D12 D16 ]
[ B12 B22 B26 D12 D22 D26 ]
[ B16 B26 B66 D16 D26 D66 ]
```

Physical meaning:

- `A`: membrane/stretching stiffness, relating in-plane force resultants to mid-surface strains.
- `B`: extension-bending and extension-twist coupling stiffness.
- `D`: bending and twisting stiffness, relating bending moments to curvatures.

For a symmetric plate, the `B` block should be close to zero. For finite-thickness, truncated, or multimaterial Lattice-skin Plate Structures, nonzero `B` terms indicate coupling between in-plane deformation and out-of-plane bending/twisting.

## Geometry Generation

### TPMS Structures

Use:

```python
from utils.tpms_generator import generate_tpms_voxel_grid
```

Supported TPMS types:

- `Primitive`
- `Diamond`
- `Gyroid`
- `I-WP`
- `F-RD`
- `L`
- `Tubular P`
- `Tubular G`
- `I2-Y`

Important parameters:

- `Nx, Ny, Nz`: number of unit cells in each direction;
- `resolution`: voxel count per unit cell along one axis;
- `relative_density`: target solid volume fraction;
- `is_sheet=True`: sheet TPMS generated by thresholding `abs(field)`;
- `is_sheet=False`: network-like TPMS generated by thresholding `field`.

The generated voxel grid follows the Cartesian `(x, y, z)` indexing convention. Solid voxels are marked as `1`; void voxels are marked as `0`.

### BCC Lattice

Use:

```python
from utils.lattice_generator import generate_lattice_voxel_grid
```

Currently supported:

- `BCC`

The BCC generator evaluates a distance field to the truss skeleton and thresholds it according to the target relative density.

## Modeling Dense Skins

Lattice-skin Plate Structures often include dense top and bottom skins. In the paper examples, skins are added by padding the voxel grid in the thickness direction:

```python
voxel_grid = np.pad(
    voxel_grid,
    pad_width=((0, 0), (0, 0), (2, 2)),
    mode="constant",
    constant_values=1,
)
```

For the Primitive TPMS case reported in the manuscript, adding skins increases `A11` from about `356.12 MPa*mm` to `973.08 MPa*mm`, and increases `D11` from about `2229.5 MPa*mm^3` to `16538.85 MPa*mm^3`.

## Example Scripts

```bash
# Base sheet-type Primitive TPMS ABD calculation
python examples/ex01_tpms_simulation.py

# BCC lattice ABD calculation
python examples/ex01_2_lattice_simulation.py

# Thickness size-effect analysis, e.g. Gyroid with multiple cells through thickness
python examples/ex02_multiple_cells.py

# Bimaterial Lattice-skin Plate Structure
python examples/ex03_multimaterial_simulation.py

# In-plane thermal conductivity homogenization
python examples/ex04_tpms_thermal_simulation.py
```

## Numerical Notes from the Manuscript

- In mesh convergence tests for the sheet-type Primitive structure, results become stable when the one-dimensional unit-cell resolution exceeds about `N = 60`.
- A practical resolution range of `N = 60 ~ 65` is recommended for balancing accuracy and GPU cost.
- In the paper's base example, `N = 96` is used for demonstration.
- For a single-layer Gyroid plate, LPS-H predicts clear in-plane orthotropy caused by physical surface truncation, while LVS-H tends to preserve the higher symmetry implied by 3D periodicity.
- For the Diamond TPMS, nonzero `B16` and `B26` terms show extension-twist coupling that can be suppressed or missed by a fully 3D periodic LVS-H treatment.
- As the number of cells through the thickness increases, LPS-H results asymptotically approach the LVS-H benchmark, which is consistent with the decay of finite-thickness surface effects.

## Thermal Homogenization

The thermal extension follows the same LPS-H assembly idea, replacing:

```text
mechanical displacement u       -> temperature T
strain epsilon                  -> temperature gradient grad(T)
stress sigma                    -> heat flux q
elastic stiffness C             -> thermal conductivity tensor k
```

Run:

```bash
python examples/ex04_tpms_thermal_simulation.py
```

The thermal solver returns a symmetric `2 x 2` in-plane effective conductivity matrix:

```text
[ k_xx k_xy ]
[ k_xy k_yy ]
```

For the paper's Primitive TPMS thermal case with `k_s = 60.5 W/(m*K)`, the reported homogenized matrix is approximately:

```text
[ 60.19  0.00 ]
[  0.00 60.19 ]
```

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{ji2026lpsh,
  title   = {A 99-Line Homogenization Code for Lattice-skin Plate Structures},
  author  = {Ji, Zhongkai and Li, Dawei and Zhao, Yong and Liao, Wenhe},
  year    = {2026},
  journal = {arXiv preprint arXiv:2604.23181},
  url     = {https://arxiv.org/abs/2604.23181}
}
```

Please update the BibTeX entry with journal, volume, pages, and DOI once the paper is published.

## Acknowledgements

The computations reported in the manuscript were run on the Siyuan1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University.

## Data Availability

The manuscript points to this repository as the open-source code and data location:

```text
https://github.com/TopJournals/ThinWalledHomogenization
```
