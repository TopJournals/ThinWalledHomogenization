"""Generate an Abaqus input file for a Diamond LPS tension-twist test.

The model is a 3 x 3 x 1 sheet-Diamond lattice-skin plate represented by a
voxel-based hexahedral mesh. Each active voxel is converted directly into one
C3D8 brick element. The boundary condition imposes a macroscopic eps11 strain
by fixing U1 on the x-min face and prescribing U1 on the x-max face. The
remaining degrees of freedom are left free except for two small anchor
constraints used to remove rigid-body motion.

The generated model is intended for Appendix-level visualization of
extension-twist coupling, not as a high-order geometry-conforming benchmark.

Example
-------
python abaqus/generate_diamond_tension_twist_inp.py --resolution 64
abaqus job=diamond_tension_twist_hex_r64 input=Paper/abaqus_diamond_tension_twist_hex_r64/diamond_tension_twist_hex_r64.inp interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tpms_generator import generate_tpms_voxel_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a voxel-hexahedral Abaqus model for Diamond extension-twist visualization.",
    )
    parser.add_argument("--resolution", type=int, default=64, help="Voxel resolution per unit cell.")
    parser.add_argument("--relative-density", type=float, default=0.15, help="Diamond sheet relative density.")
    parser.add_argument("--eps11", type=float, default=0.01, help="Applied macroscopic strain in x direction.")
    parser.add_argument("--cell-size", type=float, default=10.0, help="Unit-cell size in mm.")
    parser.add_argument("--youngs-modulus", type=float, default=1215.0, help="Base Young's modulus in MPa.")
    parser.add_argument("--poisson", type=float, default=0.35, help="Base Poisson's ratio.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Paper" / "abaqus_diamond_tension_twist_hex_r64",
        help="Directory for the generated Abaqus input and metadata files.",
    )
    return parser.parse_args()


def voxel_to_hexes(voxel: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert active voxels to a conforming C3D8 hexahedral mesh."""

    nx, ny, nz = voxel.shape
    lx = 3.0 * cell_size
    ly = 3.0 * cell_size
    lz = 1.0 * cell_size
    dx, dy, dz = lx / nx, ly / ny, lz / nz

    node_id: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[float, float, float]] = []
    elements: list[tuple[int, int, int, int, int, int, int, int]] = []

    def get_node(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        existing = node_id.get(key)
        if existing is not None:
            return existing
        x = i * dx
        y = j * dy
        z = k * dz - 0.5 * lz
        label = len(nodes) + 1
        node_id[key] = label
        nodes.append((x, y, z))
        return label

    for i, j, k in np.argwhere(voxel > 0):
        elements.append(
            (
                get_node(i, j, k),
                get_node(i + 1, j, k),
                get_node(i + 1, j + 1, k),
                get_node(i, j + 1, k),
                get_node(i, j, k + 1),
                get_node(i + 1, j, k + 1),
                get_node(i + 1, j + 1, k + 1),
                get_node(i, j + 1, k + 1),
            )
        )

    return np.asarray(nodes, dtype=float), np.asarray(elements, dtype=int)


def chunks(values: list[int], size: int = 16):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def nearest_node(nodes: np.ndarray, target: tuple[float, float, float]) -> int:
    point = np.asarray(target, dtype=float)
    distances = np.linalg.norm(nodes - point[None, :], axis=1)
    return int(np.argmin(distances)) + 1


def write_set(f, keyword: str, labels: list[int]) -> None:
    f.write(keyword + "\n")
    for group in chunks(labels):
        f.write(", ".join(str(label) for label in group) + "\n")


def write_inp(
    inp_path: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    eps11: float,
    cell_size: float,
    youngs_modulus: float,
    poisson: float,
) -> None:
    lx = 3.0 * cell_size
    ly = 3.0 * cell_size
    lz = 1.0 * cell_size
    tol = min(cell_size / 1000.0, 1.0e-6)

    xmin_nodes = [index + 1 for index, xyz in enumerate(nodes) if abs(xyz[0]) <= tol]
    xmax_nodes = [index + 1 for index, xyz in enumerate(nodes) if abs(xyz[0] - lx) <= tol]
    anchor_yz = nearest_node(nodes, (0.0, 0.0, -0.5 * lz))
    anchor_z = nearest_node(nodes, (0.0, ly, -0.5 * lz))
    loaded_disp = eps11 * lx

    with inp_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("*Heading\n")
        f.write("3x3x1 sheet-Diamond lattice-skin plate under eps11 tension, C3D8 voxel mesh\n")
        f.write("** Generated by generate_diamond_tension_twist_inp.py\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("*Part, name=DIAMOND_LPS\n")
        f.write("*Node\n")
        for label, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"{label}, {x:.8f}, {y:.8f}, {z:.8f}\n")
        f.write("*Element, type=C3D8, elset=SOLID\n")
        for label, conn in enumerate(elements, start=1):
            f.write(
                f"{label}, {conn[0]}, {conn[1]}, {conn[2]}, {conn[3]}, "
                f"{conn[4]}, {conn[5]}, {conn[6]}, {conn[7]}\n"
            )
        write_set(f, "*Nset, nset=XMIN", xmin_nodes)
        write_set(f, "*Nset, nset=XMAX", xmax_nodes)
        f.write("*Nset, nset=ANCHOR_YZ\n")
        f.write(f"{anchor_yz}\n")
        f.write("*Nset, nset=ANCHOR_Z\n")
        f.write(f"{anchor_z}\n")
        f.write("*Solid Section, elset=SOLID, material=BASE\n")
        f.write(",\n")
        f.write("*End Part\n")
        f.write("*Assembly, name=ASSEMBLY\n")
        f.write("*Instance, name=DIAMOND_LPS-1, part=DIAMOND_LPS\n")
        f.write("*End Instance\n")
        f.write("*End Assembly\n")
        f.write("*Material, name=BASE\n")
        f.write("*Elastic\n")
        f.write(f"{youngs_modulus:.8g}, {poisson:.8g}\n")
        f.write("*Step, name=EPS11_TENSION, nlgeom=NO\n")
        f.write("*Static\n")
        f.write("0.1, 1.0, 1e-05, 0.1\n")
        f.write("*Boundary\n")
        f.write("ASSEMBLY.DIAMOND_LPS-1.XMIN, 1, 1, 0.0\n")
        f.write(f"ASSEMBLY.DIAMOND_LPS-1.XMAX, 1, 1, {loaded_disp:.8g}\n")
        f.write("ASSEMBLY.DIAMOND_LPS-1.ANCHOR_YZ, 2, 3, 0.0\n")
        f.write("ASSEMBLY.DIAMOND_LPS-1.ANCHOR_Z, 3, 3, 0.0\n")
        f.write("*Output, field, frequency=1\n")
        f.write("*Node Output\n")
        f.write("U\n")
        f.write("*Element Output, directions=YES\n")
        f.write("S, E, SENER\n")
        f.write("*Output, history, frequency=1\n")
        f.write("*Node Output, nset=ASSEMBLY.DIAMOND_LPS-1.XMAX\n")
        f.write("U1, RF1\n")
        f.write("*End Step\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    voxel = generate_tpms_voxel_grid(
        tpms_type="Diamond",
        Nx=3,
        Ny=3,
        Nz=1,
        resolution=args.resolution,
        relative_density=args.relative_density,
        is_sheet=True,
    )
    nodes, elements = voxel_to_hexes(voxel, args.cell_size)

    inp_path = args.output_dir / "diamond_tension_twist_hex_r64.inp"
    write_inp(
        inp_path,
        nodes,
        elements,
        eps11=args.eps11,
        cell_size=args.cell_size,
        youngs_modulus=args.youngs_modulus,
        poisson=args.poisson,
    )

    metadata_path = args.output_dir / "diamond_tension_twist_metadata.txt"
    with metadata_path.open("w", encoding="utf-8") as f:
        f.write(f"resolution_per_cell = {args.resolution}\n")
        f.write("cells = 3, 3, 1\n")
        f.write(f"relative_density_target = {args.relative_density}\n")
        f.write(f"relative_density_actual = {float(voxel.mean()):.8g}\n")
        f.write(f"nodes = {len(nodes)}\n")
        f.write(f"hex_elements = {len(elements)}\n")
        f.write("element_type = C3D8\n")
        f.write(f"eps11 = {args.eps11}\n")
        f.write(f"cell_size_mm = {args.cell_size}\n")

    print(f"Generated {inp_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"C3D8 elements: {len(elements)}")
    print(f"Actual solid fraction: {float(voxel.mean()):.6f}")


if __name__ == "__main__":
    main()
