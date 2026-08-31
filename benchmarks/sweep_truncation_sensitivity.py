"""Run thickness-truncation sensitivity sweeps for LPS-H.

The script reuses the repository geometry generators and the core
``homogenization_plate`` solver. It generates six separate sensitivity figures:

1. lattice topology versus thickness-wise unit-cell count,
2. relative density versus thickness-wise unit-cell count,
3. skin thickness versus thickness-wise unit-cell count,

with LVS-H-referenced errors in ``A11`` and ``D11`` shown separately for each
parameter group.
Results are cached in a CSV file so interrupted sweeps can be resumed without
recomputing completed cases.

Example
-------
python benchmarks/sweep_truncation_sensitivity.py --resolution 48 --nz-values 1 2 3 4 5 6 8 10 12
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla
from scipy.sparse import coo_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plate_homogenizer import (  # noqa: E402
    compute_element_stiffness,
    get_isotropic_elasticity,
    homogenization_plate,
)
from utils.lattice_generator import generate_lattice_voxel_grid  # noqa: E402
from utils.tpms_generator import generate_tpms_voxel_grid  # noqa: E402


TPMS_TYPES = {
    "Primitive",
    "Diamond",
    "Gyroid",
    "I-WP",
    "F-RD",
    "L",
    "Tubular P",
    "Tubular G",
    "I2-Y",
}
LATTICE_TYPES = {"BCC"}


@dataclass(frozen=True)
class SweepCase:
    group: str
    label: str
    topology: str
    relative_density: float
    skin_ratio: float
    nz: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep thickness-truncation sensitivity for LPS-H ABD entries.",
    )
    parser.add_argument("--resolution", type=int, default=48, help="Voxel resolution per unit cell.")
    parser.add_argument(
        "--nz-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 10, 12],
        help="Thickness-wise unit-cell counts to evaluate.",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["Gyroid", "Diamond", "Primitive", "I-WP", "BCC"],
        help="Topologies for the topology-sensitivity sweep.",
    )
    parser.add_argument(
        "--density-topology",
        default="Gyroid",
        help="Topology used for the relative-density sensitivity sweep.",
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20],
        help="Relative densities for the density-sensitivity sweep.",
    )
    parser.add_argument(
        "--skin-topology",
        default="Gyroid",
        help="Topology used for the skin-thickness sensitivity sweep.",
    )
    parser.add_argument(
        "--skin-density",
        type=float,
        default=0.15,
        help="Core relative density used for the skin-thickness sweep.",
    )
    parser.add_argument(
        "--skin-ratios",
        type=float,
        nargs="+",
        default=[0.00, 0.05, 0.10],
        help=(
            "Single-side skin thickness ratios relative to the final plate thickness. "
            "For example, 0.05 means each solid skin is about 5 percent of the plate thickness."
        ),
    )
    parser.add_argument("--youngs-modulus", type=float, default=1215.0, help="Base Young's modulus in MPa.")
    parser.add_argument("--poisson", type=float, default=0.35, help="Base Poisson's ratio.")
    parser.add_argument("--thickness", type=float, default=10.0, help="Macroscopic plate thickness in mm.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "Paper" / "truncation_sensitivity_sweep.csv",
        help="CSV file used for cached sweep results.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "Paper" / "fig_truncation_sensitivity",
        help="Output figure prefix. Six PNG and six SVG files are generated.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=PROJECT_ROOT / "Paper" / "truncation_sensitivity_lvs_references.csv",
        help="Cached three-dimensional periodic LVS-H reference stiffnesses.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "Paper" / "truncation_sensitivity_lvs_summary.csv",
        help="Summary of absolute stiffnesses, errors, and achieved voxel fractions.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=PROJECT_ROOT / "Paper" / "fig_analysis_lvs_error",
        help="Combined six-panel output path without an extension.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation and regenerate figures from the existing CSV file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cases even if matching rows already exist in the CSV file.",
    )
    return parser.parse_args()


def build_cases(args: argparse.Namespace) -> list[SweepCase]:
    nz_values = sorted(set(args.nz_values))
    cases: list[SweepCase] = []

    for topology in args.topologies:
        for nz in nz_values:
            cases.append(
                SweepCase(
                    group="topology",
                    label=topology,
                    topology=topology,
                    relative_density=0.15,
                    skin_ratio=0.0,
                    nz=nz,
                )
            )

    for density in args.densities:
        label = rf"$\rho={density:.2f}$"
        for nz in nz_values:
            cases.append(
                SweepCase(
                    group="density",
                    label=label,
                    topology=args.density_topology,
                    relative_density=density,
                    skin_ratio=0.0,
                    nz=nz,
                )
            )

    for skin_ratio in args.skin_ratios:
        label = rf"$t_s/h={skin_ratio:.2f}$"
        for nz in nz_values:
            cases.append(
                SweepCase(
                    group="skin",
                    label=label,
                    topology=args.skin_topology,
                    relative_density=args.skin_density,
                    skin_ratio=skin_ratio,
                    nz=nz,
                )
            )

    return cases


def case_key(case: SweepCase, resolution: int) -> tuple[str, str, str, str, str, str]:
    return (
        case.group,
        case.topology,
        f"{case.relative_density:.6g}",
        f"{case.skin_ratio:.6g}",
        str(case.nz),
        str(resolution),
    )


def load_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def existing_keys(rows: Iterable[dict[str, str]]) -> set[tuple[str, str, str, str, str, str]]:
    keys = set()
    for row in rows:
        keys.add(
            (
                row["group"],
                row["topology"],
                f"{float(row['relative_density']):.6g}",
                f"{float(row['requested_skin_ratio']):.6g}",
                row["nz"],
                row["resolution"],
            )
        )
    return keys


def generate_core_voxel(case: SweepCase, resolution: int) -> np.ndarray:
    if case.topology in TPMS_TYPES:
        return generate_tpms_voxel_grid(
            tpms_type=case.topology,
            Nx=1,
            Ny=1,
            Nz=case.nz,
            resolution=resolution,
            relative_density=case.relative_density,
            is_sheet=True,
        )
    if case.topology in LATTICE_TYPES:
        return generate_lattice_voxel_grid(
            lattice_type=case.topology,
            Nx=1,
            Ny=1,
            Nz=case.nz,
            resolution=resolution,
            relative_density=case.relative_density,
        )
    raise ValueError(f"Unsupported topology: {case.topology}")


def add_solid_skins(voxel: np.ndarray, requested_skin_ratio: float) -> tuple[np.ndarray, int, float]:
    """Pad solid skins on both free surfaces.

    ``requested_skin_ratio`` is interpreted as the target thickness ratio of
    each skin relative to the final plate thickness. If the core has ``n`` z
    voxels and each side receives ``p`` solid voxels, the actual ratio is
    ``p / (n + 2p)``. The integer ``p`` is chosen to approximate the requested
    value.
    """

    if requested_skin_ratio <= 0:
        return voxel, 0, 0.0
    if requested_skin_ratio >= 0.5:
        raise ValueError("Each-side skin ratio must be smaller than 0.5.")

    core_nz = voxel.shape[2]
    pad_layers = max(1, int(round(requested_skin_ratio * core_nz / (1.0 - 2.0 * requested_skin_ratio))))
    padded = np.pad(
        voxel,
        pad_width=((0, 0), (0, 0), (pad_layers, pad_layers)),
        mode="constant",
        constant_values=1,
    )
    actual_ratio = pad_layers / padded.shape[2]
    return padded, pad_layers, actual_ratio


def build_volume_edof(voxel: np.ndarray) -> tuple[np.ndarray, int]:
    """Map all three pairs of opposite faces to enforce 3D periodicity."""

    nx, ny, nz = voxel.shape
    nodes = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    ix = np.arange(nx + 1) % nx
    iy = np.arange(ny + 1) % ny
    iz = np.arange(nz + 1) % nz
    periodic_nodes = nodes[np.ix_(ix, iy, iz)]
    dofs = np.stack((3 * periodic_nodes, 3 * periodic_nodes + 1, 3 * periodic_nodes + 2), axis=-1)
    n1, n2 = dofs[:-1, :-1, :-1], dofs[1:, :-1, :-1]
    n3, n4 = dofs[1:, 1:, :-1], dofs[:-1, 1:, :-1]
    n5, n6 = dofs[:-1, :-1, 1:], dofs[1:, :-1, 1:]
    n7, n8 = dofs[1:, 1:, 1:], dofs[:-1, 1:, 1:]
    edof = np.concatenate((n1, n2, n3, n4, n5, n6, n7, n8), axis=-1)
    return edof[voxel > 0], 3 * nx * ny * nz


def homogenization_volume(voxel: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Compute the 3D periodic effective tensor used as the LVS-H reference."""

    nx, ny, nz = voxel.shape
    C = get_isotropic_elasticity(E, nu)
    Ke, Bs, detJ = compute_element_stiffness(C, 1.0 / nx, 1.0 / ny, 1.0 / nz)
    edof, total_dofs = build_volume_edof(voxel)
    iK = np.repeat(edof, 24, axis=1).ravel()
    jK = np.tile(edof, (1, 24)).ravel()
    K = coo_matrix(
        (np.tile(Ke.ravel(), len(edof)), (iK, jK)),
        shape=(total_dofs, total_dofs),
    ).tocsr()

    macro = np.broadcast_to(np.eye(6), (len(edof), 6, 6)).copy()
    F_ele = sum(
        np.einsum("ji,kjl->kil", B, np.einsum("ij,kjl->kil", C, macro)) * detJ
        for B in Bs
    )
    F = np.column_stack(
        [
            np.bincount(edof.ravel(), weights=F_ele[:, :, c].ravel(), minlength=total_dofs)
            for c in range(6)
        ]
    )
    active = np.setdiff1d(np.unique(edof), np.unique(edof)[:3])
    K_gpu = cpsp.csr_matrix(K[active][:, active])
    F_gpu = cp.asarray(F[active])
    M_gpu = cpsp.diags(1.0 / K_gpu.diagonal())
    U_gpu = cp.zeros((len(active), 6))
    for c in range(6):
        U_gpu[:, c], info = cpspla.cg(K_gpu, F_gpu[:, c], M=M_gpu, tol=1e-6, maxiter=5000)
        if info != 0:
            raise RuntimeError(f"LVS-H load case {c} did not converge: info={info}")
    U = np.zeros((total_dofs, 6))
    U[active] = U_gpu.get()

    CH = np.zeros((6, 6))
    for B in Bs:
        stress = np.einsum(
            "ij,kjl->kil",
            C,
            macro - np.einsum("ij,kjl->kil", B, U[edof]),
        )
        CH += stress.sum(axis=0) * detJ
    return (CH + CH.T) / 2.0


def plane_stress_reduction(CH: np.ndarray) -> np.ndarray:
    in_plane = [0, 1, 5]
    out_plane = [2, 3, 4]
    return CH[np.ix_(in_plane, in_plane)] - CH[np.ix_(in_plane, out_plane)] @ np.linalg.solve(
        CH[np.ix_(out_plane, out_plane)],
        CH[np.ix_(out_plane, in_plane)],
    )


def validate_volume_homogenizer(E: float, nu: float) -> None:
    Q = plane_stress_reduction(homogenization_volume(np.ones((4, 4, 4), dtype=np.int8), E, nu))
    expected = E / (1.0 - nu**2)
    if not np.isclose(Q[0, 0], expected, rtol=1e-10, atol=1e-8):
        raise RuntimeError("The LVS-H implementation failed the homogeneous isotropic self-check.")


def load_lvs_references(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_lvs_references(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["topology", "relative_density", "resolution", "active_fraction", "Q11"],
        )
        writer.writeheader()
        writer.writerows(rows)


def lvs_key(topology: str, relative_density: float, resolution: int) -> tuple[str, str, str]:
    return topology, f"{relative_density:.8g}", str(resolution)


def ensure_lvs_references(
    records: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = load_lvs_references(args.reference_csv)
    done = {
        lvs_key(row["topology"], float(row["relative_density"]), int(row["resolution"]))
        for row in rows
    }
    requested = sorted(
        {
            (str(record["topology"]), float(record["relative_density"]), int(record["resolution"]))
            for record in records
        }
    )
    for topology, density, resolution in requested:
        key = lvs_key(topology, density, resolution)
        if key in done:
            continue
        case = SweepCase("reference", topology, topology, density, 0.0, 1)
        voxel = generate_core_voxel(case, resolution)
        Q = plane_stress_reduction(homogenization_volume(voxel, args.youngs_modulus, args.poisson))
        rows.append(
            {
                "topology": topology,
                "relative_density": f"{density:.8g}",
                "resolution": str(resolution),
                "active_fraction": f"{float(voxel.mean()):.8g}",
                "Q11": f"{float(Q[0, 0]):.12g}",
            }
        )
        write_lvs_references(args.reference_csv, rows)
        print(f"LVS-H reference {topology}, rho={density:.3f}: Q11={Q[0, 0]:.6g}")
    return {
        lvs_key(row["topology"], float(row["relative_density"]), int(row["resolution"])): {
            "active_fraction": float(row["active_fraction"]),
            "Q11": float(row["Q11"]),
        }
        for row in load_lvs_references(args.reference_csv)
    }


def run_case(case: SweepCase, args: argparse.Namespace) -> dict[str, str]:
    start = time.perf_counter()
    voxel_core = generate_core_voxel(case, args.resolution)
    voxel, skin_layers, actual_skin_ratio = add_solid_skins(voxel_core, case.skin_ratio)
    geometry_time = time.perf_counter() - start

    solve_start = time.perf_counter()
    abd = homogenization_plate(
        voxel=voxel,
        E=args.youngs_modulus,
        nu=args.poisson,
        thickness=args.thickness,
        Nx=1,
        Ny=1,
        Nz=case.nz,
    )
    solve_time = time.perf_counter() - solve_start
    total_time = time.perf_counter() - start

    return {
        "group": case.group,
        "label": case.label,
        "topology": case.topology,
        "relative_density": f"{case.relative_density:.8g}",
        "requested_skin_ratio": f"{case.skin_ratio:.8g}",
        "actual_skin_ratio": f"{actual_skin_ratio:.8g}",
        "skin_layers_each_side": str(skin_layers),
        "nz": str(case.nz),
        "resolution": str(args.resolution),
        "voxel_nx": str(voxel.shape[0]),
        "voxel_ny": str(voxel.shape[1]),
        "voxel_nz": str(voxel.shape[2]),
        "active_fraction": f"{float(voxel.mean()):.8g}",
        "A11": f"{float(abd[0, 0]):.12g}",
        "D11": f"{float(abd[3, 3]):.12g}",
        "geometry_time_s": f"{geometry_time:.6g}",
        "solve_time_s": f"{solve_time:.6g}",
        "total_time_s": f"{total_time:.6g}",
    }


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "label",
        "topology",
        "relative_density",
        "requested_skin_ratio",
        "actual_skin_ratio",
        "skin_layers_each_side",
        "nz",
        "resolution",
        "voxel_nx",
        "voxel_ny",
        "voxel_nz",
        "active_fraction",
        "A11",
        "D11",
        "geometry_time_s",
        "solve_time_s",
        "total_time_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
            "text.usetex": False,
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 8,
            "mathtext.fontset": "stix",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def rows_to_records(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        record: dict[str, object] = dict(row)
        for key in [
            "relative_density",
            "requested_skin_ratio",
            "actual_skin_ratio",
            "active_fraction",
            "A11",
            "D11",
            "geometry_time_s",
            "solve_time_s",
            "total_time_s",
        ]:
            record[key] = float(row[key])
        for key in ["skin_layers_each_side", "nz", "resolution", "voxel_nx", "voxel_ny", "voxel_nz"]:
            record[key] = int(row[key])
        records.append(record)
    return records


def attach_lvs_references(
    records: list[dict[str, object]],
    references: dict[tuple[str, str, str], dict[str, float]],
    args: argparse.Namespace,
) -> None:
    solid_q11 = args.youngs_modulus / (1.0 - args.poisson**2)
    h = args.thickness
    for record in records:
        key = lvs_key(
            str(record["topology"]),
            float(record["relative_density"]),
            int(record["resolution"]),
        )
        core_q11 = references[key]["Q11"]
        skin_ratio = float(record["actual_skin_ratio"])
        skin_thickness = skin_ratio * h
        core_thickness = h - 2.0 * skin_thickness
        a11_lvs = core_q11 * core_thickness + 2.0 * solid_q11 * skin_thickness
        d11_lvs = (
            core_q11 * core_thickness**3 / 12.0
            + 2.0
            * solid_q11
            * ((h / 2.0) ** 3 - (core_thickness / 2.0) ** 3)
            / 3.0
        )
        record["lvs_core_active_fraction"] = references[key]["active_fraction"]
        record["A11_lvs"] = a11_lvs
        record["D11_lvs"] = d11_lvs
        record["A11_error_pct"] = 100.0 * (float(record["A11"]) - a11_lvs) / a11_lvs
        record["D11_error_pct"] = 100.0 * (float(record["D11"]) - d11_lvs) / d11_lvs


def write_summary(records: list[dict[str, object]], path: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault((str(record["group"]), str(record["label"])), []).append(record)
    rows: list[dict[str, str]] = []
    group_order = {"density": 0, "topology": 1, "skin": 2}
    for (group, label), values in sorted(groups.items(), key=lambda item: (group_order[item[0][0]], item[0][1])):
        values.sort(key=lambda item: int(item["nz"]))
        first, last = values[0], values[-1]
        rows.append(
            {
                "group": group,
                "label": label,
                "topology": str(first["topology"]),
                "target_core_density": f"{float(first['relative_density']):.6f}",
                "achieved_voxel_fraction_min": f"{min(float(v['active_fraction']) for v in values):.6f}",
                "achieved_voxel_fraction_max": f"{max(float(v['active_fraction']) for v in values):.6f}",
                "actual_skin_ratio_min": f"{min(float(v['actual_skin_ratio']) for v in values):.6f}",
                "actual_skin_ratio_max": f"{max(float(v['actual_skin_ratio']) for v in values):.6f}",
                "A11_Nz1": f"{float(first['A11']):.9g}",
                "A11_LVS_Nz1": f"{float(first['A11_lvs']):.9g}",
                "A11_error_Nz1_pct": f"{float(first['A11_error_pct']):.6f}",
                "D11_Nz1": f"{float(first['D11']):.9g}",
                "D11_LVS_Nz1": f"{float(first['D11_lvs']):.9g}",
                "D11_error_Nz1_pct": f"{float(first['D11_error_pct']):.6f}",
                "last_nz": str(int(last["nz"])),
                "A11_error_last_pct": f"{float(last['A11_error_pct']):.6f}",
                "D11_error_last_pct": f"{float(last['D11_error_pct']):.6f}",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


PANEL_INFO = [
    ("density", "$A_{11}$", "A11_error_pct", "Relative density", "density_A11"),
    ("density", "$D_{11}$", "D11_error_pct", "Relative density", "density_D11"),
    ("topology", "$A_{11}$", "A11_error_pct", "Lattice topology", "topology_A11"),
    ("topology", "$D_{11}$", "D11_error_pct", "Lattice topology", "topology_D11"),
    ("skin", "$A_{11}$", "A11_error_pct", "Skin thickness", "skin_A11"),
    ("skin", "$D_{11}$", "D11_error_pct", "Skin thickness", "skin_D11"),
]
COLORS = ["#3A5BA0", "#2A9D8F", "#E76F51", "#F2B84B", "#7E6BC4", "#6C757D"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def draw_panel(
    ax: plt.Axes,
    records: list[dict[str, object]],
    group: str,
    stiffness: str,
    value_key: str,
    title: str,
) -> None:
    subset = [record for record in records if record["group"] == group]
    labels = list(dict.fromkeys(str(record["label"]) for record in subset))
    all_x: list[int] = []
    for idx, label in enumerate(labels):
        line = sorted((r for r in subset if r["label"] == label), key=lambda r: int(r["nz"]))
        x = [int(r["nz"]) for r in line]
        y = [float(r[value_key]) for r in line]
        all_x.extend(x)
        ax.plot(
            x,
            y,
            color=COLORS[idx % len(COLORS)],
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=1.2,
            markersize=4.0,
            label=label,
        )
    ax.axhline(0.0, color="#6E6E6E", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.set_xlabel("Cell count in thickness ($N_z$)")
    ax.set_ylabel(f"Relative error in {stiffness} (\%)")
    ax.set_xticks(sorted(set(all_x)))
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
    ax.grid(False)
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="best",
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def plot_sensitivity(
    records: list[dict[str, object]],
    output_prefix: Path,
    combined_output: Path,
) -> None:
    configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined_output.parent.mkdir(parents=True, exist_ok=True)

    for group, stiffness, value_key, title, suffix in PANEL_INFO:
        fig, ax = plt.subplots(figsize=(8.0 / 2.54, 6.0 / 2.54))
        draw_panel(ax, records, group, stiffness, value_key, title)
        fig.tight_layout()
        base = output_prefix.with_name(f"{output_prefix.name}_{suffix}")
        fig.savefig(base.with_suffix(".png"), dpi=600)
        fig.savefig(base.with_suffix(".svg"))
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(16.8 / 2.54, 18.0 / 2.54), constrained_layout=True)
    for panel_index, (ax, info) in enumerate(zip(axes.ravel(), PANEL_INFO)):
        draw_panel(ax, records, *info[:4])
        ax.text(
            -0.17,
            1.08,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
        )
    fig.savefig(combined_output.with_suffix(".png"), dpi=600)
    fig.savefig(combined_output.with_suffix(".svg"))
    fig.savefig(combined_output.with_suffix(".jpg"), dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_existing_rows(args.output_csv)
    done = set() if args.overwrite else existing_keys(rows)

    if args.overwrite:
        rows = []

    cases = build_cases(args)
    if not args.plot_only:
        for index, case in enumerate(cases, start=1):
            key = case_key(case, args.resolution)
            if key in done:
                print(f"[{index}/{len(cases)}] skip cached {case}")
                continue
            print(f"[{index}/{len(cases)}] run {case}")
            row = run_case(case, args)
            rows.append(row)
            write_rows(args.output_csv, rows)
            print(
                "    "
                f"A11={float(row['A11']):.4g}, D11={float(row['D11']):.4g}, "
                f"active_fraction={float(row['active_fraction']):.4f}, "
                f"time={float(row['total_time_s']):.2f}s"
            )

    records = rows_to_records(load_existing_rows(args.output_csv))
    if not records:
        raise RuntimeError(f"No records available in {args.output_csv}.")
    validate_volume_homogenizer(args.youngs_modulus, args.poisson)
    references = ensure_lvs_references(records, args)
    attach_lvs_references(records, references, args)
    write_summary(records, args.summary_csv)
    plot_sensitivity(records, args.output_prefix, args.combined_output)
    print(f"Saved six LVS-H-referenced PNG/SVG panels with prefix {args.output_prefix}")
    print(f"Saved combined figure to {args.combined_output.with_suffix('.jpg')}")
    print(f"Saved absolute-value and achieved-density summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
