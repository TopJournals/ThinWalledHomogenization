"""Run thickness-truncation sensitivity sweeps for LPS-H.

The script reuses the repository geometry generators and the core
``homogenization_plate`` solver. It generates six separate sensitivity figures:

1. lattice topology versus thickness-wise unit-cell count,
2. relative density versus thickness-wise unit-cell count,
3. skin thickness versus thickness-wise unit-cell count,

with normalized ``A11`` and ``D11`` shown separately for each parameter group.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plate_homogenizer import homogenization_plate  # noqa: E402
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
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def normalize_records(records: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["group"]), str(record["label"]))
        groups.setdefault(key, []).append(record)

    for group_records in groups.values():
        reference = max(group_records, key=lambda item: int(item["nz"]))
        ref_a11 = float(reference["A11"])
        ref_d11 = float(reference["D11"])
        for record in group_records:
            record["A11_norm"] = float(record["A11"]) / ref_a11 if ref_a11 else np.nan
            record["D11_norm"] = float(record["D11"]) / ref_d11 if ref_d11 else np.nan


def plot_sensitivity(records: list[dict[str, object]], output_prefix: Path) -> None:
    configure_matplotlib()
    normalize_records(records)

    panel_info = [
        ("topology", "$A_{11}$", "A11_norm", "Lattice topology", "topology_A11"),
        ("topology", "$D_{11}$", "D11_norm", "Lattice topology", "topology_D11"),
        ("density", "$A_{11}$", "A11_norm", "Relative density", "density_A11"),
        ("density", "$D_{11}$", "D11_norm", "Relative density", "density_D11"),
        ("skin", "$A_{11}$", "A11_norm", "Skin thickness", "skin_A11"),
        ("skin", "$D_{11}$", "D11_norm", "Skin thickness", "skin_D11"),
    ]

    colors = [
        "#4A6B8A",  # deep blue gray
        "#C27471",  # brick red
        "#E2B879",  # muted mustard
        "#6A8074",  # muted green
        "#7D6E83",  # muted violet
        "#8A7A5C",  # muted olive brown
    ]
    markers = ["o", "s", "^", "d", "v", "P"]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_nz = sorted({int(r["nz"]) for r in records})

    for group, ylabel, value_key, title, suffix in panel_info:
        fig, ax = plt.subplots(figsize=(8.0 / 2.54, 6.0 / 2.54))
        plt.subplots_adjust(left=0.18, right=0.97, bottom=0.20, top=0.86)

        subset = [r for r in records if r["group"] == group]
        labels = list(dict.fromkeys(str(r["label"]) for r in subset))
        panel_values: list[float] = []
        for idx, label in enumerate(labels):
            line = sorted((r for r in subset if r["label"] == label), key=lambda r: int(r["nz"]))
            x = [int(r["nz"]) for r in line]
            y = [float(r[value_key]) for r in line]
            panel_values.extend(y)
            ax.plot(
                x,
                y,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linewidth=1.0,
                markersize=3.0,
                label=label,
            )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylabel(f"Normalized {ylabel}")
        finite_values = [value for value in panel_values if np.isfinite(value)]
        if finite_values:
            ymin = min(0.0, min(finite_values) * 1.05)
            ymax = max(1.08, max(finite_values) * 1.08)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0.0, 1.08)
        ax.grid(False)
        ax.legend(frameon=False, fontsize=8, ncol=2, handlelength=1.8, columnspacing=0.9)
        ax.set_xlabel("Cell count in thickness ($N_z$)")
        ax.set_xticks(all_nz)
        output_png = output_prefix.with_name(f"{output_prefix.name}_{suffix}").with_suffix(".png")
        output_svg = output_prefix.with_name(f"{output_prefix.name}_{suffix}").with_suffix(".svg")
        fig.savefig(output_png)
        fig.savefig(output_svg)
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
    plot_sensitivity(records, args.output_prefix)
    print(f"Saved six PNG/SVG figures with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()
