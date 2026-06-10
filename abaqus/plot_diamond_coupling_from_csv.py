"""Plot ODB-extracted coupling curves for the Diamond tension-twist model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CM = 1.0 / 2.54
FIGSIZE = (8.0 * CM, 6.0 * CM)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "svg.fonttype": "none",
    "text.usetex": False,
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
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
})

PALETTE = {
    "blue": "#3B5BA9",
    "red": "#B44B4B",
    "green": "#4E8C6A",
    "gray": "#777777",
}


def fit_slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0.0 else 0.0


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    out_base: Path,
    color: str,
    scale_y: float = 1.0,
) -> None:
    y_scaled = y * scale_y
    slope = fit_slope_through_origin(x, y_scaled)
    xx = np.linspace(0.0, max(x) * 1.03, 100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y_scaled, "o", ms=3.8, color=color, label="ODB frames")
    ax.plot(xx, slope * xx, "-", lw=1.2, color=color, label="linear fit")
    ax.axhline(0.0, color="#999999", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel(r"Applied strain $\varepsilon_{11}$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, max(x) * 1.06)
    ymin, ymax = np.min(y_scaled), np.max(y_scaled)
    margin = max((ymax - ymin) * 0.12, abs(ymax) * 0.08, 1.0e-12)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.legend(frameon=False, loc="best", handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def save_overall_displacement_plot(data: pd.DataFrame, out_base: Path) -> None:
    x = data["eps11_bc"].to_numpy(dtype=float)
    max_u = data["max_abs_u_mm"].to_numpy(dtype=float)
    max_u3 = data["max_abs_u3_mm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, max_u, "o-", ms=3.6, lw=1.2, color=PALETTE["blue"], label=r"max $|\mathbf{U}|$")
    ax.plot(x, max_u3, "s-", ms=3.4, lw=1.2, color=PALETTE["red"], label=r"max $|U_3|$")
    ax.set_xlabel(r"Applied strain $\varepsilon_{11}$")
    ax.set_ylabel("Displacement (mm)")
    ax.set_xlim(0.0, max(x) * 1.06)
    ax.set_ylim(0.0, max(max(max_u), max(max_u3)) * 1.12)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def save_xmax_u3_profile_plot(profile: pd.DataFrame, out_base: Path) -> None:
    nonzero = profile[profile["eps11_bc"] > 0.0].copy()
    final_frame = int(nonzero["frame_index"].max())
    final = nonzero[nonzero["frame_index"] == final_frame].sort_values("y_mm")
    eps = float(final["eps11_bc"].iloc[0])
    y0 = 0.5 * (profile["y_mm"].min() + profile["y_mm"].max())
    y = final["y_mm"].to_numpy(dtype=float) - y0
    u3 = final["u3_mean_mm"].to_numpy(dtype=float)
    delta_u3 = float(np.max(u3) - np.min(u3))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(y, u3, "-", lw=1.4, color=PALETTE["blue"])
    ax.axhline(0.0, color="#999999", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel(r"Loaded-edge coordinate $y-W/2$ (mm)")
    ax.set_ylabel(r"Mean loaded-edge $U_3$ (mm)")
    ax.text(
        0.04,
        0.92,
        rf"$\varepsilon_{{11}}={eps:.3f}$" + "\n" + rf"$\Delta U_3={delta_u3:.3f}$ mm",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV produced by extract_diamond_coupling_from_odb.py.")
    parser.add_argument("--profile-csv", default=None, help="Loaded-end U3(y) profile CSV.")
    parser.add_argument("--outdir", default=None, help="Output figure directory.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.profile_csv:
        profile_csv_path = Path(args.profile_csv)
    else:
        profile_csv_path = csv_path.with_name(csv_path.stem + "_xmax_u3_profile.csv")
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    data = data[data["eps11_bc"] > 0.0].copy()
    profile = pd.read_csv(profile_csv_path)
    eps = data["eps11_bc"].to_numpy(dtype=float)

    save_overall_displacement_plot(
        data,
        outdir / "diamond_odb_overall_displacement_vs_eps11",
    )
    save_xmax_u3_profile_plot(
        profile,
        outdir / "diamond_odb_xmax_u3_profile_vs_y",
    )

    save_line_plot(
        eps,
        data["gamma12_fit"].to_numpy(dtype=float),
        r"Fitted shear strain $\gamma_{12}$",
        outdir / "diamond_odb_gamma12_vs_eps11",
        PALETTE["blue"],
    )
    save_line_plot(
        eps,
        data["theta_x_xmax_rad"].to_numpy(dtype=float),
        r"Loaded-edge twist angle $\theta_x$ (rad)",
        outdir / "diamond_odb_theta_x_vs_eps11",
        PALETTE["red"],
    )
    save_line_plot(
        eps,
        data["kappa_xy_2wxy_1_per_mm"].to_numpy(dtype=float),
        r"Fitted twist curvature $2w_{,xy}$ (mm$^{-1}$)",
        outdir / "diamond_odb_kappaxy_vs_eps11",
        PALETTE["green"],
    )


if __name__ == "__main__":
    main()
