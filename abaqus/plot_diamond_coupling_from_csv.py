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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV produced by extract_diamond_coupling_from_odb.py.")
    parser.add_argument("--outdir", default=None, help="Output figure directory.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    data = data[data["eps11_bc"] > 0.0].copy()
    eps = data["eps11_bc"].to_numpy(dtype=float)

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
