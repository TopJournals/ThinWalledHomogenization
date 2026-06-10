"""Extract extension-induced coupling metrics from an Abaqus ODB.

Run this script with Abaqus Python, for example:

abaqus python abaqus/extract_diamond_coupling_from_odb.py ^
  --odb Paper/abaqus_diamond_tension_twist_hex_r64/diamond_tension_twist_hex_r64.odb ^
  --output Paper/abaqus_diamond_tension_twist_hex_r64/diamond_odb_coupling_metrics.csv

The exported metrics are computed from ODB displacement frames. They are meant
to quantify A16-like extension-shear response and extension-induced twisting
without relying on the homogenized ABD matrix.
"""

from __future__ import print_function

import argparse
import csv
import math
import os

from odbAccess import openOdb


def solve_linear_system(matrix, vector):
    """Solve a small dense linear system using Gauss-Jordan elimination."""
    n = len(vector)
    aug = [list(matrix[i]) + [float(vector[i])] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-30:
            raise RuntimeError("Singular least-squares normal matrix.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        scale = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= scale

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def least_squares(rows, values, ncoef):
    normal = [[0.0 for _ in range(ncoef)] for _ in range(ncoef)]
    rhs = [0.0 for _ in range(ncoef)]
    for row, value in zip(rows, values):
        for i in range(ncoef):
            rhs[i] += row[i] * value
            for j in range(ncoef):
                normal[i][j] += row[i] * row[j]
    return solve_linear_system(normal, rhs)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odb", required=True, help="Path to the Abaqus ODB file.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--instance", default="DIAMOND_LPS-1", help="ODB instance name.")
    parser.add_argument("--step", default="EPS11_TENSION", help="ODB step name.")
    parser.add_argument("--xmax-set", default="XMAX", help="Loaded-end node set name.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Use every nth node for global fits.")
    return parser


def get_node_set(instance, name):
    if name in instance.nodeSets:
        return instance.nodeSets[name]
    upper = name.upper()
    for key, value in instance.nodeSets.items():
        if key.upper() == upper:
            return value
    raise KeyError("Node set not found: {}".format(name))


def main():
    args = build_arg_parser().parse_args()
    odb = openOdb(args.odb, readOnly=True)
    try:
        assembly = odb.rootAssembly
        instance = assembly.instances[args.instance]
        step = odb.steps[args.step]
        xmax_set = get_node_set(instance, args.xmax_set)

        nodes = list(instance.nodes)
        coords_by_label = {}
        for node in nodes:
            coords_by_label[node.label] = tuple(float(v) for v in node.coordinates)

        xs = [coords[0] for coords in coords_by_label.values()]
        ys = [coords[1] for coords in coords_by_label.values()]
        zs = [coords[2] for coords in coords_by_label.values()]
        xmin, xmax = min(xs), max(xs)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (min(ys) + max(ys))
        zmid = 0.5 * (min(zs) + max(zs))
        length_x = xmax - xmin

        sampled_labels = [
            node.label for index, node in enumerate(nodes)
            if index % max(args.sample_stride, 1) == 0
        ]
        xface_labels = [node.label for node in xmax_set.nodes]

        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(args.output, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "frame_index",
                "step_time",
                "eps11_bc",
                "eps11_fit",
                "eps22_fit",
                "gamma12_fit",
                "du1_dy",
                "du2_dx",
                "theta_x_xmax_rad",
                "wxy_fit_1_per_mm",
                "kappa_xy_2wxy_1_per_mm",
                "mean_u3_mm",
                "max_abs_u3_mm",
                "reaction_x_N",
            ])

            for frame_index, frame in enumerate(step.frames):
                if "U" not in frame.fieldOutputs:
                    continue
                displacement = frame.fieldOutputs["U"].getSubset(region=instance)
                disp_by_label = {}
                for value in displacement.values:
                    disp_by_label[value.nodeLabel] = tuple(float(v) for v in value.data)

                affine_rows = []
                u1_values = []
                u2_values = []
                quad_rows = []
                u3_values = []
                max_abs_u3 = 0.0
                sum_u3 = 0.0

                for label in sampled_labels:
                    if label not in disp_by_label:
                        continue
                    x, y, _z = coords_by_label[label]
                    u1, u2, u3 = disp_by_label[label]
                    xc = x - xmid
                    yc = y - ymid
                    affine_rows.append([1.0, xc, yc])
                    u1_values.append(u1)
                    u2_values.append(u2)
                    quad_rows.append([1.0, xc, yc, 0.5 * xc * xc, xc * yc, 0.5 * yc * yc])
                    u3_values.append(u3)
                    sum_u3 += u3
                    max_abs_u3 = max(max_abs_u3, abs(u3))

                u1_coef = least_squares(affine_rows, u1_values, 3)
                u2_coef = least_squares(affine_rows, u2_values, 3)
                w_coef = least_squares(quad_rows, u3_values, 6)

                eps11_fit = u1_coef[1]
                eps22_fit = u2_coef[2]
                du1_dy = u1_coef[2]
                du2_dx = u2_coef[1]
                gamma12_fit = du1_dy + du2_dx
                wxy = w_coef[4]

                u1_xmax = []
                u2_xmax = []
                u3_xmax = []
                yzc_xmax = []
                for label in xface_labels:
                    if label not in disp_by_label:
                        continue
                    _x, y, z = coords_by_label[label]
                    u1, u2, u3 = disp_by_label[label]
                    u1_xmax.append(u1)
                    u2_xmax.append(u2)
                    u3_xmax.append(u3)
                    yzc_xmax.append((y - ymid, z - zmid))

                mean_u1_xmax = sum(u1_xmax) / len(u1_xmax)
                mean_u2_xmax = sum(u2_xmax) / len(u2_xmax)
                mean_u3_xmax = sum(u3_xmax) / len(u3_xmax)
                numerator = 0.0
                denominator = 0.0
                for (yc, zc), u2, u3 in zip(yzc_xmax, u2_xmax, u3_xmax):
                    numerator += yc * (u3 - mean_u3_xmax) - zc * (u2 - mean_u2_xmax)
                    denominator += yc * yc + zc * zc
                theta_x = numerator / denominator if denominator > 0.0 else 0.0
                eps11_bc = mean_u1_xmax / length_x

                reaction_x = 0.0
                if "RF" in frame.fieldOutputs:
                    try:
                        rf_subset = frame.fieldOutputs["RF"].getSubset(region=xmax_set)
                        for value in rf_subset.values:
                            reaction_x += float(value.data[0])
                    except Exception:
                        reaction_x = float("nan")
                else:
                    reaction_x = float("nan")

                mean_u3 = sum_u3 / len(u3_values)
                writer.writerow([
                    frame_index,
                    frame.frameValue,
                    eps11_bc,
                    eps11_fit,
                    eps22_fit,
                    gamma12_fit,
                    du1_dy,
                    du2_dx,
                    theta_x,
                    wxy,
                    2.0 * wxy,
                    mean_u3,
                    max_abs_u3,
                    reaction_x,
                ])
    finally:
        odb.close()


if __name__ == "__main__":
    main()
