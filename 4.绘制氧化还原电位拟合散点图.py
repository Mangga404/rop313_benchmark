#!/usr/bin/env python3
"""Create DFT-versus-experiment fitted scatter plots for OROP redox potentials."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


class PlotError(RuntimeError):
    """An input validation error for the plotting workflow."""


@dataclass(frozen=True)
class PlotSpec:
    title: str
    input_csv: Path
    experimental_column: str
    dft_column: str
    output_png: Path
    color: str


def read_points(spec: PlotSpec) -> tuple[list[float], list[float]]:
    if not spec.input_csv.is_file():
        raise PlotError(f"Comparison CSV not found: {spec.input_csv}")
    experiment: list[float] = []
    dft: list[float] = []
    with spec.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", spec.experimental_column, spec.dft_column}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PlotError(
                f"{spec.input_csv.name} must include: {', '.join(sorted(required))}."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                x_value = float(row[spec.experimental_column])
                y_value = float(row[spec.dft_column])
            except (TypeError, ValueError) as error:
                raise PlotError(f"Invalid potential on line {line_number} of {spec.input_csv.name}.") from error
            if not (math.isfinite(x_value) and math.isfinite(y_value)):
                raise PlotError(f"Non-finite potential on line {line_number} of {spec.input_csv.name}.")
            experiment.append(x_value)
            dft.append(y_value)
    if len(experiment) < 2:
        raise PlotError(f"{spec.input_csv.name} requires at least two completed samples to fit a line.")
    return experiment, dft


def linear_fit(x_values: list[float], y_values: list[float]) -> tuple[float, float, float]:
    """Return slope, intercept, and ordinary-least-squares R² for y = slope*x + intercept."""
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    sxx = sum((value - x_mean) ** 2 for value in x_values)
    if sxx == 0.0:
        raise PlotError("Experiment values are identical; a linear fit is undefined.")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / sxx
    intercept = y_mean - slope * x_mean
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else math.nan
    return slope, intercept, r_squared


def rounded_limits(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    low = min(*x_values, *y_values)
    high = max(*x_values, *y_values)
    spread = high - low
    padding = max(0.08 * spread, 0.10)
    return low - padding, high + padding


def plot_fit(spec: PlotSpec, dpi: int) -> tuple[int, float, float, float]:
    experiment, dft = read_points(spec)
    slope, intercept, r_squared = linear_fit(experiment, dft)
    axis_min, axis_max = rounded_limits(experiment, dft)

    figure, axis = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    axis.scatter(
        experiment,
        dft,
        s=38,
        color=spec.color,
        edgecolors="black",
        linewidths=0.45,
        alpha=0.82,
        zorder=3,
    )
    axis.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="0.35",
        linestyle="--",
        linewidth=1.15,
        label="y = x",
        zorder=1,
    )
    axis.plot(
        [axis_min, axis_max],
        [slope * axis_min + intercept, slope * axis_max + intercept],
        color="#C43C39",
        linewidth=1.7,
        label="Linear fit",
        zorder=2,
    )
    sign = "+" if intercept >= 0.0 else "−"
    annotation = (
        f"DFT = {slope:.3f} × Experiment {sign} {abs(intercept):.3f}\n"
        f"R² = {r_squared:.3f}\n"
        f"n = {len(experiment)}"
    )
    axis.text(
        0.04,
        0.96,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.92},
    )
    axis.set_title(spec.title, fontsize=13, pad=10)
    axis.set_xlabel("Experiment", fontsize=12)
    axis.set_ylabel("DFT", fontsize=12)
    axis.set_xlim(axis_min, axis_max)
    axis.set_ylim(axis_min, axis_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="0.88", linewidth=0.7, zorder=0)
    axis.legend(loc="lower right", frameon=False, fontsize=10)
    axis.tick_params(labelsize=10)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

    spec.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(spec.output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return len(experiment), slope, intercept, r_squared


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot fitted OROP oxidation and reduction potential scatters.")
    parser.add_argument(
        "--oxidation-csv", type=Path, default=root / "orop_oxidation_potential_comparison.csv"
    )
    parser.add_argument(
        "--reduction-csv", type=Path, default=root / "orop_reduction_potential_comparison.csv"
    )
    parser.add_argument(
        "--oxidation-png", type=Path, default=root / "oxidation_potential_fit_scatter.png"
    )
    parser.add_argument(
        "--reduction-png", type=Path, default=root / "reduction_potential_fit_scatter.png"
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 72:
        raise PlotError("--dpi must be at least 72.")

    specifications = [
        PlotSpec(
            "Oxidation potential",
            args.oxidation_csv.resolve(),
            "experimental_oxidation_potential_v",
            "calculated_oxidation_potential_v",
            args.oxidation_png.resolve(),
            "#2878B5",
        ),
        PlotSpec(
            "Reduction potential",
            args.reduction_csv.resolve(),
            "experimental_reduction_potential_v",
            "calculated_reduction_potential_v",
            args.reduction_png.resolve(),
            "#E18727",
        ),
    ]
    for spec in specifications:
        sample_count, slope, intercept, r_squared = plot_fit(spec, args.dpi)
        print(
            f"Wrote {spec.output_png} ({sample_count} samples; "
            f"slope={slope:.6f}, intercept={intercept:.6f}, R2={r_squared:.6f})."
        )


if __name__ == "__main__":
    try:
        main()
    except PlotError as error:
        raise SystemExit(f"ERROR: {error}") from error
