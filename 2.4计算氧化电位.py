#!/usr/bin/env python3
"""Calculate OROP oxidation potentials from completed solution-phase Gibbs energies.

For every ID retained in ``orop_oxidation_potential.csv`` this script applies

    E_ox = ((G_sol(ox) - G_sol(N)) / F) - reference_shift

using one-electron transfer.  Gibbs energies from Gaussian are in Hartree per
molecule, so the energy difference is first converted to kcal mol-1 and then
divided by the Faraday constant in kcal mol-1 V-1.  The reference shift is read
directly from that ID's ``.shift`` member in the original ROP313 ZIP archive.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from pathlib import Path


# CODATA/NIST-compatible conversion constants.  Together their ratio is the
# Hartree-to-volt conversion for a one-electron process.
HARTREE_TO_KCAL_MOL = 627.5094740631
FARADAY_KCAL_MOL_PER_V = 23.060547830619
ELECTRONS_TRANSFERRED = 1


class WorkflowError(RuntimeError):
    """A validation error that identifies incomplete or inconsistent inputs."""


def parse_id(value: str | None, context: str) -> int:
    try:
        identifier = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise WorkflowError(f"{context}: invalid ID {value!r}.") from error
    if identifier < 1:
        raise WorkflowError(f"{context}: ID must be positive, got {identifier}.")
    return identifier


def parse_float(value: str | None, context: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise WorkflowError(f"{context}: invalid numeric value {value!r}.") from error
    if not math.isfinite(number):
        raise WorkflowError(f"{context}: value must be finite, got {value!r}.")
    return number


def read_experimental_potentials(csv_path: Path) -> dict[int, float]:
    if not csv_path.is_file():
        raise WorkflowError(f"Oxidation-potential CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "电位"]:
            raise WorkflowError(f"{csv_path.name} must contain exactly two columns: id,电位.")
        values: dict[int, float] = {}
        for line_number, row in enumerate(reader, start=2):
            identifier = parse_id(row.get("id"), f"{csv_path.name}, line {line_number}")
            if identifier in values:
                raise WorkflowError(f"{csv_path.name}: duplicate ID {identifier}.")
            values[identifier] = parse_float(row.get("电位"), f"{csv_path.name}, line {line_number}")
    if not values:
        raise WorkflowError(f"{csv_path.name} contains no oxidation-potential records.")
    return values


def read_gibbs_energies(csv_path: Path, state_name: str) -> dict[int, float]:
    if not csv_path.is_file():
        raise WorkflowError(f"{state_name} Gibbs-energy CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "gibbs_energy_hartree" not in reader.fieldnames:
            raise WorkflowError(
                f"{csv_path.name} must include the columns id and gibbs_energy_hartree."
            )
        values: dict[int, float] = {}
        for line_number, row in enumerate(reader, start=2):
            identifier = parse_id(row.get("id"), f"{csv_path.name}, line {line_number}")
            if identifier in values:
                raise WorkflowError(f"{csv_path.name}: duplicate ID {identifier}.")
            values[identifier] = parse_float(
                row.get("gibbs_energy_hartree"), f"{csv_path.name}, line {line_number}"
            )
    if not values:
        raise WorkflowError(f"{csv_path.name} contains no Gibbs energies.")
    return values


def find_orop_root(archive: zipfile.ZipFile) -> str:
    pattern = re.compile(r"^(?P<root>.*?rop313/orop/)\d+/\.shift$")
    roots = {match.group("root") for name in archive.namelist() if (match := pattern.match(name))}
    if len(roots) != 1:
        raise WorkflowError(f"Expected one rop313/orop root in ZIP; found {sorted(roots)!r}.")
    return roots.pop()


def read_reference_shifts(zip_path: Path, identifiers: list[int]) -> dict[int, float]:
    if not zip_path.is_file():
        raise WorkflowError(f"ROP313 ZIP not found: {zip_path}")
    shifts: dict[int, float] = {}
    with zipfile.ZipFile(zip_path) as archive:
        root = find_orop_root(archive)
        for identifier in identifiers:
            member = f"{root}{identifier}/.shift"
            try:
                raw_shift = archive.read(member).decode("utf-8-sig").strip()
            except KeyError as error:
                raise WorkflowError(f"OROP ID {identifier}: missing reference-electrode shift ({member}).") from error
            shifts[identifier] = parse_float(raw_shift, f"OROP ID {identifier} .shift")
    return shifts


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calculate OROP oxidation potentials from N/oxidized solution Gibbs energies."
    )
    parser.add_argument("--neutral-energy-csv", type=Path, default=root / "orop_neutral_single_point_energy.csv")
    parser.add_argument("--oxidized-energy-csv", type=Path, default=root / "rop313_energy_ox.csv")
    parser.add_argument("--experimental-csv", type=Path, default=root / "orop_oxidation_potential.csv")
    parser.add_argument("--zip", type=Path, default=root / "data" / "jp0c05052_si_002.zip")
    parser.add_argument(
        "--comparison-csv", type=Path, default=root / "orop_oxidation_potential_comparison.csv"
    )
    parser.add_argument(
        "--statistics-csv", type=Path, default=root / "orop_oxidation_potential_statistics.csv"
    )
    args = parser.parse_args()

    neutral = read_gibbs_energies(args.neutral_energy_csv.resolve(), "Neutral-state")
    oxidized = read_gibbs_energies(args.oxidized_energy_csv.resolve(), "Oxidized-state")
    experimental = read_experimental_potentials(args.experimental_csv.resolve())
    comparison_csv = args.comparison_csv.resolve()
    statistics_csv = args.statistics_csv.resolve()

    requested_ids = sorted(experimental)
    missing_neutral = [identifier for identifier in requested_ids if identifier not in neutral]
    missing_oxidized = [identifier for identifier in requested_ids if identifier not in oxidized]
    completed_ids = [
        identifier
        for identifier in requested_ids
        if identifier not in missing_neutral and identifier not in missing_oxidized
    ]
    if not completed_ids:
        raise WorkflowError("No oxidation-potential IDs have both neutral and oxidized Gibbs energies.")

    shifts = read_reference_shifts(args.zip.resolve(), completed_ids)
    rows: list[dict[str, str]] = []
    errors: list[float] = []
    for identifier in completed_ids:
        delta_g_hartree = oxidized[identifier] - neutral[identifier]
        delta_g_kcal_mol = delta_g_hartree * HARTREE_TO_KCAL_MOL
        unshifted_potential = delta_g_kcal_mol / (ELECTRONS_TRANSFERRED * FARADAY_KCAL_MOL_PER_V)
        calculated_potential = unshifted_potential - shifts[identifier]
        error = calculated_potential - experimental[identifier]
        errors.append(error)
        rows.append({
            "id": str(identifier),
            "experimental_oxidation_potential_v": f"{experimental[identifier]:.8f}",
            "reference_shift_v": f"{shifts[identifier]:.8f}",
            "g_sol_neutral_hartree": f"{neutral[identifier]:.12f}",
            "g_sol_oxidized_hartree": f"{oxidized[identifier]:.12f}",
            "delta_g_hartree": f"{delta_g_hartree:.12f}",
            "delta_g_kcal_mol": f"{delta_g_kcal_mol:.8f}",
            "unshifted_potential_v": f"{unshifted_potential:.8f}",
            "calculated_oxidation_potential_v": f"{calculated_potential:.8f}",
            "error_calculated_minus_experimental_v": f"{error:.8f}",
            "absolute_error_v": f"{abs(error):.8f}",
            "squared_error_v2": f"{error * error:.10f}",
        })

    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    experimental_mean = sum(experimental[identifier] for identifier in completed_ids) / len(completed_ids)
    residual_sum_squares = sum(error * error for error in errors)
    total_sum_squares = sum(
        (experimental[identifier] - experimental_mean) ** 2 for identifier in completed_ids
    )
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else math.nan

    write_csv(
        comparison_csv,
        [
            "id", "experimental_oxidation_potential_v", "reference_shift_v",
            "g_sol_neutral_hartree", "g_sol_oxidized_hartree", "delta_g_hartree",
            "delta_g_kcal_mol", "unshifted_potential_v", "calculated_oxidation_potential_v",
            "error_calculated_minus_experimental_v", "absolute_error_v", "squared_error_v2",
        ],
        rows,
    )
    statistic_rows = [
        {"metric": "formula", "value": "((G_sol(ox)-G_sol(N))/F)-reference_shift"},
        {"metric": "electron_count", "value": str(ELECTRONS_TRANSFERRED)},
        {"metric": "hartree_to_kcal_mol", "value": f"{HARTREE_TO_KCAL_MOL:.10f}"},
        {"metric": "faraday_kcal_mol_per_v", "value": f"{FARADAY_KCAL_MOL_PER_V:.12f}"},
        {"metric": "experimental_oxidation_ids", "value": str(len(requested_ids))},
        {"metric": "completed_comparison_ids", "value": str(len(completed_ids))},
        {"metric": "missing_neutral_energy_ids", "value": ",".join(map(str, missing_neutral)) or "none"},
        {"metric": "missing_oxidized_energy_ids", "value": ",".join(map(str, missing_oxidized)) or "none"},
        {"metric": "MAE_V", "value": f"{mae:.8f}"},
        {"metric": "RMSE_V", "value": f"{rmse:.8f}"},
        {"metric": "R2_coefficient_of_determination", "value": f"{r_squared:.8f}"},
    ]
    write_csv(statistics_csv, ["metric", "value"], statistic_rows)

    print(f"Wrote {len(rows)} calculated oxidation potentials to {comparison_csv}.")
    print(f"Wrote MAE/RMSE/R2 statistics to {statistics_csv}.")
    print(f"MAE = {mae:.6f} V; RMSE = {rmse:.6f} V; R2 = {r_squared:.6f}")
    if missing_neutral or missing_oxidized:
        print(
            "Excluded incomplete IDs from statistics: "
            f"missing N = {missing_neutral or 'none'}; "
            f"missing ox = {missing_oxidized or 'none'}."
        )


if __name__ == "__main__":
    try:
        main()
    except WorkflowError as error:
        raise SystemExit(f"ERROR: {error}") from error
