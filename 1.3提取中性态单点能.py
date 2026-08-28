#!/usr/bin/env python3
"""Collect M062X neutral/reference-state single-point energies after stage 1.2.

Do not run this script until Gaussian has completed
``orop_neutral_sp_out/{id}_sp_sol_N.out``.  It combines the final SCF energy
with the B3LYP optimization frequency Gibbs correction saved by stage 1.2.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


THERMO_CSV = Path("orop_neutral_thermo.csv")
SP_OUT_DIR = Path("rop313_sp_output_sol_N")
FINAL_CSV = Path("orop_neutral_single_point_energy.csv")


class StageError(RuntimeError):
    pass


def extract_last_scf_energy(output_path: Path) -> float | None:
    content = output_path.read_text(encoding="utf-8", errors="ignore")
    if "Normal termination" not in content:
        return None
    matches = re.findall(r"SCF Done:\s+E\([^\)]+\)\s+=\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", content)
    return float(matches[-1]) if matches else None


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract N-state solution single-point energies after 1.2.")
    parser.add_argument("--thermo-csv", type=Path, default=root / THERMO_CSV)
    parser.add_argument("--sp-out-dir", type=Path, default=root / SP_OUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=root / FINAL_CSV)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    thermo_csv = args.thermo_csv.resolve()
    sp_out_dir = args.sp_out_dir.resolve()
    output_csv = args.output_csv.resolve()
    if not thermo_csv.is_file():
        raise StageError(f"Thermal-correction CSV not found: {thermo_csv}")
    if not sp_out_dir.is_dir():
        raise StageError(f"Single-point output directory not found: {sp_out_dir}")
    if output_csv.exists() and not args.overwrite:
        raise StageError(f"Output already exists: {output_csv}. Use --overwrite to regenerate it.")

    with thermo_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["id", "charge", "multiplicity", "g_correction_hartree"]
        if reader.fieldnames != required:
            raise StageError(f"{thermo_csv.name} must contain exactly: {', '.join(required)}")
        thermo_rows = list(reader)

    results: list[dict[str, str]] = []
    missing: list[int] = []
    for row in thermo_rows:
        try:
            identifier = int(row["id"])
            correction = float(row["g_correction_hartree"])
        except (TypeError, ValueError) as error:
            raise StageError(f"Invalid row in {thermo_csv.name}: {row!r}") from error
        energy = extract_last_scf_energy(sp_out_dir / f"{identifier}_sp_sol_N.out")
        if energy is None:
            missing.append(identifier)
            continue
        results.append({
            "id": str(identifier),
            "charge": row["charge"],
            "multiplicity": row["multiplicity"],
            "g_correction_hartree": f"{correction:.12f}",
            "single_point_energy_hartree": f"{energy:.12f}",
            "gibbs_energy_hartree": f"{energy + correction:.12f}",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "id", "charge", "multiplicity", "g_correction_hartree",
            "single_point_energy_hartree", "gibbs_energy_hartree",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} completed N-state single-point energies to {output_csv}.")
    if missing:
        print(f"No normal single-point result for {len(missing)} IDs: {missing}")


if __name__ == "__main__":
    try:
        main()
    except StageError as error:
        raise SystemExit(f"ERROR: {error}") from error
