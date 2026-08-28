#!/usr/bin/env python3
"""Generate solution-phase neutral/reference-state single-point GJFs after 1.1 jobs.

Prerequisites (do not run this stage until they exist):

* Gaussian outputs ``orop_neutral_opt_out/{id}_opt_sol_N.out`` from stage 1.1;
* ``orop_oxidation_potential.csv`` and ``orop_reduction_potential.csv``.

Only IDs in the union of the two clean experimental-potential CSVs are handled.
For each normally terminated optimization, this script extracts its final geometry,
charge, multiplicity, and Gibbs thermal correction, then writes an M062X/Def2TZVP
SMD single-point input ``orop_neutral_sp_gjf/{id}_sp_sol_N.gjf``.  The SMD route
and any Generic/Read descriptors are copied from the corresponding stage-1.1 GJF.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Final


OXIDATION_CSV: Final = Path("orop_oxidation_potential.csv")
REDUCTION_CSV: Final = Path("orop_reduction_potential.csv")
OPT_GJF_DIR: Final = Path("orop_neutral_opt_gjf")
OPT_OUT_DIR: Final = Path("rop313_opt_output_sol_N")
SP_GJF_DIR: Final = Path("rop313_sp_input_sol_N")
THERMO_CSV: Final = Path("orop_neutral_thermo.csv")

PERIODIC_TABLE: Final = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi",
}


class StageError(RuntimeError):
    pass


def load_ids(path: Path) -> set[int]:
    if not path.is_file():
        raise StageError(f"Required potential CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "电位"]:
            raise StageError(f"{path.name} must contain exactly the columns id,电位.")
        result: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                result.add(int(row["id"]))
                float(row["电位"])
            except (TypeError, ValueError) as error:
                raise StageError(f"Invalid row {row_number} in {path.name}: {row!r}") from error
    return result


def extract_final_coordinates(output_text: str, output_path: Path) -> list[str]:
    lines = output_text.splitlines()
    start = next(
        (index for index in range(len(lines) - 1, -1, -1)
         if "Standard orientation:" in lines[index] or "Input orientation:" in lines[index]),
        None,
    )
    if start is None:
        raise StageError(f"No final Gaussian orientation found in {output_path.name}.")

    dashed_lines = 0
    coordinates: list[str] = []
    for line in lines[start + 1:]:
        if "---------------------------------------" in line:
            dashed_lines += 1
            if dashed_lines == 3:
                break
            continue
        if dashed_lines == 2:
            fields = line.split()
            if len(fields) == 6:
                try:
                    symbol = PERIODIC_TABLE[int(fields[1])]
                except (KeyError, ValueError) as error:
                    raise StageError(f"Unsupported atomic number in {output_path.name}: {line!r}") from error
                coordinates.append(f"{symbol:<2} {fields[3]:>18} {fields[4]:>18} {fields[5]:>18}")
    if not coordinates:
        raise StageError(f"No coordinates extracted from {output_path.name}.")
    return coordinates


def extract_thermo(output_text: str, output_path: Path) -> tuple[str, str, str]:
    charge_match = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", output_text)
    correction_match = re.search(r"Thermal correction to Gibbs Free Energy=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", output_text)
    if charge_match is None:
        raise StageError(f"No charge/multiplicity line found in {output_path.name}.")
    if correction_match is None:
        raise StageError(f"No Gibbs thermal correction found in {output_path.name}.")
    return charge_match.group(1), charge_match.group(2), correction_match.group(1)


def extract_smd_route_and_read_data(gjf_path: Path) -> tuple[str, str]:
    text = gjf_path.read_text(encoding="utf-8")
    route_match = re.search(r"SCRF=\(SMD,Solvent=[^\)]+\)", text, flags=re.IGNORECASE)
    if route_match is None:
        raise StageError(f"No SMD SCRF route found in {gjf_path.name}.")

    lines = text.splitlines()
    charge_line = re.compile(r"^\s*-?\d+\s+\d+\s*$")
    start = next((index for index, line in enumerate(lines) if charge_line.match(line)), None)
    if start is None:
        raise StageError(f"No charge/multiplicity line found in {gjf_path.name}.")
    end = next((index for index in range(start + 1, len(lines)) if not lines[index].strip()), None)
    if end is None:
        raise StageError(f"No end of coordinate block found in {gjf_path.name}.")
    read_data = "\n".join(lines[end + 1:]).strip()
    if "Generic,Read" not in route_match.group(0) and read_data:
        raise StageError(f"Unexpected post-coordinate data in built-in-solvent GJF {gjf_path.name}.")
    if "Generic,Read" in route_match.group(0) and not read_data:
        raise StageError(f"Generic/Read GJF {gjf_path.name} has no SMD descriptors.")
    return route_match.group(0), read_data


def sp_gjf_text(
    identifier: int,
    charge: str,
    multiplicity: str,
    coordinates: list[str],
    smd_route: str,
    read_data: str,
    nprocshared: int,
    memory: str,
) -> str:
    lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={memory}",
        f"#p M062X/Def2TZVP {smd_route} Integral=UltraFine",
        "",
        f"OROP {identifier} solution-phase N-state single-point energy",
        "",
        f"{charge} {multiplicity}",
        *coordinates,
        "",
    ]
    if read_data:
        lines.extend(read_data.splitlines())
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate M062X solution-phase N-state single-point GJFs after 1.1.")
    parser.add_argument("--oxidation-csv", type=Path, default=root / OXIDATION_CSV)
    parser.add_argument("--reduction-csv", type=Path, default=root / REDUCTION_CSV)
    parser.add_argument("--opt-gjf-dir", type=Path, default=root / OPT_GJF_DIR)
    parser.add_argument("--opt-out-dir", type=Path, default=root / OPT_OUT_DIR)
    parser.add_argument("--sp-gjf-dir", type=Path, default=root / SP_GJF_DIR)
    parser.add_argument("--thermo-csv", type=Path, default=root / THERMO_CSV)
    parser.add_argument("--nprocshared", type=int, default=28)
    parser.add_argument("--mem", default="8GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.nprocshared < 1 or not args.mem.strip():
        raise StageError("--nprocshared must be positive and --mem must not be empty.")

    oxidation_csv = args.oxidation_csv.resolve()
    reduction_csv = args.reduction_csv.resolve()
    opt_gjf_dir = args.opt_gjf_dir.resolve()
    opt_out_dir = args.opt_out_dir.resolve()
    sp_gjf_dir = args.sp_gjf_dir.resolve()
    thermo_csv = args.thermo_csv.resolve()
    identifiers = sorted(load_ids(oxidation_csv) | load_ids(reduction_csv))
    if not identifiers:
        raise StageError("The two potential CSVs contain no IDs.")
    if not opt_out_dir.is_dir():
        raise StageError(f"Optimization output directory not found: {opt_out_dir}")
    prepared: list[tuple[int, str, str, str, list[str], str, str]] = []
    skipped: list[int] = []
    for identifier in identifiers:
        output_path = opt_out_dir / f"{identifier}_opt_sol_N.out"
        if not output_path.is_file():
            skipped.append(identifier)
            continue
        output_text = output_path.read_text(encoding="utf-8", errors="ignore")
        if "Normal termination" not in output_text:
            skipped.append(identifier)
            continue
        source_gjf = opt_gjf_dir / f"{identifier}_opt_sol_N.gjf"
        if not source_gjf.is_file():
            raise StageError(f"Missing stage-1.1 GJF for ID {identifier}: {source_gjf}")
        charge, multiplicity, correction = extract_thermo(output_text, output_path)
        coordinates = extract_final_coordinates(output_text, output_path)
        smd_route, read_data = extract_smd_route_and_read_data(source_gjf)
        prepared.append((identifier, charge, multiplicity, correction, coordinates, smd_route, read_data))

    sp_gjf_dir.mkdir(parents=True, exist_ok=True)
    thermo_csv.parent.mkdir(parents=True, exist_ok=True)
    thermo_rows: list[dict[str, str]] = []
    for identifier, charge, multiplicity, correction, coordinates, smd_route, read_data in prepared:
        sp_gjf = sp_gjf_dir / f"{identifier}_sp_sol_N.gjf"
        # Default behavior is safe resume: existing generated input is retained.
        # Use --overwrite only after intentionally reoptimizing a structure.
        if args.overwrite or not sp_gjf.exists():
            sp_gjf.write_text(
                sp_gjf_text(identifier, charge, multiplicity, coordinates, smd_route, read_data, args.nprocshared, args.mem),
                encoding="utf-8",
                newline="\n",
            )
        thermo_rows.append({
            "id": str(identifier),
            "charge": charge,
            "multiplicity": multiplicity,
            "g_correction_hartree": correction,
        })
    with thermo_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "charge", "multiplicity", "g_correction_hartree"])
        writer.writeheader()
        writer.writerows(thermo_rows)

    print(f"Generated {len(prepared)} M062X/Def2TZVP single-point GJFs in {sp_gjf_dir}.")
    print(f"Wrote thermal corrections for those IDs to {thermo_csv}.")
    if skipped:
        print(f"Skipped {len(skipped)} IDs without normally terminated 1.1 outputs: {skipped}")


if __name__ == "__main__":
    try:
        main()
    except StageError as error:
        raise SystemExit(f"ERROR: {error}") from error
