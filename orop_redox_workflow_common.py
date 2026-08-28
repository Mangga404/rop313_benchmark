"""Shared implementation for the OROP oxidation/reduction stages (2.* and 3.*)."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


PERIODIC_TABLE = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi",
}
ATOMIC_NUMBERS = {symbol: number for number, symbol in PERIODIC_TABLE.items()}


class WorkflowError(RuntimeError):
    """A safe, actionable workflow validation error."""


@dataclass(frozen=True)
class RedoxStage:
    label: str
    chinese_name: str
    potential_csv: Path
    target_state: int
    opt_input_dir: Path
    opt_output_dir: Path
    sp_input_dir: Path
    sp_output_dir: Path
    thermo_csv: Path
    energy_csv: Path

    @property
    def opt_stem_suffix(self) -> str:
        return f"opt_sol_{self.label}"

    @property
    def sp_stem_suffix(self) -> str:
        return f"sp_sol_{self.label}"


def load_experimental_ids(csv_path: Path) -> list[int]:
    if not csv_path.is_file():
        raise WorkflowError(f"Experimental potential CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "电位"]:
            raise WorkflowError(f"{csv_path.name} must contain exactly two columns: id,电位.")
        identifiers: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                identifier = int(row["id"])
                float(row["电位"])
            except (TypeError, ValueError) as error:
                raise WorkflowError(f"Invalid line {line_number} in {csv_path.name}: {row!r}") from error
            identifiers.append(identifier)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise WorkflowError(f"{csv_path.name} must contain unique, nonempty IDs.")
    return sorted(identifiers)


def find_orop_root(archive: zipfile.ZipFile) -> str:
    pattern = re.compile(r"^(?P<root>.*?rop313/orop/)\d+/\.CHRG1$")
    roots = {match.group("root") for name in archive.namelist() if (match := pattern.match(name))}
    if len(roots) != 1:
        raise WorkflowError(f"Expected one rop313/orop root in ZIP; found {sorted(roots)!r}.")
    return roots.pop()


def target_charge_multiplicities(zip_path: Path, identifiers: list[int], state: int) -> dict[int, tuple[int, int]]:
    if state not in (1, 2):
        raise WorkflowError(f"Invalid ROP313 state {state}.")
    if not zip_path.is_file():
        raise WorkflowError(f"ROP313 ZIP not found: {zip_path}")
    result: dict[int, tuple[int, int]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        root = find_orop_root(archive)
        for identifier in identifiers:
            try:
                charge = int(archive.read(f"{root}{identifier}/.CHRG{state}").decode("utf-8-sig").strip())
                uhf = int(archive.read(f"{root}{identifier}/.UHF{state}").decode("utf-8-sig").strip())
            except KeyError as error:
                raise WorkflowError(f"OROP ID {identifier}: target state-{state} metadata is missing.") from error
            except ValueError as error:
                raise WorkflowError(f"OROP ID {identifier}: invalid .CHRG{state} or .UHF{state} value.") from error
            if uhf not in (0, 1):
                raise WorkflowError(
                    f"OROP ID {identifier}: UHF{state}={uhf}; only OROP UHF values 0/1 are supported."
                )
            result[identifier] = (charge, uhf + 1)
    return result


def extract_final_coordinates(output_text: str, output_path: Path) -> list[str]:
    lines = output_text.splitlines()
    start = next(
        (index for index in range(len(lines) - 1, -1, -1)
         if "Standard orientation:" in lines[index] or "Input orientation:" in lines[index]),
        None,
    )
    if start is None:
        raise WorkflowError(f"No final Gaussian orientation found in {output_path.name}.")

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
                    raise WorkflowError(f"Unsupported atomic number in {output_path.name}: {line!r}") from error
                coordinates.append(f"{symbol:<2} {fields[3]:>18} {fields[4]:>18} {fields[5]:>18}")
    if not coordinates:
        raise WorkflowError(f"No coordinates extracted from {output_path.name}.")
    return coordinates


def choose_basis(coordinates: list[str]) -> str:
    return "Def2SVP" if any(ATOMIC_NUMBERS[line.split()[0]] > 36 for line in coordinates) else "6-31G(d)"


def extract_smd_route_and_read_data(gjf_path: Path) -> tuple[str, str]:
    text = gjf_path.read_text(encoding="utf-8")
    route_match = re.search(r"SCRF=\(SMD,Solvent=[^\)]+\)", text, flags=re.IGNORECASE)
    if route_match is None:
        raise WorkflowError(f"No SMD SCRF route found in {gjf_path.name}.")
    lines = text.splitlines()
    charge_pattern = re.compile(r"^\s*-?\d+\s+\d+\s*$")
    start = next((index for index, line in enumerate(lines) if charge_pattern.match(line)), None)
    if start is None:
        raise WorkflowError(f"No charge/multiplicity line found in {gjf_path.name}.")
    end = next((index for index in range(start + 1, len(lines)) if not lines[index].strip()), None)
    if end is None:
        raise WorkflowError(f"No end of coordinate block found in {gjf_path.name}.")
    read_data = "\n".join(lines[end + 1:]).strip()
    uses_generic_read = "Generic,Read" in route_match.group(0)
    if uses_generic_read != bool(read_data):
        state = "missing" if uses_generic_read else "unexpected"
        raise WorkflowError(f"{state.capitalize()} SMD Read data in {gjf_path.name}.")
    return route_match.group(0), read_data


def opt_gjf_text(
    stage: RedoxStage,
    identifier: int,
    charge: int,
    multiplicity: int,
    coordinates: list[str],
    smd_route: str,
    read_data: str,
    nprocshared: int,
    memory: str,
) -> str:
    lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={memory}",
        (
            f"#p B3LYP/{choose_basis(coordinates)} Opt=(CalcFC,MaxCycles=200,NoEigen) Freq "
            f"Integral=UltraFine {smd_route} SCF=(XQC,MaxCycle=500)"
        ),
        "",
        f"OROP {identifier} {stage.chinese_name} state; coordinates from completed N-state optimization",
        "",
        f"{charge} {multiplicity}",
        *coordinates,
        "",
    ]
    if read_data:
        lines.extend(read_data.splitlines())
        lines.append("")
    return "\n".join(lines) + "\n"


def sp_gjf_text(
    stage: RedoxStage,
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
        f"OROP {identifier} {stage.chinese_name} state solution-phase single-point energy",
        "",
        f"{charge} {multiplicity}",
        *coordinates,
        "",
    ]
    if read_data:
        lines.extend(read_data.splitlines())
        lines.append("")
    return "\n".join(lines) + "\n"


def extract_thermo(output_text: str, output_path: Path) -> tuple[str, str, str]:
    charge_match = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", output_text)
    correction_match = re.search(r"Thermal correction to Gibbs Free Energy=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", output_text)
    if charge_match is None or correction_match is None:
        raise WorkflowError(f"Missing charge/multiplicity or Gibbs correction in {output_path.name}.")
    return charge_match.group(1), charge_match.group(2), correction_match.group(1)


def ensure_writable(output_csv: Path, output_dir: Path, overwrite: bool) -> None:
    has_gjfs = output_dir.is_dir() and bool(list(output_dir.glob("*.gjf")))
    if (output_csv.exists() or has_gjfs) and not overwrite:
        raise WorkflowError("Outputs already exist; use --overwrite to regenerate them.")


def optimization_cli(stage: RedoxStage) -> None:
    parser = argparse.ArgumentParser(description=f"Generate OROP {stage.chinese_name} opt+freq GJFs after completed N-state jobs.")
    root = stage.potential_csv.parent
    parser.add_argument("--potential-csv", type=Path, default=stage.potential_csv)
    parser.add_argument("--zip", type=Path, default=root / "data" / "jp0c05052_si_002.zip")
    parser.add_argument("--neutral-opt-gjf-dir", type=Path, default=root / "orop_neutral_opt_gjf")
    parser.add_argument("--neutral-opt-out-dir", type=Path, default=root / "rop313_opt_output_sol_N")
    parser.add_argument("--output-dir", type=Path, default=stage.opt_input_dir)
    parser.add_argument("--nprocshared", type=int, default=28)
    parser.add_argument("--mem", default="8GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.nprocshared < 1 or not args.mem.strip():
        raise WorkflowError("--nprocshared must be positive and --mem must not be empty.")

    identifiers = load_experimental_ids(args.potential_csv.resolve())
    neutral_gjf_dir = args.neutral_opt_gjf_dir.resolve()
    neutral_out_dir = args.neutral_opt_out_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not neutral_out_dir.is_dir():
        raise WorkflowError(f"Neutral optimization output directory not found: {neutral_out_dir}")
    if output_dir.is_dir() and list(output_dir.glob("*.gjf")) and not args.overwrite:
        raise WorkflowError(f"{stage.chinese_name} GJFs already exist in {output_dir}; use --overwrite to regenerate.")

    target_states = target_charge_multiplicities(args.zip.resolve(), identifiers, stage.target_state)
    prepared: list[tuple[int, int, int, list[str], str, str]] = []
    skipped: list[int] = []
    for identifier in identifiers:
        neutral_output = neutral_out_dir / f"{identifier}_opt_sol_N.out"
        if not neutral_output.is_file():
            skipped.append(identifier)
            continue
        output_text = neutral_output.read_text(encoding="utf-8", errors="ignore")
        if "Normal termination" not in output_text:
            skipped.append(identifier)
            continue
        neutral_gjf = neutral_gjf_dir / f"{identifier}_opt_sol_N.gjf"
        if not neutral_gjf.is_file():
            raise WorkflowError(f"Missing N-state GJF for ID {identifier}: {neutral_gjf}")
        charge, multiplicity = target_states[identifier]
        coordinates = extract_final_coordinates(output_text, neutral_output)
        smd_route, read_data = extract_smd_route_and_read_data(neutral_gjf)
        prepared.append((identifier, charge, multiplicity, coordinates, smd_route, read_data))

    output_dir.mkdir(parents=True, exist_ok=True)
    for identifier, charge, multiplicity, coordinates, smd_route, read_data in prepared:
        (output_dir / f"{identifier}_{stage.opt_stem_suffix}.gjf").write_text(
            opt_gjf_text(stage, identifier, charge, multiplicity, coordinates, smd_route, read_data, args.nprocshared, args.mem),
            encoding="utf-8",
            newline="\n",
        )
    print(f"Generated {len(prepared)} {stage.chinese_name} opt+freq GJFs in {output_dir}.")
    if skipped:
        print(f"Skipped {len(skipped)} IDs without normally terminated N-state outputs: {skipped}")


def single_point_cli(stage: RedoxStage) -> None:
    parser = argparse.ArgumentParser(description=f"Generate OROP {stage.chinese_name} single-point GJFs after completed optimization jobs.")
    parser.add_argument("--potential-csv", type=Path, default=stage.potential_csv)
    parser.add_argument("--opt-gjf-dir", type=Path, default=stage.opt_input_dir)
    parser.add_argument("--opt-out-dir", type=Path, default=stage.opt_output_dir)
    parser.add_argument("--sp-gjf-dir", type=Path, default=stage.sp_input_dir)
    parser.add_argument("--thermo-csv", type=Path, default=stage.thermo_csv)
    parser.add_argument("--nprocshared", type=int, default=28)
    parser.add_argument("--mem", default="8GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.nprocshared < 1 or not args.mem.strip():
        raise WorkflowError("--nprocshared must be positive and --mem must not be empty.")
    identifiers = load_experimental_ids(args.potential_csv.resolve())
    opt_gjf_dir = args.opt_gjf_dir.resolve()
    opt_out_dir = args.opt_out_dir.resolve()
    sp_gjf_dir = args.sp_gjf_dir.resolve()
    thermo_csv = args.thermo_csv.resolve()
    if not opt_out_dir.is_dir():
        raise WorkflowError(f"{stage.chinese_name} optimization output directory not found: {opt_out_dir}")
    prepared: list[tuple[int, str, str, str, list[str], str, str]] = []
    skipped: list[int] = []
    for identifier in identifiers:
        opt_output = opt_out_dir / f"{identifier}_{stage.opt_stem_suffix}.out"
        if not opt_output.is_file():
            skipped.append(identifier)
            continue
        output_text = opt_output.read_text(encoding="utf-8", errors="ignore")
        if "Normal termination" not in output_text:
            skipped.append(identifier)
            continue
        opt_gjf = opt_gjf_dir / f"{identifier}_{stage.opt_stem_suffix}.gjf"
        if not opt_gjf.is_file():
            raise WorkflowError(f"Missing {stage.chinese_name} optimization GJF: {opt_gjf}")
        charge, multiplicity, correction = extract_thermo(output_text, opt_output)
        coordinates = extract_final_coordinates(output_text, opt_output)
        smd_route, read_data = extract_smd_route_and_read_data(opt_gjf)
        prepared.append((identifier, charge, multiplicity, correction, coordinates, smd_route, read_data))

    sp_gjf_dir.mkdir(parents=True, exist_ok=True)
    thermo_csv.parent.mkdir(parents=True, exist_ok=True)
    thermo_rows: list[dict[str, str]] = []
    for identifier, charge, multiplicity, correction, coordinates, smd_route, read_data in prepared:
        sp_gjf = sp_gjf_dir / f"{identifier}_{stage.sp_stem_suffix}.gjf"
        # Resume by default.  Existing generated inputs are only replaced when
        # the user explicitly requests --overwrite after a reoptimization.
        if args.overwrite or not sp_gjf.exists():
            sp_gjf.write_text(
                sp_gjf_text(stage, identifier, charge, multiplicity, coordinates, smd_route, read_data, args.nprocshared, args.mem),
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
    print(f"Generated {len(prepared)} {stage.chinese_name} single-point GJFs in {sp_gjf_dir}.")
    print(f"Wrote matching Gibbs corrections to {thermo_csv}.")
    if skipped:
        print(f"Skipped {len(skipped)} IDs without normally terminated {stage.chinese_name} optimization outputs: {skipped}")


def extract_last_scf_energy(output_path: Path) -> float | None:
    content = output_path.read_text(encoding="utf-8", errors="ignore")
    if "Normal termination" not in content:
        return None
    matches = re.findall(r"SCF Done:\s+E\([^\)]+\)\s+=\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", content)
    return float(matches[-1]) if matches else None


def energy_extraction_cli(stage: RedoxStage) -> None:
    parser = argparse.ArgumentParser(description=f"Extract OROP {stage.chinese_name} single-point energies after completed jobs.")
    parser.add_argument("--thermo-csv", type=Path, default=stage.thermo_csv)
    parser.add_argument("--sp-out-dir", type=Path, default=stage.sp_output_dir)
    parser.add_argument("--output-csv", type=Path, default=stage.energy_csv)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    thermo_csv = args.thermo_csv.resolve()
    sp_out_dir = args.sp_out_dir.resolve()
    output_csv = args.output_csv.resolve()
    if not thermo_csv.is_file():
        raise WorkflowError(f"Thermal-correction CSV not found: {thermo_csv}")
    if not sp_out_dir.is_dir():
        raise WorkflowError(f"{stage.chinese_name} single-point output directory not found: {sp_out_dir}")
    if output_csv.exists() and not args.overwrite:
        raise WorkflowError(f"Output already exists: {output_csv}; use --overwrite to regenerate it.")

    with thermo_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = ["id", "charge", "multiplicity", "g_correction_hartree"]
        if reader.fieldnames != columns:
            raise WorkflowError(f"{thermo_csv.name} must contain exactly: {', '.join(columns)}")
        thermo_rows = list(reader)

    results: list[dict[str, str]] = []
    missing: list[int] = []
    for row in thermo_rows:
        try:
            identifier = int(row["id"])
            correction = float(row["g_correction_hartree"])
        except (TypeError, ValueError) as error:
            raise WorkflowError(f"Invalid thermal CSV row: {row!r}") from error
        energy = extract_last_scf_energy(sp_out_dir / f"{identifier}_{stage.sp_stem_suffix}.out")
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
    print(f"Wrote {len(results)} completed {stage.chinese_name} energies to {output_csv}.")
    if missing:
        print(f"No normally terminated {stage.chinese_name} single-point output for {len(missing)} IDs: {missing}")
