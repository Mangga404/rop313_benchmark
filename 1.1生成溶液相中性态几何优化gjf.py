#!/usr/bin/env python3
"""Prepare the neutral/reference-state solution-phase optimization jobs for OROP.

The script reads ``data/jp0c05052_si_002.zip`` directly and processes only
``rop313/orop``.  It first creates two deliberately minimal experiment lists:

* ``orop_oxidation_potential.csv``  (columns: ``id,电位``)
* ``orop_reduction_potential.csv``  (columns: ``id,电位``)

It then uses the union of those two lists to generate exactly one initial
``{id}_opt_sol_N.gjf`` job per experimental redox couple in
``orop_neutral_opt_gjf``.  For ordinary couples this is the net-neutral state:
state 2 for oxidation potentials (+1 -> 0) and state 1 for reduction potentials
(0 -> -1).  Three ROP313 couples are entirely charged; for them ``N`` denotes
the corresponding reference state (state 2 for oxidation, state 1 for reduction)
so that no experimental record is silently discarded.

B97-3c XYZ geometries are starting coordinates only.  Each GJF performs a new
B3LYP/SMD optimization and frequency calculation using the established basis
rule: 6-31G(d) through Kr and Def2SVP for elements heavier than Kr.

Run only this stage before Gaussian calculations are available::

    python "1.1生成溶液相中性态几何优化gjf.py"
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from orop_smd_parameters import SMD_PARAMS


DEFAULT_ZIP: Final = Path("data") / "jp0c05052_si_002.zip"
OXIDATION_CSV: Final = Path("orop_oxidation_potential.csv")
REDUCTION_CSV: Final = Path("orop_reduction_potential.csv")
GJF_DIR: Final = Path("orop_neutral_opt_gjf")
# These experimental records are permanently excluded because their neutral-state
# geometry optimizations do not converge.  All downstream stages read the two
# CSVs written here, so the exclusion propagates through the complete workflow.
PERMANENT_EXCLUDED_IDS: Final = frozenset({166, 176})

GAUSSIAN_BUILTIN_SMD_SOLVENTS: Final = {
    "acetonitrile": "Acetonitrile",
    "mecn": "Acetonitrile",
    "dmf": "N,N-Dimethylformamide",
    "n,n-dimethylformamide": "N,N-Dimethylformamide",
    "n,n dimethylformamide": "N,N-Dimethylformamide",
}

ATOMIC_NUMBERS: Final = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
    "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83,
}


class InputDataError(RuntimeError):
    """Raised when an input cannot be handled without guessing."""


@dataclass(frozen=True)
class SolventSetup:
    raw: str
    gaussian_name: str
    custom_parameters: tuple[float, ...] | None

    @property
    def is_builtin(self) -> bool:
        return self.custom_parameters is None


@dataclass(frozen=True)
class State:
    number: int
    charge: int
    uhf: int
    multiplicity: int
    coordinates: tuple[str, ...]
    basis: str


@dataclass(frozen=True)
class Record:
    identifier: int
    potential_raw: str
    potential_type: str
    solvent: SolventSetup
    state1: State  # oxidized by the ROP313 definition
    state2: State  # reduced by the ROP313 definition

    @property
    def n_reference_state(self) -> State:
        # The less oxidized partner starts an oxidation calculation; the more
        # oxidized partner starts a reduction calculation.
        return self.state2 if self.potential_type == "oxidation" else self.state1


def normalise_solvent(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def resolve_solvent(raw: str, identifier: int) -> SolventSetup:
    gaussian_name = GAUSSIAN_BUILTIN_SMD_SOLVENTS.get(normalise_solvent(raw))
    if gaussian_name is not None:
        return SolventSetup(raw=raw, gaussian_name=gaussian_name, custom_parameters=None)

    # The SMD parameter keys are SMILES and must stay case-sensitive.
    parameters = SMD_PARAMS.get(raw.strip())
    if parameters is None:
        raise InputDataError(
            f"OROP ID {identifier}: unknown solvent {raw!r}. It is neither a Gaussian "
            "built-in SMD solvent nor a key in SMD_PARAMS; parameters were not guessed."
        )
    return SolventSetup(raw=raw, gaussian_name="Generic", custom_parameters=tuple(parameters))


def read_member(archive: zipfile.ZipFile, member: str, identifier: int) -> str:
    try:
        return archive.read(member).decode("utf-8-sig").strip()
    except KeyError as error:
        raise InputDataError(f"OROP ID {identifier}: missing required archive member {member}.") from error
    except UnicodeDecodeError as error:
        raise InputDataError(f"OROP ID {identifier}: cannot decode {member} as UTF-8.") from error


def find_orop_root(names: list[str]) -> str:
    pattern = re.compile(r"^(?P<root>.*?rop313/orop/)(?P<id>\d+)/\.ref$")
    roots = {match.group("root") for name in names if (match := pattern.match(name))}
    if len(roots) != 1:
        raise InputDataError(f"Expected one rop313/orop root in ZIP, found {sorted(roots)!r}.")
    return roots.pop()


def parse_int(raw: str, member: str, identifier: int) -> int:
    try:
        return int(raw)
    except ValueError as error:
        raise InputDataError(f"OROP ID {identifier}: {member} must contain one integer, got {raw!r}.") from error


def parse_potential(raw: str, identifier: int) -> None:
    try:
        float(raw)
    except ValueError as error:
        raise InputDataError(f"OROP ID {identifier}: .ref is not numeric: {raw!r}.") from error


def parse_xyz(raw: str, identifier: int, member: str) -> tuple[str, ...]:
    lines = raw.splitlines()
    if len(lines) < 3:
        raise InputDataError(f"OROP ID {identifier}: malformed XYZ file {member}.")
    try:
        expected_count = int(lines[0].strip())
    except ValueError as error:
        raise InputDataError(f"OROP ID {identifier}: invalid atom count in {member}.") from error

    coordinate_lines = [line for line in lines[2:] if line.strip()]
    if len(coordinate_lines) != expected_count:
        raise InputDataError(
            f"OROP ID {identifier}: {member} declares {expected_count} atoms but has "
            f"{len(coordinate_lines)} coordinate lines."
        )

    result: list[str] = []
    for line_number, line in enumerate(coordinate_lines, start=3):
        fields = line.split()
        if len(fields) != 4:
            raise InputDataError(
                f"OROP ID {identifier}: {member} line {line_number} is not four-column XYZ data."
            )
        symbol = fields[0][:1].upper() + fields[0][1:].lower()
        if symbol not in ATOMIC_NUMBERS:
            raise InputDataError(f"OROP ID {identifier}: unsupported element {fields[0]!r} in {member}.")
        try:
            float(fields[1])
            float(fields[2])
            float(fields[3])
        except ValueError as error:
            raise InputDataError(
                f"OROP ID {identifier}: nonnumeric coordinate in {member} line {line_number}."
            ) from error
        # Preserve the supplied B97-3c numeric precision in the generated input.
        result.append(f"{symbol:<2} {fields[1]:>18} {fields[2]:>18} {fields[3]:>18}")
    return tuple(result)


def choose_basis(coordinates: tuple[str, ...]) -> str:
    return "Def2SVP" if any(ATOMIC_NUMBERS[line.split()[0]] > 36 for line in coordinates) else "6-31G(d)"


def parse_state(raw_charge: str, raw_uhf: str, xyz: str, identifier: int, state: int) -> State:
    charge = parse_int(raw_charge, f".CHRG{state}", identifier)
    uhf = parse_int(raw_uhf, f".UHF{state}", identifier)
    if uhf not in (0, 1):
        raise InputDataError(
            f"OROP ID {identifier} state {state}: UHF={uhf}; only 0 or 1 is permitted for this workflow."
        )
    coordinates = parse_xyz(xyz, identifier, f"{state}.b973c.xyz")
    return State(
        number=state,
        charge=charge,
        uhf=uhf,
        multiplicity=uhf + 1,
        coordinates=coordinates,
        basis=choose_basis(coordinates),
    )


def classify_potential(identifier: int, state1: State, state2: State) -> str:
    """Classify .ref without ambiguity from the one-electron charge couple."""
    if state1.charge != state2.charge + 1:
        raise InputDataError(
            f"OROP ID {identifier}: charge pair {state1.charge} -> {state2.charge} is not one-electron."
        )
    if state1.charge > 0 and state2.charge >= 0:
        return "oxidation"
    if state1.charge <= 0 and state2.charge < 0:
        return "reduction"
    raise InputDataError(
        f"OROP ID {identifier}: cannot classify charge pair {state1.charge} -> {state2.charge} "
        "as oxidation or reduction without guessing."
    )


def parse_records(zip_path: Path) -> list[Record]:
    if not zip_path.is_file():
        raise InputDataError(f"ROP313 ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        root = find_orop_root(names)
        id_pattern = re.compile(r"^" + re.escape(root) + r"(?P<id>\d+)/\.ref$")
        identifiers = sorted(int(match.group("id")) for name in names if (match := id_pattern.match(name)))
        if identifiers != list(range(1, 194)):
            raise InputDataError(
                f"Expected the complete OROP 1..193 range, found {len(identifiers)} entries."
            )

        records: list[Record] = []
        for identifier in identifiers:
            member = lambda filename: f"{root}{identifier}/{filename}"
            state1 = parse_state(
                read_member(archive, member(".CHRG1"), identifier),
                read_member(archive, member(".UHF1"), identifier),
                read_member(archive, member("1.b973c.xyz"), identifier),
                identifier,
                1,
            )
            state2 = parse_state(
                read_member(archive, member(".CHRG2"), identifier),
                read_member(archive, member(".UHF2"), identifier),
                read_member(archive, member("2.b973c.xyz"), identifier),
                identifier,
                2,
            )
            potential_raw = read_member(archive, member(".ref"), identifier)
            parse_potential(potential_raw, identifier)
            records.append(
                Record(
                    identifier=identifier,
                    potential_raw=potential_raw,
                    potential_type=classify_potential(identifier, state1, state2),
                    solvent=resolve_solvent(read_member(archive, member(".solv"), identifier), identifier),
                    state1=state1,
                    state2=state2,
                )
            )
    return [record for record in records if record.identifier not in PERMANENT_EXCLUDED_IDS]


def smd_read_lines(parameters: tuple[float, ...]) -> list[str]:
    if len(parameters) != 7:
        raise InputDataError(f"Internal error: expected seven SMD parameters, got {parameters!r}.")
    eps, epsinf, acidity, basicity, surface_tension, aromaticity, halogenicity = parameters
    return [
        f"eps={eps:.4f}",
        f"epsinf={epsinf:.4f}",
        f"HbondAcidity={acidity:.4f}",
        f"HbondBasicity={basicity:.4f}",
        f"SurfaceTensionAtInterface={surface_tension:.4f}",
        f"CarbonAromaticity={aromaticity:.4f}",
        f"ElectronegativeHalogenicity={halogenicity:.4f}",
    ]


def gjf_text(record: Record, nprocshared: int, memory: str) -> str:
    state = record.n_reference_state
    scrf = (
        f"SCRF=(SMD,Solvent={record.solvent.gaussian_name})"
        if record.solvent.is_builtin
        else "SCRF=(SMD,Solvent=Generic,Read)"
    )
    actual_state = "neutral" if state.charge == 0 else "charged reference"
    lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={memory}",
        (
            f"#p B3LYP/{state.basis} Opt=(CalcFC,MaxCycles=200,NoEigen) Freq "
            f"Integral=UltraFine {scrf} SCF=(XQC,MaxCycle=500)"
        ),
        "",
        (
            f"OROP {record.identifier} {actual_state} (N) for {record.potential_type} potential; "
            f"B97-3c state {state.number} starting geometry"
        ),
        "",
        f"{state.charge} {state.multiplicity}",
        *state.coordinates,
        "",
    ]
    if record.solvent.custom_parameters is not None:
        lines.extend(smd_read_lines(record.solvent.custom_parameters))
        lines.append("")
    return "\n".join(lines) + "\n"


def clean_csv_rows(records: list[Record], potential_type: str) -> list[dict[str, str]]:
    return [
        {"id": str(record.identifier), "电位": record.potential_raw}
        for record in records
        if record.potential_type == potential_type
    ]


def write_clean_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "电位"])
        writer.writeheader()
        writer.writerows(rows)


def ensure_targets(oxidation_csv: Path, reduction_csv: Path, gjf_dir: Path, overwrite: bool) -> None:
    existing = [path for path in (oxidation_csv, reduction_csv) if path.exists()]
    if gjf_dir.is_dir() and list(gjf_dir.glob("*.gjf")):
        existing.append(gjf_dir)
    if existing and not overwrite:
        raise InputDataError(
            "Refusing to overwrite existing 1.1 outputs: "
            + ", ".join(str(path) for path in existing)
            + ". Use --overwrite to regenerate them."
        )


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate OROP potential CSVs and N-state B3LYP/SMD opt+freq GJFs.")
    parser.add_argument("--zip", type=Path, default=root / DEFAULT_ZIP)
    parser.add_argument("--oxidation-csv", type=Path, default=root / OXIDATION_CSV)
    parser.add_argument("--reduction-csv", type=Path, default=root / REDUCTION_CSV)
    parser.add_argument("--gjf-dir", type=Path, default=root / GJF_DIR)
    parser.add_argument("--nprocshared", type=int, default=28)
    parser.add_argument("--mem", default="8GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.nprocshared < 1:
        raise InputDataError("--nprocshared must be positive.")
    if not args.mem.strip():
        raise InputDataError("--mem must not be empty.")

    zip_path = args.zip.resolve()
    oxidation_csv = args.oxidation_csv.resolve()
    reduction_csv = args.reduction_csv.resolve()
    gjf_dir = args.gjf_dir.resolve()
    ensure_targets(oxidation_csv, reduction_csv, gjf_dir, args.overwrite)

    # Fully validate the archive, potential classification, solvent mapping, and
    # starting geometries before creating either CSV or any GJF.
    records = parse_records(zip_path)
    oxidation_rows = clean_csv_rows(records, "oxidation")
    reduction_rows = clean_csv_rows(records, "reduction")
    expected_record_count = 193 - len(PERMANENT_EXCLUDED_IDS)
    if len(oxidation_rows) + len(reduction_rows) != expected_record_count:
        raise InputDataError("Potential CSV partition is incomplete.")

    oxidation_csv.parent.mkdir(parents=True, exist_ok=True)
    reduction_csv.parent.mkdir(parents=True, exist_ok=True)
    gjf_dir.mkdir(parents=True, exist_ok=True)
    write_clean_csv(oxidation_csv, oxidation_rows)
    write_clean_csv(reduction_csv, reduction_rows)
    # An --overwrite rerun must also purge input files from permanently excluded
    # IDs, rather than merely stop generating new files for them.
    for identifier in PERMANENT_EXCLUDED_IDS:
        (gjf_dir / f"{identifier}_opt_sol_N.gjf").unlink(missing_ok=True)
    for record in records:
        gjf_path = gjf_dir / f"{record.identifier}_opt_sol_N.gjf"
        gjf_path.write_text(gjf_text(record, args.nprocshared, args.mem), encoding="utf-8", newline="\n")

    charged_reference_ids = [
        record.identifier for record in records if record.n_reference_state.charge != 0
    ]
    print(f"Wrote {len(oxidation_rows)} rows to {oxidation_csv.name}.")
    print(f"Wrote {len(reduction_rows)} rows to {reduction_csv.name}.")
    print(f"Wrote {len(records)} N-state opt+freq GJFs to {gjf_dir}.")
    print(f"Permanently excluded non-convergent IDs: {sorted(PERMANENT_EXCLUDED_IDS)}")
    print(f"Charged N-reference jobs (no charge-0 structure in the redox couple): {charged_reference_ids}")


if __name__ == "__main__":
    try:
        main()
    except InputDataError as error:
        raise SystemExit(f"ERROR: {error}") from error
