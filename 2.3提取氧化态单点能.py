#!/usr/bin/env python3
"""Extract oxidation-state M062X single-point and Gibbs energies after 2.2 jobs complete."""
from pathlib import Path
from orop_redox_workflow_common import RedoxStage, WorkflowError, energy_extraction_cli

ROOT = Path(__file__).resolve().parent
STAGE = RedoxStage("ox", "氧化", ROOT / "orop_oxidation_potential.csv", 1, ROOT / "rop313_opt_input_ox", ROOT / "rop313_opt_output_sol_ox", ROOT / "rop313_sp_input_ox", ROOT / "rop313_sp_output_sol_ox", ROOT / "rop313_thermo_ox.csv", ROOT / "rop313_energy_ox.csv")

if __name__ == "__main__":
    try:
        energy_extraction_cli(STAGE)
    except WorkflowError as error:
        raise SystemExit(f"ERROR: {error}") from error
