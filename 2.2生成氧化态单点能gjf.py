#!/usr/bin/env python3
"""Generate M062X/SMD oxidation-state single-point GJFs after 2.1 jobs complete."""
from pathlib import Path
from orop_redox_workflow_common import RedoxStage, WorkflowError, single_point_cli

ROOT = Path(__file__).resolve().parent
STAGE = RedoxStage("ox", "氧化", ROOT / "orop_oxidation_potential.csv", 1, ROOT / "rop313_opt_input_ox", ROOT / "rop313_opt_output_sol_ox", ROOT / "rop313_sp_input_ox", ROOT / "rop313_sp_out_ox", ROOT / "rop313_thermo_ox.csv", ROOT / "rop313_energy_ox.csv")

if __name__ == "__main__":
    try:
        single_point_cli(STAGE)
    except WorkflowError as error:
        raise SystemExit(f"ERROR: {error}") from error
