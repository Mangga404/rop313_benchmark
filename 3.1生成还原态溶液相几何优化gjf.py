#!/usr/bin/env python3
"""Generate B3LYP/SMD reduction-state opt+freq GJFs after neutral jobs complete."""
from pathlib import Path
from orop_redox_workflow_common import RedoxStage, WorkflowError, optimization_cli

ROOT = Path(__file__).resolve().parent
STAGE = RedoxStage("red", "还原", ROOT / "orop_reduction_potential.csv", 2, ROOT / "rop313_opt_input_red", ROOT / "rop313_opt_output_sol_red", ROOT / "rop313_sp_input_sol_red", ROOT / "rop313_sp_output_sol_red", ROOT / "rop313_thermo_red.csv", ROOT / "rop313_energy_red.csv")

if __name__ == "__main__":
    try:
        optimization_cli(STAGE)
    except WorkflowError as error:
        raise SystemExit(f"ERROR: {error}") from error
