#!/usr/bin/env python3
"""Generate validated Gaussian opt+freq inputs for the OROP subset of ROP313.

This script reads the supplied SI ZIP directly.  It deliberately addresses only
``rop313/orop``; the ``rop313/omrop`` entries are never enumerated or written.

The B97-3c XYZ files are *starting geometries only*.  Every generated Gaussian
input performs a fresh B3LYP/SMD geometry optimization followed by a frequency
calculation at the same basis-set level.

Usage (from this directory)::

    python "1.生成几何优化gjf.py"

Outputs:
    orop_metadata.csv
    orop_opt_freq_gjf/orop_XXX_state[12]_{oxidized,reduced}_optfreq.gjf

Run with --overwrite when intentionally regenerating existing outputs.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# Reused from the established B3LYP workflow: main-group elements use 6-31G(d),
# while elements heavier than Kr (Z > 36) use Def2SVP.
LIGHT_BASIS: Final = "6-31G(d)"
HEAVY_BASIS: Final = "Def2SVP"

DEFAULT_ZIP: Final = Path("data") / "jp0c05052_si_002.zip"
DEFAULT_METADATA: Final = Path("orop_metadata.csv")
DEFAULT_GJF_DIR: Final = Path("orop_opt_freq_gjf")

STATE_LABELS: Final = {1: "oxidized", 2: "reduced"}
REQUIRED_FILES: Final = (
    ".solv",
    ".ref",
    ".shift",
    ".CHRG1",
    ".CHRG2",
    ".UHF1",
    ".UHF2",
    "1.b973c.xyz",
    "2.b973c.xyz",
)

# The names on the right are Gaussian built-in SMD solvent names.  The aliases
# cover the actual ROP313 .solv values and common spelling variants without
# guessing any physical parameters.
GAUSSIAN_BUILTIN_SMD_SOLVENTS: Final = {
    "acetonitrile": "Acetonitrile",
    "mecn": "Acetonitrile",
    "dmf": "N,N-Dimethylformamide",
    "n,n-dimethylformamide": "N,N-Dimethylformamide",
    "n,n dimethylformamide": "N,N-Dimethylformamide",
}

# SMD descriptors supplied with the request.  They are used only when a .solv
# value is not a mapped Gaussian built-in solvent.  In that case Gaussian is
# explicitly told to use SMD/Generic/Read; no descriptor is inferred.
SMD_PARAMS: Final = {
    'C(=S)=S': [2.64, 1.632, 0.0, 0.0, 32.33, 0.0, 0.0],
    'CS(=O)C': [46.7, 1.479, 0.0, 0.88, 43.54, 0.0, 0.0],
    'C[N+](=O)[O-]': [35.87, 1.382, 0.0, 0.31, 36.66, 0.0, 0.0],
    'CC#N': [36.64, 1.344, 0.07, 0.32, 29.29, 0.0, 0.0],
    'c1ccc(cc1)CO': [12.47, 1.54, 0.33, 0.56, 39.0, 1.0, 0.0],
    'c1ccc(cc1)[N+](=O)[O-]': [34.8, 1.556, 0.0, 0.28, 43.9, 1.0, 0.0],
    'CCC#N': [29.7, 1.366, 0.02, 0.36, 26.7, 0.0, 0.0],
    'c1ccc(cc1)C#N': [25.2, 1.528, 0.0, 0.33, 39.0, 1.0, 0.0],
    'CCCC#N': [24.8, 1.384, 0.0, 0.37, 27.6, 0.0, 0.0],
    'COc1ccccc1': [4.33, 1.517, 0.0, 0.22, 35.2, 1.0, 0.0],
    'Cc1ccc(cc1)C': [2.27, 1.495, 0.0, 0.12, 28.01, 1.0, 0.0],
    'Cc1ccccc1': [2.38, 1.496, 0.0, 0.14, 28.52, 1.0, 0.0],
    'CCN(CC)CC': [2.42, 1.4, 0.0, 0.14, 20.2, 0.0, 0.0],
    'c1ccc(cc1)Cl': [5.62, 1.524, 0.0, 0.07, 33.03, 1.0, 0.1],
    'Cc1cccc(c1)C': [2.37, 1.497, 0.0, 0.12, 28.53, 1.0, 0.0],
    'c1ccc(cc1)I': [4.66, 1.62, 0.0, 0.0, 39.14, 1.0, 0.1],
    'c1ccc(cc1)Br': [5.4, 1.559, 0.0, 0.06, 36.34, 1.0, 0.1],
    'CCc1ccccc1': [2.4, 1.495, 0.0, 0.15, 29.2, 1.0, 0.0],
    'Cc1ccccc1C': [2.57, 1.505, 0.0, 0.16, 30.1, 1.0, 0.0],
    'c1ccc(cc1)N': [6.89, 1.586, 0.26, 0.41, 42.9, 1.0, 0.0],
    'CCOC(=O)c1ccccc1': [6.02, 1.505, 0.0, 0.41, 35.5, 1.0, 0.0],
    'c1(c(c(c(c(c1F)F)F)F)F)F': [2.03, 1.367, 0.0, 0.0, 22.6, 1.0, 0.6],
    'c1ccc(cc1)F': [5.42, 1.465, 0.0, 0.1, 27.7, 1.0, 0.1],
    'c2ccc(COCc1ccccc1)cc2': [3.8, 1.562, 0.0, 0.2, 38.0, 1.0, 0.0],
    'CC[N+](=O)[O-]': [28.5, 1.392, 0.02, 0.25, 36.0, 0.0, 0.0],
    'CCOc1ccccc1': [4.22, 1.507, 0.0, 0.24, 33.5, 1.0, 0.0],
    'Cc1cccc(c1)O': [11.5, 1.539, 0.57, 0.34, 35.0, 1.0, 0.0],
    'CCCCCCCCCCCCCCC=C': [2.1, 1.44, 0.0, 0.07, 28.0, 0.0, 0.0],
    'CCCCCCCCCCCC': [2.01, 1.422, 0.0, 0.0, 25.3, 0.0, 0.0],
    'CCCCCCCCCC': [2.01, 1.412, 0.0, 0.0, 23.8, 0.0, 0.0],
    'CCCCCCCCC': [1.97, 1.405, 0.0, 0.0, 22.9, 0.0, 0.0],
    'CC(C)CC(C)(C)C': [1.94, 1.391, 0.0, 0.0, 18.77, 0.0, 0.0],
    'CCCCCCCC': [1.95, 1.397, 0.0, 0.0, 21.6, 0.0, 0.0],
    'CCCCCCC': [1.92, 1.387, 0.0, 0.0, 20.1, 0.0, 0.0],
    'CCCCCC': [1.88, 1.375, 0.0, 0.0, 18.4, 0.0, 0.0],
    'CCCCC': [1.84, 1.357, 0.0, 0.0, 16.0, 0.0, 0.0],
    'CCCCCCCCCCCCCCCC': [2.05, 1.434, 0.0, 0.0, 27.47, 0.0, 0.0],
    'CCCCCCCCCCCCCC': [2.04, 1.427, 0.0, 0.0, 26.5, 0.0, 0.0],
    'CC1CCCCC1': [2.02, 1.423, 0.0, 0.0, 23.6, 0.0, 0.0],
    'CCCCCCCCCCC': [2.01, 1.417, 0.0, 0.0, 24.9, 0.0, 0.0],
    'C1CCCCCCC1': [2.23, 1.459, 0.0, 0.0, 30.0, 0.0, 0.0],
    'CCCCCCCCO': [3.4, 1.429, 0.37, 0.48, 27.5, 0.0, 0.0],
    'CCCCCCCCCCO': [3.1, 1.437, 0.37, 0.48, 28.5, 0.0, 0.0],
    'CCCCCCO': [6.7, 1.418, 0.37, 0.48, 24.4, 0.0, 0.0],
    'CCCCCCCO': [4.6, 1.424, 0.37, 0.48, 26.1, 0.0, 0.0],
    'CCCCCC(C)O': [6.0, 1.42, 0.34, 0.48, 25.5, 0.0, 0.0],
    'CCCCCCCCCCCO': [2.8, 1.44, 0.37, 0.48, 29.3, 0.0, 0.0],
    'CCCCCCCCCO': [3.2, 1.433, 0.37, 0.48, 28.1, 0.0, 0.0],
    'CCCC(=O)OCC': [5.0, 1.4, 0.0, 0.45, 24.2, 0.0, 0.0],
    'CCCCCC(=O)OCC': [4.0, 1.413, 0.0, 0.45, 25.8, 0.0, 0.0],
    'CCCCC(CC)CO': [4.8, 1.433, 0.37, 0.48, 27.0, 0.0, 0.0],
    'CC1COC(=O)O1': [64.0, 1.422, 0.0, 0.4, 41.9, 0.0, 0.0],
    'C/N=C/O': [10.0, 1.43, 0.2, 0.5, 30.0, 0.0, 0.0],
    'CN(C)C=O': [36.7, 1.43, 0.0, 0.69, 36.76, 0.0, 0.0],
    'CC(=O)N(C)C': [37.8, 1.438, 0.0, 0.76, 33.7, 0.0, 0.0],
    'CC(=O)OC': [6.68, 1.359, 0.0, 0.42, 24.8, 0.0, 0.0],
    'CCOC(=O)C': [6.02, 1.372, 0.0, 0.45, 23.9, 0.0, 0.0],
    'CCCCOC(=O)C': [5.01, 1.394, 0.0, 0.45, 25.2, 0.0, 0.0],
    'C/C(=N/C)/O': [15.0, 1.44, 0.15, 0.6, 35.0, 0.0, 0.0],
    'CCCCN(CCCC)C=O': [12.0, 1.44, 0.0, 0.7, 28.0, 0.0, 0.0],
    'C(=O)N': [109.0, 1.447, 0.62, 0.48, 58.0, 0.0, 0.0],
    'CCCCCCOC(=O)C': [4.4, 1.409, 0.0, 0.45, 26.3, 0.0, 0.0],
    'CC(=O)O': [6.15, 1.372, 0.61, 0.45, 27.8, 0.0, 0.0],
    'CCCCOP(=O)(OCCCC)OCCCC': [8.1, 1.425, 0.0, 0.9, 27.2, 0.0, 0.0],
    'CCNC(=O)C': [50.0, 1.43, 0.3, 0.6, 32.0, 0.0, 0.0],
    'CCNC=O': [60.0, 1.43, 0.3, 0.6, 34.0, 0.0, 0.0],
    'CCN(CC)C(=O)C': [30.0, 1.44, 0.0, 0.7, 30.0, 0.0, 0.0],
    'CC(C)COC(=O)C': [5.3, 1.39, 0.0, 0.45, 23.7, 0.0, 0.0],
    'CC(C)OC(=O)C': [6.0, 1.377, 0.0, 0.45, 21.7, 0.0, 0.0],
    'CCCCCOC(=O)C': [4.7, 1.402, 0.0, 0.45, 25.8, 0.0, 0.0],
    'CCCOC(=O)C': [6.0, 1.389, 0.0, 0.45, 24.3, 0.0, 0.0],
    'CCCCCCCOC(=O)C': [4.2, 1.415, 0.0, 0.45, 27.0, 0.0, 0.0],
    'CO': [32.6, 1.329, 0.43, 0.47, 22.1, 0.0, 0.0],
    'CC(=O)c1ccccc1': [17.39, 1.534, 0.0, 0.48, 39.0, 1.0, 0.0],
    'CCCO': [20.4, 1.385, 0.37, 0.48, 23.3, 0.0, 0.0],
    'CCO': [24.3, 1.361, 0.37, 0.48, 21.97, 0.0, 0.0],
    'CC(C)O': [17.9, 1.377, 0.33, 0.56, 20.9, 0.0, 0.0],
    'CC(=O)C': [20.7, 1.359, 0.04, 0.48, 23.0, 0.0, 0.0],
    'CCCCO': [17.5, 1.399, 0.37, 0.48, 24.2, 0.0, 0.0],
    'C1CCC(=O)CC1': [15.5, 1.45, 0.0, 0.5, 34.5, 0.0, 0.0],
    'CCC(=O)C': [18.5, 1.378, 0.0, 0.48, 24.0, 0.0, 0.0],
    'CCCCCO': [13.9, 1.41, 0.37, 0.48, 25.3, 0.0, 0.0],
    'CC(C)(C)OC': [4.5, 1.369, 0.0, 0.4, 19.4, 0.0, 0.0],
    'CC(C)(C)O': [12.4, 1.387, 0.31, 0.6, 17.0, 0.0, 0.0],
    'CCCCOCCOCCOCCOCCCC': [6.0, 1.43, 0.0, 0.8, 28.0, 0.0, 0.0],
    'CCCCOCCOCCOCCCC': [6.5, 1.43, 0.0, 0.8, 28.0, 0.0, 0.0],
    'CC(C)OC(C)C': [3.8, 1.368, 0.0, 0.49, 16.0, 0.0, 0.0],
    'COCCOCCOCCOCCOC': [7.0, 1.43, 0.0, 1.0, 30.0, 0.0, 0.0],
    'CC(C)CO': [17.2, 1.395, 0.37, 0.48, 22.8, 0.0, 0.0],
    'C(CO)O': [37.7, 1.431, 0.57, 0.52, 47.7, 0.0, 0.0],
    'C=CCO': [19.0, 1.413, 0.3, 0.5, 25.0, 0.0, 0.0],
    'CCC(C)CO': [15.0, 1.406, 0.37, 0.48, 23.5, 0.0, 0.0],
    'CC(C)CCO': [14.0, 1.406, 0.37, 0.48, 23.5, 0.0, 0.0],
    'CCC(CC)O': [13.0, 1.41, 0.37, 0.48, 24.5, 0.0, 0.0],
    'CCCC(C)O': [13.0, 1.406, 0.37, 0.48, 23.5, 0.0, 0.0],
    'CCC(C)(C)O': [11.0, 1.405, 0.31, 0.6, 20.0, 0.0, 0.0],
    'COCCO': [16.9, 1.4, 0.3, 0.8, 30.0, 0.0, 0.0],
    'CCOCCO': [13.0, 1.408, 0.3, 0.8, 29.0, 0.0, 0.0],
    'CCCOCCC': [3.4, 1.405, 0.0, 0.45, 23.0, 0.0, 0.0],
    'CCCCOCCCC': [3.0, 1.41, 0.0, 0.45, 24.0, 0.0, 0.0],
    'CC(CO)O': [28.0, 1.432, 0.5, 0.6, 36.0, 0.0, 0.0],
    'CC(C)OCCO': [11.0, 1.408, 0.3, 0.8, 28.0, 0.0, 0.0],
    'CCCOCCO': [10.0, 1.415, 0.3, 0.8, 28.5, 0.0, 0.0],
    'CCCCOCCO': [9.0, 1.42, 0.3, 0.8, 29.0, 0.0, 0.0],
    'CCC(C)O': [15.8, 1.397, 0.33, 0.56, 22.5, 0.0, 0.0],
    'CCCCCC(=O)C': [10.0, 1.41, 0.0, 0.5, 26.0, 0.0, 0.0],
    'CCOCC': [4.33, 1.353, 0.0, 0.45, 17.0, 0.0, 0.0],
    'CCCCOC': [3.5, 1.39, 0.0, 0.45, 22.0, 0.0, 0.0],
    'CCC(C)(C)OC': [4.0, 1.39, 0.0, 0.45, 21.0, 0.0, 0.0],
    'CCC(=O)CC': [14.0, 1.392, 0.0, 0.5, 24.5, 0.0, 0.0],
    'CC(C)CC(=O)C': [13.0, 1.4, 0.0, 0.5, 23.5, 0.0, 0.0],
    'CCOCCOCCOCC': [12.0, 1.42, 0.0, 1.0, 31.0, 0.0, 0.0],
    'CCCCC(=O)C': [12.0, 1.4, 0.0, 0.5, 25.0, 0.0, 0.0],
    'CCCC(=O)C': [15.0, 1.39, 0.0, 0.5, 24.5, 0.0, 0.0],
    'CCOC(C)(C)C': [4.0, 1.38, 0.0, 0.45, 20.0, 0.0, 0.0],
    'CCCOCC': [3.8, 1.39, 0.0, 0.45, 21.0, 0.0, 0.0],
    'CCCC(=O)CC': [11.0, 1.4, 0.0, 0.5, 25.5, 0.0, 0.0],
    'CCCCOCC': [3.5, 1.4, 0.0, 0.45, 23.0, 0.0, 0.0],
    'CCCCC(C)O': [10.0, 1.41, 0.37, 0.48, 25.0, 0.0, 0.0],
    'CC(C)OC': [4.0, 1.37, 0.0, 0.45, 18.0, 0.0, 0.0],
    'CCCOC': [4.0, 1.38, 0.0, 0.45, 20.0, 0.0, 0.0],
    'CCCCC(CCC)O': [6.0, 1.43, 0.37, 0.48, 27.0, 0.0, 0.0],
    'CC(C)CC(C)O': [10.0, 1.41, 0.37, 0.48, 24.0, 0.0, 0.0],
    'CCCC(CCC)O': [5.0, 1.43, 0.37, 0.48, 27.5, 0.0, 0.0],
    'CCCC(CC)O': [7.0, 1.42, 0.37, 0.48, 26.5, 0.0, 0.0],
    'C(CCl)Cl': [10.36, 1.444, 0.1, 0.11, 32.42, 0.0, 0.2],
    'C(Cl)Cl': [8.93, 1.424, 0.1, 0.05, 28.12, 0.0, 0.2],
    'C(Cl)(Cl)Cl': [4.81, 1.446, 0.15, 0.02, 27.1, 0.0, 0.3],
    'C(Cl)(Cl)(Cl)Cl': [2.24, 1.46, 0.0, 0.0, 26.95, 0.0, 0.4],
    'C(I)I': [5.0, 1.73, 0.1, 0.1, 45.0, 0.0, 0.2],
    'CCCCCl': [7.39, 1.402, 0.0, 0.05, 25.8, 0.0, 0.1],
    'CCBr': [9.4, 1.424, 0.0, 0.06, 24.2, 0.0, 0.1],
    'C1CC(=O)OC1': [39.0, 1.434, 0.0, 0.4, 43.5, 0.0, 0.0],
    'CN1CCCC1=O': [32.2, 1.47, 0.0, 0.77, 40.7, 0.0, 0.0],
    'CN1CCCCC1=O': [30.0, 1.48, 0.0, 0.8, 38.0, 0.0, 0.0],
    'C1COCCO1': [2.21, 1.422, 0.0, 0.25, 32.9, 0.0, 0.0],
    'c1ccncc1': [12.4, 1.509, 0.0, 0.52, 36.5, 1.0, 0.0],
    'C1CCOC1': [7.58, 1.407, 0.0, 0.48, 26.4, 0.0, 0.0],
    'C1COCCN1C=O': [40.0, 1.48, 0.0, 0.8, 42.0, 0.0, 0.0],
    'C1CCS(=O)(=O)C1': [43.0, 1.48, 0.0, 0.8, 45.0, 0.0, 0.0],
    'C1CCOCC1': [2.8, 1.42, 0.0, 0.25, 33.0, 0.0, 0.0],
    'C1CCC(=O)OCC1': [35.0, 1.46, 0.0, 0.5, 40.0, 0.0, 0.0],
    'CCN1CCCC1=O': [28.0, 1.46, 0.0, 0.8, 38.0, 0.0, 0.0],
    'CC1CCC(=O)N1C': [25.0, 1.47, 0.0, 0.8, 35.0, 0.0, 0.0],
    'Cc1ccccn1': [9.5, 1.5, 0.0, 0.55, 33.0, 1.0, 0.0],
    'O': [78.3, 1.333, 0.82, 0.35, 72.8, 0.0, 0.0],
}

# Only elements present in valid OROP XYZ files need to be handled, but keeping
# the table through Bi makes validation explicit for all elements supported by
# the established B3LYP reference scripts.
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


class BenchmarkInputError(RuntimeError):
    """Raised when supplied ROP313 data cannot be converted without guessing."""


@dataclass(frozen=True)
class SolventSetup:
    raw_value: str
    mode: str  # gaussian_builtin or custom_read
    gaussian_name: str
    parameters: tuple[float, ...] | None = None


@dataclass(frozen=True)
class StateInput:
    state: int
    label: str
    charge: int
    uhf: int
    multiplicity: int
    basis: str
    coordinates: tuple[str, ...]
    atom_count: int


@dataclass(frozen=True)
class OROPRecord:
    identifier: int
    solvent: SolventSetup
    experimental_reference_raw: str
    experimental_reference: float
    potential_type: str  # oxidation or reduction
    reference_shift_raw: str
    reference_shift: float
    states: tuple[StateInput, StateInput]


def normalise_solvent_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def resolve_solvent(raw_value: str, identifier: int) -> SolventSetup:
    """Map a .solv entry to Gaussian built-in SMD or supplied Generic/Read data."""
    key = normalise_solvent_name(raw_value)
    builtin_name = GAUSSIAN_BUILTIN_SMD_SOLVENTS.get(key)
    if builtin_name is not None:
        return SolventSetup(raw_value, "gaussian_builtin", builtin_name)

    # SMD_PARAMS keys are SMILES and therefore intentionally case-sensitive.
    parameters = SMD_PARAMS.get(raw_value.strip())
    if parameters is None:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: solvent {raw_value!r} cannot be mapped to a "
            "Gaussian built-in SMD solvent and is absent from SMD_PARAMS. "
            "No parameters were guessed."
        )
    return SolventSetup(raw_value, "custom_read", "Generic", tuple(parameters))


def find_orop_root(names: list[str]) -> str:
    """Return the ZIP-internal path ending in rop313/orop/ and reject ambiguity."""
    pattern = re.compile(r"^(?P<root>.*?rop313/orop/)(?P<id>\d+)/\.solv$")
    roots = {match.group("root") for name in names if (match := pattern.match(name))}
    if len(roots) != 1:
        raise BenchmarkInputError(
            "Expected exactly one rop313/orop directory in the ZIP; found "
            f"{sorted(roots)!r}."
        )
    return roots.pop()


def read_required_text(archive: zipfile.ZipFile, member: str, identifier: int) -> str:
    try:
        return archive.read(member).decode("utf-8-sig").strip()
    except KeyError as error:
        raise BenchmarkInputError(f"OROP ID {identifier}: required file missing: {member}") from error
    except UnicodeDecodeError as error:
        raise BenchmarkInputError(f"OROP ID {identifier}: {member} is not valid UTF-8 text.") from error


def parse_integer(raw_value: str, member: str, identifier: int) -> int:
    try:
        return int(raw_value.strip())
    except ValueError as error:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: {member} must contain one integer, got {raw_value!r}."
        ) from error


def parse_float(raw_value: str, member: str, identifier: int) -> float:
    try:
        return float(raw_value.strip())
    except ValueError as error:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: {member} must contain one numeric value, got {raw_value!r}."
        ) from error


def canonical_element(raw_symbol: str, identifier: int, member: str) -> str:
    symbol = raw_symbol[:1].upper() + raw_symbol[1:].lower()
    if symbol not in ATOMIC_NUMBERS:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: unsupported element {raw_symbol!r} in {member}."
        )
    return symbol


def parse_xyz(xyz_text: str, identifier: int, member: str) -> tuple[tuple[str, ...], int]:
    """Validate an XYZ file and return Gaussian coordinate lines and atom count."""
    lines = xyz_text.splitlines()
    if len(lines) < 3:
        raise BenchmarkInputError(f"OROP ID {identifier}: malformed XYZ file {member}.")
    try:
        declared_atom_count = int(lines[0].strip())
    except ValueError as error:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: invalid XYZ atom count in {member}: {lines[0]!r}."
        ) from error

    coordinate_lines = [line for line in lines[2:] if line.strip()]
    if len(coordinate_lines) != declared_atom_count:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: {member} declares {declared_atom_count} atoms but "
            f"contains {len(coordinate_lines)} coordinate lines."
        )

    gaussian_lines: list[str] = []
    for line_number, line in enumerate(coordinate_lines, start=3):
        fields = line.split()
        if len(fields) != 4:
            raise BenchmarkInputError(
                f"OROP ID {identifier}: {member} line {line_number} must have exactly "
                f"four XYZ fields, got {line!r}."
            )
        symbol = canonical_element(fields[0], identifier, member)
        try:
            # Validate numeric coordinate tokens but preserve their original precision in the GJF.
            float(fields[1])
            float(fields[2])
            float(fields[3])
        except ValueError as error:
            raise BenchmarkInputError(
                f"OROP ID {identifier}: non-numeric coordinate in {member} line {line_number}."
            ) from error
        gaussian_lines.append(f"{symbol:<2} {fields[1]:>18} {fields[2]:>18} {fields[3]:>18}")

    return tuple(gaussian_lines), declared_atom_count


def choose_basis(coordinates: tuple[str, ...]) -> str:
    atomic_numbers = [ATOMIC_NUMBERS[line.split()[0]] for line in coordinates]
    return HEAVY_BASIS if any(number > 36 for number in atomic_numbers) else LIGHT_BASIS


def classify_potential_type(identifier: int, state1: StateInput, state2: StateInput) -> str:
    """Classify the experimental .ref value from the charge-defined redox couple.

    Structure 1 is oxidized and structure 2 is reduced.  For this one-electron
    data set, a non-negative charge couple (for example 0 -> +1 or +1 -> +2)
    is reported as an oxidation potential, whereas a non-positive couple
    (for example 0 -> -1 or -1 -> -2) is reported as a reduction potential.
    Any other charge pattern is rejected rather than assigned by assumption.
    """
    if state1.charge != state2.charge + 1:
        raise BenchmarkInputError(
            f"OROP ID {identifier}: state1/state2 charges ({state1.charge}, {state2.charge}) "
            "do not describe the required one-electron redox couple."
        )
    if state1.charge > 0 and state2.charge >= 0:
        return "oxidation"
    if state1.charge <= 0 and state2.charge < 0:
        return "reduction"
    raise BenchmarkInputError(
        f"OROP ID {identifier}: cannot classify charge pair "
        f"{state1.charge} -> {state2.charge} as oxidation or reduction without guessing."
    )


def parse_record(archive: zipfile.ZipFile, orop_root: str, identifier: int) -> OROPRecord:
    member = lambda filename: f"{orop_root}{identifier}/{filename}"
    raw_files = {filename: read_required_text(archive, member(filename), identifier) for filename in REQUIRED_FILES}

    solvent = resolve_solvent(raw_files[".solv"], identifier)
    states: list[StateInput] = []
    for state in (1, 2):
        charge = parse_integer(raw_files[f".CHRG{state}"], f".CHRG{state}", identifier)
        uhf = parse_integer(raw_files[f".UHF{state}"], f".UHF{state}", identifier)
        if uhf not in (0, 1):
            raise BenchmarkInputError(
                f"OROP ID {identifier} state {state}: .UHF{state} is {uhf}; "
                "OROP Gaussian multiplicity is defined only for UHF values 0 or 1."
            )
        multiplicity = uhf + 1
        xyz_member = f"{state}.b973c.xyz"
        coordinates, atom_count = parse_xyz(raw_files[xyz_member], identifier, xyz_member)
        states.append(
            StateInput(
                state=state,
                label=STATE_LABELS[state],
                charge=charge,
                uhf=uhf,
                multiplicity=multiplicity,
                basis=choose_basis(coordinates),
                coordinates=coordinates,
                atom_count=atom_count,
            )
        )

    state_pair = (states[0], states[1])
    return OROPRecord(
        identifier=identifier,
        solvent=solvent,
        experimental_reference_raw=raw_files[".ref"],
        experimental_reference=parse_float(raw_files[".ref"], ".ref", identifier),
        potential_type=classify_potential_type(identifier, *state_pair),
        reference_shift_raw=raw_files[".shift"],
        reference_shift=parse_float(raw_files[".shift"], ".shift", identifier),
        states=state_pair,
    )


def custom_smd_lines(parameters: tuple[float, ...]) -> list[str]:
    if len(parameters) != 7:
        raise BenchmarkInputError(f"Internal error: expected seven SMD parameters, got {parameters!r}.")
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


def gaussian_route(state: StateInput, solvent: SolventSetup) -> str:
    # The frequency calculation is intentionally part of the same input job and
    # uses the same B3LYP/SMD and basis level as the reoptimization.
    scrf = (
        f"SCRF=(SMD,Solvent={solvent.gaussian_name})"
        if solvent.mode == "gaussian_builtin"
        else "SCRF=(SMD,Solvent=Generic,Read)"
    )
    return (
        f"#p B3LYP/{state.basis} Opt=(CalcFC,MaxCycles=200,NoEigen) Freq "
        f"Integral=UltraFine {scrf} SCF=(XQC,MaxCycle=500)"
    )


def gjf_text(record: OROPRecord, state: StateInput, nprocshared: int, memory: str) -> str:
    lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={memory}",
        gaussian_route(state, record.solvent),
        "",
        (
            f"OROP {record.identifier} state {state.state} ({state.label}); "
            "B97-3c XYZ starting geometry, B3LYP/SMD opt+freq"
        ),
        "",
        f"{state.charge} {state.multiplicity}",
        *state.coordinates,
        "",
    ]
    if record.solvent.mode == "custom_read":
        assert record.solvent.parameters is not None
        lines.extend(custom_smd_lines(record.solvent.parameters))
        lines.append("")
    return "\n".join(lines) + "\n"


def output_filename(identifier: int, state: StateInput) -> str:
    return f"orop_{identifier:03d}_state{state.state}_{state.label}_optfreq.gjf"


def metadata_row(record: OROPRecord, gjf_dir: Path) -> dict[str, object]:
    state1, state2 = record.states
    oxidation_potential = record.experimental_reference if record.potential_type == "oxidation" else ""
    reduction_potential = record.experimental_reference if record.potential_type == "reduction" else ""
    return {
        "orop_id": record.identifier,
        "solvent_raw": record.solvent.raw_value,
        "smd_mode": record.solvent.mode,
        "gaussian_smd_solvent": record.solvent.gaussian_name,
        "experimental_reference_potential_raw": record.experimental_reference_raw,
        "experimental_reference_potential": record.experimental_reference,
        "experimental_potential_type": record.potential_type,
        "experimental_oxidation_potential": oxidation_potential,
        "experimental_reduction_potential": reduction_potential,
        "reference_shift_raw": record.reference_shift_raw,
        "reference_shift": record.reference_shift,
        "state1_assignment": state1.label,
        "state1_charge_raw": state1.charge,
        "state1_uhf_raw": state1.uhf,
        "state1_multiplicity": state1.multiplicity,
        "state1_basis": state1.basis,
        "state1_initial_xyz": "1.b973c.xyz",
        "state1_atom_count": state1.atom_count,
        "state1_gjf": (gjf_dir / output_filename(record.identifier, state1)).as_posix(),
        "state2_assignment": state2.label,
        "state2_charge_raw": state2.charge,
        "state2_uhf_raw": state2.uhf,
        "state2_multiplicity": state2.multiplicity,
        "state2_basis": state2.basis,
        "state2_initial_xyz": "2.b973c.xyz",
        "state2_atom_count": state2.atom_count,
        "state2_gjf": (gjf_dir / output_filename(record.identifier, state2)).as_posix(),
    }


def write_outputs(
    records: list[OROPRecord], metadata_path: Path, gjf_dir: Path, nprocshared: int, memory: str
) -> None:
    gjf_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    for record in records:
        for state in record.states:
            gjf_path = gjf_dir / output_filename(record.identifier, state)
            gjf_path.write_text(gjf_text(record, state, nprocshared, memory), encoding="utf-8", newline="\n")

    rows = [metadata_row(record, gjf_dir) for record in records]
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_targets_are_safe(metadata_path: Path, gjf_dir: Path, overwrite: bool) -> None:
    existing_gjfs = list(gjf_dir.glob("*.gjf")) if gjf_dir.is_dir() else []
    metadata_exists = metadata_path.exists()
    if (existing_gjfs or metadata_exists) and not overwrite:
        existing = []
        if metadata_exists:
            existing.append(str(metadata_path))
        if existing_gjfs:
            existing.append(f"{len(existing_gjfs)} GJF file(s) in {gjf_dir}")
        raise BenchmarkInputError(
            "Refusing to overwrite existing outputs: " + "; ".join(existing) + ". Use --overwrite to regenerate."
        )


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate B3LYP/SMD opt+freq GJFs for the 193-entry ROP313 OROP subset."
    )
    parser.add_argument("--zip", type=Path, default=script_dir / DEFAULT_ZIP, help="ROP313 SI ZIP path")
    parser.add_argument(
        "--metadata", type=Path, default=script_dir / DEFAULT_METADATA, help="metadata CSV output path"
    )
    parser.add_argument(
        "--gjf-dir", type=Path, default=script_dir / DEFAULT_GJF_DIR, help="directory for generated GJFs"
    )
    parser.add_argument("--nprocshared", type=int, default=28, help="Gaussian %%nprocshared value (default: 28)")
    parser.add_argument("--mem", default="8GB", help="Gaussian %%mem value (default: 8GB)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite prior CSV and GJF outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    zip_path = args.zip.resolve()
    metadata_path = args.metadata.resolve()
    gjf_dir = args.gjf_dir.resolve()

    if args.nprocshared < 1:
        raise BenchmarkInputError("--nprocshared must be a positive integer.")
    if not args.mem.strip():
        raise BenchmarkInputError("--mem cannot be empty.")
    if not zip_path.is_file():
        raise BenchmarkInputError(f"ROP313 ZIP not found: {zip_path}")
    ensure_output_targets_are_safe(metadata_path, gjf_dir, args.overwrite)

    # Parse and validate the *entire* OROP subset before writing any artifact.
    # Therefore an unmapped solvent or corrupt structure cannot yield a partly
    # generated benchmark input set.
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        orop_root = find_orop_root(names)
        id_pattern = re.compile(r"^" + re.escape(orop_root) + r"(?P<id>\d+)/\.solv$")
        identifiers = sorted(
            int(match.group("id")) for name in names if (match := id_pattern.match(name))
        )
        if len(identifiers) != 193 or len(set(identifiers)) != 193:
            raise BenchmarkInputError(
                f"Expected exactly 193 OROP IDs, found {len(identifiers)} entries: {identifiers!r}."
            )
        if identifiers != list(range(1, 194)):
            raise BenchmarkInputError(
                "OROP IDs must be the complete 1..193 range; found "
                f"{identifiers[0]}..{identifiers[-1]} with gaps or unexpected IDs."
            )
        records = [parse_record(archive, orop_root, identifier) for identifier in identifiers]

    solvent_counts = Counter(record.solvent.raw_value.casefold() for record in records)
    expected_counts = Counter({"acetonitrile": 191, "dmf": 2})
    if solvent_counts != expected_counts:
        raise BenchmarkInputError(
            "Unexpected OROP solvent distribution. Expected 191 acetonitrile and 2 dmf; "
            f"found {dict(sorted(solvent_counts.items()))!r}."
        )

    write_outputs(records, metadata_path, gjf_dir, args.nprocshared, args.mem)
    basis_counts = Counter(state.basis for record in records for state in record.states)
    potential_counts = Counter(record.potential_type for record in records)
    print(f"Validated {len(records)} OROP IDs and wrote {2 * len(records)} Gaussian opt+freq GJFs.")
    print("Solvents:", ", ".join(f"{solvent}={count}" for solvent, count in sorted(solvent_counts.items())))
    print("Potential types:", ", ".join(f"{kind}={count}" for kind, count in sorted(potential_counts.items())))
    print("Basis assignments:", ", ".join(f"{basis}={count}" for basis, count in sorted(basis_counts.items())))
    print(f"Metadata: {metadata_path}")
    print(f"GJFs: {gjf_dir}")


if __name__ == "__main__":
    try:
        main()
    except BenchmarkInputError as error:
        raise SystemExit(f"ERROR: {error}") from error
