"""Project-wide configuration: geometric constants and filesystem paths.

Originally the notebooks pointed the ``root`` variable at a Google Drive
folder (``/content/drive/MyDrive/PINN_piezo/``). When the code is run
outside Colab those locations are not available, so the paths are now
resolved relative to the repository root and can be overridden through
the ``PINN_PIEZO_DATA_DIR`` / ``PINN_PIEZO_MODELS_DIR`` /
``PINN_PIEZO_OUTPUTS_DIR`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("PINN_PIEZO_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("PINN_PIEZO_MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.environ.get("PINN_PIEZO_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

RUNS_DIR = OUTPUTS_DIR / "runs"

for _dir in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, RUNS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# Beam geometry (m)
WIDTH = 100e-3
HEIGHT = 1e-3
H2 = HEIGHT / 2
CENTER = H2
Y_TOP = HEIGHT
Y_BOTTOM = 0

# Reference scales (used by the indirect notebook)
U_c = 1e-9
SIGMA_C = 1e-6
VOLTAGE = 100
D_c = 8.854e-7

# Length / stress / potential / displacement-field reference values
L0 = 1e-3      # 1 mm
SIGMA_0 = 1e9  # Pa
PHI_0 = 1      # V
D0 = 1e-3      # C/m^2


def get_device() -> torch.device:
    """Return the preferred torch device (CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
