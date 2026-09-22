"""Make the package importable however pytest is invoked."""

import sys
from pathlib import Path

for candidate in (Path(__file__).resolve().parent,
                  Path(__file__).resolve().parents[1] / "panthera_sim"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
