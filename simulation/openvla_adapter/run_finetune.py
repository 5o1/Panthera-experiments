#!/usr/bin/env python3
"""Register Panthera's RLDS contract, then run OpenVLA-OFT finetune.py."""

from __future__ import annotations

import os
from pathlib import Path
import runpy

from panthera_rlds import register_openvla_dataset


def main() -> int:
    register_openvla_dataset()
    default_script = (
        Path(__file__).resolve().parents[1]
        / "RLinf/.venv/lib/python3.11/site-packages/vla-scripts/finetune.py"
    )
    script = Path(os.environ.get("PANTHERA_OPENVLA_FINETUNE", default_script))
    if not script.is_file():
        raise FileNotFoundError(f"OpenVLA-OFT finetune.py not found: {script}")
    from experiments.robot import openvla_utils

    package_root = Path(openvla_utils.__file__).resolve().parents[2]
    if not (package_root / "prismatic").is_dir():
        raise FileNotFoundError(
            f"OpenVLA-OFT package root lacks prismatic/: {package_root}"
        )
    os.chdir(package_root)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
