"""Make the pipeline entry points importable however pytest is invoked."""

import sys
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent
if str(PIPELINES) not in sys.path:
    sys.path.insert(0, str(PIPELINES))
