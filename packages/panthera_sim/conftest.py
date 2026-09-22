"""Make the package importable however pytest is invoked.

The modules here import each other by bare name, which is what the rest of the
repository does, and which works only when the package directory happens to be
on the path.  This removes the dependency on the caller's working directory so
the suite runs from the repository root, which is where CI runs it.
"""

import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
