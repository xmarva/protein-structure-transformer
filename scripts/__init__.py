"""Data preparation and training scripts. Run them from the repository root as ``python -m scripts.<name>``.

The ``pst`` package is imported straight from the PST submodule (``git submodule update --init``),
so it does not need to be installed; this adds the submodule to the import path.
"""

import sys
from pathlib import Path

_PST_DIR = Path(__file__).resolve().parent.parent / "PST"
if _PST_DIR.is_dir() and str(_PST_DIR) not in sys.path:
    sys.path.append(str(_PST_DIR))
