"""Desktop companion app for IINTS-AF SDK."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "iints-desktop-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

__all__ = ["__version__"]
__version__ = "0.1.0"
