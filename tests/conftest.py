from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"

if src_path.exists():
    sys.path.insert(0, str(src_path))
