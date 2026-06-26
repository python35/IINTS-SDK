#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from iints_desktop.qt_app import main as qt_main

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--smoke"]
        return qt_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
