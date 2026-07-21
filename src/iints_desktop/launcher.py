from __future__ import annotations

import sys


def main() -> int | None:
    """Launch the best available desktop shell.

    PySide6/Qt is preferred because it gives the SDK a more polished native app
    feel. Tkinter stays available as a lightweight fallback and for systems
    where Qt is not installed.
    """

    args = set(sys.argv[1:])
    if "--tk" in args:
        from iints_desktop.app import main as tk_main

        tk_main()
        return None

    if "--qt" in args:
        from iints_desktop.qt_app import main as qt_main

        return qt_main()

    from iints_desktop.qt_app import _PYSIDE_IMPORT_ERROR, main as qt_main

    if _PYSIDE_IMPORT_ERROR is None:
        return qt_main()

    print(
        "PySide6 is not installed; falling back to the Tkinter desktop app. "
        'Install the full desktop app with: python -m pip install -U "iints-sdk-python35[desktop-all]"',
        file=sys.stderr,
    )
    from iints_desktop.app import main as tk_main

    tk_main()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
