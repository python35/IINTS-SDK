import os
import shutil
from pathlib import Path

def purge_iints():
    print("=======================================================")
    print(" Purging all IINTS SDK data and configurations...")
    print("=======================================================\n")
    
    # 1. Clear QSettings
    try:
        from PySide6.QtCore import QSettings
        settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "IINTS", "DesktopApp")
        settings.clear()
        print("[-] Cleared: Application UI Preferences (QSettings)")
    except ImportError:
        print("[!] PySide6 not installed, skipping QSettings clear.")
        
    # 2. Clear history file
    history = Path.home() / ".iints-desktop-history.jsonl"
    if history.exists():
        history.unlink()
        print(f"[-] Deleted: {history}")
        
    # 3. Clear cache directory
    cache = Path.home() / ".cache" / "iints"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
        print(f"[-] Deleted: {cache}")

    print("\n=======================================================")
    print(" Purge complete!")
    print(" If you wish to entirely remove the SDK application")
    print(" from your system, you may safely delete this folder:")
    print(" " + str(Path(__file__).parent.absolute()))
    print("=======================================================")

if __name__ == "__main__":
    purge_iints()
