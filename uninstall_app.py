import os
import shutil
import sys
import platform
import time
import subprocess
from pathlib import Path

def delete_path(path: Path) -> None:
    if not path.exists():
        return
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        print(f"[-] Deleted: {path}")
    except Exception as e:
        print(f"[!] Failed to delete {path}: {e}")

def purge_iints(self_destruct: bool = False):
    print("=======================================================")
    print(" Purging all IINTS SDK data, shortcuts, and configurations...")
    print("=======================================================\n")
    
    # 1. Clear QSettings
    try:
        from PySide6.QtCore import QSettings
        settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "IINTS", "DesktopApp")
        settings.clear()
        print("[-] Cleared: Application UI Preferences (QSettings)")
    except ImportError:
        print("[-] PySide6 not found, skipping QSettings clear.")
        
    # 2. Clear history and cache
    delete_path(Path.home() / ".iints-desktop-history.jsonl")
    delete_path(Path.home() / ".cache" / "iints")
    
    # 3. Clear results
    root_dir = Path(__file__).parent.absolute()
    delete_path(root_dir / "results")

    # 4. OS-Specific Shortcuts
    system = platform.system().lower()
    
    if system == "windows":
        desktop = Path.home() / "Desktop"
        start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        for name in ["IINTS Desktop.lnk", "IINTS SDK.lnk"]:
            delete_path(desktop / name)
            if start_menu.exists():
                delete_path(start_menu / name)
                
    elif system == "darwin":
        apps_dirs = [Path("/Applications"), Path.home() / "Applications"]
        for apps_dir in apps_dirs:
            delete_path(apps_dir / "IINTS Desktop.app")
            delete_path(apps_dir / "IINTS SDK.app")
            
    elif system == "linux":
        desktop = Path.home() / "Desktop"
        apps_dir = Path.home() / ".local" / "share" / "applications"
        for name in ["iints-desktop.desktop", "iints.desktop"]:
            delete_path(desktop / name)
            delete_path(apps_dir / name)

    print("\n=======================================================")
    print(" Configuration and Shortcuts Purged.")
    
    # 5. Self-Destruct
    if self_destruct:
        print(f" Initiating Self-Destruct of source directory: {root_dir}")
        print("=======================================================")
        
        # We need to spawn a background process that waits 2 seconds, then deletes the folder,
        # so this script can exit safely without locking the folder on Windows.
        if system == "windows":
            # Create a temporary batch file
            temp_bat = Path(os.environ.get("TEMP", "C:\\Temp")) / "iints_self_destruct.bat"
            with temp_bat.open("w") as f:
                f.write("@echo off\n")
                f.write("timeout /t 2 /nobreak > NUL\n")
                f.write(f'rmdir /s /q "{root_dir}"\n')
                f.write(f'del "%~f0"\n')
            
            subprocess.Popen([str(temp_bat)], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            
        else:
            # Create a temporary shell script
            temp_sh = Path("/tmp/iints_self_destruct.sh")
            with temp_sh.open("w") as f:
                f.write("#!/bin/bash\n")
                f.write("sleep 2\n")
                f.write(f'rm -rf "{root_dir}"\n')
                f.write('rm -f "$0"\n')
            temp_sh.chmod(0o777)
            
            subprocess.Popen([str(temp_sh)], start_new_session=True)
            
        print("[-] Self-destruct sequence initiated. Goodbye.")
        sys.exit(0)
    else:
        print(" If you wish to entirely remove the SDK application")
        print(" from your system, you may safely delete this folder:")
        print(" " + str(root_dir))
        print("=======================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-destruct", action="store_true", help="Delete the SDK folder completely.")
    args = parser.parse_args()
    
    purge_iints(self_destruct=args.self_destruct)
