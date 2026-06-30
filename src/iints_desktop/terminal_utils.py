import platform
import subprocess
import shutil
from typing import Optional

def open_terminal_and_run(command: str) -> bool:
    """
    Opens a native system terminal window and executes the given command.
    Returns True if successfully launched, False otherwise.
    """
    system = platform.system().lower()

    try:
        if system == "darwin":
            # macOS: Use AppleScript to open Terminal.app and run the command
            script = f'tell application "Terminal" to do script "{command}"'
            subprocess.Popen(["osascript", "-e", script])
            return True

        elif system == "windows":
            # Windows: Launch a new cmd window that remains open (/k)
            subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", command])
            return True

        elif system == "linux":
            # Linux: Try common terminal emulators
            terminals = [
                ("gnome-terminal", ["--", "bash", "-c", f"{command}; exec bash"]),
                ("xterm", ["-e", f"{command}; bash"]),
                ("konsole", ["-e", f"bash -c '{command}; exec bash'"]),
                ("xfce4-terminal", ["-x", "bash", "-c", f"{command}; exec bash"]),
                ("alacritty", ["-e", "bash", "-c", f"{command}; exec bash"])
            ]
            
            for term, args in terminals:
                if shutil.which(term):
                    subprocess.Popen([term] + args)
                    return True
            
            return False

        else:
            return False
            
    except Exception as e:
        print(f"Failed to open terminal: {e}")
        return False
