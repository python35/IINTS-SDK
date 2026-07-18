from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Sequence

from iints_desktop.update import format_shell_command


def _command_to_shell_text(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    return format_shell_command([str(part) for part in command])


def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _hold_open_shell(command_text: str) -> str:
    return f"{command_text}; echo; echo 'IINTS update finished. You may close this terminal.'; exec bash"


def _hold_open_zsh(command_text: str) -> str:
    return f"{command_text}; echo; echo 'IINTS update finished. You may close this terminal.'; exec zsh"


def open_terminal_and_run(command: str | Sequence[str]) -> bool:
    """
    Open a native terminal window and execute a prebuilt command.

    The function accepts either a shell string or a list of argv parts. Prefer a
    list for SDK-owned commands so paths/extras with spaces or brackets are
    quoted deterministically before reaching the terminal.
    """
    system = platform.system().lower()
    command_text = _command_to_shell_text(command)

    try:
        if system == "darwin":
            script_command = _escape_applescript_string(_hold_open_zsh(command_text))
            script = f'tell application "Terminal" to do script "{script_command}"'
            subprocess.Popen(["osascript", "-e", script])
            return True

        if system == "windows":
            subprocess.Popen(["cmd.exe", "/c", "start", "IINTS SDK Update", "cmd.exe", "/k", command_text])
            return True

        if system == "linux":
            shell_command = _hold_open_shell(command_text)
            terminals = [
                ("x-terminal-emulator", ["-e", "bash", "-lc", shell_command]),
                ("gnome-terminal", ["--", "bash", "-lc", shell_command]),
                ("konsole", ["-e", "bash", "-lc", shell_command]),
                ("xfce4-terminal", ["-x", "bash", "-lc", shell_command]),
                ("xterm", ["-e", "bash", "-lc", shell_command]),
                ("alacritty", ["-e", "bash", "-lc", shell_command]),
            ]

            for term, args in terminals:
                if shutil.which(term):
                    subprocess.Popen([term, *args])
                    return True

            return False

        return False

    except Exception as exc:
        print(f"Failed to open terminal: {exc}")
        return False
