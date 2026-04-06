from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def export_uno_q_bridge(output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    template_root = files("iints.templates").joinpath("uno_q")
    sketch_path = target / "iints_supervisor_bridge.ino"
    readme_path = target / "README.md"
    protocol_path = target / "bridge_protocol.txt"

    sketch_path.write_text(template_root.joinpath("iints_supervisor_bridge.ino").read_text(encoding="utf-8"), encoding="utf-8")
    readme_path.write_text(template_root.joinpath("README.md").read_text(encoding="utf-8"), encoding="utf-8")
    protocol_path.write_text(
        "\n".join(
            [
                "IINTS UNO Q serial bridge protocol",
                "Messages:",
                "  OK",
                "  OVERRIDE",
                "  CRITICAL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(target),
        "sketch": str(sketch_path),
        "readme": str(readme_path),
        "protocol": str(protocol_path),
    }
