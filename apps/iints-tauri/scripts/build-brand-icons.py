from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
SOURCE_LOGO = REPO_ROOT / "img" / "iints_logo.png"
ICONS_DIR = APP_ROOT / "src-tauri" / "icons"
FRONTEND_DIR = APP_ROOT / "frontend"


def _contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def _trim_transparency(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("The IINTS logo does not contain visible pixels.")
    return image.crop(bbox)


def _write_brand_assets() -> Path:
    source = Image.open(SOURCE_LOGO).convert("RGBA")
    full_logo = _trim_transparency(source)

    # Native app icons must be square. Keep the complete official logo readable
    # while macOS, Windows, and Linux apply their own platform masks.
    icon = Image.new("RGBA", (1024, 1024), (242, 243, 241, 255))
    fitted_logo = _contain(full_logo, (900, 820))
    icon.alpha_composite(
        fitted_logo,
        ((icon.width - fitted_logo.width) // 2, (icon.height - fitted_logo.height) // 2),
    )
    icon_source = ICONS_DIR / "icon-source.png"
    icon.save(icon_source, optimize=True)

    # The flower is the compact in-app mark. It is cropped from the exact same
    # source artwork rather than being redrawn or generated.
    flower = _trim_transparency(source.crop((0, 0, 500, 520)))
    mark = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    fitted_flower = _contain(flower, (232, 232))
    mark.alpha_composite(
        fitted_flower,
        ((mark.width - fitted_flower.width) // 2, (mark.height - fitted_flower.height) // 2),
    )
    mark.save(FRONTEND_DIR / "app-mark.png", optimize=True)

    frontend_logo = _contain(full_logo, (900, 900))
    frontend_logo.save(FRONTEND_DIR / "iints-logo.png", optimize=True)
    return icon_source


def _tauri_executable() -> Path:
    executable = APP_ROOT / "node_modules" / ".bin" / "tauri"
    if executable.is_file():
        return executable
    windows_executable = executable.with_suffix(".cmd")
    if windows_executable.is_file():
        return windows_executable
    raise FileNotFoundError("Tauri CLI not found. Run npm install in apps/iints-tauri first.")


def _remove_mobile_assets() -> None:
    for directory in (ICONS_DIR / "android", ICONS_DIR / "ios"):
        if directory.exists():
            shutil.rmtree(directory)
    for pattern in ("Square*Logo.png", "StoreLogo.png", "64x64.png"):
        for path in ICONS_DIR.glob(pattern):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build desktop icons from the official IINTS-SDK logo.")
    parser.add_argument("--skip-tauri", action="store_true", help="Only create source/frontend assets.")
    args = parser.parse_args()

    icon_source = _write_brand_assets()
    if not args.skip_tauri:
        subprocess.run(
            [str(_tauri_executable()), "icon", str(icon_source), "--output", str(ICONS_DIR)],
            cwd=APP_ROOT,
            check=True,
        )
        _remove_mobile_assets()


if __name__ == "__main__":
    main()
