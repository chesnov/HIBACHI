#!/usr/bin/env python3
"""
make_icons.py -- generate all platform app-icon formats from one master image.

Master source (first that exists):
    assets/icon.svg        (rasterized at 1024 with cairosvg, if installed)
    assets/icon_1024.png   (fallback if cairosvg / the SVG isn't available)

Outputs (paths are where the packaging files already expect them):
    assets/icon_1024.png                 master raster (also (re)written from SVG)
    launcher/assets/hibachi.png          Linux .desktop icon (512x512)
    packaging/windows/hibachi.ico        Windows installer + shortcut icon
    packaging/macos/hibachi.icns         macOS .app icon

Run it after editing icon.svg (or after dropping in your own icon_1024.png):
    python assets/make_icons.py

Only depends on Pillow. cairosvg is optional (needed only to rebuild from SVG).
The .icns is written with a small built-in packer, so this works on any OS --
no macOS `iconutil` required.
"""

from __future__ import annotations

import io
import os
import struct
import sys

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO_ROOT, "assets")
SVG = os.path.join(ASSETS, "icon.svg")
MASTER_PNG = os.path.join(ASSETS, "icon_1024.png")

OUT_LINUX = os.path.join(REPO_ROOT, "launcher", "assets", "hibachi.png")
OUT_WIN = os.path.join(REPO_ROOT, "packaging", "windows", "hibachi.ico")
OUT_MAC = os.path.join(REPO_ROOT, "packaging", "macos", "hibachi.icns")

# ICNS OSType codes -> pixel size (all PNG-encoded entries).
ICNS_TYPES = [
    (b"icp4", 16),
    (b"icp5", 32),
    (b"ic07", 128),
    (b"ic08", 256),
    (b"ic09", 512),
    (b"ic10", 1024),
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic13", 256),
    (b"ic14", 512),
]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def load_master() -> Image.Image:
    """Return a 1024x1024 RGBA master image."""
    if os.path.isfile(SVG):
        try:
            import cairosvg  # optional

            png_bytes = cairosvg.svg2png(url=SVG, output_width=1024, output_height=1024)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            img.save(MASTER_PNG)
            print(f"[icons] rasterized {os.path.relpath(SVG, REPO_ROOT)} -> {os.path.relpath(MASTER_PNG, REPO_ROOT)}")
            return img
        except Exception as exc:
            print(f"[icons] cairosvg unavailable ({exc}); falling back to {MASTER_PNG}")
    if os.path.isfile(MASTER_PNG):
        return Image.open(MASTER_PNG).convert("RGBA")
    sys.exit(f"[icons] ERROR: need {SVG} (with cairosvg) or {MASTER_PNG}")


def _resized(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def write_png(img: Image.Image) -> None:
    os.makedirs(os.path.dirname(OUT_LINUX), exist_ok=True)
    _resized(img, 512).save(OUT_LINUX)
    print(f"[icons] wrote {os.path.relpath(OUT_LINUX, REPO_ROOT)} (512x512)")


def write_ico(img: Image.Image) -> None:
    os.makedirs(os.path.dirname(OUT_WIN), exist_ok=True)
    # Pillow builds a multi-resolution .ico from the sizes list.
    img.save(OUT_WIN, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    print(f"[icons] wrote {os.path.relpath(OUT_WIN, REPO_ROOT)} ({', '.join(str(s) for s in ICO_SIZES)})")


def write_icns(img: Image.Image) -> None:
    os.makedirs(os.path.dirname(OUT_MAC), exist_ok=True)
    entries = []
    for ostype, size in ICNS_TYPES:
        buf = io.BytesIO()
        _resized(img, size).save(buf, format="PNG")
        data = buf.getvalue()
        entries.append(ostype + struct.pack(">I", len(data) + 8) + data)
    body = b"".join(entries)
    blob = b"icns" + struct.pack(">I", len(body) + 8) + body
    with open(OUT_MAC, "wb") as fh:
        fh.write(blob)
    print(f"[icons] wrote {os.path.relpath(OUT_MAC, REPO_ROOT)} ({len(ICNS_TYPES)} sizes)")


def main() -> None:
    master = load_master()
    if master.size != (1024, 1024):
        master = _resized(master, 1024)
    write_png(master)
    write_ico(master)
    write_icns(master)
    print("[icons] done.")


if __name__ == "__main__":
    main()
