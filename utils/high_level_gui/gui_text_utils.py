"""gui_text_utils: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import re
from typing import List, Optional, Union



def app_icon_path() -> Optional[str]:
    """
    Absolute path to the application icon (a PNG), or None if not found.

    This module lives at <repo>/utils/high_level_gui/, so the repo root is two
    directories up. Returns the Linux launcher PNG if present, else the master
    raster. Kept Qt-free so it can be imported anywhere without pulling in GUI
    dependencies.
    """
    here = os.path.dirname(os.path.abspath(__file__))          # utils/high_level_gui
    repo_root = os.path.dirname(os.path.dirname(here))          # repo root
    for rel in (("launcher", "assets", "hibachi.png"),
                ("assets", "icon_1024.png")):
        candidate = os.path.join(repo_root, *rel)
        if os.path.isfile(candidate):
            return candidate
    return None


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Sorts strings containing numbers naturally (e.g., Image_2 before Image_10).
    """
    basename = os.path.basename(s)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split('([0-9]+)', basename)
    ]

def clean_filename_for_matching(name: str) -> str:
    """
    Normalizes filenames for matching.
    1. Lowercase.
    2. Remove common extensions (czi, tif, etc.).
    3. Remove trailing ' #N' suffixes often added by Zen imports.
    """
    n = name.lower()
    # Remove extensions (iteratively to handle .czi.tif)
    for ext in ['.tif', '.tiff', '.czi', '.lsm', '.nd2', '.oib', '.lif']:
        n = n.replace(ext, '')
    # Remove scene suffixes like " #1", " #2"
    n = re.sub(r'\s+#\d+$', '', n)
    return n.strip()


# --------------------------------------------------------------------------- #
# Operating-system sidecar files
# --------------------------------------------------------------------------- #
_APPLEDOUBLE_PREFIX = "._"
_JUNK_BASENAMES = {
    ".ds_store", "thumbs.db", "desktop.ini", ".spotlight-v100",
    ".trashes", ".fseventsd", ".apdisk", "__macosx",
}


def is_os_sidecar(name: str) -> bool:
    """True if `name` is an operating-system sidecar file rather than real data.

    macOS writes an AppleDouble resource fork named ``._<original>`` beside every
    file on filesystems that can't store forks natively -- exFAT, NTFS, FAT32 and
    SMB shares, which covers essentially every external drive and network volume
    mounted under /Volumes. These sidecars carry the SAME extension as the file
    they shadow, so ``._scan.tif`` passes any ``endswith('.tif')`` check while
    being a ~4 KB AppleDouble blob that no TIFF reader can open.

    Worse, ``._name`` sorts BEFORE ``name`` (0x2E precedes alphanumerics), so a
    sidecar is the first entry in any sorted listing. Code that inspects only the
    first image to guess a folder's properties therefore inspects the sidecar,
    fails, and falls back to its "unknown" branch -- which is how a folder of 2D
    images was detected as 3D on a Mac but correctly as 2D on Linux.

    Filtering these out is safe on every platform: a leading ``._`` is not a
    naming convention any microscope export uses.
    """
    base = os.path.basename(str(name))
    if base.startswith(_APPLEDOUBLE_PREFIX):
        return True
    return base.lower() in _JUNK_BASENAMES


def real_files(names) -> List[str]:
    """`names` with operating-system sidecar files removed, order preserved."""
    return [n for n in names if not is_os_sidecar(n)]