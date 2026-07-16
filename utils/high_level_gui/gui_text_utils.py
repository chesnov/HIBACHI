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
