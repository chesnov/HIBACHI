"""soma_source: seeds for cell separation taken from another channel.

Why
---
Soma extraction finds cores as local maxima of a distance transform on the
cytoplasmic mask. A cell whose nucleus is DARK in the cytoplasmic stain leaves
two thick lobes either side of it, each a local maximum, so one cell yields two
cores and step 4 dutifully splits it in half. No parameter fixes that: the
signal genuinely has two thick regions.

A nuclear stain gives exactly one marker per cell, so the split count comes out
right by construction rather than by tuning. It also fixes the converse case,
which is harder to tune away: two cells whose cytoplasm merges into a single
mask but which have two nuclei.

Nothing about step 4 changes. `separate_multi_soma_cells` already takes the
soma mask as an argument, and `cell_bodies` is read by that step alone --
feature calculation never touches it -- so the seeds are purely seeds and their
origin is contained here.

Locating the other channel
--------------------------
A results directory is named ``<sample>_processed_<mode>[_roi_<slug>]`` and the
sample folder has the same name in every channel, so the other channel's
equivalent is its copy of the SAME directory name:

    <project>/<other channel>/<sample>/<same processed dir name>

That is why an ROI needs no special handling. The full image and a named region
are different directory names, and asking for the same name in a sibling
channel can only ever find the matching one. No slug is parsed and no name is
pattern-matched, which is the failure mode this codebase keeps finding.

This module lives in `high_level_gui` and the pipeline reaches it through a
method on `ProcessingStrategy`, so `fluorescence_module` gains no new import
across the package boundary (convention 5 in its `__init__`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: Parameter naming the channel to take somas from. Empty or absent means
#: extract them from this channel's own intensity, as before.
SOMA_SOURCE_KEY = "soma_source_channel"

#: Provenance written next to the run's parameters. Added to
#: `_PASSTHROUGH_KEYS`, without which `save_config` drops it on the next save.
SOMA_SOURCE_RECORD = "soma_source"

#: Rows processed at a time when filtering seeds. A plane of these images is
#: 928 megapixels, so a whole-array boolean would be gigabytes; the filter is
#: two streaming passes instead.
_BLOCK_ROWS = 64


def sample_dir_of(processed_dir: str) -> str:
    """The sample folder holding a results directory."""
    return os.path.dirname(os.path.abspath(processed_dir))


def channel_dir_of(processed_dir: str) -> str:
    return os.path.dirname(sample_dir_of(processed_dir))


def project_root_of(processed_dir: str) -> str:
    return os.path.dirname(channel_dir_of(processed_dir))


def _artifact_in(directory: str) -> Optional[str]:
    """The cell-bodies artifact in a results directory, or None.

    By glob, like `ARTIFACT_PATTERNS`: a legacy project's file carries a mode
    string that no longer exists, and an exact name would miss it and report
    the channel as unprocessed.
    """
    import glob

    matches = sorted(glob.glob(os.path.join(directory, "cell_bodies*.dat")))
    return matches[0] if matches else None


def candidate_channels(processed_dir: str) -> List[Tuple[str, str]]:
    """[(channel_name, artifact_path)] for channels that can seed THIS segment.

    Only channels whose matching segment -- the same full image, or the same
    named region -- has actually reached soma extraction. A channel that has
    not been processed cannot seed one that is being processed now, and
    offering it would produce a run that fails at step 3.
    """
    here = os.path.basename(os.path.abspath(processed_dir.rstrip("/\\")))
    sample = os.path.basename(sample_dir_of(processed_dir))
    mine = os.path.basename(channel_dir_of(processed_dir))
    root = project_root_of(processed_dir)

    out: List[Tuple[str, str]] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for name in entries:
        if name == mine:
            continue
        other = os.path.join(root, name, sample, here)
        if not os.path.isdir(other):
            continue
        artifact = _artifact_in(other)
        if artifact is not None:
            out.append((name, artifact))
    return out


def resolve(processed_dir: str, channel_name: str) -> str:
    """The seed artifact for `channel_name`, or raise saying why not.

    Raises rather than returning None: a run configured to seed from another
    channel must not quietly fall back to extracting its own somas, because
    the result would look like a successful run of a different analysis.
    """
    sample = os.path.basename(sample_dir_of(processed_dir))
    here = os.path.basename(os.path.abspath(processed_dir.rstrip("/\\")))
    root = project_root_of(processed_dir)
    other = os.path.join(root, str(channel_name), sample, here)

    if not os.path.isdir(os.path.join(root, str(channel_name))):
        raise FileNotFoundError(
            f"this run takes somas from {channel_name!r}, but no such channel "
            f"exists in {os.path.basename(root)}. If the channel folder was "
            "renamed, set the soma source again."
        )
    if not os.path.isdir(other):
        raise FileNotFoundError(
            f"{channel_name!r} has no results for {sample!r} "
            f"({here}). Process that channel first."
        )
    artifact = _artifact_in(other)
    if artifact is None:
        raise FileNotFoundError(
            f"{channel_name!r} has results for {sample!r} but has not reached "
            "soma extraction. Run its first three steps, then this one."
        )
    return artifact


def configured_source(sample_dir: str) -> Optional[str]:
    """The channel this sample's config takes somas from, or None.

    Reads the parameter straight from the sample's own config, because that is
    where the decision lives and batch has no strategy instance to ask.
    """
    import glob

    import yaml  # type: ignore

    for path in sorted(glob.glob(os.path.join(sample_dir, "*.y*ml"))):
        try:
            with open(path) as handle:
                config = yaml.safe_load(handle) or {}
            block = ((config.get("execute_soma_extraction") or {})
                     .get("parameters") or {})
            value = (block.get(SOMA_SOURCE_KEY) or {}).get("value")
            text = str(value).strip() if value is not None else ""
            if text:
                return text
        except Exception:
            continue
    return None


def order_for_seeding(folders) -> List[str]:
    """Reorder folders so a channel's soma source is processed before it.

    Batch treats channel projects as independent and runs them in whatever
    order they were checked. A channel seeded from another one cannot: its step
    3 needs the source's cell bodies to already exist, so the wrong order turns
    a valid selection into a run that fails halfway with half the work done.

    Only reorders WITHIN the given set. A source that was not selected is left
    alone -- it may have been processed in an earlier run, and if it has not,
    step 3 says so clearly rather than this silently adding work nobody asked
    for. A dependency cycle (two channels seeding each other) keeps its
    original order rather than hanging; the run will fail with a readable
    reason, which beats a scheduler that never terminates.
    """
    from .project_selection import split_leaf_key

    keys = list(folders)
    identity = {}
    for key in keys:
        folder, _roi = split_leaf_key(str(key))
        sample = os.path.basename(os.path.normpath(folder))
        channel = os.path.basename(channel_dir_of(os.path.join(folder, "x")))
        identity[key] = (sample, channel, folder)

    # (sample, channel) -> key, so a dependency can be looked up by name.
    by_identity = {(sample, channel): key
                   for key, (sample, channel, _f) in identity.items()}

    depends = {}
    for key, (sample, _channel, folder) in identity.items():
        source = configured_source(folder)
        depends[key] = by_identity.get((sample, source)) if source else None

    ordered, placed = [], set()
    remaining = list(keys)
    while remaining:
        progressed = False
        for key in list(remaining):
            need = depends.get(key)
            if need is None or need in placed or need not in remaining:
                ordered.append(key)
                placed.add(key)
                remaining.remove(key)
                progressed = True
        if not progressed:
            # Cyclic: emit the rest in their original order.
            ordered.extend(remaining)
            break
    return ordered


def _row_blocks(rows: int):
    for start in range(0, rows, _BLOCK_ROWS):
        yield slice(start, min(start + _BLOCK_ROWS, rows))


def filter_to_mask(seed_path: str, mask_path: str, out_path: str,
                   shape: Tuple[int, ...],
                   dtype=np.int32) -> Dict[str, Any]:
    """Write the seeds that fall inside `mask_path`, relabelled from 1.

    A nuclear stain shows every nucleus in the tissue, not only those of the
    labelled population. A bystander nucleus lying inside one cell's cytoplasm
    would split that cell in two, which is the very failure this feature
    exists to remove. So a seed component is kept only if it overlaps the
    segmentation it will be used to split, and the number dropped is reported
    rather than absorbed -- if most of them go, the channels are misaligned or
    the wrong channel was chosen, and that should be visible.

    Two streaming passes over row blocks: one to decide which labels are kept,
    one to write them. A plane here is 928 megapixels, so an intermediate
    boolean of the whole array would be gigabytes.
    """
    seeds = np.memmap(seed_path, dtype=dtype, mode="r", shape=tuple(shape))
    mask = np.memmap(mask_path, dtype=np.int32, mode="r", shape=tuple(shape))
    rows = int(shape[0])

    present: set = set()
    keep: set = set()
    try:
        for block in _row_blocks(rows):
            chunk = np.asarray(seeds[block])
            nonzero = chunk > 0
            if not nonzero.any():
                continue
            present.update(int(v) for v in np.unique(chunk[nonzero]))
            inside = nonzero & (np.asarray(mask[block]) > 0)
            if inside.any():
                keep.update(int(v) for v in np.unique(chunk[inside]))
    finally:
        del mask

    kept = sorted(keep)
    highest = max(present) if present else 0
    lookup = np.zeros(highest + 1, dtype=dtype)
    for new_label, old_label in enumerate(kept, start=1):
        lookup[old_label] = new_label

    out = np.memmap(out_path, dtype=dtype, mode="w+", shape=tuple(shape))
    try:
        for block in _row_blocks(rows):
            # A copy, not a view: `seeds` is opened read-only, and the clip
            # below writes in place. The clip guards the lookup against a
            # label beyond the range this file's own first pass saw, which a
            # truncated or corrupt .dat could produce.
            chunk = np.array(seeds[block])
            np.clip(chunk, 0, highest, out=chunk)
            out[block] = lookup[chunk]
        out.flush()
    finally:
        del out
        del seeds

    return {
        "seeds_found": len(present),
        "seeds_kept": len(kept),
        "seeds_dropped": len(present) - len(kept),
    }
