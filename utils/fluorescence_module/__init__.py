"""
Fluorescence processing pipeline, one implementation per step for 2D and 3D.

Why this package exists
-----------------------
The pipeline used to live as two parallel trees, ``utils/module_3d`` and
``utils/module_2d``, meant to be kept at parity by hand. That did not work. A fix
would land in one and be forgotten in the other, and the drift was invisible
until a result looked wrong. Both trees were deleted once nothing imported them;
this package is the whole pipeline. Measured across the seven paired modules before the
merge: 61% of lines were shared, but of the divergence only about a fifth was
ever about dimensionality -- the rest was accumulated drift, including 15 paired
functions whose bodies had fallen below 60% similar and 13 functions that existed
in only one track.

So each step here takes its rank from the array it is handed, and there is one
copy of the logic. The only things that legitimately differ by rank are collected
in `dim_utils`.

Layout
------
``dim_utils.py``
    Every operation that genuinely differs between 2D and 3D: structuring
    elements, spacing conventions, physical-to-pixel conversion, tiling. No
    pipeline knowledge. Pinned by `test_dim_utils.py` against the original
    per-track implementations, so a change here cannot quietly alter behaviour.

``segmentation_helpers.py``, ``streaming_stats.py``, ``streaming_passes.py``
    Rank-agnostic infrastructure: printing, the SimpleITK watershed wrapper,
    bounded-memory label aggregates, and the post-stitch passes built on them.

Step modules, in pipeline order
    ``initial_segmentation.py``   step 1  (raw segmentation)
    ``remove_artifacts.py``       step 2  (edge trimming)
    ``soma_extraction.py``        step 3  (seed placement)
    ``cell_splitting.py``         step 4  (separation)
    ``calculate_features.py``     step 5  (dispatcher + genuinely shared code)
    ``features_2d.py`` / ``features_3d.py``
                                  step 5 implementations, see below

Conventions for anything added here
-----------------------------------
1.  **No ``_2d`` / ``_3d`` in the name of merged code.** The rank is data, not a
    module identity. A suffix is a signal that two copies exist again.

2.  **Rank comes from the array.** ``ndim = int(arr.ndim)``, near the top of the
    entry point, validated to be 2 or 3. Never from a flag, a mode string or
    which module the caller imported.

3.  **Rank-varying operations go through `dim_utils`**, not open-coded. A
    ``ball(r)`` or ``footprint_rectangle((3, 3, 3))`` or ``spacing[1:]`` written
    inline is the thing that has to be rewritten per rank, and therefore the
    thing that drifts.

4.  **Spacing is ordered like the axes**: ``(Y, X)`` in 2D, ``(Z, Y, X)`` in 3D,
    microns per voxel. "In-plane" means the LAST TWO entries at either rank --
    ``spacing[-2:]``, never ``spacing[1:]``, which silently drops Y in 2D.

5.  **Imports inside this package are relative and stay inside it.** Nothing
    reaches into a sibling package -- not ``high_level_gui``, and not the old
    ``module_3d`` / ``module_2d`` trees, which is what this rule was originally
    written against and which no longer exist to import from. Two of the merge
    session's failures were missing or cross-package imports; a self-contained
    package cannot have them. The constraint runs the other way too: the GUI
    imports this package, so this package importing the GUI would make the two
    impossible to reason about separately (``turntable._locate_layer_list_dock``
    is deliberately hand-rolled for exactly this reason).

6.  **Old entry-point names are kept as thin translating wrappers.** Existing
    callers, saved projects and batch workflows keep working, and the wrapper is
    the single place a renamed argument is mapped.

7.  **Processing parameters come from the config, not from code.** A default in
    a signature here is for direct or programmatic calls only; it is not a second
    place to configure the pipeline. In particular, no default in this package
    varies by rank -- if a value must differ between 2D and 3D, that difference
    belongs in the YAML, where it is visible and editable. Encoding it in a
    function signature hides a config decision somewhere nobody looks.

8.  **Every merged step is proven against `harness.py`.** It fingerprints all
    five steps on two real images and compares digests of the raw arrays. A
    merge is not done until ``--check`` reports IDENTICAL.

Genuinely rank-specific work
----------------------------
Feature calculation does not merge and should not be forced to: 2D has skeleton
cycle-breaking, spur pruning and topology statistics that have no 3D counterpart,
while 3D has volume, solidity and surface extraction that have no 2D one. Only 8
of its functions pair up at all. It therefore stays as two implementations,
``features_2d.py`` and ``features_3d.py``, behind ``calculate_features.py``,
which dispatches on ``ndim`` and holds whatever is genuinely shared. That is a
real difference in the science rather than an artefact of the old structure.


The same applies in part to ``remove_artifacts``: Z-erosion and hull-stack
handling are 3D-only, and live behind a rank check inside the merged module.
"""

__all__ = []
