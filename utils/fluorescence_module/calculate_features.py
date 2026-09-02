"""
Step 5: feature calculation. Rank dispatcher.

Why this step is not merged
---------------------------
Every other step in this package is one implementation that reads its rank from
the data. Feature calculation is the exception, and deliberately so: the two
implementations measure genuinely different things, not the same thing in
different ranks.

Measured before the merge, only 8 of its functions paired up at all:

    2D only   skeleton cycle-breaking, spur pruning, spur tracing,
              topology statistics, 2D morphology
    3D only   volume, solidity by convex hull, shared surface extraction

A skeleton in 2D has loops that must be broken before it can be traced; the 3D
equivalent has no counterpart. Solidity in 3D is a convex-hull volume ratio; in
2D it is an area ratio, computed from different primitives. Forcing these
together would mean one function with two disjoint halves and a rank switch
through the middle -- the same two implementations, harder to read, and with a
shared signature implying an interchangeability that does not exist.

So the split is kept where the science splits, and hidden behind one entry point
so callers do not have to care.

What this module guarantees
---------------------------
*   `analyze_segmentation` works at either rank, taking `spacing` ordered like
    the array axes.
*   `analyze_segmentation_2d` still exists and still behaves exactly as it did,
    so existing callers and saved workflows keep working.
*   `export_to_fcs` is defined once here. The two tracks' copies were identical
    apart from a docstring and one error string, so it was never rank-specific.

Anything genuinely shared that turns up later belongs here, not duplicated into
`features_2d` and `features_3d`.
"""

import os
from typing import Any, Optional, Sequence

import numpy as np

try:
    import fcswrite  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fcswrite = None

try:
    from .segmentation_helpers import flush_print
except ImportError:  # pragma: no cover - direct script execution
    from segmentation_helpers import flush_print

try:
    from .dim_utils import normalise_spacing
except ImportError:  # pragma: no cover
    from dim_utils import normalise_spacing

__all__ = ["analyze_segmentation", "analyze_segmentation_2d", "export_to_fcs"]


# --------------------------------------------------------------------------
# Shared: FCS export
# --------------------------------------------------------------------------

def export_to_fcs(metrics_df, fcs_path):
    """
    Write a metrics table to FCS.

    Defined once because it never differed by rank: the 2D and 3D copies were
    identical apart from a docstring and the wording of one error message. This
    is the 3D body unchanged -- including the no-op guards, the inf/NaN scrubbing
    that FCS requires, and re-attaching `label` after `select_dtypes` in case it
    is not numeric.
    """
    if not fcs_path or fcswrite is None or metrics_df is None or metrics_df.empty:
        return
    try:
        flush_print(f"  [Export] Writing FCS: {os.path.basename(fcs_path)}")
        num_df = metrics_df.select_dtypes(include=[np.number]).copy()
        num_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_df.fillna(0, inplace=True)
        if 'label' in metrics_df.columns:
            num_df['label'] = metrics_df['label']
        fcswrite.write_fcs(filename=fcs_path, chn_names=list(num_df.columns),
                           data=num_df.values)
    except Exception as e:
        flush_print(f"  [Export] Error during FCS write: {e}")


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def _impl(ndim: int):
    """The implementation module for a given rank."""
    if ndim == 3:
        try:
            from . import features_3d as impl
        except ImportError:  # pragma: no cover
            import features_3d as impl
        return impl
    if ndim == 2:
        try:
            from . import features_2d as impl
        except ImportError:  # pragma: no cover
            import features_2d as impl
        return impl
    raise ValueError(
        f"feature calculation handles 2D and 3D data; got a {ndim}D array"
    )


def analyze_segmentation(
    segmented_array,
    intensity_image=None,
    spacing: Optional[Sequence[float]] = None,
    calculate_distances: bool = True,
    calculate_skeletons: bool = True,
    calculate_solidity: bool = False,
    skeleton_export_path: Optional[str] = None,
    fcs_export_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    return_detailed: bool = False,
    prune_spurs_le_um: float = 0.0,
    **kwargs: Any,
):
    """
    Measure per-cell features, at whichever rank the label array has.

    `spacing` is ordered like the array axes and given in microns per voxel:
    ``(Y, X)`` in 2D, ``(Z, Y, X)`` in 3D. A 3D spacing passed with a 2D array is
    reduced to its in-plane part, which is the conversion both strategies were
    doing inline.

    Every processing parameter is supplied by the caller, which in the app means
    it comes from the YAML config. The defaults here exist only for direct or
    programmatic calls; they are not a second place to configure the pipeline, and
    none of them varies by rank. If a value needs to differ between 2D and 3D
    that belongs in the config, not in this signature.
    """
    ndim = int(np.asarray(segmented_array).ndim
               if not hasattr(segmented_array, "ndim")
               else segmented_array.ndim)
    impl = _impl(ndim)
    sp = normalise_spacing(spacing, ndim)

    common = dict(
        intensity_image=intensity_image,
        calculate_distances=calculate_distances,
        calculate_skeletons=calculate_skeletons,
        calculate_solidity=calculate_solidity,
        skeleton_export_path=skeleton_export_path,
        fcs_export_path=fcs_export_path,
        temp_dir=temp_dir,
        n_jobs=n_jobs,
        return_detailed=return_detailed,
        prune_spurs_le_um=prune_spurs_le_um,
        **kwargs,
    )
    if ndim == 3:
        return impl.analyze_segmentation(segmented_array, spacing=sp, **common)
    return impl.analyze_segmentation_2d(segmented_array, spacing_yx=sp, **common)


def analyze_segmentation_2d(segmented_array, *args, **kwargs):
    """
    2D entry point, kept so existing callers keep working.

    Forwards untouched to the 2D implementation, including its `spacing_yx`
    argument name and its `calculate_solidity=True` default. New code should call
    `analyze_segmentation`.
    """
    return _impl(2).analyze_segmentation_2d(segmented_array, *args, **kwargs)
