"""
spatial_null -- mask-preserving Monte-Carlo spatial null for HIBACHI.

Tests whether segmented objects are arranged non-randomly inside a real domain
by re-placing the ACTUAL segmented masks under random rigid motions, rather than
approximating them as points or spheres. After Andrey et al. 2010 (PLoS Comput
Biol 6:e1000853), adapted for extended objects and anisotropic voxels.

LAYOUT
    engine.py    domain construction, rigid motion, placement, distance
                 functions, the Monte-Carlo driver. No I/O beyond reading the
                 persisted hull shell.
    export.py    the on-disk schema and its comparability key.
    runner.py    HIBACHI-facing: resolves samples from the registry, handles
                 ROI cropping, runs, writes the export.
    dialog.py    the analyzer's Qt entry point, with a threaded progress dialog.
    qc_render.py JPG verification images: one per randomisation, showing the
                 stationary partner, the randomised objects and the exact
                 segment each measured distance came from.
    notebook/    hibachi_null_io.py -- standalone cross-project loader and the
                 statistics. Deliberately NOT imported from this package: it
                 must stay free of HIBACHI dependencies so it runs next to a
                 notebook on any machine. Copy it, don't import it from here.

WHY THERE IS NO INFERENCE IN THIS PACKAGE
    A HIBACHI project is normally ONE biological replicate whose images are
    technical replicates, so no test computed inside a project could be a
    biological result -- pooling images would inflate n by the images-per-project
    factor and turn a batch effect into a p-value. This package therefore emits
    raw per-object distances for every draw, plus the diagnostics needed to
    judge whether an image is interpretable, and leaves inference to the
    notebook layer where several projects can be pooled.

USAGE
    from ..spatial_null import SpatialNullDialog
    SpatialNullDialog(project_manager, checked_channels=..., recipe=...).exec_()

    # or headless
    from ..spatial_null import RunParameters, jobs_from_registry, run_project
    jobs = jobs_from_registry(pm.sample_registry, "Channel_0_Aggregates",
                              "Channel_1_Microglia")
    run_project(jobs, RunParameters(n_reference=199, n_test=199),
                out_dir=".../SPATIAL_NULL")
"""

from .engine import (
    Domain,
    NullResult,
    ObjectTemplate,
    PlacementResult,
    SDI_INTERPRETATION,
    boundary_mask,
    build_domain,
    cross_distance_field,
    derive_f_grid,
    describe_within_project,
    extract_templates,
    f_function,
    f_grid_probe,
    g_function,
    monte_carlo_null,
    nearest_cross_distances,
    nearest_cross_pairs,
    per_object_boundary,
    place_templates,
    random_rotation,
    reconstruct_hull_from_shell,
    sup_norm_signed,
    transform_mask,
)
from .qc_render import (
    estimate_qc_output,
    qc_paths,
    render_draw,
    render_observed,
)
from .export import (
    COMPARABILITY_KEY,
    SCHEMA_VERSION,
    build_manifest,
    write_project_export,
)
from .runner import (
    RunParameters,
    SampleJob,
    find_final_segmentation,
    jobs_from_registry,
    roi_crop_spec,
    run_project,
)

__all__ = [
    # engine
    "Domain", "NullResult", "ObjectTemplate", "PlacementResult",
    "SDI_INTERPRETATION", "boundary_mask", "build_domain",
    "cross_distance_field", "derive_f_grid", "describe_within_project",
    "extract_templates", "f_function", "f_grid_probe", "g_function",
    "monte_carlo_null", "nearest_cross_distances", "nearest_cross_pairs",
    "per_object_boundary",
    "place_templates", "random_rotation", "reconstruct_hull_from_shell",
    "sup_norm_signed", "transform_mask",
    # qc images
    "estimate_qc_output", "qc_paths", "render_draw", "render_observed",
    # export
    "COMPARABILITY_KEY", "SCHEMA_VERSION", "build_manifest",
    "write_project_export",
    # runner
    "RunParameters", "SampleJob", "find_final_segmentation",
    "jobs_from_registry", "roi_crop_spec", "run_project",
]


def _dialog(*args, **kwargs):
    """Lazy accessor so importing the package does not require PyQt5."""
    from .dialog import SpatialNullDialog
    return SpatialNullDialog(*args, **kwargs)


def __getattr__(name):
    # PyQt5 is only needed for the dialog, so it is imported on demand. Batch
    # and notebook use of the engine then works in a headless environment.
    if name == "SpatialNullDialog":
        from .dialog import SpatialNullDialog
        return SpatialNullDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")