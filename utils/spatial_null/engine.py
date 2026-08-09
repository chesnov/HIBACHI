"""
synthetic_null.py -- mask-preserving Monte-Carlo spatial null for HIBACHI.

Tests whether segmented objects are arranged non-randomly inside a real domain,
by re-placing the ACTUAL segmented masks under random rigid motions rather than
modelling them as points or spheres.

Method, after Andrey et al. 2010 (PLoS Comput Biol 6:e1000853), with the
substitutions we need for extended objects:

  * DOMAIN. The filled tissue hull, reconstructed from the persisted one-voxel
    boundary shell (`{mode}_edge_mask.dat`). Reconstruction is exact for hulls
    this pipeline produces: both hull generators end their per-slice pipeline
    with `binary_fill_holes`, so no enclosed void can exist, and the only
    failure mode of shell-filling is a cavity. An ROI applies the same crop the
    masks get -- bounding box AND per-slice polygon -- so a region behaves as a
    smaller copy of the full field. Falls back to the whole field when edge
    trimming was disabled (which writes an all-zero shell).

  * NULL. Each object keeps its own mask, exactly. Per draw it gets a
    Haar-uniform random ROTATION (not a reflection -- that would change
    chirality), applied in physical space so anisotropic spacing is honoured,
    resampled nearest-neighbour, then repaired to its exact original voxel
    count. It is dropped at a uniform-random position that lies wholly inside
    the domain, rejecting sites that collide with an already-placed object
    (hardcore). Because area and shape are preserved exactly, the object volume
    fraction is identical in every draw, so F(0) matches by construction and
    the comparison isolates ARRANGEMENT rather than confounding it with
    abundance -- something a point-based null cannot guarantee.

  * STATISTICS. Three families, all on HIBACHI's edge-to-edge geometry:
      - CROSS: per-object distance to the nearest object of a fixed partner
        channel. The partner never moves.
      - F (empty space): distance from every domain voxel to the nearest object
        surface. Uses all domain voxels, so there is no sampling noise and no
        N_E parameter -- the paper's objection to distance maps (loss of
        sub-voxel centroid precision) does not apply, because we are already
        working on rasterised masks at voxel resolution.
      - G: nearest-neighbour distance between objects, surface to surface.

  * INFERENCE. Two INDEPENDENT Monte-Carlo sets: the first estimates the
    reference curve, the second the spread around it. Reusing one set for both
    shrinks the spread (each draw helped define the mean it is compared to) and
    inflates significance. The per-sample index (SDI) is a mid-p Monte-Carlo
    rank, which is uniform on (0,1) under the null -- required for the
    population-level KS test to be valid.

  * DIAGNOSTICS, not decoration. Occupancy fraction, placement rejection rate
    and accepted-orientation rate are reported per sample. At high occupancy a
    hardcore null is FORCED toward regularity because there is nowhere else to
    put things, so an apparently significant result can be a packing artefact.
    A high rejection rate is the signature of that regime.

Distances reproduce `interaction_analysis*`'s edge-to-edge numbers: verified
equal to the existing contour/cKDTree path to float32 precision, for both
disjoint and overlapping objects, using one distance transform of the partner's
per-object boundary instead of N x M tree queries.
"""

from __future__ import annotations

import os
import glob
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
)
from scipy.spatial import cKDTree

__all__ = [
    "Domain", "ObjectTemplate", "PlacementResult", "NullResult",
    "build_domain", "reconstruct_hull_from_shell",
    "extract_templates", "random_rotation", "transform_mask",
    "place_templates", "boundary_mask", "per_object_boundary",
    "cross_distance_field", "nearest_cross_distances",
    "f_function", "g_function", "sup_norm_signed",
    "monte_carlo_null", "derive_f_grid", "f_grid_probe",
    "describe_within_project", "SDI_INTERPRETATION",
]


# =============================================================================
# Small helpers
# =============================================================================

def _conn_structure(ndim: int) -> np.ndarray:
    """Face connectivity, matching `_extract_contours_2d`'s erosion element."""
    return generate_binary_structure(ndim, 1)


def boundary_mask(binary: np.ndarray) -> np.ndarray:
    """Surface voxels of a binary mask (mask minus its erosion)."""
    return binary & ~ndimage.binary_erosion(
        binary, structure=_conn_structure(binary.ndim))


def per_object_boundary(labels: np.ndarray,
                        label_values: Optional[Sequence[int]] = None) -> np.ndarray:
    """Union of every object's own surface.

    Eroding each label separately -- rather than the union -- is what
    `_extract_contours_2d` does, and it matters: two objects in contact keep the
    surface along their shared face, which the union would erase.
    """
    if label_values is None:
        label_values = np.unique(labels[labels > 0])
    out = np.zeros(labels.shape, dtype=bool)
    if len(label_values) == 0:
        return out
    struct = _conn_structure(labels.ndim)
    slices = ndimage.find_objects(labels)
    for lbl in label_values:
        idx = int(lbl) - 1
        if idx < 0 or idx >= len(slices) or slices[idx] is None:
            continue
        sl = slices[idx]
        sub = labels[sl] == lbl
        out[sl] |= sub & ~ndimage.binary_erosion(sub, structure=struct)
    return out


def _radius_for_size(physical_size: float, ndim: int) -> float:
    """Radius of the ball/disc of a given physical volume/area."""
    physical_size = max(float(physical_size), 0.0)
    if physical_size <= 0:
        return 0.0
    if ndim == 3:
        return (3.0 * physical_size / (4.0 * np.pi)) ** (1.0 / 3.0)
    return (physical_size / np.pi) ** 0.5


# =============================================================================
# Domain
# =============================================================================

@dataclass
class Domain:
    """The region a synthetic object is allowed to occupy."""
    mask: np.ndarray                     # bool, shape == image shape
    spacing: Tuple[float, ...]
    source: str                          # 'hull', 'hull+roi', 'field', 'mask'
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    # When the domain is another channel's objects, containment can be required
    # per parent object rather than over their union. Union placement lets a
    # synthetic object straddle two parents, which no real child could do; but
    # per-object placement is much harder to satisfy and fails outright when a
    # child is larger than every parent. Exposed as a choice, not baked in.
    parent_labels: Optional[np.ndarray] = None

    @property
    def ndim(self) -> int:
        return self.mask.ndim

    @property
    def voxels(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def physical_size(self) -> float:
        return self.voxels * float(np.prod(self.spacing))


def reconstruct_hull_from_shell(shell: np.ndarray,
                                warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Filled hull from the persisted one-voxel boundary shell.

    `{mode}_edge_mask.dat` stores `hull ^ binary_erosion(hull)`, a hollow
    surface. Filling it recovers the solid hull exactly, because the hull
    generators fill holes per slice, so no enclosed cavity can exist and the
    shell is closed (scipy's erosion pads the array border with 0, so the rim
    is part of the shell wherever the hull meets an edge).

    The guard below is the reason this is safe to trust rather than assume: a
    cavity-free hull has exactly one enclosed background component per hull
    component, so an excess means an unexpected void is present.
    """
    diag: Dict[str, Any] = {}
    shell = np.asarray(shell, dtype=bool)
    if not shell.any():
        diag["empty_shell"] = True
        return np.zeros_like(shell), diag

    filled = binary_fill_holes(shell)
    if filled is None:                                    # pathological input
        return shell.copy(), {"fill_failed": True}

    # A cavity-free solid contributes exactly one shell component (its outer
    # skin) per connected hull component. A cavity adds a second, disjoint
    # lining inside the same filled component -- so an EXCESS of shell
    # components over filled components is the signature. Comparing shell
    # components against enclosed background regions would not work: a cavity
    # adds one of each, leaving the counts equal.
    struct = _conn_structure(shell.ndim)
    _, n_shell_components = ndimage.label(shell, structure=struct)
    _, n_filled_components = ndimage.label(filled, structure=struct)
    diag["shell_components"] = int(n_shell_components)
    diag["hull_components"] = int(n_filled_components)
    diag["hull_voxels"] = int(np.count_nonzero(filled))

    if n_shell_components > n_filled_components:
        diag["cavity_suspected"] = True
        diag["n_cavities"] = int(n_shell_components - n_filled_components)
        if warn:
            warnings.warn(
                f"Hull reconstruction found {n_shell_components} shell "
                f"component(s) for {n_filled_components} hull component(s), so "
                f"~{diag['n_cavities']} internal void(s) were filled. This "
                f"inflates the domain and biases F toward apparent clustering. "
                f"The pipeline's hull generators fill holes per slice, so this "
                f"should not happen -- inspect the reconstructed hull.",
                RuntimeWarning, stacklevel=2)
    return filled, diag


def _find_edge_mask(channel_sample_dir: str,
                    mode: Optional[str] = None) -> Optional[str]:
    """Locate `{mode}_edge_mask.dat` inside a full-image `*_processed_*` folder."""
    try:
        entries = sorted(os.listdir(channel_sample_dir))
    except OSError:
        return None
    for name in entries:
        full = os.path.join(channel_sample_dir, name)
        if "_processed_" not in name or not os.path.isdir(full):
            continue
        # Skip ROI sessions: their masks are a different shape than the field.
        base = os.path.basename(name.rstrip("/\\"))
        if base.endswith("_roi") or "_roi_" in base:
            continue
        if mode:
            exact = os.path.join(full, f"{mode}_edge_mask.dat")
            if os.path.exists(exact):
                return exact
        hits = sorted(glob.glob(os.path.join(full, "*edge_mask*.dat")))
        if hits:
            return hits[0]
    return None


def build_domain(shape: Tuple[int, ...],
                 spacing: Sequence[float],
                 channel_sample_dir: Optional[str] = None,
                 mode: Optional[str] = None,
                 use_hull: bool = True,
                 roi_crop: Optional[Dict[str, Any]] = None,
                 explicit_mask: Optional[np.ndarray] = None,
                 erode_um: float = 0.0) -> Domain:
    """Assemble the placement domain.

    Args:
        shape: shape the domain must have -- the CROP's shape under an ROI.
        spacing: per-axis microns.
        channel_sample_dir: sample folder of the channel whose hull to use.
        mode: pipeline mode string, for the exact edge-mask filename.
        use_hull: when False, or when no hull exists, the domain is the field.
        roi_crop: `{'bbox': {...}, 'slice_masks': {z: bool2d} or None}`. The
            hull is read at FULL image shape, then cropped by the bounding box
            and masked by the same per-slice polygon the masks received --
            bounding box alone would admit the corners between box and polygon,
            where the masks were forcibly zeroed and no real object can exist.
        explicit_mask: overrides everything (already in `shape`).
        erode_um: optional inward margin, e.g. to keep objects clear of the
            tissue edge. Reported, never silent.
    """
    spacing = tuple(float(s) for s in spacing)
    ndim = len(shape)
    if len(spacing) != ndim:
        raise ValueError(f"spacing {spacing} does not match shape {shape}")

    diag: Dict[str, Any] = {}

    if explicit_mask is not None:
        mask = np.asarray(explicit_mask, dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError(f"explicit_mask {mask.shape} != shape {tuple(shape)}")
        source = "mask"
    else:
        mask = None
        source = "field"
        if use_hull and channel_sample_dir:
            path = _find_edge_mask(channel_sample_dir, mode)
            if path:
                full_shape = tuple(shape)
                if roi_crop and roi_crop.get("full_shape"):
                    full_shape = tuple(roi_crop["full_shape"])
                try:
                    shell = np.array(np.memmap(path, dtype=bool, mode="r",
                                               shape=full_shape))
                except (ValueError, OSError) as exc:
                    diag["hull_read_error"] = str(exc)
                    shell = None
                if shell is not None:
                    hull, hdiag = reconstruct_hull_from_shell(shell)
                    diag.update(hdiag)
                    if hull.any():
                        if roi_crop:
                            hull = _apply_roi_crop(hull, roi_crop)
                            source = "hull+roi"
                        else:
                            source = "hull"
                        if hull.shape != tuple(shape):
                            diag["roi_shape_mismatch"] = (hull.shape, tuple(shape))
                        else:
                            mask = hull
                    else:
                        diag["hull_disabled"] = True   # all-zero shell
            else:
                diag["no_edge_mask"] = True

        if mask is None:
            mask = np.ones(tuple(shape), dtype=bool)

    if erode_um > 0:
        inner = distance_transform_edt(mask, sampling=spacing)
        mask = inner >= erode_um
        diag["eroded_um"] = float(erode_um)

    diag["domain_voxels"] = int(np.count_nonzero(mask))
    diag["domain_fraction_of_field"] = float(
        diag["domain_voxels"] / max(1, mask.size))
    return Domain(mask=mask, spacing=spacing, source=source, diagnostics=diag)


def _apply_roi_crop(full: np.ndarray, roi_crop: Dict[str, Any]) -> np.ndarray:
    """Crop by bounding box then apply the per-slice polygon mask.

    Mirrors `roi_sharing._build_crop_memmap`: a crop slice uses the nearest
    defined polygon, so one polygon applies to every slice and slices between
    two drawn levels take the nearer one.
    """
    bbox = roi_crop.get("bbox") or {}
    if full.ndim == 3:
        z0 = int(bbox.get("z0") or 0)
        z1 = int(bbox["z1"]) if bbox.get("z1") is not None else full.shape[0]
        y0, y1 = int(bbox.get("y0") or 0), int(bbox.get("y1") or full.shape[1])
        x0, x1 = int(bbox.get("x0") or 0), int(bbox.get("x1") or full.shape[2])
        out = full[z0:z1, y0:y1, x0:x1].copy()
    else:
        y0, y1 = int(bbox.get("y0") or 0), int(bbox.get("y1") or full.shape[0])
        x0, x1 = int(bbox.get("x0") or 0), int(bbox.get("x1") or full.shape[1])
        out = full[y0:y1, x0:x1].copy()

    slice_masks = roi_crop.get("slice_masks")
    if not slice_masks:
        return out

    if out.ndim == 2:
        m2 = next(iter(slice_masks.values()))
        out &= np.asarray(m2, dtype=bool)
        return out

    keys = sorted(int(k) for k in slice_masks)
    for local_z in range(out.shape[0]):
        global_z = local_z + (int(bbox.get("z0") or 0))
        nearest = min(keys, key=lambda k: abs(k - global_z))
        out[local_z] &= np.asarray(slice_masks[nearest], dtype=bool)
    return out


# =============================================================================
# Object templates
# =============================================================================

@dataclass
class ObjectTemplate:
    """One real segmented object, kept as its own mask."""
    label: int
    mask: np.ndarray                     # cropped boolean patch
    spacing: Tuple[float, ...]

    @property
    def voxels(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def physical_size(self) -> float:
        return self.voxels * float(np.prod(self.spacing))

    @property
    def surface_voxels(self) -> int:
        return int(np.count_nonzero(boundary_mask(self.mask)))


def extract_templates(labels: np.ndarray,
                      spacing: Sequence[float],
                      keep: Optional[Sequence[int]] = None) -> List[ObjectTemplate]:
    """Per-object mask patches from a label image."""
    spacing = tuple(float(s) for s in spacing)
    values = np.unique(labels[labels > 0]) if keep is None else np.asarray(keep)
    slices = ndimage.find_objects(labels)
    out: List[ObjectTemplate] = []
    for lbl in values:
        idx = int(lbl) - 1
        if idx < 0 or idx >= len(slices) or slices[idx] is None:
            continue
        sub = labels[slices[idx]] == lbl
        if not sub.any():
            continue
        out.append(ObjectTemplate(label=int(lbl), mask=sub.copy(), spacing=spacing))
    return out


# =============================================================================
# Rigid motion: Haar-uniform rotation in physical space, exact size
# =============================================================================

def random_rotation(ndim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrix (proper: det = +1, no reflection)."""
    if ndim == 2:
        t = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])
    if ndim == 3:
        # Random unit quaternion -> uniform on SO(3).
        u1, u2, u3 = rng.uniform(size=3)
        q = np.array([
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ])
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    raise ValueError(f"unsupported ndim {ndim}")


def _repair_size(mask: np.ndarray, target: int,
                 spacing: Sequence[float]) -> np.ndarray:
    """Force `mask` to exactly `target` voxels, changing only its surface.

    Rotating a rasterised mask perturbs its digitised size by O(surface), so
    size and shape cannot both be preserved exactly on a lattice. Size is
    preserved exactly; the residual shape change is measured and reported by
    the caller rather than hidden.
    """
    count = int(np.count_nonzero(mask))
    if count == target or count == 0:
        return mask

    if count < target:
        # Grow outward, nearest surface first.
        outside = distance_transform_edt(~mask, sampling=spacing)
        cand = np.flatnonzero((outside > 0).reshape(-1))
        if cand.size == 0:
            return mask
        order = np.argsort(outside.reshape(-1)[cand], kind="stable")
        take = cand[order[:target - count]]
        flat = mask.reshape(-1).copy()
        flat[take] = True
        return flat.reshape(mask.shape)

    # Shrink inward, shallowest surface first, so the core survives.
    inside = distance_transform_edt(mask, sampling=spacing)
    idx = np.flatnonzero(mask.reshape(-1))
    order = np.argsort(inside.reshape(-1)[idx], kind="stable")
    drop = idx[order[:count - target]]
    flat = mask.reshape(-1).copy()
    flat[drop] = False
    return flat.reshape(mask.shape)


def transform_mask(mask: np.ndarray,
                   rotation: np.ndarray,
                   spacing: Sequence[float],
                   preserve_size: bool = True) -> np.ndarray:
    """Rotate a mask about its centroid in PHYSICAL space, nearest-neighbour.

    The rotation is defined on microns, not indices, so an anisotropic grid does
    not shear the object. In index space the map is S^-1 R^T S with
    S = diag(spacing), which is what `affine_transform` needs.
    """
    spacing = np.asarray(spacing, dtype=float)
    ndim = mask.ndim
    S = np.diag(spacing)
    S_inv = np.diag(1.0 / spacing)

    # Output grid: bounding box of the rotated input box, in voxels.
    corners = np.array(np.meshgrid(*[[0, s - 1] for s in mask.shape],
                                   indexing="ij")).reshape(ndim, -1).T
    phys = corners * spacing
    centre_in = (np.asarray(mask.shape, dtype=float) - 1.0) / 2.0 * spacing
    rot = (rotation @ (phys - centre_in).T).T
    extent = rot.max(axis=0) - rot.min(axis=0)
    out_shape = tuple(int(np.ceil(e / sp)) + 2 for e, sp in zip(extent, spacing))
    centre_out = (np.asarray(out_shape, dtype=float) - 1.0) / 2.0 * spacing

    matrix = S_inv @ rotation.T @ S
    offset = S_inv @ (centre_in - rotation.T @ centre_out)

    rotated = ndimage.affine_transform(
        mask.astype(np.uint8), matrix=matrix, offset=offset,
        output_shape=out_shape, order=0, mode="constant", cval=0
    ).astype(bool)

    if preserve_size:
        rotated = _repair_size(rotated, int(np.count_nonzero(mask)), spacing)

    coords = np.argwhere(rotated)
    if coords.size == 0:
        return mask.copy()
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    return rotated[tuple(slice(lo[d], hi[d]) for d in range(ndim))].copy()


# =============================================================================
# Placement
# =============================================================================

@dataclass
class PlacementResult:
    labels: np.ndarray
    placed: int
    failed: int
    orientation_attempts: int
    orientation_accepts: int
    site_attempts: int
    # Placed label -> the template it came from. Without this the export cannot
    # be re-subset downstream (e.g. "large objects only"), because a draw's
    # label numbering is arbitrary and unrelated to the observed labelling.
    label_to_template: Dict[int, int] = field(default_factory=dict)
    centroids: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    per_object: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return (self.orientation_accepts / self.orientation_attempts
                if self.orientation_attempts else float("nan"))


def place_templates(templates: Sequence[ObjectTemplate],
                    domain: Domain,
                    rng: np.random.Generator,
                    rotate: bool = True,
                    hardcore: bool = True,
                    min_separation_um: float = 0.0,
                    max_attempts: int = 2000,
                    largest_first: bool = True,
                    per_parent_containment: bool = False,
                    record_per_object: bool = False) -> PlacementResult:
    """Place every template at a uniform-random valid site inside the domain.

    A site is valid when the whole footprint lies inside the domain (no
    clipping, since that would change the object's size) and, with `hardcore`,
    does not touch an already-placed object.

    Orientation and position are proposed JOINTLY and rejected together. Near a
    tissue boundary this biases the orientation distribution, because elongated
    objects fit in some orientations and not others. That bias is accepted by
    design and measured: `orientation_accepts / orientation_attempts` is
    reported per sample, and per-object rates are available.

    Objects are placed largest-first, which raises the chance that a crowded
    field can be packed at all. It does not bias the null: every object still
    lands uniformly over the sites available to it.
    """
    shape = domain.mask.shape
    ndim = domain.ndim
    spacing = domain.spacing
    out = np.zeros(shape, dtype=np.int32)
    occupied = np.zeros(shape, dtype=bool) if hardcore else None

    order = list(range(len(templates)))
    if largest_first:
        order.sort(key=lambda i: -templates[i].voxels)

    placed = failed = 0
    o_attempts = o_accepts = s_attempts = 0
    per_object: List[Dict[str, Any]] = []
    label_to_template: Dict[int, int] = {}
    centroids: Dict[int, Tuple[float, ...]] = {}

    for new_label, ti in enumerate(order, start=1):
        tpl = templates[ti]
        obj_attempts = 0
        success = False

        for _ in range(max_attempts):
            obj_attempts += 1
            o_attempts += 1

            patch = (transform_mask(tpl.mask, random_rotation(ndim, rng), spacing)
                     if rotate else tpl.mask)

            room = [shape[d] - patch.shape[d] for d in range(ndim)]
            if any(r < 0 for r in room):
                continue                       # cannot fit in this orientation

            origin = tuple(int(rng.integers(0, room[d] + 1)) for d in range(ndim))
            sl = tuple(slice(origin[d], origin[d] + patch.shape[d])
                       for d in range(ndim))
            s_attempts += 1

            # Containment: every voxel of the object inside the domain.
            if not np.all(domain.mask[sl][patch]):
                continue
            # Optionally inside ONE parent object, not merely inside the union.
            if per_parent_containment and domain.parent_labels is not None:
                ids = np.unique(domain.parent_labels[sl][patch])
                if ids.size != 1 or ids[0] <= 0:
                    continue
            o_accepts += 1

            if hardcore:
                test = patch
                if min_separation_um > 0:
                    test = _expand_physical(patch, spacing, min_separation_um)
                    pad = [(t - p) // 2 for t, p in zip(test.shape, patch.shape)]
                    o2 = [origin[d] - pad[d] for d in range(ndim)]
                    if any(o2[d] < 0 or o2[d] + test.shape[d] > shape[d]
                           for d in range(ndim)):
                        continue
                    sl2 = tuple(slice(o2[d], o2[d] + test.shape[d])
                                for d in range(ndim))
                else:
                    sl2 = sl
                if np.any(occupied[sl2] & test):
                    continue
                occupied[sl] |= patch

            region = out[sl]
            region[patch] = new_label
            placed += 1
            success = True

            label_to_template[new_label] = tpl.label
            local = np.argwhere(patch).mean(axis=0)
            centroids[new_label] = tuple(
                float((origin[d] + local[d]) * spacing[d]) for d in range(ndim))

            if record_per_object:
                per_object.append({
                    "label": tpl.label, "attempts": obj_attempts,
                    "voxels": tpl.voxels,
                    "realised_voxels": int(np.count_nonzero(patch)),
                    "surface_template": tpl.surface_voxels,
                    "surface_realised": int(np.count_nonzero(boundary_mask(patch))),
                })
            break

        if not success:
            failed += 1
            if record_per_object:
                per_object.append({"label": tpl.label, "attempts": obj_attempts,
                                   "voxels": tpl.voxels, "realised_voxels": 0,
                                   "placed": False})

    return PlacementResult(labels=out, placed=placed, failed=failed,
                           orientation_attempts=o_attempts,
                           orientation_accepts=o_accepts,
                           site_attempts=s_attempts,
                           label_to_template=label_to_template,
                           centroids=centroids, per_object=per_object)


def _expand_physical(mask: np.ndarray, spacing: Sequence[float],
                     margin_um: float) -> np.ndarray:
    """Mask grown outward by a physical margin (anisotropy-aware)."""
    pad = [int(np.ceil(margin_um / float(s))) + 1 for s in spacing]
    padded = np.pad(mask, [(p, p) for p in pad], mode="constant",
                    constant_values=False)
    return distance_transform_edt(~padded, sampling=spacing) <= margin_um


# =============================================================================
# Distances -- edge to edge, matching interaction_analysis*
# =============================================================================

def cross_distance_field(partner_labels: np.ndarray,
                         spacing: Sequence[float]) -> np.ndarray:
    """Microns from every voxel to the nearest partner SURFACE voxel.

    Computed once, because the partner never moves across draws. Taking the
    minimum of this field over an object's own surface reproduces the existing
    contour/cKDTree distance exactly -- boundary-to-boundary semantics, so an
    object sitting inside a partner reports its depth rather than collapsing to
    zero the way a solid-union transform would.
    """
    surf = per_object_boundary(partner_labels)
    if not surf.any():
        return np.full(partner_labels.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(~surf, sampling=spacing).astype(np.float32)


def nearest_cross_distances(labels: np.ndarray,
                            field: np.ndarray) -> np.ndarray:
    """Per object, the distance to the nearest partner surface (microns)."""
    values = np.unique(labels[labels > 0])
    if values.size == 0:
        return np.empty(0, dtype=float)
    struct = _conn_structure(labels.ndim)
    slices = ndimage.find_objects(labels)
    out = np.full(values.size, np.nan, dtype=float)
    for k, lbl in enumerate(values):
        idx = int(lbl) - 1
        if idx < 0 or idx >= len(slices) or slices[idx] is None:
            continue
        sl = slices[idx]
        sub = labels[sl] == lbl
        surf = sub & ~ndimage.binary_erosion(sub, structure=struct)
        if not surf.any():
            surf = sub
        vals = field[sl][surf]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[k] = float(vals.min())
    return out


def f_function(labels: np.ndarray, domain: Domain) -> np.ndarray:
    """Empty-space distances: every domain voxel to the nearest object surface.

    Returns the raw per-voxel distances; the CDF is formed later on a shared
    grid. Voxels inside an object contribute 0, so F(0) equals the object
    volume fraction -- identical between observed and null here, because the
    masks are preserved exactly. The comparison therefore isolates arrangement.
    """
    surf = per_object_boundary(labels)
    if not surf.any():
        return np.empty(0, dtype=np.float32)
    dist = distance_transform_edt(~surf, sampling=domain.spacing).astype(np.float32)
    inside = labels > 0
    dist[inside] = 0.0
    return dist[domain.mask]


def g_function(labels: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Nearest-neighbour surface-to-surface distance per object (microns).

    Exact, but without one distance transform per object. A generalised Voronoi
    map gives each background voxel its nearest object; the true nearest
    neighbour of an object is always Voronoi-adjacent to it, because the
    midpoint of the shortest connecting segment lies where those two objects
    are the two closest. Candidate pairs come from that adjacency and are then
    measured exactly with a local tree on their surface voxels.
    """
    values = np.unique(labels[labels > 0])
    n = values.size
    if n < 2:
        return np.empty(0, dtype=float)

    spacing = tuple(float(s) for s in spacing)
    solid = labels > 0
    _, indices = distance_transform_edt(~solid, sampling=spacing,
                                        return_indices=True)
    nearest = labels[tuple(indices)]          # nearest object id per voxel

    # Voronoi adjacency: labels meeting across any axis-aligned step.
    pairs = set()
    for axis in range(labels.ndim):
        a = np.take(nearest, np.arange(0, nearest.shape[axis] - 1), axis=axis)
        b = np.take(nearest, np.arange(1, nearest.shape[axis]), axis=axis)
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            for u, v in zip(a[m].ravel(), b[m].ravel()):
                pairs.add((min(int(u), int(v)), max(int(u), int(v))))

    struct = _conn_structure(labels.ndim)
    slices = ndimage.find_objects(labels)
    surf_pts: Dict[int, np.ndarray] = {}

    def points(lbl: int) -> np.ndarray:
        if lbl not in surf_pts:
            sl = slices[lbl - 1]
            sub = labels[sl] == lbl
            b = sub & ~ndimage.binary_erosion(sub, structure=struct)
            if not b.any():
                b = sub
            coords = np.argwhere(b) + np.array([s.start for s in sl])
            surf_pts[lbl] = coords * np.asarray(spacing)
        return surf_pts[lbl]

    best = {int(v): np.inf for v in values}
    for u, v in pairs:
        try:
            tree = cKDTree(points(u))
            d = float(tree.query(points(v), k=1)[0].min())
        except (IndexError, ValueError):
            continue
        if d < best[u]:
            best[u] = d
        if d < best[v]:
            best[v] = d

    out = np.array([best[int(v)] for v in values], dtype=float)
    return out[np.isfinite(out)]


# =============================================================================
# Curves and the sup-norm statistic
# =============================================================================

def _empirical_cdf(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """CDF of `samples` evaluated on `grid` (P[X < g], matching the paper)."""
    if samples.size == 0:
        return np.zeros(grid.size, dtype=float)
    s = np.sort(np.asarray(samples, dtype=float))
    return np.searchsorted(s, grid, side="left") / float(s.size)


def sup_norm_signed(curve: np.ndarray, reference: np.ndarray) -> float:
    """Signed difference of maximum amplitude between two curves.

    The paper's statistic: the value of `curve - reference` at the argument
    where its absolute value peaks. Keeping the sign is what makes clustered
    and regular patterns land in opposite tails.
    """
    diff = np.asarray(curve, dtype=float) - np.asarray(reference, dtype=float)
    if diff.size == 0:
        return float("nan")
    return float(diff[int(np.argmax(np.abs(diff)))])


def _mid_p_rank(observed: float, null: np.ndarray) -> float:
    """Mid-p Monte-Carlo rank of `observed` in the upper tail of `null`.

    Mid-p rather than the plain `(1 + #ge) / (N + 1)` because the index is
    afterwards KS-tested against Uniform(0,1); for a discrete statistic the
    mid-p version is much closer to uniform under the null, so the population
    test stays calibrated.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    greater = float(np.sum(null > observed))
    equal = float(np.sum(null == observed))
    return (greater + 0.5 * equal + 0.5) / (null.size + 1.0)


# =============================================================================
# Shared F grid
# =============================================================================

def _round_outward_125(value: float) -> float:
    """Round up to the next 1/2/5 step at the value's own magnitude.

    Two projects imaging the same tissue under the same settings then almost
    always derive an IDENTICAL grid, so their F curves pool with no resampling
    at all. That is the practical thing that makes a derived grid workable
    rather than merely defensible.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    base = 10.0 ** exp
    for step in (1.0, 2.0, 5.0, 10.0):
        if value <= step * base * (1.0 + 1e-12):
            return step * base
    return 10.0 * base


def f_grid_probe(labels: np.ndarray, domain: "Domain",
                 percentile: float = 99.9) -> float:
    """One scalar per image: the upper tail of its empty-space distances.

    Only the scalar is retained. Holding every image's full F distribution to
    derive a grid would cost hundreds of megabytes on a modest 3D project, for
    information that collapses to one number.
    """
    vals = f_function(labels, domain)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, percentile)) if vals.size else 0.0


def derive_f_grid(probes: Sequence[float], grid_points: int = 512
                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The run's shared F grid, from per-image probes.

    Every image and every draw in the run is evaluated on this one grid, so the
    grid is a shared coordinate frame rather than part of the statistic --
    which is why deriving it from the data does not compromise calibration.
    """
    finite = [p for p in probes if np.isfinite(p) and p > 0]
    raw = max(finite) if finite else 1.0
    top = _round_outward_125(raw)
    grid = np.linspace(0.0, top, int(grid_points))
    return grid, {"f_grid_max_um": float(top),
                  "f_grid_raw_max_um": float(raw),
                  "f_grid_points": int(grid_points)}


# =============================================================================
# Monte-Carlo driver
# =============================================================================

@dataclass
class NullResult:
    """One sample's contribution, including the reasons to distrust it."""
    sample: str
    n_objects: int
    n_partner: int
    statistics: Dict[str, float] = field(default_factory=dict)
    sdi: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    observed_objects: Optional[pd.DataFrame] = None
    null_objects: Optional[pd.DataFrame] = None
    f_curves: Dict[str, np.ndarray] = field(default_factory=dict)
    first_draw_labels: Optional[np.ndarray] = None

    def metadata_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"sample": self.sample,
                               "n_objects": self.n_objects,
                               "n_partner": self.n_partner}
        row.update({f"stat_{k}": v for k, v in self.statistics.items()})
        row.update({f"sdi_{k}": v for k, v in self.sdi.items()})
        row.update({k: v for k, v in self.diagnostics.items()
                    if isinstance(v, (int, float, str, bool, type(None)))})
        return row


# One formula, opposite meanings. Stated explicitly because it is the single
# easiest thing to get backwards.
SDI_INTERPRETATION = {
    "F": "low = regular/evenly spread, high = clustered",
    "G": "low = clustered, high = regular/evenly spread",
    "cross": "low = closer to the partner than chance, high = farther",
}


def _object_centroids_um(labels: np.ndarray,
                         spacing: Sequence[float]) -> Dict[int, Tuple[float, ...]]:
    values = np.unique(labels[labels > 0])
    if values.size == 0:
        return {}
    cents = ndimage.center_of_mass(labels > 0, labels, values)
    if len(values) == 1:
        cents = [cents] if not isinstance(cents, list) else cents
    return {int(v): tuple(float(c * s) for c, s in zip(cent, spacing))
            for v, cent in zip(values, cents)}


def _object_sizes(labels: np.ndarray) -> Dict[int, int]:
    counts = np.bincount(labels.reshape(-1))
    return {int(i): int(counts[i]) for i in range(1, counts.size) if counts[i]}


def monte_carlo_null(labels: np.ndarray,
                     domain: Domain,
                     f_grid: Optional[np.ndarray] = None,
                     sample: str = "sample",
                     partner_labels: Optional[np.ndarray] = None,
                     n_reference: int = 199,
                     n_test: int = 199,
                     rng: Optional[np.random.Generator] = None,
                     rotate: bool = True,
                     hardcore: bool = True,
                     min_separation_um: float = 0.0,
                     compute_f: bool = True,
                     compute_g: bool = True,
                     cross_statistic: str = "median",
                     keep_first_draw: bool = True,
                     max_attempts: int = 2000,
                     record_centroids: bool = True,
                     per_parent_containment: bool = False,
                     progress: Optional[Any] = None) -> NullResult:
    """Per-sample Monte-Carlo null for one object set in one domain.

    Two INDEPENDENT sets of draws: `n_reference` estimates the reference curve,
    `n_test` estimates the spread of the sup-norm statistic around it. Sharing
    one set would shrink that spread -- every draw would have helped define the
    mean it is then compared against -- and overstate significance.

    `partner_labels` never moves, so its distance field is built once and
    reused, making cross-distances cost one distance transform for the whole
    sample rather than one per draw.

    `f_grid` must be the run's SHARED grid (see `derive_f_grid`). A grid derived
    per image would give every image a different x-axis, so the curves could not
    be pooled downstream -- which is the entire point of the export.

    Emits tidy per-object frames for observed and null. The null frame carries
    `template_label`, so a notebook can re-subset both sides to the same objects
    and recompute any statistic without re-running the Monte-Carlo.
    """
    rng = rng or np.random.default_rng()
    spacing = domain.spacing
    templates = extract_templates(labels, spacing)
    n_obj = len(templates)

    n_partner = (int(np.unique(partner_labels[partner_labels > 0]).size)
                 if partner_labels is not None else 0)
    result = NullResult(sample=sample, n_objects=n_obj, n_partner=n_partner)
    d = result.diagnostics
    d["domain_source"] = domain.source
    d["domain_voxels"] = domain.voxels
    d["domain_um"] = domain.physical_size
    d.update({f"domain_{k}": v for k, v in domain.diagnostics.items()
              if isinstance(v, (int, float, str, bool))})

    if n_obj == 0:
        d["skipped"] = "no objects"
        return result

    object_voxels = sum(t.voxels for t in templates)
    occupancy = object_voxels / max(1, domain.voxels)
    d["occupancy_fraction"] = float(occupancy)
    d["object_voxels"] = int(object_voxels)
    # Above roughly a third, a hardcore null is FORCED toward regularity --
    # there is nowhere else to put things -- so "significantly regular" may be
    # a packing artefact rather than biology.
    d["packing_warning"] = bool(occupancy > 0.30)
    d["per_parent_containment"] = bool(per_parent_containment)

    # ---- observed ----------------------------------------------------------
    cross_field = (cross_distance_field(partner_labels, spacing)
                   if partner_labels is not None else None)

    obs_values = np.unique(labels[labels > 0])
    obs_sizes = _object_sizes(labels)
    obs_cent = _object_centroids_um(labels, spacing) if record_centroids else {}

    obs_frame: Dict[str, Any] = {
        "template_label": obs_values.astype(np.int32),
        "voxels": np.array([obs_sizes.get(int(v), 0) for v in obs_values],
                           dtype=np.int32),
    }
    unit = float(np.prod(spacing))
    obs_frame["physical_size"] = obs_frame["voxels"] * unit

    obs: Dict[str, np.ndarray] = {}
    if cross_field is not None:
        obs["cross"] = nearest_cross_distances(labels, cross_field)
        obs_frame["cross_um"] = obs["cross"].astype(np.float32)
    if compute_g:
        obs["G"] = g_function(labels, spacing)
    if compute_f:
        obs["F"] = f_function(labels, domain)
    if record_centroids and obs_cent:
        for ax, name in enumerate(_axis_names(labels.ndim)):
            obs_frame[f"centroid_{name}_um"] = np.array(
                [obs_cent.get(int(v), (np.nan,) * labels.ndim)[ax]
                 for v in obs_values], dtype=np.float32)
    result.observed_objects = pd.DataFrame(obs_frame)

    # G is per object but its length can differ from the object count when an
    # object has no Voronoi neighbour; join on the labels it was computed for.
    if compute_g and obs["G"].size == obs_values.size:
        result.observed_objects["g_um"] = obs["G"].astype(np.float32)

    # ---- draws -------------------------------------------------------------
    total = n_reference + n_test
    iterator = progress(range(total)) if progress is not None else range(total)

    ref_F: List[np.ndarray] = []
    test_F: List[np.ndarray] = []
    ref_scalar: List[float] = []
    test_scalar: List[float] = []
    ref_G: List[np.ndarray] = []
    test_G: List[np.ndarray] = []
    null_chunks: List[pd.DataFrame] = []

    o_att = o_acc = failed_total = incomplete = 0
    agg = {"median": np.median, "mean": np.mean, "min": np.min}[cross_statistic]

    def _scalar(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return float(agg(v)) if v.size else float("nan")

    for i in iterator:
        pr = place_templates(templates, domain, rng, rotate=rotate,
                             hardcore=hardcore,
                             min_separation_um=min_separation_um,
                             max_attempts=max_attempts,
                             per_parent_containment=per_parent_containment)
        o_att += pr.orientation_attempts
        o_acc += pr.orientation_accepts
        failed_total += pr.failed
        if pr.failed:
            incomplete += 1
        if i == 0 and keep_first_draw:
            result.first_draw_labels = pr.labels

        vals = np.unique(pr.labels[pr.labels > 0])
        chunk: Dict[str, Any] = {
            "draw": np.full(vals.size, i, dtype=np.int16),
            "set": np.full(vals.size, 0 if i < n_reference else 1, dtype=np.int8),
            "template_label": np.array(
                [pr.label_to_template.get(int(v), 0) for v in vals], dtype=np.int32),
        }
        sizes = _object_sizes(pr.labels)
        chunk["voxels"] = np.array([sizes.get(int(v), 0) for v in vals],
                                   dtype=np.int32)

        if cross_field is not None:
            cd = nearest_cross_distances(pr.labels, cross_field)
            chunk["cross_um"] = cd.astype(np.float32)
            (ref_scalar if i < n_reference else test_scalar).append(_scalar(cd))
        if compute_g:
            gd = g_function(pr.labels, spacing)
            (ref_G if i < n_reference else test_G).append(gd)
            if gd.size == vals.size:
                chunk["g_um"] = gd.astype(np.float32)
        if compute_f:
            (ref_F if i < n_reference else test_F).append(
                f_function(pr.labels, domain))
        if record_centroids:
            for ax, name in enumerate(_axis_names(labels.ndim)):
                chunk[f"centroid_{name}_um"] = np.array(
                    [pr.centroids.get(int(v), (np.nan,) * labels.ndim)[ax]
                     for v in vals], dtype=np.float32)

        null_chunks.append(pd.DataFrame(chunk))

    result.null_objects = (pd.concat(null_chunks, ignore_index=True)
                           if null_chunks else pd.DataFrame())

    d["orientation_acceptance_rate"] = float(o_acc / o_att) if o_att else float("nan")
    d["mean_unplaced_per_draw"] = float(failed_total / max(1, total))
    d["draws_incomplete"] = int(incomplete)
    d["n_reference"] = int(n_reference)
    d["n_test"] = int(n_test)
    # Crowded samples are exactly the ones where the null matters. Flag them;
    # never drop them, because dropping biases which samples reach the export.
    d["placement_warning"] = bool(incomplete)

    # ---- indices -----------------------------------------------------------
    if cross_field is not None:
        obs_s = _scalar(obs["cross"])
        ref_arr = np.asarray(ref_scalar, dtype=float)
        result.statistics[f"cross_{cross_statistic}_observed"] = obs_s
        result.statistics[f"cross_{cross_statistic}_null"] = (
            float(np.nanmedian(ref_arr)) if ref_arr.size else float("nan"))
        result.statistics["cross_effect_um"] = (
            float(np.nanmedian(ref_arr)) - obs_s if ref_arr.size else float("nan"))
        sd = float(np.nanstd(ref_arr, ddof=1)) if ref_arr.size > 1 else np.nan
        # Standardised alongside the microns: the same shift against a tight
        # null is far stronger evidence than against a wide one, and pooling
        # raw microns across images weights them as if it were not.
        result.statistics["cross_effect_z"] = (
            (float(np.nanmean(ref_arr)) - obs_s) / sd
            if np.isfinite(sd) and sd > 0 else np.nan)
        result.statistics["cross_null_sd_um"] = sd
        result.sdi["cross"] = 1.0 - _mid_p_rank(
            obs_s, np.asarray(test_scalar, dtype=float))

    if compute_g and ref_G:
        grid_g = _quantile_grid(
            [obs["G"]] + ref_G + test_G, points=len(f_grid) if f_grid is not None else 512)
        ref_curves = np.array([_empirical_cdf(x, grid_g) for x in ref_G])
        reference = ref_curves.mean(axis=0)
        obs_delta = sup_norm_signed(_empirical_cdf(obs["G"], grid_g), reference)
        null_delta = np.array([sup_norm_signed(_empirical_cdf(x, grid_g), reference)
                               for x in test_G])
        result.statistics["G_delta"] = obs_delta
        result.sdi["G"] = _mid_p_rank(obs_delta, null_delta)

    if compute_f and ref_F:
        if f_grid is None:
            raise ValueError(
                "compute_f=True requires the run's shared f_grid; see "
                "derive_f_grid(). A per-image grid cannot be pooled.")
        ref_curves = np.array([_empirical_cdf(x, f_grid) for x in ref_F])
        reference = ref_curves.mean(axis=0)
        obs_curve = _empirical_cdf(obs["F"], f_grid)
        obs_delta = sup_norm_signed(obs_curve, reference)
        null_delta = np.array([sup_norm_signed(_empirical_cdf(x, f_grid), reference)
                               for x in test_F])
        result.statistics["F_delta"] = obs_delta
        result.sdi["F"] = _mid_p_rank(obs_delta, null_delta)
        result.f_curves = {
            "grid": f_grid.astype(np.float32),
            "observed": obs_curve.astype(np.float32),
            "reference": reference.astype(np.float32),
            "null": np.array(
                [_empirical_cdf(x, f_grid) for x in (ref_F + test_F)],
                dtype=np.float32),
            "set": np.array([0] * len(ref_F) + [1] * len(test_F), dtype=np.int8),
        }

    return result


def _axis_names(ndim: int) -> Tuple[str, ...]:
    return ("z", "y", "x") if ndim == 3 else ("y", "x")


def _quantile_grid(arrays: Sequence[np.ndarray], points: int = 512) -> np.ndarray:
    """Grid for G, spanning the pooled support of the supplied samples.

    G stays per-sample: unlike F it is not exported as a curve, so it has no
    cross-project pooling requirement, and a local grid keeps its resolution
    where the data actually are.
    """
    pooled = np.concatenate([np.asarray(a, dtype=float).ravel()
                             for a in arrays if np.size(a)]) \
        if any(np.size(a) for a in arrays) else np.array([0.0, 1.0])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return np.linspace(0.0, 1.0, points)
    top = float(np.percentile(pooled, 99.5))
    return np.linspace(0.0, max(top, 1e-6), points)


# =============================================================================
# Within-project description (NOT inference)
# =============================================================================

def describe_within_project(metadata: pd.DataFrame,
                            sdi_columns: Optional[Sequence[str]] = None
                            ) -> pd.DataFrame:
    """Descriptive summary of one project's per-image indices.

    A project is typically ONE biological replicate whose images are technical
    replicates, so nothing here is a biological claim and none of it should be
    reported as one. The KS statistic is a within-replicate description: it is
    useful mainly for spotting a project where nothing departs from the null at
    all, which usually means a domain or segmentation problem rather than
    biology. Inference belongs downstream, across projects.
    """
    from scipy.stats import kstest

    if sdi_columns is None:
        sdi_columns = [c for c in metadata.columns if c.startswith("sdi_")]

    rows: List[Dict[str, Any]] = []
    for col in sdi_columns:
        vals = pd.to_numeric(metadata[col], errors="coerce").dropna().values
        key = col[4:] if col.startswith("sdi_") else col
        row: Dict[str, Any] = {
            "index": key, "n_images": int(vals.size),
            "mean_sdi": float(np.mean(vals)) if vals.size else np.nan,
            "median_sdi": float(np.median(vals)) if vals.size else np.nan,
            "interpretation": SDI_INTERPRETATION.get(key, ""),
        }
        if vals.size >= 3:
            D, p = kstest(vals, "uniform")
            row["ks_D"] = float(D)
            row["ks_p_descriptive"] = float(p)
        else:
            row["ks_D"] = np.nan
            row["ks_p_descriptive"] = np.nan
        row["note"] = "within-replicate description only; not a biological test"
        rows.append(row)
    return pd.DataFrame(rows)
