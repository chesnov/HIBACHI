"""Elongated somas: seeds for spindle- and fibre-shaped cells (step 3, soma_shape="elongated").

Split out of soma_extraction.py, which calls `_elongated_label_candidates` for every
label when soma_shape is "elongated" and keeps the shared seed placement, the worker
pool and the compact mode. Nothing here depends on soma_extraction.

Pipeline for one object (see `_elongated_label_candidates` and `_paths_from_skeleton`):

1. The object's mask is skeletonised per tile (with a halo), and the skeleton graph is
   decomposed into fibre paths by `_paths_from_skeleton`:
     - hairs are removed and junction zones merged; lanes between zones are traced
       longest-first, each path choosing the straightest lane at every junction;
     - `_stitch_paths` and `_mend_breaks` join pieces of one fibre;
     - `_fix_sigmoids` undoes sideways switches between parallel fibres, swaps crossed
       paths, attaches orphans and hands over/bridges fibres through double "blobs";
     - `_leftover_chains` traces skeleton the paths did not cover;
     - `_rejoin_at_bends`, `_end_depth_reversals`, `_keep_order_at_crossings`,
       `_cut_climbs`, `_hand_on_at_ends`, `_join_lined_up_ends`, `_join_side_by_side`
       and `_march_free_ends` repair the remaining tracing errors, in that order.
2. Each path becomes a tube of the mask around it; the tubes are the seed candidates,
   which soma_extraction places as it does compact ones.

Lengths are in micrometres unless a name ends in _R (a multiple of the object's fibre
radius r, measured from the mask) or _DEG (degrees).
"""

import os
import math

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .soma_extraction import _CandidateParams

# =============================================================================
# Elongated somas (soma_shape="elongated")
# =============================================================================
#: The two soma shapes step 3 knows. "compact" is the only behaviour this step
#: had before the setting existed, and stays the default.
SOMA_SHAPES = ("compact", "elongated")

# Elongated mode reads the fibres from the MASK: its skeleton gives their
# course, its width their number. Every scale is a multiple of the object's
# fibre radius r, measured from the mask:
_ARM_WINDOW_R = 10.0   # an arm's direction is taken over this length of branch
_CORE_R = 2.0          # junctions joined by a branch shorter than one fibre
#                        width (2 r) lie inside one contact: one junction zone
_RIBBON_R = 1.5        # a branch whose in-plane half-width exceeds 1.5 r is a
#                        ribbon of several fibres fused side by side
_HAIR_K = 4.0          # an end branch no longer than 4 half-widths where it
#                        leaves is a hair -- a bump of the mask -- not a fibre
_LAYER_HOP_R = 3.5     # a lane that changes a path's depth by more than this many r leaves its fibre layer
_LAYER_SAME_R = 1.5    # one that changes it by less than this many r stays in it
_MAX_TURN_DEG = 30.0   # a fibre continues through a contact onto a branch
#                        that turns it by less than this
_STACK_K = 2.0         # a mask column more than this many single-fibre heights
#                        tall holds two fibres stacked in z
_STACK_LOW_K = 1.25    # ... and next to such columns, one at least this tall
#                        still holds part of both
_STACK_FOLLOW_UM = 5.0 # a stacked contact is followed this far beyond its
#                        clearly stacked columns
# Necks (`_neck_lines`): two fibres that touch side by side along a thin neck. Each fibre's core is where
# the mask is at least _NECK_CORE_R r deep; the neck between two cores is the watershed line of that depth,
# and it is a contact between fibres if it is at least _NECK_LEN_UM long in-plane. Such a neck is cut only
# where a seed path runs along it for at least _NECK_RUN_UM, and the cut is kept only if the mask within
# _NECK_CHECK_UM of it ends up at least as well seeded as without it.
_NECK_CORE_R = 0.85
_NECK_LEN_UM = 3.0
_NECK_RUN_UM = 2.0
_NECK_CHECK_UM = 6.0
# Break mending (`_stitch_paths`, second phase): two pieces of one fibre whose ends a contact has
# bent are judged on their COURSES -- each piece's path from _JOIN_SKIP_UM to _JOIN_WINDOW_UM back
# from its end, leaving out the bend itself --
_JOIN_SKIP_UM = 2.0
_JOIN_WINDOW_UM = 12.0
_JOIN_MIN_UM = 6.0     # a course needs at least this much path to have a direction
_JOIN_SHIFT_R = 4.0    # the two courses may be this far (in r) off one line, midway between the ends
_JOIN_MASK_GAP_R = 3.0 # and the straight link between the ends may leave the mask for this far (in r)
_JOIN_FAR_R = 20.0     # a second, longer reach (in r), for pieces the mask still joins: the same tests, but
_JOIN_FAR_TURN_DEG = 15.0  # the pieces must continue each other this straight
_JOIN_OVERLAP_R = 4.0  # pieces may overlap this far (in r) along the fibre, if they lie on one line (< 2 r apart)
_MEND_ROUTE_RATIO = 1.3  # a mend of a released leg may follow the mask on a route at most this much longer than the gap
# Leftover skeleton (`_leftover_chains`): skeleton that no path of at least _CRUMB_UM covers is
# kept as a candidate fibre piece if its longest chain is at least _SALVAGE_UM long
_CRUMB_UM = 3.0
_SALVAGE_UM = 8.0
# Sigmoid correction (`_fix_sigmoids`): a fibre's course is judged over this
# long a stretch on either side of a place (long-range direction), leaving out
# this much right at the place ...
_SIG_WINDOW_UM = 10.0
_SIG_GAP_UM = 1.0
_SIG_STEP_UM = 0.44    # ... at places this far apart along a path;
_SIG_MAX_DEG = 20.0    # courses within this angle are parallel ...
_SIG_MIN_SHIFT_UM = 0.5  # ... and one shifted sideways by more than this onto the
#                        other is a sigmoid step, if
_SIG_END_NEAR_UM = 3.0 # another fibre ends within this distance of the step,
_SIG_MIN_FIBRE_UM = 8.0  # at least this long (a crumb has no course of its own),
_SIG_MAX_MISS_UM = 1.0 # whose course continues the new track to within this,
_SIG_MARGIN_UM = 0.1   # better than the path's own old course by this margin,
#                        and whose thickness matches the new track as well
_SIG_PASSES = 3        # repeated on the corrected paths at most this often
_SIG_BEND_MISS_UM = 1.5  # a fibre leaving the contact on a gentle bend may miss the new track by up to this,
_SIG_BEND_SHIFT = 0.5    # if it still lands nearer than this share of the step (halfway to the old course)
_ORPHAN_UM = 25.0          # a free-standing piece this short is a fragment of a fibre, not a fibre (`_fix_sigmoids`):
_ORPHAN_NEAR_UM = 3.5     # it starts within this distance of a contact where one fibre passes and another ends,
_ORPHAN_DEPTH_UM = 1.1    # and fibres keep their order in depth across a contact (at least this far apart)
_ORPHAN_REACH_UM = 2.2    # the fibres it is paired with lie this close (2.5 fibre widths) to each other at the contact
_BLOB_RATIO = 1.6          # two fibres that overlap make a blob this many times the lone fibre's cross-section (`blob_handover`);
_BLOB_MIN_UM = 2.5         # it lasts at least this long and starts within _BLOB_NEAR_UM of where a fibre ends against the path;
_BLOB_NEAR_UM = 3.5        # the single fibre that comes out of it goes to the fibre whose depth range (top and bottom of the raw mask)
_BLOB_EDGE_MAX = 4.0       # it matches: both edges together within this many slices, and
_BLOB_EDGE_RATIO = 0.5     # at most this share of what the other fibre's edges differ by
_BRIDGE_NEAR_UM = 3.5      # a fibre that goes into a blob and one that comes out of it (`bridge_blobs`): both ends within this of the path
_BRIDGE_ALONG_UM = (4.0, 14.0)   # that runs through the blob, this far apart along it,
_BRIDGE_TURN_DEG = 25.0    # lined up within this angle and
_BRIDGE_SHIFT_UM = 1.8     # this far off one line,
_BRIDGE_BLOB_SHARE = 0.6   # with at least this share of the stretch between them a blob (_BLOB_RATIO times a lone fibre's section)
_BLOB_COURSES = ((6, 16), (8, 20), (12, 28), (16, 32))   # or: courses taken this far before the contact (um, so that the bend into
_BLOB_COURSE_MED = 0.2     # the contact stays out of them), extrapolated across the blob, miss it at most this share (median) of the
_BLOB_COURSE_ALL = 0.4     # other's miss, and at most this share in every window
_BLOB_LONG_SCAN_UM = 24.0  # a blob that outlasts the first 16 um is looked for this far, against the lone fibre that comes out of it;
_BLOB_INSIDE_UM = 4.0      # a fibre that ends inside a blob (double over this much before its end) goes on through it with
_BLOB_SIDE_UM = 4.0        # the passing fibre and takes what comes out, if the two lie in layers at least this far apart in depth
_BLOB_LAYER_INSET_UM = 0.5 # the passing fibre then keeps to its own layer of the blob, this far inside the mask's edge
_BLOB_SIDE_MATCH_UM = 0.5  # and what comes out is at the ending fibre's own depth, to within this; a blob found only by the
#                            longer scan is handed over on this evidence alone (and only such a blob is)
_BLOB_LEVEL_DEG = 10.0     # depth ranges are compared only for courses this close to level: one that climbs or dips
#                            more steeply is at another depth at the contact than over the stretch before it
_SWAP_MIN_SHIFT_UM = 0.9  # crossing fibres (`_fix_sigmoids`): both steps are at least a fibre width (2 r),
_SWAP_FIT_RATIO = 0.5    # and each swapped course misses by at most this fraction of what the original did
# Bend repair (`_rejoin_at_bends`): a path that turns by more than _BEND_MIN_DEG (judged on its courses
# _JOIN_SKIP_UM to _JOIN_WINDOW_UM either side, as in `_end_course`) where another path of at least
# _SIG_MIN_FIBRE_UM touches it (within _JOIN_SKIP_UM, the stretch the courses leave out) and ends there
# (up to _BEND_HOOK_UM of a hook beyond the contact is ignored) is re-paired with it if that path lines up
# with one of the two legs to within _BEND_LINE_DEG and _BEND_LINE_RATIO of the bend, lies off that leg's
# line by at most _JOIN_SHIFT_R r, and stays in the leg's depth layer (_LAYER_SAME_R r)
_BEND_MIN_DEG = 20.0
_BEND_HOOK_UM = 8.0
_BEND_LINE_DEG = 12.0
_BEND_LINE_RATIO = 0.5
_DIVE_IN_DEG = 16.0      # a path that dips (or climbs) into a contact at least this steeply (`_end_depth_reversals`)
_DIVE_OUT_DEG = 8.0      # and leaves it climbing (or dipping) back at least this steeply,
_DIVE_LEG_UM = 8.0       # both courses measured over this much path on either side,
_DIVE_NEAR_UM = 3.5      # with a piece too short to be a fibre (< _SIG_MIN_FIBRE_UM) within this of the turn,
_DIVE_AIM_DEG = 30.0     # starting where the incoming course points and running along it (both to within this), on its far side in depth,
_DIVE_MIN_UM = 2.5       # reaching at least this far along the incoming course (and no further than _DIVE_REACH_UM),
_DIVE_REACH_UM = 8.0     # and ending free, goes on into that piece and ends there; the outgoing leg is a path of its own
_DIVE_PATH_UM = 30.0     # only paths at least this long are repaired (both legs must stand as fibres afterwards)
_DIVE_EDGE_SLICES = 2    # and not where the piece runs to within this many slices of the top or bottom of the box
_ORDER_TOUCH_UM = 1.0    # two paths of at least 30 um that touch (come within this; `_keep_order_at_crossings`)
_ORDER_SPAN_UM = 8.0     # along no more than this much of their length,
_ORDER_PAR_DEG = 20.0    # parallel within this on both sides of the contact (courses 2 to 12 um away),
_ORDER_SIDE_UM = 0.9     # and on opposite sides of each other before and after it, at least this far apart,
_ORDER_TURN_DEG = 20.0   # swap their continuations if each then turns by no more than this: parallel fibres keep their order
_CLIMB_DEG = 40.0        # a path of at least 30 um that, after a course within _CLIMB_LEVEL_DEG of level over 12 um,
_CLIMB_LEVEL_DEG = 20.0  # climbs or dips at least this steeply (over 2 um steps) through at least _CLIMB_DEPTH_UM of depth
_HOP_LEVEL_DEG = 10.0    # a climb in mid-path is a hop between two fibres if the path runs within this of level, 12 um before and after
_CLIMB_DEPTH_UM = 3.0    # to its end has cut across the layers (`_cut_climbs`): it is cut where the climb starts,
_CLIMB_NEAR_UM = 5.0     # and it, and the paths that end within this of the climb, go on along their own courses
_MARCH_STEP_UM = 0.5     # through mask that no other path claims: steps of this length, each centred on the free mask
_MARCH_DISC_UM = 1.2     # within this of the course (across it), at least _MARCH_CLEAR_UM from every other path,
_MARCH_CLEAR_UM = 0.9    # turning at most _MARCH_TURN_DEG per step, until no free mask is left ahead (at most _MARCH_MAX_UM)
_MARCH_TURN_DEG = 10.0
_MARCH_MAX_UM = 60.0
_OWN_DEV_UM = 0.8        # a stretch where a path leaves its own line (`_follow_own_line`) by more than this
_OWN_FLANK_UM = 7.0      # (the line through this much path on either side, straight within 0.45 um, parallel within
_OWN_PAR_DEG = 15.0      # this), up to _OWN_MAX_UM long, is put back on its line if it rides along another fibre's
_OWN_MAX_UM = 15.0       # path (within _OWN_RIDE_UM) for at least _OWN_RIDE_LEN_UM, or if the line there runs through
_OWN_RIDE_UM = 0.8       # a gap in the mask (_OWN_GAP_SHARE of it outside) and the stretch veers to within
_OWN_RIDE_LEN_UM = 2.5   # _OWN_VEER_UM of another fibre's path
_OWN_GAP_SHARE = 0.6
_OWN_VEER_UM = 1.3
_CONTACT_UM = 1.0        # two paths of at least 30 um that touch (within this, `_resolve_contacts_by_intensity`)
_CONTACT_SPAN_UM = 10.0  # along at most this much of each, at most _CONTACT_DEG apart, are each followed through the
_CONTACT_DEG = 45.0      # contact by their bright core: from the _CONTACT_LEG_UM of path that ends _CONTACT_GAP_UM
_CONTACT_GAP_UM = 4.0    # before it, for the contact's length + _CONTACT_GAP_UM + _CONTACT_EXTRA_UM. If each core comes
_CONTACT_LEG_UM = 8.0    # out on the other's continuation (nearer than _CONTACT_RATIO of its distance to its own, and
_CONTACT_EXTRA_UM = 10.0 # within _CONTACT_HIT_UM of it), the two continuations are swapped
_CONTACT_RATIO = 0.7
_CONTACT_HIT_UM = 1.0
_LOOP_BACK_UM = 1.0      # a path that runs back against its own course (`_cut_loops`) by more than this has a loop
_STEP_LEG_UM = 8.0       # a path of at least 30 um that steps sideways (`_split_overlapping_steps`) between two
_STEP_PAR_DEG = 10.0     # straight stretches of this length (median miss <= _STEP_STRAIGHT_UM), parallel within this,
_STEP_STRAIGHT_UM = 0.3
_STEP_SHIFT_UM = (0.6, 1.5)  # offset by this much at the step, where a short dead-end piece of skeleton (3 to 8 um)
_STEP_NEAR_UM = 2.0      # lies within this of the step, parallel within _STEP_PIECE_DEG, and carries the old line on
_STEP_PIECE_DEG = 20.0   # past the step (or the new one back before it), clearly (miss < _STEP_RATIO of the other
_STEP_RATIO = 0.7        # line's): two fibres overlap there, and the path is split at the step
_HAND_TOUCH_UM = 1.5     # a path of at least 30 um passing within this of where a parallel path ends (`_hand_on_at_ends`),
_HAND_SIDE_UM = 0.9      # that path having run beside it at least this far off its course,
_HAND_ON_RATIO = 0.6     # while a third path starts within 3 um of there on the passing path's own line (missing it by at
#                          most this share of that offset, and by less than the passing path's own rest does): the passing
#                          path goes on into the third path, and the ending one into the passing path's rest
_LINE_GAP_UM = 20.0      # two path ends at most this far apart (`_join_lined_up_ends`), facing each other,
_LINE_TURN_DEG = 15.0    # whose courses continue each other within this (_LINE_LONG_TURN_DEG beyond _JOIN_FAR_R * r),
_LINE_LONG_TURN_DEG = 10.0
_LINE_SHIFT_UM = 2.0     # and lie on one line to within this midway, are joined along the mask (a route at most
_LINE_ROUTE_RATIO = 1.5  # this many times the gap, out of the mask for at most _JOIN_MASK_GAP_R * r, and at least
_LINE_CLEAR_UM = 0.9     # this far from other paths over _LINE_CLEAR_SHARE of it), or, if no longer than _JOIN_FAR_R * r
_LINE_CLEAR_SHARE = 0.8  # and on one line to within _LINE_STRAIGHT_UM, straight across a gap in the mask. An end may
_LINE_STRAIGHT_UM = 1.2  # first drop a tail of up to 6 um that runs past the other piece (a hook off the course).
_LINE_OVERLAP_UM = 3.0   # ends that run past each other by up to this, on one line to within _LINE_STRAIGHT_UM and
_LINE_OVERLAP_TURN_DEG = 22.0  # continuing each other's course within this, are joined too
_FREE_MARCH_MIN_UM = 8.0 # a free path end then goes on through mask nobody claims (`_march_free_ends`), if that is this long
_SIDE_TURN_DEG = 10.0    # a path end and the end of a parallel path (within this) beside it (`_join_side_by_side`),
_SIDE_OFF_UM = 3.5       # at most this far to the side, overlapping it by up to _SIDE_OVERLAP_UM or at most
_SIDE_OVERLAP_UM = 10.0  # _SIDE_AHEAD_UM ahead of it, are one fibre with a sideways jog: joined through the mask (a route
_SIDE_ROUTE_RATIO = 2.5  # (the route through the mask between them may be up to this many times their distance + 1.5 um)
_SIDE_AHEAD_UM = 3.0     # at most 1.6 x the direct distance + 1.5 um, clear of other paths), dropping the overlap


# ---- tiling ------------------------------------------------------------------ #
def _tile_targets(box, tile):
    """The tiles' own regions: `box` cut into blocks of `tile`, each voxel in
    exactly one."""
    import itertools
    axes = [range(b.start, b.stop, t) for b, t in zip(box, tile)]
    for starts in itertools.product(*axes):
        yield tuple(slice(s0, min(s0 + t, b.stop))
                    for s0, t, b in zip(starts, tile, box))


def _grow(target, halo, bounds):
    """`target` grown by `halo` voxels per axis, clamped to `bounds`."""
    return tuple(slice(max(bd.start, t.start - h), min(bd.stop, t.stop + h))
                 for t, h, bd in zip(target, halo, bounds))


def _local(inner, outer):
    """`inner` expressed in the coordinates of the crop `outer`."""
    return tuple(slice(i.start - o.start, i.stop - o.start)
                 for i, o in zip(inner, outer))


class _Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


# ---- the skeleton graph -------------------------------------------------------- #
_OFFSETS_26 = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
                        for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)])


def _skeleton_graph(pts, bshape):
    """Graph of a 26-connected skeleton given as voxel coordinates.

    Voxels with three or more skeleton neighbours are junction voxels; touching
    ones form one junction (so the small false junctions thinning leaves on a
    diagonal staircase become a junction with two arms, which the pairing
    simply passes through). Returns (node_of_voxel, n_nodes, edges): each edge
    is (node_a, node_b, ordered voxel indices from a to b, including the end
    voxels); node -1 marks a free end, and a closed loop has both ends -1.
    """
    n = len(pts)
    lin = np.ravel_multi_index(pts.T, bshape)
    order = np.argsort(lin)
    lin_s = lin[order]
    nbrs = [[] for _ in range(n)]
    for off in _OFFSETS_26:
        q = pts + off
        ok = np.all((q >= 0) & (q < np.asarray(bshape)), axis=1)
        ql = np.full(n, -1, np.int64)
        ql[ok] = np.ravel_multi_index(q[ok].T, bshape)
        pos = np.searchsorted(lin_s, ql)
        pos = np.minimum(pos, n - 1)
        hit = ok & (lin_s[pos] == ql)
        for i, j in zip(np.nonzero(hit)[0], order[pos[hit]]):
            nbrs[i].append(int(j))
    deg = np.array([len(x) for x in nbrs])
    junction = deg >= 3
    uf = _Union(n)
    for i in np.nonzero(junction)[0]:
        for j in nbrs[i]:
            if junction[j]:
                uf.union(int(i), j)
    node_of = np.full(n, -1, np.int64)
    roots = {}
    for i in np.nonzero(junction)[0]:
        rt = uf.find(int(i))
        node_of[i] = roots.setdefault(rt, len(roots))
    n_nodes = len(roots)
    is_node = junction | (deg <= 1)
    visited = np.zeros(n, bool)
    edges = []
    for start in np.nonzero(is_node)[0]:
        for first in nbrs[start]:
            if junction[start] and junction[first] and node_of[first] == node_of[start]:
                continue
            if not is_node[first] and visited[first]:
                continue
            chain = [int(start)]
            prev, cur = int(start), int(first)
            while not is_node[cur]:
                visited[cur] = True
                chain.append(cur)
                nxt = [j for j in nbrs[cur] if j != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            chain.append(cur)
            a_node = int(node_of[start]) if junction[start] else -1
            b_node = int(node_of[cur]) if junction[cur] else -1
            if len(chain) == 2 and is_node[chain[0]] and is_node[chain[1]]:
                if chain[0] > chain[1]:
                    continue            # a direct node-node link, recorded once
            edges.append((a_node, b_node, np.array(chain)))
    # Closed loops with no junction on them.
    for i in np.nonzero(~is_node & ~visited)[0]:
        if visited[i]:
            continue
        chain = [int(i)]
        visited[i] = True
        prev, cur = int(i), nbrs[i][0]
        while cur != i and not visited[cur]:
            visited[cur] = True
            chain.append(int(cur))
            nxt = [j for j in nbrs[cur] if j != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        edges.append((-1, -1, np.array(chain)))
    return node_of, n_nodes, edges


def _arc_points(xyz, dist_from_start):
    """Cumulative arc length along an ordered polyline (physical units)."""
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.zeros(0)
    arc = np.r_[0.0, np.cumsum(seg)]
    return arc


def _arm(xyz, r):
    """An arm leaving its junction: `xyz` ordered from the junction outward.
    Returns (anchor, direction): the direction from 2 r to 10 r along the arm,
    so the bending right at the junction does not count."""
    arc = _arc_points(xyz, None)
    L = arc[-1]
    i0 = int(np.searchsorted(arc, min(2 * r, 0.25 * L)))
    i1 = int(np.searchsorted(arc, min(_ARM_WINDOW_R * r, L)))
    i1 = min(max(i1, i0 + 1), len(xyz) - 1)
    i0 = min(i0, i1 - 1) if i1 > 0 else 0
    d = xyz[i1] - xyz[i0]
    nrm = np.linalg.norm(d)
    if nrm == 0:
        d = xyz[-1] - xyz[0]
        nrm = np.linalg.norm(d) or 1.0
    return xyz[0], d / nrm


def _line_voxels(p0, p1):
    """Voxels on the straight segment from p0 to p1, in order along it."""
    n = int(np.max(np.abs(p1 - p0))) + 1
    t = np.linspace(0.0, 1.0, n)[:, None]
    v = np.round(p0 + (p1 - p0) * t).astype(np.int64)
    keep = np.r_[True, np.any(np.diff(v, axis=0) != 0, axis=1)]
    return v[keep]


def _z_run_length(obj):
    """Length (voxels) of the run of object voxels along z through each voxel;
    0 outside the object."""
    Z = obj.shape[0]
    up = np.zeros(obj.shape, np.int16)
    dn = np.zeros(obj.shape, np.int16)
    for z in range(Z):
        up[z] = np.where(obj[z], (up[z - 1] + 1) if z else 1, 0)
    for z in range(Z - 1, -1, -1):
        dn[z] = np.where(obj[z], (dn[z + 1] + 1) if z < Z - 1 else 1, 0)
    return np.where(obj, up + dn - 1, 0)


def _cut_stacked_fibres(obj, inten, h0, noise, spacing_yx):
    """The mask with every contact of two stacked fibres cut between them.

    Two fibres lying one above the other merge, where they touch, into mask
    columns up to about twice a fibre's height. Thinning such a stretch leaves
    a single centre line between the fibres and bends both into it -- so the
    fibres swap, or break, at the contact. Cut first, each fibre keeps a
    centre line of its own.

    The cut has to separate the whole contact: a cut with a hole lets
    thinning join the fibres through the hole, and cuts at different heights
    in neighbouring columns leave a diagonal step between them. So, working
    on runs of object voxels along z: runs longer than _STACK_K x `h0` (the
    object's single-fibre column height) are clearly two fibres; the contact
    is followed from them into neighbouring runs at least _STACK_LOW_K x `h0`
    long that overlap them in depth, up to _STACK_FOLLOW_UM beyond them; and
    every run of the contact is cut at one smooth height -- the median over
    the clearly stacked runs around it of their own cut heights: the
    intensity valley between the run's two brightest maxima where `inten`
    shows two maxima that stand out of the noise, else the run's middle. The
    cut is 3 voxels thick, and never within 2 voxels of a run's top or bottom,
    where it would only shave a single fibre. `inten`: the intensity crop
    (None: no intensity); `noise`: the noise of its 3 x 3 in-plane mean.
    """
    from scipy.signal import find_peaks
    from scipy.spatial import cKDTree
    pad = np.pad(obj, ((1, 1), (0, 0), (0, 0))).astype(np.int8)
    d = np.diff(pad, axis=0).transpose(1, 2, 0)
    starts = np.argwhere(d == 1)                # (y, x, first z), sorted like ends
    ends = np.argwhere(d == -1)                 # (y, x, z after the last)
    y, x, a = starts[:, 0], starts[:, 1], starts[:, 2]
    b = ends[:, 2]
    n = b - a
    stacked = n > _STACK_K * h0
    if not stacked.any():
        return obj
    # cut height of each clearly stacked run
    sidx = np.nonzero(stacked)[0]
    target = np.empty(len(sidx))
    for k, i in enumerate(sidx):
        t = a[i] + int(round(n[i] / 2))
        if inten is not None and noise > 0:
            prof = inten[a[i]:b[i], max(0, y[i] - 1):y[i] + 2, max(0, x[i] - 1):x[i] + 2].mean(axis=(1, 2))
            prof = ndimage.gaussian_filter1d(prof.astype(np.float64), 1.0)
            pk, _ = find_peaks(prof, prominence=3.0 * noise, distance=max(2, int(h0) // 2))
            if len(pk) >= 2:
                i0, i1 = sorted(pk[np.argsort(prof[pk])[-2:]])
                t = a[i] + i0 + int(np.argmin(prof[i0:i1 + 1]))
        target[k] = t
    # follow each contact from its clearly stacked runs into the runs next to them
    by_column = {}
    for i in np.nonzero(n >= _STACK_LOW_K * h0)[0]:
        by_column.setdefault((y[i], x[i]), []).append(i)
    stacked_xy = cKDTree(np.c_[y[sidx], x[sidx]].astype(float))
    reach = _STACK_FOLLOW_UM / spacing_yx
    contact = set(sidx.tolist())
    frontier = list(contact)
    while frontier:
        nxt = []
        for i in frontier:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j in by_column.get((y[i] + dy, x[i] + dx), ()):
                        if j in contact or min(b[i], b[j]) - max(a[i], a[j]) <= 0:
                            continue            # already in, or no overlap in depth
                        if stacked_xy.query([y[j], x[j]])[0] > reach:
                            continue
                        contact.add(j)
                        nxt.append(j)
        frontier = nxt
    # one smooth cut through the whole contact
    out = obj.copy()
    for i in sorted(contact):
        near = stacked_xy.query_ball_point([y[i], x[i]], r=4.0)
        if not near:
            near = [stacked_xy.query([y[i], x[i]])[1]]
        t = int(round(np.median(target[near])))
        if t - 1 < a[i] + 2 or t + 1 > b[i] - 3:
            continue                            # would only shave a single fibre
        out[t - 1:t + 2, y[i], x[i]] = False
    return out


def _neck_lines(obj, sp, r):
    """The thin necks between fibres that touch side by side, as voxel coordinates of `obj`.

    Two fibres that run next to each other -- side by side, one above the other, or
    diagonally offset -- merge in the mask into a cross-section of two lobes joined by a
    neck. Thinning leaves one centre line for both, and it can run along the neck: that
    path sits between the two fibres, neither gets a centre line of its own, and where the
    neck ends the path steps onto one of them. `_cut_stacked_fibres` cuts only contacts
    whose mask columns are clearly two fibres tall; a diagonal offset keeps every column
    shorter. Here each fibre's core is where the mask is at least _NECK_CORE_R r deep
    (physical distance to the background); the neck between two cores is the watershed line
    of that depth between them; it is returned if it is at least _NECK_LEN_UM long in-plane,
    i.e. the cores lie side by side. Where a single fibre only narrows, the line between its
    two pieces is a short disc across it and is not returned. `sp`: spacing (z, y, x).
    """
    sp = np.asarray(sp, float)
    dt = ndimage.distance_transform_edt(obj, sampling=sp).astype(np.float32)
    lab, n = ndimage.label(dt >= _NECK_CORE_R * r, structure=np.ones((3, 3, 3)))
    if n < 2:
        return []
    ws = watershed(-dt, lab, mask=obj, watershed_line=True)
    del dt, lab
    lz, ly, lx = np.nonzero(obj & (ws == 0))
    if not len(lz):
        return []
    lo = np.full(len(lz), np.iinfo(np.int64).max, np.int64)
    hi = np.zeros(len(lz), np.int64)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                z2, y2, x2 = lz + dz, ly + dy, lx + dx
                ok = ((z2 >= 0) & (z2 < ws.shape[0]) & (y2 >= 0) & (y2 < ws.shape[1])
                      & (x2 >= 0) & (x2 < ws.shape[2]))
                v = np.zeros(len(lz), np.int64)
                v[ok] = ws[z2[ok], y2[ok], x2[ok]]
                lo = np.where(v > 0, np.minimum(lo, v), lo)
                hi = np.maximum(hi, v)
    del ws
    two = np.nonzero((hi > 0) & (lo < hi))[0]          # line voxels between two cores
    if not len(two):
        return []
    key = lo[two] * (n + 1) + hi[two]
    order = np.argsort(key, kind="stable")
    two, key = two[order], key[order]
    bounds = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    lines = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = two[a:b]
        if len(sel) < 3:
            continue
        yx = np.c_[ly[sel], lx[sel]] * sp[1:]
        yx = yx - yx.mean(0)
        u = np.linalg.svd(yx, full_matrices=False)[2][0]
        if float(np.ptp(yx @ u)) >= _NECK_LEN_UM:
            lines.append(np.c_[lz[sel], ly[sel], lx[sel]].astype(np.int64))
    return lines


def _remove_hairs(edges, half_w, length, xyz, r):
    """Remove the skeleton's hairs and merge what they leave behind.

    A rough mask surface gives its skeleton a hair every few fibre widths,
    and every hair is a junction a fibre has to be traced through. Repeated
    until nothing changes: end branches no longer than _HAIR_K half-widths
    where they leave are removed, shortest first, never taking a node below
    two branches; a node left with one branch becomes a free end; the two
    branches of a node left with two are merged into one, so a fibre that only
    had hairs on it becomes one long branch -- but only if they continue each
    other, turning by less than _MAX_TURN_DEG (each branch's direction taken
    from 2 r to 10 r out, as `_arm`). The node a hair leaves behind can just
    as well be where two DIFFERENT fibres meet at an angle; merging those
    without looking welds the fibres into one, which no later step can undo.
    A node that fails the test stays a junction, and the tracer decides there
    by the same rule. `edges` as from `_skeleton_graph`, `length(edge)` its
    physical length, `xyz` the skeleton voxels in microns.
    """

    def continues(c1, c2):
        """Do branch c1 (ending at the node) and c2 (starting there) continue
        each other?"""
        _, d1 = _arm(xyz[c1[::-1]], r)
        _, d2 = _arm(xyz[c2], r)
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(d1 @ d2))))) < _MAX_TURN_DEG
    E = {i: e for i, e in enumerate(edges)}
    nxt = len(edges)
    inc = {}                            # node -> ids of the edges ending there

    def add_inc(i):
        na, nb, _ = E[i]
        for nd in (na, nb):
            if nd >= 0:
                inc.setdefault(nd, []).append(i)

    def remove(i):
        na, nb, _ = E.pop(i)
        for nd in (na, nb):
            if nd >= 0:
                inc[nd].remove(i)

    for i in list(E):
        add_inc(i)
    for _it in range(1000):
        changed = False
        hairs = []
        for i, (na, nb, ch) in E.items():
            if (na >= 0) == (nb >= 0):
                continue
            nd, jv = (na, ch[0]) if na >= 0 else (nb, ch[-1])
            L = length(E[i])
            if L <= _HAIR_K * float(half_w[jv]):
                hairs.append((L, i, nd))
        for L, i, nd in sorted(hairs):
            if i in E and len(inc.get(nd, ())) >= 3:
                remove(i)
                changed = True
        for nd in [n for n, l in inc.items() if len(l) == 1]:
            i = inc[nd][0]
            na, nb, ch = E[i]
            remove(i)
            E[i] = (-1 if na == nd else na, -1 if nb == nd else nb, ch)
            add_inc(i)
            changed = True
        for nd in [n for n, l in inc.items() if len(l) == 2]:
            l = inc.get(nd, [])
            if len(l) != 2 or l[0] == l[1]:
                continue
            i, j = l
            a1, b1, c1 = E[i]
            a2, b2, c2 = E[j]
            if b1 != nd:
                a1, b1, c1 = b1, a1, c1[::-1]
            if a2 != nd:
                a2, b2, c2 = b2, a2, c2[::-1]
            if not continues(c1, c2):
                continue                # a corner between two fibres: keep the junction
            if len(c2) and len(c1) and c1[-1] == c2[0]:
                c2 = c2[1:]
            remove(i)
            remove(j)
            E[nxt] = (a1, b2, np.concatenate([c1, c2]))
            add_inc(nxt)
            nxt += 1
            changed = True
        if not changed:
            break
    return [E[i] for i in sorted(E)]


def _leftover_chains(pts, paths, sp):
    """Chains through the skeleton that no real path covers.

    A bright fibre that lies on a dim one shares a ladder of very short branches
    with it: every branch is shorter than the baseline the tracer needs to give a
    path a direction, so the dim fibre takes one rail and the other rail is left as
    crumbs. Skeleton voxels more than a voxel from every path of at least _CRUMB_UM
    are grouped into connected pieces, and the longest chain through each piece
    (found with two shortest-path searches) is returned if it is at least
    _SALVAGE_UM long, as an ordered voxel array.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree

    def plen(p):
        return float(np.sum(np.linalg.norm(np.diff(p * sp, axis=0), axis=1))) if len(p) > 1 else 0.0

    real = [p for p in paths if plen(p) >= _CRUMB_UM]
    if not real or len(pts) == 0:
        return []
    covered = cKDTree(np.vstack(real) * sp)
    keep = covered.query(pts * sp)[0] > 0.36            # more than one voxel step from every real path
    U = pts[keep]
    if len(U) < 8:
        return []
    pairs = cKDTree(U * sp).query_pairs(0.36, output_type="ndarray")
    if not len(pairs):
        return []
    w = np.linalg.norm((U[pairs[:, 0]] - U[pairs[:, 1]]) * sp, axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                   shape=(len(U), len(U))).tocsr()
    n_comp, label = connected_components(G, directed=False)
    out = []
    for c in range(n_comp):
        members = np.nonzero(label == c)[0]
        if len(members) < 8:
            continue
        sub = G[members][:, members]
        d0 = dijkstra(sub, indices=0)
        a = int(np.argmax(np.where(np.isinf(d0), -1.0, d0)))
        d1, pred = dijkstra(sub, indices=a, return_predecessors=True)
        b = int(np.argmax(np.where(np.isinf(d1), -1.0, d1)))
        if d1[b] < _SALVAGE_UM:
            continue
        chain = [b]
        while chain[-1] != a:
            chain.append(int(pred[chain[-1]]))
        out.append(U[members[chain[::-1]]])
    return out


def _paths_from_skeleton(pts, half_w, bshape, spacing, r, obj_lookup):
    """Decompose a mask skeleton into fibre paths, each a chain, never a tree.

    `pts` are skeleton voxels (box coordinates), `half_w` the mask's in-plane
    half-width at each. Steps: remove hairs (`_remove_hairs`); merge
    junctions joined by a branch shorter than one fibre width into one
    junction zone (they lie inside one contact); split ribbons (branches
    wider than one fibre) into k parallel lanes, k = half-width / r; then
    trace fibres through the zones, each branch used once, so every path is
    a chain (see the tracing comment below); join pieces end to end
    (`_stitch_paths`); and undo sigmoid switches between parallel fibres
    (`_fix_sigmoids`). A bridge between fibres -- a touch point, a ladder rung, which
    meets them at an angle -- is never taken as a continuation and is left as
    a short path of its own. Branch length is not used to call something a
    bridge: in a dense bundle most fibre stretches between two touch points
    are short too. `obj_lookup(coords)` says which voxel
    coordinates lie in the object's mask. Returns a list of voxel arrays, one
    line per path.
    """
    sp = np.asarray(spacing, float)
    if len(pts) == 0:
        return []
    node_of, n_nodes, edges = _skeleton_graph(pts, bshape)
    xyz_all = pts * sp

    def length(e):
        return float(_arc_points(xyz_all[e[2]], None)[-1])

    # ---- remove hairs --------------------------------------------------------
    edges = _remove_hairs(edges, half_w, length, xyz_all, r)

    # ---- junctions inside one contact form one zone ---------------------------
    bridge = set()
    for k, (na, nb, ch) in enumerate(edges):
        if na >= 0 and nb >= 0 and na != nb and length(edges[k]) <= _CORE_R * r:
            bridge.add(k)
    zone = _Union(max(n_nodes, 1))
    for k in bridge:
        zone.union(edges[k][0], edges[k][1])

    # Each non-bridge branch becomes one or more lanes (ordered voxel arrays).
    lanes = []                          # (zone_a or -1, zone_b or -1, voxels)
    for k, (na, nb, ch) in enumerate(edges):
        if k in bridge:
            continue
        za = zone.find(na) if na >= 0 else -1
        zb = zone.find(nb) if nb >= 0 else -1
        v = pts[ch]
        w = half_w[ch]
        kk = int(round(float(np.median(w)) / r)) if float(np.median(w)) > _RIBBON_R * r else 1
        if kk <= 1 or len(ch) < 3:
            lanes.append((za, zb, v))
            continue
        # Ribbon: kk lanes, spaced evenly across its local width, in-plane.
        xyz = xyz_all[ch]
        arc = _arc_points(xyz, None)
        lane_pts = [[] for _ in range(kk)]
        for i in range(len(ch)):
            j0 = int(np.searchsorted(arc, arc[i] - 2 * r))
            j1 = min(len(ch) - 1, int(np.searchsorted(arc, arc[i] + 2 * r)))
            t = xyz[j1] - xyz[j0]
            t2 = t[1:] / (np.linalg.norm(t[1:]) + 1e-9)
            nrm = np.array([0.0, -t2[1], t2[0]])
            for j in range(kk):
                off = ((j + 0.5) / kk * 2.0 - 1.0) * float(w[i])
                q = np.round((xyz[i] + off * nrm) / sp).astype(np.int64)
                lane_pts[j].append(q)
        for j in range(kk):
            q = np.array(lane_pts[j])
            q = q[np.all((q >= 0) & (q < np.asarray(bshape)), axis=1)]
            q = q[obj_lookup(q)] if len(q) else q
            if len(q) >= 2:
                lanes.append((za, zb, q))

    # ---- trace fibres through the zones ---------------------------------------
    # Each branch between two contacts is short, too short a baseline for its
    # own direction to decide a junction. So fibres are traced: starting from
    # the longest unused branch, a path is extended through each junction onto
    # the branch that stays closest to the line of the path's OWN last 10 r --
    # a long baseline -- provided it turns the path by less than
    # _MAX_TURN_DEG; more than that is a step onto another fibre.
    # Inside a contact the skeleton is bent towards the junction, which sits
    # between the fibres rather than on either one's centre line. So near a
    # junction a path's own voxels are dropped -- those within the contact's
    # size, twice the mask's half-width there -- and the two sides are joined
    # by a straight segment: on either side of a contact a fibre is locally
    # straight, so that follows its real course instead of a jog towards the
    # contact point. The same trim applies where a path ends at a junction.
    # The continuation is DECIDED on that same trimmed geometry: the path's
    # direction and end point without its bend into this contact, and each
    # candidate branch without its bends into this contact and the next one.
    # Decided on the untrimmed skeleton, a straight fibre looks like a turn
    # at every contact, and a bundle breaks at every contact.
    # Each branch is used once, so paths are chains. Where two fibres touch
    # over a stretch they share one skeleton branch; letting a second path
    # run along it let that path follow the other fibre's route well past the
    # contact -- two seeds interleaved on one fibre, and fibres swapped at the
    # contact -- so one of the two fibres ends there instead. Step 4 can join
    # the pieces; it cannot undo a swap. Longest-first lets clear fibres claim
    # their continuations before short fragments can.
    zone_pts, zone_w = {}, {}
    for i in np.nonzero(node_of >= 0)[0]:
        zn = zone.find(int(node_of[i]))
        zone_pts.setdefault(zn, []).append(xyz_all[i])
        zone_w[zn] = max(zone_w.get(zn, 0.0), float(half_w[i]))
    for k in bridge:
        zn = zone.find(edges[k][0])
        for i in edges[k][2]:
            zone_pts.setdefault(zn, []).append(xyz_all[i])
            zone_w[zn] = max(zone_w.get(zn, 0.0), float(half_w[i]))
    zone_pts = {zn: np.array(v) for zn, v in zone_pts.items()}

    def trim_tail(path_vox, zn):
        """Drop the path's last voxels that lie inside contact `zn`."""
        if zn < 0 or zn not in zone_pts:
            return path_vox
        reach = 2.0 * zone_w[zn]
        zp = zone_pts[zn]
        keep = len(path_vox)
        floor = max(2, len(path_vox) // 2)       # never trim more than half
        while keep > floor:
            d = np.min(np.linalg.norm(zp - path_vox[keep - 1] * sp, axis=1))
            if d >= reach:
                break
            keep -= 1
        return path_vox[:keep]

    incident = {}                       # zone -> [(lane, end index 0 | -1)]
    for li, (za, zb, v) in enumerate(lanes):
        if za >= 0:
            incident.setdefault(za, []).append((li, 0))
        if zb >= 0:
            incident.setdefault(zb, []).append((li, -1))
    lane_len = [float(_arc_points(v * sp, None)[-1]) for _za, _zb, v in lanes]
    used = np.zeros(len(lanes), bool)
    win = _ARM_WINDOW_R * r

    def head(coords_xyz):
        """End point and direction of a path over its last `win`."""
        arc = _arc_points(coords_xyz[::-1], None)
        i1 = min(int(np.searchsorted(arc, win)), len(coords_xyz) - 1)
        d = coords_xyz[-1] - coords_xyz[-1 - i1] if i1 > 0 else np.zeros(3)
        nrm = np.linalg.norm(d)
        return coords_xyz[-1], (d / nrm if nrm > 0 else None)

    def extend(path_vox, zone_id):
        """Grow an ordered path (voxels) from its last voxel at `zone_id`."""
        while zone_id >= 0:
            p_end, u = head(trim_tail(path_vox, zone_id) * sp)
            if u is None:
                return path_vox
            best = None
            cands = []
            for li, e in incident.get(zone_id, ()):
                if used[li]:
                    continue
                v = lanes[li][2]
                v = v if e == 0 else v[::-1]
                far = lanes[li][1] if e == 0 else lanes[li][0]
                vd = trim_tail(v[::-1], zone_id)[::-1]
                vd = trim_tail(vd, far) if far != zone_id else vd
                xyz = vd * sp
                arc = _arc_points(xyz, None)
                q = xyz[min(int(np.searchsorted(arc, win)), len(xyz) - 1)]
                w = q - p_end
                fwd = float(w @ u)
                if fwd <= 0:
                    continue
                lat = float(np.linalg.norm(w - fwd * u))
                if math.degrees(math.atan2(lat, fwd)) >= _MAX_TURN_DEG:
                    continue            # a step onto another fibre
                cost = lat / fwd
                dz = abs(float(w[0]))               # depth change onto this lane (um)
                cands.append((cost, li, v, e, dz))
            if cands:
                # A fibre's depth is the least certain of its coordinates (a fibre is ~8
                # slices tall), so a small drift in z can make a hop onto the fibre
                # layer below look like a straight continuation. Where a lane that stays
                # in the path's own layer is also open, the path does not hop layers.
                level = [c for c in cands if c[4] <= _LAYER_SAME_R * r]
                pick = min(cands, key=lambda c: c[0])       # the first of equal costs, as before
                if level and pick[4] > _LAYER_HOP_R * r:
                    pick = min(level, key=lambda c: c[0])
                best = pick[:4]
            if best is None:
                return trim_tail(path_vox, zone_id)
            _c, li, v, e = best
            used[li] = True
            tail = trim_tail(path_vox, zone_id)
            lead = trim_tail(v[::-1], zone_id)[::-1]
            link = _line_voxels(tail[-1], lead[0])
            link = link[obj_lookup(link)]
            path_vox = np.concatenate([tail, link, lead])
            za, zb = lanes[li][0], lanes[li][1]
            zone_id = zb if e == 0 else za
        return path_vox

    paths = []
    for li in np.argsort(lane_len)[::-1]:
        if used[li]:
            continue
        used[li] = True
        za, zb, v = lanes[li]
        fwd_path = extend(v, zb)                     # grow past the far end
        back = extend(fwd_path[::-1], za)            # then past the near end
        keep = np.r_[True, np.any(np.diff(back, axis=0) != 0, axis=1)]
        paths.append(back[keep])                     # ordered along the fibre
    paths = _stitch_paths(paths, sp, r, obj_lookup)
    paths = _fix_sigmoids(paths, pts, half_w, sp, obj_lookup, bshape)
    # The sigmoid fixes trim and re-join paths, so some ends are final only now: mend again.
    paths = _mend_breaks(paths, sp, r, obj_lookup)
    # Skeleton that no path covers (see `_leftover_chains`) may still be the continuation of a
    # fibre that ran out of branches to follow. It is attached to a path end by the same test
    # as above, and only to a real path; whatever attaches to nothing is dropped.
    leftover = _leftover_chains(pts, paths, sp)
    if leftover:
        paths = _mend_breaks(paths, sp, r, obj_lookup, extras=leftover)
    # Only now are the pieces of every fibre in place, so a bend that a greedy choice at a contact
    # left in one path, while another path carries straight on from it, can be told apart.
    paths = _rejoin_at_bends(paths, sp, r, obj_lookup)
    paths = _end_depth_reversals(paths, sp, r, obj_lookup, bshape)
    paths = _keep_order_at_crossings(paths, sp, obj_lookup)
    paths = _cut_climbs(paths, sp, obj_lookup, bshape)
    paths = _hand_on_at_ends(paths, sp, obj_lookup)
    paths = _join_lined_up_ends(paths, sp, r, obj_lookup, bshape)
    paths = _join_side_by_side(paths, sp, r, obj_lookup, bshape)
    paths = _march_free_ends(paths, sp, obj_lookup, bshape)
    paths = _follow_own_line(paths, sp, obj_lookup)
    paths = _cut_loops(paths, sp, obj_lookup)
    return _split_overlapping_steps(paths, sp, obj_lookup)


def _fix_sigmoids(paths, pts, half_w, sp, obj_lookup, bshape):
    """Undo sigmoid switches: a path that slides sideways from one straight
    course onto a parallel one where another fibre touches it.

    Two fibres that touch either stay parallel, or one ends where it touches
    the other. The tracer can instead lead one fibre along an S-shaped step
    onto the other's course, so that one seed ends abruptly and its
    neighbour carries on along the track that was its. Such a step shows
    along the path as two parallel courses (within _SIG_MAX_DEG, each fitted
    over _SIG_WINDOW_UM) shifted sideways by more than _SIG_MIN_SHIFT_UM. A
    kink -- a sideways step of one fibre, an artefact of the mask -- looks the
    same, so the step is only corrected when the other fibre is there:
    another path of at least _SIG_MIN_FIBRE_UM ends within _SIG_END_NEAR_UM of
    the step, its own course over its last _SIG_WINDOW_UM continues the new
    track to within _SIG_MAX_MISS_UM and better than the path's old course
    does (a fibre that leaves the contact on a gentle bend misses a straight
    line by more: up to _SIG_BEND_MISS_UM is accepted when that is less than
    _SIG_BEND_SHIFT of the step, i.e. it lands on the new track nearer than
    halfway to the old one), and its thickness (the mask's half-width)
    matches the new track at least as well. Then the path ends at the step, and the other fibre is
    joined to the new track by a straight segment. Every path is examined in
    both directions; a path takes part in one correction per pass, and the
    passes repeat on the corrected paths, at most _SIG_PASSES times.

    Two fibres can also cross: both paths slide onto each other's course in the
    same place, each stepping the opposite way, so that neither ends. Then the
    two tails are swapped: each path's old course continues the other's new
    track (see `swap_crossings`).

    A short free-standing piece that starts at a contact where one fibre passes and
    another ends belongs to one of the two. Fibres keep their order in depth across a
    contact: the one above stays above. So the piece joins the incoming fibre that is on
    the same side of the other incoming fibre as the piece is of the passing fibre's
    continuation (see `attach_orphans`).

    Where a fibre ends against a passing path and the mask just beyond is a double blob
    (the two fibres overlapping), they coexist through the blob and one of them ends at its
    end. The single fibre that comes out of the blob lies in the depth range of one of the
    two, read off the raw mask where only one fibre is in the cross-section, or its line,
    from a course taken well before the contact, runs into the one far better than the other's
    does; that one goes on along it and the other ends (see `blob_handover`).
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    tree = cKDTree(pts * sp)

    def thickness(xyz):
        return float(np.median(half_w[tree.query(xyz)[1]]))

    def arc(xyz):
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]

    def fit(xyz):
        c = xyz.mean(0)
        return c, np.linalg.svd(xyz - c, full_matrices=False)[2][0]

    def towards(u, frm, to):
        return u if u @ (to - frm) >= 0 else -u

    def miss(c, u, xyz):
        w = xyz - c
        f = w @ u
        w, f = w[f > 0], f[f > 0]
        return float(np.median(np.linalg.norm(w - np.outer(f, u), axis=1))) if len(f) else None

    W, D = _SIG_WINDOW_UM, _SIG_GAP_UM
    cos_max = np.cos(np.radians(_SIG_MAX_DEG))

    def one_pass(P):
        X = [p_ * sp for p_ in P]
        ends = []
        for i, x in enumerate(X):
            if len(x) >= 6:
                ends += [(i, 0, x[0]), (i, 1, x[-1])]
        if not ends:
            return P, 0
        etree = cKDTree(np.array([e[2] for e in ends]))
        touched, out, n_fixed = set(), list(P), 0
        for i, x0 in enumerate(X):
            if i in touched or len(x0) < 30:
                continue
            best = None
            for rev in (False, True):
                x = x0[::-1] if rev else x0
                a = arc(x)
                for s in np.arange(D + 3.0, a[-1] - D - 3.0, _SIG_STEP_UM):
                    A = x[(a >= s - W) & (a <= s - D)]
                    B = x[(a >= s + D) & (a <= s + W)]
                    if len(A) < 8 or len(B) < 8:
                        continue
                    cA, uA = fit(A)
                    cB, uB = fit(B)
                    uA = towards(uA, A.mean(0), B.mean(0))
                    uB = towards(uB, A.mean(0), B.mean(0))
                    if np.degrees(np.arccos(np.clip(uA @ uB, -1, 1))) > _SIG_MAX_DEG:
                        continue
                    p = x[min(np.searchsorted(a, s), len(x) - 1)]
                    m = (uA + uB) / np.linalg.norm(uA + uB)
                    d = (cB + ((p - cB) @ uB) * uB) - (cA + ((p - cA) @ uA) * uA)
                    shift = float(np.linalg.norm(d - (d @ m) * m))
                    if shift < _SIG_MIN_SHIFT_UM:
                        continue
                    for k in etree.query_ball_point(p, _SIG_END_NEAR_UM):
                        j, which, _pe = ends[k]
                        if j == i or j in touched:
                            continue
                        q = X[j] if which == 1 else X[j][::-1]      # towards its end at the step
                        aq = arc(q)
                        Q = q[aq >= aq[-1] - W]
                        if len(Q) < 8 or aq[-1] < _SIG_MIN_FIBRE_UM:
                            continue
                        cQ, uQ = fit(Q)
                        uQ = towards(uQ, Q[0], Q[-1])
                        if uQ @ uB < cos_max:
                            continue                                # not heading along the new track
                        ahead = B[(B - q[-1]) @ uQ > 0]
                        if len(ahead) < 5:
                            continue
                        mQ, mA = miss(cQ, uQ, ahead), miss(cA, uA, ahead)
                        if mQ is None or mA is None or mQ >= mA - _SIG_MARGIN_UM:
                            continue
                        # A fibre that leaves the contact on a gentle bend misses a straight
                        # extrapolation of its own course by more than _SIG_MAX_MISS_UM. It is
                        # still accepted when it lands on the new track nearer than halfway to
                        # the old course: then the step is the other fibre's, not a kink.
                        if mQ > _SIG_MAX_MISS_UM and (mQ > _SIG_BEND_MISS_UM
                                                      or mQ > _SIG_BEND_SHIFT * shift):
                            continue
                        tA, tB, tQ = thickness(A), thickness(B), thickness(Q)
                        if abs(tQ - tB) > abs(tA - tB):
                            continue
                        if best is None or mA - mQ > best[0]:
                            best = (mA - mQ, s, j, which, rev)
            if best is None:
                continue
            _score, s, j, which, rev = best
            Pi = P[i][::-1] if rev else P[i]
            a = arc(Pi * sp)
            ia = int(np.searchsorted(a, s - D))
            ib = int(np.searchsorted(a, s + D))
            head, tail = Pi[:ia], Pi[ib:]
            q = P[j] if which == 1 else P[j][::-1]
            link = _line_voxels(q[-1], tail[0])
            link = link[obj_lookup(link)] if len(link) else link
            out[i] = head
            out[j] = np.concatenate([q, link, tail])
            touched.update((i, j))
            n_fixed += 1
        return [o for o in out if len(o)], n_fixed

    def steps_of(x):
        """Places along the oriented path `x` where two parallel courses are shifted sideways."""
        a = arc(x)
        out = []
        for s_ in np.arange(D + 3.0, a[-1] - D - 3.0, _SIG_STEP_UM):
            A = x[(a >= s_ - W) & (a <= s_ - D)]
            B = x[(a >= s_ + D) & (a <= s_ + W)]
            if len(A) < 8 or len(B) < 8:
                continue
            cA, uA = fit(A)
            cB, uB = fit(B)
            uA = towards(uA, A.mean(0), B.mean(0))
            uB = towards(uB, A.mean(0), B.mean(0))
            if uA @ uB < cos_max:
                continue
            p_ = x[min(np.searchsorted(a, s_), len(x) - 1)]
            m = (uA + uB) / np.linalg.norm(uA + uB)
            d = (cB + ((p_ - cB) @ uB) * uB) - (cA + ((p_ - cA) @ uA) * uA)
            d = d - (d @ m) * m
            if float(np.linalg.norm(d)) < _SWAP_MIN_SHIFT_UM:
                continue
            out.append((s_, p_, A, B, cA, uA, d))
        return out

    def swap_crossings(P):
        """Swap the tails of two paths that cross: both step onto each other's course at the
        same place, the steps point opposite ways, and each path's old course continues the
        other's new track clearly better than its own."""
        X = [p_ * sp for p_ in P]
        oriented = []                     # (path index, reversed, oriented xyz, steps)
        for i, x in enumerate(X):
            if len(x) < 30:
                continue
            for rev in (False, True):
                xo = x[::-1] if rev else x
                oriented.append((i, rev, xo, steps_of(xo)))
        if not oriented:
            return list(P), 0
        owner = np.concatenate([[k] * len(o[2]) for k, o in enumerate(oriented)])
        tree_all = cKDTree(np.vstack([o[2] for o in oriented]))
        found = {}
        for k1, (i, r1, _x1, L1) in enumerate(oriented):
            for s1, p1, A1, B1, cA1, uA1, d1 in L1:
                for hit in tree_all.query_ball_point(p1, _SIG_END_NEAR_UM):
                    k2 = int(owner[hit])
                    j, r2, _x2, L2 = oriented[k2]
                    if j <= i:
                        continue
                    for s2, p2, A2, B2, cA2, uA2, d2 in L2:
                        if np.linalg.norm(p2 - p1) > _SIG_END_NEAR_UM or uA1 @ uA2 <= 0 or d1 @ d2 >= 0:
                            continue
                        m1q, m1a = miss(cA2, uA2, B1), miss(cA1, uA1, B1)
                        m2q, m2a = miss(cA1, uA1, B2), miss(cA2, uA2, B2)
                        if None in (m1q, m1a, m2q, m2a):
                            continue
                        if (m1q > _SIG_MAX_MISS_UM or m2q > _SIG_MAX_MISS_UM
                                or m1q > _SWAP_FIT_RATIO * m1a or m2q > _SWAP_FIT_RATIO * m2a):
                            continue
                        score = (m1a - m1q) + (m2a - m2q)
                        if (i, j) not in found or score > found[(i, j)][0]:
                            found[(i, j)] = (score, r1, s1, r2, s2)
        touched, out, n_done = set(), list(P), 0
        for (i, j), (_sc, r1, s1, r2, s2) in sorted(found.items(), key=lambda t: -t[1][0]):
            if i in touched or j in touched:
                continue
            parts = []
            for idx, rev, s_ in ((i, r1, s1), (j, r2, s2)):
                Pi = P[idx][::-1] if rev else P[idx]
                a = arc(Pi * sp)
                parts.append((Pi[:int(np.searchsorted(a, s_ - D))], Pi[int(np.searchsorted(a, s_ + D)):]))
            (head_i, tail_i), (head_j, tail_j) = parts
            if not (len(head_i) and len(tail_i) and len(head_j) and len(tail_j)):
                continue
            new = []
            for head, tail in ((head_i, tail_j), (head_j, tail_i)):
                link = _line_voxels(head[-1], tail[0])
                link = link[obj_lookup(link)] if len(link) else link
                new.append(np.concatenate([head, link, tail]))
            out[i], out[j] = new
            touched.update((i, j))
            n_done += 1
        return [o for o in out if len(o)], n_done

    def attach_orphans(P):
        """Give each short free-standing piece to the incoming fibre that keeps the depth order."""
        X = [p_ * sp for p_ in P]
        L = [float(arc(x)[-1]) if len(x) > 1 else 0.0 for x in X]
        ends = [(i, w, x[0] if w == 0 else x[-1]) for i, x in enumerate(X) if len(x) >= 6 for w in (0, 1)]
        if not ends:
            return list(P), 0
        etree = cKDTree(np.array([e[2] for e in ends]))
        allp = np.vstack(X)
        owner = np.concatenate([[k] * len(x) for k, x in enumerate(X)])
        first = np.concatenate([[0], np.cumsum([len(x) for x in X])[:-1]])
        ptree = cKDTree(allp)
        near = _ORPHAN_NEAR_UM
        found = {}
        for r, xr_all in enumerate(X):
            if len(xr_all) < 6 or not (_SIG_MIN_FIBRE_UM <= L[r] <= _ORPHAN_UM):
                continue
            for wr in (0, 1):
                xr = xr_all if wr == 0 else xr_all[::-1]                # it starts at the contact
                if any(ends[k][0] != r for k in etree.query_ball_point(xr[-1], near)):
                    continue                                              # its far end must be free
                ar = arc(xr)
                Rsel = xr[ar <= W]
                cR, uR = fit(Rsel)
                uR = towards(uR, Rsel[0], Rsel[-1])
                for pk in ptree.query_ball_point(xr[0], near):
                    i = int(owner[pk])
                    if i == r or L[i] < 30.0:
                        continue
                    xi = X[i]
                    ai = arc(xi)
                    s_ = float(ai[pk - first[i]])
                    if s_ < D + 3.0 or s_ > ai[-1] - D - 3.0:
                        continue
                    A = xi[(ai >= s_ - W) & (ai <= s_ - D)]
                    B = xi[(ai >= s_ + D) & (ai <= s_ + W)]
                    if len(A) < 8 or len(B) < 8:
                        continue
                    cA, uA = fit(A)
                    cB, uB = fit(B)
                    uA = towards(uA, A.mean(0), B.mean(0))
                    uB = towards(uB, A.mean(0), B.mean(0))
                    if uR @ uB < 0.5:
                        continue                                          # it must leave the way the path goes on
                    p = xi[pk - first[i]]
                    for ek in etree.query_ball_point(p, near):
                        q, wq, _pe = ends[ek]
                        if q in (i, r) or L[q] < _SIG_MIN_FIBRE_UM:
                            continue
                        xq = X[q] if wq == 1 else X[q][::-1]              # it ends at the contact
                        aq = arc(xq)
                        Qsel = xq[aq >= aq[-1] - W]
                        if len(Qsel) < 6:
                            continue
                        cQ, uQ = fit(Qsel)
                        uQ = towards(uQ, Qsel[0], Qsel[-1])
                        if uQ @ uA < 0.5:
                            continue                                      # it must arrive the way the path comes in
                        m = (uA + uB) / np.linalg.norm(uA + uB)

                        def depth_at(c, u):
                            return float((c + u * (((p - c) @ m) / (u @ m)))[0])

                        def plane_at(c, u):
                            return c + u * (((p - c) @ m) / (u @ m))

                        def apart(c1, u1, c2, u2):
                            d_ = plane_at(c1, u1) - plane_at(c2, u2)
                            return float(np.linalg.norm(d_ - (d_ @ m) * m))

                        if apart(cA, uA, cR, uR) > _ORPHAN_REACH_UM or apart(cQ, uQ, cB, uB) > _ORPHAN_REACH_UM:
                            continue                                      # too far apart to be one fibre
                        dz_in = depth_at(cA, uA) - depth_at(cQ, uQ)
                        dz_out = depth_at(cR, uR) - depth_at(cB, uB)
                        if abs(dz_in) < _ORPHAN_DEPTH_UM or abs(dz_out) < _ORPHAN_DEPTH_UM or dz_in * dz_out <= 0:
                            continue                                      # no clear order, or the order is kept as it is
                        turns = [np.degrees(np.arccos(np.clip(u1 @ u2, -1, 1))) for u1, u2 in ((uA, uR), (uQ, uB))]
                        if max(turns) > _SIG_MAX_DEG * 2.25:
                            continue
                        key = (r, i, q)
                        score = min(abs(dz_in), abs(dz_out))
                        if key not in found or score > found[key][0]:
                            found[key] = (score, wr, wq, s_)
        touched, out, n_done = set(), list(P), 0
        for (r, i, q), (_sc, wr, wq, s_) in sorted(found.items(), key=lambda kv: -kv[1][0]):
            if {r, i, q} & touched:
                continue
            a = arc(X[i])
            ia, ib = int(np.searchsorted(a, s_ - D)), int(np.searchsorted(a, s_ + D))
            head, tail = P[i][:ia], P[i][ib:]
            piece = P[r] if wr == 0 else P[r][::-1]
            incoming = P[q] if wq == 1 else P[q][::-1]
            if not (len(head) and len(tail)):
                continue
            new = []
            for first_part, second_part in ((head, piece), (incoming, tail)):
                link = _line_voxels(first_part[-1], second_part[0])
                link = link[obj_lookup(link)] if len(link) else link
                new.append(np.concatenate([first_part, link, second_part]))
            out[i], out[q], out[r] = new[0], new[1], P[r][:0]
            touched.update((r, i, q))
            n_done += 1
        return [o for o in out if len(o)], n_done

    def section_stats(xyz, stride=4):
        """(index, area um^2, top z, bottom z) of the raw mask's cross-section at points along a course."""
        from scipy import ndimage
        da_, db_ = 0.15, 0.18
        ga = np.arange(-3.0, 3.0 + 1e-9, da_)
        gb = np.arange(-5.0, 5.0 + 1e-9, db_)
        GA, GB = np.meshgrid(ga, gb, indexing="ij")
        out_ = []
        for k in range(2, len(xyz) - 2, stride):
            t = xyz[min(k + 5, len(xyz) - 1)] - xyz[max(k - 5, 0)]
            nt = np.linalg.norm(t)
            if nt < 1e-6:
                continue
            t = t / nt
            n1 = np.array([0.0, -t[2], t[1]])
            n1 /= np.linalg.norm(n1) + 1e-12
            n2 = np.cross(t, n1)
            n2 /= np.linalg.norm(n2) + 1e-12
            vox = np.round((xyz[k][None, None, :] + GA[..., None] * n1 + GB[..., None] * n2).reshape(-1, 3) / sp).astype(np.int64)
            ok = np.all((vox >= 0) & (vox < np.asarray(bshape)), axis=1)
            flat = np.zeros(len(vox), bool)
            if ok.any():
                flat[ok] = obj_lookup(vox[ok])
            lab, _n = ndimage.label(flat.reshape(GA.shape))
            c = (len(ga) // 2, len(gb) // 2)
            if lab[c] == 0:
                continue
            comp = (lab == lab[c]).reshape(-1)
            zz = vox[comp, 0]
            out_.append((k, float(comp.sum()) * da_ * db_, float(zz.min()), float(zz.max())))
        return out_

    def depth_range(stats):
        """Depth range (top, bottom) of a course from its thinnest cross-sections, where only the one fibre is in the section."""
        if len(stats) < 3:
            return None
        rows = np.array(sorted((r_[1], r_[2], r_[3]) for r_ in stats))
        thin = rows[:max(3, int(0.4 * len(rows)))]
        return float(np.median(thin[:, 1])), float(np.median(thin[:, 2]))

    def blob_handover(P):
        """Where a fibre ends against a passing path and a double blob follows, the single fibre
        that comes out of the blob goes to the fibre whose depth range it matches.

        The path carries on through the blob (it coexists there with the fibre that ends against
        it) and is cut where the blob ends; the fibre that ends takes the rest. Both depth ranges
        are those of the fibres' own courses 12 to 2 um before the contact; they are
        compared only when both courses are within _BLOB_LEVEL_DEG of level there, since a
        course that climbs or dips is at another depth at the contact. Otherwise the courses
        decide."""
        X = [p_ * sp for p_ in P]
        L = [float(arc(x)[-1]) if len(x) > 1 else 0.0 for x in X]
        ends = [(i, w, x[0] if w == 0 else x[-1]) for i, x in enumerate(X) if len(x) >= 6 for w in (0, 1)]
        if not ends:
            return list(P), 0
        allp = np.vstack(X)
        owner = np.concatenate([[k] * len(x) for k, x in enumerate(X)])
        first = np.concatenate([[0], np.cumsum([len(x) for x in X])[:-1]])
        ptree = cKDTree(allp)
        sin_level = float(np.sin(np.radians(_BLOB_LEVEL_DEG)))
        found = {}
        for q, wq, pe in ends:
            if L[q] < _SIG_MIN_FIBRE_UM:
                continue
            xq = X[q] if wq == 1 else X[q][::-1]                      # it ends at the contact
            aq = arc(xq)
            Qsel = xq[aq >= aq[-1] - W]
            Qc = xq[(aq >= aq[-1] - 12.0) & (aq <= aq[-1] - 2.0)]
            if len(Qsel) < 6 or len(Qc) < 8:
                continue
            uQ = towards(fit(Qsel)[1], Qsel[0], Qsel[-1])
            nearest = {}                                              # per passing path, the point nearest the tip
            for pk in ptree.query_ball_point(pe, _BLOB_NEAR_UM):
                o_ = int(owner[pk])
                dd_ = float(np.linalg.norm(allp[pk] - pe))
                if o_ != q and L[o_] >= 30.0 and (o_ not in nearest or dd_ < nearest[o_][0]):
                    nearest[o_] = (dd_, pk)
            for pk in [v_[1] for v_ in nearest.values()]:
                i = int(owner[pk])
                xi = X[i]
                ai = arc(xi)
                s0 = float(ai[pk - first[i]])
                loc = xi[(ai >= s0 - 5.0) & (ai <= s0 + 5.0)]
                if len(loc) < 6:
                    continue
                ui = towards(fit(loc)[1], loc[0], loc[-1])
                d_ = float(ui @ uQ)
                if abs(d_) < 0.5:
                    continue
                fwd = d_ > 0                                          # the way the fibre ending here is heading
                xo = xi if fwd else xi[::-1]
                ao = arc(xo)
                s0o = s0 if fwd else float(ai[-1] - s0)
                if ao[-1] - s0o < 12.0:
                    continue
                det = None
                for scan_, long_ in ((16.0, False), (_BLOB_LONG_SCAN_UM, True)):
                    down = xo[(ao >= s0o) & (ao <= s0o + scan_)]
                    if len(down) < 20:
                        break
                    st = section_stats(down)
                    if len(st) < 7:
                        break
                    s_k = arc(down)[np.array([r_[0] for r_ in st])]
                    ar = np.array([r_[1] for r_ in st])
                    thin_a = float(np.percentile(ar, 25))
                    if long_:
                        # A blob can outlast the 16 um scan, which then holds no lone fibre to compare
                        # with: scan further, take the lone section from the far end of the scan (where
                        # the single fibre has come out), and let one thin section not break the blob.
                        far_ = ar[s_k >= s_k[-1] * 2.0 / 3.0]
                        if len(far_) >= 3:
                            thin_a = min(thin_a, float(np.percentile(far_, 25)))
                    big = ar >= _BLOB_RATIO * thin_a
                    if long_ and len(big) > 2:
                        big[1:-1] |= big[:-2] & big[2:]
                    near = np.nonzero(big & (s_k <= 6.0))[0]
                    if not len(near):
                        continue
                    run_end = int(near[0])
                    while run_end + 1 < len(big) and big[run_end + 1]:
                        run_end += 1
                    if s_k[run_end] - s_k[near[0]] < _BLOB_MIN_UM:
                        continue
                    beyond = [b for b in range(run_end + 1, len(big)) if not big[b]]
                    if len(beyond) < 3 or s_k[beyond[0]] - s_k[run_end] > 2.0:
                        continue                                      # no single fibre comes out of it
                    det = (down, st, s_k, run_end, beyond, thin_a, long_)
                    break
                if det is None:
                    continue
                down, st, s_k, run_end, beyond, thin_a, long_det = det
                # Is the fibre that ends here already inside the blob before it ends (the passing
                # path's section double over the last _BLOB_INSIDE_UM before the contact)? Then the
                # two run through it as a pair. If the ending fibre lies at least _BLOB_SIDE_UM from
                # the passing fibre's own depth (where that runs alone, upstream), and the fibre that
                # comes out lies as far from it, at the ending fibre's depth (_BLOB_SIDE_MATCH_UM),
                # the ending fibre takes it: the passing path had hopped layers inside the blob.
                side_ok, side_layer = False, None
                up_ = xo[(ao >= s0o - _BLOB_LONG_SCAN_UM) & (ao <= s0o)]
                st_up = section_stats(up_) if long_det and len(up_) >= 20 else []
                if len(st_up) >= 7:
                    s_up = arc(up_)[np.array([r_[0] for r_ in st_up])]
                    a_up = np.array([r_[1] for r_ in st_up])
                    last_ = s_up >= s_up[-1] - _BLOB_INSIDE_UM
                    lone_up = [r_ for r_ in st_up if r_[1] < _BLOB_RATIO * thin_a]
                    if last_.sum() >= 2 and np.mean(a_up[last_] >= _BLOB_RATIO * thin_a) >= 0.6 and len(lone_up) >= 3:
                        dI = depth_range(lone_up)
                        dX = depth_range([st[b_] for b_ in beyond[:8]])
                        zq_ = float(np.mean(xq[aq >= aq[-1] - 3.0][:, 0])) / sp[0]
                        if dI is not None and dX is not None:
                            zi_, zx_ = 0.5 * (dI[0] + dI[1]), 0.5 * (dX[0] + dX[1])
                            sg_ = np.sign(zq_ - zi_)
                            side_ok = (abs(zq_ - zi_) * sp[0] >= _BLOB_SIDE_UM
                                       and (zx_ - zi_) * sg_ * sp[0] >= _BLOB_SIDE_UM
                                       and abs(zx_ - zq_) * sp[0] <= _BLOB_SIDE_MATCH_UM)
                            side_gain = 100.0 + abs(zx_ - zi_)
                            side_layer = (s0o, float(sg_))
                depth_ok, gain = False, 0.0
                Ac = xo[(ao >= s0o - 12.0) & (ao <= s0o - 2.0)]
                if len(Ac) >= 8 and all(abs(fit(c_)[1][0]) <= sin_level for c_ in (Ac, Qc)):
                    dB = depth_range([st[b_] for b_ in beyond[:8]])
                    dQ = depth_range(section_stats(Qc))
                    dA = depth_range(section_stats(Ac))
                    if None not in (dB, dQ, dA):
                        edgeQ = abs(dB[0] - dQ[0]) + abs(dB[1] - dQ[1])
                        edgeA = abs(dB[0] - dA[0]) + abs(dB[1] - dA[1])
                        depth_ok = edgeQ <= _BLOB_EDGE_MAX and edgeQ <= _BLOB_EDGE_RATIO * edgeA
                        gain = edgeA - edgeQ
                course_ok = False
                if not depth_ok:
                    ad_ = arc(down)
                    Bc = down[ad_ >= float(s_k[beyond[0]])]
                    Bc = Bc[arc(Bc) <= 8.0] if len(Bc) > 1 else Bc
                    ratios = []
                    for w0_, w1_ in _BLOB_COURSES:
                        if aq[-1] < w1_ or s0o < w1_:
                            continue                                      # a window the path cannot supply in full is no evidence
                        Qw = xq[(aq >= aq[-1] - w1_) & (aq <= aq[-1] - w0_)]
                        Aw = xo[(ao >= s0o - w1_) & (ao <= s0o - w0_)]
                        if len(Qw) < 8 or len(Aw) < 8 or len(Bc) < 8:
                            continue
                        cQw, uQw = fit(Qw)
                        uQw = towards(uQw, Qw[0], Qw[-1])
                        cAw, uAw = fit(Aw)
                        uAw = towards(uAw, Aw[0], Aw[-1])
                        mq_ = miss(cQw, uQw, Bc[(Bc - xq[-1]) @ uQw > 0])
                        ma_ = miss(cAw, uAw, Bc[(Bc - xq[-1]) @ uAw > 0])
                        if mq_ is None or ma_ is None:
                            continue
                        ratios.append(mq_ / max(ma_, 1e-6))
                    course_ok = (len(ratios) >= 2 and max(ratios) <= _BLOB_COURSE_ALL
                                 and float(np.median(ratios)) <= _BLOB_COURSE_MED)
                    if course_ok:
                        gain = 10.0 * (1.0 - float(np.median(ratios)))
                if long_det:
                    depth_ok = course_ok = False                      # past 16 um only a pair in clearly separate layers is handed over
                if side_ok:
                    gain = side_gain                                  # a pair already in the blob comes first
                if not (depth_ok or course_ok or side_ok):
                    continue
                if (q, i) not in found or gain > found[(q, i)][0]:
                    found[(q, i)] = (gain, wq, fwd, s0o + float(s_k[run_end]), side_layer if side_ok else None)

        def to_own_layer(seg_, side):
            """Move a stretch of path to the passing fibre's own layer of a two-layer blob: the
            mask's edge in depth on that layer's side (`side` = -1 shallower, +1 deeper), kept
            _BLOB_LAYER_INSET_UM inside it."""
            off = max(1, int(round(_BLOB_LAYER_INSET_UM / sp[0])))
            moved = []
            for v in seg_:
                z_ = int(v[0])
                while 0 <= z_ + side < bshape[0] and obj_lookup(np.array([[z_ + side, v[1], v[2]]]))[0]:
                    z_ += side
                z_ -= side * off
                if (z_ - int(v[0])) * side < 0:               # the mask is no thicker than that here
                    z_ = int(v[0])
                moved.append([z_, v[1], v[2]])
            return np.array(moved, dtype=seg_.dtype)

        touched, out, n_done = set(), list(P), 0
        for (q, i), (_g, wq, fwd, s_b, layer) in sorted(found.items(), key=lambda kv: -kv[1][0]):
            if q in touched or i in touched:
                continue
            Pi = P[i] if fwd else P[i][::-1]
            a = arc(Pi * sp)
            head, tail = Pi[:int(np.searchsorted(a, s_b))], Pi[int(np.searchsorted(a, s_b + D)):]
            if len(head) < 6 or len(tail) < 6:
                continue
            incoming = P[q] if wq == 1 else P[q][::-1]
            link = _line_voxels(incoming[-1], tail[0])
            link = link[obj_lookup(link)] if len(link) else link
            out[q] = np.concatenate([incoming, link, tail])
            if layer is not None:
                # The passing path hopped layers inside the blob; keep its rest of the blob on its own side.
                k0 = int(np.searchsorted(a, layer[0]))
                if k0 < len(head):
                    moved = to_own_layer(head[k0:], -int(layer[1]))
                    seq, prev = [head[:k0]] if k0 else [], (head[k0 - 1] if k0 else None)
                    for v in moved:
                        if prev is not None:
                            lk = _line_voxels(prev, v)
                            lk = lk[obj_lookup(lk)] if len(lk) else lk
                            if len(lk):
                                seq.append(lk)
                        else:
                            seq.append(v[None])
                        prev = v
                    head = np.concatenate(seq)
                    head = head[np.r_[True, np.any(np.diff(head, axis=0) != 0, axis=1)]]
            out[i] = head
            touched.update((q, i))
            n_done += 1
        return [o for o in out if len(o)], n_done

    def bridge_blobs(P):
        """Join a fibre that goes into a double blob and one that comes out of it, on the same line.

        "On the same line" allows for the passing path's own climb or dip between the two contacts:
        a fibre that runs through the blob alongside it changes depth with it."""
        from skimage.graph import MCP_Geometric
        X = [p_ * sp for p_ in P]
        L = [float(arc(x)[-1]) if len(x) > 1 else 0.0 for x in X]
        ends = [(i, w, x[0] if w == 0 else x[-1]) for i, x in enumerate(X) if len(x) >= 6 and L[i] >= _SIG_MIN_FIBRE_UM for w in (0, 1)]
        if len(ends) < 2:
            return list(P), 0
        allp = np.vstack(X)
        owner = np.concatenate([[k] * len(x) for k, x in enumerate(X)])
        first = np.concatenate([[0], np.cumsum([len(x) for x in X])[:-1]])
        ptree = cKDTree(allp)
        by_path = {}
        for a, wa, pe in ends:
            nearest = {}
            for pk in ptree.query_ball_point(pe, _BRIDGE_NEAR_UM):
                o_ = int(owner[pk])
                dd_ = float(np.linalg.norm(allp[pk] - pe))
                if o_ != a and L[o_] >= 30.0 and (o_ not in nearest or dd_ < nearest[o_][0]):
                    nearest[o_] = (dd_, pk - first[o_])
            for o_, (dd_, k_) in nearest.items():
                by_path.setdefault(o_, []).append((a, wa, pe, k_))
        found = []
        for i, lst in by_path.items():
            ai = arc(X[i])
            for u_ in range(len(lst)):
                for v_ in range(u_ + 1, len(lst)):
                    a, wa, ea, ka = lst[u_]
                    b, wb, eb, kb = lst[v_]
                    if a == b:
                        continue
                    along = abs(float(ai[ka] - ai[kb]))
                    gap = float(np.linalg.norm(ea - eb))
                    if not (_BRIDGE_ALONG_UM[0] <= along <= _BRIDGE_ALONG_UM[1]) or gap > _BRIDGE_ALONG_UM[1]:
                        continue
                    A_ = _end_course(P[a], wa, sp)
                    B_ = _end_course(P[b], wb, sp)
                    if A_ is None or B_ is None:
                        continue
                    ea_, ca, ua = A_
                    eb_, cb, ub = B_
                    g = eb_ - ea_
                    turn = float(np.degrees(np.arccos(np.clip(-ua @ ub, -1, 1))))
                    if turn > _BRIDGE_TURN_DEG or float(g @ ua) < 0 or float(-g @ ub) < 0:
                        continue
                    n_ = ua - ub
                    n_ = n_ / np.linalg.norm(n_)
                    mid = 0.5 * (ea_ + eb_)
                    dv_ = (ca + ua * (((mid - ca) @ n_) / (ua @ n_))) - (cb + ub * (((mid - cb) @ n_) / (ub @ n_)))
                    # Two fibres that run through a blob together climb or dip together: the depth the
                    # passing path changes by between the two contacts is no offset between the pieces.
                    dz_i = float(X[i][ka][0] - X[i][kb][0])
                    shift = min(float(np.linalg.norm(dv_)), float(np.linalg.norm(dv_ - np.array([dz_i, 0.0, 0.0]))))
                    if shift > _BRIDGE_SHIFT_UM:
                        continue
                    lo_, hi_ = sorted((ka, kb))
                    st = section_stats(X[i][lo_:hi_ + 1], stride=3)
                    ref = []
                    for xx_, w_ in ((X[a], wa), (X[b], wb)):
                        xo_ = xx_[::-1] if w_ == 0 else xx_
                        ref += [r_[1] for r_ in section_stats(xo_[arc(xo_) >= arc(xo_)[-1] - 10.0], stride=3)]
                    if len(st) < 3 or len(ref) < 4:
                        continue
                    lone = float(np.percentile(ref, 40))
                    share = float(np.mean([r_[1] >= _BLOB_RATIO * lone for r_ in st]))
                    if share < _BRIDGE_BLOB_SHARE:
                        continue
                    found.append((shift + 0.3 * gap, a, wa, ea, b, wb, eb, i, along, gap, turn, shift, share))
        found.sort(key=lambda t: t[0])
        used, out, n_done = set(), list(P), 0
        for cost, a, wa, ea, b, wb, eb, i, along, gap, turn, shift, share in found:
            if a in used or b in used:
                continue
            va = np.round(ea / sp).astype(np.int64)
            vb = np.round(eb / sp).astype(np.int64)
            lo = np.maximum(np.minimum(va, vb) - np.array([6, 25, 25]), 0)
            hi = np.minimum(np.maximum(va, vb) + np.array([7, 26, 26]), np.asarray(bshape))
            gz, gy, gx = np.meshgrid(*[np.arange(lo[k_], hi[k_]) for k_ in range(3)], indexing="ij")
            box = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1)
            inside = obj_lookup(box).reshape(gz.shape)
            costs = np.where(inside, 1.0, 12.0)                      # the mask has gaps: crossing one is allowed, at a price
            near_i = cKDTree(X[i]).query(box * sp)[0].reshape(gz.shape) < 0.45
            costs = np.where(inside & near_i, 4.0, costs)            # keep off the passing path's own line
            s0, e0 = tuple(va - lo), tuple(vb - lo)
            m_ = MCP_Geometric(costs, fully_connected=True, sampling=tuple(sp))
            m_.find_costs([s0], [e0])
            try:
                route = np.array(m_.traceback(e0)) + lo
            except ValueError:
                continue
            rlen = float(np.sum(np.linalg.norm(np.diff(route * sp, axis=0), axis=1)))
            if rlen > 1.6 * gap:
                continue
            route = route[obj_lookup(route)]                       # only what is in the mask, like the other joins
            pa = P[a] if wa == 1 else P[a][::-1]                   # ends at the blob
            pb = P[b] if wb == 0 else P[b][::-1]                   # starts at the blob
            out[a] = np.concatenate([pa, route[1:-1], pb])
            out[b] = P[b][:0]
            used.update((a, b))
            n_done += 1
        return [o for o in out if len(o)], n_done

    P = [np.asarray(p_) for p_ in paths]
    for _pass in range(_SIG_PASSES):
        P, n_fixed = one_pass(P)
        if not n_fixed:
            break
    P, _ = swap_crossings(P)        # once: swapping the same two paths again would only undo it
    P, _ = attach_orphans(P)
    P, _ = blob_handover(P)
    P, _ = bridge_blobs(P)
    return P


def _rejoin_at_bends(paths, sp, r, obj_lookup):
    """Re-pair a bend with the path that carries straight on through it.

    At a contact the tracer takes paths one at a time, longest first, and each continues onto the
    best branch left. A long path can thereby be bent onto a branch (up to _MAX_TURN_DEG) while the
    branch that carries straight on from where it came is left to a path of its own, which then starts
    at the contact. That shows in the finished paths as a bend of more than _BEND_MIN_DEG with a third
    path touching it near its end and lined up with one of the two legs (within _BEND_LINE_DEG, and
    within _BEND_LINE_RATIO of the bend) far better than the legs are with each other. It must also be
    the same fibre as that leg: on its line to within _JOIN_SHIFT_R r, and in its depth layer
    (_LAYER_SAME_R r), as a lane has to be to be followed (see `extend`). The touching path is then
    joined to that leg, and the other leg is left as a path of its own, ending at the contact. Whatever
    the touching path has beyond the contact (a hook, within _BEND_HOOK_UM) is dropped.

    Every path takes part in at most one repair, and nothing is repeated. Paths with no such bend are
    returned untouched.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    if not P:
        return P
    X = [p_ * sp for p_ in P]
    A = [_arc_points(x, None) for x in X]
    L = [float(a[-1]) for a in A]

    def line(i, s0, s1):
        """(centroid, direction along the path) of path i between arc positions s0 and s1, or None."""
        a = A[i]
        sel = (a >= s0) & (a <= s1)
        if sel.sum() < 4 or a[sel].max() - a[sel].min() < _JOIN_MIN_UM:
            return None
        q = X[i][sel]
        c = q.mean(0)
        u = np.linalg.svd(q - c, full_matrices=False)[2][0]
        return c, (u if u @ (q[-1] - q[0]) >= 0 else -u)

    def angle(u, v):
        return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))

    def offset(l1, l2, pt):
        """Sideways distance (um) between two nearly parallel lines at `pt`, and its depth part."""
        (c1, u1), (c2, u2) = l1, l2
        m = u1 + (u2 if u1 @ u2 >= 0 else -u2)
        m = m / np.linalg.norm(m)
        d = (c1 + (((pt - c1) @ m) / (u1 @ m)) * u1) - (c2 + (((pt - c2) @ m) / (u2 @ m)) * u2)
        return float(np.linalg.norm(d)), abs(float(d[0]))

    # ---- bends: local maxima of the turn between the courses either side of a place ----------
    sites = []
    near_n = int(round(2.0 * _JOIN_SKIP_UM / _SIG_STEP_UM))
    for i in range(len(P)):
        if L[i] < 2.0 * _JOIN_WINDOW_UM:
            continue
        steps = np.arange(_JOIN_WINDOW_UM, L[i] - _JOIN_WINDOW_UM, _SIG_STEP_UM)
        turn = np.zeros(len(steps))
        legs = [None] * len(steps)
        for n, s in enumerate(steps):
            l_in = line(i, s - _JOIN_WINDOW_UM, s - _JOIN_SKIP_UM)
            l_out = line(i, s + _JOIN_SKIP_UM, s + _JOIN_WINDOW_UM)
            if l_in is not None and l_out is not None:
                turn[n] = angle(l_in[1], l_out[1])
                legs[n] = (l_in, l_out)
        for n in np.flatnonzero(turn >= _BEND_MIN_DEG):
            lo, hi = max(0, n - near_n), min(len(steps), n + near_n + 1)
            if n == lo + int(np.argmax(turn[lo:hi])):
                sites.append((i, float(steps[n]), float(turn[n]), legs[n][0], legs[n][1]))
    if not sites:
        return P

    # ---- a path that touches a bend near its end and carries on along one of its legs ---------
    allx = np.vstack(X)
    owner = np.concatenate([np.full(len(x), k) for k, x in enumerate(X)])
    index = np.concatenate([np.arange(len(x)) for x in X])
    tree = cKDTree(allx)
    found = []
    for i, s, bend, l_in, l_out in sites:
        kx = min(int(np.searchsorted(A[i], s)), len(X[i]) - 1)
        pt = X[i][kx]
        touch = {}
        for m in tree.query_ball_point(pt, _JOIN_SKIP_UM):
            j = int(owner[m])
            if j == i or L[j] < _SIG_MIN_FIBRE_UM:
                continue
            d = float(np.linalg.norm(allx[m] - pt))
            if j not in touch or d < touch[j][0]:
                touch[j] = (d, int(index[m]))
        for j, (_d, kk) in touch.items():
            at_start = A[j][kk] <= _BEND_HOOK_UM
            at_end = L[j] - A[j][kk] <= _BEND_HOOK_UM
            if at_start == at_end:              # it passes the bend, or is too short to tell
                continue
            if at_start:
                l_q = line(j, A[j][kk] + _JOIN_SKIP_UM, A[j][kk] + _JOIN_WINDOW_UM)
            else:
                l_q = line(j, A[j][kk] - _JOIN_WINDOW_UM, A[j][kk] - _JOIN_SKIP_UM)
                l_q = (l_q[0], -l_q[1]) if l_q is not None else None
            if l_q is None:                     # l_q: its line, directed from the bend into the path
                continue
            off_before, off_after = angle(l_q[1], l_in[1]), angle(l_q[1], -l_out[1])
            leg, off, l_leg = (("before", off_before, l_in) if off_before < off_after
                               else ("after", off_after, l_out))
            if off > min(_BEND_LINE_DEG, _BEND_LINE_RATIO * bend):
                continue
            side, depth = offset(l_leg, l_q, pt)
            if side <= _JOIN_SHIFT_R * r and depth <= _LAYER_SAME_R * r:
                found.append((off / bend, i, kx, j, kk, at_start, leg))

    out, touched = list(P), set()
    for _rank, i, kx, j, kk, at_start, leg in sorted(found, key=lambda f: f[:5]):
        if i in touched or j in touched:
            continue
        body = P[j][kk:][::-1] if at_start else P[j][:kk + 1]      # the touching path, ending at the bend
        if leg == "after":
            carried, rest = P[i][kx:], P[i][:kx + 1]
        else:
            carried, rest = P[i][:kx + 1][::-1], P[i][kx:]
        link = _line_voxels(body[-1], carried[0])
        link = link[obj_lookup(link)] if len(link) else link
        joined = np.concatenate([body, link, carried])
        out[j] = joined[np.r_[True, np.any(np.diff(joined, axis=0) != 0, axis=1)]]
        out[i] = rest
        touched.update((i, j))
    return [o for o in out if len(o)]


def _end_depth_reversals(paths, sp, r, obj_lookup, bshape):
    """End a dive at the dead end it goes into, instead of turning it back up.

    A fibre that dips (or climbs) steeply into a contact keeps going that way. Where the tracer
    instead turns the path back in depth (in at least _DIVE_IN_DEG, out at least _DIVE_OUT_DEG
    the other way, over _DIVE_LEG_UM either side), and the skeleton beyond the turn holds a
    piece too short to be traced as a fibre of its own (< _SIG_MIN_FIBRE_UM) that starts within
    _DIVE_NEAR_UM of the turn, lies where the incoming course points and runs along it (both
    within _DIVE_AIM_DEG) on its far side in depth, reaches at least _DIVE_MIN_UM along it and ends free (no other fibre
    within _JOIN_SKIP_UM of its tip), the incoming leg is continued into that piece and ends
    there. The outgoing leg is left as a path of its own, starting at the turn. Only paths of at
    least _DIVE_PATH_UM are repaired, and not where the piece runs to within _DIVE_EDGE_SLICES of
    the top or bottom of the box, where the fibre may leave the volume rather than end. The
    released legs are then mended (`_mend_breaks`) to the pieces they continue, if any.

    Every path takes part in at most one such repair. Paths without one are returned untouched.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    if not P:
        return P
    X = [p_ * sp for p_ in P]
    L = [float(_arc_points(x, None)[-1]) if len(x) > 1 else 0.0 for x in X]
    frag = [i for i in range(len(P)) if len(P[i]) and L[i] < _SIG_MIN_FIBRE_UM]
    big = [i for i in range(len(P)) if L[i] >= _SIG_MIN_FIBRE_UM]
    if not frag or not big:
        return P
    fowner = np.concatenate([[i] * len(X[i]) for i in frag])
    ftree = cKDTree(np.vstack([X[i] for i in frag]))
    bowner = np.concatenate([[i] * len(X[i]) for i in big])
    btree = cKDTree(np.vstack([X[i] for i in big]))

    def fit(q):
        c = q.mean(0)
        return c, np.linalg.svd(q - c, full_matrices=False)[2][0]

    def towards(u, f, t):
        return u if u @ (t - f) >= 0 else -u

    s_in = np.sin(np.radians(_DIVE_IN_DEG))
    s_out = np.sin(np.radians(_DIVE_OUT_DEG))
    c_aim = np.cos(np.radians(_DIVE_AIM_DEG))
    gap, part = 1.0, 3.0
    found = []
    z_last = int(bshape[0]) - 1
    for i in big:
        if L[i] < _DIVE_PATH_UM:
            continue
        for rev in (False, True):
            x = X[i][::-1] if rev else X[i]
            a = _arc_points(x, None)
            best = None
            for s in np.arange(_DIVE_LEG_UM, a[-1] - _DIVE_LEG_UM, _SIG_STEP_UM):
                Aq = x[(a >= s - _DIVE_LEG_UM) & (a <= s - gap)]
                Bq = x[(a >= s + gap) & (a <= s + _DIVE_LEG_UM)]
                if len(Aq) < 6 or len(Bq) < 6:
                    continue
                uA = towards(fit(Aq)[1], Aq[0], Aq[-1])
                uB = towards(fit(Bq)[1], Bq[0], Bq[-1])
                if not (abs(uA[0]) >= s_in and abs(uB[0]) >= s_out and uA[0] * uB[0] < 0):
                    continue                                  # the path does not turn back in depth here
                p = x[min(int(np.searchsorted(a, s)), len(x) - 1)]
                for f in sorted(set(int(fowner[h]) for h in ftree.query_ball_point(p, _DIVE_NEAR_UM))):
                    U = X[f]
                    fw = (U - p) @ uA
                    U = U[(fw > 0) & (fw <= _DIVE_REACH_UM) & ((U[:, 0] - p[0]) * np.sign(uA[0]) > 0)]
                    if len(U) < 6:
                        continue
                    fU = (U - p) @ uA
                    d = U[fU <= fU.min() + part].mean(0) - p
                    nd = float(np.linalg.norm(d))
                    aim = float(d @ uA) / nd if nd > 0 else 1.0
                    if aim < c_aim or float(fU.max()) < _DIVE_MIN_UM:
                        continue
                    if len(X[f]) >= 3 and abs(float(fit(X[f])[1] @ uA)) < c_aim:
                        continue                              # a spur standing across the course, not its continuation
                    tip = U[int(np.argmax(fU))]
                    if set(int(bowner[h]) for h in btree.query_ball_point(tip, _JOIN_SKIP_UM)) - {i}:
                        continue                              # the piece leads on to another fibre
                    zt = float(X[f][:, 0].min() if uA[0] < 0 else X[f][:, 0].max()) / sp[0]
                    if zt <= _DIVE_EDGE_SLICES or zt >= z_last - _DIVE_EDGE_SLICES:
                        continue                              # it runs out of the volume: where it ends is unknown
                    score = float(fU.max()) * aim
                    if best is None or score > best[0]:
                        best = (score, s, f, rev, p, uA)
            if best is not None:
                found.append((best[0], i) + best[1:])
    touched, out, extra = set(), list(P), []
    for _sc, i, s, f, rev, p, uA in sorted(found, key=lambda t: -t[0]):
        if i in touched or f in touched:
            continue
        Pi = P[i][::-1] if rev else P[i]
        a = _arc_points(Pi * sp, None)
        head, tail = Pi[:int(np.searchsorted(a, s))], Pi[int(np.searchsorted(a, s + gap)):]
        if len(head) < 6 or len(tail) < 6:
            continue
        U = P[f]
        fw = (U * sp - p) @ uA
        keep = (fw > 0) & (fw <= _DIVE_REACH_UM) & ((U[:, 0] * sp[0] - p[0]) * np.sign(uA[0]) > 0)
        U, fw = U[keep], fw[keep]
        bins = np.floor(fw / sp.min()).astype(int)
        seq, prev = [head], head[-1]
        for b in np.unique(bins):                             # the piece's centre line, along the course
            v = np.round(U[bins == b].mean(0)).astype(int)
            lk = _line_voxels(prev, v)
            lk = lk[obj_lookup(lk)] if len(lk) else lk
            if len(lk):
                seq.append(lk)
            prev = v
        new = np.concatenate(seq)
        out[i] = new[np.r_[True, np.any(np.diff(new, axis=0) != 0, axis=1)]]
        out[f] = P[f][:0]
        extra.append(tail)
        touched.update((i, f))
    if not extra:
        return P
    kept = [o for o in out if len(o)]
    # The released legs end where the dive was: mend them like any other break (only they may join).
    return _mend_breaks(kept + extra, sp, r, obj_lookup, focus=range(len(kept), len(kept) + len(extra)), bshape=bshape)


def _keep_order_at_crossings(paths, sp, obj_lookup):
    """Swap the continuations of two parallel paths that cross where they touch.

    Fibres that run parallel through a bundle keep their order. Where two long paths (at least
    30 um) touch along no more than _ORDER_SPAN_UM, are parallel on both sides of the contact
    (within _ORDER_PAR_DEG, courses 2 to 12 um away from it) and swap sides there (at least
    _ORDER_SIDE_UM apart before and after, on opposite sides), the tracer has carried each onto
    the other. Their continuations are swapped if each then turns by no more than _ORDER_TURN_DEG.
    Every path takes part in at most one swap.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    A = [_arc_points(x, None) if len(x) > 1 else np.zeros(len(x)) for x in X]
    L = [float(a[-1]) if len(a) else 0.0 for a in A]
    big = [i for i in range(len(P)) if L[i] >= 30.0]
    if len(big) < 2:
        return P

    def fit(q):
        c = q.mean(0)
        return c, np.linalg.svd(q - c, full_matrices=False)[2][0]

    def towards(u, f, t):
        return u if u @ (t - f) >= 0 else -u

    def legs(x, a, s):
        Aq = x[(a >= s - _JOIN_WINDOW_UM) & (a <= s - _JOIN_SKIP_UM)]
        Bq = x[(a >= s + _JOIN_SKIP_UM) & (a <= s + _JOIN_WINDOW_UM)]
        if len(Aq) < 8 or len(Bq) < 8:
            return None
        cA, uA = fit(Aq)
        cB, uB = fit(Bq)
        return cA, towards(uA, Aq[0], Aq[-1]), cB, towards(uB, Bq[0], Bq[-1])

    def side(c1, u1, c2, u2):
        m = u1 + u2
        m = m / np.linalg.norm(m)
        d = c2 - c1
        return d - (d @ m) * m

    c_par = np.cos(np.radians(_ORDER_PAR_DEG))
    c_turn = np.cos(np.radians(_ORDER_TURN_DEG))
    boxes = {i: (X[i].min(0) - _ORDER_TOUCH_UM, X[i].max(0) + _ORDER_TOUCH_UM) for i in big}
    trees = {}
    found = {}
    for n_, i in enumerate(big):
        for j in big[n_ + 1:]:
            if np.any(boxes[i][1] < boxes[j][0]) or np.any(boxes[j][1] < boxes[i][0]):
                continue
            if i not in trees:
                trees[i] = cKDTree(X[i])
            d, k = trees[i].query(X[j], distance_upper_bound=_ORDER_TOUCH_UM)
            kj = np.nonzero(np.isfinite(d))[0]
            if not len(kj) or A[j][kj].max() - A[j][kj].min() > _ORDER_SPAN_UM:
                continue                                      # apart, or running together rather than crossing
            m = int(kj[np.argmin(d[kj])])
            si, sj = float(A[i][int(k[m])]), float(A[j][m])
            li, lj = legs(X[i], A[i], si), legs(X[j], A[j], sj)
            if li is None or lj is None:
                continue
            cAi, uAi, cBi, uBi = li
            cAj, uAj, cBj, uBj = lj
            revj = bool(uAi @ uAj < 0)
            if revj:                                          # j runs the other way
                cAj, uAj, cBj, uBj = cBj, -uBj, cAj, -uAj
            if uAi @ uAj < c_par or uBi @ uBj < c_par:
                continue                                      # not two parallel fibres
            db, da = side(cAi, uAi, cAj, uAj), side(cBi, uBi, cBj, uBj)
            if min(np.linalg.norm(db), np.linalg.norm(da)) < _ORDER_SIDE_UM or db @ da >= 0:
                continue                                      # they keep their order already
            if uAi @ uBj < c_turn or uAj @ uBi < c_turn:
                continue
            found[(i, j)] = (si, sj, revj, float(min(np.linalg.norm(db), np.linalg.norm(da))))
    touched, out = set(), list(P)
    half = 0.5 * _JOIN_SKIP_UM
    for (i, j), (si, sj, revj, _sc) in sorted(found.items(), key=lambda t: -t[1][3]):
        if i in touched or j in touched:
            continue
        Pj = P[j][::-1] if revj else P[j]
        sj_ = float(A[j][-1] - sj) if revj else sj
        aj = _arc_points(Pj * sp, None)
        ai = A[i]
        hi_, ti_ = P[i][:int(np.searchsorted(ai, si - half))], P[i][int(np.searchsorted(ai, si + half)):]
        hj_, tj_ = Pj[:int(np.searchsorted(aj, sj_ - half))], Pj[int(np.searchsorted(aj, sj_ + half)):]
        if min(len(hi_), len(ti_), len(hj_), len(tj_)) < 6:
            continue
        new = []
        for h, t in ((hi_, tj_), (hj_, ti_)):
            lk = _line_voxels(h[-1], t[0])
            lk = lk[obj_lookup(lk)] if len(lk) else lk
            new.append(np.concatenate([h, lk, t]))
        out[i], out[j] = new
        touched.update((i, j))
    return out


def _cut_climbs(paths, sp, obj_lookup, bshape):
    """Cut a path where it leaves its course to climb (or dip) through the layers, and let it,
    and the paths it cut off, go on along their own courses.

    In a bundle too thick to be split by the skeleton, the tracer can follow a skeleton branch
    that crosses the layers, carrying a fibre steeply up or down out of its course. A path of at
    least 30 um that, after a course within _CLIMB_LEVEL_DEG of level over its last 12 um,
    climbs or dips at least _CLIMB_DEG steeply through at least _CLIMB_DEPTH_UM of depth, to
    (near) its end, is cut where the climb starts and the climb is dropped. If instead the path
    goes on past the climb for at least 12 um, and runs within _HOP_LEVEL_DEG of level over
    12 um both before and after it, it hopped from one fibre onto another: the climb is dropped and the rest
    becomes a path of its own. That path, and every
    path of at least _SIG_MIN_FIBRE_UM that ends within _CLIMB_NEAR_UM of the dropped climb, is
    then marched on along its course through mask no other path claims (`_MARCH_*`).
    Paths without a climb near them are returned untouched.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    bsh = np.asarray(bshape)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    A = [_arc_points(x, None) if len(x) > 1 else np.zeros(len(x)) for x in X]
    L = [float(a[-1]) if len(a) else 0.0 for a in A]

    def fit_dir(q):
        c = q.mean(0)
        u = np.linalg.svd(q - c, full_matrices=False)[2][0]
        return u if u @ (q[-1] - q[0]) >= 0 else -u

    s_climb = np.sin(np.radians(_CLIMB_DEG))
    s_level = np.sin(np.radians(_CLIMB_LEVEL_DEG))
    s_hop = np.sin(np.radians(_HOP_LEVEL_DEG))
    win, course_um = 2.0, _JOIN_WINDOW_UM
    cuts = []
    for i in range(len(P)):
        if L[i] < 30.0:
            continue
        for rev in (False, True):
            x = X[i][::-1] if rev else X[i]
            a = _arc_points(x, None)
            n, k, found = len(x), 0, False
            while k < n and not found:
                k2 = int(np.searchsorted(a, a[k] + win))
                if k2 >= n:
                    break
                d = x[k2] - x[k]
                nd = float(np.linalg.norm(d))
                if nd > 0 and abs(d[0]) / nd >= s_climb and a[k] >= course_um:
                    uc = fit_dir(x[(a >= a[k] - course_um) & (a <= a[k])])
                    if abs(uc[0]) <= s_level:
                        j, sgn = k, np.sign(d[0])
                        while j < n - 1:
                            j2 = int(np.searchsorted(a, a[j] + win))
                            if j2 >= n:
                                j = n - 1
                                break
                            dd = x[j2] - x[j]
                            ndd = float(np.linalg.norm(dd))
                            j = j2
                            if ndd == 0 or np.sign(dd[0]) != sgn or abs(dd[0]) / ndd < s_climb:
                                break
                        if abs(x[j][0] - x[k][0]) >= _CLIMB_DEPTH_UM:
                            if n - 1 - j < 20:
                                cuts.append((i, rev, k, uc, None))   # the climb runs to the end
                                found = True
                            elif a[-1] - a[j] >= course_um and abs(uc[0]) <= s_hop:
                                # The path goes on, level again, in another layer: it hopped from
                                # one fibre onto another. Its rest is a fibre of its own.
                                after = x[(a >= a[j]) & (a <= a[j] + course_um)]
                                if len(after) >= 6 and abs(fit_dir(after)[0]) <= s_hop:
                                    cuts.append((i, rev, k, uc, j))
                                    found = True
                k += 1
            if found:
                break
    if not cuts:
        return P
    out, released, removed, hopped = list(P), [], [], []
    for i, rev, k, uc, j in cuts:
        Pi = P[i][::-1] if rev else P[i]
        if j is None:
            removed.append(Pi[k + 1:])
        else:
            removed.append(Pi[k + 1:j])
            hopped.append(Pi[j:])
        out[i] = Pi[:k + 1]
        released.append((i, uc, False))
    rtree = cKDTree(np.vstack(removed) * sp)
    cut_ids = {c[0] for c in cuts}
    for m in range(len(out)):
        if m in cut_ids or L[m] < _SIG_MIN_FIBRE_UM:
            continue
        for at_start in (False, True):
            x = out[m] * sp if not at_start else (out[m] * sp)[::-1]
            if rtree.query(x[-1])[0] <= _CLIMB_NEAR_UM:
                a = _arc_points(x, None)
                released.append((m, fit_dir(x[a >= a[-1] - course_um]), at_start))
    others = [k for k in range(len(out)) if L[k] >= _SIG_MIN_FIBRE_UM]
    g = np.arange(-_MARCH_DISC_UM, _MARCH_DISC_UM + 1e-9, float(sp.min()))
    grid = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=1) <= _MARCH_DISC_UM]
    cos_t, sin_t = np.cos(np.radians(_MARCH_TURN_DEG)), np.sin(np.radians(_MARCH_TURN_DEG))
    trails = []
    for i, u, at_start in released:
        Pi = out[i][::-1] if at_start else out[i]
        if len(Pi) < 2:
            continue
        tree = cKDTree(np.vstack([out[k] * sp for k in others if k != i] + trails))
        p, d = Pi[-1] * sp, u / np.linalg.norm(u)
        trail, total = [], 0.0
        while total < _MARCH_MAX_UM:
            c = p + d * _MARCH_STEP_UM
            n1 = np.array([0.0, -d[2], d[1]])
            n1 /= np.linalg.norm(n1) + 1e-12
            n2 = np.cross(d, n1)
            cand = c + grid[:, :1] * n1 + grid[:, 1:] * n2
            vox = np.round(cand / sp).astype(np.int64)
            ok = np.all((vox >= 0) & (vox < bsh), axis=1)
            if not ok.any():
                break
            vox, cand = vox[ok], cand[ok]
            free = obj_lookup(vox) & (tree.query(cand)[0] >= _MARCH_CLEAR_UM)
            if free.sum() < 4:
                break                                         # no free mask ahead: the fibre ends here
            nc = cand[free].mean(0)
            nd = (nc - p) / (np.linalg.norm(nc - p) + 1e-12)
            if nd @ d < cos_t:                                # turn at most _MARCH_TURN_DEG per step
                ax = nd - (nd @ d) * d
                ax /= np.linalg.norm(ax) + 1e-12
                nd = cos_t * d + sin_t * ax
                nc = p + nd * _MARCH_STEP_UM
            trail.append(nc)
            total += float(np.linalg.norm(nc - p))
            p = nc
            d = 0.8 * d + 0.2 * nd
            d /= np.linalg.norm(d)
        if not trail:
            continue
        seq, prev = [Pi], Pi[-1]
        for t in trail:
            v = np.round(t / sp).astype(np.int64)
            if np.all(v == prev):
                continue
            lk = _line_voxels(prev, v)
            lk = lk[obj_lookup(lk)] if len(lk) else lk
            if len(lk):
                seq.append(lk)
            prev = v
        new = np.concatenate(seq)
        new = new[np.r_[True, np.any(np.diff(new, axis=0) != 0, axis=1)]]
        out[i] = new[::-1] if at_start else new
        trails.append(np.array(trail))
    return [o for o in out + hopped if len(o)]


def _hand_on_at_ends(paths, sp, obj_lookup):
    """Re-pair three paths that meet where one ends and another starts.

    A long path i (at least 30 um) passes within _HAND_TOUCH_UM of where a parallel path j ends
    (within _ORDER_PAR_DEG, at least _HAND_SIDE_UM to the side of i's course before the contact).
    If i's rest, after the contact, lies on j's line (within _HAND_ON_RATIO of that offset), and a
    path k starts within 3 um of the contact on i's own line (missing it by no more than
    _HAND_ON_RATIO of the offset, by less than it misses j's line, and by no more than i's own rest
    misses i's line), then i stepped onto j's course there: i goes on into k, and j into i's rest.
    Every path takes part in at most one such repair.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    A = [_arc_points(x, None) if len(x) > 1 else np.zeros(len(x)) for x in X]
    L = [float(a[-1]) if len(a) else 0.0 for a in A]
    ends = [(j, w) for j in range(len(P)) if L[j] >= _SIG_MIN_FIBRE_UM for w in (0, -1)]
    if len(ends) < 3:
        return P
    epts = np.array([X[j][w] for j, w in ends])
    etree = cKDTree(epts)

    def fit(q):
        c = q.mean(0)
        return c, np.linalg.svd(q - c, full_matrices=False)[2][0]

    def tow(u, f, t):
        return u if u @ (t - f) >= 0 else -u

    def miss(c, u, pts):
        w = pts - c
        return float(np.median(np.linalg.norm(w - np.outer(w @ u, u), axis=1)))

    W, G, near = _JOIN_WINDOW_UM, _JOIN_SKIP_UM, 3.0
    c_par = np.cos(np.radians(_ORDER_PAR_DEG))
    c_turn = np.cos(np.radians(_ORDER_TURN_DEG))
    found = []
    for i in range(len(P)):
        if L[i] < 30.0:
            continue
        ti = cKDTree(X[i])
        for h in sorted(set(h_ for hs in etree.query_ball_point(X[i], _HAND_TOUCH_UM) for h_ in hs)):
            j, wj = ends[h]
            pe = epts[h]
            if j == i:
                continue
            s = float(A[i][int(ti.query(pe)[1])])
            for rev in (False, True):
                xi = X[i][::-1] if rev else X[i]
                ai = _arc_points(xi, None)
                si = float(ai[-1] - s) if rev else s
                Ai = xi[(ai >= si - W) & (ai <= si - G)]
                Bi = xi[(ai >= si + G) & (ai <= si + W)]
                if len(Ai) < 8 or len(Bi) < 8:
                    continue
                cA, uA = fit(Ai)
                uA = tow(uA, Ai[0], Ai[-1])
                xj = X[j] if wj == -1 else X[j][::-1]
                aj = _arc_points(xj, None)
                Qj = xj[aj >= aj[-1] - W + G]
                if len(Qj) < 6:
                    continue
                cQ, uQ = fit(Qj)
                uQ = tow(uQ, Qj[0], Qj[-1])
                if uQ @ uA < c_par:
                    continue                                  # j does not run along i before the contact
                side = float(np.linalg.norm((cQ - cA) - ((cQ - cA) @ uA) * uA))
                if side < _HAND_SIDE_UM:
                    continue
                mBj, mBi = miss(cQ, uQ, Bi), miss(cA, uA, Bi)
                if mBj > _HAND_ON_RATIO * side:
                    continue                                  # i's rest is not on j's line
                best = None
                for h2 in etree.query_ball_point(pe, near + W):
                    kk, wk = ends[h2]
                    if kk in (i, j):
                        continue
                    xk = X[kk] if wk == 0 else X[kk][::-1]
                    if np.linalg.norm(xk[0] - pe) > near or (xk[0] - pe) @ uA < -near:
                        continue
                    ak = _arc_points(xk, None)
                    Rk = xk[(ak >= G / 2) & (ak <= W)]
                    if len(Rk) < 8:
                        continue
                    uk = tow(fit(Rk)[1], Rk[0], Rk[-1])
                    mki, mkj = miss(cA, uA, Rk), miss(cQ, uQ, Rk)
                    if uk @ uA < c_turn or mki > _HAND_ON_RATIO * side or mki >= mkj or mki > mBi:
                        continue
                    if best is None or mki < best[0]:
                        best = (mki, kk, wk)
                if best is not None:
                    found.append((side, i, rev, si, j, wj, best[1], best[2]))
    touched, out = set(), list(P)

    def join(a, b):
        lk = _line_voxels(a[-1], b[0])
        lk = lk[obj_lookup(lk)] if len(lk) else lk
        return np.concatenate([a, lk, b])

    for _side, i, rev, si, j, wj, kk, wk in sorted(found, key=lambda t: -t[0]):
        if touched & {i, j, kk}:
            continue
        Pi = P[i][::-1] if rev else P[i]
        ai = _arc_points(Pi * sp, None)
        head = Pi[:int(np.searchsorted(ai, si - G / 2))]
        tail = Pi[int(np.searchsorted(ai, si + G / 2)):]
        if min(len(head), len(tail)) < 6:
            continue
        out[i] = join(head, P[kk] if wk == 0 else P[kk][::-1])
        out[j] = join(P[j] if wj == -1 else P[j][::-1], tail)
        out[kk] = P[kk][:0]
        touched.update((i, j, kk))
    return [o for o in out if len(o)]


def _join_lined_up_ends(paths, sp, r, obj_lookup, bshape):
    """Join path ends that line up across a gap the earlier mending left open.

    `_mend_breaks` joins ends within _JOIN_FAR_R * r whose straight link stays in the mask. Two
    ends can line up as well across a longer gap, along a mask that curves, or across a short gap
    in the mask; or one end can carry a short hook that runs past the other piece. Ends that face
    each other within _LINE_GAP_UM, continue each other's course within _LINE_TURN_DEG
    (_LINE_LONG_TURN_DEG for gaps beyond _JOIN_FAR_R * r) and lie on one line to within
    _LINE_SHIFT_UM are joined, each with its best partner (mutually), along the shortest route
    through the mask (see the constants), or straight if the gap is short and the ends lie on one
    line to within _LINE_STRAIGHT_UM. An end may first drop up to 6 um of tail, but only if that
    tail runs past the other piece. Repeated for up to three rounds, so a piece can be joined at
    both ends.
    """
    from scipy.spatial import cKDTree
    from skimage.graph import MCP_Geometric
    sp = np.asarray(sp, float)
    bsh = np.asarray(bshape)
    far_um, mask_gap = _JOIN_FAR_R * r, _JOIN_MASK_GAP_R * r
    P = [np.asarray(p_) for p_ in paths]

    def fit(q):
        c = q.mean(0)
        return c, np.linalg.svd(q - c, full_matrices=False)[2][0]

    cos_t = math.cos(math.radians(_LINE_TURN_DEG))
    cos_long = math.cos(math.radians(_LINE_LONG_TURN_DEG))
    cos_overlap = math.cos(math.radians(_LINE_OVERLAP_TURN_DEG))
    W, G = _JOIN_WINDOW_UM, _JOIN_SKIP_UM
    for _round in range(3):
        X = [p_ * sp for p_ in P]
        A = [_arc_points(x, None) if len(x) > 1 else np.zeros(len(x)) for x in X]
        L = [float(a[-1]) if len(a) else 0.0 for a in A]
        ends = []
        for i in range(len(P)):
            if L[i] < _SIG_MIN_FIBRE_UM:
                continue
            for w in (0, 1):
                x = X[i] if w == 1 else X[i][::-1]
                a = A[i] if w == 1 else _arc_points(x, None)
                for t in (0.0, 3.0, 6.0):
                    if a[-1] - t < W:
                        continue
                    kc = int(np.searchsorted(a, a[-1] - t, side="right")) - 1
                    q = x[(a >= a[kc] - W) & (a <= a[kc] - G)]
                    if len(q) < 8:
                        continue
                    c, u = fit(q)
                    u = u if u @ (q[-1] - q[0]) >= 0 else -u
                    ends.append((i, w, t, x[kc], c, u, kc, x[-1]))
        if len(ends) < 2:
            break
        E = np.array([e[3] for e in ends])
        etree = cKDTree(E)
        keep = [k for k in range(len(P)) if L[k] >= _SIG_MIN_FIBRE_UM]
        owner = np.concatenate([[k] * len(X[k]) for k in keep])
        ptree = cKDTree(np.vstack([X[k] for k in keep]))
        cands = []
        for a_, ea in enumerate(ends):
            for b_ in etree.query_ball_point(ea[3], _LINE_GAP_UM):
                if b_ <= a_:
                    continue
                eb = ends[b_]
                if ea[0] == eb[0]:
                    continue
                g = eb[3] - ea[3]
                gap = float(np.linalg.norm(g))
                if gap < 1e-6:
                    continue
                cosab = -float(ea[5] @ eb[5])
                overlap = g @ ea[5] <= 0 or -g @ eb[5] <= 0
                if overlap:
                    # Ends that run past each other by up to _LINE_OVERLAP_UM, on one line to within
                    # _LINE_STRAIGHT_UM: one fibre whose two pieces overlap at the break.
                    if (ea[2] or eb[2] or g @ ea[5] < -_LINE_OVERLAP_UM or -g @ eb[5] < -_LINE_OVERLAP_UM
                            or cosab < cos_overlap):
                        continue
                elif cosab < cos_t:
                    continue
                if gap > far_um and (ea[2] or eb[2] or cosab < cos_long):
                    continue
                ok = True
                for e_, o_ in ((ea, eb), (eb, ea)):
                    if e_[2] and (o_[3] - e_[7]) @ e_[5] > 0:
                        ok = False                            # a dropped tail must run past the other piece
                if not ok:
                    continue
                n = ea[5] - eb[5]
                n = n / np.linalg.norm(n)
                mid = 0.5 * (ea[3] + eb[3])
                dv = (ea[4] + ea[5] * (((mid - ea[4]) @ n) / (ea[5] @ n))) - (eb[4] + eb[5] * (((mid - eb[4]) @ n) / (eb[5] @ n)))
                shift = float(np.linalg.norm(dv))
                if shift > (_LINE_STRAIGHT_UM if overlap else _LINE_SHIFT_UM):
                    continue
                cands.append((shift + 0.05 * gap + 0.3 * (ea[2] + eb[2]), a_, b_, gap, shift, overlap))
        cands.sort()
        best_of = {}
        for sc, a_, b_, gap, shift, _ov in cands:
            for e in (a_, b_):
                best_of.setdefault(ends[e][:2], (a_, b_))
        touched, out, n_done = set(), list(P), 0
        for sc, a_, b_, gap, shift, overlap in cands:
            ea, eb = ends[a_], ends[b_]
            if best_of.get(ea[:2]) != (a_, b_) or best_of.get(eb[:2]) != (a_, b_):
                continue
            if ea[0] in touched or eb[0] in touched:
                continue
            va = np.round(ea[3] / sp).astype(np.int64)
            vb = np.round(eb[3] / sp).astype(np.int64)
            lo = np.maximum(np.minimum(va, vb) - np.array([6, 25, 25]), 0)
            hi = np.minimum(np.maximum(va, vb) + np.array([7, 26, 26]), bsh)
            gz, gy, gx = np.meshgrid(*[np.arange(lo[k], hi[k]) for k in range(3)], indexing="ij")
            inside = obj_lookup(np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1)).reshape(gz.shape)
            m_ = MCP_Geometric(np.where(inside, 1.0, 12.0), fully_connected=True, sampling=tuple(sp))
            m_.find_costs([tuple(va - lo)], [tuple(vb - lo)])
            try:
                route = np.array(m_.traceback(tuple(vb - lo))) + lo
            except ValueError:
                continue
            st = np.r_[0.0, np.linalg.norm(np.diff(route * sp, axis=0), axis=1)]
            out_ = ~obj_lookup(route)
            run_ = longest = 0.0
            for k in range(1, len(route)):
                run_ = run_ + st[k] if out_[k] else 0.0
                longest = max(longest, run_)
            dd, hh = ptree.query(route * sp)
            clear = float(np.mean((dd >= _LINE_CLEAR_UM) | np.isin(owner[hh], [ea[0], eb[0]])))
            routed = (st.sum() <= _LINE_ROUTE_RATIO * gap + (1.5 if overlap else 0.0)
                      and longest <= mask_gap and clear >= _LINE_CLEAR_SHARE)
            straight = gap <= far_um and shift <= _LINE_STRAIGHT_UM and not overlap
            if not (routed or straight):
                continue
            Pa = P[ea[0]] if ea[1] == 1 else P[ea[0]][::-1]
            Pb = P[eb[0]] if eb[1] == 1 else P[eb[0]][::-1]
            Pa, Pb = Pa[:ea[6] + 1], Pb[:eb[6] + 1][::-1]
            if routed:
                link = route[obj_lookup(route)]
            else:
                link = _line_voxels(Pa[-1], Pb[0])
                link = link[obj_lookup(link)] if len(link) else link
            out[ea[0]] = np.concatenate([Pa, link, Pb])
            out[eb[0]] = P[eb[0]][:0]
            touched.update((ea[0], eb[0]))
            n_done += 1
        P = [o for o in out if len(o)]
        if not n_done:
            break
    return P


def _join_side_by_side(paths, sp, r, obj_lookup, bshape):
    """Join a path end to a parallel path that starts beside it.

    Two pieces of one fibre can be left side by side, offset sideways (up to _SIDE_OFF_UM) and
    overlapping by up to _SIDE_OVERLAP_UM or a little apart, when the skeleton splits a fibre that
    jogs sideways in a dense bundle. An end and the oppositely-facing end of a parallel path
    (within _SIDE_TURN_DEG) placed like that, each the other's best such partner, are joined along
    the shortest route through the mask between them (at most _SIDE_ROUTE_RATIO x their direct
    distance plus 1.5 um, out of the mask for at most _JOIN_MASK_GAP_R * r, and clear of other paths over
    _LINE_CLEAR_SHARE of it). Where they overlap, of the two stretches beside each other the one
    that bends further off its own piece's course is dropped, as it belongs to another fibre. Pieces that
    simply line up are left to `_join_lined_up_ends`. Repeated for up to three rounds.
    """
    from scipy.spatial import cKDTree
    from skimage.graph import MCP_Geometric
    sp = np.asarray(sp, float)
    bsh = np.asarray(bshape)
    mask_gap = _JOIN_MASK_GAP_R * r
    W, G = _JOIN_WINDOW_UM, _JOIN_SKIP_UM
    cos_t = math.cos(math.radians(_SIDE_TURN_DEG))
    P = [np.asarray(p_) for p_ in paths]

    def fit(q):
        c = q.mean(0)
        return c, np.linalg.svd(q - c, full_matrices=False)[2][0]

    for _round in range(3):
        X = [p_ * sp for p_ in P]
        L = [float(_arc_points(x, None)[-1]) if len(x) > 1 else 0.0 for x in X]
        ends = []
        for i in range(len(P)):
            if L[i] < _SIG_MIN_FIBRE_UM:
                continue
            for w in (0, 1):
                x = X[i] if w == 1 else X[i][::-1]
                a = _arc_points(x, None)
                q = x[(a >= a[-1] - W) & (a <= a[-1] - G)]
                if len(q) < 8:
                    continue
                c, u = fit(q)
                ends.append((i, w, x[-1], c, u if u @ (q[-1] - q[0]) >= 0 else -u))
        if len(ends) < 2:
            break
        etree = cKDTree(np.array([e[2] for e in ends]))
        keep = [k for k in range(len(P)) if L[k] >= _SIG_MIN_FIBRE_UM]
        owner = np.concatenate([[k] * len(X[k]) for k in keep])
        ptree = cKDTree(np.vstack([X[k] for k in keep]))
        cands = []
        for a_, ea in enumerate(ends):
            for b_ in etree.query_ball_point(ea[2], _SIDE_OVERLAP_UM + _SIDE_OFF_UM):
                if b_ <= a_:
                    continue
                eb = ends[b_]
                if ea[0] == eb[0] or -float(ea[4] @ eb[4]) < cos_t:
                    continue
                u = ea[4] - eb[4]
                u = u / np.linalg.norm(u)
                g = eb[2] - ea[2]
                along = float(g @ u)
                if along < -_SIDE_OVERLAP_UM or along > _SIDE_AHEAD_UM:
                    continue
                d = eb[3] - ea[3]
                off = float(np.linalg.norm(d - (d @ u) * u))
                if off > _SIDE_OFF_UM or (along > 0 and off < 2.0):
                    continue                                  # too far aside, or simply lined up
                cands.append((off + 0.1 * abs(along), a_, b_, along))
        cands.sort()
        best_of = {}
        for _sc, a_, b_, _al in cands:
            for e in (a_, b_):
                best_of.setdefault(ends[e][:2], (a_, b_))
        touched, out, n_done = set(), list(P), 0
        for _sc, a_, b_, along in cands:
            ea, eb = ends[a_], ends[b_]
            if best_of.get(ea[:2]) != (a_, b_) or best_of.get(eb[:2]) != (a_, b_):
                continue
            if ea[0] in touched or eb[0] in touched:
                continue
            u = ea[4] - eb[4]
            u = u / np.linalg.norm(u)
            Pa = P[ea[0]] if ea[1] == 1 else P[ea[0]][::-1]
            Pb = P[eb[0]] if eb[1] == 0 else P[eb[0]][::-1]   # starts at b's end
            if along < 0:
                # The two pieces overlap. Of the two stretches that lie beside each other, the one
                # that bends off its own piece's course belongs to another fibre: drop that one.
                ahead = ((Pb * sp - ea[2]) @ u) >= 0          # b's points past a's end
                past = ((Pa * sp - eb[2]) @ u) > 0            # a's points past b's end
                if not ahead.any():
                    continue
                k0 = int(np.argmax(ahead))
                k1 = int(np.argmax(past)) if past.any() else len(Pa)

                def tail_miss(body, tail):
                    if len(body) < 8 or len(tail) < 3:
                        return 0.0
                    bx = body * sp
                    ab = _arc_points(bx, None)
                    q = bx[ab >= ab[-1] - W] if len(bx) else bx
                    c_, u_ = fit(q)
                    w_ = tail * sp - c_
                    return float(np.median(np.linalg.norm(w_ - np.outer(w_ @ u_, u_), axis=1)))

                miss_a = tail_miss(Pa[:k1], Pa[k1:])
                miss_b = tail_miss(Pb[k0:][::-1], Pb[:k0][::-1])
                if miss_a > miss_b and k1 >= 6:
                    Pa = Pa[:k1]                              # a's stretch beside b bends off: drop it
                else:
                    Pb = Pb[k0:]                              # b's stretch beside a bends off: drop it
            if len(Pb) < 6:
                continue
            va, vb = Pa[-1], Pb[0]
            lo = np.maximum(np.minimum(va, vb) - np.array([6, 25, 25]), 0)
            hi = np.minimum(np.maximum(va, vb) + np.array([7, 26, 26]), bsh)
            gz, gy, gx = np.meshgrid(*[np.arange(lo[k], hi[k]) for k in range(3)], indexing="ij")
            inside = obj_lookup(np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1)).reshape(gz.shape)
            m_ = MCP_Geometric(np.where(inside, 1.0, 12.0), fully_connected=True, sampling=tuple(sp))
            m_.find_costs([tuple(va - lo)], [tuple(vb - lo)])
            try:
                route = np.array(m_.traceback(tuple(vb - lo))) + lo
            except ValueError:
                continue
            st = np.r_[0.0, np.linalg.norm(np.diff(route * sp, axis=0), axis=1)]
            out_ = ~obj_lookup(route)
            run_ = longest = 0.0
            for k in range(1, len(route)):
                run_ = run_ + st[k] if out_[k] else 0.0
                longest = max(longest, run_)
            dd, hh = ptree.query(route * sp)
            clear = float(np.mean((dd >= _LINE_CLEAR_UM) | np.isin(owner[hh], [ea[0], eb[0]])))
            direct = float(np.linalg.norm((vb - va) * sp))
            if st.sum() > _SIDE_ROUTE_RATIO * direct + 1.5 or longest > mask_gap or clear < _LINE_CLEAR_SHARE:
                continue
            out[ea[0]] = np.concatenate([Pa, route[obj_lookup(route)], Pb])
            out[eb[0]] = P[eb[0]][:0]
            touched.update((ea[0], eb[0]))
            n_done += 1
        P = [o for o in out if len(o)]
        if not n_done:
            break
    return P


def _march_free_ends(paths, sp, obj_lookup, bshape):
    """Carry free path ends on through mask that no path claims.

    In a thick bundle the skeleton can leave a fibre's continuation without a branch of its own
    (it runs alongside or under another fibre), so its path stops although the mask goes on. Each
    end of a path of at least _SIG_MIN_FIBRE_UM is marched on along its course as in
    `_cut_climbs` (steps centred on mask at least _MARCH_CLEAR_UM from every other path, turning at
    most _MARCH_TURN_DEG per step); the extension is kept if it reaches _FREE_MARCH_MIN_UM.
    Longer paths march first, and each extension is avoided by those that follow.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    bsh = np.asarray(bshape)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    L = [float(_arc_points(x, None)[-1]) if len(x) > 1 else 0.0 for x in X]
    keep = [k for k in range(len(P)) if L[k] >= _SIG_MIN_FIBRE_UM]
    if not keep:
        return P
    g = np.arange(-_MARCH_DISC_UM, _MARCH_DISC_UM + 1e-9, float(sp.min()))
    grid = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=1) <= _MARCH_DISC_UM]
    cos_t, sin_t = np.cos(np.radians(_MARCH_TURN_DEG)), np.sin(np.radians(_MARCH_TURN_DEG))
    base = np.vstack([X[k] for k in keep])
    base_owner = np.concatenate([[k] * len(X[k]) for k in keep])
    trails = []
    out = list(P)
    for i in sorted(keep, key=lambda k: -L[k]):
        for at_start in (False, True):
            Pi = out[i][::-1] if at_start else out[i]
            x = Pi * sp
            a = _arc_points(x, None)
            q = x[a >= a[-1] - _JOIN_WINDOW_UM]
            if len(q) < 6:
                continue
            c_ = q.mean(0)
            u = np.linalg.svd(q - c_, full_matrices=False)[2][0]
            u = u if u @ (q[-1] - q[0]) >= 0 else -u
            pts = base[base_owner != i]
            if trails:
                pts = np.vstack([pts] + trails)
            tree = cKDTree(pts)
            p, dvec = x[-1], u
            trail, total = [], 0.0
            while total < _MARCH_MAX_UM:
                c = p + dvec * _MARCH_STEP_UM
                n1 = np.array([0.0, -dvec[2], dvec[1]])
                n1 /= np.linalg.norm(n1) + 1e-12
                n2 = np.cross(dvec, n1)
                cand = c + grid[:, :1] * n1 + grid[:, 1:] * n2
                vox = np.round(cand / sp).astype(np.int64)
                ok = np.all((vox >= 0) & (vox < bsh), axis=1)
                if not ok.any():
                    break
                vox, cand = vox[ok], cand[ok]
                free = obj_lookup(vox) & (tree.query(cand)[0] >= _MARCH_CLEAR_UM)
                if free.sum() < 4:
                    break
                nc = cand[free].mean(0)
                nd = (nc - p) / (np.linalg.norm(nc - p) + 1e-12)
                if nd @ dvec < cos_t:
                    ax = nd - (nd @ dvec) * dvec
                    ax /= np.linalg.norm(ax) + 1e-12
                    nd = cos_t * dvec + sin_t * ax
                    nc = p + nd * _MARCH_STEP_UM
                trail.append(nc)
                total += float(np.linalg.norm(nc - p))
                p = nc
                dvec = 0.8 * dvec + 0.2 * nd
                dvec /= np.linalg.norm(dvec)
            if total < _FREE_MARCH_MIN_UM:
                continue
            seq, prev = [Pi], Pi[-1]
            for t in trail:
                v = np.round(t / sp).astype(np.int64)
                if np.all(v == prev):
                    continue
                lk = _line_voxels(prev, v)
                lk = lk[obj_lookup(lk)] if len(lk) else lk
                if len(lk):
                    seq.append(lk)
                prev = v
            new = np.concatenate(seq)
            new = new[np.r_[True, np.any(np.diff(new, axis=0) != 0, axis=1)]]
            out[i] = new[::-1] if at_start else new
            trails.append(np.array(trail))
    return out


def _follow_own_line(paths, sp, obj_lookup):
    """Put a path back on its own line where it strays onto a neighbour.

    Where the mask along a fibre is thin or broken, the skeleton (and so the path) can
    swing onto the track of the fibre next to it and come back, so the soma takes a bite of
    its neighbour. A stretch (up to `_OWN_MAX_UM`) that leaves the line through the path on
    either side by more than `_OWN_DEV_UM`, where that line is straight and the two sides run
    parallel, is replaced by the line itself if it rides along another fibre's path (within
    `_OWN_RIDE_UM`, for at least `_OWN_RIDE_LEN_UM`), or if the line there crosses a gap in
    the mask and the stretch veers to within `_OWN_VEER_UM` of another fibre: across a gap
    the fibre jumps straight rather than detouring into its neighbour. Only mask voxels of
    the line are kept.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    L = [float(_arc_points(x, None)[-1]) if len(x) > 1 else 0.0 for x in X]
    longs = [k for k in range(len(P)) if L[k] >= _SIG_MIN_FIBRE_UM]
    if len(longs) < 2:
        return P
    own = np.concatenate([[k] * len(X[k]) for k in longs])
    tree = cKDTree(np.vstack([X[k] for k in longs]))
    c_par = np.cos(np.radians(_OWN_PAR_DEG))

    def fit(q):
        c = q.mean(0)
        return c, np.linalg.svd(q - c, full_matrices=False)[2][0]

    def miss(c, u, q):
        w = q - c
        return np.linalg.norm(w - np.outer(w @ u, u), axis=1)

    out = list(P)
    for i in range(len(P)):
        if L[i] < 30.0:
            continue
        x = X[i]
        a = _arc_points(x, None)
        ss = np.arange(10.0, a[-1] - 10.0, 0.5)
        dev = np.zeros(len(ss))
        for n, s in enumerate(ss):
            q = x[((a >= s - 10.0) & (a <= s - 3.0)) | ((a >= s + 3.0) & (a <= s + 10.0))]
            if len(q) < 8:
                continue
            c, u = fit(q)
            dev[n] = miss(c, u, x[min(int(np.searchsorted(a, s)), len(x) - 1)][None])[0]
        runs, n = [], 0
        while n < len(ss):
            if dev[n] > _OWN_DEV_UM:
                m = n
                while m + 1 < len(ss) and dev[m + 1] > _OWN_DEV_UM:
                    m += 1
                runs.append((ss[n] - 1.0, ss[m] + 1.0))
                n = m + 1
            else:
                n += 1
        reps = []
        for s0, s1 in runs:
            if s1 - s0 > _OWN_MAX_UM:
                continue
            A = x[(a >= s0 - _OWN_FLANK_UM) & (a < s0)]
            B = x[(a > s1) & (a <= s1 + _OWN_FLANK_UM)]
            if len(A) < 6 or len(B) < 6 or abs(fit(A)[1] @ fit(B)[1]) < c_par:
                continue
            c, u = fit(np.vstack([A, B]))
            if np.median(miss(c, u, np.vstack([A, B]))) > 0.45:
                continue
            S = x[(a >= s0) & (a <= s1)]
            dd, hh = tree.query(S, k=min(12, len(own)))
            dd, hh = dd.reshape(len(S), -1), hh.reshape(len(S), -1)
            d = np.where(own[np.minimum(hh, len(own) - 1)] != i, dd, np.inf).min(1)
            ride = float((d < _OWN_RIDE_UM).mean()) * (s1 - s0)
            ts = np.linspace(float((S[0] - c) @ u), float((S[-1] - c) @ u), max(4, len(S)))
            gap = 1.0 - float(obj_lookup(np.round((c + np.outer(ts, u)) / sp).astype(np.int64)).mean())
            if ride >= _OWN_RIDE_LEN_UM or (gap >= _OWN_GAP_SHARE and float(d.min()) <= _OWN_VEER_UM):
                reps.append((s0, s1))
        if not reps:
            continue
        p_ = P[i]
        seq, k = [], 0
        for s0, s1 in reps:
            k0 = int(np.searchsorted(a, s0))
            k1 = min(int(np.searchsorted(a, s1)), len(p_) - 1)
            if k0 < k:
                continue
            seq.append(p_[k:k0])
            lk = _line_voxels(p_[max(k0 - 1, 0)], p_[k1])
            lk = lk[obj_lookup(lk)] if len(lk) else lk
            if len(lk):
                seq.append(lk)
            k = k1
        seq.append(p_[k:])
        q = np.concatenate([s_ for s_ in seq if len(s_)])
        out[i] = q[np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]]
    return out


def _resolve_contacts_by_intensity(paths, sp, obj_lookup, intensity_box):
    """Let the image decide which way two fibres go where their paths touch.

    Where two fibres cross at a shallow angle in a thick part of the mask, the skeleton
    gives them a short shared stretch and the path shapes alone cannot tell whether they
    crossed or touched and turned back, so the tracer may hand each the other's
    continuation. Each fibre's bright core is then followed through the contact: from the
    straight course of its path just before it (`_CONTACT_LEG_UM`, ending `_CONTACT_GAP_UM`
    before the contact), step by step (0.5 um, at most 8 degrees of turn), each step centred
    on the smoothed intensity across the course, for the contact's length plus
    `_CONTACT_GAP_UM` + `_CONTACT_EXTRA_UM`. If both cores come out on the other path's
    continuation (nearer than `_CONTACT_RATIO` of their distance to their own, and within
    `_CONTACT_HIT_UM` of it), the continuations are swapped. Geometry alone decides
    everywhere else.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    A = [_arc_points(x, None) if len(x) > 1 else np.zeros(len(x)) for x in X]
    L = [float(a[-1]) if len(a) else 0.0 for a in A]
    big = [k for k in range(len(P)) if L[k] >= 30.0]
    if len(big) < 2:
        return P
    c_ang = np.cos(np.radians(_CONTACT_DEG))
    disc = 0.9
    g = np.arange(-disc, disc + 1e-9, 0.15)
    grid = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=1) <= disc]
    c8, s8 = np.cos(np.radians(8.0)), np.sin(np.radians(8.0))

    def fit(q):
        c = q.mean(0)
        u = np.linalg.svd(q - c, full_matrices=False)[2][0]
        return c, (u if u @ (q[-1] - q[0]) >= 0 else -u)

    def follow(start, n_steps, img, org):
        _, d = fit(start)
        p = start[-1].copy()
        out = []
        for _ in range(n_steps):
            n1 = np.array([0.0, -d[2], d[1]])
            n1 /= np.linalg.norm(n1) + 1e-12
            n2 = np.cross(d, n1)
            cand = p + d * 0.5 + grid[:, :1] * n1 + grid[:, 1:] * n2
            v = np.round(cand / sp).astype(np.int64) - org
            v = np.clip(v, 0, np.array(img.shape) - 1)
            w = img[tuple(v.T)]
            w = np.clip(w - np.percentile(w, 30), 0, None)
            if w.sum() <= 0:
                break
            nc = (cand * w[:, None]).sum(0) / w.sum()
            nd = (nc - p) / (np.linalg.norm(nc - p) + 1e-12)
            if nd @ d < c8:
                ax = nd - (nd @ d) * d
                ax /= np.linalg.norm(ax) + 1e-12
                nd = c8 * d + s8 * ax
                nc = p + nd * 0.5
            p = nc
            d = 0.85 * d + 0.15 * nd
            d /= np.linalg.norm(d)
            out.append(p.copy())
        return np.array(out)

    found = []
    for n_, i in enumerate(big):
        ti = cKDTree(X[i])
        for j in big[n_ + 1:]:
            d, k = ti.query(X[j], distance_upper_bound=_CONTACT_UM)
            ks = np.nonzero(np.isfinite(d))[0]
            if not len(ks):
                continue
            sj0, sj1 = float(A[j][ks].min()), float(A[j][ks].max())
            si = A[i][k[ks]]
            si0, si1 = float(si.min()), float(si.max())
            if sj1 - sj0 > _CONTACT_SPAN_UM or si1 - si0 > _CONTACT_SPAN_UM:
                continue
            xi, xj, ai, aj = X[i], X[j], A[i], A[j]
            ui = xi[min(int(np.searchsorted(ai, si1)), len(xi) - 1)] - xi[int(np.searchsorted(ai, si0))]
            uj = xj[min(int(np.searchsorted(aj, sj1)), len(xj) - 1)] - xj[int(np.searchsorted(aj, sj0))]
            rev = float(ui @ uj) < 0
            G, W = _CONTACT_GAP_UM, _CONTACT_LEG_UM
            i_in = xi[(ai >= si0 - G - W) & (ai <= si0 - G)]
            i_out = xi[(ai >= si1 + 1.0) & (ai <= si1 + 25.0)]
            if rev:
                j_in = xj[(aj <= sj1 + G + W) & (aj >= sj1 + G)][::-1]
                j_out = xj[(aj <= sj0 - 1.0) & (aj >= sj0 - 25.0)][::-1]
            else:
                j_in = xj[(aj >= sj0 - G - W) & (aj <= sj0 - G)]
                j_out = xj[(aj >= sj1 + 1.0) & (aj <= sj1 + 25.0)]
            if min(len(i_in), len(i_out), len(j_in), len(j_out)) < 6:
                continue
            if abs(fit(i_in)[1] @ fit(j_in)[1]) < c_ang:
                continue
            n_steps = int((max(si1 - si0, sj1 - sj0) + G + _CONTACT_EXTRA_UM) / 0.5)
            allp = np.vstack([i_in, j_in, i_out, j_out])
            lo = np.floor(allp.min(0) / sp).astype(np.int64) - 12
            hi = np.ceil(allp.max(0) / sp).astype(np.int64) + 13
            img, org = intensity_box(lo, hi)
            if img.size == 0:
                continue
            img = ndimage.gaussian_filter(img, (0.7, 1.2, 1.2))
            res = []
            for inc, own, oth in ((i_in, i_out, j_out), (j_in, j_out, i_out)):
                tr = follow(inc, n_steps, img, org)
                if not len(tr):
                    res = None
                    break
                e = tr[-1]
                res.append((float(np.linalg.norm(own - e, axis=1).min()),
                            float(np.linalg.norm(oth - e, axis=1).min())))
            if res is None:
                continue
            if all(b <= _CONTACT_RATIO * a and b <= _CONTACT_HIT_UM for a, b in res):
                found.append((max(b / a for a, b in res), i, j, si0, si1, sj0, sj1, rev))
    used, out = set(), list(P)
    for _sc, i, j, si0, si1, sj0, sj1, rev in sorted(found):
        if i in used or j in used:
            continue
        mi = 0.5 * (si0 + si1)
        mj = 0.5 * (sj0 + sj1)
        ki = int(np.searchsorted(A[i], mi))
        kj = int(np.searchsorted(A[j], mj))
        head_i, tail_i = P[i][:ki], P[i][ki:]
        if rev:
            head_j, tail_j = P[j][kj:][::-1], P[j][:kj][::-1]
        else:
            head_j, tail_j = P[j][:kj], P[j][kj:]
        if min(len(head_i), len(tail_i), len(head_j), len(tail_j)) < 6:
            continue
        new = []
        for h, t in ((head_i, tail_j), (head_j, tail_i)):
            lk = _line_voxels(h[-1], t[0])
            lk = lk[obj_lookup(lk)] if len(lk) else lk
            new.append(np.concatenate([h, lk, t]))
        out[i], out[j] = new
        used.update((i, j))
    return out


def _cut_loops(paths, sp, obj_lookup):
    """Cut out stretches where a path doubles back on itself.

    A fibre does not run backwards. Joins of overlapping pieces, and links across
    knots of skeleton, can leave a path that goes forward, back by a few micrometres
    (often along a neighbour's track) and forward again. Walking along the path, its
    direction is taken from the 6 um it has just covered; a stretch whose progress along
    that direction falls more than `_LOOP_BACK_UM` behind the furthest point reached is
    dropped, up to where the path passes that point again, and the two ends are linked.
    """
    sp = np.asarray(sp, float)
    out = []
    for p_ in paths:
        p_ = np.asarray(p_)
        x = p_ * sp
        if len(x) < 20:
            out.append(p_)
            continue
        a = _arc_points(x, None)
        keep = np.ones(len(x), bool)
        k = 1
        while k < len(x):
            ref = x[(a >= a[k] - 6.0) & (a <= a[k])]
            if len(ref) < 4 or a[k] < 6.0:
                k += 1
                continue
            u = ref[-1] - ref[0]
            nu = float(np.linalg.norm(u))
            if nu < 1e-6:
                k += 1
                continue
            u = u / nu
            front = x[k - 1]
            j = k
            while j < len(x) and float((x[j] - front) @ u) < 0.3:
                j += 1
            if j > k and float(np.min((x[k:j] - front) @ u)) < -_LOOP_BACK_UM:
                keep[k:j] = False                         # back and forth: cut out
                k = j
            else:
                k += 1
        if keep.all():
            out.append(p_)
            continue
        q = p_[keep]
        seq = [q[:1]]
        for v in q[1:]:
            prev = seq[-1][-1]
            if np.abs(v - prev).max() > 1:
                lk = _line_voxels(prev, v)
                lk = lk[obj_lookup(lk)] if len(lk) else lk
                if len(lk):
                    seq.append(lk)
            seq.append(v[None])
        q = np.concatenate(seq)
        out.append(q[np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]])
    return out


def _split_overlapping_steps(paths, sp, obj_lookup):
    """Split a path at a sideways step where two fibres overlap.

    Where one fibre ends and another starts beside it, overlapping it for a few micrometres,
    the tracer follows one into the other with a small sideways step, and the overlapping
    end of one of them is left as a short dead-end piece of skeleton. A path of at least
    30 um that steps sideways (offset `_STEP_SHIFT_UM`) between two straight stretches of
    `_STEP_LEG_UM` that run parallel (within `_STEP_PAR_DEG`), with such a piece (3 to 8 um)
    within `_STEP_NEAR_UM` of the step that either carries the old line on past the step or
    the new line back before it (missing that line by less than `_STEP_RATIO` of the
    other's), where the path does not jump a gap in the mask at the step nor come within
    `_MARCH_CLEAR_UM` of another fibre's path within 2 um of it, and that keeps clear (`_MARCH_CLEAR_UM`) of every other fibre's path over most
    of its length (a piece lying along another path is that fibre's spur), is split at the step; the piece goes to the fibre whose line it carries.
    Every path and piece takes part in at most one split.
    """
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    P = [np.asarray(p_) for p_ in paths]
    X = [p_ * sp for p_ in P]
    L = [float(_arc_points(x, None)[-1]) if len(x) > 1 else 0.0 for x in X]
    pieces = [k for k in range(len(P)) if 3.0 <= L[k] < _SIG_MIN_FIBRE_UM]
    if not pieces:
        return P
    fowner = np.concatenate([[k] * len(X[k]) for k in pieces])
    ftree = cKDTree(np.vstack([X[k] for k in pieces]))
    longs = [k for k in range(len(P)) if L[k] >= _SIG_MIN_FIBRE_UM]
    if not longs:
        return P
    lowner = np.concatenate([[k] * len(X[k]) for k in longs])
    ltree = cKDTree(np.vstack([X[k] for k in longs]))

    def clear_of_others(f, i):
        # A piece lying along another fibre's path is a spur of that fibre, not the end of one here.
        d, h = ltree.query(X[f], k=min(4, len(lowner)))
        d, h = d.reshape(len(X[f]), -1), h.reshape(len(X[f]), -1)
        far = np.ones(len(X[f]), bool)
        for col in range(d.shape[1]):
            other = lowner[np.minimum(h[:, col], len(lowner) - 1)] != i
            far &= ~((d[:, col] < _MARCH_CLEAR_UM) & other)
        return float(far.mean()) >= 0.8

    c_par = np.cos(np.radians(_STEP_PAR_DEG))
    c_piece = np.cos(np.radians(_STEP_PIECE_DEG))
    gap = 0.75

    def fit(q):
        c = q.mean(0)
        u = np.linalg.svd(q - c, full_matrices=False)[2][0]
        return c, (u if u @ (q[-1] - q[0]) >= 0 else -u)

    def miss(c, u, q):
        w = q - c
        return np.linalg.norm(w - np.outer(w @ u, u), axis=1)

    found = []
    for i in range(len(P)):
        if L[i] < 30.0:
            continue
        x = X[i]
        a = _arc_points(x, None)
        best = None
        for s in np.arange(_STEP_LEG_UM + gap, a[-1] - _STEP_LEG_UM - gap, _SIG_STEP_UM):
            Aq = x[(a >= s - _STEP_LEG_UM - gap) & (a <= s - gap)]
            Bq = x[(a >= s + gap) & (a <= s + _STEP_LEG_UM + gap)]
            if len(Aq) < 8 or len(Bq) < 8:
                continue
            cA, uA = fit(Aq)
            cB, uB = fit(Bq)
            if uA @ uB < c_par:
                continue
            if np.median(miss(cA, uA, Aq)) > _STEP_STRAIGHT_UM or np.median(miss(cB, uB, Bq)) > _STEP_STRAIGHT_UM:
                continue
            pt = x[min(int(np.searchsorted(a, s)), len(x) - 1)]
            m = (uA + uB) / np.linalg.norm(uA + uB)
            d = (cB + ((pt - cB) @ uB) * uB) - (cA + ((pt - cA) @ uA) * uA)
            shift = float(np.linalg.norm(d - (d @ m) * m))
            if not (_STEP_SHIFT_UM[0] <= shift <= _STEP_SHIFT_UM[1]):
                continue
            near_step = np.nonzero((a >= s - 1.5) & (a <= s + 1.5))[0]
            if len(near_step) > 1 and np.abs(np.diff(P[i][near_step], axis=0)).max() > 1:
                continue                                      # the step jumps a gap in the mask: one fibre
            around = x[(a >= s - 2.0) & (a <= s + 2.0)]
            dd, hh = ltree.query(around, k=min(8, len(lowner)))
            dd, hh = dd.reshape(len(around), -1), hh.reshape(len(around), -1)
            if np.any((dd < _MARCH_CLEAR_UM) & (lowner[np.minimum(hh, len(lowner) - 1)] != i)):
                continue                                      # the step touches another fibre: a contact, not an overlap
            for f in sorted(set(int(fowner[h]) for h in ftree.query_ball_point(pt, _STEP_NEAR_UM))):
                F = X[f]
                if abs(fit(F)[1] @ uA) < c_piece or not clear_of_others(f, i):
                    continue
                mA, mB = float(np.median(miss(cA, uA, F))), float(np.median(miss(cB, uB, F)))
                back = float((((F - pt) @ uA) < 0).mean())
                if back >= 0.7 and mB < _STEP_RATIO * mA:
                    case = "B"                                # the new fibre started before the step
                elif back <= 0.3 and mA < _STEP_RATIO * mB:
                    case = "A"                                # the old fibre carries on past it
                else:
                    continue
                score = min(mA, mB) / max(mA, mB)
                if best is None or score < best[0]:
                    best = (score, s, f, case, uA)
        if best is not None:
            found.append((best[0], i) + best[1:])
    used, out, extra = set(), list(P), []
    for _sc, i, s, f, case, uA in sorted(found, key=lambda t: t[0]):
        if i in used or f in used:
            continue
        a = _arc_points(X[i], None)
        head = P[i][:int(np.searchsorted(a, s - gap))]
        tail = P[i][int(np.searchsorted(a, s + gap)):]
        if len(head) < 6 or len(tail) < 6:
            continue
        F = P[f][np.argsort((X[f] - X[i][0]) @ uA)]          # the piece, in the path's direction

        def join(p1, p2):
            lk = _line_voxels(p1[-1], p2[0])
            lk = lk[obj_lookup(lk)] if len(lk) else lk
            return np.concatenate([p1, lk, p2])

        if case == "A":
            head = join(head, F)
        else:
            tail = join(F, tail)
        out[i] = head
        out[f] = P[f][:0]
        extra.append(tail)
        used.update((i, f))
    return [o for o in out + extra if len(o)]


def _end_course(pv, which, sp):
    """The course of a path's end: (end point, centroid, outward direction), all in
    physical units, from the stretch of the path between _JOIN_SKIP_UM and
    _JOIN_WINDOW_UM back from the end (`which`: 0 the first voxel, 1 the last). None if
    that stretch is shorter than _JOIN_MIN_UM."""
    x = pv * sp
    if which == 0:
        x = x[::-1]
    back = float(np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))) - np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    sel = (back >= _JOIN_SKIP_UM) & (back <= _JOIN_WINDOW_UM)
    if sel.sum() < 4 or back[sel].max() - back[sel].min() < _JOIN_MIN_UM:
        return None
    c = x[sel].mean(0)
    u = np.linalg.svd(x[sel] - c, full_matrices=False)[2][0]
    if u @ (x[-1] - c) < 0:
        u = -u
    return x[-1], c, u


def _mend_breaks(chains, sp, r, obj_lookup, extras=None, focus=None, bshape=None):
    """Join the pieces of one fibre that a contact left apart, comparing them on their
    courses rather than on their ends (see the comment below). Returns the paths.

    With `extras` (a list of further paths, appended after `chains`) only a pair made of one
    extra and one real path may join, and an extra that joins none is dropped: that is how
    leftover skeleton is attached to the fibres it continues without taking part in the
    competition between the real paths."""
    # ---- second phase: mend the breaks a contact left in a fibre -----------------
    # Near a contact the mask's bumps and the crossing fibres bend a piece's last few
    # microns, so its END direction and position can be off by more than a fibre width
    # while the fibre's course a little further back is straight and continues on the
    # other side. The pieces still free after the rounds above are therefore compared on
    # their courses (`_end_course`): two ends within 10 r of each other, continuing each
    # other within _MAX_TURN_DEG, whose courses are off one line by less than
    # _JOIN_SHIFT_R r midway between them, with the mask missing along the straight link
    # for no more than _JOIN_MASK_GAP_R r. As before, a pair is joined only if each is the
    # other's first choice and never so as to close a loop.
    from scipy.spatial import cKDTree
    win = _ARM_WINDOW_R * r
    n_real = len(chains)
    if extras:
        chains = list(chains) + list(extras)
    is_extra = [i >= n_real for i in range(len(chains))]
    # With `focus` (indices into `chains`) only pairs with at least one end on those paths may join,
    # and a path made by a join takes no further part: the rest was mended already.
    # Such a join may also follow the mask around a corner (with `bshape`, the box's shape): where
    # the straight link leaves the mask for too long, the shortest route through the mask between
    # the ends is used instead, if it is no longer than _MEND_ROUTE_RATIO times the gap and leaves
    # the mask for no longer than the straight link may.
    is_focus = [focus is None or i in set(focus) for i in range(len(chains))]
    routes = {}

    def mask_route(va, vb, max_out):
        from skimage.graph import MCP_Geometric
        lo = np.maximum(np.minimum(va, vb) - np.array([6, 25, 25]), 0)
        hi = np.minimum(np.maximum(va, vb) + np.array([7, 26, 26]), np.asarray(bshape))
        gz, gy, gx = np.meshgrid(*[np.arange(lo[k], hi[k]) for k in range(3)], indexing="ij")
        inside = obj_lookup(np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1)).reshape(gz.shape)
        m_ = MCP_Geometric(np.where(inside, 1.0, 12.0), fully_connected=True, sampling=tuple(sp))
        m_.find_costs([tuple(va - lo)], [tuple(vb - lo)])
        try:
            route = np.array(m_.traceback(tuple(vb - lo))) + lo
        except ValueError:
            return None
        st = np.r_[0.0, np.linalg.norm(np.diff(route * sp, axis=0), axis=1)]
        if st.sum() > _MEND_ROUTE_RATIO * float(np.linalg.norm((vb - va) * sp)):
            return None
        out_ = ~obj_lookup(route)
        run_ = longest_ = 0.0
        for k in range(1, len(route)):
            run_ = run_ + st[k] if out_[k] else 0.0
            longest_ = max(longest_, run_)
        return route if longest_ <= max_out else None
    cos_turn = math.cos(math.radians(_MAX_TURN_DEG))
    # Two reaches, the shorter first: within 10 r, then within _JOIN_FAR_R r, where the pieces
    # must continue each other more straightly. At both, the mask may be missing along the
    # straight link for at most _JOIN_MASK_GAP_R r: a longer stretch without mask means the
    # segmentation does not connect the two pieces, and a straight line across it says nothing
    # about which fibre they belong to.
    tiers = ((win, _JOIN_MASK_GAP_R * r, cos_turn, _JOIN_SHIFT_R * r),
             (_JOIN_FAR_R * r, _JOIN_MASK_GAP_R * r, math.cos(math.radians(_JOIN_FAR_TURN_DEG)),
              _JOIN_SHIFT_R * r))
    for reach, mask_gap, tier_cos, tier_shift in tiers:
        if extras:
            mask_gap = reach        # leftover skeleton on the far side is itself the evidence of fibre there
        for _round in range(64):
            routes.clear()                  # keyed by path indices, which change every round
            ends = []                       # (path index, which end 0|1, end point, centroid, direction)
            for i, pv in enumerate(chains):
                for which in (0, 1):
                    c = _end_course(pv, which, sp)
                    if c is not None:
                        ends.append((i, which) + c)
            if len(ends) < 2:
                break
            tree = cKDTree(np.array([e[2] for e in ends]))
            cand = {}
            for a, b in sorted(tree.query_pairs(reach)):
                ia, wa, ea, ca, ua = ends[a]
                ib, wb, eb, cb, ub = ends[b]
                if ia == ib or -float(ua @ ub) < tier_cos or (extras and is_extra[ia] == is_extra[ib]):
                    continue
                if not (is_focus[ia] or is_focus[ib]):
                    continue
                g = eb - ea
                n = ua - ub
                n = n / np.linalg.norm(n)
                mid = 0.5 * (ea + eb)
                shift = float(np.linalg.norm((ca + ua * (((mid - ca) @ n) / (ua @ n)))
                                             - (cb + ub * (((mid - cb) @ n) / (ub @ n)))))
                if shift >= tier_shift:
                    continue                # off-line: a step onto another fibre
                ahead = min(float(g @ ua), float(-g @ ub))
                if ahead < 0 and (-ahead > _JOIN_OVERLAP_R * r or shift >= 2.0 * r):
                    continue                # overlapping along the fibre but not on one line: two fibres side by side
                link = _line_voxels(np.round(ea / sp).astype(np.int64), np.round(eb / sp).astype(np.int64))
                gone = ~obj_lookup(link)
                if gone.any():
                    step = np.r_[0.0, np.linalg.norm(np.diff(link * sp, axis=0), axis=1)]
                    run = longest = 0.0
                    for k in range(1, len(link)):
                        run = run + step[k] if gone[k] else 0.0
                        longest = max(longest, run)
                    if longest > mask_gap:
                        route = None
                        if focus is not None and bshape is not None:
                            route = mask_route(np.round(ea / sp).astype(np.int64), np.round(eb / sp).astype(np.int64), mask_gap)
                        if route is None:
                            continue        # the mask is missing for too long between them
                        routes[(ia, wa, ib, wb)] = route
                cost = shift + 0.3 * float(np.linalg.norm(g))
                cand.setdefault(a, []).append((cost, b))
                cand.setdefault(b, []).append((cost, a))
            best = {e: min(c)[1] for e, c in cand.items()}
            pairs = [(a, b) for a, b in best.items() if a < b and best.get(b) == a]
            if not pairs:
                break
            taken, merged, drop = set(), [], set()
            for a, b in sorted(pairs):
                ia, wa = ends[a][:2]
                ib, wb = ends[b][:2]
                if ia in taken or ib in taken:
                    continue
                taken.update((ia, ib))
                pa_seq = chains[ia] if wa == 1 else chains[ia][::-1]      # joined end last
                pb_seq = chains[ib] if wb == 0 else chains[ib][::-1]      # joined end first
                # Pieces that overlap along the fibre are cut where they cross midway between
                # the ends, so the joined path runs on without doubling back.
                ea, ua, eb, ub = ends[a][2], ends[a][4], ends[b][2], ends[b][4]
                n = (ua - ub) / np.linalg.norm(ua - ub)
                mid = 0.5 * (ea + eb)
                ka = np.nonzero((pa_seq * sp - mid) @ n <= 0)[0]
                kb = np.nonzero((pb_seq * sp - mid) @ n >= 0)[0]
                if len(ka) and len(kb):
                    pa_seq, pb_seq = pa_seq[:ka[-1] + 1], pb_seq[kb[0]:]
                rt = routes.get((ia, wa, ib, wb))
                if rt is None and (ib, wb, ia, wa) in routes:
                    rt = routes[(ib, wb, ia, wa)][::-1]
                if rt is not None:
                    link = rt[obj_lookup(rt)]
                    pa_seq, pb_seq = (chains[ia] if wa == 1 else chains[ia][::-1]), (chains[ib] if wb == 0 else chains[ib][::-1])
                else:
                    link = _line_voxels(pa_seq[-1], pb_seq[0])
                    link = link[obj_lookup(link)]
                merged.append(np.concatenate([pa_seq, link, pb_seq]))
                drop.update((ia, ib))
            is_extra = [f for i, f in enumerate(is_extra) if i not in drop] + [False] * len(merged)
            is_focus = [f for i, f in enumerate(is_focus) if i not in drop] + [focus is None] * len(merged)
            chains = [pv for i, pv in enumerate(chains) if i not in drop] + merged

    if extras:
        chains = [c for c, f in zip(chains, is_extra) if not f]      # an extra that joined nothing is dropped
    return chains


def _stitch_paths(paths, sp, r, obj_lookup):
    """Join traced paths that are consecutive pieces of one fibre.

    Only a path's two ENDS are ever joined, never places where paths touch
    along their sides -- that is how side-by-side fibres and bridges were
    welded before. Every path takes part, short ones included: in a dense
    bundle a fibre is traced as a chain of pieces, many of them shorter than
    the direction window, and leaving those out left the chain broken. An end
    is its end point plus the direction of its last 10 r (of the whole path,
    if shorter). Two ends from different paths are candidates if they are
    within 10 r, the straight segment between them stays inside the mask,
    they continue each other (point towards each other), and each end's
    extended line passes within one fibre spacing (2 r) of the other end.
    Every end ranks its candidates by how far off-line they are; a pair is
    joined only if each is the other's first choice, and never so as to close
    a loop. Joined paths keep their outer ends and directions, so this repeats
    until nothing changes. Returns the paths, ordered along their fibres.

    A second phase then mends breaks whose ends a contact has bent, comparing the
    pieces left on their courses rather than on their ends (see the comment there).
    """
    win = _ARM_WINDOW_R * r
    paths = [np.asarray(p_) for p_ in paths if len(p_)]

    def arc(pv):
        xyz = pv * sp
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]

    def end_geom(pv):
        """(end point, outward direction) at the path's last voxel."""
        xyz = pv * sp
        a = arc(pv)
        j = int(np.searchsorted(a, a[-1] - win))
        d = xyz[-1] - xyz[min(j, len(xyz) - 2)]
        n = np.linalg.norm(d)
        return xyz[-1], (d / n if n > 0 else None)

    chains = list(paths)

    for _round in range(64):
        ends = []                       # (path index, which end 0|1, point, direction)
        for i, pv in enumerate(chains):
            for which, seq in ((0, pv[::-1]), (1, pv)):
                pt, d = end_geom(seq)
                if d is not None:
                    ends.append((i, which, pt, d))
        cand = {}
        for a in range(len(ends)):
            ia, wa, pa, da = ends[a]
            for b in range(a + 1, len(ends)):
                ib, wb, pb, db = ends[b]
                if ia == ib:
                    continue            # both ends of one path: a loop
                g = pb - pa
                dist = float(np.linalg.norm(g))
                if dist > win:
                    continue
                if float(da @ db) >= 0 or (dist > 0 and (float(g @ da) < 0 or float(-g @ db) < 0)):
                    continue            # they do not continue each other
                lat_a = float(np.linalg.norm(g - (g @ da) * da))
                lat_b = float(np.linalg.norm(-g - (-g @ db) * db))
                if lat_a >= 2 * r or lat_b >= 2 * r:
                    continue            # off-line: a step onto another fibre
                seg_v = _line_voxels(np.round(pa / sp).astype(np.int64),
                                     np.round(pb / sp).astype(np.int64))
                if len(seg_v) and not bool(np.all(obj_lookup(seg_v))):
                    continue            # would cross background
                cost = lat_a + lat_b + r * (1.0 + float(da @ db))
                cand.setdefault(a, []).append((cost, b))
                cand.setdefault(b, []).append((cost, a))
        best = {e: min(c)[1] for e, c in cand.items()}
        pairs = [(a, b) for a, b in best.items() if a < b and best.get(b) == a]
        if not pairs:
            break
        # Join each mutual pair; a path joined twice this round waits.
        taken = set()
        merged = []
        drop = set()
        for a, b in pairs:
            ia, wa, pa, _ = ends[a]
            ib, wb, pb, _ = ends[b]
            if ia in taken or ib in taken:
                continue
            taken.update((ia, ib))
            pa_seq = chains[ia] if wa == 1 else chains[ia][::-1]      # joined end last
            pb_seq = chains[ib] if wb == 0 else chains[ib][::-1]      # joined end first
            seg_v = _line_voxels(pa_seq[-1], pb_seq[0])
            merged.append(np.concatenate([pa_seq, seg_v, pb_seq]))
            drop.update((ia, ib))
        chains = [pv for i, pv in enumerate(chains) if i not in drop] + merged


    return _mend_breaks(chains, sp, r, obj_lookup)


def _elongated_fibre_candidates(lbl, sl, segmentation_mask, intensity_image,
                                params: "_CandidateParams"):
    """Seeds for spindle- or fibre-shaped cells, from the MASK's skeleton.

    The mask carries what is needed: its skeleton gives each fibre's course and
    its width -- and in a stack its height -- tells how many fibres a stretch
    holds. Where two fibres lie one above the other the mask is cut between
    them before skeletonizing, at the intensity valley where there is one;
    that is the only use of the intensity image. A plain skeleton merges
    fibres -- every touch point is a junction, a ladder of fibres joined by
    bridges is a loop, fibres fused side by side are one branch -- so it is
    decomposed into paths that can never branch (`_paths_from_skeleton`):
    hairs are removed, bridges carry no fibre, ribbons are split into lanes,
    and at every junction a fibre continues onto its straightest branch,
    judged on the geometry with the bends into the contacts removed. Each path becomes
    one seed: a tube of radius r around it, clipped to the mask, the nearest
    path winning where tubes meet, so neighbouring seeds never overlap.

    Tiled like compact mode. Dense work (distance transforms, skeleton, tubes)
    runs per tile with a halo wide enough to be exact inside the tile; what
    crosses tiles is only the skeleton -- about one voxel per voxel of fibre
    length -- so the graph is built and paired for the whole object at once.

    Config input: min_fragment_size (voxels per seed). Every scale derives
    from the object's fibre radius r (see the constants above).
    """
    import shutil
    import tempfile
    from skimage.morphology import skeletonize

    diag = {"cores_evaluated": 0, "cores_too_small": 0, "thickness_rejected": 0,
            "aspect_ratio_rejected": 0, "spatial_overlap_rejected": 0,
            "pushed_and_dropped": 0}

    is_stack = params.ndim == 3
    seg = segmentation_mask if is_stack else segmentation_mask[None]
    box = tuple(sl) if is_stack else (slice(0, 1),) + tuple(sl)
    tile = tuple(int(t) for t in params.tile_size)
    tile = tile if is_stack else (1,) + tile
    sp = np.asarray(params.spacing, float)
    sp = sp if is_stack else np.r_[1.0, sp]
    image_bounds = tuple(slice(0, n) for n in seg.shape)
    targets = list(_tile_targets(box, tile))
    bshape = tuple(b.stop - b.start for b in box)
    boff = np.array([b.start for b in box])

    # ---- pass 1: fibre radius and the object's thickest point ---------------
    # In a stack also the single-fibre column height -- how many slices one
    # fibre spans, at the same centre-line voxels -- and the noise of the
    # intensity around the object, which the stacked-fibre cut uses.
    ridge_vals, dt_max = [], 0.0
    col_vals, bg_n, bg_s, bg_ss = [], 0, 0.0, 0.0
    for tg in targets:
        h_um = 5.0
        while True:
            halo = [int(np.ceil(h_um / s)) + 1 for s in sp]
            if not is_stack:
                halo[0] = 0
            crop = _grow(tg, halo, image_bounds)
            obj = np.asarray(seg[crop]) == lbl
            loc = _local(tg, crop)
            if not obj[loc].any():
                dt = None
                break
            dt = ndimage.distance_transform_edt(obj, sampling=sp)
            # Exact inside the tile when no inner distance could reach past
            # the halo; otherwise widen it and recompute.
            if float(dt[loc].max()) < h_um - float(sp.max()):
                break
            h_um *= 2.0
        if dt is None:
            continue
        ridge = obj & (dt >= ndimage.maximum_filter(dt, size=3)) & (dt > 0)
        ridge_vals.append(dt[loc][ridge[loc]].astype(np.float32))
        dt_max = max(dt_max, float(dt[loc].max()))
        if is_stack:
            col_vals.append(_z_run_length(obj)[loc][ridge[loc]])
            if intensity_image is not None:
                bg = np.asarray(intensity_image[tg])[np.asarray(seg[tg]) == 0].astype(np.float64)
                bg_n += bg.size
                bg_s += float(bg.sum())
                bg_ss += float((bg * bg).sum())
        del obj, dt, ridge
    rv = np.concatenate(ridge_vals) if ridge_vals else np.zeros(0, np.float32)
    if rv.size == 0:
        return [], diag
    r = float(np.median(rv))
    del ridge_vals, rv
    if not (r > 0):
        return [], diag
    h0 = float(np.median(np.concatenate(col_vals))) if col_vals else 0.0
    noise = 0.0
    if bg_n > 1:
        var = max(0.0, bg_ss / bg_n - (bg_s / bg_n) ** 2)
        noise = math.sqrt(var) / 3.0            # noise of a 3 x 3 in-plane mean

    # ---- pass 2: skeleton and in-plane half-width, exact per tile ------------
    # Thinning reaches as far as the object is thick, so a halo of a few times
    # its thickest radius (and at least one arm window) makes each tile's
    # skeleton the object's own inside the tile. In a stack the mask is first
    # cut between fibres stacked in z (`_cut_stacked_fibres`); only the
    # skeleton sees the cut -- the half-width, links and seeds use the mask.
    # The necks between fibres that touch side by side are found here too
    # (`_neck_lines`), but cut only later, where a seed path turns out to run
    # along one (see below).
    h_um = max(4.0 * dt_max, _ARM_WINDOW_R * r)
    halo = [int(np.ceil(h_um / s)) + 2 for s in sp]
    if not is_stack:
        halo[0] = 0
    neck_lines = []                     # box coordinates, one array per neck

    def skeleton_tile(tg, cut=None, find_necks=False):
        """(skeleton points in box coordinates, their half-widths) inside tile `tg`, or None.
        `cut`: box coordinates of neck voxels to remove before thinning."""
        crop = _grow(tg, halo, image_bounds)
        loc = _local(tg, crop)
        obj = np.asarray(seg[crop]) == lbl
        if not obj[loc].any():
            return None
        c0 = np.array([c.start for c in crop]) - boff
        if not is_stack:
            sk = skeletonize(obj[0])[None]
        elif h0 >= 1:
            inten = None
            if intensity_image is not None and noise > 0:
                inten = np.asarray(intensity_image[crop], np.float32)
            cobj = _cut_stacked_fibres(obj, inten, h0, noise, float(sp[1]))
            del inten
            if find_necks:
                t0 = np.array([t.start for t in tg]) - boff
                t1 = np.array([t.stop for t in tg]) - boff
                for line in _neck_lines(cobj, sp, r):
                    g = line + c0
                    g = g[np.all((g >= t0) & (g < t1), axis=1)]     # the tile's own part
                    if len(g):
                        neck_lines.append(g)
            if cut is not None:
                q = cut - c0
                q = q[np.all((q >= 0) & (q < np.array(cobj.shape)), axis=1)]
                if len(q):
                    if cobj is obj:
                        cobj = obj.copy()
                    cobj[tuple(q.T)] = False
            sk = skeletonize(cobj)
            del cobj
        else:
            sk = skeletonize(obj)
        sk = sk.astype(bool)
        # In-plane half-width: a flat ribbon is one fibre deep, so the 3D
        # distance to the background would see its depth, not its width.
        w2 = np.zeros(obj.shape, np.float32)
        for z in range(obj.shape[0]):
            if obj[z].any():
                w2[z] = ndimage.distance_transform_edt(obj[z], sampling=sp[1:])
        if obj.shape[0] > 1:
            w2 = ndimage.maximum_filter1d(w2, size=3, axis=0)
        inner = sk[loc]
        q = np.argwhere(inner)
        if not len(q):
            return None
        return ((q + np.array([t.start for t in tg]) - boff).astype(np.int64),
                w2[loc][inner].astype(np.float32))

    tile_skel = [skeleton_tile(tg, find_necks=is_stack and h0 >= 1) for tg in targets]

    def obj_lookup(coords):
        coords = np.asarray(coords, np.int64)
        if not len(coords):
            return np.zeros(0, bool)
        g = coords + boff
        return np.asarray(seg[tuple(g.T)]) == lbl

    full_shape = np.asarray(segmentation_mask.shape)

    def intensity_box(lo, hi):
        """The intensity image over box coordinates [lo, hi) (clipped to the image)."""
        g0 = np.maximum(np.asarray(lo) + boff, 0)
        g1 = np.minimum(np.asarray(hi) + boff, full_shape)
        return np.asarray(intensity_image[tuple(slice(int(a), int(b)) for a, b in zip(g0, g1))],
                          np.float32), g0 - boff

    def trace(skel):
        """The whole object's graph, decomposed into paths."""
        found = [t_ for t_ in skel if t_ is not None]
        if not found:
            return []
        paths_ = _paths_from_skeleton(np.concatenate([t_[0] for t_ in found]),
                                      np.concatenate([t_[1] for t_ in found]),
                                      bshape, sp, r, obj_lookup)
        if intensity_image is not None and is_stack:
            paths_ = _resolve_contacts_by_intensity(paths_, sp, obj_lookup, intensity_box)
        return paths_

    def skeleton_with_cut(cut):
        """The tiles' skeletons with `cut` removed; tiles whose halo misses it are reused."""
        out_ = []
        for tg, prev in zip(targets, tile_skel):
            crop = _grow(tg, halo, image_bounds)
            c0 = np.array([c.start for c in crop]) - boff
            c1 = np.array([c.stop for c in crop]) - boff
            if np.any(np.all((cut >= c0) & (cut < c1), axis=1)):
                out_.append(skeleton_tile(tg, cut=cut))
            else:
                out_.append(prev)
        return out_

    paths = trace(tile_skel)
    if not paths:
        return [], diag

    # ---- pass 3: each path widened to the mask's own width, per tile ---------
    # Around each path voxel the seed takes the mask's in-plane half-width at
    # that voxel as its radius, so it has the mask's cross-section along the
    # fibre -- a fixed radius made seeds narrower than wide fibres and notched
    # where neighbouring tubes met. The nearest path still wins, so seeds never
    # overlap. The halo covers the widest half-width twice -- once to reach the
    # nearest path voxel, once more for that voxel's own width to be measured
    # with the background around it in view -- so each tile is exact.
    th = [int(np.ceil((2.0 * max(r, dt_max) + float(sp.max())) / s)) + 2 for s in sp]
    if not is_stack:
        th[0] = 0
    in_box = lambda t: tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(t, box))
    workdir = tempfile.mkdtemp(prefix=f"elongated_{lbl}_", dir=params.temp_dir)
    try:
        piece_mm = np.memmap(os.path.join(workdir, "seeds.dat"), np.int32, "w+", shape=bshape)

        def build_tubes(path_list):
            """Tubes for `path_list` into piece_mm; returns voxel count per label."""
            path_lab = np.concatenate([np.full(len(p_), k + 1, np.int32)
                                       for k, p_ in enumerate(path_list)])
            path_vox = np.concatenate(path_list)
            counts = np.zeros(len(path_list) + 1, np.int64)
            for tg in targets:
                crop = _grow(tg, th, box)
                loc = _local(tg, crop)
                obj = np.asarray(seg[crop]) == lbl
                if not obj[loc].any():
                    continue
                c0 = np.array([c.start for c in crop]) - boff
                c1 = np.array([c.stop for c in crop]) - boff
                sel = np.all((path_vox >= c0) & (path_vox < c1), axis=1)
                if not sel.any():
                    piece_mm[in_box(tg)] = 0
                    continue
                lab_img = np.zeros(obj.shape, np.int32)
                lab_img[tuple((path_vox[sel] - c0).T)] = path_lab[sel]
                w_img = np.zeros(obj.shape, np.float32)
                for z in range(obj.shape[0]):
                    if obj[z].any():
                        w_img[z] = ndimage.distance_transform_edt(obj[z], sampling=sp[1:])
                dist, idx = ndimage.distance_transform_edt(lab_img == 0, sampling=sp,
                                                           return_indices=True)
                reach = w_img[tuple(idx)] + float(sp.max())
                seedt = np.where(obj & (dist <= reach), lab_img[tuple(idx)], 0)[loc]
                piece_mm[in_box(tg)] = seedt
                counts += np.bincount(seedt.ravel(), minlength=counts.size)[:counts.size]
                del lab_img, w_img, dist, idx, reach, obj
            return counts

        # A path too short to make a seed must not take tube voxels from its
        # neighbours: drop it and rebuild, until every tube reaches the size.
        min_vol = max(1, params.min_seed_vol)

        def make_seeds(path_list):
            """Tubes of the paths that make seeds, in piece_mm: (those paths, voxel counts,
            how many paths were too small), or (None, None, n) if none is left."""
            n_small = 0
            counts_ = None
            small_ = []
            for _round in range(4):
                counts_ = build_tubes(path_list)
                small_ = [k for k in range(len(path_list)) if counts_[k + 1] < min_vol]
                n_small += len(small_)
                if not small_:
                    break
                path_list = [p_ for k, p_ in enumerate(path_list) if counts_[k + 1] >= min_vol]
                if not path_list:
                    return None, None, n_small
            counts_ = build_tubes(path_list) if small_ else counts_
            return path_list, counts_, n_small

        def seeded_near(line):
            """Seeded voxels of the mask within _NECK_CHECK_UM of the neck `line`."""
            ext = np.ceil(_NECK_CHECK_UM / sp).astype(np.int64)
            lo_ = np.maximum(line.min(0) - ext, 0)
            hi_ = np.minimum(line.max(0) + ext + 1, np.asarray(bshape))
            bx = tuple(slice(int(a_), int(b_)) for a_, b_ in zip(lo_, hi_))
            off = np.zeros(tuple(int(b_ - a_) for a_, b_ in zip(lo_, hi_)), bool)
            off[tuple((line - lo_).T)] = True
            near = ndimage.distance_transform_edt(~off, sampling=sp) <= _NECK_CHECK_UM
            near &= np.asarray(seg[tuple(slice(b_.start + o_, b_.stop + o_) for b_, o_ in zip(bx, boff))]) == lbl
            return int((np.asarray(piece_mm[bx])[near] > 0).sum())

        def carried(line, path_list):
            """Does a seed path run along the neck `line` for at least _NECK_RUN_UM?"""
            lo_ = line.min(0) - 1
            hi_ = line.max(0) + 2
            near = np.zeros(tuple(hi_ - lo_), bool)
            near[tuple((line - lo_).T)] = True
            near = ndimage.binary_dilation(near, structure=np.ones((3, 3, 3)))
            for p_ in path_list:
                inside = np.all((p_ >= lo_) & (p_ < hi_), axis=1)
                if inside.sum() < 2:
                    continue
                on = np.zeros(len(p_), bool)
                on[inside] = near[tuple((p_[inside] - lo_).T)]
                if on.sum() < 2:
                    continue
                a_ = _arc_points(p_ * sp, None)
                k_ = np.flatnonzero(on)
                start_ = last_ = k_[0]
                for k in k_[1:]:
                    if a_[k] - a_[last_] > 0.5:
                        if a_[last_] - a_[start_] >= _NECK_RUN_UM:
                            return True
                        start_ = k
                    last_ = k
                if a_[last_] - a_[start_] >= _NECK_RUN_UM:
                    return True
            return False

        all_paths = paths
        paths, counts, n_small = make_seeds(all_paths)
        # A seed path that runs along a neck sits between two fibres: cut those necks and
        # trace again, and keep each cut only where it leaves the mask around it at least as
        # well seeded (a neck that is really a thin fibre of its own is not a neck to cut).
        if paths is not None and neck_lines:
            sites = [g for g in neck_lines if carried(g, paths)]
            if sites:
                before = [seeded_near(g) for g in sites]
                trial_all = trace(skeleton_with_cut(np.concatenate(sites)))
                t_paths, t_counts, t_small = make_seeds(trial_all) if trial_all else (None, None, 0)
                after = [seeded_near(g) for g in sites] if t_paths is not None else [-1] * len(sites)
                keep = [g for g, b_, a_ in zip(sites, before, after) if a_ >= b_]
                if keep and len(keep) == len(sites):
                    all_paths, paths, counts, n_small = trial_all, t_paths, t_counts, t_small
                elif keep:
                    all_paths = trace(skeleton_with_cut(np.concatenate(keep)))
                    paths, counts, n_small = make_seeds(all_paths) if all_paths else (None, None, 0)
                else:
                    paths, counts, n_small = make_seeds(all_paths)
        diag["cores_evaluated"] = len(all_paths)
        diag["cores_too_small"] += n_small
        if paths is None:
            del piece_mm
            return [], diag
        piece_mm.flush()

        # ---- one candidate per path ------------------------------------------
        keep = counts >= min_vol
        keep[0] = False
        coords, best = {}, {}
        for tg in targets:
            ids = np.asarray(piece_mm[in_box(tg)])
            if not ids.any():
                continue
            vox = np.argwhere(keep[ids])
            if not len(vox):
                continue
            lab_v = ids[tuple(vox.T)]
            g = vox + np.array([t.start for t in tg])
            for rt in np.unique(lab_v):
                coords.setdefault(int(rt), []).append(g[lab_v == rt].astype(np.int32))
        candidates = []
        for rt in sorted(coords):
            cc = np.concatenate(coords[rt])
            pk = cc[len(cc) // 2]
            if not is_stack:
                cc, pk = cc[:, 1:], pk[1:]
            candidates.append({
                "coords": cc.astype(np.int32),
                "peak_coord": np.asarray(pk),
                "vol": int(len(cc)),
                "score": 1.0,
                "strat_name": "Fibre",
                "frag_max_thick": r,
                "family": "Fibre",
                "dt_vals": np.zeros(len(cc), np.float32),
                "rank_vals": np.zeros(len(cc), np.float32),
            })
        del piece_mm
        return candidates, diag
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _elongated_label_candidates(lbl, sl, segmentation_mask, intensity_image, params):
    """Seed candidates for one object in elongated mode.

    The object's fibres (`_elongated_fibre_candidates`). An object none of whose fibre pieces
    reaches the minimum seed size, but which is itself at least that large, becomes one seed
    as a whole: it is a short piece of fibre the trimming cut off from its neighbours, and
    leaving it without a soma would drop it from the segmentation.
    """
    candidates, diag = _elongated_fibre_candidates(lbl, sl, segmentation_mask, intensity_image, params)
    if candidates:
        return candidates, diag
    own = np.asarray(segmentation_mask[tuple(sl)]) == lbl
    n = int(own.sum())
    if n < int(params.min_seed_vol):
        return candidates, diag
    cc = (np.argwhere(own) + np.array([s_.start for s_ in sl])).astype(np.int32)
    candidates = [{
        "coords": cc,
        "peak_coord": np.asarray(cc[len(cc) // 2]),
        "vol": n,
        "score": 1.0,
        "strat_name": "WholeObject",
        "frag_max_thick": 0.0,
        "family": "Fibre",
        "dt_vals": np.zeros(n, np.float32),
        "rank_vals": np.zeros(n, np.float32),
    }]
    return candidates, diag