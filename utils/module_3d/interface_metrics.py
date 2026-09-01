"""
Interface statistics and the merge decision, shared by 2D and 3D.

Why this module exists
----------------------
The decision whether two adjacent basins are one cell or two is taken in two
different places:

*   in the chunk worker, on a crop that holds both somata; and
*   in the post-stitch pass, on interfaces the worker deferred because neither
    side had a soma in its crop.

Both must apply identical thresholds to identically-defined quantities, or the
same boundary can be judged differently depending on where the chunk seams
happened to fall. So the arithmetic lives here once, and both callers reach it
through `decide_merge`.

The quantities
--------------
Every test reduces to sums and counts over three voxel sets:

*   the **interface** -- voxels of B adjacent to A, inside the parent object;
*   **zone A** -- voxels of A within `local_analysis_radius` of the interface;
*   **zone B** -- voxels of B within `local_analysis_radius` of the interface.

plus one scalar the caller supplies: the mean intensity of the somata involved.

Nothing here depends on the size of the parent object, the area of the interface,
or the distance between the somata. The only spatial extent involved is
`local_analysis_radius`. That is what lets the post-stitch pass accumulate the
same statistics by streaming fixed-size tiles (see `InterfaceStats.merge`) and
hold one tile at a time, whatever the data looks like.

Sums are accumulated in float64. The two paths visit voxels in a different order,
and in float32 that produced answers differing in the eighth significant figure --
harmless for any threshold you would tune to, but enough to make a verdict sitting
exactly on a threshold depend on the tiling, which is not a thing anyone should
have to debug.
"""

from typing import Any, Dict, Optional
import numpy as np


class InterfaceStats:
    """
    Additive accumulator for one interface.

    Instances sum, which is the whole point: a tiled sweep produces one instance
    per tile and the total is their sum, exactly equal to what a single pass over
    the whole interface would have produced.
    """

    __slots__ = ("iface_sum", "iface_n", "a_sum", "a_n", "b_sum", "b_n")

    def __init__(self) -> None:
        self.iface_sum = 0.0
        self.iface_n = 0
        self.a_sum = 0.0
        self.a_n = 0
        self.b_sum = 0.0
        self.b_n = 0

    def add(self, intensity: np.ndarray, iface: np.ndarray,
            zone_a: np.ndarray, zone_b: np.ndarray) -> None:
        """Accumulate one tile. Masks must already be restricted to the tile's
        own non-overlapping target region, so no voxel is counted twice."""
        i64 = np.asarray(intensity, dtype=np.float64)
        self.iface_sum += float(i64[iface].sum())
        self.iface_n += int(iface.sum())
        self.a_sum += float(i64[zone_a].sum())
        self.a_n += int(zone_a.sum())
        self.b_sum += float(i64[zone_b].sum())
        self.b_n += int(zone_b.sum())

    def merge(self, other: "InterfaceStats") -> "InterfaceStats":
        self.iface_sum += other.iface_sum
        self.iface_n += other.iface_n
        self.a_sum += other.a_sum
        self.a_n += other.a_n
        self.b_sum += other.b_sum
        self.b_n += other.b_n
        return self

    # -- derived quantities -------------------------------------------------

    @property
    def interface_mean(self) -> float:
        return self.iface_sum / self.iface_n if self.iface_n else 0.0

    @property
    def zone_a_mean(self) -> float:
        return self.a_sum / self.a_n if self.a_n else 0.0

    @property
    def zone_b_mean(self) -> float:
        return self.b_sum / self.b_n if self.b_n else 0.0

    @property
    def zone_mean(self) -> float:
        """
        Mean intensity of the neighbourhood the interface sits in -- both basins
        within `local_analysis_radius`, taken together.

        This is the reference for the bright-cut test. It replaces the mean of the
        whole parent object, which is not a usable reference at every scale: a
        parent object can be a single doublet or the entire volume, and in the
        latter case its mean describes the image rather than the boundary being
        judged. The neighbourhood mean asks what the test is actually trying to
        ask -- is this boundary darker than the tissue immediately around it --
        and is well defined whatever the object turns out to be.
        """
        n = self.a_n + self.b_n
        return (self.a_sum + self.b_sum) / n if n else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "iface_sum": self.iface_sum, "iface_n": self.iface_n,
            "a_sum": self.a_sum, "a_n": self.a_n,
            "b_sum": self.b_sum, "b_n": self.b_n,
        }


def decide_merge(
    stats: InterfaceStats,
    avg_soma_intensity: float,
    min_local_intensity_difference: float,
    min_path_intensity_ratio: float,
    max_interface_to_zone_ratio: float = 0.85,
    min_zone_voxels: int = 20,
) -> Dict[str, Any]:
    """
    Apply the three merge tests to accumulated statistics.

    Returns a dict carrying `should_merge_decision` plus every intermediate value,
    so the caller can log exactly why a boundary went the way it did.

    1.  **Valley depth** (soma-relative). interface / soma_mean must be below
        `min_path_intensity_ratio` to keep the basins apart. Failing it merges.

    2.  **Local contrast**. The two basins must differ near the interface by at
        least `min_local_intensity_difference`. Failing it merges. Skipped, and
        treated as passing, when either basin has fewer than `min_zone_voxels` in
        the neighbourhood -- too little to compare.

    3.  **Bright cut** (neighbourhood-relative). interface / zone_mean at or above
        `max_interface_to_zone_ratio` means the boundary is no darker than the
        tissue around it. This never merges on its own: it can only tip a
        valley-depth verdict that passed but sits within 70% of its threshold.
        It is the same interface measurement as test 1 under a different
        reference, so it amounts to applying test 1 at an adjusted threshold.
    """
    m = {"should_merge_decision": False}

    if stats.iface_n == 0:
        m["reason"] = "no_interface"
        return m

    iface = stats.interface_mean
    m["interface_mean"] = iface
    m["soma_ref"] = avg_soma_intensity
    m["zone_mean"] = stats.zone_mean
    m["zone_a_mean"] = stats.zone_a_mean
    m["zone_b_mean"] = stats.zone_b_mean
    m["iface_n"] = stats.iface_n

    # 1. valley depth
    ratio_soma = iface / max(avg_soma_intensity, 1e-6)
    soma_passed = ratio_soma < min_path_intensity_ratio
    m["ratio_soma"] = ratio_soma
    m["soma_ratio_passed"] = soma_passed

    # 2. local contrast
    if stats.a_n < min_zone_voxels or stats.b_n < min_zone_voxels:
        lid_passed = True
        rel_diff = float("nan")
    else:
        m1, m2 = stats.zone_a_mean, stats.zone_b_mean
        ref = max(m1, m2)
        if ref < 1e-6:
            lid_passed, rel_diff = True, float("nan")
        else:
            rel_diff = abs(m1 - m2) / ref
            lid_passed = rel_diff >= min_local_intensity_difference
    m["local_rel_diff"] = rel_diff
    m["lid_passed"] = lid_passed

    # 3. bright cut
    ratio_zone = iface / max(stats.zone_mean, 1e-6)
    zone_passed = ratio_zone < max_interface_to_zone_ratio
    m["ratio_zone"] = ratio_zone
    m["zone_ratio_passed"] = zone_passed

    soma_borderline = soma_passed and ratio_soma > min_path_intensity_ratio * 0.7
    if not soma_passed or not lid_passed:
        m["should_merge_decision"] = True
        m["reason"] = "valley_depth" if not soma_passed else "local_contrast"
    elif not zone_passed and soma_borderline:
        m["should_merge_decision"] = True
        m["reason"] = "bright_cut_on_borderline_valley"
    else:
        m["reason"] = "kept"
    return m


def format_decision(m: Dict[str, Any], thr_soma: float, thr_zone: float) -> str:
    """One-line log rendering, identical from either caller."""
    if m.get("reason") == "no_interface":
        return "no interface voxels"
    warn = " *** BRIGHT CUT ***" if not m.get("zone_ratio_passed", True) else ""
    return (
        f"iface={m['interface_mean']:.1f} (n={m['iface_n']}) | "
        f"soma_ref={m['soma_ref']:.1f} | "
        f"soma_ratio={m['ratio_soma']:.4f} (thr={thr_soma}, "
        f"passed={m['soma_ratio_passed']}) | "
        f"zone_mean={m['zone_mean']:.1f} | "
        f"zone_ratio={m['ratio_zone']:.4f} (thr={thr_zone}, "
        f"passed={m['zone_ratio_passed']}){warn} | "
        f"lid_passed={m['lid_passed']} (diff={m['local_rel_diff']:.4f}) | "
        f"=> merge={m['should_merge_decision']} [{m['reason']}]"
    )
