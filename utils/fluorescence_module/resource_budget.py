"""
One resource budget for the whole application.

Why this exists
---------------
Every step of the pipeline used to carry its own hardcoded memory shape: a
``(64, 512, 512)`` chunk here, a ``(128, 256, 256)`` there, ``(2048, 2048)`` at
rank 2, ``mp.Pool(os.cpu_count() - 2)`` in the enhancement pass, ``32`` planes
in one scan loop and ``64`` in the next. Each number was chosen so the step
would survive on the smallest machine anyone had run it on, and nothing
consulted the machine it was actually running on. The consequence is that a
workstation with 68 GB and 24 cores processes a dataset at the speed of an 8 GB
laptop, and -- in the other direction -- a 16-core 8 GB laptop can still be
pushed over the edge, because the worker count came from the CPU count while
the memory cost came from the chunk shape and the two were never multiplied
together.

So there is one budget, set once per install, and the steps ask it for their
shapes. Three numbers, entered by the user:

    RAM (GB)      the ceiling for the WHOLE application, not just the pipeline
    VRAM (GB)     the ceiling for GPU work (0 disables GPU entirely)
    cores         the maximum number of processes/threads to run at once

Nothing else is exposed. How a step spends those three numbers -- whether it
buys more workers or bigger blocks -- is decided here and in the step, not by
the user. Fifteen knobs is not a setting, it is a maintenance burden.

Two signals, and why they are different
---------------------------------------
Sizing and braking use DIFFERENT measurements, deliberately.

*Sizing* uses a ledger: the planner knows exactly how many bytes it just told a
step to allocate, because it computed that number. This is what makes the
budget honourable in advance rather than discovered after an OOM.

*Braking* uses ``psutil.virtual_memory().available``, NOT resident set size.
This matters more than it looks. Almost every large array in this pipeline is a
``np.memmap``, and touching one pulls its pages into the page cache, where they
count toward RSS while remaining fully reclaimable -- the kernel drops them
under pressure and nothing crashes. A brake driven by RSS would therefore
throttle a correctly-behaving streaming step to a crawl the moment it read a
big file, which is the opposite of the intent. ``available`` already excludes
reclaimable cache, so it is the honest predictor of the thing the user actually
asked not to happen: the machine running out of memory while a browser is open.

RSS of the process tree is still reported, because it is what you want in a log
when a step misbehaves. It is diagnostics, not control.

What the budget is NOT allowed to touch
---------------------------------------
Some "chunk sizes" in this pipeline are algorithm parameters wearing a memory
parameter's clothes, and wiring them to a budget would change results between
two machines -- which is indistinguishable from a bug in the science. They are
listed in `PINNED` with the reason, and the planner refuses to size them. The
rule for telling them apart: a geometry is budget-driven only if the operator
is computed with enough halo that the result is bit-identical to the
whole-array computation. If the result depends on where a boundary falls, the
geometry is pinned.

Dependencies
------------
Standard library, numpy and psutil only. NO Qt, and nothing from
`high_level_gui`: this module is imported by pipeline steps that run inside
`multiprocessing` workers, where importing the GUI stack is both slow and, on
some drivers, fatal. The settings dialog reads this module, never the reverse.
"""

from __future__ import annotations

import json
import math
import threading
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is in environment.yml
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


__all__ = [
    "GB",
    "ProcessingCancelled",
    "cancel_requested",
    "check_cancelled",
    "clear_cancel",
    "request_cancel",
    "MIN_RAM_GB",
    "PINNED",
    "BlockPlan",
    "Budget",
    "DeviceCapabilities",
    "Monitor",
    "ResourceSettings",
    "cost_bytes_per_voxel",
    "describe_environment",
    "feasible_range",
    "iter_leading_spans",
    "load_settings",
    "open_budget",
    "pinned",
    "pinned_reason",
    "probe_device",
    "save_settings",
    "settings_path",
    "state_dir",
]


GB = 1024 ** 3

#: Smallest RAM ceiling the pipeline is expected to run under. Not a guess: at
#: 2 GB there is room for the interpreter, one 4096x4096 float32 working set and
#: its temporaries, which is the smallest block any step is allowed to fall back
#: to. Below this the honest answer is "this machine cannot run this", and the
#: planner says so rather than pretending.
MIN_RAM_GB = 2.0

#: Never drive the SYSTEM this close to empty, whatever the user's ceiling says.
#: This is the "a browser is open" clause: the budget is our own ceiling, and
#: this is the floor under everyone else. Checked at runtime by `Monitor`.
MIN_SYSTEM_FREE_GB = 1.0

#: Fraction of the budget at which the brake engages. Below it, nothing happens;
#: above it, the step is asked to shed concurrency and collect garbage before
#: taking the next block. Not 1.0, because a block is committed before it is
#: measured -- the brake has to act while there is still room to act in.
BRAKE_WATERMARK = 0.85

#: Default share of physical RAM proposed on a machine that has never been
#: configured, and the headroom left for the rest of the system. A fresh install
#: therefore gets a budget appropriate to the machine rather than the one the
#: constants were tuned on, and the user can raise or lower it afterwards.
_DEFAULT_RAM_FRACTION = 0.60
_DEFAULT_RAM_HEADROOM_GB = 4.0


# --------------------------------------------------------------------------
# Where the setting lives
# --------------------------------------------------------------------------

def state_dir() -> str:
    """
    The per-install state directory: ``$HIBACHI_STATE_DIR`` or ``~/.hibachi``.

    Same rule as `project_selection._default_state_dir`, which is where the
    config library and the recent-projects list already live. It is duplicated
    here rather than imported because that module pulls in Qt, and this one is
    imported inside `multiprocessing` workers. When `project_selection` is next
    edited its function should be changed to delegate here, so there is one
    definition again.

    Deliberately NOT the install directory. A settings file inside the git
    checkout makes the working tree dirty, and `updater.describe_version`
    stamps that dirty flag into every processed config -- so a preference would
    show up as "this result came from modified code". It would also be at the
    mercy of the self-update.
    """
    return os.environ.get("HIBACHI_STATE_DIR") or os.path.join(
        os.path.expanduser("~"), ".hibachi"
    )


def settings_path() -> str:
    """Absolute path to the resource settings file (may not exist yet)."""
    return os.path.join(state_dir(), "resources.json")


# --------------------------------------------------------------------------
# What the machine actually has
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceCapabilities:
    """Measured hardware limits. The upper bounds the UI offers."""

    total_ram_gb: float
    available_ram_gb: float
    logical_cores: int
    physical_cores: int
    gpus: Tuple[Tuple[str, float], ...] = ()   # (name, total VRAM in GB)

    @property
    def total_vram_gb(self) -> float:
        """VRAM of the LARGEST single GPU, not the sum.

        A buffer cannot straddle two cards, so the sum would advertise a
        capacity no single allocation can use. Multi-GPU work would need its
        own scheduling and does not exist here.
        """
        return max((v for _n, v in self.gpus), default=0.0)

    @property
    def gpu_name(self) -> Optional[str]:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda nv: nv[1])[0]


_DEVICE_CACHE: Optional[DeviceCapabilities] = None


def _probe_gpus() -> Tuple[Tuple[str, float], ...]:
    """
    Installed GPUs and their VRAM, via ``nvidia-smi``.

    A subprocess rather than a library on purpose: there is no CUDA package in
    the environment (no cupy, no cucim, no torch), so there is nothing to import
    and asking would mean adding a dependency in order to answer a question the
    driver already answers. Anything that is not an NVIDIA card reports no GPU,
    which disables GPU work rather than guessing at a capacity.

    Best-effort in every direction: a missing binary, a driver error, a timeout
    or unparseable output all mean "no usable GPU". This function must never be
    the reason the application fails to start.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if out.returncode != 0 or not out.stdout.strip():
        return ()

    found: List[Tuple[str, float]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mib = float(parts[1])
        except ValueError:
            continue
        # nvidia-smi reports MiB; the setting is in GB (1024^3) so that a
        # 24576 MiB card reads as 24.0 rather than 25.8.
        found.append((parts[0], mib / 1024.0))
    return tuple(found)


def probe_device(refresh: bool = False) -> DeviceCapabilities:
    """
    Measure the machine. Cached, because probing the GPU spawns a process.

    ``refresh=True`` re-measures, which is worth doing when the UI opens the
    settings dialog -- ``available_ram_gb`` moves constantly and is the number
    that tells the user how much of their machine is currently free.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None and not refresh:
        return _DEVICE_CACHE

    logical = os.cpu_count() or 1
    physical = logical
    total = available = 0.0
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            total = vm.total / GB
            available = vm.available / GB
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            physical = psutil.cpu_count(logical=False) or logical
        except Exception:  # pragma: no cover - defensive
            physical = logical
    if total <= 0:
        # No psutil, or a platform it cannot read. Assume the minimum rather
        # than something optimistic: under-using a machine is a performance
        # problem, over-committing one is a crash.
        total = available = MIN_RAM_GB

    _DEVICE_CACHE = DeviceCapabilities(
        total_ram_gb=round(total, 2),
        available_ram_gb=round(available, 2),
        logical_cores=int(logical),
        physical_cores=int(physical),
        gpus=_probe_gpus(),
    )
    return _DEVICE_CACHE


def feasible_range(device: Optional[DeviceCapabilities] = None) -> Dict[str, Any]:
    """
    The min/max the settings UI should offer, per field.

    Cores are bounded by the LOGICAL count while the default is derived from the
    physical count: hyperthreads help the filter passes (which are memory-bound
    but not memory-latency-bound) and hurt nothing except by consuming a block's
    worth of RAM each, which the planner already accounts for. So the user may
    opt into them; they are not chosen for them.
    """
    dev = device or probe_device()
    return {
        "ram_gb": (MIN_RAM_GB, max(MIN_RAM_GB, round(dev.total_ram_gb, 2))),
        "vram_gb": (0.0, round(dev.total_vram_gb, 2)),
        "cores": (1, max(1, dev.logical_cores)),
    }


# --------------------------------------------------------------------------
# The setting
# --------------------------------------------------------------------------

@dataclass
class ResourceSettings:
    """
    The three numbers, plus how they were arrived at.

    `origin` is recorded so a log can say whether a run was sized by a value the
    user typed, by the first-run default, or by an environment override -- which
    is the first question worth asking when two machines disagree about
    throughput.
    """

    ram_gb: float
    vram_gb: float
    cores: int
    origin: str = "default"

    @property
    def ram_bytes(self) -> int:
        return int(self.ram_gb * GB)

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gb * GB)

    @property
    def gpu_enabled(self) -> bool:
        """GPU work is enabled by giving it a VRAM budget above zero.

        One switch, not two. A `gpu_enabled` flag alongside a VRAM number can
        disagree with itself, and the disagreement is silent.

        NOTE: no step performs GPU computation yet, so this currently gates
        nothing. It is here so the setting exists and is persisted before the
        first GPU path lands, rather than being retrofitted afterwards.
        """
        return self.vram_gb > 0.0

    def clamped(self, device: Optional[DeviceCapabilities] = None) -> "ResourceSettings":
        """
        This setting, forced inside what the machine can actually do.

        Applied on load as well as on save, because a settings file travels: a
        project folder synced between the 68 GB workstation and the 32 GB laptop
        would otherwise hand the laptop a 60 GB ceiling. Clamping DOWN silently
        is right (the machine is the authority on its own size); clamping up is
        not, so a deliberately small ceiling is left alone.
        """
        rng = feasible_range(device)
        ram_lo, ram_hi = rng["ram_gb"]
        vram_lo, vram_hi = rng["vram_gb"]
        core_lo, core_hi = rng["cores"]
        return ResourceSettings(
            ram_gb=float(min(max(self.ram_gb, ram_lo), ram_hi)),
            vram_gb=float(min(max(self.vram_gb, vram_lo), vram_hi)),
            cores=int(min(max(int(self.cores), core_lo), core_hi)),
            origin=self.origin,
        )


def _default_settings(device: Optional[DeviceCapabilities] = None) -> ResourceSettings:
    """
    What a machine that has never been configured gets.

    A share of physical RAM, minus a fixed headroom for everything else on the
    machine, whichever is smaller. On 8 GB that yields ~4 GB rather than 4.8,
    because the headroom term dominates on small machines -- which is the case
    the pipeline has to survive. On 68 GB it yields ~41 GB, which is a large
    increase over the previous hardcoded behaviour, so the first run after this
    lands will be noticeably faster and noticeably hungrier. That is the point,
    and it is visible in the log line `describe_environment` writes.

    Cores default to the PHYSICAL count minus two, preserving the intent of the
    `os.cpu_count() - 2` that the enhancement pass used -- leave the machine
    usable while a long run is going.
    """
    dev = device or probe_device()
    by_fraction = dev.total_ram_gb * _DEFAULT_RAM_FRACTION
    by_headroom = dev.total_ram_gb - _DEFAULT_RAM_HEADROOM_GB
    ram = max(MIN_RAM_GB, min(by_fraction, by_headroom))
    return ResourceSettings(
        ram_gb=round(ram, 2),
        # Zero, not the card's capacity: no step computes on the GPU yet, and a
        # non-zero default would advertise an acceleration that does not exist.
        vram_gb=0.0,
        cores=max(1, dev.physical_cores - 2),
        origin="default",
    ).clamped(dev)


def _env_override() -> Optional[ResourceSettings]:
    """
    Settings from ``HIBACHI_RAM_GB`` / ``HIBACHI_VRAM_GB`` / ``HIBACHI_CORES``.

    Exists for the regression gate. The property worth testing is that a run at
    a low ceiling and a run at a high ceiling produce BYTE-IDENTICAL artifacts,
    and that test needs to set a ceiling without editing the user's saved
    preference. Any subset may be given; whatever is absent falls back to the
    saved value, so ``HIBACHI_RAM_GB=2`` is enough to run the low-memory leg.
    """
    keys = ("HIBACHI_RAM_GB", "HIBACHI_VRAM_GB", "HIBACHI_CORES")
    if not any(k in os.environ for k in keys):
        return None
    base = _read_settings_file() or _default_settings()
    out = ResourceSettings(base.ram_gb, base.vram_gb, base.cores, "environment")
    try:
        if "HIBACHI_RAM_GB" in os.environ:
            out.ram_gb = float(os.environ["HIBACHI_RAM_GB"])
        if "HIBACHI_VRAM_GB" in os.environ:
            out.vram_gb = float(os.environ["HIBACHI_VRAM_GB"])
        if "HIBACHI_CORES" in os.environ:
            out.cores = int(os.environ["HIBACHI_CORES"])
    except ValueError as exc:
        print(f"[resources] ignoring malformed environment override: {exc}")
        return None
    return out


def _read_settings_file() -> Optional[ResourceSettings]:
    """The saved settings, or None. Corruption-tolerant, like RecentProjects."""
    try:
        with open(settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ResourceSettings(
            ram_gb=float(data["ram_gb"]),
            vram_gb=float(data.get("vram_gb", 0.0)),
            cores=int(data["cores"]),
            origin="saved",
        )
    except (KeyError, TypeError, ValueError):
        # A partially-written or hand-edited file. Treated as absent rather than
        # repaired: a half-understood ceiling is worse than a fresh default,
        # which at least matches the machine.
        print(f"[resources] {settings_path()} is not readable as settings; "
              "using defaults.")
        return None


def load_settings(device: Optional[DeviceCapabilities] = None) -> ResourceSettings:
    """
    The settings in force: environment override, else saved file, else default.

    Always clamped to the machine, so a caller never has to check.
    """
    dev = device or probe_device()
    return (_env_override() or _read_settings_file()
            or _default_settings(dev)).clamped(dev)


def save_settings(settings: ResourceSettings,
                  device: Optional[DeviceCapabilities] = None) -> str:
    """
    Persist, clamped, atomically. Returns the path written.

    Written to a temporary file and renamed, so an interrupted save cannot leave
    a truncated JSON file that reads as "never configured" and silently reverts
    the user's ceiling.
    """
    dev = device or probe_device()
    final = settings.clamped(dev)
    payload = {
        "ram_gb": round(final.ram_gb, 3),
        "vram_gb": round(final.vram_gb, 3),
        "cores": int(final.cores),
        # Recorded for the reader's benefit only. Nothing reads them back: they
        # are what the machine looked like when the choice was made, which is
        # the context needed to understand a number that now looks odd.
        "_measured_total_ram_gb": dev.total_ram_gb,
        "_measured_logical_cores": dev.logical_cores,
        "_measured_gpu": dev.gpu_name,
        "_measured_total_vram_gb": dev.total_vram_gb,
        "_written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# Geometry the budget must not size
# --------------------------------------------------------------------------

#: Shapes and extents that LOOK like memory parameters but change results.
#: Each entry is (value, reason). `pinned()` is the only way a step should read
#: one, so that the reason travels with the number and a later reader cannot
#: mistake it for something the budget forgot to scale.
#:
#: The distinguishing test, applied to each: is the operator computed with
#: enough halo that the answer equals the whole-array answer? If yes it belongs
#: in the planner. If the answer depends on where the boundary falls, it is here.
PINNED: Dict[str, Tuple[Any, str]] = {
    "soma_tile_shape_3d": (
        (128, 512, 512),
        "Soma extraction computes max_dt_val per tile and thresholds the "
        "distance transform at max_dt_val * ratio, so every DT strategy's "
        "threshold is a function of the tile extent. It also attributes a "
        "detection to the tile whose target region contains its centroid. "
        "Changing the tile size changes which somata are found.",
    ),
    "soma_tile_shape_2d": (
        (2048, 2048),
        "As soma_tile_shape_3d, at rank 2.",
    ),
    "split_chunk_shape_3d": (
        (128, 512, 512),
        "Step 4's chunk geometry decides which cells are judged inside a "
        "chunk (extent <= overlap + 1) and which are deferred to the global "
        "merge pass, and the stitcher's conflict resolution runs only where "
        "chunks overlap. Two machines with different chunk shapes would take "
        "different merge decisions.",
    ),
    "split_chunk_shape_2d": (
        (1024, 1024),
        "As split_chunk_shape_3d, at rank 2.",
    ),
    "split_overlap": (
        64,
        "Same reason as split_chunk_shape_*, and it is the parameter that "
        "directly encodes the inside-chunk/deferred boundary.",
    ),
    "dask_chunk_shape_3d": (
        (128, 512, 512),
        "One function supplies this shape to dask_image's gaussian_filter, "
        "binary_closing AND ndmeasure.label. The filters go through "
        "map_overlap with a sigma-derived depth and SHOULD be chunk-invariant, "
        "but `label` provably is not: it numbers each block independently and "
        "then resolves equivalences, so the partition is chunk-independent "
        "while the IDs are not. Step 1 uses those IDs structurally (soma_lut, "
        "the size/seed filter's 1..maxid walk), and step 3's peak grid "
        "resolves cross-label soma conflicts in ascending label order -- so a "
        "renumbering changes which somata survive, not just their names. "
        "Rather than split this on an unverified assumption about dask_image's "
        "depth, the shape is fixed at both ranks and only the scheduler's "
        "concurrency is scaled.",
    ),
    "dask_chunk_shape_2d": (
        (2048, 2048),
        "As dask_chunk_shape_3d, at rank 2. The 8x footprint difference "
        "between the ranks is inherited from the pre-merge tracks and is "
        "deliberately NOT unified: doing so would renumber one rank's labels.",
    ),
    "relabel_chunk_shape_3d": (
        (128, 256, 256),
        "Step 2's fragment filter feeds this to dask_image.ndmeasure.label, so "
        "the same renumbering argument as dask_chunk_shape_3d applies: the "
        "partition is chunk-independent, the IDs are not, and step 3's peak "
        "grid resolves cross-label soma conflicts in ascending label order. "
        "Note the value differs from dask_chunk_shape_3d -- step 1 and step 2 "
        "chose different shapes and both are baked into existing results, so "
        "they are pinned separately rather than unified.",
    ),
    "relabel_chunk_shape_2d": (
        (2048, 2048),
        "As relabel_chunk_shape_3d, at rank 2.",
    ),
    "threshold_sample_stride_inplane_max": (
        16,
        "Step 1 estimates its percentile thresholds from a strided sample. "
        "The sampled SET is the estimate, so changing the stride changes "
        "every threshold. The memory cost of this sample is fixed instead by "
        "streaming it to disk and taking the percentile out of core, which "
        "leaves the sampled set -- and therefore the threshold -- identical.",
    ),
    "threshold_sample_stride_leading_max": (
        4,
        "As threshold_sample_stride_inplane_max, along the leading axis.",
    ),
}


def pinned_reason(name: str) -> str:
    """Why `name` is pinned. For the settings UI and for log detail on demand."""
    try:
        return PINNED[name][1]
    except KeyError:
        raise KeyError(f"{name!r} is not a pinned geometry.") from None


def pinned(name: str) -> Any:
    """
    A pinned value, by name. Raises on an unknown name rather than returning a
    default, so a typo is a startup error and not a silently different result.
    """
    try:
        return PINNED[name][0]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a pinned geometry. Known: "
            f"{', '.join(sorted(PINNED))}. If this is a memory-only shape it "
            "belongs in the planner, not here."
        ) from None


# --------------------------------------------------------------------------
# What an operation costs
# --------------------------------------------------------------------------

#: Bytes of working set per element, per named operation. These are the numbers
#: the planner divides the budget by, so a wrong one here is a wrong block size
#: everywhere -- each is derived from the arrays the operation demonstrably
#: allocates, not estimated.
#:
#: "Working set" means everything live at the same time for one block: the input
#: cast, the output, and the temporaries the library allocates internally.
_COST: Dict[str, float] = {
    # A pass that reads a block and writes a same-shaped result: input cast to
    # float32, output float32, one temporary. Covers the merge/threshold passes.
    "float32_pass": 12.0,
    # scipy's distance_transform_edt returns float64 and, for the distance-only
    # form, allocates an int32 index workspace per axis plus the boolean input.
    # 8 (out) + 4*ndim (workspace) + 1 (input) at rank 3 = 21, rounded up for
    # the copy the sampling argument forces.
    "edt_3d": 24.0,
    "edt_2d": 16.0,
    # Vesselness: the block is held as float32 and each plane's Frangi/Sato
    # response, crest weight and grey dilation are plane-sized rather than
    # block-sized, so the block term dominates. Input float32 + owned-region
    # output float32 + two plane temporaries amortised over a 64-plane block.
    "vesselness": 10.0,
    # Binary morphology on a labelled block: int32 in, bool mask, bool result.
    "binary_morphology": 6.0,
    # The streaming label/interface accumulators: int32 labels plus an
    # intensity plane and the forward-offset comparison temporaries.
    "label_statistics": 10.0,
    # A plain copy between two same-dtype arrays, e.g. writing a checkpoint.
    "copy_int32": 8.0,
    # One step-4 chunk in flight, per voxel of the chunk. The chunk-level
    # arrays are the labels (int32, 4), the intensity crop (uint16, 2), the
    # soma crop (int32, 4), the copy of the prior mask (int32, 4) and the
    # result (int32, 4) = 18. Inside it, one cell's crop can span the whole
    # chunk, and that path holds the cell mask (bool, 1), the markers (int32,
    # 4), the watershed labels and the merged copy (int32, 4+4), and five
    # float64 fields -- the two distance transforms, the speed field, the
    # normalised intensity and the landscape (8 x 5 = 40) = 53. The per-cell
    # crop cache is budgeted separately by the caller and is not counted here.
    "cell_separation_chunk": 72.0,
}


def cost_bytes_per_voxel(operation: str) -> float:
    """
    Working-set cost of a named operation, in bytes per element.

    Named rather than passed as a number at each call site, so the estimate for
    "an EDT" is stated once and can be corrected once. An unknown name raises:
    guessing a cost is how a planner ends up confidently over-committing.
    """
    try:
        return _COST[operation]
    except KeyError:
        raise KeyError(
            f"unknown operation cost {operation!r}. Known: "
            f"{', '.join(sorted(_COST))}. Add it to _COST with the derivation "
            "of the number, rather than passing a literal here."
        ) from None


# --------------------------------------------------------------------------
# Runtime brake
# --------------------------------------------------------------------------

class Monitor:
    """
    Watches actual memory use and says when a step should back off.

    Sampled rather than continuous: reading ``virtual_memory()`` is cheap but
    not free, and a block loop can run thousands of iterations, so consecutive
    checks inside ``_MIN_SAMPLE_INTERVAL`` reuse the last reading.

    The brake never changes geometry on its own. It reports pressure; the step
    decides what to shed, and a step whose geometry is pinned may only shed
    concurrency. That asymmetry is the whole reason `BlockPlan.adaptive` exists.
    """

    _MIN_SAMPLE_INTERVAL = 0.25   # seconds

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = int(budget_bytes)
        self._last_sample = 0.0
        self._available = float("inf")
        self._tree_rss = 0
        self.brake_events = 0

    # -- measurement ----------------------------------------------------
    def _sample(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_sample) < self._MIN_SAMPLE_INTERVAL:
            return
        self._last_sample = now
        if not _HAS_PSUTIL:
            return
        try:
            self._available = float(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - defensive
            self._available = float("inf")
        try:
            proc = psutil.Process()
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    continue
            self._tree_rss = int(total)
        except Exception:  # pragma: no cover - defensive
            self._tree_rss = 0

    def system_available_bytes(self) -> float:
        """Memory the OS says is available WITHOUT reclaiming anything."""
        self._sample()
        return self._available

    def tree_rss_bytes(self) -> int:
        """Resident set of this process and its children. Diagnostics only.

        Includes page-cache pages backing every memmap the pipeline has touched,
        so on a healthy streaming step this number climbs toward the size of the
        files on disk and means nothing about pressure. Do not brake on it.
        """
        self._sample()
        return self._tree_rss

    # -- control --------------------------------------------------------
    def pressure(self) -> float:
        """
        How close the machine is to the point of failure, in [0, inf).

        1.0 means the system has exactly `MIN_SYSTEM_FREE_GB` left, which is the
        line the budget promised not to cross. Above 1.0 the step is eating into
        the reserve that belongs to everything else on the machine.
        """
        avail = self.system_available_bytes()
        if not math.isfinite(avail):
            return 0.0
        floor = MIN_SYSTEM_FREE_GB * GB
        if avail <= 0:
            return float("inf")
        return floor / avail

    def should_shed(self) -> bool:
        """
        True when the step should reduce concurrency and collect before going on.

        Two independent triggers, because they catch different failures:
        the system floor catches everything else on the machine growing (the
        browser), and the budget watermark catches our own estimate having been
        too optimistic for this particular data.
        """
        self._sample()
        if math.isfinite(self._available) and \
                self._available < MIN_SYSTEM_FREE_GB * GB:
            return True
        if self.budget_bytes > 0 and self._tree_rss > 0:
            return self._tree_rss > self.budget_bytes * BRAKE_WATERMARK
        return False

    def note_brake(self, what: str, log=print) -> None:
        """Record and report one braking event, so the log explains a slowdown."""
        self.brake_events += 1
        log(f"  [resources|BRAKE] {what} "
            f"(system available {self.system_available_bytes() / GB:.2f} GB, "
            f"tree RSS {self.tree_rss_bytes() / GB:.2f} GB, "
            f"budget {self.budget_bytes / GB:.2f} GB)")


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

@dataclass
class BlockPlan:
    """
    A sized traversal: how big a block, and how many at once.

    `fits` is False when even the smallest permitted block exceeds the budget.
    The plan is still returned, with the smallest block and one worker, because
    refusing to return anything would leave the caller with nothing to report;
    it is the caller's job to warn or abort. `note` says why.

    `adaptive` records whether the brake is allowed to shrink the geometry
    mid-run. False for anything derived from `PINNED`: shedding workers is
    always safe, changing a pinned extent is not.
    """

    block_shape: Tuple[int, ...]
    workers: int
    overlap: int = 0
    bytes_per_block: int = 0
    adaptive: bool = True
    fits: bool = True
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return int(self.bytes_per_block * max(1, self.workers))

    def describe(self) -> str:
        state = "" if self.fits else "  *** DOES NOT FIT ***"
        return (f"block={self.block_shape} workers={self.workers} "
                f"overlap={self.overlap} "
                f"peak={self.total_bytes / GB:.2f} GB"
                f"{(' -- ' + self.note) if self.note else ''}{state}")


class Budget:
    """
    A budget scoped to one step.

    Constructed per step rather than once per process, because the amount of
    room available depends on what the application is already holding: a napari
    session with four label layers open has less to give than a headless batch
    run, and the user asked for the ceiling to cover the whole app. So the
    budget measures the current footprint at construction and plans inside what
    is LEFT, instead of assuming it has the whole ceiling to itself.
    """

    def __init__(self, settings: ResourceSettings, step: str = "",
                 log=print) -> None:
        self.settings = settings
        self.step = step
        self._log = log
        self.monitor = Monitor(settings.ram_bytes)

        baseline = self.monitor.tree_rss_bytes()
        # The baseline is the app's own footprint, but on Linux it also contains
        # page-cache pages for every artifact already touched, which are
        # reclaimable and must not be treated as spent. Charging the larger of
        # a fixed floor and a quarter of the measured baseline is the compromise:
        # it reserves real room for the GUI without letting a big memmap read
        # earlier in the session consume the entire budget.
        self.reserved_bytes = int(max(0.25 * GB, 0.25 * baseline))
        self.plannable_bytes = max(
            int(0.5 * GB), settings.ram_bytes - self.reserved_bytes
        )
        self.cores = max(1, int(settings.cores))

    # -- properties -----------------------------------------------------
    @property
    def gpu_enabled(self) -> bool:
        return self.settings.gpu_enabled

    @property
    def vram_bytes(self) -> int:
        return self.settings.vram_bytes

    # -- planning -------------------------------------------------------
    def plan_spans(
        self,
        shape: Sequence[int],
        operation: str,
        *,
        overlap: int = 0,
        axis: int = 0,
        min_extent: int = 1,
        max_workers: Optional[int] = None,
        share: float = 1.0,
        name: str = "",
    ) -> BlockPlan:
        """
        Size a traversal that walks `axis` in spans, taking the other axes whole.

        This is the shape of most of the pipeline's loops: a run of Z planes in
        3D, a run of rows in 2D, with the in-plane extent left full because the
        operator's statistics or its halo require it. The span is grown to fill
        the budget and the worker count is chosen with it, which is the thing
        the old hardcoded constants could not do -- a chunk size and a CPU count
        chosen independently multiply into a peak nobody computed.

        `overlap` is the halo in elements on each side along `axis`. It is a
        correctness input, never a budget output: the planner spends memory to
        satisfy it and reduces the span if it cannot, but it never shrinks the
        halo, because that is what would change the result.

        Worker count and span are chosen together by maximising
        ``workers * span / (span + 2 * overlap)`` -- parallelism discounted by
        the fraction of each block that is redundant halo computation. That
        prefers a few large blocks when the halo is deep (an EDT with a 70-plane
        margin) and many small ones when it is shallow, which is the correct
        trade in both directions and is not a preference anyone has to set.
        """
        shape = tuple(int(v) for v in shape)
        nd = len(shape)
        axis = int(axis) % nd
        per_element = cost_bytes_per_voxel(operation)
        cross = 1
        for k in range(nd):
            if k != axis:
                cross *= shape[k]
        budget = max(int(0.25 * GB), int(self.plannable_bytes * float(share)))
        axis_len = shape[axis]
        cap = max(1, int(max_workers) if max_workers else self.cores)
        cap = min(cap, self.cores)

        best: Optional[Tuple[float, int, int]] = None   # (score, workers, span)
        for workers in range(cap, 0, -1):
            per_worker = budget // workers
            # Two independent caps on the span, and the smaller wins.
            #
            # `span_mem` is what one worker's share of the budget pays for.
            # `span_bal` is what keeps every worker busy: with W workers there
            # must be at least W blocks, or concurrency is wasted on a queue
            # that runs dry. Taking the largest span that merely FITS is the
            # trap -- it produces two enormous blocks, three workers idle, and a
            # slower run than a smaller span would have given.
            span_mem = int(per_worker / (per_element * cross)) - 2 * overlap
            span_bal = math.ceil(axis_len / workers)
            span = min(span_mem, span_bal)
            span = max(1, min(axis_len, span))
            if span < min_extent and workers > 1:
                continue
            # A plan that cannot hold its own blocks is not a plan. Only the
            # single-worker case is allowed through, because that is the honest
            # "this does not fit at all" report the caller has to see.
            if workers > 1 and \
                    per_element * cross * (span + 2 * overlap) * workers > budget:
                continue
            efficiency = span / float(span + 2 * overlap)
            n_blocks = math.ceil(axis_len / span)
            useful = min(workers, n_blocks)
            score = useful * efficiency
            if best is None or score > best[0]:
                best = (score, useful, span)

        if best is None:
            best = (0.0, 1, max(1, min(axis_len, min_extent)))
        _score, workers, span = best
        block = list(shape)
        block[axis] = span
        padded = span + 2 * overlap
        bytes_per_block = int(per_element * cross * padded)
        # No `or span <= 1` escape here. When a single one-element-deep span
        # still exceeds the budget -- a brain cross-section under an EDT, say --
        # the answer is that this operator cannot run as a span traversal on
        # this machine, and saying otherwise would hand the caller a plan that
        # reports success and then dies. The caller must either warn or, better,
        # re-express the operator as a haloed tile traversal
        # (`plan_scaled_block`), which is exact whenever the halo covers the
        # operator's reach.
        fits = (bytes_per_block * workers) <= budget
        note = name or operation
        if not fits:
            note += (f"; a single {span}-element span of this cross-section "
                     f"(plus {overlap} halo either side) needs "
                     f"{bytes_per_block / GB:.2f} GB, above the "
                     f"{budget / GB:.2f} GB available to this step")
        return BlockPlan(
            block_shape=tuple(block), workers=workers, overlap=overlap,
            bytes_per_block=bytes_per_block, adaptive=True, fits=fits, note=note,
        )

    def plan_scaled_block(
        self,
        shape: Sequence[int],
        base_block: Sequence[int],
        operation: str,
        *,
        overlap: int = 0,
        max_workers: Optional[int] = None,
        share: float = 1.0,
        max_factor: int = 16,
        name: str = "",
    ) -> BlockPlan:
        """
        Size a grid traversal by scaling `base_block` by an integer factor.

        For the loops that tile every axis rather than walking one: the dask
        chunking, the streaming statistics blocks, the vesselness grid. Scaling a
        base shape by a whole factor rather than solving for each axis keeps the
        block's aspect ratio, which matters because these shapes were chosen so
        that the in-plane extent covers a filter's reach and the leading extent
        is the cheap axis to grow.

        The factor is applied to every axis and then clipped to the array, so a
        thin stack does not get a block deeper than it is.
        """
        shape = tuple(int(v) for v in shape)
        base = tuple(int(v) for v in base_block)
        if len(base) != len(shape):
            base = (base[-len(shape):] if len(base) > len(shape)
                    else (base[0],) * (len(shape) - len(base)) + base)
        per_element = cost_bytes_per_voxel(operation)
        budget = max(int(0.25 * GB), int(self.plannable_bytes * float(share)))
        cap = min(max(1, int(max_workers) if max_workers else self.cores),
                  self.cores)

        total_volume = 1
        for v in shape:
            total_volume *= v

        best: Optional[Tuple[float, int, Tuple[int, ...]]] = None
        for workers in range(cap, 0, -1):
            per_worker = budget // workers
            # Same two caps as `plan_spans`, expressed as volumes: what one
            # worker's budget share pays for, and what leaves at least one block
            # per worker. Without the second, the largest fitting factor wins and
            # a 20038^2 image gets two 18432^2 blocks -- most of the machine idle
            # and most of each block redundant halo.
            volume_balance = max(1.0, total_volume / float(workers))
            chosen: Optional[Tuple[int, ...]] = None
            for factor in range(max_factor, 0, -1):
                cand = tuple(min(shape[k], base[k] * factor)
                             for k in range(len(shape)))
                padded = 1
                volume = 1
                for v in cand:
                    padded *= (v + 2 * overlap)
                    volume *= v
                if per_element * padded > per_worker:
                    continue
                if volume > volume_balance and factor > 1:
                    continue
                chosen = cand
                break
            if chosen is None:
                continue
            n_blocks = 1
            for k in range(len(shape)):
                n_blocks *= math.ceil(shape[k] / chosen[k])
            useful = min(workers, n_blocks)
            volume = 1
            for v in chosen:
                volume *= v
            padded = 1
            for v in chosen:
                padded *= (v + 2 * overlap)
            score = useful * (volume / float(padded))
            if best is None or score > best[0]:
                best = (score, useful, chosen)

        if best is None:
            # Nothing fits, not even one base block. Report the base block so
            # the caller behaves exactly as it did before this module existed.
            padded = 1
            for v in base:
                padded *= (v + 2 * overlap)
            return BlockPlan(
                block_shape=tuple(min(shape[k], base[k])
                                  for k in range(len(shape))),
                workers=1, overlap=overlap,
                bytes_per_block=int(per_element * padded), adaptive=True,
                fits=False,
                note=(f"{name or operation}; the base block {base} needs "
                      f"{per_element * padded / GB:.2f} GB, above the "
                      f"{budget / GB:.2f} GB available to this step"),
            )

        _score, workers, chosen = best
        padded = 1
        for v in chosen:
            padded *= (v + 2 * overlap)
        return BlockPlan(
            block_shape=chosen, workers=workers, overlap=overlap,
            bytes_per_block=int(per_element * padded), adaptive=True,
            fits=True, note=name or operation,
        )

    def plan_pinned(
        self,
        shape: Sequence[int],
        pinned_name: str,
        operation: str,
        *,
        overlap: int = 0,
        max_workers: Optional[int] = None,
        share: float = 1.0,
        name: str = "",
    ) -> BlockPlan:
        """
        A plan whose GEOMETRY comes from `PINNED` and whose CONCURRENCY does not.

        This is how a result-affecting shape still participates in the budget:
        the extent is fixed, so the answer cannot move between machines, but the
        number of those blocks in flight at once is sized to the budget. On the
        68 GB workstation that is where the speed comes from; on an 8 GB laptop
        it is what keeps the step inside its ceiling.

        `adaptive=False` on the returned plan, so a brake can only shed workers.
        """
        shape = tuple(int(v) for v in shape)
        block = tuple(pinned(pinned_name))
        if len(block) != len(shape):
            block = (block[-len(shape):] if len(block) > len(shape)
                     else (block[0],) * (len(shape) - len(block)) + block)
        block = tuple(min(shape[k], block[k]) for k in range(len(shape)))
        per_element = cost_bytes_per_voxel(operation)
        padded = 1
        for v in block:
            padded *= (v + 2 * overlap)
        per_block = int(per_element * padded)
        budget = max(int(0.25 * GB), int(self.plannable_bytes * float(share)))
        cap = min(max(1, int(max_workers) if max_workers else self.cores),
                  self.cores)
        workers = max(1, min(cap, budget // max(1, per_block)))
        n_blocks = 1
        for k in range(len(shape)):
            n_blocks *= math.ceil(shape[k] / block[k])
        workers = int(min(workers, max(1, n_blocks)))
        fits = per_block <= budget
        # The reason is deliberately NOT interpolated into the note: it is a
        # paragraph, and a log line per block loop carrying a paragraph is a log
        # nobody reads. `pinned_reason` is there for the settings UI and for
        # anyone asking why this one is not scaled.
        note = name or f"{pinned_name} (pinned geometry)"
        if not fits:
            note += (f"; one block needs {per_block / GB:.2f} GB, above the "
                     f"{budget / GB:.2f} GB available to this step")
        return BlockPlan(
            block_shape=block, workers=workers, overlap=overlap,
            bytes_per_block=per_block, adaptive=False, fits=fits, note=note,
        )

    # -- reporting ------------------------------------------------------
    def report(self, plan: BlockPlan) -> BlockPlan:
        """
        Log a plan and hand it back, so a call site can wrap and stay one line.

        Every step logs its plan. Without it, "why was this run slower on the
        other machine" has no answer in the log, and the whole point of a budget
        is that its effect is visible.
        """
        self._log(f"  [resources] {self.step or 'step'}: {plan.describe()}")
        if not plan.fits:
            self._log(
                "  [resources] *** the smallest permitted block for this step "
                "does not fit the configured RAM ceiling. It will be attempted "
                "anyway and may fail or swap. Raise the ceiling in Settings, or "
                "process a smaller region. ***"
            )
        return plan


def open_budget(step: str = "", log=print,
                settings: Optional[ResourceSettings] = None) -> Budget:
    """
    The budget for one step. The normal entry point.

    Takes `settings` only for tests; in the application it always reads the
    setting in force, so a change made in the dialog applies to the next step
    without anything having to be plumbed through.
    """
    return Budget(settings or load_settings(), step=step, log=log)


# --------------------------------------------------------------------------
# Adaptive traversal
# --------------------------------------------------------------------------

def iter_leading_spans(
    shape: Sequence[int],
    plan: BlockPlan,
    monitor: Optional[Monitor] = None,
    axis: int = 0,
    log=print,
) -> Iterator[Tuple[Tuple[slice, ...], Tuple[slice, ...]]]:
    """
    Walk `axis` in spans, yielding ``(read_slices, write_slices)``.

    The write regions tile the array exactly once; the read region is the write
    region grown by ``plan.overlap`` and clipped. Same contract as
    `dim_utils.chunk_read_write_slices`, so a step can swap one for the other,
    but the span is read from a mutable local on every iteration rather than
    fixed up front. That is what lets the brake take effect: when the monitor
    reports pressure, the span for the REMAINING blocks is halved.

    Halving the span cannot change the result, because the halo is preserved and
    the write regions still tile the array exactly once -- which is precisely the
    property that qualified this geometry as budget-driven in the first place. On
    a plan with ``adaptive=False`` the span is left alone and the brake is
    reported but not acted on here; such a step must shed concurrency instead.
    """
    shape = tuple(int(v) for v in shape)
    nd = len(shape)
    axis = int(axis) % nd
    overlap = max(0, int(plan.overlap))
    span = max(1, int(plan.block_shape[axis]))
    length = shape[axis]

    start = 0
    while start < length:
        stop = min(start + span, length)
        read = []
        write = []
        for k in range(nd):
            if k == axis:
                write.append(slice(start, stop))
                read.append(slice(max(0, start - overlap),
                                  min(length, stop + overlap)))
            else:
                write.append(slice(0, shape[k]))
                read.append(slice(0, shape[k]))
        yield tuple(read), tuple(write)
        start = stop

        if monitor is not None and monitor.should_shed():
            if plan.adaptive and span > 1:
                span = max(1, span // 2)
                monitor.note_brake(
                    f"span along axis {axis} reduced to {span}", log=log)
            else:
                monitor.note_brake(
                    "under pressure but this geometry is pinned; "
                    "shed concurrency instead", log=log)
            import gc
            gc.collect()


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def describe_environment(settings: Optional[ResourceSettings] = None) -> str:
    """
    One block of text: what the machine has, and what it has been allowed.

    Printed once at the start of a run. This is the line that makes two machines
    comparable -- and the line that tells a user their first run after an update
    is using more memory than the last one, and why.
    """
    dev = probe_device(refresh=True)
    cfg = settings or load_settings(dev)
    gpu = (f"{dev.gpu_name} ({dev.total_vram_gb:.1f} GB)"
           if dev.gpu_name else "none detected")
    gpu_use = (f"{cfg.vram_gb:.1f} GB" if cfg.gpu_enabled
               else "disabled (no step computes on the GPU yet)")
    return (
        "  [resources] machine: "
        f"{dev.total_ram_gb:.1f} GB RAM ({dev.available_ram_gb:.1f} free), "
        f"{dev.physical_cores} physical / {dev.logical_cores} logical cores, "
        f"GPU: {gpu}\n"
        "  [resources] budget:  "
        f"{cfg.ram_gb:.1f} GB RAM, {cfg.cores} cores, VRAM {gpu_use} "
        f"[{cfg.origin}]\n"
        f"  [resources] settings file: {settings_path()}"
    )


# --------------------------------------------------------------------------
# Cooperative cancellation
# --------------------------------------------------------------------------
# Here rather than in a module of its own because it is the other half of the
# same question -- a budget says how much of the machine a run may use, this
# says for how long -- and because every step already imports this module, so a
# cancellation point costs no new import.
#
# Why it is needed: `gui_manager._stop_worker_safely` used to stop a step by
# killing the child processes it had spawned, on the stated assumption that the
# worker thread would then unwind on its own. That holds only for steps that
# USE a pool. Steps 2 and 4 are single-threaded and in-process, so there were no
# children to kill: the thread was detached and ran the entire step to
# completion, holding a core and several GB long after the window had closed.
#
# `QThread.terminate()` is not the alternative. It stops a thread at an
# arbitrary instruction, which halfway through writing a memmap leaves a
# truncated artifact and inside a C extension aborts the process. Hence a
# cooperative flag, checked at boundaries the step chooses.
#
# Usage, once per chunk / tile / window -- never per voxel:
#
#     resource_budget.check_cancelled()
#
# A cancellation point must be somewhere stopping is SAFE: every artifact
# written so far either complete, or about to be deleted by the caller. Do not
# put one between two writes that must both land.
#
# Process-global, deliberately: one window runs one step at a time, and
# threading a token through five modules' call signatures would touch far more
# code than the problem is worth. It is not inherited by multiprocessing
# children -- those are still stopped by killing them, which stays correct.

class ProcessingCancelled(Exception):
    """Raised by `check_cancelled()` when the user has asked the current step to stop.

    Caught by `StepWorker.run`, which reports it as a cancellation rather than
    an error -- it is not a failure, and it should not raise an error dialog or
    a crash report.
    """


_cancelled = threading.Event()


def request_cancel() -> None:
    """Ask the running step to stop at its next cancellation point."""
    _cancelled.set()


def clear_cancel() -> None:
    """Reset before starting a step.

    Called when a step is launched, not when one finishes: a step that ended by
    being cancelled must leave the flag set until the next one starts, or a
    second step queued behind it would run with a stale request.
    """
    _cancelled.clear()


def cancel_requested() -> bool:
    """True when cancellation has been asked for. Does not raise."""
    return _cancelled.is_set()


def check_cancelled() -> None:
    """Raise `ProcessingCancelled` if cancellation has been requested."""
    if _cancelled.is_set():
        raise ProcessingCancelled("processing cancelled by the user")


if __name__ == "__main__":  # pragma: no cover - manual inspection
    # `python -m ...resource_budget` prints what this machine would be given
    # and what a few representative steps would be planned at. Useful for
    # checking a ceiling before starting a long run.
    print(describe_environment())
    demo_shape = (512, 8192, 8192)
    b = open_budget("demo")
    print(f"  [resources] plannable this step: "
          f"{b.plannable_bytes / GB:.2f} GB (reserved "
          f"{b.reserved_bytes / GB:.2f} GB for the app)")
    b.report(b.plan_spans(demo_shape, "edt_3d", overlap=8, name="per-plane EDT"))
    b.report(b.plan_scaled_block(demo_shape, (64, 512, 512), "vesselness",
                                 overlap=32, name="vesselness grid"))
    b.report(b.plan_pinned(demo_shape, "split_chunk_shape_3d",
                           "binary_morphology", overlap=pinned("split_overlap"),
                           name="step 4 chunks"))
    sys.exit(0)