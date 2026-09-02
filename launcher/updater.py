"""
updater.py -- safe, offline-tolerant git self-update for the HIBACHI checkout.

Design principles (see INSTALL.md for the full rationale):

* The repository is treated as *disposable application code*. All user data
  (projects, per-project configs, outputs) and scratch/temp files live OUTSIDE
  the repository, so the checkout can be fast-forwarded / reset freely.
* Updating must NEVER prevent the app from launching. Any failure (no network,
  git error, detached HEAD, ...) is caught and reported, and the caller
  proceeds to launch whatever version is currently on disk.
* If the working tree has local modifications (e.g. a power-user hand-edited a
  shipped file), those changes are preserved in a timestamped `git stash`
  before the update, so nothing is silently destroyed.

Update flow is split into two steps so the launcher can ask before installing:

    check_for_update(...)  -> UpdateResult (fetches, decides; changes nothing)
    apply_update(...)      -> UpdateResult (stash + fast-forward merge)

`check_and_update(...)` runs both back-to-back (the old auto-install behaviour)
and is kept for compatibility / headless use.

Rollback is just git: list_versions() enumerates recent commits and
rollback_to() does a guarded `git reset --hard` to the chosen one.

The only external requirement is a `git` executable on PATH -- which the conda
environment provides, so it works even on machines with no system git.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# Status values returned in UpdateResult.status
UP_TO_DATE = "up_to_date"
UPDATE_AVAILABLE = "update_available"  # a fast-forward update exists but is NOT yet applied
UPDATED = "updated"
OFFLINE = "offline"
SKIPPED = "skipped"
LOCAL_AHEAD = "local_ahead"   # checkout has un-pushed / diverged commits; left untouched
PINNED = "pinned"             # HEAD is detached at a user-chosen rev; tracking nothing
ERROR = "error"

# Path (relative to repo root) whose change triggers a dependency-env update.
ENV_FILE_REL = os.path.join("install", "environment.yml")

# --------------------------------------------------------------------------- #
# Release channels
# --------------------------------------------------------------------------- #
# A channel is a user-facing name for a branch. `stable` is what every install
# tracks unless the user opts in; `dev` carries work that may break. The
# mapping exists so the UI and the persisted state talk about channels while
# git only ever sees branch names -- renaming a branch is then a one-line
# change here rather than a search across the launcher and both installers.
#
# NOTE: install.sh and install.ps1 default BRANCH/-Branch to "main"
# independently. If STABLE_BRANCH ever changes, those two must change with it.
STABLE_BRANCH = "main"
DEV_BRANCH = "dev"
CHANNELS: Dict[str, str] = {"stable": STABLE_BRANCH, "dev": DEV_BRANCH}
DEFAULT_CHANNEL = "stable"


@dataclass
class UpdateResult:
    status: str
    old_rev: Optional[str] = None
    new_rev: Optional[str] = None
    env_changed: bool = False
    update_available: bool = False
    branch: Optional[str] = None
    channel: Optional[str] = None
    pinned: bool = False
    message: str = ""
    stashed: bool = False
    changelog: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


def _default_logger(msg: str) -> None:
    print(f"[updater] {msg}")


def find_repo_root(start: Optional[str] = None) -> Optional[str]:
    """Walk upward from `start` (default: this file) to find a dir containing .git."""
    here = os.path.abspath(start or __file__)
    if os.path.isfile(here):
        here = os.path.dirname(here)
    while True:
        if os.path.isdir(os.path.join(here, ".git")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


_GIT_EXE: Optional[str] = None


def _find_git() -> str:
    """
    Resolve a git executable, preferring an absolute path over the bare name.

    Order: $HIBACHI_GIT, then PATH, then the known conda-env locations relative
    to this interpreter's prefix (git ships inside the env). Falls back to the
    bare "git" if nothing is found, preserving the old behaviour. Cached.

    This makes the self-updater robust when the app is launched with the env's
    interpreter directly (no conda activation), where git is installed in the
    env but not on PATH.
    """
    global _GIT_EXE
    if _GIT_EXE is not None:
        return _GIT_EXE

    import shutil
    import sys

    override = os.environ.get("HIBACHI_GIT")
    if override and os.path.isfile(override):
        _GIT_EXE = override
        return _GIT_EXE

    found = shutil.which("git")
    if found:
        _GIT_EXE = found
        return _GIT_EXE

    prefix = sys.prefix
    if sys.platform.startswith("win"):
        candidates = [
            os.path.join(prefix, "Library", "bin", "git.exe"),
            os.path.join(prefix, "Library", "cmd", "git.exe"),
            os.path.join(prefix, "Library", "mingw64", "bin", "git.exe"),
            os.path.join(prefix, "Library", "mingw-w64", "bin", "git.exe"),
        ]
    else:
        candidates = [os.path.join(prefix, "bin", "git")]
    for cand in candidates:
        if os.path.isfile(cand):
            _GIT_EXE = cand
            return _GIT_EXE

    _GIT_EXE = "git"  # last resort; may still resolve via PATH at call time
    return _GIT_EXE


# Windows: launching a console program (git) from a GUI process (pythonw) pops a
# console window for a split second. Passing CREATE_NO_WINDOW suppresses it, so
# the self-update's many git calls don't flash terminals on the user's screen.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0


def _git(
    args: List[str],
    cwd: str,
    timeout: int = 60,
) -> Tuple[int, str, str]:
    """Run a git command, returning (returncode, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(
            [_find_git(), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git executable not found (checked PATH and the env)"
    except subprocess.TimeoutExpired:
        return 124, "", f"git {' '.join(args)} timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Small query helpers (used by the launcher's rollback UI)
# --------------------------------------------------------------------------- #
def head_branch(repo_root: str) -> Optional[str]:
    """
    The branch HEAD is on, or None when HEAD is detached.

    Returns the truth and nothing else. The old `current_branch` answered this
    question with the literal string "main" when HEAD was detached, and
    `describe_version` stamped that answer into every processed dataset -- an
    invented value presented as provenance. Callers that need *something* to
    track should use `resolve_branch`; callers recording what happened must use
    this and accept None.
    """
    rc, cur, _ = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if rc == 0 and cur and cur != "HEAD":
        return cur
    return None


def is_pinned(repo_root: str) -> bool:
    """True when HEAD is detached -- i.e. the user pinned a specific version."""
    return head_branch(repo_root) is None


def resolve_branch(repo_root: str, branch: Optional[str] = None) -> str:
    """
    Which branch this install should track, in order of precedence:

        1. an explicit `branch` argument            (caller knows best)
        2. $HIBACHI_BRANCH                          (operator override)
        3. HEAD's branch, if it is NOT a channel    (a dev's feature branch:
           branch and not detached                   respect where they are)
        4. the persisted channel                    (the normal case)
        5. STABLE_BRANCH                            (never-configured install)

    Rule 3 keeps a developer working on `feature/x` from being told about
    updates to `main`; rule 4 is what makes the channel choice sticky across
    launches, including while pinned (HEAD detached), where there is no branch
    to infer from.
    """
    if branch:
        return branch
    env = os.environ.get("HIBACHI_BRANCH")
    if env:
        return env
    head = head_branch(repo_root)
    if head and head not in CHANNELS.values():
        return head
    return channel_branch()


def current_branch(repo_root: str) -> str:
    """Deprecated alias for `resolve_branch`, kept for external callers."""
    return resolve_branch(repo_root)


def _fetch_branch(repo_root: str, branch: str, timeout: int = 30) -> Tuple[int, str]:
    """
    Fetch one branch, creating/updating refs/remotes/origin/<branch>.

    The refspec is explicit rather than a bare `git fetch origin <branch>`,
    which only guarantees FETCH_HEAD: on a clone made with --single-branch (or
    a shallow clone), remote.origin.fetch matches only the cloned branch, so no
    remote-tracking ref is created for any other one and `rev-parse
    origin/<branch>` fails -- making a channel that exists look nonexistent.
    """
    rc, _, err = _git(
        ["fetch", "--quiet", "origin",
         f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
        repo_root, timeout=timeout,
    )
    return rc, err


def _remote_reachable(repo_root: str, timeout: int = 15) -> bool:
    """
    Can we talk to origin at all?

    Only called when a fetch has already failed, to tell "no network" apart
    from "that branch does not exist on the server". Both make `git fetch`
    exit non-zero, and reporting the second as "working offline" sends the user
    to check their wifi over a branch-name problem.
    """
    rc, _, _ = _git(["ls-remote", "--exit-code", "--heads", "origin"], repo_root,
                    timeout=timeout)
    return rc == 0


def current_rev(repo_root: str) -> Optional[str]:
    rc, rev, _ = _git(["rev-parse", "HEAD"], repo_root)
    return rev if rc == 0 and rev else None


def remote_tip(repo_root: str, branch: Optional[str] = None) -> Optional[str]:
    branch = branch or current_branch(repo_root)
    rc, rev, _ = _git(["rev-parse", f"origin/{branch}"], repo_root)
    return rev if rc == 0 and rev else None


def describe_version(repo_root: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Best-effort snapshot of the running HIBACHI version, for stamping into
    processed output so an analysis can be reproduced later (`git checkout
    <commit>` restores the exact code that produced it).

    Returns a dict with keys: commit (full sha), short (abbrev sha), date
    (commit date, YYYY-MM-DD), branch, and dirty (True if the working tree had
    uncommitted changes at processing time -- important, because a dirty tree
    means the commit alone does NOT fully reproduce the code). Never raises; any
    field it can't determine is None (e.g. not a git checkout -> commit is None).

    Two fields describe *where the code came from* rather than which commit it
    is: `channel` (the release channel the install tracks) and `branch` (the
    branch HEAD is actually on, or None when the version is pinned). They can
    legitimately disagree -- a pinned dev install reports channel 'dev' and
    branch None -- and that disagreement is the useful signal, so neither is
    inferred from the other.
    """
    info: Dict[str, Optional[str]] = {
        "commit": None, "short": None, "date": None, "branch": None,
        "channel": None, "pinned": None, "tag": None, "dirty": None,
    }
    try:
        if not repo_root:
            repo_root = find_repo_root(os.path.dirname(os.path.abspath(__file__)))
        if not repo_root:
            return info
        rc, out, _ = _git(["log", "-1", "--format=%H%x1f%h%x1f%cs", "HEAD"], repo_root)
        if rc == 0 and out:
            parts = (out.split("\x1f") + ["", "", ""])[:3]
            info["commit"] = parts[0] or None
            info["short"] = parts[1] or None
            info["date"] = parts[2] or None
        # `head_branch`, not `resolve_branch`: this field records what the
        # checkout WAS, so a detached (pinned) HEAD must report None rather
        # than the branch it would track if it were following one.
        info["branch"] = head_branch(repo_root)
        info["channel"] = get_channel()
        info["pinned"] = info["branch"] is None
        # Nearest semantic tag, e.g. 'v1.2.0' (exact) or 'v1.2.0-3-gabc123' (3
        # commits after the tag). Empty/None if the repo has no tags. This is the
        # human-facing version to cite when reproducing an analysis.
        rc, tag, _ = _git(["describe", "--tags", "--always"], repo_root)
        if rc == 0 and tag:
            info["tag"] = tag
        # Only tracked-file modifications count as "dirty". We ignore untracked
        # files because importing modules writes __pycache__/*.pyc into the tree,
        # which would otherwise mark every run dirty. What matters for
        # reproducibility is whether the committed source was hand-edited.
        rc, out, _ = _git(["status", "--porcelain", "--untracked-files=no"], repo_root)
        if rc == 0:
            info["dirty"] = bool(out.strip())
    except Exception:  # pragma: no cover - defensive; version stamping is best-effort
        pass
    return info


# --------------------------------------------------------------------------- #
# Step 1: check (no side effects beyond `git fetch`)
# --------------------------------------------------------------------------- #
def check_for_update(
    repo_root: Optional[str] = None,
    branch: Optional[str] = None,
    fetch_timeout: int = 30,
    logger: Optional[Callable[[str], None]] = None,
) -> UpdateResult:
    """
    Fetch and decide what (if anything) can be updated -- WITHOUT applying it.

    Returns an UpdateResult. status is one of UP_TO_DATE / UPDATE_AVAILABLE /
    OFFLINE / LOCAL_AHEAD / SKIPPED / ERROR. When UPDATE_AVAILABLE, new_rev,
    env_changed and changelog are populated so the caller can prompt.
    """
    log = logger or _default_logger
    result = UpdateResult(status=ERROR)

    root = repo_root or find_repo_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        result.status = SKIPPED
        result.message = "Not a git checkout; skipping self-update."
        log(result.message)
        return result

    branch = resolve_branch(root, branch)
    result.branch = branch
    result.channel = get_channel()

    rc, old_rev, _ = _git(["rev-parse", "HEAD"], root)
    result.old_rev = old_rev or None

    # A pinned install (detached HEAD) tracks nothing by definition. Report
    # what is available so the UI can offer to unpin, but never present it as
    # an applicable update: `update_available` stays False, and the status is
    # one no caller treats as actionable.
    if is_pinned(root):
        result.status = PINNED
        result.pinned = True
        result.message = (
            f"Pinned to version {(old_rev or '')[:8]}; not tracking "
            f"'{branch}'. Switch version to resume updates."
        )
        rc, _ = _fetch_branch(root, branch, fetch_timeout)
        if rc == 0:
            result.new_rev = remote_tip(root, branch)
        log(result.message)
        return result

    log(f"Checking for updates on '{branch}' ({result.channel} channel)...")
    rc, err = _fetch_branch(root, branch, fetch_timeout)
    if rc != 0:
        if _remote_reachable(root):
            result.status = ERROR
            result.message = (
                f"The update server has no branch '{branch}'. If this install "
                f"tracks a channel that has been retired, switch channel to "
                f"resume updates."
            )
        else:
            result.status = OFFLINE
            result.message = f"Could not reach the update server (working offline). {err}".strip()
        log(result.message)
        return result

    rc, remote_rev, err = _git(["rev-parse", f"origin/{branch}"], root)
    if rc != 0 or not remote_rev:
        result.status = ERROR
        result.message = f"Could not resolve origin/{branch}: {err}"
        log(result.message)
        return result
    result.new_rev = remote_rev

    if old_rev == remote_rev:
        result.status = UP_TO_DATE
        result.message = "Already up to date."
        log(result.message)
        return result

    # Only a clean fast-forward (HEAD is an ancestor of the remote tip) is an
    # auto-updatable "update". If the checkout is AHEAD or DIVERGED (a dev's
    # machine with un-pushed commits), leave it untouched.
    rc, _, _ = _git(["merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"], root)
    if rc != 0:
        result.status = LOCAL_AHEAD
        result.message = (
            "Local checkout is ahead of or has diverged from the server; "
            "skipping auto-update to preserve local changes. "
            "(Push/pull manually to sync.)"
        )
        log(result.message)
        return result

    # Did the dependency spec change between here and the remote tip?
    rc, changed, _ = _git(["diff", "--name-only", f"{old_rev}..{remote_rev}"], root)
    if rc == 0:
        norm = {os.path.normpath(f.strip()) for f in changed.splitlines() if f.strip()}
        result.env_changed = os.path.normpath(ENV_FILE_REL) in norm

    # Collect commit subjects for a short "what's changed" list (best-effort).
    rc, subjects, _ = _git(
        ["log", "--no-merges", "--format=%s", f"{old_rev}..{remote_rev}"], root
    )
    if rc == 0 and subjects:
        result.changelog = [s.strip() for s in subjects.splitlines() if s.strip()][:20]

    result.status = UPDATE_AVAILABLE
    result.update_available = True
    result.message = f"Update available: {(old_rev or '')[:8]} -> {remote_rev[:8]}."
    log(result.message)
    return result


# --------------------------------------------------------------------------- #
# Step 2: apply (stash + fast-forward)
# --------------------------------------------------------------------------- #
def apply_update(
    repo_root: Optional[str] = None,
    result: Optional[UpdateResult] = None,
    branch: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> UpdateResult:
    """
    Apply the fast-forward update discovered by check_for_update.

    Re-verifies the fast-forward before touching anything, preserves any
    uncommitted changes in a timestamped stash, then `git merge --ff-only`.
    """
    log = logger or _default_logger
    root = repo_root or find_repo_root()
    if result is None:
        result = check_for_update(root, branch=branch, logger=logger)
    if result.status not in (UPDATE_AVAILABLE, UPDATED):
        return result  # nothing to apply (up-to-date, offline, diverged, ...)

    branch = result.branch or branch or current_branch(root)

    # Re-verify we are still strictly behind origin/branch.
    rc, _, _ = _git(["merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"], root)
    if rc != 0:
        result.status = LOCAL_AHEAD
        result.message = "Checkout changed since the update check; skipping to be safe."
        log(result.message)
        return result

    rc, dirty, _ = _git(["status", "--porcelain"], root)
    if rc == 0 and dirty:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        rc_s, _, err_s = _git(
            ["stash", "push", "--include-untracked", "-m", f"hibachi-autobackup-{stamp}"],
            root,
        )
        if rc_s == 0:
            result.stashed = True
            log(f"Local changes detected; backed them up to a git stash ({stamp}).")
        else:
            log(f"Warning: could not stash local changes: {err_s}")

    log("Downloading and applying updates...")
    rc, _, err = _git(["merge", "--ff-only", f"origin/{branch}"], root)
    if rc != 0:
        result.status = ERROR
        result.message = f"Update failed while applying changes: {err}"
        log(result.message)
        return result

    result.status = UPDATED
    result.update_available = False
    result.message = f"Updated {(result.old_rev or '')[:8]} -> {(result.new_rev or '')[:8]}."
    log(result.message)
    if result.env_changed:
        set_pending_env_update(True)
        log("Dependency list changed; the environment will be updated.")
    return result


def check_and_update(
    repo_root: Optional[str] = None,
    branch: Optional[str] = None,
    fetch_timeout: int = 30,
    logger: Optional[Callable[[str], None]] = None,
) -> UpdateResult:
    """
    Check and, if a fast-forward update exists, apply it immediately.

    This is the original auto-install behaviour, kept for compatibility and for
    headless/non-interactive use. Interactive callers should use
    check_for_update() + apply_update() so they can prompt in between.
    """
    res = check_for_update(repo_root, branch=branch, fetch_timeout=fetch_timeout, logger=logger)
    if res.status == UPDATE_AVAILABLE:
        return apply_update(repo_root or find_repo_root(), res, logger=logger)
    return res


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #
def _versions_along(root: str, ref: str, limit: int) -> List[dict]:
    """Commits reachable from `ref`, newest first. Empty if `ref` is unknown."""
    rc, out, _ = _git(
        ["log", f"-n{int(limit)}", "--first-parent",
         "--format=%H%x1f%h%x1f%cs%x1f%s", ref],
        root,
    )
    versions: List[dict] = []
    if rc != 0 or not out:
        return versions
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            versions.append(
                {"rev": parts[0], "short": parts[1], "date": parts[2], "subject": parts[3]}
            )
    return versions


def list_versions(
    repo_root: Optional[str] = None,
    limit: int = 15,
    branch: Optional[str] = None,
) -> List[dict]:
    """
    Recent versions (newest first) as dicts: {rev, short, date, subject}.

    Listed along origin/<branch> when available (so you can also switch forward
    to a fetched-but-not-installed version), otherwise along local HEAD.

    NOTE the HEAD fallback: this cannot distinguish "that branch has no commits"
    from "that branch was never fetched", and answers the second case with the
    CURRENT branch's history. That is fine for its one caller, which only ever
    asks about the branch it is already on. Do not use it to populate a list for
    some OTHER channel -- use `channel_overview`, which reports a channel as
    unavailable rather than substituting the wrong commits.
    """
    root = repo_root or find_repo_root()
    if not root:
        return []
    branch = branch or current_branch(root)
    ref = f"origin/{branch}"
    rc, _, _ = _git(["rev-parse", "--verify", "--quiet", ref], root)
    if rc != 0:
        ref = "HEAD"
    return _versions_along(root, ref, limit)


def channel_overview(
    repo_root: Optional[str] = None,
    limit: int = 15,
    fetch: bool = True,
    fetch_timeout: int = 30,
) -> Dict:
    """
    Everything a channel-picking UI needs, in one call.

    Returns:
        {
          "current":  "stable",              # the tracked channel
          "pinned":   False,                 # HEAD detached?
          "head":     "<sha>",               # what is checked out now
          "channels": {
             "stable": {"branch": "main", "available": True,
                        "tip": "<sha>", "versions": [ {...}, ... ]},
             "dev":    {"branch": "dev", "available": False, "tip": None,
                        "versions": [], "reason": "never fetched"},
          },
        }

    `available` is False when `origin/<branch>` cannot be resolved even after a
    fetch attempt -- the channel does not exist on the server, or we are offline
    and never fetched it. Such a channel gets an EMPTY version list and a reason,
    never a borrowed one: showing the current channel's commits under the other
    channel's name would invite picking a version that is not what it says.

    Fetches each channel's branch (pass fetch=False to work purely offline).
    Read-only otherwise: it never moves HEAD or writes state.
    """
    root = repo_root or find_repo_root()
    out: Dict = {"current": get_channel(), "pinned": False, "head": None,
                 "channels": {}}
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        for name, branch in CHANNELS.items():
            out["channels"][name] = {"branch": branch, "available": False,
                                     "tip": None, "versions": [],
                                     "reason": "not a git checkout"}
        return out

    out["pinned"] = is_pinned(root)
    out["head"] = current_rev(root)

    for name, branch in CHANNELS.items():
        entry: Dict = {"branch": branch, "available": False, "tip": None,
                       "versions": []}
        if fetch:
            _fetch_branch(root, branch, fetch_timeout)
        ref = f"origin/{branch}"
        rc, tip, _ = _git(["rev-parse", "--verify", "--quiet", ref], root)
        if rc != 0 or not tip:
            entry["reason"] = ("not on the server, or never fetched"
                               if fetch else "never fetched (offline mode)")
        else:
            entry["available"] = True
            entry["tip"] = tip
            entry["versions"] = _versions_along(root, ref, limit)
        out["channels"][name] = entry
    return out


def _stash_guard(root: str, tag: str, log: Callable[[str], None]) -> bool:
    """Stash uncommitted changes before a destructive checkout. Never raises."""
    rc, dirty, _ = _git(["status", "--porcelain"], root)
    if rc != 0 or not dirty:
        return False
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    rc_s, _, err_s = _git(
        ["stash", "push", "--include-untracked", "-m", f"hibachi-{tag}-{stamp}"], root
    )
    if rc_s == 0:
        log(f"Local changes detected; backed them up to a git stash ({stamp}).")
        return True
    log(f"Warning: could not stash local changes: {err_s}")
    return False


def pin_to(
    repo_root: Optional[str],
    rev: str,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Pin the checkout to `rev` by detaching HEAD there.

    Detaching, rather than `reset --hard` on the current branch, is deliberate.
    A reset rewrites the local branch pointer, so `main` stops meaning what
    `origin/main` means: the next launch sees a clean fast-forward and offers to
    pull the user straight back to the tip they just left. (That is what the
    old `set_skipped_rev(remote_tip)` call in the launcher existed to suppress.)
    A detached HEAD instead says exactly what the user asked for -- this commit,
    tracking nothing -- so `check_for_update` reports PINNED and nothing is
    offered until the pin is released. The tracked channel is left alone, so
    unpinning returns to whichever channel the user was on.

    Uncommitted changes are stashed first. Returns (ok, message).
    """
    log = logger or _default_logger
    root = repo_root or find_repo_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        return False, "Not a git checkout."

    rc, full, _ = _git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], root)
    if rc != 0:
        return False, f"Unknown version: {rev}"
    rev = full or rev

    _stash_guard(root, "pin-backup", log)

    rc, _, err = _git(["checkout", "--detach", "--quiet", rev], root)
    if rc != 0:
        return False, f"Could not switch to that version: {err}"

    log(f"Pinned to {rev[:8]} (updates paused until you switch back).")
    return True, f"Pinned to version {rev[:8]}. Updates are paused."


# Kept so existing callers keep working; the behaviour is now a pin, not a
# branch-rewinding reset. Prefer `pin_to` in new code.
rollback_to = pin_to


def unpin(
    repo_root: Optional[str] = None,
    channel: Optional[str] = None,
    fetch_timeout: int = 30,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Release a pin and resume tracking `channel` (default: the tracked one).

    Works offline: if origin cannot be reached, HEAD is attached to the local
    branch and the next launch fast-forwards normally.
    """
    log = logger or _default_logger
    root = repo_root or find_repo_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        return False, "Not a git checkout."

    branch = channel_branch(channel)
    _stash_guard(root, "unpin-backup", log)

    _fetch_branch(root, branch, fetch_timeout)
    rc, _, _ = _git(["rev-parse", "--verify", "--quiet", f"origin/{branch}"], root)
    if rc == 0:
        rc, _, err = _git(["checkout", "-B", branch, f"origin/{branch}", "--quiet"], root)
    else:
        rc, _, err = _git(["checkout", "--quiet", branch], root)
    if rc != 0:
        return False, f"Could not resume updates on '{branch}': {err}"

    log(f"Resumed tracking '{branch}'.")
    return True, f"Resumed tracking the {channel or get_channel()} channel."


def switch_channel(
    repo_root: Optional[str] = None,
    channel: str = DEFAULT_CHANNEL,
    fetch_timeout: int = 30,
    logger: Optional[Callable[[str], None]] = None,
) -> UpdateResult:
    """
    Move the checkout onto `channel` and persist the choice.

    This cannot reuse apply_update: that path is `merge --ff-only` gated on
    HEAD being an ancestor of the remote tip, and two channels diverge by
    construction, so it would report LOCAL_AHEAD and refuse. A switch is a
    deliberate, guarded replacement of the working tree instead --
    `checkout -B <branch> origin/<branch>` -- which also re-attaches HEAD, so
    switching channels releases a pin as a side effect.

    Order matters: the channel is persisted only AFTER the checkout succeeds.
    Recording the intent first would leave an install claiming to be on dev
    while running stable code, which is worse than not switching at all.

    `env_changed` is computed by diffing the two trees directly rather than
    over a commit range, because the channels are not ancestors of one another.
    It is set in BOTH directions: returning to stable also needs the dependency
    environment rebuilt, or stable code runs against dev's pinned numerics.
    """
    log = logger or _default_logger
    result = UpdateResult(status=ERROR)

    if channel not in CHANNELS:
        result.message = f"Unknown channel {channel!r}."
        log(result.message)
        return result

    root = repo_root or find_repo_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        result.status = SKIPPED
        result.message = "Not a git checkout; cannot switch channel."
        log(result.message)
        return result

    branch = CHANNELS[channel]
    result.branch = branch
    result.channel = channel
    result.old_rev = current_rev(root)

    log(f"Switching to the {channel} channel ('{branch}')...")
    rc, err = _fetch_branch(root, branch, fetch_timeout)
    if rc != 0:
        if _remote_reachable(root):
            result.message = (
                f"The {channel} channel does not exist on the server "
                f"(no branch '{branch}'). Nothing was changed."
            )
        else:
            result.status = OFFLINE
            result.message = (
                f"Could not reach the update server, so the {channel} channel "
                f"was not installed. Nothing was changed. {err}".strip()
            )
        log(result.message)
        return result

    rc, remote_rev, _ = _git(["rev-parse", f"origin/{branch}"], root)
    if rc != 0 or not remote_rev:
        result.message = (
            f"The {channel} channel does not exist on the server "
            f"(no branch '{branch}'). Nothing was changed."
        )
        log(result.message)
        return result
    result.new_rev = remote_rev

    rc, changed, _ = _git(["diff", "--name-only", "HEAD", remote_rev], root)
    if rc == 0:
        norm = {os.path.normpath(f.strip()) for f in changed.splitlines() if f.strip()}
        result.env_changed = os.path.normpath(ENV_FILE_REL) in norm

    rc, subjects, _ = _git(
        ["log", "--no-merges", "--format=%s", f"{result.old_rev}..{remote_rev}"], root
    )
    if rc == 0 and subjects:
        result.changelog = [s.strip() for s in subjects.splitlines() if s.strip()][:20]

    result.stashed = _stash_guard(root, f"channel-{channel}-backup", log)

    # Check out the exact rev that was diffed above, not `origin/<branch>`, so a
    # push landing mid-switch cannot leave the tree at a commit we never
    # inspected. Upstream is then set separately, best-effort, so a power user's
    # manual `git pull` still works.
    rc, _, err = _git(["checkout", "-B", branch, remote_rev, "--quiet"], root)
    if rc != 0:
        result.message = f"Could not switch to the {channel} channel: {err}"
        log(result.message)
        return result
    _git(["branch", f"--set-upstream-to=origin/{branch}", branch], root)

    if not set_channel(channel):
        # The checkout moved but the choice could not be saved, so the next
        # launch would resolve the channel from HEAD's branch and appear to
        # stick anyway -- until the user pins. Say so rather than imply it took.
        log(f"Warning: switched to {channel}, but the choice could not be saved "
            f"to {_state_path()}; it may not persist.")

    result.status = UPDATED
    result.message = (
        f"Now on the {channel} channel ({(remote_rev or '')[:8]})."
    )
    log(result.message)
    if result.env_changed:
        # Set only now, after the checkout has actually moved. Recording it
        # earlier would leave a flag demanding an environment rebuild for code
        # that was never installed.
        set_pending_env_update(True)
        log("Dependency list differs on this channel; the environment will be updated.")
    return result


# --------------------------------------------------------------------------- #
# Tiny persisted state: the tracked channel, and per-channel skipped revs
# --------------------------------------------------------------------------- #
# Kept in $HIBACHI_STATE_DIR/state.json (default ~/.hibachi/state.json), which
# lives OUTSIDE the repository -- so it survives every reset/fast-forward the
# updater performs, which is exactly why the channel choice can be sticky.
#
# Writes are atomic (temp file + os.replace): a half-written state.json would
# lose the channel and silently drop a dev install back to stable.
def _state_path() -> str:
    base = os.environ.get("HIBACHI_STATE_DIR") or os.path.join(os.path.expanduser("~"), ".hibachi")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "state.json")


def _read_state() -> Dict:
    import json
    try:
        with open(_state_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(data: Dict) -> bool:
    """Best-effort atomic write. Returns True on success; never raises."""
    import json
    path = _state_path()
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def get_channel() -> str:
    """
    The channel this install tracks. Unknown/absent values fall back to stable.

    An install that has never made a choice reads as `stable`, which is why no
    migration was needed when the dev channel was introduced: every existing
    install keeps tracking the same branch it always did.
    """
    ch = _read_state().get("channel")
    return ch if ch in CHANNELS else DEFAULT_CHANNEL


def set_channel(channel: str) -> bool:
    """Persist the tracked channel. Returns False if it could not be saved."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {sorted(CHANNELS)}")
    data = _read_state()
    data["channel"] = channel
    return _write_state(data)


def channel_branch(channel: Optional[str] = None) -> str:
    """Branch name for `channel` (default: the tracked one)."""
    return CHANNELS.get(channel or get_channel(), STABLE_BRANCH)


def get_pending_env_update() -> bool:
    """
    True when the checkout moved to code whose dependency spec differs from
    what is installed, and that spec has not been applied yet.

    Set by `apply_update` and `switch_channel`; cleared by the launcher once it
    has attempted the update. It exists because the two are separate processes
    and separate moments: the code changes now, but only a launcher start can
    rebuild the environment and re-exec into it. Without the flag, a switch made
    from inside the running app -- which cannot rebuild its own environment --
    would leave the next launch seeing an up-to-date checkout, `env_changed`
    False, and no reason to update anything. The result is one channel's code
    running against the other's pinned numerics, silently.

    It also survives a kill: a solve interrupted halfway leaves the flag set, so
    the next launch tries again instead of proceeding on a half-built env.
    """
    return bool(_read_state().get("pending_env_update"))


def set_pending_env_update(value: bool) -> None:
    data = _read_state()
    if value:
        data["pending_env_update"] = True
    else:
        data.pop("pending_env_update", None)
    _write_state(data)


def get_skipped_rev(channel: Optional[str] = None) -> Optional[str]:
    """
    The rev the user chose to skip on `channel` (default: the tracked one).

    Skips are per-channel: a version dismissed on dev must not suppress the
    update prompt on stable, which is what a single shared value did.
    """
    data = _read_state()
    skip = data.get("skip_rev")
    ch = channel or get_channel()
    if isinstance(skip, dict):
        val = skip.get(ch)
        return val if isinstance(val, str) else None
    # Legacy scalar, written before channels existed. Such an install was on
    # stable by definition, so honour it there and nowhere else.
    if isinstance(skip, str):
        return skip if ch == DEFAULT_CHANNEL else None
    return None


def set_skipped_rev(rev: Optional[str], channel: Optional[str] = None) -> None:
    data = _read_state()
    ch = channel or get_channel()
    skip = data.get("skip_rev")
    if isinstance(skip, dict):
        table = dict(skip)
    elif isinstance(skip, str):
        table = {DEFAULT_CHANNEL: skip}   # migrate the legacy scalar in place
    else:
        table = {}
    if rev is None:
        table.pop(ch, None)
    else:
        table[ch] = rev
    data["skip_rev"] = table
    _write_state(data)