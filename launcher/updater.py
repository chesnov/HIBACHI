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
ERROR = "error"

# Path (relative to repo root) whose change triggers a dependency-env update.
ENV_FILE_REL = os.path.join("install", "environment.yml")


@dataclass
class UpdateResult:
    status: str
    old_rev: Optional[str] = None
    new_rev: Optional[str] = None
    env_changed: bool = False
    update_available: bool = False
    branch: Optional[str] = None
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
def current_branch(repo_root: str) -> str:
    rc, cur, _ = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if rc == 0 and cur and cur != "HEAD":
        return cur
    return os.environ.get("HIBACHI_BRANCH") or "main"


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
    """
    info: Dict[str, Optional[str]] = {
        "commit": None, "short": None, "date": None, "branch": None, "dirty": None,
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
        info["branch"] = current_branch(repo_root)
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

    if not branch:
        branch = os.environ.get("HIBACHI_BRANCH")
    if not branch:
        branch = current_branch(root)
    result.branch = branch

    rc, old_rev, _ = _git(["rev-parse", "HEAD"], root)
    result.old_rev = old_rev or None

    log(f"Checking for updates on '{branch}'...")
    rc, _, err = _git(["fetch", "--quiet", "origin", branch], root, timeout=fetch_timeout)
    if rc != 0:
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
def list_versions(
    repo_root: Optional[str] = None,
    limit: int = 15,
    branch: Optional[str] = None,
) -> List[dict]:
    """
    Recent versions (newest first) as dicts: {rev, short, date, subject}.

    Listed along origin/<branch> when available (so you can also switch forward
    to a fetched-but-not-installed version), otherwise along local HEAD.
    """
    root = repo_root or find_repo_root()
    if not root:
        return []
    branch = branch or current_branch(root)
    ref = f"origin/{branch}"
    rc, _, _ = _git(["rev-parse", "--verify", "--quiet", ref], root)
    if rc != 0:
        ref = "HEAD"

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


def rollback_to(
    repo_root: Optional[str],
    rev: str,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Switch the checkout to `rev` (a guarded `git reset --hard`).

    Uncommitted changes are stashed first, so nothing is silently lost. Returns
    (ok, message).
    """
    log = logger or _default_logger
    root = repo_root or find_repo_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        return False, "Not a git checkout."

    rc, _, _ = _git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], root)
    if rc != 0:
        return False, f"Unknown version: {rev}"

    rc, dirty, _ = _git(["status", "--porcelain"], root)
    if rc == 0 and dirty:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        _git(["stash", "push", "--include-untracked", "-m", f"hibachi-rollback-backup-{stamp}"], root)
        log(f"Backed up local changes to a git stash ({stamp}).")

    rc, _, err = _git(["reset", "--hard", rev], root)
    if rc != 0:
        return False, f"Rollback failed: {err}"

    log(f"Switched to {rev[:8]}.")
    return True, f"Switched to version {rev[:8]}."


# --------------------------------------------------------------------------- #
# Tiny persisted state (only: which update version the user chose to skip)
# --------------------------------------------------------------------------- #
def _state_path() -> str:
    base = os.environ.get("HIBACHI_STATE_DIR") or os.path.join(os.path.expanduser("~"), ".hibachi")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "state.json")


def get_skipped_rev() -> Optional[str]:
    import json
    try:
        with open(_state_path()) as fh:
            return json.load(fh).get("skip_rev")
    except Exception:
        return None


def set_skipped_rev(rev: Optional[str]) -> None:
    import json
    path = _state_path()
    data = {}
    try:
        if os.path.isfile(path):
            with open(path) as fh:
                data = json.load(fh)
    except Exception:
        data = {}
    data["skip_rev"] = rev
    try:
        with open(path, "w") as fh:
            json.dump(data, fh)
    except Exception:
        pass