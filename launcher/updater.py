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

The only external requirement is a `git` executable on PATH -- which the conda
environment provides, so it works even on machines with no system git.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# Status values returned in UpdateResult.status
UP_TO_DATE = "up_to_date"
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
    branch: Optional[str] = None
    message: str = ""
    stashed: bool = False
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


def _git(
    args: List[str],
    cwd: str,
    timeout: int = 60,
) -> Tuple[int, str, str]:
    """Run a git command, returning (returncode, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git executable not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"git {' '.join(args)} timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{type(exc).__name__}: {exc}"


def check_and_update(
    repo_root: Optional[str] = None,
    branch: Optional[str] = None,
    fetch_timeout: int = 30,
    logger: Optional[Callable[[str], None]] = None,
) -> UpdateResult:
    """
    Bring the checkout up to date with origin/<branch>, safely.

    Returns an UpdateResult describing what happened. The caller should launch
    the app regardless of status (except it may want to trigger a dependency
    update when result.env_changed is True).
    """
    log = logger or _default_logger
    result = UpdateResult(status=ERROR)

    root = repo_root or find_repo_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        result.status = SKIPPED
        result.message = "Not a git checkout; skipping self-update."
        log(result.message)
        return result

    # Determine the branch to track: explicit arg > env var > current branch.
    if not branch:
        branch = os.environ.get("HIBACHI_BRANCH")
    if not branch:
        rc, cur, _ = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
        branch = cur if (rc == 0 and cur and cur != "HEAD") else "main"
    result.branch = branch

    rc, old_rev, _ = _git(["rev-parse", "HEAD"], root)
    result.old_rev = old_rev or None

    # 1) Fetch. A failure here almost always means "offline" -> launch anyway.
    log(f"Checking for updates on '{branch}'...")
    rc, _, err = _git(["fetch", "--quiet", "origin", branch], root, timeout=fetch_timeout)
    if rc != 0:
        result.status = OFFLINE
        result.message = f"Could not reach the update server (working offline). {err}".strip()
        log(result.message)
        return result

    # 2) Compare local HEAD against the fetched remote tip.
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

    # Only auto-update when the local checkout is strictly BEHIND the remote,
    # i.e. HEAD is an ancestor of origin/<branch> and the change is a clean
    # fast-forward. If the checkout is AHEAD or has DIVERGED (a developer's
    # machine with un-pushed commits), we must NOT move it -- doing so would
    # silently discard local work. In that case we leave the tree exactly as-is
    # and just launch whatever is on disk.
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

    # Detect whether the environment spec changed between old and new.
    rc, changed, _ = _git(
        ["diff", "--name-only", f"{old_rev}..{remote_rev}"], root
    )
    if rc == 0:
        changed_files = {line.strip() for line in changed.splitlines() if line.strip()}
        norm = {os.path.normpath(f) for f in changed_files}
        result.env_changed = os.path.normpath(ENV_FILE_REL) in norm

    # Preserve any uncommitted changes to tracked files so the fast-forward can
    # apply cleanly (untracked files never block a fast-forward).
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

    # Fast-forward only. We already confirmed HEAD is an ancestor of the remote,
    # so this cannot lose commits; there is deliberately no hard-reset fallback.
    log("Downloading and applying updates...")
    rc, _, err = _git(["merge", "--ff-only", f"origin/{branch}"], root)
    if rc != 0:
        result.status = ERROR
        result.message = f"Update failed while applying changes: {err}"
        log(result.message)
        return result

    result.status = UPDATED
    short_old = (old_rev or "")[:8]
    short_new = remote_rev[:8]
    result.message = f"Updated {short_old} -> {short_new}."
    log(result.message)
    if result.env_changed:
        log("Dependency list changed; the environment will be updated.")
    return result