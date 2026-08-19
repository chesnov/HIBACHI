#!/usr/bin/env bash
# =============================================================================
# HIBACHI installer for macOS / Linux
# -----------------------------------------------------------------------------
# One-line install (from your GitHub README):
#
#   curl -fsSL https://raw.githubusercontent.com/chesnov/HIBACHI/main/install/install.sh | bash
#
# What it does (no admin rights, nothing touched system-wide):
#   1. Downloads micromamba (a single ~5 MB binary that bundles conda).
#   2. Builds the 'hibachi' environment from environment.yml (Python + git + all deps).
#   3. git-clones the app into the install directory.
#   4. Creates a double-click launcher (.desktop on Linux, .app on macOS).
#
# =============================================================================
# WHY THERE IS A PROGRESS WINDOW  (the most important thing in this file)
# -----------------------------------------------------------------------------
# The first install downloads ~1 GB of scientific packages and takes several
# minutes. Transient macOS notifications were not enough: users concluded the
# app had glitched and quit mid-install. That left a half-built environment
# which (before the fix below) could never be repaired by retrying -- so the app
# then crashed on every launch, deep in an import, with no visible error.
#
# So this script puts up a real, always-visible progress window with a moving
# bar, the current package name, and an elapsed timer. It is written in tkinter
# and EMBEDDED in this file as a heredoc rather than shipped as a separate
# module, because this script must work in two contexts where a sibling file
# would not exist: piped straight from curl, and copied alone into
# HIBACHI.app/Contents/Resources by build_dmg.sh.
#
# ORDERING CONSTRAINT: tkinter comes from the conda env's own Python, which does
# not exist yet at the start. So the environment build is split in two --
#   phase A: conda-level packages (python, git, pip)  -- fast, ~1 min
#   phase B: the pip: subsection (napari, PyQt5, ...) -- slow, many minutes
# and the window is raised between them, before the slow part. If some other
# tkinter-capable Python is already on the machine we raise it immediately
# instead. Either way the window is up for the whole slow phase.
#
# -----------------------------------------------------------------------------
# RECOVERY FROM A HALF-BUILT ENVIRONMENT
# -----------------------------------------------------------------------------
# Splitting the build as above also fixes the original trap. The old logic
# branched on "does the env prefix exist?" and took `micromamba env update` on
# every retry -- which skips the pip: subsection entirely when the conda-level
# solve finds nothing to do. Missing pip packages were therefore never
# reinstalled, no matter how often the user re-ran the installer.
#
# Now pip is ALWAYS run explicitly by us (step 4), on both the create and the
# update path, mirroring pass 2 of run_app.py::_update_environment(). On top of
# that we validate an existing env by importing what the app needs and rebuild
# from scratch if the probe fails, and a Cancel from the progress window deletes
# the partial env so the next attempt starts clean.
# =============================================================================
set -euo pipefail

# ------------------------- CONFIG (edit for your repo) ----------------------- #
GH_OWNER="${HIBACHI_OWNER:-chesnov}"        # <-- your GitHub username / org
GH_REPO="${HIBACHI_REPO:-HIBACHI}"           # <-- your GitHub repository name
BRANCH="${HIBACHI_BRANCH:-main}"          # branch to install / track
INSTALL_DIR="${HIBACHI_HOME:-$HOME/HIBACHI}"
ENV_NAME="hibachi"
# Set HIBACHI_FORCE_REBUILD=1 to discard any existing environment and rebuild.
FORCE_REBUILD="${HIBACHI_FORCE_REBUILD:-0}"
# Set HIBACHI_NO_INSTALLER_GUI=1 for a headless/CI install (console output only).
NO_GUI="${HIBACHI_NO_INSTALLER_GUI:-0}"
# Rough number of wheels pip ends up installing, used to scale the progress bar
# during the download phase. Only affects bar smoothness, never correctness.
EXPECTED_PKGS="${HIBACHI_EXPECTED_PKGS:-220}"
# ----------------------------------------------------------------------------- #

REPO_URL="https://github.com/${GH_OWNER}/${GH_REPO}.git"
ENV_YML_URL="https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${BRANCH}/install/environment.yml"
MAMBA_ROOT="${INSTALL_DIR}/micromamba"
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
APP_DIR="${INSTALL_DIR}/app"
ENV_PREFIX="${MAMBA_ROOT}/envs/${ENV_NAME}"
ENV_PY="${ENV_PREFIX}/bin/python"
ENV_YML="${INSTALL_DIR}/environment.yml"

GUI_SCRIPT="${INSTALL_DIR}/.installer_progress.py"
STATUS_FILE="${INSTALL_DIR}/.installer_status.json"
CANCEL_FILE="${INSTALL_DIR}/.installer_cancelled"
PIP_LOG="${INSTALL_DIR}/pip-install.log"
GUI_PID=""

# Modules the app imports at startup, in roughly the order the failing chain
# hits them (segment.py -> utils.high_level_gui.helper_funcs -> app_launch,
# relational_engine, cross_channel_window, metadata, ...). Import names, not
# distribution names: scikit-learn -> sklearn, simpleitk -> SimpleITK, etc.
# Probing these is what distinguishes a complete env from a half-built one.
REQUIRED_MODULES="yaml numpy pandas scipy tifffile PyQt5.QtWidgets vispy \
napari magicgui dask.array dask_image.ndmeasure sklearn seaborn skan \
SimpleITK slideio numba zarr plotly nbformat napari_animation aicspylibczi \
fcswrite PartSegCore_compiled_backend"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required." >&2; exit 1; }
command -v tar  >/dev/null 2>&1 || { echo "ERROR: tar is required." >&2; exit 1; }

# ============================================================================ #
# Progress reporting
# ============================================================================ #

# progress <pct> <ceil> <title> <detail>
# Writes the status file atomically (the GUI polls it 4x/second). `ceil` is the
# upper bound the GUI may creep towards while waiting for the next update --
# that slow creep is what keeps the bar from ever looking frozen.
progress() {
  local pct="$1" ceil="$2" title="$3" detail="${4:-}"
  [ -d "${INSTALL_DIR}" ] || return 0
  printf '{"pct": %s, "ceil": %s, "title": "%s", "detail": "%s", "state": "running"}\n' \
    "${pct}" "${ceil}" "${title//\"/}" "${detail//\"/}" > "${STATUS_FILE}.tmp" 2>/dev/null || return 0
  mv -f "${STATUS_FILE}.tmp" "${STATUS_FILE}" 2>/dev/null || true
  printf '    %s%s\n' "${title}" "${detail:+ -- ${detail}}"
}

progress_state() {   # progress_state <done|failed> [message]
  local state="$1" msg="${2:-}"
  [ -d "${INSTALL_DIR}" ] || return 0
  printf '{"pct": 100, "ceil": 100, "title": "%s", "detail": "", "state": "%s"}\n' \
    "${msg//\"/}" "${state}" > "${STATUS_FILE}.tmp" 2>/dev/null || return 0
  mv -f "${STATUS_FILE}.tmp" "${STATUS_FILE}" 2>/dev/null || true
}

# Did the user press Cancel in the progress window?
cancelled() { [ -f "${CANCEL_FILE}" ]; }

# Abort cleanly on cancel: remove the partial env so the NEXT attempt starts
# from a clean `create` rather than inheriting a half-built environment.
check_cancel() {
  cancelled || return 0
  trap - ERR
  say "Installation cancelled. Cleaning up the partial environment..."
  rm -rf "${ENV_PREFIX}"
  rm -f "${CANCEL_FILE}"
  progress_state failed "Installation cancelled."
  sleep 2
  gui_stop
  echo "Nothing was left half-installed; re-run the installer to try again." >&2
  exit 1
}

err() {
  trap - ERR
  printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2
  progress_state failed "$*"
  # Leave the window up briefly so the user can read the failure and find the log.
  [ -n "${GUI_PID}" ] && sleep 12
  gui_stop
  exit 1
}

on_error() {
  local rc=$?
  trap - ERR
  printf '\n\033[1;31mSetup did not complete (exit %s).\033[0m\n' "${rc}" >&2
  printf 'If an earlier attempt was interrupted, force a clean rebuild with:\n' >&2
  printf '  HIBACHI_FORCE_REBUILD=1 bash %s\n\n' "$0" >&2
  progress_state failed "Setup failed. See ${INSTALL_DIR}/setup.log"
  [ -n "${GUI_PID}" ] && sleep 12
  gui_stop
}
trap on_error ERR
trap gui_stop EXIT

# ============================================================================ #
# The progress window (tkinter, embedded so this file stays self-contained)
# ============================================================================ #
write_gui_script() {
  mkdir -p "${INSTALL_DIR}"
  cat > "${GUI_SCRIPT}" <<'PYGUI'
"""HIBACHI installer progress window (spawned by install.sh; not imported).

Polls a small JSON status file written by the installer and renders a progress
bar, the current activity, and an elapsed timer. Deliberately tkinter-only: it
runs before the application's dependencies exist.

Two behaviours matter more than looks:
  * The bar creeps slowly towards `ceil` between real updates, so a long
    download never looks frozen -- the reason users were quitting mid-install.
  * Closing the window (or pressing Cancel) writes a cancel file instead of
    killing the installer, so install.sh can tear down the partial environment
    and leave a clean slate for the next attempt.
"""

import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

STATUS_FILE = sys.argv[1]
CANCEL_FILE = sys.argv[2]

POLL_MS = 250
CREEP_PER_SEC = 0.18          # % per second while waiting for the next update
EASE = 0.18                   # fraction of the remaining gap closed per tick


class ProgressWindow:
    def __init__(self, root):
        self.root = root
        self.shown = 0.0          # what the bar currently displays
        self.target = 0.0         # last pct reported by the installer
        self.ceil = 0.0           # do not creep past this
        self.started = time.time()
        self.finished = False
        self.cancelling = False

        root.title("Installing HIBACHI")
        root.resizable(False, False)
        w, h = 460, 250
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        tk.Label(root, text="HIBACHI", font=("Helvetica", 20, "bold")).pack(pady=(20, 2))
        tk.Label(root, text="Setting up for first use", font=("Helvetica", 11)).pack()

        self.title_var = tk.StringVar(value="Starting...")
        tk.Label(root, textvariable=self.title_var, font=("Helvetica", 11, "bold")).pack(pady=(14, 3))

        self.bar = ttk.Progressbar(
            root, orient="horizontal", length=390, mode="determinate", maximum=100
        )
        self.bar.pack(pady=2)

        self.detail_var = tk.StringVar(value="")
        tk.Label(
            root, textvariable=self.detail_var, font=("Helvetica", 9), fg="#666666",
            wraplength=390, justify="center",
        ).pack(pady=(3, 0))

        self.elapsed_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.elapsed_var, font=("Helvetica", 9), fg="#888888").pack()

        tk.Label(
            root,
            text="Downloading ~1 GB of scientific packages.\n"
                 "This takes several minutes -- please leave this window open.",
            font=("Helvetica", 9), fg="#888888", justify="center",
        ).pack(pady=(8, 4))

        self.button = tk.Button(root, text="Cancel", width=10, command=self.on_close)
        self.button.pack(pady=(0, 10))

        # Come to the front once, then stop being obnoxious about it.
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(1800, lambda: root.attributes("-topmost", False))
        except tk.TclError:
            pass

        self.tick()

    # -- status file -------------------------------------------------------- #
    def read_status(self):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None      # mid-write, absent or truncated: just keep creeping

    def tick(self):
        status = self.read_status()
        if status:
            state = status.get("state", "running")
            if state in ("done", "failed"):
                self.finish(state, str(status.get("title") or ""))
                return
            try:
                self.target = max(self.target, float(status.get("pct", 0)))
                self.ceil = max(self.ceil, float(status.get("ceil", self.target)))
            except (TypeError, ValueError):
                pass
            # While cancelling, keep our own message: the installer is still
            # reporting the work it was in the middle of, and overwriting the
            # "Cancelling..." text would look like the Cancel had been ignored.
            if not self.cancelling:
                self.title_var.set(str(status.get("title") or ""))
                self.detail_var.set(str(status.get("detail") or ""))

        # Ease towards the reported target; once caught up, creep slowly towards
        # the ceiling so the bar is always visibly alive.
        if self.shown < self.target:
            self.shown += max((self.target - self.shown) * EASE, 0.05)
        elif self.shown < self.ceil:
            self.shown = min(self.ceil, self.shown + CREEP_PER_SEC * POLL_MS / 1000.0)
        self.bar["value"] = min(self.shown, 100.0)

        secs = int(time.time() - self.started)
        self.elapsed_var.set(f"{secs // 60}:{secs % 60:02d} elapsed")

        self.root.after(POLL_MS, self.tick)

    # -- terminal states ---------------------------------------------------- #
    def finish(self, state, message):
        self.finished = True
        if state == "done":
            self.root.destroy()
            return
        self.bar["value"] = 0
        self.title_var.set("Setup failed")
        self.detail_var.set(message or "See setup.log for details.")
        self.button.configure(text="Close", state="normal", command=self.root.destroy)

    def on_close(self):
        if self.finished or self.cancelling:
            self.root.destroy()
            return
        self.cancelling = True
        self.title_var.set("Cancelling...")
        self.detail_var.set("Removing partially installed files.")
        self.button.configure(state="disabled")
        try:
            with open(CANCEL_FILE, "w", encoding="utf-8") as fh:
                fh.write("cancelled by user\n")
        except Exception:
            pass
        # install.sh notices the cancel file, cleans up, and writes state=failed,
        # which brings us back through finish(). Close on our own after a while
        # in case it is wedged inside a long download.
        self.root.after(20000, self.root.destroy)


def main():
    if sys.platform not in ("darwin", "win32") and not os.environ.get("DISPLAY"):
        return 0                    # headless Linux: console output only
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"[installer-gui] unavailable ({exc}); console output only.")
        return 0
    ProgressWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYGUI
}

# Find a Python that can actually open a tkinter window. Skips /usr/bin/python3
# on macOS: when the Command Line Tools are absent it is a stub that pops its own
# install dialog, which is exactly the kind of confusion we are trying to remove.
#
# The env's own interpreter is only offered once GUI_ALLOW_ENV_PY=1, i.e. after
# the environment is final. Before that the env is a deletion candidate (step 2
# rebuilds a half-built env), and running the window from a prefix we are about
# to `rm -rf` would pull Tcl/Tk's library files out from under a live process --
# the window could disappear mid-install, which is the exact failure this whole
# mechanism exists to prevent.
GUI_ALLOW_ENV_PY=0

pick_gui_python() {
  local cand candidates=""
  [ "${GUI_ALLOW_ENV_PY}" = "1" ] && candidates="${ENV_PY}"
  candidates="${candidates} $(command -v python3 2>/dev/null || true)"
  for cand in ${candidates}; do
    [ -n "${cand}" ] && [ -x "${cand}" ] || continue
    # Only macOS has the CLT stub problem; on Linux /usr/bin/python3 is real and
    # is usually the only interpreter available before the env is built.
    if [ "$(uname -s)" = "Darwin" ] && [ "${cand}" = "/usr/bin/python3" ]; then
      continue
    fi
    "${cand}" -c "import tkinter" >/dev/null 2>&1 || continue
    printf '%s' "${cand}"
    return 0
  done
  return 1
}

gui_start() {
  [ "${NO_GUI}" = "1" ] && return 0
  [ -n "${GUI_PID}" ] && return 0                  # already up
  local py
  py="$(pick_gui_python)" || return 0              # nothing usable yet; try later
  write_gui_script
  "${py}" "${GUI_SCRIPT}" "${STATUS_FILE}" "${CANCEL_FILE}" >/dev/null 2>&1 &
  GUI_PID=$!
  sleep 1                                          # let the window map before work starts
}

gui_stop() {
  [ -n "${GUI_PID}" ] || return 0
  kill "${GUI_PID}" >/dev/null 2>&1 || true
  wait "${GUI_PID}" 2>/dev/null || true
  GUI_PID=""
  rm -f "${GUI_SCRIPT}" "${STATUS_FILE}" "${STATUS_FILE}.tmp"
}

# ============================================================================ #
# Environment helpers
# ============================================================================ #

# Print the `pip:` subsection of environment.yml, one requirement per line.
# Prefers PyYAML when the env can import it; falls back to awk, because PyYAML
# is only a transitive dependency here (via napari) and so may itself be one of
# the packages missing from a half-built env.
pip_requirements() {
  if [ -x "${ENV_PY}" ] && "${ENV_PY}" -c "import yaml" >/dev/null 2>&1; then
    "${ENV_PY}" - "${ENV_YML}" <<'PY'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1])) or {}
for dep in spec.get("dependencies") or []:
    if isinstance(dep, dict) and dep.get("pip"):
        for req in dep["pip"]:
            req = str(req).strip()
            if req:
                print(req)
PY
  else
    awk '
      /^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ {
        match($0, /^[[:space:]]*/); pipind = RLENGTH; inpip = 1; next
      }
      inpip {
        if ($0 ~ /^[[:space:]]*(#.*)?$/) next
        match($0, /^[[:space:]]*/)
        if (RLENGTH <= pipind) { inpip = 0; next }
        line = $0
        sub(/^[[:space:]]*-[[:space:]]*/, "", line)
        sub(/[[:space:]]+#.*$/, "", line)     # strip trailing YAML comment
        gsub(/[[:space:]]+$/, "", line)
        if (length(line)) print line
      }
    ' "${ENV_YML}"
  fi
}

# environment.yml with the `pip:` subsection removed, so the conda-level solve
# can run on its own and we can drive (and report progress for) pip ourselves.
write_conda_only_yml() {
  awk '
    /^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ {
      match($0, /^[[:space:]]*/); pipind = RLENGTH; inpip = 1; next
    }
    inpip {
      if ($0 ~ /^[[:space:]]*(#.*)?$/) next
      match($0, /^[[:space:]]*/)
      if (RLENGTH <= pipind) { inpip = 0 } else { next }
    }
    { print }
  ' "${ENV_YML}" > "$1"
}

# Import every REQUIRED_MODULE in one interpreter. Prints one line per failure
# and returns non-zero if anything is missing. Always call from an `if`.
env_health_check() {
  [ -x "${ENV_PY}" ] || { echo "python interpreter missing at ${ENV_PY}"; return 1; }
  # shellcheck disable=SC2086  # word splitting of the module list is intended
  "${ENV_PY}" - ${REQUIRED_MODULES} <<'PY'
import importlib, sys
missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except BaseException as exc:   # ImportError, but also DLL/ABI load failures
        missing.append(f"  {name}  ({type(exc).__name__}: {exc})")
if missing:
    print("\n".join(missing))
    sys.exit(1)
PY
}

# An interrupted `pip install` can leave `~`-prefixed directories behind from a
# killed uninstall step. pip treats those as real packages, so clear them first.
prune_pip_debris() {
  local sp
  for sp in "${ENV_PREFIX}"/lib/python*/site-packages; do
    [ -d "${sp}" ] || continue
    find "${sp}" -maxdepth 1 -name '~*' -exec rm -rf {} + 2>/dev/null || true
  done
}

# ============================================================================ #
# 1. Platform + micromamba
# ============================================================================ #
mkdir -p "${INSTALL_DIR}"
rm -f "${CANCEL_FILE}"
gui_start                     # opens now if the machine already has a usable python3
progress 1 4 "Preparing" ""

say "Detecting platform"
OS="$(uname -s)"; ARCH="$(uname -m)"
case "${OS}-${ARCH}" in
  Linux-x86_64)   PLATFORM="linux-64" ;;
  Linux-aarch64)  PLATFORM="linux-aarch64" ;;
  Darwin-x86_64)  PLATFORM="osx-64" ;;
  Darwin-arm64)   PLATFORM="osx-arm64" ;;
  *) err "Unsupported platform: ${OS}-${ARCH}" ;;
esac
echo "Platform: ${PLATFORM}"

if [ ! -x "${MAMBA_BIN}" ]; then
  say "Downloading micromamba"
  progress 2 5 "Downloading package manager" "micromamba (${PLATFORM})"
  mkdir -p "${MAMBA_ROOT}"
  # The official endpoint streams a tarball containing bin/micromamba.
  curl -fsSL "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" \
    | tar -xj -C "${MAMBA_ROOT}" bin/micromamba
else
  echo "micromamba already present."
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
check_cancel

say "Fetching dependency list"
progress 5 6 "Fetching dependency list" ""
curl -fsSL "${ENV_YML_URL}" -o "${ENV_YML}" \
  || err "Could not download environment.yml from ${ENV_YML_URL}"

# ============================================================================ #
# 2. Validate any existing environment
# ============================================================================ #
if [ -d "${ENV_PREFIX}" ] && [ "${FORCE_REBUILD}" = "1" ]; then
  say "Discarding the existing environment (HIBACHI_FORCE_REBUILD=1)"
  progress 6 8 "Removing the old environment" ""
  rm -rf "${ENV_PREFIX}"
elif [ -d "${ENV_PREFIX}" ]; then
  say "Checking the existing '${ENV_NAME}' environment"
  progress 6 10 "Checking the existing environment" "importing required packages"
  if health_output="$(env_health_check 2>&1)"; then
    echo "Environment looks complete."
  else
    warn "The existing environment is incomplete (likely an interrupted install):"
    printf '%s\n' "${health_output}" >&2
    say "Rebuilding the '${ENV_NAME}' environment from scratch"
    progress 7 10 "Repairing a previous interrupted install" "rebuilding from scratch"
    rm -rf "${ENV_PREFIX}"
  fi
fi
check_cancel

# ============================================================================ #
# 3. Phase A: conda-level packages (python, git, pip)
# ============================================================================ #
CONDA_ONLY_YML="${INSTALL_DIR}/.environment-conda-only.yml"
write_conda_only_yml "${CONDA_ONLY_YML}"

if [ -d "${ENV_PREFIX}" ]; then
  say "Updating the '${ENV_NAME}' environment (conda-level packages)"
  progress 10 28 "Updating Python environment" "this can take a minute"
  "${MAMBA_BIN}" env update -n "${ENV_NAME}" -f "${CONDA_ONLY_YML}" -y
else
  say "Creating the '${ENV_NAME}' environment (conda-level packages)"
  progress 10 28 "Creating Python environment" "downloading Python, git and pip"
  "${MAMBA_BIN}" create -y -n "${ENV_NAME}" -f "${CONDA_ONLY_YML}"
fi
rm -f "${CONDA_ONLY_YML}"

[ -x "${ENV_PY}" ] || err "Environment build did not produce an interpreter at ${ENV_PY}"
check_cancel

# The env's Python now exists and is final (nothing deletes it past this point),
# so it is safe to use for the window. If we could not open one earlier -- no
# system python3 with tkinter -- this is the moment, right before the slow phase.
GUI_ALLOW_ENV_PY=1
gui_start
progress 30 32 "Preparing to install packages" ""

# ============================================================================ #
# 4. Phase B: the pip: subsection, with live progress
# ============================================================================ #
# Always explicit, on both the create and update paths: `micromamba env update`
# skips the pip subsection when the conda solve is a no-op, which is what made
# interrupted installs unrepairable. Version specifiers mean pip no-ops on
# already-satisfied packages, so this is safe to run every time.
say "Installing scientific packages (the slow part)"
prune_pip_debris
PIP_REQS_FILE="${INSTALL_DIR}/.pip-requirements.txt"
pip_requirements > "${PIP_REQS_FILE}"

if [ ! -s "${PIP_REQS_FILE}" ]; then
  warn "No pip: subsection found in ${ENV_YML}; skipping the pip phase."
else
  : > "${PIP_LOG}"
  # Parse pip's own narration into progress updates. `Collecting <pkg>` fires
  # once per wheel it resolves, so counting those against EXPECTED_PKGS gives a
  # bar that tracks real work; the exact total does not matter, because the GUI
  # clamps to the phase ceiling and creeps between updates.
  set +e
  "${ENV_PY}" -m pip install --no-input --progress-bar off -r "${PIP_REQS_FILE}" 2>&1 \
    | tee -a "${PIP_LOG}" \
    | { n=0; pct=32
        while IFS= read -r line; do
          case "${line}" in
            *Collecting[[:space:]]*)
              pkg="${line#*Collecting }"
              pkg="${pkg%% (from *}"      # drop pip's "(from -r ... (line N))" noise
              n=$((n + 1))
              pct=$(awk -v n="${n}" -v t="${EXPECTED_PKGS}" \
                    'BEGIN { p = 32 + 46 * (n / t); printf "%.1f", (p > 78 ? 78 : p) }')
              progress "${pct}" 79 "Downloading packages" "${pkg}"
              ;;
            *"Installing collected packages"*)
              progress 80 92 "Installing packages" "unpacking wheels"
              ;;
            *"Successfully installed"*)
              progress 93 94 "Installed all packages" ""
              ;;
            ERROR:*)
              progress "${pct}" 79 "Resolving a problem" "${line}"
              ;;
          esac
        done; }
  pip_rc=${PIPESTATUS[0]}
  set -e
  if [ "${pip_rc}" -ne 0 ]; then
    check_cancel      # a Cancel mid-download kills pip; report it as a cancel
    err "Package installation failed (exit ${pip_rc}). Full log: ${PIP_LOG}"
  fi
fi
rm -f "${PIP_REQS_FILE}"
check_cancel

# ============================================================================ #
# 5. Verify before letting the caller record success
# ============================================================================ #
# The .app stub writes .bootstrapped_version only when this script exits 0, so
# failing here is what makes it report "setup failed" instead of recording a
# successful bootstrap over a broken install.
say "Verifying the environment"
progress 94 95 "Verifying installation" "importing required packages"
if health_output="$(env_health_check 2>&1)"; then
  echo "All required packages import cleanly."
else
  printf '%s\n' "${health_output}" >&2
  err "Some packages are still missing (see setup.log); HIBACHI would crash on startup. Re-run with HIBACHI_FORCE_REBUILD=1."
fi

# ============================================================================ #
# 6. Clone (or force-sync) the application
# ============================================================================ #
if [ -d "${APP_DIR}/.git" ]; then
  say "Updating existing checkout to the latest ${BRANCH}"
  progress 95 97 "Updating HIBACHI" "branch ${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git -C "${APP_DIR}" fetch origin "${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git -C "${APP_DIR}" checkout "${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"
else
  say "Cloning ${REPO_URL} (branch: ${BRANCH})"
  progress 95 97 "Downloading HIBACHI" "branch ${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git clone --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

# ============================================================================ #
# 7. Create the double-click launcher
# ============================================================================ #
# Publish the chosen install dir so make_shortcuts.py builds a shortcut for THIS
# location rather than assuming the default (matches the Windows fix).
export HIBACHI_HOME="${INSTALL_DIR}"
if [ "${HIBACHI_SKIP_SHORTCUT:-0}" = "1" ]; then
  say "Skipping shortcut creation (launched from a native app bundle)"
else
  say "Creating desktop launcher"
  progress 98 99 "Creating launcher" ""
  # Prefer the env's own interpreter over `micromamba run`: it needs nothing on
  # PATH and doesn't swallow stdout/stderr the way `micromamba run` can.
  "${ENV_PY}" "${APP_DIR}/launcher/make_shortcuts.py"
fi

trap - ERR
progress_state done "Setup complete."
sleep 1                       # let the window read the final state and self-close
gui_stop

say "Done!"
cat <<EOF

HIBACHI is installed at:
  ${APP_DIR}

Launch it from your Applications menu / Desktop shortcut, or directly with:
  "${ENV_PREFIX}/bin/python" "${APP_DIR}/launcher/run_app.py"

It will check for updates automatically each time it starts.
EOF