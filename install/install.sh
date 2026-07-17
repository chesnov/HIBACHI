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
# This script is IDEMPOTENT: re-running it over an existing install updates the
# environment in place and force-syncs the checkout to the current release,
# instead of erroring out or leaving the old version behind. `set -e` makes any
# failure (e.g. a failed git fetch) abort loudly rather than silently proceed.
# =============================================================================
set -euo pipefail

# ------------------------- CONFIG (edit for your repo) ----------------------- #
GH_OWNER="${HIBACHI_OWNER:-chesnov}"        # <-- your GitHub username / org
GH_REPO="${HIBACHI_REPO:-HIBACHI}"           # <-- your GitHub repository name
BRANCH="${HIBACHI_BRANCH:-main}"          # branch to install / track
INSTALL_DIR="${HIBACHI_HOME:-$HOME/HIBACHI}"
ENV_NAME="hibachi"
# ----------------------------------------------------------------------------- #

REPO_URL="https://github.com/${GH_OWNER}/${GH_REPO}.git"
ENV_YML_URL="https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${BRANCH}/install/environment.yml"
MAMBA_ROOT="${INSTALL_DIR}/micromamba"
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
APP_DIR="${INSTALL_DIR}/app"
ENV_PREFIX="${MAMBA_ROOT}/envs/${ENV_NAME}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
err() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || err "curl is required but not found."
command -v tar  >/dev/null 2>&1 || err "tar is required but not found."

# --- 1. Detect platform and install micromamba ------------------------------- #
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
  mkdir -p "${MAMBA_ROOT}"
  # The official endpoint streams a tarball containing bin/micromamba.
  curl -fsSL "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" \
    | tar -xj -C "${MAMBA_ROOT}" bin/micromamba
else
  echo "micromamba already present."
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"

# --- 2. Build (or update) the environment from environment.yml --------------- #
say "Fetching dependency list"
mkdir -p "${INSTALL_DIR}"
ENV_YML="${INSTALL_DIR}/environment.yml"
curl -fsSL "${ENV_YML_URL}" -o "${ENV_YML}" \
  || err "Could not download environment.yml from ${ENV_YML_URL}"

# `create` is NOT idempotent (it aborts if the prefix already exists), so on a
# re-install we update the existing env in place instead. This is what lets a
# new installer safely run over an old one.
if [ -d "${ENV_PREFIX}" ]; then
  say "Updating the '${ENV_NAME}' environment"
  "${MAMBA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_YML}" -y
else
  say "Creating the '${ENV_NAME}' environment (first run downloads packages; be patient)"
  "${MAMBA_BIN}" create -y -n "${ENV_NAME}" -f "${ENV_YML}"
fi

# --- 3. Clone (or force-sync) the application -------------------------------- #
# A failure here aborts the whole script (set -e), so the caller/.app launcher
# reports "setup failed" instead of silently launching the stale checkout.
if [ -d "${APP_DIR}/.git" ]; then
  say "Updating existing checkout to the latest ${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git -C "${APP_DIR}" fetch origin "${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git -C "${APP_DIR}" checkout "${BRANCH}"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"
else
  say "Cloning ${REPO_URL} (branch: ${BRANCH})"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" git clone --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

# --- 4. Create the double-click launcher ------------------------------------- #
if [ "${HIBACHI_SKIP_SHORTCUT:-0}" = "1" ]; then
  say "Skipping shortcut creation (launched from a native app bundle)"
else
  say "Creating desktop launcher"
  "${MAMBA_BIN}" run -n "${ENV_NAME}" python "${APP_DIR}/launcher/make_shortcuts.py"
fi

say "Done!"
cat <<EOF

HIBACHI is installed at:
  ${APP_DIR}

Launch it from your Applications menu / Desktop shortcut, or directly with:
  "${MAMBA_BIN}" run -n ${ENV_NAME} python "${APP_DIR}/launcher/run_app.py"

It will check for updates automatically each time it starts.
EOF
