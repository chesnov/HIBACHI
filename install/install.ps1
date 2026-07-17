# =============================================================================
# HIBACHI installer for Windows (PowerShell)
# -----------------------------------------------------------------------------
# One-line install (from your GitHub README), run in PowerShell:
#
#   iwr -useb https://raw.githubusercontent.com/chesnov/HIBACHI/main/install/install.ps1 | iex
#
# No admin rights required; nothing is installed system-wide.
#
# Install location (highest priority first):
#   1. -InstallDir <path>   (the Inno Setup wizard passes the user's chosen {app})
#   2. $env:HIBACHI_HOME
#   3. default: %USERPROFILE%\HIBACHI
#
# This script is IDEMPOTENT and FAILS LOUDLY: re-running it over an existing
# install updates the env in place and force-syncs the checkout. Native command
# (git / micromamba) exit codes are checked explicitly, because
# $ErrorActionPreference = "Stop" does NOT catch non-zero exits from external
# programs in Windows PowerShell 5.1 -- which is exactly how a failed `git fetch`
# used to slip through and leave the old version installed while reporting success.
# =============================================================================
param([string]$InstallDir = "")

$ErrorActionPreference = "Stop"

# ------------------------- CONFIG (edit for your repo) ----------------------- #
$GhOwner    = if ($env:HIBACHI_OWNER)  { $env:HIBACHI_OWNER }  else { "chesnov" }
$GhRepo     = if ($env:HIBACHI_REPO)   { $env:HIBACHI_REPO }   else { "HIBACHI" }
$Branch     = if ($env:HIBACHI_BRANCH) { $env:HIBACHI_BRANCH } else { "main" }
if (-not $InstallDir) {
    $InstallDir = if ($env:HIBACHI_HOME) { $env:HIBACHI_HOME } else { Join-Path $env:USERPROFILE "HIBACHI" }
}
# Publish the chosen location so every child process (git, micromamba, and
# especially make_shortcuts.py) uses the SAME install dir instead of assuming
# the default. This is what makes a non-default install produce a working icon.
$env:HIBACHI_HOME = $InstallDir
$EnvName    = "hibachi"
# ----------------------------------------------------------------------------- #

$RepoUrl   = "https://github.com/$GhOwner/$GhRepo.git"
$EnvYmlUrl = "https://raw.githubusercontent.com/$GhOwner/$GhRepo/$Branch/install/environment.yml"
$MambaRoot = Join-Path $InstallDir "micromamba"
$MambaBin  = Join-Path $MambaRoot "micromamba.exe"
$AppDir    = Join-Path $InstallDir "app"
$EnvPrefix = Join-Path (Join-Path $MambaRoot "envs") $EnvName

function Say($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# Run a native command and abort if it returns a non-zero exit code. Without
# this, git/micromamba failures are silently ignored and the installer "succeeds"
# with the old version still in place.
function Assert-Native([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit code $LASTEXITCODE). The previous version was left untouched; fix the problem (often no internet / a proxy) and re-run the installer."
    }
}

# --- 1. Install micromamba --------------------------------------------------- #
Say "Downloading micromamba"
New-Item -ItemType Directory -Force -Path $MambaRoot | Out-Null
if (-not (Test-Path $MambaBin)) {
    # win-64 tarball contains Library\bin\micromamba.exe
    $tmp = Join-Path $env:TEMP "micromamba.tar.bz2"
    Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile $tmp -UseBasicParsing
    tar -xf $tmp -C $MambaRoot
    Assert-Native "Extracting micromamba"
    $found = Get-ChildItem -Path $MambaRoot -Recurse -Filter "micromamba.exe" | Select-Object -First 1
    if ($null -eq $found) { throw "micromamba.exe not found after extraction." }
    Copy-Item $found.FullName $MambaBin -Force
}
$env:MAMBA_ROOT_PREFIX = $MambaRoot

# --- 2. Build (or update) environment ---------------------------------------- #
Say "Fetching dependency list"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$EnvYml = Join-Path $InstallDir "environment.yml"
Invoke-WebRequest -Uri $EnvYmlUrl -OutFile $EnvYml -UseBasicParsing

# `create` aborts if the prefix already exists, so update in place on re-install.
if (Test-Path $EnvPrefix) {
    Say "Updating the '$EnvName' environment"
    & $MambaBin env update -n $EnvName -f $EnvYml -y
    Assert-Native "Updating the '$EnvName' environment"
} else {
    Say "Creating the '$EnvName' environment (first run downloads packages; be patient)"
    & $MambaBin create -y -n $EnvName -f $EnvYml
    Assert-Native "Creating the '$EnvName' environment"
}

# --- 3. Clone / force-sync the application ----------------------------------- #
if (Test-Path (Join-Path $AppDir ".git")) {
    Say "Updating existing checkout to the latest $Branch"
    & $MambaBin run -n $EnvName git -C $AppDir fetch origin $Branch
    Assert-Native "git fetch"
    & $MambaBin run -n $EnvName git -C $AppDir checkout $Branch
    Assert-Native "git checkout"
    & $MambaBin run -n $EnvName git -C $AppDir reset --hard "origin/$Branch"
    Assert-Native "git reset --hard origin/$Branch"
} else {
    Say "Cloning $RepoUrl (branch: $Branch)"
    & $MambaBin run -n $EnvName git clone --branch $Branch $RepoUrl $AppDir
    Assert-Native "git clone"
}

# --- 4. Desktop launcher ----------------------------------------------------- #
Say "Creating desktop launcher"
& $MambaBin run -n $EnvName python (Join-Path $AppDir "launcher\make_shortcuts.py")
Assert-Native "Creating desktop launcher"

Say "Done!"
Write-Host @"

HIBACHI is installed at:
  $AppDir

Launch it from the Desktop shortcut (HIBACHI), or directly with:
  "$MambaBin" run -n $EnvName pythonw "$AppDir\launcher\run_app.py"

It will check for updates automatically each time it starts.
"@
