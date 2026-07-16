# =============================================================================
# HIBACHI installer for Windows (PowerShell)
# -----------------------------------------------------------------------------
# One-line install (from your GitHub README), run in PowerShell:
#
#   iwr -useb https://raw.githubusercontent.com/chesnov/HIBACHI/main/install/install.ps1 | iex
#
# No admin rights required; nothing is installed system-wide.
# =============================================================================
$ErrorActionPreference = "Stop"

# ------------------------- CONFIG (edit for your repo) ----------------------- #
$GhOwner    = if ($env:HIBACHI_OWNER)  { $env:HIBACHI_OWNER }  else { "chesnov" }
$GhRepo     = if ($env:HIBACHI_REPO)   { $env:HIBACHI_REPO }   else { "HIBACHI" }
$Branch     = if ($env:HIBACHI_BRANCH) { $env:HIBACHI_BRANCH } else { "main" }
$InstallDir = if ($env:HIBACHI_HOME)   { $env:HIBACHI_HOME }   else { Join-Path $env:USERPROFILE "HIBACHI" }
$EnvName    = "hibachi"
# ----------------------------------------------------------------------------- #

$RepoUrl   = "https://github.com/$GhOwner/$GhRepo.git"
$EnvYmlUrl = "https://raw.githubusercontent.com/$GhOwner/$GhRepo/$Branch/install/environment.yml"
$MambaRoot = Join-Path $InstallDir "micromamba"
$MambaBin  = Join-Path $MambaRoot "micromamba.exe"
$AppDir    = Join-Path $InstallDir "app"

function Say($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --- 1. Install micromamba --------------------------------------------------- #
Say "Downloading micromamba"
New-Item -ItemType Directory -Force -Path $MambaRoot | Out-Null
if (-not (Test-Path $MambaBin)) {
    # win-64 tarball contains Library\bin\micromamba.exe
    $tmp = Join-Path $env:TEMP "micromamba.tar.bz2"
    Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile $tmp -UseBasicParsing
    tar -xf $tmp -C $MambaRoot
    $found = Get-ChildItem -Path $MambaRoot -Recurse -Filter "micromamba.exe" | Select-Object -First 1
    if ($null -eq $found) { throw "micromamba.exe not found after extraction." }
    Copy-Item $found.FullName $MambaBin -Force
}
$env:MAMBA_ROOT_PREFIX = $MambaRoot

# --- 2. Build environment ---------------------------------------------------- #
Say "Fetching dependency list"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$EnvYml = Join-Path $InstallDir "environment.yml"
Invoke-WebRequest -Uri $EnvYmlUrl -OutFile $EnvYml -UseBasicParsing

Say "Creating the '$EnvName' environment (first run downloads packages; be patient)"
& $MambaBin create -y -n $EnvName -f $EnvYml

# --- 3. Clone / update the application --------------------------------------- #
if (Test-Path (Join-Path $AppDir ".git")) {
    Say "Updating existing checkout"
    & $MambaBin run -n $EnvName git -C $AppDir fetch origin $Branch
    & $MambaBin run -n $EnvName git -C $AppDir checkout $Branch
    & $MambaBin run -n $EnvName git -C $AppDir reset --hard "origin/$Branch"
} else {
    Say "Cloning $RepoUrl (branch: $Branch)"
    & $MambaBin run -n $EnvName git clone --branch $Branch $RepoUrl $AppDir
}

# --- 4. Desktop launcher ----------------------------------------------------- #
Say "Creating desktop launcher"
& $MambaBin run -n $EnvName python (Join-Path $AppDir "launcher\make_shortcuts.py")

Say "Done!"
Write-Host @"

HIBACHI is installed at:
  $AppDir

Launch it from the Desktop shortcut (HIBACHI), or directly with:
  "$MambaBin" run -n $EnvName pythonw "$AppDir\launcher\run_app.py"

It will check for updates automatically each time it starts.
"@
