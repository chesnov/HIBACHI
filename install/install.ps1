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
#
# =============================================================================
# WHY THERE IS A PROGRESS WINDOW  (the most important thing in this file)
# -----------------------------------------------------------------------------
# The first install downloads ~1 GB of scientific packages and takes several
# minutes. Users who cannot see progress conclude the app has glitched and quit
# mid-install, which leaves a half-built environment that (before the fix below)
# could never be repaired by retrying -- the app then crashed on every launch,
# deep in an import, with no visible error.
#
# So this script raises a real WinForms progress window with a moving bar, the
# current package name and an elapsed timer. It is EMBEDDED in this file as a
# here-string rather than shipped as a separate module, because this script must
# work in two contexts where a sibling file would not exist: piped straight from
# `iwr | iex`, and copied alone into the Inno Setup payload.
#
# WinForms (not tkinter) is deliberate: it needs no Python, so the window is up
# from the first second -- including during the conda-level solve, before any
# interpreter exists. This is also why hibachi.iss now runs this script HIDDEN
# instead of showing a console: the window is the progress UI on every OS.
#
# -----------------------------------------------------------------------------
# RECOVERY FROM A HALF-BUILT ENVIRONMENT
# -----------------------------------------------------------------------------
# The environment build is split in two, matching install.sh:
#   phase A: conda-level packages (python, git, pip)  -- fast
#   phase B: the pip: subsection (napari, PyQt5, ...) -- slow
# That split fixes the original trap. The old logic branched on "does the env
# prefix exist?" and took `micromamba env update` on every retry -- which skips
# the pip: subsection entirely when the conda-level solve finds nothing to do.
# Missing pip packages were therefore never reinstalled, no matter how often the
# user re-ran the installer.
#
# Now pip is ALWAYS run explicitly by us, on both the create and the update
# path, mirroring pass 2 of run_app.py::_update_environment(). On top of that we
# validate an existing env by importing what the app needs and rebuild from
# scratch if the probe fails, and Cancel deletes the partial env so the next
# attempt starts clean.
# =============================================================================
param(
    [string]$InstallDir = "",
    # Headless / CI: skip the progress window and log to the console only.
    [switch]$NoGui
)

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
# Set HIBACHI_FORCE_REBUILD=1 to discard any existing environment and rebuild.
$ForceRebuild = ($env:HIBACHI_FORCE_REBUILD -eq "1")
if ($env:HIBACHI_NO_INSTALLER_GUI -eq "1") { $NoGui = $true }
# Rough number of wheels pip ends up installing, used to scale the bar during
# the download phase. Only affects bar smoothness, never correctness.
$ExpectedPkgs = if ($env:HIBACHI_EXPECTED_PKGS) { [int]$env:HIBACHI_EXPECTED_PKGS } else { 220 }
# ----------------------------------------------------------------------------- #

$RepoUrl   = "https://github.com/$GhOwner/$GhRepo.git"
$EnvYmlUrl = "https://raw.githubusercontent.com/$GhOwner/$GhRepo/$Branch/install/environment.yml"
$MambaRoot = Join-Path $InstallDir "micromamba"
$MambaBin  = Join-Path $MambaRoot "micromamba.exe"
$AppDir    = Join-Path $InstallDir "app"
$EnvPrefix = Join-Path (Join-Path $MambaRoot "envs") $EnvName
$EnvPy     = Join-Path $EnvPrefix "python.exe"
$EnvPyw    = Join-Path $EnvPrefix "pythonw.exe"
$EnvYml    = Join-Path $InstallDir "environment.yml"

$GuiScript  = Join-Path $InstallDir ".installer_progress.ps1"
$StatusFile = Join-Path $InstallDir ".installer_status.json"
$CancelFile = Join-Path $InstallDir ".installer_cancelled"
$PipLog     = Join-Path $InstallDir "pip-install.log"
$SetupLog   = Join-Path $InstallDir "setup.log"
$GuiProc    = $null

# Modules the app imports at startup, in roughly the order the failing chain
# hits them (segment.py -> utils.high_level_gui.helper_funcs -> app_launch,
# relational_engine, cross_channel_window, metadata, ...). Import names, not
# distribution names: scikit-learn -> sklearn, simpleitk -> SimpleITK, etc.
# Probing these is what distinguishes a complete env from a half-built one.
$RequiredModules = @(
    "yaml", "numpy", "pandas", "scipy", "tifffile", "PyQt5.QtWidgets", "vispy",
    "napari", "magicgui", "dask.array", "dask_image.ndmeasure", "sklearn",
    "seaborn", "skan", "SimpleITK", "slideio", "numba", "zarr", "plotly",
    "nbformat", "napari_animation", "aicspylibczi", "fcswrite",
    "PartSegCore_compiled_backend"
)

function Say($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "`nWARNING: $msg" -ForegroundColor Yellow }

# ============================================================================ #
# Progress reporting
# ============================================================================ #

# Write-Progress-File <pct> <ceil> <title> <detail>
# Written atomically (the GUI polls it 4x/second). `ceil` is the upper bound the
# GUI may creep towards while waiting for the next update -- that slow creep is
# what keeps the bar from ever looking frozen.
function Write-ProgressFile([double]$Pct, [double]$Ceil, [string]$Title, [string]$Detail = "", [string]$State = "running") {
    if (-not (Test-Path $InstallDir)) { return }
    $t = ($Title -replace '"', '').Replace('\', '/')
    $d = ($Detail -replace '"', '').Replace('\', '/')
    $json = '{{"pct": {0}, "ceil": {1}, "title": "{2}", "detail": "{3}", "state": "{4}"}}' -f `
        [math]::Round($Pct, 1), [math]::Round($Ceil, 1), $t, $d, $State
    try {
        Set-Content -LiteralPath "$StatusFile.tmp" -Value $json -Encoding UTF8 -Force
        Move-Item -LiteralPath "$StatusFile.tmp" -Destination $StatusFile -Force
    } catch { return }
    if ($Detail) { Write-Host "    $Title -- $Detail" } else { Write-Host "    $Title" }
}

function Test-Cancelled { return (Test-Path $CancelFile) }

# Abort cleanly on cancel: remove the partial env so the NEXT attempt starts
# from a clean `create` rather than inheriting a half-built environment.
function Assert-NotCancelled {
    if (-not (Test-Cancelled)) { return }
    Say "Installation cancelled. Cleaning up the partial environment..."
    Remove-Item -LiteralPath $EnvPrefix -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $CancelFile -Force -ErrorAction SilentlyContinue
    Write-ProgressFile 100 100 "Installation cancelled." "" "failed"
    Start-Sleep -Seconds 2
    Stop-Gui
    Write-Host "Nothing was left half-installed; re-run the installer to try again."
    exit 1
}

function Fail([string]$Message) {
    Write-Host "`nERROR: $Message" -ForegroundColor Red
    Write-ProgressFile 100 100 $Message "" "failed"
    # Leave the window up briefly so the user can read the failure and find the log.
    if ($GuiProc) { Start-Sleep -Seconds 12 }
    Stop-Gui
    exit 1
}

# Run a native command and abort if it returns a non-zero exit code. Without
# this, git/micromamba failures are silently ignored and the installer "succeeds"
# with the old version still in place.
function Assert-Native([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        Fail "$What failed (exit code $LASTEXITCODE). The previous version was left untouched; fix the problem (often no internet / a proxy) and re-run the installer."
    }
}

# ============================================================================ #
# The progress window (WinForms, embedded so this file stays self-contained)
# ============================================================================ #
function Write-GuiScript {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $gui = @'
# HIBACHI installer progress window (spawned by install.ps1; not dot-sourced).
#
# Polls a small JSON status file written by the installer and renders a progress
# bar, the current activity, and an elapsed timer. WinForms rather than tkinter
# so it needs no Python and can be shown before the environment exists.
#
# Two behaviours matter more than looks:
#   * The bar creeps slowly towards `ceil` between real updates, so a long
#     download never looks frozen -- the reason users were quitting mid-install.
#   * Closing the window (or pressing Cancel) writes a cancel file instead of
#     killing the installer, so install.ps1 can tear down the partial
#     environment and leave a clean slate for the next attempt.
param([string]$StatusFile, [string]$CancelFile)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:shown      = 0.0    # what the bar currently displays
$script:target     = 0.0    # last pct reported by the installer
$script:ceiling    = 0.0    # do not creep past this
$script:started    = Get-Date
$script:finished   = $false
$script:cancelling = $false

$POLL_MS       = 250
$CREEP_PER_SEC = 0.18       # % per second while waiting for the next update
$EASE          = 0.18       # fraction of the remaining gap closed per tick

$form                 = New-Object System.Windows.Forms.Form
$form.Text            = "Installing HIBACHI"
$form.Size            = New-Object System.Drawing.Size(470, 285)
$form.StartPosition   = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox     = $false
$form.MinimizeBox     = $false
$form.TopMost         = $true

$heading           = New-Object System.Windows.Forms.Label
$heading.Text      = "HIBACHI"
$heading.Font      = New-Object System.Drawing.Font("Segoe UI", 18, [System.Drawing.FontStyle]::Bold)
$heading.TextAlign = "MiddleCenter"
$heading.Location  = New-Object System.Drawing.Point(20, 18)
$heading.Size      = New-Object System.Drawing.Size(410, 32)
$form.Controls.Add($heading)

$sub           = New-Object System.Windows.Forms.Label
$sub.Text      = "Setting up for first use"
$sub.Font      = New-Object System.Drawing.Font("Segoe UI", 9)
$sub.TextAlign = "MiddleCenter"
$sub.Location  = New-Object System.Drawing.Point(20, 50)
$sub.Size      = New-Object System.Drawing.Size(410, 18)
$form.Controls.Add($sub)

$titleLabel           = New-Object System.Windows.Forms.Label
$titleLabel.Text      = "Starting..."
$titleLabel.Font      = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
$titleLabel.TextAlign = "MiddleCenter"
$titleLabel.Location  = New-Object System.Drawing.Point(20, 80)
$titleLabel.Size      = New-Object System.Drawing.Size(410, 18)
$form.Controls.Add($titleLabel)

$bar          = New-Object System.Windows.Forms.ProgressBar
$bar.Style    = "Continuous"
$bar.Minimum  = 0
$bar.Maximum  = 1000                  # x10 for smooth sub-percent movement
$bar.Location = New-Object System.Drawing.Point(30, 104)
$bar.Size     = New-Object System.Drawing.Size(390, 18)
$form.Controls.Add($bar)

$detailLabel           = New-Object System.Windows.Forms.Label
$detailLabel.Text      = ""
$detailLabel.Font      = New-Object System.Drawing.Font("Segoe UI", 8)
$detailLabel.ForeColor = [System.Drawing.Color]::DimGray
$detailLabel.TextAlign = "MiddleCenter"
$detailLabel.Location  = New-Object System.Drawing.Point(20, 128)
$detailLabel.Size      = New-Object System.Drawing.Size(410, 30)
$form.Controls.Add($detailLabel)

$elapsedLabel           = New-Object System.Windows.Forms.Label
$elapsedLabel.Text      = ""
$elapsedLabel.Font      = New-Object System.Drawing.Font("Segoe UI", 8)
$elapsedLabel.ForeColor = [System.Drawing.Color]::Gray
$elapsedLabel.TextAlign = "MiddleCenter"
$elapsedLabel.Location  = New-Object System.Drawing.Point(20, 158)
$elapsedLabel.Size      = New-Object System.Drawing.Size(410, 16)
$form.Controls.Add($elapsedLabel)

$note           = New-Object System.Windows.Forms.Label
$note.Text      = "Downloading ~1 GB of scientific packages." + [Environment]::NewLine + "This takes several minutes - please leave this window open."
$note.Font      = New-Object System.Drawing.Font("Segoe UI", 8)
$note.ForeColor = [System.Drawing.Color]::Gray
$note.TextAlign = "MiddleCenter"
$note.Location  = New-Object System.Drawing.Point(20, 180)
$note.Size      = New-Object System.Drawing.Size(410, 32)
$form.Controls.Add($note)

$button          = New-Object System.Windows.Forms.Button
$button.Text     = "Cancel"
$button.Size     = New-Object System.Drawing.Size(90, 26)
$button.Location = New-Object System.Drawing.Point(190, 216)
$form.Controls.Add($button)

function Request-Cancel {
    if ($script:finished -or $script:cancelling) { $form.Close(); return }
    $script:cancelling = $true
    $titleLabel.Text   = "Cancelling..."
    $detailLabel.Text  = "Removing partially installed files."
    $button.Enabled    = $false
    try { Set-Content -LiteralPath $CancelFile -Value "cancelled by user" -Force } catch { }
    # install.ps1 notices the cancel file, cleans up, and writes state=failed,
    # which brings us through Complete-Run. Close on our own after a while in
    # case it is wedged inside a long download.
    $bail          = New-Object System.Windows.Forms.Timer
    $bail.Interval = 20000
    $bail.Add_Tick({ $bail.Stop(); $form.Close() })
    $bail.Start()
}

function Complete-Run([string]$State, [string]$Message) {
    $script:finished = $true
    if ($State -eq "done") { $form.Close(); return }
    $bar.Value        = 0
    $titleLabel.Text  = "Setup failed"
    $detailLabel.Text = if ($Message) { $Message } else { "See setup.log for details." }
    $button.Text      = "Close"
    $button.Enabled   = $true
}

$button.Add_Click({
    if ($script:finished) { $form.Close() } else { Request-Cancel }
})
$form.Add_FormClosing({
    param($sender, $e)
    if (-not $script:finished -and -not $script:cancelling) {
        $e.Cancel = $true       # do not just die: hand off to install.ps1 first
        Request-Cancel
    }
})

$timer          = New-Object System.Windows.Forms.Timer
$timer.Interval = $POLL_MS
$timer.Add_Tick({
    $status = $null
    try {
        $raw = Get-Content -LiteralPath $StatusFile -Raw -ErrorAction Stop
        if ($raw) { $status = $raw | ConvertFrom-Json }
    } catch {
        $status = $null          # mid-write, absent or truncated: keep creeping
    }

    if ($status) {
        $state = if ($status.state) { [string]$status.state } else { "running" }
        if ($state -eq "done" -or $state -eq "failed") {
            $timer.Stop()
            Complete-Run $state ([string]$status.title)
            return
        }
        try {
            $p = [double]$status.pct
            $c = if ($null -ne $status.ceil) { [double]$status.ceil } else { $p }
            if ($p -gt $script:target)  { $script:target  = $p }
            if ($c -gt $script:ceiling) { $script:ceiling = $c }
        } catch { }
        # While cancelling, keep our own message: the installer is still
        # reporting the work it was in the middle of, and overwriting the
        # "Cancelling..." text would look like the Cancel had been ignored.
        if (-not $script:cancelling) {
            $titleLabel.Text  = [string]$status.title
            $detailLabel.Text = [string]$status.detail
        }
    }

    # Ease towards the reported target; once caught up, creep slowly towards the
    # ceiling so the bar is always visibly alive.
    if ($script:shown -lt $script:target) {
        $step = ($script:target - $script:shown) * $EASE
        if ($step -lt 0.05) { $step = 0.05 }
        $script:shown = $script:shown + $step
    } elseif ($script:shown -lt $script:ceiling) {
        $script:shown = [math]::Min($script:ceiling, $script:shown + $CREEP_PER_SEC * $POLL_MS / 1000.0)
    }
    if ($script:shown -gt 100) { $script:shown = 100 }
    $bar.Value = [int]($script:shown * 10)

    $secs = [int]((Get-Date) - $script:started).TotalSeconds
    $elapsedLabel.Text = "{0}:{1:D2} elapsed" -f [math]::Floor($secs / 60), ($secs % 60)
})
$timer.Start()

# Come to the front once, then stop being obnoxious about it.
$untop          = New-Object System.Windows.Forms.Timer
$untop.Interval = 1800
$untop.Add_Tick({ $untop.Stop(); $form.TopMost = $false })
$untop.Start()

[void][System.Windows.Forms.Application]::Run($form)
'@
    Set-Content -LiteralPath $GuiScript -Value $gui -Encoding UTF8 -Force
}

function Start-Gui {
    if ($NoGui -or $GuiProc) { return }
    try {
        Write-GuiScript
        $psArgs = @(
            "-ExecutionPolicy", "Bypass", "-NoProfile", "-Sta",
            "-WindowStyle", "Hidden",
            "-File", $GuiScript, "-StatusFile", $StatusFile, "-CancelFile", $CancelFile
        )
        $script:GuiProc = Start-Process -FilePath "powershell.exe" -ArgumentList $psArgs `
            -WindowStyle Hidden -PassThru
        Start-Sleep -Milliseconds 900        # let the window paint before work starts
    } catch {
        Warn "Could not open the progress window ($($_.Exception.Message)); continuing with console output."
        $script:GuiProc = $null
    }
}

function Stop-Gui {
    if (-not $GuiProc) { return }
    try { Stop-Process -Id $GuiProc.Id -Force -ErrorAction SilentlyContinue } catch { }
    $script:GuiProc = $null
    Remove-Item -LiteralPath $GuiScript, $StatusFile, "$StatusFile.tmp" -Force -ErrorAction SilentlyContinue
}

# ============================================================================ #
# Environment helpers
# ============================================================================ #

# The `pip:` subsection of environment.yml, one requirement per line. Parsed in
# pure PowerShell rather than via PyYAML, because on a half-built env PyYAML may
# itself be one of the missing packages (it is only a transitive dependency here,
# via napari).
function Get-PipRequirements([string]$Path) {
    $reqs   = @()
    $inPip  = $false
    $pipInd = 0
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^(\s*)-\s*pip:\s*$') {
            $inPip  = $true
            $pipInd = $Matches[1].Length
            continue
        }
        if (-not $inPip) { continue }
        if ($line -match '^\s*(#.*)?$') { continue }          # blank / comment
        $indent = ($line -replace '^(\s*).*$', '$1').Length
        if ($indent -le $pipInd) { $inPip = $false; continue } # block ended
        $req = $line -replace '^\s*-\s*', ''
        $req = $req -replace '\s+#.*$', ''                     # trailing comment
        $req = $req.Trim()
        if ($req) { $reqs += $req }
    }
    return $reqs
}

# environment.yml with the `pip:` subsection removed, so the conda-level solve
# can run on its own and we can drive (and report progress for) pip ourselves.
function Write-CondaOnlyYml([string]$Source, [string]$Destination) {
    $out    = @()
    $inPip  = $false
    $pipInd = 0
    foreach ($line in Get-Content -LiteralPath $Source) {
        if ($line -match '^(\s*)-\s*pip:\s*$') {
            $inPip  = $true
            $pipInd = $Matches[1].Length
            continue
        }
        if ($inPip) {
            if ($line -match '^\s*(#.*)?$') { continue }
            $indent = ($line -replace '^(\s*).*$', '$1').Length
            if ($indent -le $pipInd) { $inPip = $false } else { continue }
        }
        $out += $line
    }
    Set-Content -LiteralPath $Destination -Value $out -Encoding UTF8 -Force
}

# Import every required module in one interpreter. Returns the failures as text
# (empty string means healthy).
function Get-EnvHealth {
    if (-not (Test-Path $EnvPy)) { return "python interpreter missing at $EnvPy" }
    $probe = @'
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
'@
    $probeFile = Join-Path $InstallDir ".env_probe.py"
    Set-Content -LiteralPath $probeFile -Value $probe -Encoding UTF8 -Force
    $out = & $EnvPy $probeFile @RequiredModules 2>&1
    $rc  = $LASTEXITCODE
    Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
    if ($rc -eq 0) { return "" }
    return ($out | Out-String)
}

# An interrupted `pip install` can leave `~`-prefixed directories behind from a
# killed uninstall step. pip treats those as real packages, so clear them first.
function Remove-PipDebris {
    $sp = Join-Path $EnvPrefix "Lib\site-packages"
    if (-not (Test-Path $sp)) { return }
    Get-ChildItem -LiteralPath $sp -Filter "~*" -Force -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
}

# ============================================================================ #
# 1. micromamba
# ============================================================================ #
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Remove-Item -LiteralPath $CancelFile -Force -ErrorAction SilentlyContinue
Start-Gui                     # WinForms needs no Python: the window opens now
Write-ProgressFile 1 4 "Preparing" ""

try {
    Say "Downloading micromamba"
    New-Item -ItemType Directory -Force -Path $MambaRoot | Out-Null
    if (-not (Test-Path $MambaBin)) {
        Write-ProgressFile 2 5 "Downloading package manager" "micromamba (win-64)"
        # win-64 tarball contains Library\bin\micromamba.exe
        $tmp = Join-Path $env:TEMP "micromamba.tar.bz2"
        Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile $tmp -UseBasicParsing
        tar -xf $tmp -C $MambaRoot
        Assert-Native "Extracting micromamba"
        $found = Get-ChildItem -Path $MambaRoot -Recurse -Filter "micromamba.exe" | Select-Object -First 1
        if ($null -eq $found) { Fail "micromamba.exe not found after extraction." }
        Copy-Item $found.FullName $MambaBin -Force
    }
    $env:MAMBA_ROOT_PREFIX = $MambaRoot
    Assert-NotCancelled

    Say "Fetching dependency list"
    Write-ProgressFile 5 6 "Fetching dependency list" ""
    Invoke-WebRequest -Uri $EnvYmlUrl -OutFile $EnvYml -UseBasicParsing

    # ======================================================================== #
    # 2. Validate any existing environment
    # ======================================================================== #
    if ((Test-Path $EnvPrefix) -and $ForceRebuild) {
        Say "Discarding the existing environment (HIBACHI_FORCE_REBUILD=1)"
        Write-ProgressFile 6 8 "Removing the old environment" ""
        Remove-Item -LiteralPath $EnvPrefix -Recurse -Force
    } elseif (Test-Path $EnvPrefix) {
        Say "Checking the existing '$EnvName' environment"
        Write-ProgressFile 6 10 "Checking the existing environment" "importing required packages"
        $health = Get-EnvHealth
        if ($health) {
            Warn "The existing environment is incomplete (likely an interrupted install):"
            Write-Host $health
            Say "Rebuilding the '$EnvName' environment from scratch"
            Write-ProgressFile 7 10 "Repairing a previous interrupted install" "rebuilding from scratch"
            Remove-Item -LiteralPath $EnvPrefix -Recurse -Force
        } else {
            Write-Host "Environment looks complete."
        }
    }
    Assert-NotCancelled

    # ======================================================================== #
    # 3. Phase A: conda-level packages (python, git, pip)
    # ======================================================================== #
    $CondaOnlyYml = Join-Path $InstallDir ".environment-conda-only.yml"
    Write-CondaOnlyYml $EnvYml $CondaOnlyYml

    if (Test-Path $EnvPrefix) {
        Say "Updating the '$EnvName' environment (conda-level packages)"
        Write-ProgressFile 10 28 "Updating Python environment" "this can take a minute"
        & $MambaBin env update -n $EnvName -f $CondaOnlyYml -y
        Assert-Native "Updating the '$EnvName' environment"
    } else {
        Say "Creating the '$EnvName' environment (conda-level packages)"
        Write-ProgressFile 10 28 "Creating Python environment" "downloading Python, git and pip"
        & $MambaBin create -y -n $EnvName -f $CondaOnlyYml
        Assert-Native "Creating the '$EnvName' environment"
    }
    Remove-Item -LiteralPath $CondaOnlyYml -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path $EnvPy)) { Fail "Environment build did not produce an interpreter at $EnvPy" }
    Assert-NotCancelled
    Write-ProgressFile 30 32 "Preparing to install packages" ""

    # ======================================================================== #
    # 4. Phase B: the pip: subsection, with live progress
    # ======================================================================== #
    # Always explicit, on both the create and update paths: `micromamba env
    # update` skips the pip subsection when the conda solve is a no-op, which is
    # what made interrupted installs unrepairable. Version specifiers mean pip
    # no-ops on already-satisfied packages, so this is safe to run every time.
    Say "Installing scientific packages (the slow part)"
    Remove-PipDebris
    $reqs = Get-PipRequirements $EnvYml
    if ($reqs.Count -eq 0) {
        Warn "No pip: subsection found in $EnvYml; skipping the pip phase."
    } else {
        $reqsFile = Join-Path $InstallDir ".pip-requirements.txt"
        Set-Content -LiteralPath $reqsFile -Value $reqs -Encoding UTF8 -Force
        Set-Content -LiteralPath $PipLog -Value "" -Encoding UTF8 -Force

        # Parse pip's own narration into progress updates. `Collecting <pkg>`
        # fires once per wheel it resolves, so counting those against
        # $ExpectedPkgs gives a bar that tracks real work; the exact total does
        # not matter, because the GUI clamps to the ceiling and creeps between
        # updates.
        $n   = 0
        $pct = 32.0
        & $EnvPy -m pip install --no-input --progress-bar off -r $reqsFile 2>&1 | ForEach-Object {
            $line = [string]$_
            Add-Content -LiteralPath $PipLog -Value $line
            if ($line -match 'Collecting\s+(.+)$') {
                $pkg = $Matches[1] -replace '\s+\(from .*$', ''   # drop pip's "(from -r ...)" noise
                $n++
                $pct = 32 + 46 * ($n / $ExpectedPkgs)
                if ($pct -gt 78) { $pct = 78 }
                Write-ProgressFile $pct 79 "Downloading packages" $pkg
            } elseif ($line -match 'Installing collected packages') {
                Write-ProgressFile 80 92 "Installing packages" "unpacking wheels"
            } elseif ($line -match 'Successfully installed') {
                Write-ProgressFile 93 94 "Installed all packages" ""
            } elseif ($line -match '^ERROR:') {
                Write-ProgressFile $pct 79 "Resolving a problem" $line
            }
        }
        $pipRc = $LASTEXITCODE
        Remove-Item -LiteralPath $reqsFile -Force -ErrorAction SilentlyContinue
        if ($pipRc -ne 0) {
            Assert-NotCancelled       # a Cancel mid-download kills pip
            Fail "Package installation failed (exit $pipRc). Full log: $PipLog"
        }
    }
    Assert-NotCancelled

    # ======================================================================== #
    # 5. Verify before reporting success
    # ======================================================================== #
    # The Inno wizard and the launcher both treat a zero exit as "installed",
    # so failing here is what prevents a broken install being recorded as good.
    Say "Verifying the environment"
    Write-ProgressFile 94 95 "Verifying installation" "importing required packages"
    $health = Get-EnvHealth
    if ($health) {
        Write-Host $health
        Fail "Some packages are still missing (see above); HIBACHI would crash on startup. Re-run with HIBACHI_FORCE_REBUILD=1."
    }
    Write-Host "All required packages import cleanly."

    # ======================================================================== #
    # 6. Clone / force-sync the application
    # ======================================================================== #
    if (Test-Path (Join-Path $AppDir ".git")) {
        Say "Updating existing checkout to the latest $Branch"
        Write-ProgressFile 95 97 "Updating HIBACHI" "branch $Branch"
        & $MambaBin run -n $EnvName git -C $AppDir fetch origin $Branch
        Assert-Native "git fetch"
        & $MambaBin run -n $EnvName git -C $AppDir checkout $Branch
        Assert-Native "git checkout"
        & $MambaBin run -n $EnvName git -C $AppDir reset --hard "origin/$Branch"
        Assert-Native "git reset --hard origin/$Branch"
    } else {
        Say "Cloning $RepoUrl (branch: $Branch)"
        Write-ProgressFile 95 97 "Downloading HIBACHI" "branch $Branch"
        & $MambaBin run -n $EnvName git clone --branch $Branch $RepoUrl $AppDir
        Assert-Native "git clone"
    }

    # ======================================================================== #
    # 7. Desktop launcher
    # ======================================================================== #
    Say "Creating desktop launcher"
    Write-ProgressFile 98 99 "Creating launcher" ""
    & $EnvPy (Join-Path $AppDir "launcher\make_shortcuts.py")
    Assert-Native "Creating desktop launcher"

    Write-ProgressFile 100 100 "Setup complete." "" "done"
    Start-Sleep -Seconds 1        # let the window read the final state and self-close
    Stop-Gui
}
catch {
    # Anything not already routed through Fail(): report it the same way rather
    # than dying silently behind a hidden console.
    $msg = $_.Exception.Message
    Write-Host "`nSetup did not complete: $msg" -ForegroundColor Red
    Write-Host "If an earlier attempt was interrupted, force a clean rebuild with:"
    Write-Host "  `$env:HIBACHI_FORCE_REBUILD=1; <re-run the installer>"
    Write-ProgressFile 100 100 "Setup failed: $msg" "" "failed"
    if ($GuiProc) { Start-Sleep -Seconds 12 }
    Stop-Gui
    exit 1
}

Say "Done!"
Write-Host @"

HIBACHI is installed at:
  $AppDir

Launch it from the Desktop shortcut (HIBACHI), or directly with:
  "$EnvPyw" "$AppDir\launcher\run_app.py"

It will check for updates automatically each time it starts.
"@