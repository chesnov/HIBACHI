# HIBACHI — Installation, Troubleshooting & Releasing

For what HIBACHI is and how to use it, see the [README](README.md). This file
covers getting it installed, fixing it when it will not start, and — in the
second half — publishing it.

---

# Part 1 — For users

## Installing

You do not need Python, conda or git. The installer builds a private,
self-contained environment.

*   **Windows** — download `HIBACHI-Setup.exe` from the
    [latest release](https://github.com/chesnov/HIBACHI/releases/latest) and
    double-click it. If SmartScreen appears: *More info → Run anyway*.
*   **macOS** — download `HIBACHI.dmg` from the
    [latest release](https://github.com/chesnov/HIBACHI/releases/latest) and drag
    **HIBACHI** to Applications. macOS blocks it on first open because the app is
    unsigned. To allow it, once:

    1.  Right-click **HIBACHI** → *Open* → *Open*. On newer macOS this route has
        been removed and the app is blocked anyway; if so, continue below.
    2.  Open **System Settings → Privacy & Security** and scroll to the
        **Security** section. A line reads *"HIBACHI was blocked from use because
        it is not from an identified developer"* (or *"...blocked to protect your
        Mac"*). Click **Open Anyway**, then confirm with Touch ID or your
        password.
    3.  Open HIBACHI again. It launches normally from now on.

    On older macOS the same control is under
    **System Preferences → Security & Privacy → General**.
*   **Linux** —

    ```bash
    curl -fsSL https://raw.githubusercontent.com/chesnov/HIBACHI/main/install/install.sh | bash
    ```

The first run downloads the scientific packages, which takes a few minutes and
needs a connection. After that, launch from the shortcut.

Both installers are unsigned, which is why Windows and macOS block them once.
See [Code signing](#code-signing--notarization) for what would remove that.

### What the installer does

Worth knowing if it fails partway:

1.  Fetches a `micromamba` binary — one self-contained file, no system Python
    touched.
2.  Builds the environment in **two phases**: the conda-level packages first,
    then the pip packages explicitly. A single-phase build that failed midway
    left an environment that retrying could not repair, because the retry saw an
    environment that already existed.
3.  Runs an **import health check** over the packages the app needs, so a
    half-built environment is reported at install time rather than at first use.
4.  Clones the repository and force-syncs it to the tracked branch.
5.  Creates the shortcut.

There is a progress window with a working **Cancel**. Cancelling tears down a
partially-built environment rather than leaving it behind.

**If an install fails or is interrupted, run it again.** The installer checks an
existing environment on entry and rebuilds it from scratch when it is incomplete,
so a half-finished attempt repairs itself with no flags and no terminal. On macOS
that means opening HIBACHI again; on Windows, running `HIBACHI-Setup.exe` again;
on Linux, re-running the one-liner.

The success marker is written only when the install completes and every required
package imports, so an interrupted attempt is never mistaken for a good one.

## Updating

On launch, HIBACHI checks GitHub for a newer version. If there is one it shows
what changed and asks: **Update**, **Later**, or **Skip this version**. A skipped
version is not offered again.

Local edits are stashed and the checkout fast-forwarded. A checkout with commits
of its own is left alone rather than being moved.

Offline, it opens the version you have.

To change this: `HIBACHI_AUTO_UPDATE=1` installs updates without asking, and
`HIBACHI_NO_UPDATE=1` skips the check entirely. Both are read by the app, so see
[Setting one](#setting-one) for how to set them on your system.

## Version and rollback

The project window's status bar shows the running version. Clicking it offers a
version check and a list of earlier versions to switch to. Rolling back records
the current remote tip as skipped, so the next launch does not immediately offer
the version you just left.

Starting with `HIBACHI_ROLLBACK=1` — or passing `--rollback` — opens the chooser
directly instead of launching. That is the route on macOS, where the launcher
`exec`s into the app and so cannot offer rollback after a crash. See
[Setting one](#setting-one) for how to set it.

## Uninstalling

The version dialog has an **Uninstall** option, which removes the installation
and the shortcut. Deletion runs from a detached script, because the targets
include the interpreter running it.

Removing it by hand means deleting two things:

*   `~/HIBACHI/` — the environment and the checkout.
*   `~/.hibachi/` — recent projects, the config library and logs.

Deleting only the first leaves your config library intact for a reinstall, which
may be what you want. **Your images and results are never inside either**, and
are not touched.

## Troubleshooting

### It will not start, or dies immediately

Send `crash-report.txt` from `~/.hibachi/logs/`. It bundles the app log, the
native crash traceback, the launcher log and the app's console output. See
[Diagnostics](wiki/diagnostics.md).

### A crash mentioning the graphics driver

Messages such as `context is lost`, `guilty of a hard recovery`, `Xid` or
`DEVICE_LOST` mean the graphics driver reset the GPU. Start HIBACHI with
`HIBACHI_SOFTWARE_OPENGL=1` to render without it — see
[Setting one](#setting-one) for how to set that on your system.

This is slower to draw but bypasses the driver. It is the supported answer for
virtual machines, remote desktops and machines without a usable GPU driver. Data
and results on disk are unaffected by this class of crash — it happens in the
display layer.

### The app bounces in the Dock and disappears (macOS)

Two causes, worth ruling out in this order:

1.  **Gatekeeper.** An unquarantined-but-unapproved app is killed before its code
    runs, so nothing can report anything. Follow the
    **Privacy & Security → Open Anyway** steps above.
2.  **A setup running with no window.** Setup shows a progress window once it has
    a usable Python; before that it posts macOS notifications instead. If you see
    a *"Setting up HIBACHI"* notification, it is working — the download takes
    several minutes. Progress is written to `~/HIBACHI/setup.log`.

If neither applies, `~/HIBACHI/setup.log` and `~/HIBACHI/launch.log` hold what
happened.

### "MSVCP140.dll was not found" on Windows

The app registers the environment's DLL directories itself, so this should not
occur. If it does, installing Microsoft's VC++ Redistributable resolves it.

### The environment looks broken

Run the installer again — opening HIBACHI on macOS, `HIBACHI-Setup.exe` on
Windows, the one-liner on Linux. An incomplete environment is detected and
rebuilt automatically.

`HIBACHI_FORCE_REBUILD=1` forces that rebuild even when the environment passes
its checks. Only the installer reads it, never the app, so see
[Install-time variables](#install-time-variables) for how to supply it.

## Environment variables

These change how HIBACHI installs or starts. You do not need any of them for
normal use — they exist for troubleshooting and for unusual setups.

### Setting one

HIBACHI normally starts from a shortcut, which does not read your shell profile.
There are two ways to set a variable so it takes effect.

**For one run**, launch from a terminal with the variable in front of the command.
The shortcut runs the launcher inside HIBACHI's own environment, so the command is:

*   **Linux / macOS**

    ```bash
    HIBACHI_SOFTWARE_OPENGL=1 ~/HIBACHI/micromamba/envs/hibachi/bin/python \
        ~/HIBACHI/app/launcher/run_app.py
    ```

*   **Windows** (PowerShell)

    ```powershell
    $env:HIBACHI_SOFTWARE_OPENGL = "1"
    & "$env:USERPROFILE\HIBACHI\micromamba\envs\hibachi\pythonw.exe" `
        "$env:USERPROFILE\HIBACHI\app\launcher\run_app.py"
    ```

**Permanently**, so the desktop shortcut picks it up too:

*   **Windows** — press Start, type *environment variables*, choose **Edit
    environment variables for your account**, click **New**, enter the name and
    `1` as the value, then **OK**. Restart HIBACHI.
*   **macOS** — in Terminal:

    ```bash
    launchctl setenv HIBACHI_SOFTWARE_OPENGL 1
    ```

    This applies to apps launched afterwards, and lasts until you log out. To
    remove it: `launchctl unsetenv HIBACHI_SOFTWARE_OPENGL`.
*   **Linux** — edit `~/.local/share/applications/hibachi.desktop` and put the
    variable on the `Exec=` line:

    ```
    Exec=env HIBACHI_SOFTWARE_OPENGL=1 /home/you/HIBACHI/micromamba/envs/hibachi/bin/python /home/you/HIBACHI/app/launcher/run_app.py
    ```

### Install-time variables

`HIBACHI_HOME`, `HIBACHI_BRANCH`, `HIBACHI_FORCE_REBUILD`,
`HIBACHI_EXPECTED_PKGS`, `HIBACHI_NO_INSTALLER_GUI` and `HIBACHI_SKIP_SHORTCUT`
are read by the installer, not by the app. How you supply one depends on how you
install:

*   **Linux, or any platform running the script directly** — put it in front of
    the command:

    ```bash
    HIBACHI_FORCE_REBUILD=1 bash install.sh
    ```

*   **Windows `HIBACHI-Setup.exe`, or the macOS app's first launch** — there is no
    command to prefix, so set the variable **permanently first** (using the
    Windows or macOS instructions above), then run the installer. It inherits
    your account's environment.

    On macOS, `launchctl setenv` only reaches apps launched afterwards, so set it
    before opening HIBACHI.

Alternatively, run the script by hand. It is already on disk after an install:

*   macOS / Linux: `HIBACHI_FORCE_REBUILD=1 bash ~/HIBACHI/app/install/install.sh`
*   Windows (PowerShell):

    ```powershell
    $env:HIBACHI_FORCE_REBUILD = "1"
    & "$env:USERPROFILE\HIBACHI\app\install\install.ps1"
    ```

The app itself cannot rebuild the environment — only the installer can — so a
rebuild always means running one of the above.

### Reference

| Variable | Effect |
| :--- | :--- |
| `HIBACHI_HOME` | Install location (default `~/HIBACHI`) |
| `HIBACHI_STATE_DIR` | State: recent projects, config library, logs (default `~/.hibachi`) |
| `HIBACHI_LOG_DIR` | Logs only |
| `HIBACHI_BRANCH` | Branch to install and track (default `main`) |
| `HIBACHI_OWNER`, `HIBACHI_REPO` | Which GitHub repository to install from |
| `HIBACHI_NO_UPDATE=1` | Skip the update check |
| `HIBACHI_AUTO_UPDATE=1` | Install updates without asking |
| `HIBACHI_ROLLBACK=1` | Open the version chooser instead of launching |
| `HIBACHI_NO_SPLASH=1` | No splash window |
| `HIBACHI_SOFTWARE_OPENGL=1` | Software OpenGL (see above) |
| `HIBACHI_FORCE_REBUILD=1` | Rebuild the environment from scratch |
| `HIBACHI_SKIP_ENV_CHECK=1` | Skip the environment health check |
| `HIBACHI_EXPECTED_PKGS` | Package count the installer's progress bar scales to |
| `HIBACHI_NO_INSTALLER_GUI=1` | Install without the progress window |
| `HIBACHI_SKIP_SHORTCUT=1` | Do not create a shortcut |
| `HIBACHI_GIT` | Path to a specific `git` binary |

`HIBACHI_ENV_UPDATED` is set internally as a re-exec guard and is not for you to
set.

## Running from source

For development, with [conda / Miniforge](https://github.com/conda-forge/miniforge)
and Git:

```bash
git clone https://github.com/chesnov/HIBACHI.git
cd HIBACHI
conda env create -f install/environment.yml
conda activate hibachi
python segment.py
```

That launches the app directly, skipping the splash, the update check, crash
reporting and rollback. See [`segment.py`](wiki/segment.md).

---

# Part 2 — For the maintainer

## What gets installed where

```
~/HIBACHI/
├── micromamba/            the package manager (one self-contained binary)
├── environment.yml        copy fetched during install
└── app/                   the git checkout — auto-updated
    ├── segment.py         entry point
    ├── utils/             the package (needs utils/__init__.py)
    │   ├── high_level_gui/
    │   ├── module_2d/  module_3d/  spatial_null/
    ├── install/           environment.yml, install.sh, install.ps1
    ├── launcher/          run_app.py, updater.py, splash.py, dialogs.py,
    │                      make_shortcuts.py, uninstall.py
    ├── packaging/         windows/ (Inno .iss) + macos/ (.app, build_dmg.sh)
    ├── wiki/              the documentation
    └── .github/workflows/ build-installers.yml

~/.hibachi/                state, never touched by an update
├── configs/{2d,3d}/       the config library
├── logs/                  app, launcher, child, faulthandler, crash report
└── recent_projects.json
```

The conda env is named `hibachi` under `~/HIBACHI/micromamba`.

**No user data lives inside `app/`.** Preset templates are copied into project
folders, per-project configs and outputs live in those folders, scratch uses the
project temp dir, and cross-project state is under `~/.hibachi`. That invariant
is what makes force-syncing `app/` on update safe.

## Publishing

1.  Set `GH_OWNER`/`GH_REPO` in `install/install.sh` and `$GhOwner`/`$GhRepo` in
    `install/install.ps1`.
2.  Ensure `utils/__init__.py` exists.
3.  Optionally add an icon at `launcher/assets/hibachi.png`.
4.  Push, and test the one-liner on a clean machine.

Release: `git tag v1.2.3 && git push --tags`. The
`.github/workflows/build-installers.yml` workflow builds both installers and
attaches them to the GitHub Release, which is where the README sends users.

Both installers are thin wrappers around the same bootstrap:

*   `packaging/windows/hibachi.iss` — [Inno Setup](https://jrsoftware.org/isinfo.php),
    compiles to `HIBACHI-Setup.exe`, runs `install.ps1` behind a progress page.
*   `packaging/macos/HIBACHI.app` + `build_dmg.sh` — a self-bootstrapping app in
    a `.dmg` via `hdiutil`. First launch runs `install.sh`; later launches run the
    launcher.

To build locally: `iscc packaging\windows\hibachi.iss`, or
`packaging/macos/build_dmg.sh 1.2.3`.

## Tracking a `stable` branch

Set `HIBACHI_BRANCH=stable` inside `install/install.sh` and `install/install.ps1`
(or prefix it on the published one-liner), and merge into `stable` only when you
want users to receive a change. Users then track `stable` while you
work on `main`.

## Reproducible builds

`environment.yml` uses loose version floors so conda-forge can solve on every
platform. For a locked environment per release, generate a lock file with
[`conda-lock`](https://github.com/conda/conda-lock) and have the installer
consume it instead.

## Testing on macOS / Windows

Confirm on each: the environment solves; the shortcut appears and launches with
no visible terminal; the splash shows; an update is detected after a push; a
changed `environment.yml` triggers a dependency update. On Windows also confirm
the `.lnk` is created — it falls back to a `.bat` if PowerShell COM is
unavailable.

## Code signing & notarization

The CI artifacts are unsigned, so users see a one-time warning: SmartScreen on
Windows, Gatekeeper on macOS. To remove it:

*   **Windows** — an OV or EV code-signing certificate, then a
    `signtool sign /fd sha256 /tr <timestamp-url> ...` step (or Inno's
    `SignTool`) after compilation.
*   **macOS** — the Apple Developer Program, then uncomment the `codesign`,
    `notarytool` and `stapler` steps in `build_dmg.sh` and supply your Developer
    ID.

Signing secrets belong in GitHub Actions secrets, not the repo.

Until then the workarounds in Part 1 are the intended route: *Run anyway* on
Windows, and **Privacy & Security → Open Anyway** on macOS.

## Outstanding

*   A first-run resource check (RAM and disk, via `psutil`, already a
    dependency).
*   A signed and notarized macOS `.app`.
