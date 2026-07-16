# HIBACHI — Installation & Updating

This app installs with **one command** and **updates itself** every time it
opens. No prior Python, no admin rights, works on Windows, macOS, and Linux.

---

## For users (biologists)

You install once, and the app updates itself after that. Choose your system:

* **Windows** — download `HIBACHI-Setup.exe` from the
  [Releases page](https://github.com/chesnov/HIBACHI/releases/latest) and
  double-click it. If SmartScreen appears: *More info → Run anyway*.
* **macOS** — download `HIBACHI.dmg` from the
  [Releases page](https://github.com/chesnov/HIBACHI/releases/latest), drag
  **HIBACHI** to Applications, then right-click it → *Open* the first time.
* **Linux** — open a terminal and paste:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/chesnov/HIBACHI/main/install/install.sh | bash
  ```

The first launch downloads the scientific packages (a few minutes). After that,
double-click the **HIBACHI** icon; it shows a small "Starting up…" window while
it checks for updates, then opens. Offline is fine — it opens the version you
have.

To remove HIBACHI, delete the `HIBACHI` folder in your home directory (and the
shortcut). Your image data and results live elsewhere and are never touched.

---

## For the maintainer (you)

### What gets installed where

```
~/HIBACHI/
├── micromamba/            # the package manager (one binary, self-contained)
├── environment.yml        # copy fetched during install
└── app/                   # the git checkout (this repo) — auto-updated
    ├── segment.py         # real entry point (unchanged)
    ├── utils/             # your package (needs utils/__init__.py)
    ├── install/           # environment.yml, install.sh, install.ps1
    ├── launcher/          # run_app.py, updater.py, splash.py, make_shortcuts.py
    ├── packaging/         # windows/ (Inno .iss) + macos/ (.app, build_dmg.sh)
    └── .github/workflows/ # build-installers.yml (builds the .exe / .dmg)
```

The conda env is named `hibachi` under `~/HIBACHI/micromamba`. **User data lives
entirely outside `app/`**: preset templates are copied into the user's project
folders, per-project configs/outputs live in those folders, and scratch uses the
system temp dir. That invariant is what makes auto-update safe — see below.

### One-time setup before you publish

1. Put these files in the repo at the paths shown above (`install/` and
   `launcher/`), and ensure `utils/__init__.py` exists (an empty file is fine).
2. In `install/install.sh` and `install/install.ps1`, set `GH_OWNER`/`GH_REPO`
   (or `$GhOwner`/`$GhRepo`) to your GitHub owner/repo. Replace `chesnov/HIBACHI` in
   the two one-line commands above.
3. (Optional) add an icon at `launcher/assets/hibachi.png`.
4. Push. Test the one-liner on a clean machine.

### How updating works (and why it's safe)

On every launch, `launcher/run_app.py`:

1. `git fetch` + compares your checkout to `origin/<branch>`.
2. If behind: any local edits to tracked files are first saved to a timestamped
   `git stash` (recoverable via `git stash list`), then the checkout is
   fast-forwarded (or hard-reset) to the remote tip.
3. If — and only if — `install/environment.yml` changed, it runs
   `micromamba env update` and relaunches once so new packages take effect.
4. Launches `segment.py`.

Any failure (offline, git error, …) is caught and the app launches the version
already on disk. Adding or bumping a dependency is therefore as simple as
editing `environment.yml` and pushing — users pick it up on next launch.

### Recommended: track a `stable` branch, not `main`

Auto-pulling `main` means every commit reaches users immediately. Safer for a
non-technical audience: develop on `main`, and only fast-forward a `stable`
branch (or move a tag) when you've verified a build. Point the installers at it:

```bash
HIBACHI_BRANCH=stable curl -fsSL .../install.sh | bash
```

The launcher tracks whatever branch the checkout is on (override with the
`HIBACHI_BRANCH` env var).

### Reproducible builds (optional, recommended for releases)

`environment.yml` uses loose version floors so conda-forge can solve on all
platforms. For a locked, reproducible environment per release, generate a lock
file with [`conda-lock`](https://github.com/conda/conda-lock) and have the
installer consume it instead of `environment.yml`.

### Useful knobs (environment variables)

| Variable            | Effect                                             |
|---------------------|----------------------------------------------------|
| `HIBACHI_BRANCH`    | Branch to install / track (default `main`)         |
| `HIBACHI_HOME`      | Install location (default `~/HIBACHI`)             |
| `HIBACHI_NO_UPDATE` | `1` skips the self-update (offline / development)  |
| `HIBACHI_NO_SPLASH` | `1` disables the splash window                     |

### Testing on macOS / Windows

Give your testers the matching one-line command. Things to confirm on each OS:
the env solves, the shortcut appears and launches without a visible terminal,
the splash shows, an update is detected after you push a commit, and a forced
`environment.yml` change triggers a dependency update. On Windows also confirm
the `.lnk` is created (falls back to a `.bat` if PowerShell COM is unavailable).

### Building the native installers (.exe / .dmg)

Biologists on Windows/macOS get a double-click installer instead of a terminal
command. Both are thin wrappers around the same bootstrap (`install.ps1` /
`install.sh`) and are **built by CI**, so you don't need a Windows or Mac
machine yourself:

* `packaging/windows/hibachi.iss` — an [Inno Setup](https://jrsoftware.org/isinfo.php)
  script that compiles to `HIBACHI-Setup.exe` and runs `install.ps1` with a
  progress page.
* `packaging/macos/HIBACHI.app` + `packaging/macos/build_dmg.sh` — a
  self-bootstrapping app packaged into `HIBACHI.dmg` (`hdiutil`). First launch
  runs `install.sh`; every launch after runs the self-updating launcher.
* `.github/workflows/build-installers.yml` — on every pushed tag (`vX.Y.Z`),
  builds both artifacts and **attaches them to the GitHub Release**. That
  Releases page is where the README links users to download.

Release flow: `git tag v1.2.3 && git push --tags` → wait for the workflow →
the `.exe` and `.dmg` appear on the release. To build locally instead:
`iscc packaging\windows\hibachi.iss` (Windows) or
`packaging/macos/build_dmg.sh 1.2.3` (macOS).

### Code signing & notarization (removes the scary warnings)

The CI-built artifacts are **unsigned**, so users see a one-time warning:
Windows SmartScreen ("More info → Run anyway") and macOS Gatekeeper
(right-click → Open the first time). This is cosmetic but off-putting for a
non-technical audience. To remove it:

* **Windows:** obtain an OV or EV code-signing certificate and add a
  `signtool sign /fd sha256 /tr <timestamp-url> ...` step (or Inno's
  `SignTool`) after compilation.
* **macOS:** join the Apple Developer Program ($99/yr), then in `build_dmg.sh`
  uncomment the `codesign` + `notarytool` + `stapler` steps and supply your
  Developer ID. Signing secrets live in GitHub Actions secrets, not in the repo.

Until then, the "Run anyway" / right-click-Open instructions in the README are
the intended workaround.

### Suggested next user-friendly touches (not yet built)

- A **"Check for updates"** menu item + visible version string in the app.
- A **first-run resource check** (RAM/disk via `psutil`, already a dependency).
- **File logging** to `~/HIBACHI/logs/` so users can send you a log on failure.
- An **uninstaller** script and a signed/notarized macOS `.app` (removes the
  Gatekeeper warning on first open).
