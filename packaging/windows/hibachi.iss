; ============================================================================
; HIBACHI Windows installer (Inno Setup script)
; ----------------------------------------------------------------------------
; Compiles to  HIBACHI-Setup.exe  with the Inno Setup Compiler (ISCC.exe).
; This is a thin wizard: it lays down the bundled bootstrap files and runs
; install.ps1, which downloads micromamba, builds the conda env, clones the
; repo, and creates the launcher shortcut. The app self-updates thereafter.
;
; Build locally on Windows:
;   iscc packaging\windows\hibachi.iss
; Or let GitHub Actions build it (see .github/workflows/build-installers.yml).
;
; NOTE ON SIGNING: an unsigned .exe shows a Microsoft SmartScreen warning
; ("Windows protected your PC" -> More info -> Run anyway). To remove it, sign
; the compiled exe with an OV/EV code-signing certificate (see INSTALL.md).
; ============================================================================

#define AppName "HIBACHI"
#define AppPublisher "HIBACHI"
; AppVersion can be overridden on the ISCC command line:  /DAppVersion=1.2.3
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{7B2C6E9A-4E2E-4B1E-9E5F-HIBACHI000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; Per-user install: no admin rights required (matches the "biologist" audience).
PrivilegesRequired=lowest
DefaultDirName={localappdata}\{#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=HIBACHI-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; The installer itself is small; the heavy download happens during [Run].
SetupIconFile=hibachi.ico

[Files]
; Bundle the bootstrap logic and dependency spec so install.ps1 can run offline
; of the repo (it still downloads micromamba + packages from the internet).
Source: "..\..\install\install.ps1";     DestDir: "{app}\bootstrap"; Flags: ignoreversion
Source: "..\..\install\environment.yml";  DestDir: "{app}\bootstrap"; Flags: ignoreversion

[Run]
; RUN HIDDEN. install.ps1 now raises its own WinForms progress window (a bar,
; the current package name and an elapsed timer), which is the progress UI on
; every OS -- matching the macOS .app and the Linux one-liner.
;
; This used to run VISIBLE so that users could see micromamba's live download
; output and not think the installer had hung. That worked, but a raw console
; full of solver output alarms a non-technical audience, and it duplicated the
; wizard's own status page. The dedicated window is friendlier and says the one
; thing that actually matters: leave this open, it takes a few minutes.
;
; If you need to debug a failing bootstrap, run it by hand to see the console:
;   powershell -ExecutionPolicy Bypass -File "%LOCALAPPDATA%\HIBACHI\bootstrap\install.ps1"
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\bootstrap\install.ps1"" -InstallDir ""{app}"""; \
  StatusMsg: "Setting up HIBACHI. A progress window shows what is happening; this takes several minutes..."; \
  Flags: runhidden waituntilterminated

[UninstallRun]
; Best-effort cleanup of the installed environment + checkout on uninstall.
; Remove the chosen install dir ({app}), not a hardcoded default path.
Filename: "cmd.exe"; Parameters: "/c rmdir /s /q ""{app}"""; Flags: runhidden; RunOnceId: "RemoveHibachiHome"

[UninstallDelete]
; make_shortcuts.py writes these at RUNTIME, so they are not in [Icons] and the
; uninstaller does not otherwise know about them -- they used to survive as dead
; icons. It writes both a .lnk and a .bat fallback, to the Desktop and the Start
; Menu, so all four are removed here.
;
; On managed/corporate machines the Desktop is often redirected into OneDrive,
; which {userdesktop} resolves correctly (make_shortcuts.py reads the same shell
; folder from the registry), so both agree on the location.
Type: files; Name: "{userdesktop}\HIBACHI.lnk"
Type: files; Name: "{userdesktop}\HIBACHI.bat"
Type: files; Name: "{userprograms}\HIBACHI.lnk"
Type: files; Name: "{userprograms}\HIBACHI.bat"
; Logs and launcher state (~/.hibachi). Left behind, this carried a stale
; "skipped_rev" across a reinstall, so a freshly installed app could silently
; decline to offer an update it had been told to skip months earlier.
Type: filesandordirs; Name: "{%USERPROFILE}\.hibachi"

[Messages]
WelcomeLabel2=This will install [name] for the current user.%n%nThe first setup downloads the scientific packages (a few hundred MB) and may take several minutes. A progress window will show you what is happening. An internet connection is required.
