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
; To brand the installer, add packaging\windows\hibachi.ico and uncomment:
; SetupIconFile=hibachi.ico

[Files]
; Bundle the bootstrap logic and dependency spec so install.ps1 can run offline
; of the repo (it still downloads micromamba + packages from the internet).
Source: "..\..\install\install.ps1";     DestDir: "{app}\bootstrap"; Flags: ignoreversion
Source: "..\..\install\environment.yml";  DestDir: "{app}\bootstrap"; Flags: ignoreversion

[Run]
; Run the bootstrap with a visible progress message. install.ps1 reads config
; from environment variables so we don't have to edit it here.
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\bootstrap\install.ps1"""; \
  StatusMsg: "Downloading and setting up HIBACHI (this can take several minutes)..."; \
  Flags: runhidden waituntilterminated

[UninstallRun]
; Best-effort cleanup of the installed environment + checkout on uninstall.
Filename: "cmd.exe"; Parameters: "/c rmdir /s /q ""{%USERPROFILE}\HIBACHI"""; Flags: runhidden; RunOnceId: "RemoveHibachiHome"

[Messages]
WelcomeLabel2=This will install [name] for the current user.%n%nThe first setup downloads the scientific packages (a few hundred MB) and may take several minutes. An internet connection is required.
