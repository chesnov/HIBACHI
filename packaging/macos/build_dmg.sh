#!/usr/bin/env bash
# =============================================================================
# build_dmg.sh -- package HIBACHI.app into HIBACHI.dmg (run on macOS)
# -----------------------------------------------------------------------------
# Usage:   packaging/macos/build_dmg.sh [VERSION]
# Produces <repo-root>/HIBACHI.dmg containing a drag-to-Applications app that
# self-bootstraps on first launch. Intended to run on a macOS machine or a
# macOS GitHub Actions runner.
# =============================================================================
set -euo pipefail

VERSION="${1:-0.0.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"     # repo root

STAGE="$(mktemp -d)"
APP="${STAGE}/HIBACHI.app"

echo "==> Assembling app bundle (version ${VERSION})"
cp -R "${HERE}/HIBACHI.app" "${APP}"
mkdir -p "${APP}/Contents/Resources"

# Bundle the bootstrap installer so first launch can set everything up.
cp "${ROOT}/install/install.sh" "${APP}/Contents/Resources/install.sh"
chmod +x "${APP}/Contents/MacOS/HIBACHI" "${APP}/Contents/Resources/install.sh"

# Stamp the version into Info.plist.
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${VERSION}" "${APP}/Contents/Info.plist" || true
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion ${VERSION}" "${APP}/Contents/Info.plist" || true

# Optional icon.
if [ -f "${HERE}/hibachi.icns" ]; then
    cp "${HERE}/hibachi.icns" "${APP}/Contents/Resources/"
fi

# A symlink so users can drag the app onto Applications inside the DMG window.
ln -s /Applications "${STAGE}/Applications"

OUT="${ROOT}/HIBACHI.dmg"
rm -f "${OUT}"
echo "==> Creating ${OUT}"
hdiutil create -volname "HIBACHI" -srcfolder "${STAGE}" -ov -format UDZO "${OUT}"

echo "==> Done: ${OUT}"

# --- Optional: code sign + notarize (requires an Apple Developer ID) --------
# Without this, users see a Gatekeeper warning and must right-click -> Open once.
#   codesign --deep --force --options runtime \
#     --sign "Developer ID Application: YOUR NAME (TEAMID)" "${APP}"
#   hdiutil create ...   # (re-create dmg after signing the app)
#   xcrun notarytool submit "${OUT}" --keychain-profile "AC_PASSWORD" --wait
#   xcrun stapler staple "${OUT}"
