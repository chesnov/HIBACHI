"""
splash.py -- a tiny, dependency-light status window shown during startup.

Biologists shouldn't stare at a blank screen (or a scary terminal) while the
app checks for updates. This shows a small "HIBACHI is starting..." window with
a status line. It uses only tkinter (bundled with the conda Python), and if
tkinter is somehow unavailable it degrades silently to console prints -- so it
can never block startup.

Kept deliberately independent of PyQt/napari: the heavy GUI stack must NOT be
imported until AFTER a potential dependency update has run.
"""

from __future__ import annotations

from typing import Optional


class Splash:
    def __init__(self, title: str = "HIBACHI", subtitle: str = "Starting up..."):
        self._tk = None
        self._root = None
        self._status_var = None
        try:
            import tkinter as tk  # noqa: WPS433 (intentional local import)

            self._tk = tk
            root = tk.Tk()
            root.title(title)
            root.resizable(False, False)
            # Center a small window.
            w, h = 420, 150
            root.update_idletasks()
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")

            tk.Label(root, text=title, font=("Helvetica", 20, "bold")).pack(pady=(22, 4))
            self._status_var = tk.StringVar(value=subtitle)
            tk.Label(root, textvariable=self._status_var, font=("Helvetica", 11)).pack(pady=4)
            tk.Label(
                root,
                text="This window closes automatically.",
                font=("Helvetica", 8),
                fg="#888888",
            ).pack(side="bottom", pady=8)

            self._root = root
            self._pump()
        except Exception as exc:  # tkinter missing / no display / etc.
            print(f"[splash] GUI splash unavailable ({exc}); using console output.")
            self._tk = None
            self._root = None

    def _pump(self) -> None:
        if self._root is not None:
            try:
                self._root.update_idletasks()
                self._root.update()
            except Exception:
                pass

    def set_status(self, text: str) -> None:
        print(f"[startup] {text}")
        if self._root is not None and self._status_var is not None:
            try:
                self._status_var.set(text)
                self._pump()
            except Exception:
                pass

    def close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            finally:
                self._root = None


def get_splash(enabled: bool = True) -> Optional["Splash"]:
    """Return a Splash, or None if disabled (e.g. headless / --no-splash)."""
    if not enabled:
        return None
    return Splash()
