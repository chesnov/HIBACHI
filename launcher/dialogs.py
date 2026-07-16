"""
dialogs.py -- tiny tkinter dialogs for the launcher (update prompt + rollback).

Like splash.py, this uses ONLY tkinter (bundled with the conda Python) and never
imports the heavy GUI stack (PyQt/napari). Every function degrades safely when
there is no display or tkinter is missing:

    * ask_update      -> "later"  (never update without explicit consent)
    * choose_rollback -> None     (cancel)
    * ask_yes_no      -> False
    * notify          -> prints to the console

Only one Tk root is ever alive at a time: if the splash's root still exists we
attach a Toplevel to it; otherwise we create a hidden throwaway root and destroy
it on the way out.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional


@contextlib.contextmanager
def _ui():
    """Yield (tk_module, parent_root) or (None, None) when no UI is available."""
    try:
        import tkinter as tk
    except Exception as exc:  # no tkinter at all
        print(f"[dialogs] tkinter unavailable ({exc}).")
        yield None, None
        return

    existing = getattr(tk, "_default_root", None)
    owns = existing is None
    root = None
    try:
        if owns:
            root = tk.Tk()
            root.withdraw()  # keep the throwaway root off-screen
        else:
            root = existing
        yield tk, root
    except Exception as exc:  # display error, etc.
        print(f"[dialogs] dialog error ({exc}).")
    finally:
        if owns and root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _center(win) -> None:
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"+{(sw - w) // 2}+{(sh - h) // 3}")


def ask_update(current: str, latest: str, changelog: List[str], env_changed: bool) -> str:
    """
    Ask whether to install an available update.

    Returns 'update', 'later', or 'skip'. Defaults to 'later' with no UI, so we
    never install without explicit consent.
    """
    with _ui() as (tk, root):
        if tk is None:
            return "later"

        result = {"choice": "later"}
        dlg = tk.Toplevel(root)
        dlg.title("HIBACHI update available")
        dlg.resizable(False, False)
        dlg.transient(root)

        msg = (
            "A new version of HIBACHI is available.\n\n"
            f"Installed:  {current}\n"
            f"Available:  {latest}"
        )
        if env_changed:
            msg += (
                "\n\nThis update also changes dependencies, so installing it may\n"
                "take a few minutes and will restart the app once."
            )
        tk.Label(dlg, text=msg, justify="left", font=("Helvetica", 11)).pack(
            padx=20, pady=(18, 8), anchor="w"
        )

        if changelog:
            tk.Label(dlg, text="What's changed:", font=("Helvetica", 10, "bold")).pack(
                padx=20, anchor="w"
            )
            box = tk.Text(dlg, width=58, height=min(8, len(changelog) + 1), wrap="word")
            for line in changelog:
                box.insert("end", f"\u2022 {line}\n")
            box.configure(state="disabled")
            box.pack(padx=20, pady=(2, 8))

        btns = tk.Frame(dlg)
        btns.pack(padx=20, pady=(4, 16), anchor="e")

        def choose(value: str) -> None:
            result["choice"] = value
            dlg.destroy()

        tk.Button(btns, text="Skip this version", width=15,
                  command=lambda: choose("skip")).pack(side="left", padx=4)
        tk.Button(btns, text="Not now", width=10,
                  command=lambda: choose("later")).pack(side="left", padx=4)
        tk.Button(btns, text="Update now", width=12, default="active",
                  command=lambda: choose("update")).pack(side="left", padx=4)

        dlg.protocol("WM_DELETE_WINDOW", lambda: choose("later"))
        _center(dlg)
        dlg.grab_set()
        dlg.wait_window()
        return result["choice"]


def choose_rollback(versions: List[Dict[str, str]], current_rev: str) -> Optional[str]:
    """
    Show recent versions and let the user pick one to switch to.

    `versions` is a list of dicts with keys rev/short/date/subject (newest
    first). Returns the chosen full rev, or None if cancelled / no UI.
    """
    with _ui() as (tk, root):
        if tk is None:
            return None

        result: Dict[str, Optional[str]] = {"rev": None}
        dlg = tk.Toplevel(root)
        dlg.title("Roll back HIBACHI")
        dlg.resizable(False, False)
        dlg.transient(root)

        tk.Label(dlg, text="Select a version to switch to:",
                 font=("Helvetica", 11)).pack(padx=20, pady=(16, 6), anchor="w")

        lb = tk.Listbox(dlg, width=70, height=min(12, max(3, len(versions))),
                        activestyle="dotbox", font=("Courier", 10))
        for v in versions:
            here = v["rev"].startswith(current_rev) or (current_rev and current_rev.startswith(v["rev"]))
            marker = "  <- current" if here else ""
            lb.insert("end", f'{v["date"]}  {v["short"]}  {v["subject"]}{marker}')
        lb.pack(padx=20, pady=4)
        if versions:
            lb.selection_set(0)

        btns = tk.Frame(dlg)
        btns.pack(padx=20, pady=(6, 16), anchor="e")

        def do_rollback() -> None:
            sel = lb.curselection()
            if sel:
                result["rev"] = versions[sel[0]]["rev"]
            dlg.destroy()

        tk.Button(btns, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=4)
        tk.Button(btns, text="Switch to this version", width=20, default="active",
                  command=do_rollback).pack(side="left", padx=4)

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        _center(dlg)
        dlg.grab_set()
        dlg.wait_window()
        return result["rev"]


def ask_yes_no(title: str, message: str) -> bool:
    with _ui() as (tk, root):
        if tk is None:
            return False
        from tkinter import messagebox
        return bool(messagebox.askyesno(title, message, parent=root))


def notify(title: str, message: str) -> None:
    with _ui() as (tk, root):
        if tk is None:
            print(f"[dialogs] {title}: {message}")
            return
        from tkinter import messagebox
        messagebox.showinfo(title, message, parent=root)
