"""
dialogs.py -- small, themed tkinter dialogs for the launcher.

Uses ONLY tkinter/ttk (bundled with the conda Python); never imports the heavy
GUI stack (PyQt/napari). Every function degrades safely with no display:

    * ask_update      -> "later"  (never update without explicit consent)
    * choose_rollback -> None     (cancel)
    * ask_yes_no      -> False
    * notify          -> prints to the console

Each dialog builds its UI on its own visible root and runs a short mainloop,
brought to the front. We deliberately avoid a withdrawn parent + transient() +
grab_set() modal, which on some Linux window managers grabs input without ever
mapping a window (a silent hang).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

# --- palette -------------------------------------------------------------- #
_BG = "#f4f5f7"          # window background
_CARD = "#ffffff"        # content surface
_HEADER = "#22252a"      # header strip
_HEADER_FG = "#ffffff"
_TEXT = "#1f2328"        # primary text
_MUTED = "#6b7280"       # secondary text
_ACCENT = "#c75b39"      # primary action (warm "hibachi" ember)
_ACCENT_ACTIVE = "#a94a2d"
_BORDER = "#e2e4e8"
_SEL = "#f3d9cf"         # list selection tint

_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hibachi.png")


def _pick_font(tk_root):
    """Choose a pleasant UI font that exists on this system."""
    try:
        import tkinter.font as tkfont
        available = set(tkfont.families(tk_root))
    except Exception:
        available = set()
    for fam in ("SF Pro Text", "Segoe UI", "Helvetica Neue", "Cantarell",
                "Noto Sans", "DejaVu Sans", "Helvetica", "Arial"):
        if fam in available:
            return fam
    return "TkDefaultFont"


def _new_root():
    """Create a themed, front-most Tk root. Returns (tk, root, ui) or (None, None, None)."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        print(f"[dialogs] no GUI available ({exc}); skipping dialog.")
        return None, None, None
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"[dialogs] could not open a window ({exc}); skipping dialog.")
        return None, None, None

    root.title("HIBACHI")
    root.configure(bg=_BG)
    root.resizable(False, False)

    fam = _pick_font(root)
    ui = {
        "font": (fam, 11),
        "font_small": (fam, 9),
        "font_head": (fam, 15, "bold"),
        "font_bold": (fam, 10, "bold"),
        "mono": ("DejaVu Sans Mono", 10),
    }

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TFrame", background=_CARD)
    style.configure("Bg.TFrame", background=_BG)
    style.configure("TLabel", background=_CARD, foreground=_TEXT, font=ui["font"])
    style.configure("Muted.TLabel", background=_CARD, foreground=_MUTED, font=ui["font_small"])
    style.configure("Head.TLabel", background=_HEADER, foreground=_HEADER_FG, font=ui["font_head"])
    style.configure("HeadSub.TLabel", background=_HEADER, foreground="#c9ccd1", font=ui["font_small"])
    # Secondary (default) button
    style.configure("TButton", font=ui["font_bold"], padding=(14, 7),
                    background="#e9eaed", foreground=_TEXT, borderwidth=0)
    style.map("TButton", background=[("active", "#dcdee2")])
    # Primary (accent) button
    style.configure("Accent.TButton", font=ui["font_bold"], padding=(14, 7),
                    background=_ACCENT, foreground="#ffffff", borderwidth=0)
    style.map("Accent.TButton",
              background=[("active", _ACCENT_ACTIVE), ("pressed", _ACCENT_ACTIVE)],
              foreground=[("disabled", "#f0d8cf")])

    try:
        root.lift()
        root.attributes("-topmost", True)
        root.after(300, lambda: root.attributes("-topmost", False))
        root.focus_force()
    except Exception:
        pass
    try:
        if os.path.isfile(_ICON):
            root._icon_img = tk.PhotoImage(file=_ICON)  # keep a ref
            root.iconphoto(True, root._icon_img)
    except Exception:
        pass
    return tk, root, ui


def _header(tk, root, ui, title: str, subtitle: str = "HIBACHI"):
    from tkinter import ttk
    bar = tk.Frame(root, bg=_HEADER)
    bar.pack(fill="x", side="top")
    inner = tk.Frame(bar, bg=_HEADER)
    inner.pack(fill="x", padx=22, pady=(16, 14))
    ttk.Label(inner, text=title, style="Head.TLabel").pack(anchor="w")
    ttk.Label(inner, text=subtitle, style="HeadSub.TLabel").pack(anchor="w", pady=(2, 0))


def _body(tk, root):
    body = tk.Frame(root, bg=_CARD)
    body.pack(fill="both", expand=True)
    return body


def _center(win) -> None:
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")


def _finish(root) -> None:
    try:
        root.quit()
    except Exception:
        pass


def _run(root):
    _center(root)
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass


def ask_update(current: str, latest: str, changelog: List[str], env_changed: bool) -> str:
    """Return 'update', 'later', or 'skip'. Defaults to 'later' with no UI."""
    tk, root, ui = _new_root()
    if tk is None:
        return "later"
    from tkinter import ttk

    result = {"choice": "later"}
    _header(tk, root, ui, "Update available")
    body = _body(tk, root)

    grid = ttk.Frame(body)
    grid.pack(fill="x", padx=22, pady=(18, 6), anchor="w")
    ttk.Label(grid, text="Installed", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 16))
    ttk.Label(grid, text=current, font=ui["mono"]).grid(row=0, column=1, sticky="w")
    ttk.Label(grid, text="Available", style="Muted.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 16), pady=(3, 0))
    ttk.Label(grid, text=latest, font=ui["mono"]).grid(row=1, column=1, sticky="w", pady=(3, 0))

    if env_changed:
        ttk.Label(body,
                  text="This update also changes dependencies, so it may take a few\n"
                       "minutes to install and will restart the app once.",
                  style="Muted.TLabel").pack(fill="x", padx=22, pady=(8, 0), anchor="w")

    if changelog:
        ttk.Label(body, text="What's changed", style="TLabel", font=ui["font_bold"]).pack(
            anchor="w", padx=22, pady=(14, 4)
        )
        wrap = tk.Frame(body, bg=_BORDER, highlightthickness=0)
        wrap.pack(fill="x", padx=22)
        box = tk.Text(wrap, width=54, height=min(8, len(changelog) + 1), wrap="word",
                      relief="flat", bg=_CARD, fg=_TEXT, font=ui["font"], padx=12, pady=8,
                      highlightthickness=1, highlightbackground=_BORDER)
        for line in changelog:
            box.insert("end", f"\u2022  {line}\n")
        box.configure(state="disabled")
        box.pack(fill="x")

    btns = ttk.Frame(body)
    btns.pack(fill="x", padx=22, pady=(18, 18))

    def choose(value: str) -> None:
        result["choice"] = value
        _finish(root)

    ttk.Button(btns, text="Skip this version", command=lambda: choose("skip")).pack(side="left")
    ttk.Button(btns, text="Update now", style="Accent.TButton",
               command=lambda: choose("update")).pack(side="right")
    ttk.Button(btns, text="Not now", command=lambda: choose("later")).pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", lambda: choose("later"))
    _run(root)
    return result["choice"]


def choose_rollback(versions: List[Dict[str, str]], current_rev: str) -> Optional[str]:
    """Show recent versions; return the chosen full rev, or None if cancelled/no UI."""
    tk, root, ui = _new_root()
    if tk is None:
        return None
    from tkinter import ttk

    result: Dict[str, Optional[str]] = {"rev": None}
    _header(tk, root, ui, "Switch version")
    body = _body(tk, root)

    ttk.Label(body, text="Select a version to switch to:").pack(
        anchor="w", padx=22, pady=(16, 8)
    )

    listwrap = tk.Frame(body, bg=_CARD)
    listwrap.pack(fill="both", expand=True, padx=22)
    sb = ttk.Scrollbar(listwrap, orient="vertical")
    lb = tk.Listbox(listwrap, width=64, height=min(12, max(3, len(versions))),
                    activestyle="none", font=ui["mono"], bd=0, relief="flat",
                    highlightthickness=1, highlightbackground=_BORDER,
                    selectbackground=_SEL, selectforeground=_TEXT,
                    bg=_CARD, fg=_TEXT, yscrollcommand=sb.set)
    sb.config(command=lb.yview)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)

    cur_index = 0
    for i, v in enumerate(versions):
        here = v["rev"].startswith(current_rev) or (current_rev and current_rev.startswith(v["rev"]))
        if here:
            cur_index = i
        marker = "  \u25c0 current" if here else ""
        lb.insert("end", f' {v["date"]}   {v["short"]}   {v["subject"]}{marker}')
    if versions:
        lb.selection_set(cur_index)
        lb.see(cur_index)

    hint = ttk.Label(body,
                     text="Tip: you can also switch back later from the app's status bar.",
                     style="Muted.TLabel")
    hint.pack(anchor="w", padx=22, pady=(8, 0))

    btns = ttk.Frame(body)
    btns.pack(fill="x", padx=22, pady=(14, 18))

    def do_switch() -> None:
        sel = lb.curselection()
        if sel:
            result["rev"] = versions[sel[0]]["rev"]
        _finish(root)

    def cancel() -> None:
        result["rev"] = None
        _finish(root)

    ttk.Button(btns, text="Cancel", command=cancel).pack(side="left")
    ttk.Button(btns, text="Switch to this version", style="Accent.TButton",
               command=do_switch).pack(side="right")
    lb.bind("<Double-Button-1>", lambda _e: do_switch())

    root.protocol("WM_DELETE_WINDOW", cancel)
    _run(root)
    return result["rev"]


def ask_yes_no(title: str, message: str) -> bool:
    tk, root, ui = _new_root()
    if tk is None:
        return False
    from tkinter import ttk

    result = {"yes": False}
    _header(tk, root, ui, title)
    body = _body(tk, root)
    ttk.Label(body, text=message, wraplength=360, justify="left").pack(
        anchor="w", padx=22, pady=(18, 12)
    )
    btns = ttk.Frame(body)
    btns.pack(fill="x", padx=22, pady=(4, 18))

    def answer(value: bool) -> None:
        result["yes"] = value
        _finish(root)

    ttk.Button(btns, text="No", command=lambda: answer(False)).pack(side="left")
    ttk.Button(btns, text="Yes", style="Accent.TButton",
               command=lambda: answer(True)).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", lambda: answer(False))
    _run(root)
    return result["yes"]


def notify(title: str, message: str) -> None:
    tk, root, ui = _new_root()
    if tk is None:
        print(f"[dialogs] {title}: {message}")
        return
    from tkinter import ttk

    _header(tk, root, ui, title)
    body = _body(tk, root)
    ttk.Label(body, text=message, wraplength=360, justify="left").pack(
        anchor="w", padx=22, pady=(18, 12)
    )
    btns = ttk.Frame(body)
    btns.pack(fill="x", padx=22, pady=(4, 18))
    ttk.Button(btns, text="OK", style="Accent.TButton",
               command=lambda: _finish(root)).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", lambda: _finish(root))
    _run(root)