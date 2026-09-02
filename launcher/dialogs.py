"""
dialogs.py -- small, themed tkinter dialogs for the launcher.

Uses ONLY tkinter/ttk (bundled with the conda Python); never imports the heavy
GUI stack (PyQt/napari). Every function degrades safely with no display:

    * ask_update      -> "later"  (never update without explicit consent)
    * choose_version  -> None     (cancel)
    * confirm_uninstall -> False  (never delete without explicit consent)
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

# Returned by choose_version() when the user picks Uninstall instead of a
# version, rather than a {channel, rev} selection. A sentinel keeps the
# return type flat and avoids a second out-parameter.
UNINSTALL = "__uninstall__"

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


CHANNEL_LABELS = {"stable": "Stable", "dev": "Development"}

# Shown when the Development channel is selected. Not a scare message -- the
# point is that the choice is reversible from this same dialog.
_DEV_WARNING = ("Development builds are unreleased and may be broken. "
                "You can switch back to Stable here at any time.")

# The synthetic first row. Selecting it means "follow this channel's tip and
# keep auto-updating"; selecting any real commit below means "pin to exactly
# this version and stop updating".
_LATEST_ROW = " Latest  --  follow this channel (auto-update)"


def choose_version(overview: Dict, current_rev: str = "") -> Optional[object]:
    """
    Pick a channel and a version in one dialog.

    `overview` is exactly what `updater.channel_overview()` returns; this
    function does no git work of its own, matching the rest of this module.

    Returns:
        None                                cancelled, or no display
        UNINSTALL                           the uninstall button
        {"channel": <name>, "rev": None}    track that channel's tip
        {"channel": <name>, "rev": <sha>}   pin to that exact commit

    The caller decides what those mean in git terms (switch / unpin / pin) --
    see run_app._run_rollback. Switching the radio only re-renders the list;
    nothing is applied until the primary button is pressed, so a mis-click
    while recovering from a crash costs nothing.
    """
    tk, root, ui = _new_root()
    if tk is None:
        return None
    from tkinter import ttk

    # `clam` renders radiobuttons on its own grey; restyle them onto the card.
    # Done here rather than in _new_root so no other dialog is affected.
    try:
        _st = ttk.Style(root)
        _st.configure("Chan.TRadiobutton", background=_CARD, foreground=_TEXT,
                      font=ui["font"])
        _st.map("Chan.TRadiobutton",
                background=[("active", _CARD)],
                foreground=[("disabled", _MUTED)])
        _RB = "Chan.TRadiobutton"
    except Exception:
        _RB = "TRadiobutton"

    channels: Dict = overview.get("channels") or {}
    order = [c for c in ("stable", "dev") if c in channels] or sorted(channels)
    if not order:
        _finish(root)
        return None
    current_channel = overview.get("current") or order[0]
    if current_channel not in order:
        current_channel = order[0]
    head = overview.get("head") or current_rev or ""
    pinned = bool(overview.get("pinned"))

    result: Dict[str, Optional[object]] = {"value": None}
    _header(tk, root, ui, "Switch version")
    body = _body(tk, root)

    # --- channel toggle -------------------------------------------------- #
    chan_var = tk.StringVar(value=current_channel)
    chan_row = ttk.Frame(body)
    chan_row.pack(anchor="w", fill="x", padx=22, pady=(16, 0))
    ttk.Label(chan_row, text="Channel:", font=ui["font_bold"]).pack(side="left")
    for name in order:
        entry = channels.get(name) or {}
        rb = ttk.Radiobutton(chan_row, text=CHANNEL_LABELS.get(name, name.title()),
                             value=name, variable=chan_var, style=_RB)
        rb.pack(side="left", padx=(10, 0))
        if not entry.get("available"):
            # Unreachable channels are disabled rather than hidden, so the
            # option is visibly there and the reason is stated.
            rb.state(["disabled"])

    note = ttk.Label(body, text="", style="Muted.TLabel", wraplength=430,
                     justify="left")
    note.pack(anchor="w", fill="x", padx=22, pady=(6, 8))

    listwrap = tk.Frame(body, bg=_CARD)
    listwrap.pack(fill="both", expand=True, padx=22)
    sb = ttk.Scrollbar(listwrap, orient="vertical")
    # Wide enough that the longest row -- a subject plus a trailing marker --
    # is not clipped; the earlier 64 cut "<- current" to "<- curren".
    lb = tk.Listbox(listwrap, width=78, height=12,
                    activestyle="none", font=ui["mono"], bd=0, relief="flat",
                    highlightthickness=1, highlightbackground=_BORDER,
                    selectbackground=_SEL, selectforeground=_TEXT,
                    bg=_CARD, fg=_TEXT, yscrollcommand=sb.set)
    sb.config(command=lb.yview)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)

    btns = ttk.Frame(body)
    btns.pack(fill="x", padx=22, pady=(14, 18))
    go = ttk.Button(btns, text="Switch to this version", style="Accent.TButton")

    # `rows` maps a listbox index to the rev it means (None = the Latest row).
    rows: List[Optional[str]] = []

    def render(*_args) -> None:
        name = chan_var.get()
        entry = channels.get(name) or {}
        lb.delete(0, "end")
        rows.clear()

        if not entry.get("available"):
            reason = entry.get("reason") or "unavailable"
            note.config(text=f"The {CHANNEL_LABELS.get(name, name)} channel is "
                             f"not available: {reason}.")
            go.state(["disabled"])
            return
        go.state(["!disabled"])

        msgs = []
        if name != current_channel:
            msgs.append(f"You are currently on "
                        f"{CHANNEL_LABELS.get(current_channel, current_channel)}. "
                        f"Switching replaces the application files and may "
                        f"update dependencies.")
        if name == "dev":
            msgs.append(_DEV_WARNING)
        if pinned and name == current_channel:
            msgs.append("This version is pinned, so updates are paused. "
                        "Choose Latest to resume them.")
        note.config(text=" ".join(msgs))

        is_here = (name == current_channel)
        # When following a channel, its tip IS the Latest row -- so only that
        # row gets the marker. Marking the tip commit too (as an earlier version
        # did) labelled two rows "current" and implied that selecting the commit
        # was a no-op, when it would in fact pin and stop updates.
        tracking_tip = is_here and not pinned
        lb.insert("end", _LATEST_ROW + ("   <- current" if tracking_tip else ""))
        rows.append(None)

        select_at = 0   # Latest: correct default whenever nothing is pinned
        for v in entry.get("versions") or []:
            here = is_here and bool(head) and v["rev"] == head
            tag = "   <- pinned here" if (here and pinned) else ""
            lb.insert("end", f' {v["date"]}   {v["short"]}   {v["subject"]}{tag}')
            rows.append(v["rev"])
            if here and pinned:
                select_at = len(rows) - 1
        lb.selection_clear(0, "end")
        lb.selection_set(select_at)
        lb.see(select_at)

    chan_var.trace_add("write", render)
    render()

    def do_switch() -> None:
        sel = lb.curselection()
        if sel and sel[0] < len(rows):
            result["value"] = {"channel": chan_var.get(), "rev": rows[sel[0]]}
        _finish(root)

    def cancel() -> None:
        result["value"] = None
        _finish(root)

    def uninstall() -> None:
        result["value"] = UNINSTALL
        _finish(root)

    go.config(command=do_switch)
    ttk.Button(btns, text="Cancel", command=cancel).pack(side="left")
    ttk.Button(btns, text="Uninstall HIBACHI...", command=uninstall).pack(
        side="left", padx=(8, 0)
    )
    go.pack(side="right")
    lb.bind("<Double-Button-1>", lambda _e: do_switch())

    root.protocol("WM_DELETE_WINDOW", cancel)
    _run(root)
    return result["value"]


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


def confirm_uninstall(paths: List[str]) -> bool:
    """Confirm complete removal. Lists exactly what will be deleted.

    Returns False with no display, so an uninstall never proceeds unattended.
    The confirm button stays disabled until the checkbox is ticked: this dialog
    is one keypress away from the version list, and the action is irreversible.
    """
    tk, root, ui = _new_root()
    if tk is None:
        return False
    from tkinter import ttk

    result = {"yes": False}
    _header(tk, root, ui, "Uninstall HIBACHI")
    body = _body(tk, root)

    ttk.Label(
        body,
        text="This permanently deletes HIBACHI, its Python environment and its "
             "logs. Nothing is backed up and this cannot be undone.",
        wraplength=430, justify="left",
    ).pack(anchor="w", padx=22, pady=(16, 4))

    ttk.Label(body, text="The following will be removed:",
              style="Muted.TLabel").pack(anchor="w", padx=22, pady=(6, 4))

    listwrap = tk.Frame(body, bg=_CARD)
    listwrap.pack(fill="both", expand=True, padx=22)
    lb = tk.Listbox(listwrap, width=60, height=min(6, max(2, len(paths))),
                    activestyle="none", font=ui["mono"], bd=0, relief="flat",
                    highlightthickness=1, highlightbackground=_BORDER,
                    bg=_CARD, fg=_TEXT)
    for p in paths:
        lb.insert("end", f" {p}")
    lb.pack(fill="both", expand=True)

    ttk.Label(
        body,
        text="Your image data and exported results are NOT stored here and are "
             "not affected.",
        style="Muted.TLabel", wraplength=430, justify="left",
    ).pack(anchor="w", padx=22, pady=(8, 0))

    # Bind to THIS root explicitly. A masterless BooleanVar attaches to
    # tkinter._default_root, which is whatever root was created last -- and
    # _finish() only quits the mainloop, leaving destroy() to _run(). Any path
    # that skips _run leaves a live stale root, and the variable then reads from
    # the wrong interpreter so the checkbox silently never takes effect.
    agreed = tk.BooleanVar(master=root, value=False)
    btns = ttk.Frame(body)

    def answer(value: bool) -> None:
        result["yes"] = value
        _finish(root)

    confirm = ttk.Button(btns, text="Uninstall", style="Accent.TButton",
                         command=lambda: answer(True), state="disabled")

    def toggled() -> None:
        confirm.configure(state="normal" if agreed.get() else "disabled")

    ttk.Checkbutton(body, text="I understand these files will be deleted",
                    variable=agreed, command=toggled).pack(
        anchor="w", padx=22, pady=(10, 0)
    )

    btns.pack(fill="x", padx=22, pady=(12, 18))
    ttk.Button(btns, text="Cancel", command=lambda: answer(False)).pack(side="left")
    confirm.pack(side="right")

    root.protocol("WM_DELETE_WINDOW", lambda: answer(False))
    _run(root)
    return result["yes"]


def _open_path(path: str) -> None:
    """Open a file/folder in the OS file manager (best-effort, no heavy deps)."""
    import subprocess
    import sys
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print(f"[dialogs] could not open {path}: {exc}")


def crash_report(
    summary: str,
    details: str,
    log_dir: str,
    report_path: Optional[str] = None,
    offer_rollback: bool = False,
) -> bool:
    """The crash window shown after HIBACHI exits abnormally.

    Displays the collected diagnostics in a scrollable, selectable box and gives
    the user one-click ways to hand them over: a **Copy** button, an **Open logs
    folder** button, and (when written) the path to a consolidated
    ``crash-report.txt``. When *offer_rollback* is True a "Roll back" action is
    included.

    Parameters
    ----------
    summary : str
        One-line human summary, e.g. "HIBACHI stopped unexpectedly - killed by
        SIGABRT (raw exit code -6)".
    details : str
        The full diagnostic text to show and to copy to the clipboard.
    log_dir : str
        Folder the "Open logs folder" button opens.
    report_path : str, optional
        Path to a saved consolidated report, shown so the user can find it even
        after this window closes (the reliable cross-platform artefact: some
        Linux clipboards are cleared when the owning app exits).
    offer_rollback : bool
        Whether to include the "Roll back to a previous version" action.

    Returns
    -------
    bool
        True if the user chose to roll back; otherwise False. With no display it
        prints the summary and report path to the console and returns False.
    """
    tk, root, ui = _new_root()
    if tk is None:
        # Headless / no display: make sure the user still learns where to look.
        print(f"[crash] {summary}")
        if report_path:
            print(f"[crash] Full report saved to: {report_path}")
        print(f"[crash] Logs folder: {log_dir}")
        return False
    from tkinter import ttk

    result = {"rollback": False}
    root.resizable(True, True)  # let the user enlarge to read the trace
    _header(tk, root, ui, "HIBACHI stopped unexpectedly")
    body = _body(tk, root)

    ttk.Label(
        body,
        text=(summary + "\n\nThe details below help us find the cause. Please send "
              "them to the developers \u2014 use Copy, or attach the saved report file."),
        wraplength=620, justify="left",
    ).pack(anchor="w", padx=22, pady=(18, 10))

    # --- diagnostics box (monospace, scrollable, selectable) --------------- #
    box_wrap = tk.Frame(body, bg=_BORDER)
    box_wrap.pack(fill="both", expand=True, padx=22)
    sb = ttk.Scrollbar(box_wrap, orient="vertical")
    box = tk.Text(box_wrap, width=92, height=18, wrap="none", relief="flat",
                  bg=_CARD, fg=_TEXT, font=ui["mono"], padx=10, pady=8,
                  highlightthickness=1, highlightbackground=_BORDER,
                  yscrollcommand=sb.set)
    sb.config(command=box.yview)
    sb.pack(side="right", fill="y")
    box.pack(side="left", fill="both", expand=True)
    box.insert("1.0", details)
    box.mark_set("insert", "1.0")
    box.see("1.0")  # show the top (header + crash traceback), not the short tail
    # Native selection + Ctrl/Cmd-C still work; the Copy button copies the
    # canonical `details` string regardless of any stray edits, so leaving the
    # widget editable is harmless and gives the best manual-copy behaviour.

    if report_path:
        ttk.Label(body, text=f"A copy was saved to:  {report_path}",
                  style="Muted.TLabel").pack(anchor="w", padx=22, pady=(8, 0))

    # --- actions ----------------------------------------------------------- #
    btns = ttk.Frame(body)
    btns.pack(fill="x", padx=22, pady=(14, 18))

    def do_copy() -> None:
        try:
            root.clipboard_clear()
            root.clipboard_append(details)
            root.update()  # flush to the X selection while we're still alive
            copy_btn.configure(text="Copied \u2713")
            root.after(1500, lambda: copy_btn.configure(text="Copy details"))
        except Exception as exc:
            print(f"[dialogs] clipboard copy failed: {exc}")

    copy_btn = ttk.Button(btns, text="Copy details", style="Accent.TButton", command=do_copy)
    copy_btn.pack(side="left")
    ttk.Button(btns, text="Open logs folder",
               command=lambda: _open_path(log_dir)).pack(side="left", padx=(8, 0))

    def close() -> None:
        result["rollback"] = False
        _finish(root)

    if offer_rollback:
        def roll_back() -> None:
            result["rollback"] = True
            _finish(root)
        ttk.Button(btns, text="Roll back to a previous version",
                   command=roll_back).pack(side="right")
        ttk.Button(btns, text="Close", command=close).pack(side="right", padx=(0, 8))
    else:
        ttk.Button(btns, text="Close", command=close).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", close)
    _run(root)
    return result["rollback"]


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