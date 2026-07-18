"""
batch_runner: the child-process entry point for batch segmentation.

The GUI runs the batch in a separate OS process so a long, uninterruptible
native stage can be cancelled *instantly* by terminating that process — killing
a worker *thread* mid-call is unsafe (it can strand Python's GIL and hang the
whole app). A crash in a native segmentation library also can't take the GUI
down with it.

The parent (see batch_progress_dialog.py) does the scan + reprocess prompt on
the GUI thread, then spawns run_batch_process() with a resolved per-folder plan.
This module has no import-time side effects and creates no Qt objects, so it is
safe to import in a spawned child.

IPC: a single multiprocessing.Queue carries tagged messages back to the parent:
    ("log",      str)                  captured stdout/stderr text
    ("progress", dict)                 folder/step progress events
    ("done",     {success,failed,skipped})
    ("error",    traceback_str)
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List


class _QueueWriter:
    """A minimal file-like object that ships everything written to it back to
    the parent as ("log", text) messages. Installed as stdout/stderr in the
    child so all existing print()/traceback output shows up in the GUI console."""

    def __init__(self, queue):
        self._q = queue

    def write(self, text):
        if text:
            try:
                self._q.put(("log", text))
            except Exception:
                pass
        return len(text) if text else 0

    def flush(self):
        pass

    def isatty(self):
        return False


def run_batch_process(folders: List[str], force_map: Dict[str, bool], queue) -> None:
    """
    Child-process entry: process `folders` per `force_map`, reporting over queue.

    Must be a module-level function so it is picklable for a spawned process.
    All arguments are plain picklable data; the ProjectManager/BatchProcessor are
    rebuilt here from the folder list (they read everything else from disk).
    """
    # Route all output back to the parent's console pane.
    sys.stdout = _QueueWriter(queue)
    sys.stderr = _QueueWriter(queue)

    # Detach into our own process group / session (POSIX) so the GUI can kill
    # this entire subtree on Cancel — including any worker Pool the feature
    # pipeline spawns — without ever touching the GUI's own process group.
    if os.name == "posix":
        try:
            os.setsid()
        except Exception:
            pass

    try:
        # Imported here (not at module top) so the import cost lands in the child
        # and any import error is reported through the queue rather than crashing
        # silently during spawn.
        from .project_manager import ProjectManager
        from .batch_processor import BatchProcessor

        pm = ProjectManager()
        pm.image_folders = list(folders)
        processor = BatchProcessor(pm)

        def _cb(event):
            try:
                queue.put(("progress", event))
            except Exception:
                pass

        success, failed, skipped = processor.run_folders(force_map, progress_callback=_cb)
        queue.put(("done", {"success": success, "failed": failed, "skipped": skipped}))
    except Exception:
        try:
            queue.put(("error", traceback.format_exc()))
        except Exception:
            pass