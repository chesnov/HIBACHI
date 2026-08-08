"""project_view_window: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import yaml  # type: ignore
from PyQt5.QtGui import QCloseEvent, QIcon  # type: ignore
from PyQt5.QtCore import Qt, QEvent  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QFileDialog, QMessageBox, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QLabel, QInputDialog
)

from .gui_text_utils import app_icon_path, clean_filename_for_matching
from .cross_channel_window import (
    CrossChannelAnalyzerWindow, list_relational_analyses, open_sample_overlay,
)
from .metadata import MetadataExtractor
from .project_manager import ProjectManager
from .project_scaffolding import apply_template_config_to_project
from .project_selection import (
    classify_path, RecentProjects, PROJECT, RAW_IMAGES, PARENT_OF_PROJECTS,
    MULTICHANNEL_PROJECT, EMPTY, MISSING, build_channel_registry,
    build_single_channel_registry,
)
try:
    from .project_selection import WelcomeWidget, ProjectContentsView  # need Qt; always here
except Exception:  # pragma: no cover
    WelcomeWidget = None  # type: ignore
    ProjectContentsView = None  # type: ignore

# --- Optional BatchProcessor import ---
try:
    from .batch_processor import BatchProcessor
except ImportError as e:
    print(f"WARNING: Failed to import BatchProcessor: {e}. "
          "Batch processing button will be disabled.")
    BatchProcessor = None  # type: ignore



class ProjectViewWindow(QMainWindow):
    """The main entry window for selecting a project."""

    def __init__(self, project_manager: ProjectManager):
        super().__init__()
        self.project_manager = project_manager
        self.recent = RecentProjects()
        self._content_view = None       # the unified ProjectContentsView (or None)
        self._cross_scan_dir = None     # dir the cross-channel analyzer should scan
        self._project_root = None       # project root that holds RELATIONAL_ANALYSIS
        self._channel_dirs = []          # channel folders currently shown (to detect new ones)
        self._batch_dialog = None       # live batch progress dialog (if running)
        # Set when we open a sample in the napari view; on returning (window
        # re-activates) we refresh the tree so status / "last edited" reflect any
        # processing just done, instead of showing stale values.
        self._pending_content_refresh = False
        self._last_opened_folder = None  # re-highlighted on return from an image
        # Folder currently being set up, used to break the organize -> open_path ->
        # organize cycle if setup ever reports success without creating anything.
        self._organizing = None
        self.initUI()
        self.setAttribute(Qt.WA_QuitOnClose)

    def initUI(self) -> None:
        self.setWindowTitle("Image Segmentation Project")
        _icon = app_icon_path()
        if _icon:
            self.setWindowIcon(QIcon(_icon))
        # Open occupying the full vertical span of the screen (moderate width),
        # since the project tree is tall — avoids the squished default height.
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            width = min(900, avail.width())
            self.setGeometry(avail.x() + 60, avail.y(), width, avail.height())
        except Exception:
            self.setGeometry(100, 100, 860, 560)

        central_widget = QWidget()
        layout = QVBoxLayout()

        # Guided welcome panel: recent projects, drag-and-drop, and forgiving
        # Browse buttons. Every selection is routed through open_path(), which
        # classifies it and does the right thing (open / organize / drill in).
        self.welcome = None
        if WelcomeWidget is not None:
            self.welcome = WelcomeWidget(self.recent)
            self.welcome.path_chosen.connect(self.open_path)
            layout.addWidget(self.welcome)

        self.project_path_label = QLabel("Project Path: Not Selected")
        layout.addWidget(self.project_path_label)

        # Content area. Single- and multi-channel projects both render into the
        # same ProjectContentsView (a checkbox tree); we just swap the instance.
        self._content_container = QWidget()
        self._content_holder_layout = QVBoxLayout(self._content_container)
        self._content_holder_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._content_container)

        # Fixed bottom action bar -- identical for every project kind, so nothing
        # appears/disappears as you move between single- and multi-channel views.
        button_layout = QHBoxLayout()

        self.process_selected_btn = QPushButton("Process Selected")
        self.process_selected_btn.setToolTip(
            "Batch-process every checked image/channel. Use the selection tools "
            "at the top to check images or a whole channel at once."
        )
        self.process_selected_btn.clicked.connect(self._process_selected)
        self.process_selected_btn.setEnabled(False)
        button_layout.addWidget(self.process_selected_btn)

        self.cross_channel_btn = QPushButton("Open Cross-Channel Analyzer")
        self.cross_channel_btn.clicked.connect(self.open_cross_channel_analyzer)
        self.cross_channel_btn.setEnabled(False)  # enable once a project loads
        button_layout.addWidget(self.cross_channel_btn)

        self.set_config_btn = QPushButton("⚙ Set New Channel Config…")
        self.set_config_btn.setToolTip(
            "Choose a YAML config template and apply its processing parameters to\n"
            "the checked images (image dimensions are preserved). Available only\n"
            "when the checked images belong to a single channel."
        )
        self.set_config_btn.clicked.connect(self.set_channel_config)
        self.set_config_btn.setEnabled(False)
        button_layout.addWidget(self.set_config_btn)

        self.optimize_btn = QPushButton("\U0001f39b Optimize Parameters\u2026")
        self.optimize_btn.setToolTip(
            "Find ONE shared initial-segmentation config for the checked images "
            "that minimizes how far each image's result drifts from its own "
            "optimum (curvature-weighted, using the actual masks). Available when "
            "two or more images of the same mode are checked."
        )
        self.optimize_btn.clicked.connect(self._optimize_parameters)
        self.optimize_btn.setEnabled(False)
        button_layout.addWidget(self.optimize_btn)

        self.config_library_btn = QPushButton("\U0001f4da Config Library\u2026")
        self.config_library_btn.setToolTip(
            "Browse, import, duplicate, rename and export the configs in your "
            "cross-project library (and see any that failed to load)."
        )
        self.config_library_btn.clicked.connect(self.open_config_library_manager)
        button_layout.addWidget(self.config_library_btn)

        self.export_run_btn = QPushButton("\u2b07 Export run config\u2026")
        self.export_run_btn.setToolTip(
            "Export a single checked, processed image's run config — either as a "
            "reusable preset in your Config Library, or to a file (byte-for-byte, "
            "keeping saved thresholds, dimensions and the pipeline version) so a "
            "collaborator can reproduce the run."
        )
        self.export_run_btn.clicked.connect(self._export_run_config)
        self.export_run_btn.setEnabled(False)
        button_layout.addWidget(self.export_run_btn)

        layout.addLayout(button_layout)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Unobtrusive version indicator in the status bar (check for updates /
        # switch versions). Guarded so it can never block the home window from
        # opening -- if this isn't a git checkout it simply won't appear.
        try:
            from .version_manager import attach_version_status
            attach_version_status(self)
        except Exception as _exc:
            print(f"[version] status widget unavailable: {_exc}")

    # ---- content view plumbing ------------------------------------------- #
    def _install_content_view(self, view) -> None:
        """Swap the embedded contents view (or clear it when view is None)."""
        if self._content_view is not None:
            self._content_holder_layout.removeWidget(self._content_view)
            self._content_view.deleteLater()
        self._content_view = view
        if view is not None:
            self._content_holder_layout.addWidget(view)

    def _update_action_buttons(self) -> None:
        """Enable the bottom bar according to the current checked set."""
        view = self._content_view
        checked = view.checked_folders() if view is not None else []
        self.process_selected_btn.setEnabled(bool(checked))

        # Set Config applies per-channel, so it is only valid when the checked
        # images belong to at most one channel (a single image, several images in
        # the same channel, or a whole single-channel project). Checking a whole
        # multi-channel image spans several channels and therefore disables it.
        keys = view.checked_channel_keys() if view is not None else set()
        self.set_config_btn.setEnabled(bool(checked) and len(keys) <= 1)

        # Export run config is a single-folder, reproducibility action: enable it
        # only when exactly one checked folder actually has a processed run config.
        one = checked[0] if len(checked) == 1 else None
        self.export_run_btn.setEnabled(
            one is not None and self._run_config_path(one) is not None
        )

        # Parameter optimization reconciles several images into one shared
        # config, so it needs >= 2 checked images that all share one mode.
        self.optimize_btn.setEnabled(
            len(checked) >= 2 and self._uniform_mode(checked) is not None
        )

    def _uniform_mode(self, folders: list):
        """Return the common processing mode of `folders`, or None if they are
        mixed / undetermined."""
        modes = set()
        for folder in folders:
            try:
                mode = self.project_manager.get_image_details(folder).get('mode')
            except Exception:
                return None
            if not mode or mode in ('unknown', 'error'):
                return None
            modes.add(mode)
        return next(iter(modes)) if len(modes) == 1 else None

    def _optimize_parameters(self) -> None:
        """Optimize one shared initial-segmentation config across the checked
        images (curvature-weighted compromise, using their masks)."""
        if self._content_view is None:
            return
        checked = self._content_view.checked_folders()
        if len(checked) < 2:
            return
        mode = self._uniform_mode(checked)
        if mode is None:
            QMessageBox.information(
                self, "Mixed modes",
                "Parameter optimization needs two or more checked images that "
                "share the same processing mode (all 2D, or all 3D)."
            )
            return
        try:
            from .parameter_optimizer import run_optimization_dialog
        except Exception as exc:
            QMessageBox.warning(
                self, "Unavailable",
                f"Parameter optimizer could not be loaded:\n{exc}")
            return
        run_optimization_dialog(self, list(checked), mode)

    def open_cross_channel_analyzer(self):
        if self._cross_scan_dir:
            # build_consolidated_sample_registry scans os.path.dirname(project_path),
            # so anchoring project_path at a channel dir makes it scan the whole
            # project root for sibling channels.
            self.project_manager.project_path = self._cross_scan_dir
        registry = self.project_manager.build_consolidated_sample_registry()
        if not registry:
            QMessageBox.warning(
                self,
                "No Compatible Data",
                "Could not find any multi-channel samples in the parent directory.\n\n"
                "Ensure your project is organized into 'Channel_X' folders, and that "
                "they share matching sample names."
            )
            return

        self.analyzer_window = CrossChannelAnalyzerWindow(self.project_manager)
        self.analyzer_window.show()
        self.analyzer_window.raise_()
        self.analyzer_window.activateWindow()

    def open_path(self, selected_path: str) -> None:
        """
        Act on any user-selected path (from Browse, a recent row, or a drop),
        deciding what it is instead of assuming the user picked correctly.

          * a project            -> open it
          * loose raw images      -> offer to organize into a project
          * a folder of projects  -> let the user pick which one
          * an image file         -> use its containing folder (handled by classify)
          * empty / missing       -> explain, don't fail silently
        """
        info = classify_path(selected_path)

        if info.redirected_from_file:
            QMessageBox.information(
                self, "Using folder",
                "You selected an image file, so HIBACHI will use its folder:\n\n"
                f"{info.path}"
            )

        if info.kind == MISSING:
            QMessageBox.warning(self, "Not found", info.note)
            return

        if info.kind == EMPTY:
            QMessageBox.warning(
                self, "Nothing to open",
                f"{info.note}\n\nPick a folder that contains images, or a project "
                "folder (whose sub-folders each hold one image and one config)."
            )
            return

        if info.kind == PARENT_OF_PROJECTS:
            names = [os.path.basename(p) for p in info.project_roots]
            choice, ok = QInputDialog.getItem(
                self, "Choose a project",
                f"{os.path.basename(info.path)} contains several projects.\n"
                "Which would you like to open?",
                names, 0, False
            )
            if ok and choice:
                self.open_path(info.project_roots[names.index(choice)])
            return

        if info.kind == MULTICHANNEL_PROJECT:
            self.open_multichannel(info)
            return

        # PROJECT or RAW_IMAGES: hand off to the existing loader/scaffolder, which
        # already knows how to organize raw images and populate the view.
        self.project_manager.project_path = info.path
        self._cross_scan_dir = info.path
        self._project_root = info.path
        self.cross_channel_btn.setEnabled(True)
        self.project_path_label.setText(f"Project Path: {info.path}")
        self._load_or_organize(info.path)

    def open_multichannel(self, info) -> None:
        """Show the sample→channel tree for a multi-channel project, in-place."""
        if ProjectContentsView is None:
            QMessageBox.information(
                self, "Multi-channel project",
                f"{info.note}\nOpen a specific Channel_* folder to work on it."
            )
            return

        registry = build_channel_registry(info.channel_dirs)
        if not registry:
            QMessageBox.warning(self, "Empty project",
                                "No samples were found in the channel folders.")
            return

        self.project_path_label.setText(f"Project Path: {info.path}  (multi-channel)")
        self.recent.add(info.path)
        if self.welcome is not None:
            self.welcome.refresh_recents()

        view = ProjectContentsView(
            registry, channel_dirs=info.channel_dirs,
            project_dir=info.path, multichannel=True,
            analyses=list_relational_analyses(info.path),
        )
        view.open_requested.connect(self._open_sample_folder)
        view.overlay_requested.connect(self._open_overlay)
        view.selection_changed.connect(self._update_action_buttons)
        view.add_channel_requested.connect(self._add_channel)
        view.resetup_requested.connect(self._resetup_project)
        self._install_content_view(view)

        self._project_root = info.path
        self._channel_dirs = list(info.channel_dirs or [])
        # Anchor cross-channel scanning at a channel dir so its parent (the
        # project root) is what gets scanned for sibling channels.
        self._cross_scan_dir = info.channel_dirs[0] if info.channel_dirs else info.path
        self.project_manager.project_path = self._cross_scan_dir
        self.cross_channel_btn.setEnabled(True)
        self._update_action_buttons()

    def _open_sample_folder(self, folder: str) -> None:
        """Open one image/channel folder in the interactive segmentation view."""
        if not folder:
            return
        # Re-entrancy guard. Launching the segmentation viewer runs its setup --
        # including a nested modal dialog (restore-from-checkpoint's "View
        # results / Restart") -- SYNCHRONOUSLY inside this double-click handler.
        # That modal spins a nested Qt event loop; without this guard, an event
        # delivered during that loop (e.g. a queued/duplicate open) can re-invoke
        # this method and open another viewer whose setup opens another modal,
        # stacking dialogs without bound until the machine is unresponsive.
        # Ignoring re-entrant calls while a launch is in flight prevents that.
        if getattr(self, "_launching_viewer", False):
            return
        self._launching_viewer = True
        try:
            # Processing may change this folder's status/outputs; refresh the tree
            # when we come back so it doesn't show stale "in progress / N ago".
            self._pending_content_refresh = True
            self._last_opened_folder = folder
            self.hide()
            from .app_launch import interactive_segmentation_with_config  # lazy: avoid cycle
            interactive_segmentation_with_config(folder, project_manager=self.project_manager)
        finally:
            self._launching_viewer = False

    def _open_overlay(self, sample_name: str) -> None:
        """Open a sample's multi-channel viewer (parent row double-click).

        Always shows the raw intensity channels (segmentation hidden). If an
        analysis is selected in the picker, its cross-channel layers are added
        on top and shown.
        """
        view = self._content_view
        if view is None:
            return
        analysis = view.current_analysis()  # None on the neutral entry

        # Re-anchor the scan at THIS project's channel dir and rebuild the
        # consolidated registry fresh every time. project_path can drift (opening
        # a channel leaf, the analyzer, another project) and the registry may be
        # stale from a previous project, so relying on a cached one is unsafe.
        if self._cross_scan_dir:
            self.project_manager.project_path = self._cross_scan_dir
        self.project_manager.build_consolidated_sample_registry()

        # The tree keys samples by folder basename; the consolidated registry and
        # analysis folders use the "clean" name. Map between the two.
        clean = clean_filename_for_matching(sample_name)
        if clean not in self.project_manager.sample_registry:
            QMessageBox.warning(
                self, "No cross-channel data",
                f"Could not find consolidated channels for '{sample_name}'."
            )
            return
        try:
            open_sample_overlay(self.project_manager, clean, analysis, parent=self)
        except Exception as exc:
            QMessageBox.critical(self, "Overlay Error", f"Could not open overlay:\n{exc}")

    def _maybe_rediscover_channels(self) -> None:
        """Rebuild the multi-channel tree if channel folders were added or removed
        on disk since it was last built (e.g. a newly generated synthetic
        channel). Cheap no-op when the set of channel folders is unchanged, so
        it's safe to call on every re-activation."""
        view = self._content_view
        if view is None or not getattr(view, "_multichannel", False) or not self._project_root:
            return
        try:
            info = classify_path(self._project_root)
        except Exception as exc:
            print(f"[project view] channel rediscover failed: {exc}")
            return
        if getattr(info, "kind", None) != MULTICHANNEL_PROJECT:
            return
        new_dirs = set(info.channel_dirs or [])
        if new_dirs and new_dirs != set(self._channel_dirs):
            # The structure changed; rebuild from the fresh discovery. This
            # re-renders the whole tree, so it only runs when something actually
            # changed rather than on every focus event.
            self.open_multichannel(info)

    def _rescan_analyses(self) -> None:
        """Refresh the overlay picker from disk (cheap; called on re-activation)."""
        view = self._content_view
        if view is None or not getattr(view, "_multichannel", False) or not self._project_root:
            return
        try:
            view.set_analyses(list_relational_analyses(self._project_root))
        except Exception as exc:
            print(f"[analyses] rescan failed: {exc}")

    def changeEvent(self, event) -> None:
        # When the window regains focus (e.g. after closing the analyzer or an
        # overlay viewer), rescan so freshly-run analyses appear in the picker.
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._rescan_analyses()
            # Pick up channel folders added while we were away (e.g. a synthetic
            # channel generated from the analyzer) without a manual reload.
            self._maybe_rediscover_channels()
            # If we just came back from processing a sample, recompute the tree so
            # status ("Step k/n …") and "last edited" reflect what's now on disk.
            if self._pending_content_refresh:
                self._pending_content_refresh = False
                if self._content_view is not None:
                    try:
                        self._content_view.refresh()
                        self._update_action_buttons()
                        # Keep the image we just came back from highlighted so
                        # it's easy to see where we were in the list.
                        if self._last_opened_folder:
                            self._content_view.highlight_folder(self._last_opened_folder)
                    except Exception as exc:
                        print(f"[project view] tree refresh failed: {exc}")
        super().changeEvent(event)

    def _process_selected(self) -> None:
        if self._content_view is None:
            return
        self._batch_process_folders(self._content_view.checked_folders())

    def _batch_process_folders(self, folders: list) -> None:
        """
        Batch-process the checked folders in a separate process, with a live
        progress dialog (spinner, per-image and per-stage bars, console) and an
        immediate Cancel. The scan + reprocess prompt run here on the GUI thread;
        only the actual processing runs in the child process.
        """
        if not folders:
            return
        if not BatchProcessor:
            QMessageBox.warning(self, "Unavailable", "Batch processor is not available.")
            return

        folders = list(folders)

        # Prescan + reprocess prompt must run on the GUI thread (they may show a
        # QMessageBox). Point a BatchProcessor at exactly the checked set.
        saved = self.project_manager.image_folders
        self.project_manager.image_folders = folders
        try:
            processor = BatchProcessor(self.project_manager)
            complete, partial, scan = processor.prescan_folders()

            restart_complete = restart_partial = False
            if complete or partial:
                choice = processor._prompt_reprocess_choice(complete, partial)
                if choice == 'cancel':
                    return
                restart_complete = restart_partial = (choice == 'restart_all')
            else:
                confirm = QMessageBox.question(
                    self, "Confirm",
                    f"Process {len(folders)} selected image folder"
                    f"{'s' if len(folders) != 1 else ''}?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if confirm != QMessageBox.Yes:
                    return

            force_map = {}
            for fp, info in scan.items():
                st = info.get('status')
                force_map[fp] = ((st == 'complete' and restart_complete)
                                 or (st == 'partial' and restart_partial))
        finally:
            self.project_manager.image_folders = saved

        # Hand the resolved plan to the process-backed progress dialog.
        from .batch_progress_dialog import BatchProgressDialog
        dlg = BatchProgressDialog(folders, force_map, parent=self)
        self._batch_dialog = dlg  # keep a reference alive
        dlg.finished_batch.connect(self._on_batch_finished)
        dlg.show()
        dlg.start()

    def _on_batch_finished(self, success: int, failed: int, skipped: int,
                           cancelled: bool) -> None:
        """Refresh the view after a batch run (or cancellation) completes."""
        if self._content_view is not None:
            self._content_view.refresh()
        self._update_action_buttons()

    def _add_channel(self, project_dir: str) -> None:
        """Extract one more channel from the leftover raw images."""
        from .organize_wizard import run_organize_wizard
        if run_organize_wizard(self, project_dir, mode="add", project_dir=project_dir):
            self.open_path(project_dir)  # re-classify + rebuild the tree with the new channel

    def _resetup_project(self, project_dir: str) -> None:
        """Delete the organized structure, then set up again from scratch.

        Multi-channel projects re-extract from the loose raw source images;
        single-channel projects have their images moved back to the project root
        first. Either way the raw images are preserved.
        """
        from .organize_wizard import (
            reset_multichannel_project, reset_single_channel_project,
            purge_derived_artifacts, existing_channel_indices, run_organize_wizard,
        )
        is_multi = bool(existing_channel_indices(project_dir))
        if is_multi:
            detail = (
                "This deletes ALL channel folders and their processed results for:"
            )
        else:
            detail = (
                "This moves your images back to the project root and deletes the "
                "organized folders and their processed results for:"
            )
        reply = QMessageBox.warning(
            self, "Re-set up project?",
            f"{detail}\n\n{project_dir}\n\n"
            "Any saved cross-channel analyses and synthetic channels from previous "
            "runs are also removed. The raw source images are kept. This cannot be "
            "undone. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        # Clear derived artifacts (cross-channel analyses + synthetic channels)
        # first, then tear down the channel/image structure.
        purged = purge_derived_artifacts(project_dir)
        if is_multi:
            removed = reset_multichannel_project(project_dir)
        else:
            removed = reset_single_channel_project(project_dir)
        print(f"[resetup] removed {len(removed)} folder(s), "
              f"purged {len(purged)} derived artifact(s).")
        if run_organize_wizard(self, project_dir, mode="new", project_dir=project_dir):
            self.open_path(project_dir)

    def _load_or_organize(self, selected_path: str) -> None:
        try:
            # Already organized? Just list it.
            self.project_manager._find_valid_image_folders()
            if self.project_manager.image_folders:
                self._populate_image_list(selected_path)
                return

            # Not organized yet -- are there raw images to set up into a project?
            from .gui_text_utils import is_os_sidecar
            raw_files = [
                f for f in os.listdir(selected_path)
                if f.lower().endswith(('.tif', '.tiff', '.czi'))
                and os.path.isfile(os.path.join(selected_path, f))
                and not is_os_sidecar(f)
            ]
            if raw_files:
                # Re-entry guard. When setup succeeds, this folder is a project, so
                # the open_path() below routes it to a project view and never comes
                # back here. Arriving here twice for the same folder means setup
                # reported success while creating nothing -- previously that spun
                # forever, re-opening the wizard on every pass. Report it once
                # instead.
                if self._organizing == os.path.abspath(selected_path):
                    self._organizing = None
                    QMessageBox.warning(
                        self, "Setup didn't create a project",
                        "Project setup finished but this folder still contains "
                        f"only unorganized images:\n\n{selected_path}\n\n"
                        "Your images were not modified. Check that they can be "
                        "read, then try setting the project up again."
                    )
                    self._install_content_view(None)
                    self._update_action_buttons()
                    return

                from .organize_wizard import run_organize_wizard
                self._organizing = os.path.abspath(selected_path)
                try:
                    if run_organize_wizard(self, selected_path, mode="new",
                                           project_dir=selected_path):
                        # After setup the folder is a single- or multi-channel
                        # project; re-route so it opens in the right view.
                        self.open_path(selected_path)
                finally:
                    self._organizing = None
                return

            # Nothing organized and nothing to organize.
            self._install_content_view(None)
            self._update_action_buttons()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _populate_image_list(self, selected_path: str) -> None:
        """Build the single-channel contents view and record the project as recent."""
        registry = build_single_channel_registry(self.project_manager.image_folders)
        view = ProjectContentsView(
            registry, channel_dirs=None,
            project_dir=selected_path, multichannel=False,
        )
        view.open_requested.connect(self._open_sample_folder)
        view.selection_changed.connect(self._update_action_buttons)
        view.resetup_requested.connect(self._resetup_project)
        self._install_content_view(view)

        if self.project_manager.image_folders:
            self.recent.add(selected_path)
            if self.welcome is not None:
                self.welcome.refresh_recents()

        self.cross_channel_btn.setEnabled(True)
        self._update_action_buttons()

    def set_channel_config(self) -> None:
        """Apply a template YAML to the checked (single-channel) image folders."""
        if self._content_view is None:
            return
        folders = self._content_view.checked_folders()
        if not folders:
            return

        # Resolve the mode of the checked folders so the picker only offers
        # compatible configs. All checked folders belong to at most one channel
        # (the button is disabled otherwise), so they share a mode in practice.
        folder_mode = self._checked_folders_mode(folders)

        template_path = self._pick_template(folder_mode)
        if not template_path:
            return

        # Preview what mode the template targets
        try:
            with open(template_path, 'r') as fh:
                template_preview = yaml.safe_load(fh) or {}
            template_mode = template_preview.get('mode', 'unknown')
            execute_keys = [k for k in template_preview if k.startswith('execute_')]
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not read template:\n{exc}")
            return

        total = len(folders)
        reply = QMessageBox.question(
            self,
            "Apply Config Template",
            f"Template:  {os.path.basename(template_path)}\n"
            f"Mode:      {template_mode}\n"
            f"Steps:     {len(execute_keys)}\n\n"
            f"Apply to the {total} checked image folder(s)?\n\n"
            f"• Processing parameters will be replaced.\n"
            f"• Image dimensions are always preserved.\n"
            f"• Folders with a different mode will be skipped.\n"
            f"• Any already-processed results whose parameters change will be "
            f"CLEARED, so those images reopen unprocessed (they were computed "
            f"with the old parameters).\n\n"
            f"If the template was tuned on a different pipeline version, you'll be "
            f"shown exactly what changes before anything is written.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Reconcile the template against the current canonical schema first, and
        # show the diff for confirmation. Any config problem (bad/no mode, no
        # canonical reference) is raised by the logic layer and surfaced here --
        # never worked around silently.
        from .reconcile_dialog import make_reconcile_confirm
        from .config_library import ConfigLibraryError

        QApplication.setOverrideCursor(Qt.WaitCursor)
        saved = self.project_manager.image_folders
        self.project_manager.image_folders = list(folders)
        try:
            results = apply_template_config_to_project(
                template_path, self.project_manager,
                reconcile_confirm=make_reconcile_confirm(
                    self, context="Set New Channel Config"),
                clear_stale_results=True,
            )
        except ConfigLibraryError as exc:
            self.project_manager.image_folders = saved
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Config error", str(exc))
            return
        except Exception as exc:
            self.project_manager.image_folders = saved
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Config application failed:\n{exc}")
            return
        finally:
            self.project_manager.image_folders = saved
        QApplication.restoreOverrideCursor()

        # The user cancelled the reconcile diff: nothing was written, so we say
        # nothing (no misleading "Done" dialog).
        if results.get('aborted'):
            return

        summary = (
            f"Config template applied.\n\n"
            f"Updated : {results['success']}\n"
            f"Skipped : {results['skipped']}  (different mode or invalid)\n"
            f"Failed  : {results['failed']}\n"
            f"Results cleared : {results.get('cleared', 0)}  "
            f"(parameters changed — these images reopen unprocessed)\n"
        )
        if results['updated_folders']:
            preview = results['updated_folders'][:8]
            summary += "\nUpdated folders:\n" + "\n".join(f"  \u2022 {n}" for n in preview)
            if len(results['updated_folders']) > 8:
                summary += f"\n  \u2026 and {len(results['updated_folders']) - 8} more"

        if results['failed'] > 0:
            QMessageBox.warning(self, "Partial Success", summary)
        else:
            QMessageBox.information(self, "Done", summary)

        if self._content_view is not None:
            self._content_view.refresh()
        self._update_action_buttons()

    # ---- config library plumbing ----------------------------------------- #
    def _checked_folders_mode(self, folders: list):
        """Resolve the common mode of the checked folders, or None if ambiguous.

        Reads each folder's mode via ``get_image_details``; returns the single
        shared mode when they agree (ignoring 'unknown'/'error'), else None so the
        picker falls back to showing all configs.
        """
        modes = set()
        for folder in folders:
            try:
                m = self.project_manager.get_image_details(folder).get('mode')
            except Exception:
                m = None
            if m and m not in ('unknown', 'error'):
                modes.add(m)
        return next(iter(modes)) if len(modes) == 1 else None

    def _pick_template(self, folder_mode):
        """Offer library configs (filtered by mode) plus Import / Browse options.

        Returns an absolute template path, or None if the user cancelled. Any
        config-library problem is surfaced via a message box rather than silently
        skipped.
        """
        from . import config_library as cl
        from .config_library import ConfigLibraryError, ConfigModeError

        _IMPORT = "\u2795  Import from file\u2026 (adds it to your library)"
        _BROWSE = "\U0001f4c1  Browse for a file\u2026 (use once, don't save)"

        try:
            entries = cl.list_all(mode=folder_mode)
        except ConfigLibraryError as exc:
            QMessageBox.critical(self, "Config error", str(exc))
            entries = []

        labels = [e.label for e in entries]
        options = labels + [_IMPORT, _BROWSE]

        mode_note = (f" for mode '{folder_mode}'" if folder_mode
                     else " (mode could not be resolved; showing all)")
        choice, ok = QInputDialog.getItem(
            self, "Choose a config",
            f"Pick a config to apply{mode_note}:",
            options, 0, False
        )
        if not ok or not choice:
            return None

        if choice == _BROWSE:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Config Template", "",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            return path or None

        if choice == _IMPORT:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import config into library", "",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if not path:
                return None
            try:
                entry = cl.import_config(path)
            except ConfigModeError as exc:
                QMessageBox.critical(
                    self, "Config error",
                    f"That file has no valid 'mode' and can't be imported:\n\n{exc}"
                )
                return None
            except FileExistsError:
                reply = QMessageBox.question(
                    self, "Already exists",
                    "A library config with that name already exists.\n\nOverwrite it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return None
                try:
                    entry = cl.import_config(path, overwrite=True)
                except (ConfigLibraryError, OSError) as exc:
                    QMessageBox.critical(self, "Config error", str(exc))
                    return None
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return None
            return entry.path

        # Otherwise the user picked a discovered entry by its label.
        idx = labels.index(choice)
        return entries[idx].path

    def open_config_library_manager(self) -> None:
        """Open the Config Library manager dialog."""
        try:
            from .config_library_dialog import open_config_library
        except Exception as exc:
            QMessageBox.critical(self, "Unavailable",
                                 f"Config Library manager unavailable:\n{exc}")
            return
        open_config_library(self)

    def _stamp_config_name(self, folder: str, name: str) -> None:
        """Record `name` as this folder's config name in its main YAML, so the
        project view's Config column shows it. Also updates the processed run
        config if present, keeping the two in sync."""
        import yaml  # type: ignore
        targets = []
        try:
            yml = next((f for f in os.listdir(folder)
                        if f.lower().endswith((".yaml", ".yml"))), None)
            if yml:
                targets.append(os.path.join(folder, yml))
        except OSError:
            return
        run_cfg = self._run_config_path(folder)
        if run_cfg:
            targets.append(run_cfg)
        for path in targets:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                data["config_name"] = name
                with open(path, "w", encoding="utf-8") as fh:
                    yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
            except Exception:
                pass

    def _run_config_path(self, folder: str):
        """Absolute path to a folder's processed run config, or None if absent.

        Looks for ``<basename>_processed_<mode>/processing_config_<mode>.yaml``
        next to the image, matching how the pipeline writes it.
        """
        if not folder:
            return None
        try:
            details = self.project_manager.get_image_details(folder)
        except Exception:
            return None
        tif_file = details.get('tif_file')
        mode = details.get('mode')
        if not tif_file or not mode or mode in ('unknown', 'error'):
            return None
        basename = os.path.splitext(tif_file)[0]
        path = os.path.join(
            folder, f"{basename}_processed_{mode}", f"processing_config_{mode}.yaml"
        )
        return path if os.path.isfile(path) else None

    def _export_run_config(self) -> None:
        """Export one processed folder's run config verbatim (reproducibility)."""
        if self._content_view is None:
            return
        checked = self._content_view.checked_folders()
        if len(checked) != 1:
            QMessageBox.information(
                self, "Select one image",
                "Check exactly one processed image to export its run config."
            )
            return
        run_cfg = self._run_config_path(checked[0])
        if not run_cfg:
            QMessageBox.information(
                self, "No run config",
                "That image has no processed run config yet. Process it first."
            )
            return

        from . import config_library as cl
        from .config_library import ConfigLibraryError

        # Preview provenance so the user knows they're sharing a full run record.
        try:
            prov = cl.read_provenance(run_cfg)
        except Exception:
            prov = {}
        ver = prov.get("hibachi_version")
        ver_text = (ver.get("short") or ver.get("commit")) if isinstance(ver, dict) else ver

        # Choose a destination: a reusable preset in the user's library, or a
        # standalone file on disk.
        dest_box = QMessageBox(self)
        dest_box.setWindowTitle("Export run config")
        dest_box.setIcon(QMessageBox.Question)
        dest_box.setText("Where should this config go?")
        dest_box.setInformativeText(
            "Save it as a preset in your Config Library to reuse it on other "
            "images, or export it to a file to share (a file keeps the full "
            "reproducibility record; a preset is sanitised for reuse)."
        )
        preset_btn = dest_box.addButton("Save as preset", QMessageBox.AcceptRole)
        file_btn = dest_box.addButton("Export to file\u2026", QMessageBox.ActionRole)
        dest_box.addButton("Cancel", QMessageBox.RejectRole)
        dest_box.setDefaultButton(preset_btn)
        dest_box.exec_()
        clicked = dest_box.clickedButton()

        if clicked == preset_btn:
            default_name = os.path.basename(checked[0])
            name, ok = QInputDialog.getText(
                self, "Save preset", "Preset name:", text=default_name
            )
            if not ok or not name.strip():
                return
            name = name.strip()
            try:
                entry = cl.import_config(run_cfg, name=name)
            except FileExistsError:
                reply = QMessageBox.question(
                    self, "Already exists",
                    f"A library preset named '{name}' already exists.\n\nOverwrite it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                try:
                    entry = cl.import_config(run_cfg, name=name, overwrite=True)
                except (ConfigLibraryError, OSError) as exc:
                    QMessageBox.critical(self, "Config error", str(exc))
                    return
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            # Record the chosen name on the source folder's config too, so the
            # project view's Config column reflects the name you just gave it.
            try:
                self._stamp_config_name(checked[0], entry.name)
                if self._content_view is not None:
                    self._content_view.refresh()
            except Exception:
                pass
            QMessageBox.information(
                self, "Saved to library",
                f"Saved '{entry.name}' to your Config Library. It will now appear "
                "in the config picker for matching images."
            )
            return

        if clicked != file_btn:
            return  # cancelled

        default_path = os.path.join(
            cl.desktop_dir(), f"{os.path.basename(checked[0])}_run_config.yaml"
        )
        dst, _ = QFileDialog.getSaveFileName(
            self, "Export run config", default_path,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not dst:
            return
        try:
            cl.export_run_config(run_cfg, dst)
        except (ConfigLibraryError, FileNotFoundError, OSError) as exc:
            QMessageBox.critical(self, "Config error", str(exc))
            return
        QMessageBox.information(
            self, "Exported",
            "Exported the full run config (saved_state, dimensions and pipeline "
            f"version{f' {ver_text}' if ver_text else ''} preserved) to:\n\n{dst}"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        reply = QMessageBox.question(
            self, 'Exit', "Exit application?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QApplication.instance().quit()
            event.accept()
        else:
            event.ignore()