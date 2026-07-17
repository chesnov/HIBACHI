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
        self.initUI()
        self.setAttribute(Qt.WA_QuitOnClose)

    def initUI(self) -> None:
        self.setWindowTitle("Image Segmentation Project")
        _icon = app_icon_path()
        if _icon:
            self.setWindowIcon(QIcon(_icon))
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
        self.hide()
        from .app_launch import interactive_segmentation_with_config  # lazy: avoid cycle
        interactive_segmentation_with_config(folder, project_manager=self.project_manager)

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
        super().changeEvent(event)

    def _process_selected(self) -> None:
        if self._content_view is None:
            return
        self._batch_process_folders(self._content_view.checked_folders())

    def _batch_process_folders(self, folders: list) -> None:
        """Route the checked set of image/channel folders to batch processing."""
        if not folders:
            return
        if not BatchProcessor:
            QMessageBox.warning(self, "Unavailable", "Batch processor is not available.")
            return
        reply = QMessageBox.question(
            self, "Confirm",
            f"Process {len(folders)} selected image folder"
            f"{'s' if len(folders) != 1 else ''}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # BatchProcessor iterates project_manager.image_folders, so point it at
        # exactly the checked set (which may span several channels), then restore.
        saved = self.project_manager.image_folders
        self.project_manager.image_folders = list(folders)
        try:
            processor = BatchProcessor(self.project_manager)
            processor.process_all_folders(force_restart_all=False)
        finally:
            self.project_manager.image_folders = saved
        QMessageBox.information(self, "Done", "Batch processing complete.")
        if self._content_view is not None:
            self._content_view.refresh()  # refresh status badges / last-edited
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
            raw_files = [
                f for f in os.listdir(selected_path)
                if f.lower().endswith(('.tif', '.tiff', '.czi')) and
                os.path.isfile(os.path.join(selected_path, f))
            ]
            if raw_files:
                from .organize_wizard import run_organize_wizard
                if run_organize_wizard(self, selected_path, mode="new",
                                       project_dir=selected_path):
                    # After setup the folder is a single- or multi-channel project;
                    # re-route so it opens in the right view (list or tree).
                    self.open_path(selected_path)
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

        template_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config Template", "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
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
            f"• Existing computed state (e.g. thresholds) is preserved.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        saved = self.project_manager.image_folders
        self.project_manager.image_folders = list(folders)
        try:
            results = apply_template_config_to_project(
                template_path, self.project_manager
            )
        except Exception as exc:
            self.project_manager.image_folders = saved
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Config application failed:\n{exc}")
            return
        finally:
            self.project_manager.image_folders = saved
        QApplication.restoreOverrideCursor()

        summary = (
            f"Config template applied.\n\n"
            f"Updated : {results['success']}\n"
            f"Skipped : {results['skipped']}  (different mode or invalid)\n"
            f"Failed  : {results['failed']}\n"
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