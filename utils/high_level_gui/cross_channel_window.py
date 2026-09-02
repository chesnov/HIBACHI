"""cross_channel_window: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
from typing import Optional
import traceback
import yaml  # type: ignore
import numpy as np
import pandas as pd
import tifffile as tiff  # type: ignore
import napari  # type: ignore
from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QMessageBox, QMainWindow, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QWidget, QLabel, QInputDialog, QComboBox
)
from .relational_engine import RelationalEngine

from .metadata import get_sample_metadata



def _safe_name(name: str) -> str:
    """Folder-safe form of a region name, e.g. 'ROI 2' -> 'ROI_2'.

    Region results are written to a subfolder named after the region, so the same
    recipe run on the full image and on a region cannot overwrite each other.
    """
    return "".join(c if (c.isalnum() or c in "-_") else "_"
                   for c in str(name)).strip("_") or "region"


class CrossChannelAnalyzerWindow(QMainWindow):
    def __init__(self, project_manager):
        # Resolved lazily by project_is_2d(); objects are areas in 2D and volumes
        # in 3D, and the UI should say which.
        self._is_2d_cache = None
        super().__init__()
        self.pm = project_manager
        self.setWindowTitle("Cross-Channel Relational Analyzer")
        self.setGeometry(150, 150, 1000, 650)
        
        main_layout = QHBoxLayout()
        
        # --- 1. LEFT PANEL: Channels ---
        left_panel = QVBoxLayout()
        self.channel_list = QListWidget()
        
        # Safely get channels
        if not self.pm.sample_registry:
            self.pm.build_consolidated_sample_registry()
            
        if self.pm.sample_registry:
            first_sample = list(self.pm.sample_registry.keys())[0]
            channels = sorted(list(self.pm.sample_registry[first_sample].keys()))
            for ch in channels:
                item = QListWidgetItem(ch)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.channel_list.addItem(item)
        else:
            item = QListWidgetItem("No channels found")
            self.channel_list.addItem(item)
        
        left_panel.addWidget(QLabel("<b>1. Select Input Channels:</b>"))
        left_panel.addWidget(self.channel_list)
        
        # --- 2. MIDDLE PANEL: Recipe List & Controls ---
        mid_panel = QVBoxLayout()
        mid_panel.addWidget(QLabel("<b>2. Analysis Recipe (Order Matters):</b>"))
        
        recipe_hbox = QHBoxLayout()
        self.recipe_list = QListWidget()
        recipe_hbox.addWidget(self.recipe_list)
        
        step_controls = QVBoxLayout()
        self.btn_remove = QPushButton("❌ Remove")
        self.btn_up = QPushButton("🔼 Up")
        self.btn_down = QPushButton("🔽 Down")
        self.btn_clear = QPushButton("🗑️ Clear All")
        
        self.btn_remove.clicked.connect(self.remove_step)
        self.btn_up.clicked.connect(lambda: self.move_step(-1))
        self.btn_down.clicked.connect(lambda: self.move_step(1))
        self.btn_clear.clicked.connect(self.clear_recipe)
        
        step_controls.addWidget(self.btn_remove)
        step_controls.addWidget(self.btn_up)
        step_controls.addWidget(self.btn_down)
        step_controls.addStretch()
        step_controls.addWidget(self.btn_clear)
        recipe_hbox.addLayout(step_controls)
        
        mid_panel.addLayout(recipe_hbox)
        
        # --- 3. ADD STEP BUTTONS ---
        add_step_layout = QHBoxLayout()
        
        # No longer generates an intensity image: it randomises the segmented
        # masks themselves and exports raw distances for downstream statistics.
        self.btn_synth = QPushButton("🎲 Spatial Null (randomise masks)")
        self.btn_synth.setStyleSheet("background-color: #8A2BE2; color: white;") # Purple
        
        self.btn_intersect = QPushButton("+ Intersection")
        self.btn_filter = QPushButton("+ Size Filter")
        self.btn_dist = QPushButton("+ Distance Analysis")
        
        self.btn_synth.clicked.connect(self.open_synthetic_dialog)
        self.btn_intersect.clicked.connect(self.add_intersect_step)
        self.btn_filter.clicked.connect(self.add_filter_step)
        self.btn_dist.clicked.connect(self.add_analysis_step)
        
        add_step_layout.addWidget(self.btn_synth)
        add_step_layout.addWidget(self.btn_intersect)
        add_step_layout.addWidget(self.btn_filter)
        add_step_layout.addWidget(self.btn_dist)
        mid_panel.addLayout(add_step_layout)

        # --- 4. EXECUTION PANEL ---
        exec_layout = QVBoxLayout()
        
        # Sample Selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Preview Target Sample:"))
        self.sample_selector = QComboBox()
        self.sample_selector.addItems(sorted(list(self.pm.sample_registry.keys())))
        selector_layout.addWidget(self.sample_selector)
        exec_layout.addLayout(selector_layout)

        # Region selector. Only regions that exist in EVERY channel of the sample
        # are offered: the analysis compares one region's mask across channels, so a
        # region missing from one channel cannot be analysed. Rebuilt whenever the
        # sample changes.
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Region:"))
        self.region_selector = QComboBox()
        region_layout.addWidget(self.region_selector)
        exec_layout.addLayout(region_layout)
        self.sample_selector.currentTextChanged.connect(self._reload_regions)
        self._reload_regions(self.sample_selector.currentText())

        self.btn_preview = QPushButton("👁️ Preview Recipe (Napari)")
        self.btn_preview.setFixedHeight(40)
        self.btn_preview.clicked.connect(self.preview_recipe)

        self.btn_batch = QPushButton("🚀 RUN RECIPE ON ALL SAMPLES")
        self.btn_batch.setFixedHeight(50)
        self.btn_batch.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold;")
        self.btn_batch.clicked.connect(self.run_batch_analysis)
        
        exec_layout.addWidget(self.btn_preview)
        exec_layout.addWidget(self.btn_batch)
        mid_panel.addLayout(exec_layout)

        container = QWidget()
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(mid_panel, 3)
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.recipe_steps = []

    # =========================================================================
    # RECIPE EDITING METHODS
    # =========================================================================

    def remove_step(self):
        row = self.recipe_list.currentRow()
        if row >= 0:
            self.recipe_steps.pop(row)
            self.recipe_list.takeItem(row)

    def move_step(self, direction):
        """direction: -1 for up, 1 for down"""
        row = self.recipe_list.currentRow()
        new_row = row + direction
        if 0 <= new_row < self.recipe_list.count():
            # Swap in logic list
            self.recipe_steps[row], self.recipe_steps[new_row] = \
                self.recipe_steps[new_row], self.recipe_steps[row]
            
            # Swap in UI list
            item = self.recipe_list.takeItem(row)
            self.recipe_list.insertItem(new_row, item)
            self.recipe_list.setCurrentRow(new_row)

    def clear_recipe(self):
        reply = QMessageBox.question(self, "Confirm", "Clear entire recipe?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.recipe_steps = []
            self.recipe_list.clear()

    # =========================================================================
    # ADD STEP METHODS
    # =========================================================================

    def get_checked_channels(self):
        return [self.channel_list.item(i).text() for i in range(self.channel_list.count()) 
                if self.channel_list.item(i).checkState() == Qt.Checked]

    def open_synthetic_dialog(self):
        """Launch the spatial-null dialog, informed by the current recipe.

        The recipe matters: an intersection step means the OVERLAP objects are
        what get randomised, and it makes the parent-object domains meaningful,
        so the dialog needs the steps and the selected region -- not just the
        project.

        Catches Exception rather than only ImportError: the previous handler let
        any failure inside the dialog's constructor escape unhandled, which took
        the window down instead of reporting.
        """
        try:
            from ..spatial_null import SpatialNullDialog
            dialog = SpatialNullDialog(
                self.pm,
                checked_channels=self.get_checked_channels(),
                recipe=self.recipe_steps,
                roi_name=self._selected_region(),
                parent=self,
            )
            dialog.exec_()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to open the spatial null dialog:\n{e}")

    def add_intersect_step(self):
        checked = self.get_checked_channels()
        if not (len(checked) == 2 or (len(checked) == 1 and self.recipe_steps)):
            QMessageBox.warning(self, "Error", "Select channels for intersection.")
            return

        # NEW: Ask for Labeling Mode
        modes = {
            "Binary (All overlaps = ID 1)": "binary",
            "Connected Components (Every fragment unique)": "connected",
            "Inherit Parent A (Keep IDs of first channel)": "parent_a",
            "Inherit Parent B (Keep IDs of second channel)": "parent_b"
        }
        
        mode_display, ok = QInputDialog.getItem(
            self, "Intersection Mode", 
            "How should the resulting overlap mask be labeled?", 
            list(modes.keys()), 0, False
        )
        
        if not ok: return
        label_mode = modes[mode_display]

        # For parent modes, ask whether to preserve the original IDs for traceability
        preserve_ids = False
        if label_mode in ("parent_a", "parent_b"):
            parent_label = "A (first channel)" if label_mode == "parent_a" else "B (second channel)"
            id_choice, ok2 = QInputDialog.getItem(
                self, "ID Preservation",
                f"Inherit {parent_label} — should the result mask keep the\n"
                f"original object IDs from that channel?\n\n"
                f"• Keep original IDs — result IDs match the source mask exactly\n"
                f"  (enables direct traceability to the original segmentation).\n"
                f"• Reset to sequential — result is renumbered 1\u2026N as usual.",
                ["Keep original IDs (preserve for traceability)",
                 "Reset to sequential (default)"],
                0, False
            )
            if not ok2: return
            preserve_ids = id_choice.startswith("Keep")

        id_suffix = " [IDs preserved]" if preserve_ids else ""

        if len(checked) == 2:
            step = {
                "type": "intersect", "inputs": checked, "label_mode": label_mode,
                "preserve_ids": preserve_ids,
                "name": f"Overlap ({label_mode}){id_suffix}: {checked[0]} & {checked[1]}"
            }
        else:
            step = {
                "type": "intersect", "inputs": [checked[0], "PREVIOUS_RESULT"], "label_mode": label_mode,
                "preserve_ids": preserve_ids,
                "name": f"Overlap ({label_mode}){id_suffix}: {checked[0]} with previous"
            }
        
        self.recipe_steps.append(step)
        self.recipe_list.addItem(step["name"])

    def project_is_2d(self):
        """True when this project was processed by the 2D pipeline.

        Read from a sample's saved config rather than inferred from array shapes:
        a 3-axis array can be (Z, Y, X) or (C, Y, X), so the recorded mode is the
        only reliable discriminator. Cached, because it cannot change within a
        project and the lookup touches the disk.
        """
        if getattr(self, "_is_2d_cache", None) is not None:
            return self._is_2d_cache
        is_2d = None
        for sample_data in self.pm.sample_registry.values():
            for ch_path in sample_data.values():
                try:
                    _, mode = get_sample_metadata(ch_path)
                except Exception:
                    continue
                if mode:
                    is_2d = str(mode).endswith("_2d")
                    break
            if is_2d is not None:
                break
        self._is_2d_cache = bool(is_2d) if is_2d is not None else False
        return self._is_2d_cache

    def size_unit(self):
        """'um²' in 2D, 'um³' in 3D — objects are areas or volumes, not both."""
        return "um\u00b2" if self.project_is_2d() else "um\u00b3"

    def size_word(self):
        return "Area" if self.project_is_2d() else "Volume"

    def add_filter_step(self):
        """Add a size filter, recording WHAT it filters.

        A filter with a preceding mask-producing step applies to that result. With
        nothing before it, it needs a channel, and that channel has to be recorded
        in the step -- it cannot be inferred later from the checked boxes, because
        `get_checked_channels` returns list order rather than click order, so
        "the first checked channel" is not "the one the user meant". Leaving it
        unrecorded also made the step a silent no-op in `run_recipe`, which only
        acted when a previous result existed.
        """
        has_previous = any(st.get("type") in ("intersect", "filter")
                           for st in self.recipe_steps)

        source = None
        if not has_previous:
            checked = self.get_checked_channels()
            if len(checked) == 1:
                source = checked[0]
            else:
                options = self.get_checked_channels() or sorted(
                    {c for d in self.pm.sample_registry.values() for c in d})
                if not options:
                    QMessageBox.warning(self, "No channels",
                                        "No channels are available to filter.")
                    return
                pick, ok = QInputDialog.getItem(
                    self, "Size Filter",
                    "This filter has nothing before it in the recipe, so it needs "
                    "a channel to filter.\n\nFilter which channel?",
                    options, 0, False)
                if not ok:
                    return
                source = pick

        unit = self.size_unit()
        val, ok = QInputDialog.getDouble(
            self, "Size Filter", f"Minimum {self.size_word().lower()} ({unit}):",
            10.0, 0, 1000000, 2)
        if not ok:
            return

        target = (source.split("_", 2)[-1] if source else "previous result")
        # 'min_vol' is kept as the key for backward compatibility with recipes
        # already saved to disk; 'input' and 'size_unit' are additions.
        step = {"type": "filter", "min_vol": val, "size_unit": unit,
                "input": source,
                "name": f"Size filter ({target}): keep objects > {val:g} {unit}"}
        self.recipe_steps.append(step)
        self.recipe_list.addItem(step["name"])

    def add_analysis_step(self):
        checked = self.get_checked_channels()
        if not checked:
            QMessageBox.warning(self, "Error", "Check at least one channel for distance analysis.")
            return

        # Determine whether there is an accumulated pipeline result (intersection / filter)
        # that could act as one side of the analysis.
        has_previous_result = any(
            s['type'] in ('intersect', 'filter') for s in self.recipe_steps
        )

        for ch in checked:
            if has_previous_result:
                # --- Case A: previous accumulated result exists ---
                # Ask which role the checked channel plays.
                role, ok = QInputDialog.getItem(
                    self,
                    "Select Primary Channel",
                    "Which side should be the PRIMARY (objects distances are reported FOR)?",
                    [
                        f"{ch}  →  primary  (measure FROM {ch} TO the previous result)",
                        f"Previous result  →  primary  (measure FROM previous result TO {ch})",
                    ],
                    0, False
                )
                if not ok:
                    return

                if role.startswith(ch):
                    # ch is primary, previous result is the partner
                    step = {
                        "type":    "analyze",
                        "primary": ch,
                        "target":  "PREVIOUS_RESULT",
                        "name":    f"Analyze {ch} → distance to previous result",
                    }
                else:
                    # previous result is primary, ch is the partner
                    step = {
                        "type":   "analyze",
                        "target": ch,
                        "name":   f"Analyze previous result → distance to {ch}",
                    }

            else:
                # --- Case B: simple two-channel analysis, no prior pipeline result ---
                # The second channel must be chosen from the channel list.
                other_channels = [
                    self.channel_list.item(i).text()
                    for i in range(self.channel_list.count())
                    if self.channel_list.item(i).text() != ch
                ]
                if not other_channels:
                    QMessageBox.warning(self, "Error", "Need at least two channels for distance analysis.")
                    return

                partner, ok = QInputDialog.getItem(
                    self,
                    "Select Partner Channel",
                    f"Measure distance FROM  '{ch}'  TO which channel?",
                    other_channels, 0, False
                )
                if not ok:
                    return

                # Ask which of the two is primary
                role, ok = QInputDialog.getItem(
                    self,
                    "Select Primary Channel",
                    "Which side should be the PRIMARY (objects distances are reported FOR)?",
                    [
                        f"{ch}  →  primary  (measure FROM {ch} TO {partner})",
                        f"{partner}  →  primary  (measure FROM {partner} TO {ch})",
                    ],
                    0, False
                )
                if not ok:
                    return

                if role.startswith(ch):
                    primary_ch, partner_ch = ch, partner
                else:
                    primary_ch, partner_ch = partner, ch

                step = {
                    "type":    "analyze",
                    "primary": primary_ch,
                    "target":  partner_ch,
                    "name":    f"Analyze {primary_ch} → distance to {partner_ch}",
                }

            self.recipe_steps.append(step)
            self.recipe_list.addItem(step["name"])

    FULL_IMAGE_LABEL = "Full image"

    def _reload_regions(self, sample_name: str = "") -> None:
        """Repopulate the region picker for the selected sample."""
        self.region_selector.clear()
        self.region_selector.addItem(self.FULL_IMAGE_LABEL)
        try:
            channels = list(
                (self.pm.sample_registry.get(sample_name) or {}).values())
            from .roi_sharing import regions_common_to_channels
            for name in regions_common_to_channels(channels,
                                                   require_segmentation=True):
                self.region_selector.addItem(name)
        except Exception as exc:
            print(f"  [Relational] could not list regions: {exc}")

    def _selected_region(self):
        """Region name chosen in the picker, or None for the full image."""
        try:
            label = self.region_selector.currentText()
        except Exception:
            return None
        return None if (not label or label == self.FULL_IMAGE_LABEL) else label

    def _geometry_for(self, sample_data: dict, roi_name):
        """(shape, spacing) for a sample, honouring the selected region.

        A region's masks are the CROP, so its shape and spacing must come from the
        region rather than the channel's full-resolution TIFF -- memmapping a
        region's mask against the full shape would read past the end of the file.
        Returns (None, None) when it cannot be determined.
        """
        first_ch = list(sample_data.values())[0]
        if roi_name:
            from .roi_sharing import region_geometry
            geo = region_geometry(first_ch, roi_name)
            if geo is None:
                return None, None
            return geo["shape"], geo["spacing"]

        tif_path = next((os.path.join(first_ch, f) for f in os.listdir(first_ch)
                         if f.lower().endswith((".tif", ".tiff"))), None)
        if not tif_path:
            return None, None
        with tiff.TiffFile(tif_path) as tif:
            shape = tuple(int(s) for s in tif.series[0].shape)
        meta, mode = get_sample_metadata(first_ch)
        meta = meta or {}
        # A (C, Z, Y, X) or (C, Y, X) file would otherwise hand back a shape the
        # .dat memmaps do not have. The config's mode says how many trailing
        # axes are spatial, which is the only reliable discriminator: a 3-axis
        # array may be (Z, Y, X) or (C, Y, X). Keep exactly the `want` trailing
        # spatial axes and drop any leading (channel) axes. Do NOT squeeze
        # singleton axes generally: a genuine thin-Z volume is (1, Y, X) and must
        # stay 3D, or the 3D viewport and turntable would be lost for
        # single-channel / few-slice samples.
        want = 2 if str(mode or "").endswith("_2d") else 3
        if len(shape) > want:
            shape = shape[-want:]
        if len(shape) == 3:
            spacing = (meta.get('z', 1.0) / shape[0],
                       meta.get('y', 1.0) / shape[1],
                       meta.get('x', 1.0) / shape[2])
        else:
            spacing = (meta.get('y', 1.0) / shape[0],
                       meta.get('x', 1.0) / shape[1])
        return shape, spacing

    def preview_recipe(self):
        if not self.recipe_steps:
            QMessageBox.information(self, "Info", "Recipe is empty.")
            return

        # 1. Ask for a name to make this a "Single Run"
        analysis_name, ok = QInputDialog.getText(
            self, "Single Sample Run", 
            "Enter a name for this analysis (files will be saved):",
            text="Preview_Run"
        )
        if not ok or not analysis_name: return

        sample_name = self.sample_selector.currentText()
        sample_data = self.pm.sample_registry[sample_name]
        roi_name = self._selected_region()

        # 2. Setup Viewer
        # Setup permanent path for this sample's relational results. A region's
        # results go in their own subfolder so running the same recipe on the full
        # image and on a region cannot overwrite one another.
        project_root = os.path.dirname(self.pm.project_path)
        leaf = sample_name if not roi_name else os.path.join(
            sample_name, _safe_name(roi_name))
        sample_out_dir = os.path.join(
            project_root, "RELATIONAL_ANALYSIS", analysis_name, leaf
        )
        os.makedirs(sample_out_dir, exist_ok=True)
        # Self-describing results, matching the batch path.
        try:
            with open(os.path.join(project_root, "RELATIONAL_ANALYSIS",
                                   analysis_name, "region.txt"), "w") as fh:
                fh.write((roi_name or "Full image") + "\n")
        except OSError:
            pass

        viewer = napari.Viewer(title=f"Cross-Channel Preview: {sample_name}")
        # Any failure below leaves a live napari window (Qt owns it, so it is not
        # collected when this frame unwinds), holding a GL context and the loaded
        # image for the rest of the session. Close it instead.
        #
        # Imported lazily: a module-level import would close the cycle
        # cross_channel_window -> app_launch -> project_view_window -> here.
        from .app_launch import close_viewer_on_error
        with close_viewer_on_error(viewer):
            try:
                _qw = viewer.window._qt_window
                _qw.showMaximized(); _qw.raise_(); _qw.activateWindow()
            except Exception:
                pass
        
            # Predefined colormaps for raw channels
            colormaps = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
        
            # 3. Load Raw Data and Segmentation for EVERY channel in this sample
            shape = None
            spacing = (1.0, 1.0, 1.0)
        
            for i, (ch_name, ch_path) in enumerate(sample_data.items()):
                # Find files
                tif_file = next((os.path.join(ch_path, f) for f in os.listdir(ch_path) if f.lower().endswith(('.tif', '.tiff'))), None)
                dat_file = RelationalEngine._find_dat(ch_path)
            
                # Fetch metadata from the first valid channel we find
                if shape is None and tif_file:
                    with tiff.TiffFile(tif_file) as tif:
                        shape = tif.series[0].shape
                    # Try to get spacing from strategy config
                    meta, _ = get_sample_metadata(ch_path)
                    if meta:
                        # Very simple spacing calc: total_um / pixels
                        # (Note: In a production version, we use the exact strategy spacing)
                        spacing = (meta.get('z', 1.0)/shape[0], meta.get('y', 1.0)/shape[1], meta.get('x', 1.0)/shape[2]) if len(shape)==3 else (meta.get('y', 1.0)/shape[0], meta.get('x', 1.0)/shape[1])

                # Add Raw Intensity
                if tif_file:
                    raw_img = tiff.imread(tif_file)
                    cmap = colormaps[i % len(colormaps)]
                    viewer.add_image(raw_img, name=f"Raw: {ch_name}", colormap=cmap, blending='additive', opacity=0.5)

                # Add Segmentation Labels (Semi-transparent)
                if dat_file:
                    seg_data = np.memmap(dat_file, dtype=np.int32, mode='r', shape=shape)
                    viewer.add_labels(seg_data, name=f"Seg: {ch_name}", opacity=0.3)

            # 4. Execute Relational Recipe
            temp_dir = os.path.join(list(sample_data.values())[0], "relational_preview_temp")
            os.makedirs(temp_dir, exist_ok=True)
        
            # Run calculation
            derived_masks, metrics_df = RelationalEngine.run_recipe(
                sample_name, self.pm.sample_registry, self.recipe_steps,
                sample_out_dir, shape, spacing, roi_name=roi_name
            )

            # Add the Red Proximity Lines
            if metrics_df is not None:
                self._draw_proximity_bridges(viewer, metrics_df, shape, spacing)

            # 5. Add Derived Results (High Opacity Labels)
            for res in derived_masks:
                data = np.memmap(res['path'], dtype=np.int32, mode='r', shape=shape)
                viewer.add_labels(data, name=f"DERIVED: {res['name']}")

            # Final Adjustments
            if len(shape) == 3:
                viewer.dims.ndisplay = 3
                # Set scale to handle anisotropy if 3D
                # (Napari scale is z_scale, y_scale, x_scale)
                # Use spacing[0]/spacing[2] for z-scale factor
                z_scale = spacing[0]/spacing[2] if len(spacing)==3 else 1.0
                for layer in viewer.layers:
                    layer.scale = (z_scale, 1, 1)

            # One-click hide/show-all toggle under the layer list.
            try:
                from .app_launch import add_channel_visibility_toggle
                add_channel_visibility_toggle(viewer)
            except Exception as exc:
                print(f"Could not add channel visibility toggle: {exc}")

            # 3D rotation recorder (3D samples only), docked beneath the layer list.
            if shape and len(shape) == 3:
                try:
                    from ..fluorescence_module.turntable import add_turntable_button
                    add_turntable_button(viewer)
                except Exception as exc:
                    print(f"Could not add 3D rotation recorder: {exc}")

    def _draw_proximity_bridges(self, viewer, df, shape, spacing):
        """Delegate to the module-level bridge drawer (used by preview_recipe)."""
        draw_proximity_bridges(viewer, df, shape, spacing)

    def run_batch_analysis(self):
        if not self.recipe_steps:
            return

        # 1. Ask for an Analysis Name (to create a subfolder)
        roi_name = self._selected_region()
        analysis_name, ok = QInputDialog.getText(self, "Batch Run", "Enter name for this analysis:")
        if not ok or not analysis_name:
            return

        # 2. Setup root results folder
        project_root = os.path.dirname(self.pm.project_path)
        batch_out_dir = os.path.join(project_root, "RELATIONAL_ANALYSIS", analysis_name)
        os.makedirs(batch_out_dir, exist_ok=True)

        # 3. Save the recipe itself for reproducibility
        # Record which region the run covered, so a results folder is
        # self-describing rather than depending on someone remembering.
        try:
            with open(os.path.join(batch_out_dir, "region.txt"), 'w') as f:
                f.write((roi_name or "Full image") + "\n")
        except OSError:
            pass
        with open(os.path.join(batch_out_dir, "recipe.yaml"), 'w') as f:
            yaml.dump(self.recipe_steps, f)

        # 4. Process all samples
        samples = sorted(list(self.pm.sample_registry.keys()))
        total = len(samples)
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        print(f"\n{'='*60}\nSTARTING RELATIONAL BATCH: {analysis_name}\n{'='*60}")
        
        try:
            for i, s_name in enumerate(samples):
                print(f"Processing {i+1}/{total}: {s_name}...")
                sample_data = self.pm.sample_registry[s_name]
                
                # Shape/spacing for this sample, from the region when one is
                # selected. Samples lacking that region are skipped with a note
                # rather than analysed against the wrong geometry.
                shape, spacing = self._geometry_for(sample_data, roi_name)
                if shape is None:
                    print(f"  [Skip] {s_name}: "
                          + (f"region {roi_name!r} not available in every channel"
                             if roi_name else "no readable image"))
                    continue

                # Sample-specific output folder, kept separate per region.
                sample_out = os.path.join(batch_out_dir, s_name) if not roi_name \
                    else os.path.join(batch_out_dir, s_name, _safe_name(roi_name))

                # EXECUTE ENGINE
                RelationalEngine.run_recipe(
                    s_name, self.pm.sample_registry, self.recipe_steps,
                    sample_out, shape, spacing, roi_name=roi_name
                )

            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Success", f"Batch Complete!\nResults saved to: {batch_out_dir}")
            print(f"\n{'='*60}\nBATCH FINISHED\n{'='*60}")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            print(f"FATAL ERROR IN BATCH: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Batch Error", str(e))
            # Without this the summary block below still runs and can report
            # "Successfully generated" for a batch that died.
            return

        # 5. Create Master Summary Table
        all_csvs = []
        for s_name in samples:
            leaf_dir = os.path.join(batch_out_dir, s_name) if not roi_name \
                else os.path.join(batch_out_dir, s_name, _safe_name(roi_name))
            # run_recipe writes "_relational_metrics.csv"; nothing has ever
            # written "_relational_results.csv", so this list was always empty
            # and MASTER_RELATIONAL_RESULTS.csv was never produced.
            csv_p = os.path.join(leaf_dir, f"{s_name}_relational_metrics.csv")
            if os.path.exists(csv_p):
                df = pd.read_csv(csv_p)
                df['sample_name'] = s_name
                all_csvs.append(df)
        
        if all_csvs:
            master_df = pd.concat(all_csvs, ignore_index=True)
            master_df.to_csv(os.path.join(batch_out_dir, "MASTER_RELATIONAL_RESULTS.csv"), index=False)
            print("Successfully generated MASTER_RELATIONAL_RESULTS.csv")


# ============================================================================
# Reusable cross-channel overlay helpers (module-level, callable from the main
# project window as well as this analyzer).
# ============================================================================

# Separates an analysis name from a region name inside it, matching the
# convention used for slide scenes and tree leaves.
ANALYSIS_SEP = "::"


def make_analysis_key(analysis: str, region_dir: Optional[str] = None) -> str:
    """Identity for one overlay choice: an analysis, optionally scoped to a region."""
    return f"{analysis}{ANALYSIS_SEP}{region_dir}" if region_dir else analysis


def split_analysis_key(key: str):
    """(analysis, region_dir or None) for an overlay key."""
    text = str(key)
    if ANALYSIS_SEP in text:
        analysis, _, region = text.partition(ANALYSIS_SEP)
        return analysis, (region or None)
    return text, None


def _has_result_files(directory: str) -> bool:
    """True if a directory holds relational outputs directly inside it."""
    try:
        return any(f.endswith(".dat") or f.endswith(".csv")
                   for f in os.listdir(directory))
    except OSError:
        return False


def list_relational_analyses(project_root: str):
    """Saved cross-channel analyses under <project_root>/RELATIONAL_ANALYSIS.

    Returns (display, key) pairs. An analysis run on a saved region writes to
    ``<analysis>/<sample>/<region>/`` rather than ``<analysis>/<sample>/``, so one
    analysis name can hold several variants -- the full image, and one per region.
    Listing only the top-level name meant selecting it found nothing at the level
    the overlay looks, or worse, silently showed the FULL-IMAGE result when a
    region was intended. Each variant is therefore its own entry.

    Detected from the directory layout rather than from a recorded label, so an
    analysis name used for both a full image and a region reports both.
    """
    rel_dir = os.path.join(project_root, "RELATIONAL_ANALYSIS")
    if not os.path.isdir(rel_dir):
        return []
    try:
        analyses = sorted(d for d in os.listdir(rel_dir)
                          if os.path.isdir(os.path.join(rel_dir, d)))
    except OSError:
        return []

    out = []
    for analysis in analyses:
        a_dir = os.path.join(rel_dir, analysis)
        has_full = False
        regions = set()
        try:
            samples = [s for s in os.listdir(a_dir)
                       if os.path.isdir(os.path.join(a_dir, s))]
        except OSError:
            samples = []
        for sample in samples:
            s_dir = os.path.join(a_dir, sample)
            if _has_result_files(s_dir):
                has_full = True
            try:
                for sub in os.listdir(s_dir):
                    sub_path = os.path.join(s_dir, sub)
                    if os.path.isdir(sub_path) and _has_result_files(sub_path):
                        regions.add(sub)
            except OSError:
                pass

        # An analysis with no detectable outputs still gets a plain entry, so a
        # run that failed part-way is visible rather than vanishing.
        if has_full or not regions:
            out.append((analysis, make_analysis_key(analysis)))
        for region in sorted(regions):
            # Folder names are slugified ("ROI_2"); show them as drawn ("ROI 2").
            out.append((f"{analysis}  \u203a  {region.replace('_', ' ')}",
                        make_analysis_key(analysis, region)))
    return out


def draw_proximity_bridges(viewer, df, shape, spacing):
    """
    Parse a metrics dataframe for Source/Target coordinates and draw red
    connection lines (bridges) between interacting biological objects.
    """
    partners = [c.replace('src_y_', '') for c in df.columns if c.startswith('src_y_')]
    is_3d = (len(shape) == 3)
    z_scale = spacing[0] / spacing[-1] if is_3d else 1.0
    display_scale = (z_scale, 1, 1) if is_3d else (1, 1)

    for p in partners:
        lines = []
        for _, row in df.iterrows():
            if pd.notna(row.get(f'dist_um_{p}')):
                try:
                    if is_3d:
                        src = [row[f'src_z_{p}'], row[f'src_y_{p}'], row[f'src_x_{p}']]
                        tgt = [row[f'tgt_z_{p}'], row[f'tgt_y_{p}'], row[f'tgt_x_{p}']]
                    else:
                        src = [row[f'src_y_{p}'], row[f'src_x_{p}']]
                        tgt = [row[f'tgt_y_{p}'], row[f'tgt_x_{p}']]
                    lines.append([src, tgt])
                except KeyError:
                    continue
        if lines:
            viewer.add_shapes(
                lines, shape_type='line', edge_color='red',
                edge_width=2 if not is_3d else 1,
                name=f"Bridges to {p}", scale=display_scale, blending='additive',
            )
    print(f"  [Visualizer] Plotted connection bridges for partners: {partners}")


def open_sample_overlay(project_manager, sample_name, analysis_name=None, parent=None):
    """
    Open a napari viewer for one multi-channel sample.

    Always loads every channel's raw intensity (visible) and its base
    segmentation (added but hidden, so the viewer isn't a mess of overlapping
    label layers — the user can toggle any on). When `analysis_name` is given,
    the cross-channel-specific layers are added on top and shown: the derived
    masks (.dat) from that analysis and the proximity bridges from its metrics.

    `sample_name` is the consolidated-registry key (the clean sample name), which
    is also the analysis output subfolder name. Returns True if a viewer opened.
    """
    pm = project_manager
    if not pm.sample_registry:
        pm.build_consolidated_sample_registry()

    sample_data = pm.sample_registry.get(sample_name)
    if not sample_data:
        QMessageBox.warning(
            parent, "Not Found", f"No channels found for sample '{sample_name}'."
        )
        return False

    sample_out_dir = None
    analysis_label = analysis_name
    region_dir = None
    if analysis_name:
        # The picker hands over a key, which may name a region inside the analysis.
        analysis_name, region_dir = split_analysis_key(analysis_name)
        project_root = os.path.dirname(pm.project_path)
        sample_out_dir = os.path.join(
            project_root, "RELATIONAL_ANALYSIS", analysis_name, sample_name
        )
        if region_dir:
            sample_out_dir = os.path.join(sample_out_dir, region_dir)
        analysis_label = (f"{analysis_name} \u203a {region_dir.replace('_', ' ')}"
                          if region_dir else analysis_name)
        if not os.path.isdir(sample_out_dir):
            QMessageBox.warning(
                parent, "Not Found",
                f"No data found for sample '{sample_name}' in "
                f"'{analysis_label}'."
                + ("\n\nThis sample may not have that region." if region_dir else "")
            )
            return False

    title = (f"Overlay: {analysis_label} | {sample_name}"
             if analysis_name else f"Sample: {sample_name}")
    viewer = napari.Viewer(title=title)
    # Any failure below leaves a live napari window (Qt owns it, so it is not
    # collected when this frame unwinds), holding a GL context and the loaded
    # image for the rest of the session. Close it instead.
    #
    # Imported lazily: a module-level import would close the cycle
    # cross_channel_window -> app_launch -> project_view_window -> here.
    from .app_launch import close_viewer_on_error
    with close_viewer_on_error(viewer):
        try:
            _qw = viewer.window._qt_window
            _qw.showMaximized(); _qw.raise_(); _qw.activateWindow()
        except Exception:
            pass
        colormaps = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
        shape = None
        spacing = (1.0, 1.0, 1.0)

        # 1. Raw intensity (visible) + base segmentation (hidden, toggle-able).
        for i, (ch_name, ch_path) in enumerate(sample_data.items()):
            tif_file = next((os.path.join(ch_path, f) for f in os.listdir(ch_path)
                             if f.lower().endswith(('.tif', '.tiff'))), None)
            dat_file = RelationalEngine._find_dat(ch_path)

            if shape is None and tif_file:
                with tiff.TiffFile(tif_file) as tif:
                    shape = tif.series[0].shape
                meta, _ = get_sample_metadata(ch_path)
                if meta:
                    spacing = (
                        (meta.get('z', 1.0)/shape[0], meta.get('y', 1.0)/shape[1], meta.get('x', 1.0)/shape[2])
                        if len(shape) == 3 else
                        (meta.get('y', 1.0)/shape[0], meta.get('x', 1.0)/shape[1])
                    )

            if tif_file:
                raw_img = tiff.imread(tif_file)
                cmap = colormaps[i % len(colormaps)]
                viewer.add_image(raw_img, name=f"Raw: {ch_name}", colormap=cmap,
                                 blending='additive', opacity=0.5)
            if dat_file:
                seg_data = np.memmap(dat_file, dtype=np.int32, mode='r', shape=shape)
                viewer.add_labels(seg_data, name=f"Seg: {ch_name}", opacity=0.3,
                                  visible=False)

        # 2. Cross-channel-specific layers (only for a selected analysis; shown).
        if sample_out_dir:
            for f in [x for x in os.listdir(sample_out_dir) if x.endswith('.dat')]:
                try:
                    data = np.memmap(os.path.join(sample_out_dir, f), dtype=np.int32, mode='r', shape=shape)
                    viewer.add_labels(data, name=f"DERIVED: {f.replace('.dat', '')}")
                except Exception as e:
                    print(f"Could not load {f}: {e}")

            csv_path = os.path.join(sample_out_dir, f"{sample_name}_relational_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    draw_proximity_bridges(viewer, pd.read_csv(csv_path), shape, spacing)
                except Exception as e:
                    print(f"Could not draw bridges: {e}")

        # 3. Viewport for 3D.
        if shape and len(shape) == 3:
            viewer.dims.ndisplay = 3
            z_scale = spacing[0]/spacing[2] if len(spacing) == 3 else 1.0
            for layer in viewer.layers:
                layer.scale = (z_scale, 1, 1)

        # One-click hide/show-all toggle under the layer list (shared with the
        # per-channel segmenter view). Lazy import avoids a circular import.
        try:
            from .app_launch import add_channel_visibility_toggle
            add_channel_visibility_toggle(viewer)
        except Exception as exc:
            print(f"Could not add channel visibility toggle: {exc}")

        # 3D rotation recorder (3D samples only), docked beneath the layer list.
        if shape and len(shape) == 3:
            try:
                from ..fluorescence_module.turntable import add_turntable_button
                add_turntable_button(viewer)
            except Exception as exc:
                print(f"Could not add 3D rotation recorder: {exc}")

        # Shared sub-region controls. Every channel of a sample has the same pixel
        # dimensions, so a polygon drawn here is valid in all of them -- this is the
        # one place an ROI can be defined once and handed to several channels.
        # Requires `shape`, which is the coordinate frame the polygon is stored in.
        if shape:
            try:
                from .roi_overlay_panel import add_overlay_roi_panel
                add_overlay_roi_panel(
                    viewer, sample_name, list(sample_data.values()), shape
                )
            except Exception as exc:
                print(f"Could not add shared ROI panel: {exc}")

        return True