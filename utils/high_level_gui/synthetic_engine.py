import os
import yaml
import shutil
import numpy as np
import pandas as pd
import tifffile as tiff
import scipy.ndimage as ndi
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QLineEdit, QPushButton, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt

class SyntheticDataDialog(QDialog):
    def __init__(self, project_manager, parent=None):
        super().__init__(parent)
        self.pm = project_manager
        self.setWindowTitle("Generate Procedural Synthetic Data")
        self.setMinimumWidth(450)
        
        self.project_root = os.path.dirname(self.pm.project_path)
        self.rel_dir = os.path.join(self.project_root, "RELATIONAL_ANALYSIS")
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # 1. Template Channel Selection
        layout.addWidget(QLabel("1. Select Template Channel (Source of stats):"))
        self.cb_channel = QComboBox()
        # Get unique channels from the sample registry
        channels = set()
        for sample, ch_dict in self.pm.sample_registry.items():
            channels.update(ch_dict.keys())
        self.cb_channel.addItems(sorted(list(channels)))
        layout.addWidget(self.cb_channel)
        
        # 2. Relational Filter (Optional)
        layout.addWidget(QLabel("2. Filter by Relational Analysis (Optional):"))
        layout.addWidget(QLabel("<i>(Only generates objects that matched this condition)</i>"))
        self.cb_filter = QComboBox()
        self.cb_filter.addItem("None (Use all objects in channel)")
        if os.path.exists(self.rel_dir):
            runs =[d for d in os.listdir(self.rel_dir) if os.path.isdir(os.path.join(self.rel_dir, d))]
            self.cb_filter.addItems(runs)
        layout.addWidget(self.cb_filter)
        
        # 3. Output Name
        layout.addWidget(QLabel("3. New Channel Name:"))
        self.le_output = QLineEdit("Synthetic_Data")
        layout.addWidget(self.le_output)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_run = QPushButton("Generate")
        btn_run.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold;")
        btn_run.clicked.connect(self.run_generation)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_run)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)

    def run_generation(self):
        template_ch = self.cb_channel.currentText()
        rel_filter = self.cb_filter.currentText()
        if rel_filter.startswith("None"):
            rel_filter = None
            
        out_name = self.le_output.text().strip()
        if not out_name:
            QMessageBox.warning(self, "Error", "Please provide an output name.")
            return
            
        # Determine highest channel number to format "Channel_X_Name"
        existing_channels = [d for d in os.listdir(self.project_root) if d.startswith("Channel_")]
        next_idx = max([int(c.split('_')[1]) for c in existing_channels if c.split('_')[1].isdigit()] + [-1]) + 1
        out_channel_dir = os.path.join(self.project_root, f"Channel_{next_idx}_{out_name}")
        os.makedirs(out_channel_dir, exist_ok=True)
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            generate_synthetic_channel(self.pm, template_ch, rel_filter, out_channel_dir)
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Success", f"Synthetic data generated!\nSaved to: {out_channel_dir}\n\nIt will appear in the project view automatically when you return to it.")
            self.accept()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Generation failed:\n{e}")

def _draw_line_nd(mask, start, end):
    """Draws an N-dimensional line on a boolean mask."""
    start, end = np.array(start), np.array(end)
    max_dist = int(np.max(np.abs(end - start))) + 1
    if max_dist == 0:
        mask[tuple(start)] = 1
        return
    for i in range(max_dist):
        t = i / max_dist
        pt = tuple(np.round(start + t * (end - start)).astype(int))
        mask[pt] = 1

def _generate_procedural_object(shape, pixel_count, branches, skel_pixels):
    """Generates a local boolean mask of a procedurally drawn biological object."""
    ndim = len(shape)
    
    radius = int(np.ceil((pixel_count / np.pi) ** (1/ndim)))
    branch_len = int(skel_pixels / max(1, branches)) if branches > 0 else 0
    grid_size = max(10, branch_len * 3, radius * 3)
    local_mask = np.zeros((grid_size,) * ndim, dtype=bool)
    center = np.array([grid_size // 2] * ndim)
    
    if branches > 0 and skel_pixels > 0:
        # 1. Draw Cell Body (Soma) - assume ~30% of total volume
        soma_vol = pixel_count * 0.3
        soma_rad = max(1, int(np.ceil((soma_vol / np.pi) ** (1/ndim))))
        
        grid_coords = np.indices((grid_size,) * ndim)
        dist_from_center = np.sqrt(np.sum((grid_coords - center.reshape((-1,) + (1,)*ndim))**2, axis=0))
        local_mask[dist_from_center <= soma_rad] = True
        
        # 2. Draw Branches
        for _ in range(int(branches)):
            dir_vec = np.random.randn(ndim)
            if np.linalg.norm(dir_vec) == 0: continue
            dir_vec /= np.linalg.norm(dir_vec)
            
            # Randomize branch length a bit for a natural, non-perfect look
            actual_len = branch_len * np.random.uniform(0.5, 1.5)
            end_pt = center + dir_vec * actual_len
            end_pt = np.clip(end_pt, 0, grid_size - 1)
            _draw_line_nd(local_mask, center, end_pt)
            
        # 3. Dilate carefully (Cap iterations so branches don't merge into a circle)
        current_vol = np.sum(local_mask)
        iterations = 0
        max_dilations = max(1, int(soma_rad * 0.5)) 
        while current_vol < pixel_count and iterations < max_dilations:
            local_mask = ndi.binary_dilation(local_mask)
            current_vol = np.sum(local_mask)
            iterations += 1
            
    else:
        # It's a Blob (Aggregate)
        local_mask[tuple(center)] = 1
        current_vol = 1
        iterations = 0
        while current_vol < pixel_count and iterations < (grid_size // 2):
            local_mask = ndi.binary_dilation(local_mask)
            current_vol = np.sum(local_mask)
            iterations += 1

    # Fast crop to bounding box to speed up merging
    coords = np.argwhere(local_mask)
    if len(coords) == 0: return local_mask
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    slices = tuple(slice(mins[d], maxs[d]) for d in range(ndim))
    
    return local_mask[slices]

def _add_local_mask(main_img, local_mask, center_coords, intensity):
    """Safely adds a local mask into an N-Dimensional image, handling boundary clipping."""
    slices_main, slices_local = [],[]
    for dim, c in enumerate(center_coords):
        size = local_mask.shape[dim]
        start = c - (size // 2)
        end = start + size
        
        m_start = max(0, start)
        m_end = min(main_img.shape[dim], end)
        if m_start >= m_end: return # Out of bounds
            
        slices_main.append(slice(m_start, m_end))
        l_start = m_start - start
        slices_local.append(slice(l_start, l_start + (m_end - m_start)))
        
    main_img[tuple(slices_main)] += (local_mask[tuple(slices_local)] * intensity)

def generate_synthetic_channel(pm, template_ch, rel_filter, out_dir):
    """Core engine to generate synthetic channels based on extracted stats."""
    rel_dir = os.path.join(os.path.dirname(pm.project_path), "RELATIONAL_ANALYSIS")
    
    for sample_name, ch_dict in pm.sample_registry.items():
        if template_ch not in ch_dict:
            continue
            
        ch_path = ch_dict[template_ch]
        
        # Find raw TIF to get background/shape
        tif_file = next((f for f in os.listdir(ch_path) if f.lower().endswith(('.tif', '.tiff'))), None)
        if not tif_file: continue
        real_img = tiff.imread(os.path.join(ch_path, tif_file))
        shape = real_img.shape[-3:] if real_img.ndim >= 3 else real_img.shape[-2:]
        ndim = len(shape)
        
        # Extract Background Noise Profile
        bg_mean = np.median(real_img)
        bg_mask = real_img < np.percentile(real_img, 90)
        bg_std = np.std(real_img[bg_mask]) if np.any(bg_mask) else 1.0
        synth_float = np.random.normal(loc=bg_mean, scale=bg_std, size=shape)
        
        # Load Stats
        df = None
        for d in os.listdir(ch_path):
            if "_processed_" in d:
                csv_p = os.path.join(ch_path, d, "metrics_df_fluorescence_2d.csv")
                if not os.path.exists(csv_p):
                    csv_p = os.path.join(ch_path, d, "metrics_df.csv") # 3D fallback
                if os.path.exists(csv_p):
                    df = pd.read_csv(csv_p)
                    break
                    
        # Apply Relational Filter if requested
        if df is not None and not df.empty and rel_filter:
            rel_csv = os.path.join(rel_dir, rel_filter, sample_name, f"{sample_name}_relational_metrics.csv")
            if os.path.exists(rel_csv):
                rel_df = pd.read_csv(rel_csv)
                # Find the parent_id column matching the template channel
                parent_col =[c for c in rel_df.columns if c.startswith('parent_id_') and template_ch.split('_')[-1] in c]
                if parent_col:
                    valid_ids = rel_df[parent_col[0]].unique()
                    df = df[df['label'].isin(valid_ids)]
                    
        # Draw Objects
        spots_drawn = 0
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                # Extract template parameters (fallback to defaults if missing)
                pixel_count = int(row.get('pixel_count', 10))
                intensity = float(row.get('mean_intensity', bg_mean + bg_std*10))
                branches = int(row.get('true_num_branches', 0))
                
                # CHANGED: Use raw pixel length instead of physical micron length
                skel_pixels = float(row.get('skan_num_skeleton_pixels', 0)) 
                
                coords = [np.random.randint(0, shape[d]) for d in range(ndim)]
                
                # CHANGED: Pass skel_pixels to the generator
                obj_mask = _generate_procedural_object(shape, pixel_count, branches, skel_pixels)
                _add_local_mask(synth_float, obj_mask, coords, intensity)
                spots_drawn += 1

        # Optical Blur & Save
        synth_float = ndi.gaussian_filter(synth_float, sigma=1.0)
        max_val = np.iinfo(real_img.dtype).max if np.issubdtype(real_img.dtype, np.integer) else 1.0
        synth_img = np.clip(synth_float, 0, max_val).astype(real_img.dtype)
        
        # Save exact folder structure
        sample_out = os.path.join(out_dir, sample_name)
        os.makedirs(sample_out, exist_ok=True)
        tiff.imwrite(os.path.join(sample_out, f"{sample_name}.tif"), synth_img)
        
        # Copy YAML configuration, tagging it as procedurally generated so the
        # channel can be robustly identified later (e.g. cleaned up on re-setup).
        yml = next((f for f in os.listdir(ch_path) if f.lower().endswith(('.yaml', '.yml'))), None)
        if yml:
            try:
                with open(os.path.join(ch_path, yml), 'r') as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
            cfg['synthetic'] = True
            with open(os.path.join(sample_out, yml), 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        print(f"[{sample_name}] Generated {spots_drawn} procedural objects.")