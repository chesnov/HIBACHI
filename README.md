# HIBACHI
### Heuristic-Informed Batch Analysis for Cell Histological Identification

**HIBACHI** is a modular, memory-safe application designed for the automated segmentation, separation, and analysis of cells in large 2D and 3D microscopy datasets. It is engineered to process terabyte-scale volumes on standard laptops by utilizing aggressive memory mapping, chunked processing, and heuristic graph-based separation.

## 🚀 Key Features

*   **Memory Efficiency:** Uses `numpy.memmap` and `dask` to process images significantly larger than physical RAM.
*   **Robust Separation:** Solves the "clumped cell" problem using a custom graph-based merge algorithm that analyzes geometric "necks" and intensity valleys.
*   **Batch Processing:** "Set and Forget" automated processing of entire experiment folders with crash recovery and resume capability.
*   **Interactive GUI:** Built on **Napari** and **PyQt5** for visualizing results and tuning parameters step-by-step.
*   **Modular Workflow:** 5-step pipeline covering everything from raw signal enhancement to skeletonization.

---

## 🛠️ Installation

### Prerequisites
*   **Anaconda** or **Miniconda** installed on your system.
*   **Git** (optional, to clone the repo).

### 1. Clone the Repository
```bash
git clone https://github.com/chesnov/HIBACHI.git
cd HIBACHI
```

### 2. Create the Environment
We provide an `environment.yaml` file that contains all necessary dependencies (Napari, SimpleITK, Dask, etc.).

```bash
# Create the environment from the file
conda env create -f environment.yaml

# Activate the environment
conda activate hibachi
```

### 3. Run the Application
```bash
python segment.py
```

---

## 📂 Data Preparation & Project Structure

To use the **Batch Processor** and **Automatic Initialization**, your data must be organized specifically.

### 1. Input Directory
Place all your raw `.tif` or `.tiff` images in a single folder. Do not create subfolders yet.

### 2. The Metadata CSV
You must provide **exactly one** `.csv` file in that same folder. This file contains the physical dimensions of your images. This is crucial because HIBACHI uses physical units (microns) for thresholds, not pixels.

**Required CSV Format:**

| Filename | Width (um) | Height (um) | Depth (um) | Slices |
| :--- | :--- | :--- | :--- | :--- |
| `image_01.tif` | 1024.5 | 1024.5 | 50.0 | 100 |
| `image_02.tif` | 512.0 | 512.0 | 20.0 | 40 |

*   **Filename**: Exact name of the image file (including extension).
*   **Width/Height (um)**: Total physical dimensions of the XY plane.
*   **Depth (um)**: Total physical depth (Z-axis). For 2D images, use 0.
*   **Slices**: Number of Z-slices (used for validation).

### 3. Automatic Initialization
When you load this folder in the GUI for the first time:
1.  HIBACHI will detect the flat structure.
2.  It will ask to **Organize** the project.
3.  It will move every image into its own subfolder (e.g., `./image_01/image_01.tif`).
4.  It will generate a `ramified_config.yaml` for each image, pre-filled with the voxel dimensions calculated from your CSV.

---

## 🧠 Processing Workflow

The pipeline consists of 5 modular steps. You can tune parameters for each step via the GUI sidebar.

1.  **Raw Segmentation:** Uses Hessian filters (Frangi/Sato) to detect tubular structures and global thresholding for volume.
2.  **Edge Trimming:** Generates a 3D convex hull around the tissue and removes artifacts caused by edge damage/fluorescence.
3.  **Soma Extraction:** "Peels" the binary mask using iterative erosion to find the core seeds (cell bodies) within dense clumps.
4.  **Cell Separation:** Uses the seeds from Step 3 to perform a Marker-Controlled Watershed on the remaining mask, followed by a graph-based merge to fix over-segmentation.
5.  **Feature Calculation:** Computes Volume, Surface Area, Sphericity, Skeleton/Ramification stats, and Pairwise Distances.

---

## 📚 Documentation (Wiki)

Detailed documentation for the architecture, logic, and parameter tuning of each module can be found in the `wiki/` folder.

### Core Application
*   **[Entry Point (segment.py)](wiki/segment.md):** Application bootstrap and error handling.
*   **[GUI Manager (gui_manager.py)](wiki/gui_manager.md):** Manages the sidebar interactions and state.
*   **[Batch Processor (batch_processor.py)](wiki/batch_processor.md):** Logic for iterating over folders and managing memory during batch runs.

### The 3D Pipeline (Modules)
*   **[Strategy Controller (_3D_strategy.py)](wiki/_3D_strategy.md):** The orchestrator that manages data flow between steps.
*   **[Step 1: Raw Segmentation (initial_3d_segmentation.py)](wiki/initial_3d_segmentation.md):** Tubular enhancement and thresholding logic.
*   **[Step 2: Edge Trimming (remove_artifacts.py)](wiki/remove_artifacts.md):** 3D Convex hull and boundary cleaning.
*   **[Step 3: Soma Extraction (soma_extraction.py)](wiki/soma_extraction.md):** **(Read for Tuning)** How to detect cell seeds.
*   **[Step 4: Cell Separation (cell_splitting.py)](wiki/cell_splitting.md):** **(Read for Tuning)** How to split touching cells and fix straight-line artifacts.
*   **[Step 5: Features (calculate_features_3d.py)](wiki/calculate_features_3d.md):** Metrics, skeletonization, and distance mapping.
*   **[Helpers (segmentation_helpers.py)](wiki/segmentation_helpers.md):** Shared math and system utilities.

---

## 📄 Outputs

For every image, the following files are generated in `image_folder/image_name_processed_ramified/`:

*   `final_segmentation_ramified.dat`: The final integer label mask (Memmap).
*   `metrics_df_ramified.csv`: Spreadsheet containing Volume, Depth, Shape, Neighbor, and Skeleton stats per cell.
*   `skeleton_array_ramified.dat`: Visualization of the 1-pixel skeletons.
*   `cell_bodies.dat`: Intermediate mask of detected seeds.
*   `points_matrix_ramified.csv`: Data used to draw neighbor connection lines.
*   `processing_config_ramified.yaml`: A record of the exact parameters used for this run.

---
