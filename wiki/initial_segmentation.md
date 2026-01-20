# Step 1: Raw Segmentation

**Corresponding Modules:**
*   **3D:** `utils/module_3d/initial_3d_segmentation.py`
*   **2D:** `utils/module_2d/initial_2d_segmentation.py`

## Overview
The goal of Step 1 is to convert the raw, noisy, grayscale microscopy image into a clean binary mask (Foreground vs. Background). It is specifically designed to detect **fluorescence structures** (tubular processes) which are often lost by simple thresholding.

### The Algorithm
1.  **Preprocessing (Optional):**
    *   **Background Subtraction:** Applies a white top-hat filter to remove broad background gradients while preserving local structures.
    *   **Smoothing:** Applies a Gaussian blur to suppress camera noise.
2.  **Tubular Enhancement (Hessian Filtering):**
    *   We use **Frangi** and **Sato** vesselness filters. These mathematical filters analyze the second-order derivatives (curvature) of the image intensity.
    *   They respond strongly to tube-like structures (neurites/processes) and suppress planar (background) or spherical (noise) signals.
    *   **Hybrid Approach:** The algorithm calculates both Frangi and Sato filters and takes the *maximum* response of the two for every pixel.
    *   **Soma Preservation:** Since vesselness filters suppress spheres (cell bodies), we optionally blend the original intensity back into the result to ensure somas are preserved.
3.  **Normalization:**
    *   The filtered output is normalized using a robust Z-profile (3D) or global percentile (2D) to correct for uneven illumination or signal attenuation in deep tissue.
4.  **Thresholding:**
    *   A global percentile-based threshold is applied to the normalized volume to create a binary mask.
5.  **Morphological Cleaning:**
    *   **Closing:** Small gaps in processes are bridged.
    *   **Size Filtering:** Connected components smaller than a specific size are removed as noise.

---

## 🔧 Parameter Tuning Guide

### 1. `tubular_scales`
**Type:** List of Floats (e.g., `[0.5, 1.0, 2.0]`)

*   **High Level:** The approximate "thickness" (radius in microns) of the branches you want to detect.
*   **Detailed Explanation:** These values represent the $\sigma$ (sigma) of the Gaussian derivative kernels used in the Hessian analysis. The filter responds maximally when the scale matches the physical radius of the tubular structure in the image.
*   **Image Features:**
    *   **Fine Processes:** Require small scales (e.g., `0.3`, `0.5`).
    *   **Thick Primary Branches / Somas:** Require larger scales (e.g., `2.0`, `3.0`).
*   **How to Tune:**
    *   **Scenario A: Missing fine tips.**
        *   *Fix:* Add a smaller number to the list (e.g., add `0.3` or `0.5`).
    *   **Scenario B: The segmentation looks "bubbly" or fragmented.**
        *   *Fix:* You might be missing intermediate scales. Ensure you have a range (e.g., `[0.5, 1.0, 2.0]`) rather than just `[0.5]`.
    *   **Scenario C: Too much noise/grain.**
        *   *Fix:* Remove the smallest scale (e.g., remove `0.3`). Extremely small scales often detect pixel noise as "tubes".
    *   **Disable Filter:** Set to `[0.0]` to skip vesselness filtering and use raw intensity only.

### 2. `low_threshold_percentile`
**Type:** Float (0.0 - 100.0) | **Default:** ~95.0 - 98.0

*   **High Level:** The "sensitivity" of the segmentation. Lower values = More Foreground; Higher values = Less Foreground.
*   **Detailed Explanation:** After filtering and normalization, the code constructs a histogram of the image values. This parameter picks a cutoff point. A value of `95.0` means "Keep the brightest 5% of pixels."
*   **Image Features:**
    *   **High Contrast / Clean Background:** Can use a lower percentile (e.g., `90.0`) to capture everything.
    *   **Noisy Background:** Requires a higher percentile (e.g., `99.0`) to exclude background noise.
*   **How to Tune:**
    *   **Scenario A: Cells look "anoretic" or broken; processes are disconnected.**
        *   *Fix:* **Decrease** the value (e.g., `98.0` -> `95.0`).
    *   **Scenario B: The background is filled with "snow" or static.**
        *   *Fix:* **Increase** the value (e.g., `95.0` -> `99.5`).

### 3. `high_threshold_percentile`
**Type:** Float (0.0 - 100.0) | **Default:** 95.0 - 100.0

*   **High Level:** Used for normalization. Defines what "Maximum Brightness" means for your image.
*   **Detailed Explanation:** This sets the denominator for normalization. Pixels brighter than this percentile are clamped to 1.0. This prevents a single ultra-bright artifact from squashing the rest of the signal to near-zero.
*   **How to Tune:**
    *   **Standard:** Leave at `95.0` or `99.0`.
    *   **Scenario:** If you have massive bright blobs (artifacts) that make the rest of the cell dim, **lower** this value (e.g., to `90.0`) to saturate the artifacts and boost the contrast of the dimmer structures.

### 4. `smooth_sigma`
**Type:** Float (Microns) | **Default:** ~0.5 - 1.1

*   **High Level:** Pre-blurring to remove camera noise before analysis.
*   **Detailed Explanation:** The standard deviation of the Gaussian blur applied to the raw input.
*   **How to Tune:**
    *   **Scenario A: Image is very sharp/noisy.**
        *   *Fix:* Increase sigma (e.g., `1.5`). Note: This may blur away very fine processes.
    *   **Scenario B: Image is already blurry or low-res.**
        *   *Fix:* Decrease sigma (e.g., `0.1` or `0.0`).
    *   *Warning:* If `smooth_sigma` is larger than your smallest `tubular_scale`, you will blur away the features you are trying to detect.

### 5. `connect_max_gap_physical`
**Type:** Float (Microns)

*   **High Level:** The size of the "bridge" used to connect broken parts of a cell.
*   **Detailed Explanation:** Technically, this defines the radius of the structuring element used for a **Morphological Closing** operation. It smears the binary mask slightly to fuse nearby pixels, then erodes it back.
*   **How to Tune:**
    *   **Scenario A: Branches look like dashed lines.**
        *   *Fix:* **Increase** the gap (e.g., `1.0` -> `2.5`).
    *   **Scenario B: Distinct neighboring cells are fusing together.**
        *   *Fix:* **Decrease** the gap (e.g., `1.0` -> `0.5`).

### 6. `min_size` (or `min_size_voxels`/`min_size_pixels`)
**Type:** Integer

*   **High Level:** The smallest object (dust/noise) to keep.
*   **Detailed Explanation:** Any connected component in the binary mask with fewer voxels/pixels than this count is deleted.
*   **How to Tune:**
    *   **Scenario:** You see lots of tiny 1-3 pixel speckles in the background.
        *   *Fix:* Increase this value until the speckles disappear.
    *   *Note:* In 3D, voxel counts are much higher than pixel counts. A value of `50` might be fine for 2D, but `2000` might be needed for 3D.

### 7. `soma_preservation_factor`
**Type:** Float (0.0 - 1.0+) | **Default:** 0.0

*   **High Level:** How much raw intensity (cell body signal) to mix back into the filter result.
*   **Detailed Explanation:** Vesselness filters suppress spheres (somas). This can lead to "hollow" cells. This parameter takes the normalized raw image, scales it by this factor, and takes the maximum of (Filter Result, Raw Result).
*   **How to Tune:**
    *   **Scenario: Processes are detected perfectly, but the cell body is missing or has holes.**
        *   *Fix:* Set to `1.0` or `1.5`.
    *   **Scenario: The filter is picking up too much background haze.**
        *   *Fix:* Set to `0.0`.

### 8. `subtract_background_radius`
**Type:** Integer (Pixels) | **Default:** 0

*   **High Level:** Removes broad, glowing background gradients.
*   **Detailed Explanation:** Applies a **White Top-Hat** transform. This subtracts the morphological opening from the image. It effectively removes structures larger than the specified radius.
*   **How to Tune:**
    *   **Scenario: You have large, glowing smears of background fluorescence.**
        *   *Fix:* Set this to a value *larger* than the thickest part of your cells (e.g., `20` or `30` pixels).
    *   *Warning:* If you set this too small (smaller than a cell body), you will delete the cell bodies. Default `0` disables it.