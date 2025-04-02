def segment_microglia_first_pass_2d(volume,
                                    tubular_scales=[0.5, 1.0, 2.0],
                                    smooth_sigma=1.0,
                                    min_size=50,
                                    sensitivity=0.8,
                                    background_level=50,
                                    target_level=75):
    """
    First pass microglia segmentation for 2D images with focus on capturing processes.
    Supports both single 2D images and stacks of 2D images.
    
    Parameters:
    -----------
    volume : ndarray
        2D input image or stack of 2D images (2D or 3D array)
    tubular_scales : list
        Scales for tubular structure enhancement
    smooth_sigma : float or list
        Gaussian smoothing sigma(s)
    min_size : int
        Minimum object size in pixels
    sensitivity : float
        Sensitivity factor for thresholding (0-1)
    
    Returns:
    --------
    upsampled_first_pass : ndarray
        First pass segmentation with same shape as input
    first_pass_params : dict
        Dictionary containing parameters from first pass
    """
    from scipy.ndimage import label
    import numpy as np
    from tqdm import tqdm
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    
    # Ensure input is 3D even if it's a single 2D image
    if volume.ndim == 2:
        volume = volume[np.newaxis, :, :]
    
    # Ensure sensitivity is within expected range
    sensitivity = max(0.01, min(1.0, sensitivity))

    # Create a params dictionary to store parameters
    first_pass_params = {
        'tubular_scales': tubular_scales,
        'smooth_sigma': smooth_sigma,
        'min_size': min_size,
        'sensitivity': sensitivity
    }
    
    # Preallocate output arrays
    first_pass = np.zeros_like(volume, dtype=np.int32)
    
    # Process each 2D image in the stack
    for z in tqdm(range(volume.shape[0]), desc="Processing 2D images"):
        # Get current 2D image
        image = volume[z]
        
        # Smooth the image
        if isinstance(smooth_sigma, (list, tuple)):
            smooth_sigma_value = smooth_sigma[0]
        else:
            smooth_sigma_value = smooth_sigma
        smoothed = gaussian_filter(image, sigma=smooth_sigma_value)
        
        # Enhance tubular structures
        scaled_tubular_scales = [scale for scale in tubular_scales]
        enhanced = np.zeros_like(smoothed, dtype=smoothed.dtype)
        for scale in scaled_tubular_scales:
            # 2D Frangi vesselness or similar tubular enhancement
            # Note: You might want to replace this with a specific 2D tubular enhancement method
            enhanced = np.maximum(enhanced, gaussian_filter(smoothed, sigma=scale))
        
        # Normalize intensity
        # Calculate intensity statistics
        noise_level = np.percentile(enhanced, background_level)
        foreground = enhanced[enhanced > noise_level]
        
        if len(foreground) > 0:
            median = np.median(foreground)
            std = np.std(foreground)
            
            # Target intensity scaling
            target_intensity = np.percentile([median], target_level)
            if median > 0:
                scale_factor = target_intensity / max(median, 1e-6)
                normalized = enhanced * scale_factor
            else:
                normalized = enhanced
        else:
            normalized = enhanced
        
        # Otsu thresholding with sensitivity adjustment
        sample_size = min(1000000, normalized.size)
        samples = normalized.ravel()
        threshold = threshold_otsu(samples)
        adjusted_threshold = threshold * (1 - sensitivity * 0.5)
        
        # Binary segmentation
        binary = normalized > adjusted_threshold
        
        # Connect fragmented processes
        # For 2D, we'll use simple morphological operations
        from scipy.ndimage import binary_closing, binary_dilation
        max_gap = 3  # Fixed gap for 2D
        connected_binary = binary_closing(binary, structure=np.ones((3,3)))
        connected_binary = binary_dilation(connected_binary, iterations=max_gap)
        
        # Remove small objects
        from skimage.morphology import remove_small_objects
        cleaned_binary = remove_small_objects(connected_binary, min_size=min_size)
        
        # Label connected components
        labeled, num_labels = label(cleaned_binary)
        
        # Store labeled result
        first_pass[z] = labeled
    
    # Squeeze back to original dimensionality
    if first_pass.shape[0] == 1:
        first_pass = first_pass[0]
    
    return first_pass, first_pass_params