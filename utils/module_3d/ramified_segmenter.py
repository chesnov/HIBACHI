# --- START OF FILE utils/module_3d/ramified_segmenter.py ---
"""
Facade for the Ramified Segmentation Logic.
The actual implementations have been moved to:
- soma_extraction.py
- cell_splitting.py
- segmentation_helpers.py
"""

# Import from the new modules
try:
    from .soma_extraction import extract_soma_masks
    from .cell_splitting import separate_multi_soma_cells
except ImportError:
    # Fallback for direct execution
    from soma_extraction import extract_soma_masks
    from cell_splitting import separate_multi_soma_cells

# Re-export for compatibility
__all__ = ['extract_soma_masks', 'separate_multi_soma_cells']
# --- END OF FILE utils/module_3d/ramified_segmenter.py ---