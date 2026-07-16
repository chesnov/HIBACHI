"""helper_funcs: backward-compatible facade.

The implementation was split into focused submodules along functional seams.
This module re-exports the original public names so existing imports keep
working unchanged (e.g. `from .helper_funcs import create_parameter_widget`).
"""

from .gui_text_utils import natural_sort_key, clean_filename_for_matching
from .metadata import MetadataExtractor, get_sample_metadata
from .parameter_widgets import ScalesTableWidget, create_parameter_widget
from .project_scaffolding import scan_available_presets, apply_template_config_to_project, organize_channel_project, organize_processing_dir
from .project_manager import ApplicationState, app_state, ProjectManager
from .cross_channel_window import CrossChannelAnalyzerWindow
from .project_view_window import ProjectViewWindow
from .app_launch import _check_if_last_window, _handle_napari_close, interactive_segmentation_with_config, launch_image_segmentation_tool, create_back_to_project_button
from .metadata import HAS_CZI
from .relational_engine import RelationalEngine
from .project_view_window import BatchProcessor

__all__ = [
    "natural_sort_key",
    "clean_filename_for_matching",
    "MetadataExtractor",
    "get_sample_metadata",
    "ScalesTableWidget",
    "create_parameter_widget",
    "scan_available_presets",
    "apply_template_config_to_project",
    "organize_channel_project",
    "organize_processing_dir",
    "ApplicationState",
    "app_state",
    "ProjectManager",
    "ProjectViewWindow",
    "CrossChannelAnalyzerWindow",
    "_check_if_last_window",
    "_handle_napari_close",
    "interactive_segmentation_with_config",
    "launch_image_segmentation_tool",
    "create_back_to_project_button",
    "HAS_CZI",
    "RelationalEngine",
    "BatchProcessor",
]