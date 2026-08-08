"""metadata: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import re
import traceback
import yaml  # type: ignore
import numpy as np
import tifffile as tiff  # type: ignore
from xml.etree import ElementTree as ET
from typing import Dict, Any, Union

# --- Optional CZI support ---
try:
    from aicspylibczi import CziFile  # type: ignore
    HAS_CZI = True
except ImportError:
    HAS_CZI = False
    print("Warning: 'aicspylibczi' not installed. CZI support disabled.")



class ChannelExtractionError(RuntimeError):
    """Raised when a channel could not be written, carrying the specific reason.

    Extraction used to fail silently: every failure path returned None, identical
    to success, so project setup created image-less folders and reported nothing.
    The reason is the whole value of this exception -- "aicspylibczi is not
    installed" and "not a TIFF file" call for completely different user actions.
    """


class MetadataExtractor:
    """Helper class to parse dimensions and physical scales from microscopy files."""

    @staticmethod
    def _slide_source(path: str):
        """(FormatSpec, source_key) if `path` is a slide source, else (None, None).

        Accepts a plain path or a source key like ``/data/Image.vsi::20x_01``, so
        callers already threading a filename through don't need to know that one
        slide file can contain several scenes.
        """
        try:
            from .slide_formats import spec_for_path
            from .slide_reader import parse_source_key
        except Exception:
            return None, None
        filename, _scene = parse_source_key(path)
        return spec_for_path(filename), path

    @staticmethod
    def get_channel_count(path: str) -> int:
        """Determines the number of channels in a file (slide, CZI or TIFF)."""
        spec, key = MetadataExtractor._slide_source(path)
        if spec is not None:
            from .slide_reader import scene_channel_count
            return scene_channel_count(key)

        ext = os.path.splitext(path)[1].lower()
        if ext == '.czi' and HAS_CZI:
            try:
                czi = CziFile(path)
                dims_list = czi.get_dims_shape() if hasattr(czi, 'get_dims_shape') else czi.dims_shape()
                if dims_list:
                    dims = dims_list[0]
                    if 'C' in dims: return dims['C'][1] - dims['C'][0]
                return 1
            except Exception: return 1

        elif ext in ['.tif', '.tiff']:
            try:
                with tiff.TiffFile(path) as tif:
                    if tif.imagej_metadata:
                        return int(tif.imagej_metadata.get('channels', 1))
                    if tif.ome_metadata:
                        match = re.search(r'SizeC="(\d+)"', str(tif.ome_metadata))
                        if match: return int(match.group(1))
                    if len(tif.series) > 0:
                        shape = tif.series[0].shape
                        if len(shape) == 3 and shape[0] < 10 and shape[0] < shape[1]: return shape[0]
                        if len(shape) == 4: return min(shape[0], shape[1])
            except Exception: return 1
        return 1

    @staticmethod
    def read_slide_metadata(path: str) -> Dict[str, Union[float, bool]]:
        """Physical scale of a slide source, in the read_tiff_metadata shape."""
        from .slide_reader import scene_metadata
        _spec, key = MetadataExtractor._slide_source(path)
        return scene_metadata(key or path)

    @staticmethod
    def read_tiff_metadata(path: str) -> Dict[str, Union[float, bool]]:
        """Attempts to read physical scale (microns) with robust ImageJ support."""
        meta: Dict[str, Union[float, bool]] = {'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False}
        try:
            with tiff.TiffFile(path) as tif:
                ij = tif.imagej_metadata or {}
                # 1. Capture Z-Spacing from ImageJ immediately
                if 'spacing' in ij:
                    meta['z'] = float(ij['spacing'])
                    meta['found'] = True

                # 2. Capture X/Y from Tags or ImageJ
                if tif.pages:
                    page = tif.pages[0]
                    x_res = page.tags.get('XResolution')
                    y_res = page.tags.get('YResolution')
                    u_tag = page.tags.get('ResolutionUnit')
                    
                    if x_res and y_res:
                        x_val, y_val = x_res.value, y_res.value
                        x_dens = x_val[0]/x_val[1] if isinstance(x_val, tuple) else x_val
                        y_dens = y_val[0]/y_val[1] if isinstance(y_val, tuple) else y_val
                        
                        # Unit detection: Tag says 'None' (1), but ImageJ string might say 'micron'
                        unit_str = str(ij.get('unit', '')).lower()
                        u_val = u_tag.value if u_tag else 1
                        
                        if x_dens > 0:
                            # Case: Unit is Microns (Standard for Fiji calibration)
                            if u_val == 3 or unit_str in ['micron', 'µm', 'um']:
                                # If unit is cm (3), density is px/cm. 10000/dens = um/px
                                # If unit is micron, density is px/um. 1/dens = um/px
                                factor = 10000.0 if u_val == 3 else 1.0
                                meta['x'], meta['y'] = factor/x_dens, factor/y_dens
                                meta['found'] = True
                            # Case: Unit is Inches (DPI)
                            elif u_val == 2:
                                meta['x'], meta['y'] = 25400.0/x_dens, 25400.0/y_dens
                                meta['found'] = True
                            # Case: Unit is "None" but we have numbers (often happens in bio-formats)
                            elif u_val == 1:
                                if x_dens < 1.0: # Likely already microns per pixel
                                    meta['x'], meta['y'] = x_dens, y_dens
                                else: # Likely pixels per micron
                                    meta['x'], meta['y'] = 1.0/x_dens, 1.0/y_dens
                                meta['found'] = True

                # 3. OME-XML Fallback
                if not meta['found'] and tif.ome_metadata:
                    txt = str(tif.ome_metadata)
                    for ax in ['X', 'Y', 'Z']:
                        m = re.search(rf'PhysicalSize{ax}="([\d\.]+)"', txt)
                        if m: 
                            meta[ax.lower()] = float(m.group(1))
                            meta['found'] = True

        except Exception as e:
            print(f"Metadata read error: {e}")
        return meta

    @staticmethod
    def _parse_czi_xml_scaling(xml_input: Any) -> Dict[str, float]:
        """Parses CZI XML object/string to find scaling in MICRONS."""
        scales = {}
        try:
            root = None
            if hasattr(xml_input, 'getroot'):
                root = xml_input.getroot()
            elif ET.iselement(xml_input):
                root = xml_input
            elif isinstance(xml_input, (str, bytes)):
                try:
                    if len(str(xml_input)) < 255 and os.path.exists(xml_input):
                        root = ET.parse(xml_input).getroot()
                    else:
                        root = ET.fromstring(xml_input)
                except Exception:
                    pass

            if root is not None:
                for dist in root.iter('Distance'):
                    axis_id = dist.get('Id')
                    val_node = dist.find('Value')
                    if axis_id and val_node is not None and val_node.text:
                        try:
                            scales[axis_id] = float(val_node.text) * 1e6
                        except ValueError:
                            pass
        except Exception as e:
            print(f"    Error parsing CZI XML: {e}")
        return scales

    @staticmethod
    def extract_channel_to_tiff(src_path: str, dest_path: str, channel_idx: int,
                                progress=None, should_cancel=None) -> bool:
        """Extracts a channel and preserves the spatial resolution tags.

        Returns True on success and raises ChannelExtractionError otherwise.
        Previously this returned None in every case -- including the several early
        ``return`` paths and the blanket ``except`` -- so a caller could not tell a
        successful extraction from one that wrote nothing. Project setup
        consequently created image folders containing only a config, which then
        failed validation with no indication of why.
        """
        spec, key = MetadataExtractor._slide_source(src_path)
        if spec is not None:
            # Slides extract tile-by-tile straight to disk instead of via ch_data
            # below: one scene of the tested VSI is 997 megapixels, so assembling
            # a channel in memory would mean a 2 GB allocation.
            from .slide_reader import extract_scene_channel
            try:
                if extract_scene_channel(key, dest_path, channel_idx,
                                         progress=progress,
                                         should_cancel=should_cancel):
                    return True
                raise ChannelExtractionError(
                    "slide extraction produced no image data")
            except ChannelExtractionError:
                raise
            except Exception as exc:
                from .slide_reader import SetupCancelled
                if isinstance(exc, SetupCancelled):
                    raise  # user-initiated, not a failure to report
                raise ChannelExtractionError(
                    f"{type(exc).__name__}: {exc}") from exc

        try:
            ext = os.path.splitext(src_path)[1].lower()
            ch_data = None
            source_meta = {'x': 1.0, 'y': 1.0, 'z': 1.0}

            # --- BRANCH 1: CZI FILES ---
            if ext == '.czi' and HAS_CZI:
                # 1. Get Metadata specifically for CZI
                source_meta = MetadataExtractor.get_czi_metadata(src_path)
                
                # 2. Extract Data using aicspylibczi
                try:
                    czi = CziFile(src_path)
                    # Read specific channel. explicit T=0 ensures we get a volume, not a 4D hyperstack if time exists
                    # This returns (data, list_of_dims). Data usually has shape (1, 1, Z, Y, X) or similar.
                    data, dims = czi.read_image(C=channel_idx)
                    ch_data = np.squeeze(data)
                except Exception as czi_e:
                    raise ChannelExtractionError(
                        f"CZI read error: {czi_e}") from czi_e

            # --- BRANCH 2: TIFF FILES ---
            elif ext in ['.tif', '.tiff']:
                # 1. Get Metadata specifically for TIFF
                source_meta = MetadataExtractor.read_tiff_metadata(src_path)
                
                # 2. Extract Data using tifffile
                vol = tiff.imread(src_path)
                
                # Handle ImageJ Hyperstacks (Z vs C vs T)
                if vol.ndim == 3:
                    # Differentiate (C,Y,X) from (Z,Y,X)
                    # Heuristic: Channels usually < 10, Z usually < Y/X
                    if vol.shape[0] < 10 and vol.shape[0] < vol.shape[1]: 
                        ch_data = vol[channel_idx]
                    else: 
                        # Assumes single channel Z-stack
                        ch_data = vol
                elif vol.ndim == 4:
                    # Usually (C, Z, Y, X) or (Z, C, Y, X). 
                    # Simplistic assumption: Smallest dim is C.
                    if vol.shape[0] < vol.shape[1]: # (C, Z, Y, X)
                        ch_data = vol[channel_idx]
                    else: # (Z, C, Y, X)
                        ch_data = vol[:, channel_idx, :, :]
                else:
                    ch_data = vol
            
            elif ext == '.czi' and not HAS_CZI:
                # Called out separately from "unsupported": the format IS supported,
                # the optional reader just isn't installed, and that distinction is
                # what tells the user how to fix it.
                raise ChannelExtractionError(
                    "cannot read .czi because 'aicspylibczi' is not installed "
                    "in this environment"
                )

            else:
                raise ChannelExtractionError(
                    f"unsupported file type for extraction: {ext}")

            # --- COMMON: SAVE TO DISK ---
            if ch_data is None:
                raise ChannelExtractionError("no channel data could be selected")

            res_x = MetadataExtractor._safe_resolution(source_meta.get('x'))
            res_y = MetadataExtractor._safe_resolution(source_meta.get('y'))
            spacing = MetadataExtractor._safe_spacing(source_meta.get('z'))

            # imagej=True is required for the 'micron' unit to survive a round
            # trip. Without it tifffile records ResolutionUnit=INCH and reading the
            # file back gives a pixel size ~25400x too large -- masked inside
            # HIBACHI (dimensions come from the config) but wrong for anyone who
            # opens the extracted TIFF in Fiji.
            tiff.imwrite(
                dest_path, ch_data,
                imagej=True,
                photometric='minisblack',
                resolution=(res_x, res_y),
                metadata={'unit': 'micron', 'spacing': spacing}
            )
            # Confirm rather than assume: imwrite can leave a zero-length file if
            # the volume runs out of space mid-write, and a 0-byte .tif would pass
            # a bare existence check downstream.
            if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
                raise ChannelExtractionError(
                    "the write completed but produced no data on disk "
                    "(is the volume full or read-only?)")
            return True

        except ChannelExtractionError:
            raise  # already carries a precise reason
        except Exception as e:
            # Wrap anything unexpected so the caller still gets a usable reason
            # instead of the old silent None.
            print(f"Extraction failed for {os.path.basename(src_path)}: {e}")
            traceback.print_exc()
            raise ChannelExtractionError(f"{type(e).__name__}: {e}") from e

    # TIFF stores resolution as a RATIONAL (two uint32s). Converting a float
    # outside roughly 1e-6..1e6 overflows that conversion and makes imwrite raise
    # -- which used to abort extraction silently, per file, depending only on what
    # scale tag the source happened to carry.
    _MIN_TIFF_RESOLUTION = 1e-6
    _MAX_TIFF_RESOLUTION = 1e6

    @staticmethod
    def _safe_resolution(spacing: Any) -> float:
        """Pixels-per-micron for a micron-per-pixel spacing, clamped to writable.

        Falls back to 1.0 for missing, non-finite, non-positive or physically
        implausible spacings (below a picometre or above a metre per pixel), since
        such a value can only come from a junk resolution tag.
        """
        try:
            s = float(spacing)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(s) or s <= 0:
            return 1.0
        res = 1.0 / s
        if not (MetadataExtractor._MIN_TIFF_RESOLUTION
                <= res <= MetadataExtractor._MAX_TIFF_RESOLUTION):
            print(f"    Warning: implausible pixel size {s:g} um; writing "
                  "resolution 1.0 instead.")
            return 1.0
        return res

    @staticmethod
    def _safe_spacing(z: Any) -> float:
        """Z spacing for the ImageJ 'spacing' tag, sanitised the same way."""
        try:
            s = float(z)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(s) or s <= 0:
            return 1.0
        return s

    @staticmethod
    def get_czi_metadata(path: str) -> Dict[str, Union[float, bool]]:
        """Wrapper to get metadata specifically for CZI files."""
        if not HAS_CZI:
            return {'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False}
        czi = CziFile(path)
        scale_map = {}
        if hasattr(czi, 'pixel_scaling'):
            try:
                scale_map = {k: v * 1e6 for k, v in czi.pixel_scaling.items()}
            except Exception:
                pass
        if not scale_map and hasattr(czi, 'meta'):
            xml = czi.meta() if callable(czi.meta) else czi.meta
            scale_map = MetadataExtractor._parse_czi_xml_scaling(xml)
        return {
            'x': scale_map.get('X', 1.0),
            'y': scale_map.get('Y', 1.0),
            'z': scale_map.get('Z', 1.0),
            'found': bool(scale_map)
        }

def get_sample_metadata(folder_path):
    """Retrieves shape and spacing from the YAML in a project folder."""
    for f in os.listdir(folder_path):
        if f.endswith(('.yaml', '.yml')):
            with open(os.path.join(folder_path, f), 'r') as file:
                cfg = yaml.safe_load(file)
                mode = cfg.get('mode', '')
                is_2d = mode.endswith('_2d')
                dim_key = 'pixel_dimensions' if is_2d else 'voxel_dimensions'
                dims = cfg.get(dim_key, {'x':1, 'y':1, 'z':1})
                # Note: We'd need actual pixel counts to calculate spacing, 
                # but for preview, we can often rely on the Strategy to provide this.
                return dims, mode
    return None, None