"""Lesion dimension quantification and image-derived stenosis estimation module.

Calculates physical dimensions (width, height, length) from pixel bounding boxes
and computes image-derived diameter stenosis percentage.

DISCLAIMER:
All measurements and stenosis percentages produced here are research/hackathon
image-derived estimates and are NOT clinically validated diagnostic values.
"""

import math
from typing import Dict, Optional, Tuple, Union
from ml.quantification.calibration import pixels_to_mm


def calculate_image_derived_stenosis_percent(
    min_diameter_px: Optional[Union[int, float]],
    reference_diameter_px: Optional[Union[int, float]],
) -> Optional[float]:
    """Calculate image-derived diameter stenosis percentage from lumen diameters.
    
    Formula:
        stenosis_percent = (1 - min_diameter_px / reference_diameter_px) * 100
        
    Clamped to range [0.0, 100.0].
    
    Args:
        min_diameter_px: Minimal lumen diameter (MLD) in pixels.
        reference_diameter_px: Normal reference vessel diameter (RVD) in pixels.
        
    Returns:
        Optional[float]: Estimated stenosis percentage clamped to [0.0, 100.0],
                         or None if diameters cannot be reliably obtained or RVD <= 0.
    """
    if min_diameter_px is None or reference_diameter_px is None:
        return None
        
    if reference_diameter_px <= 0 or min_diameter_px < 0:
        return None
        
    raw_percent = (1.0 - (float(min_diameter_px) / float(reference_diameter_px))) * 100.0
    return float(max(0.0, min(100.0, round(raw_percent, 2))))


def compute_bbox_dimensions_px(
    bbox: Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]
) -> Dict[str, float]:
    """Extract width, height, and diagonal length from [x, y, w, h] bounding box.
    
    Args:
        bbox: Bounding box as [x, y, width, height] or (x, y, width, height).
        
    Returns:
        Dict[str, float]: width_px, height_px, length_px
    """
    _, _, w, h = bbox
    w_px = float(max(0.0, w))
    h_px = float(max(0.0, h))
    diag_length_px = float(math.hypot(w_px, h_px))
    
    return {
        "width_px": round(w_px, 2),
        "height_px": round(h_px, 2),
        "length_px": round(diag_length_px, 2),
    }


def compute_physical_lesion_measurements(
    bbox: Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]],
    mm_per_pixel: float,
    min_diameter_px: Optional[float] = None,
    reference_diameter_px: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Compute physical lesion dimensions in mm and image-derived stenosis percent.
    
    Args:
        bbox: Bounding box [x, y, w, h] on original image coordinate space.
        mm_per_pixel: Calibration factor in millimeters per pixel.
        min_diameter_px: Optional minimal lumen diameter inside lesion.
        reference_diameter_px: Optional reference vessel diameter.
        
    Returns:
        Dict[str, Optional[float]]: Physical measurements in millimeters and stenosis percent.
    """
    px_dims = compute_bbox_dimensions_px(bbox)
    
    w_mm = round(pixels_to_mm(px_dims["width_px"], mm_per_pixel), 3)
    h_mm = round(pixels_to_mm(px_dims["height_px"], mm_per_pixel), 3)
    len_mm = round(pixels_to_mm(px_dims["length_px"], mm_per_pixel), 3)
    
    stenosis = calculate_image_derived_stenosis_percent(
        min_diameter_px=min_diameter_px,
        reference_diameter_px=reference_diameter_px,
    )
    
    return {
        "lesion_width_px": px_dims["width_px"],
        "lesion_height_px": px_dims["height_px"],
        "lesion_length_px": px_dims["length_px"],
        "lesion_width_mm": w_mm,
        "lesion_height_mm": h_mm,
        "lesion_length_mm": len_mm,
        "image_derived_stenosis_percent": stenosis,
    }
