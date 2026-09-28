"""Catheter-based physical scale calibration module.

Converts catheter French size and measured pixel diameter into physical scaling factor
(mm per pixel) for clinical angiography quantification.
"""

from typing import Union


def fr_to_mm(fr: Union[int, float]) -> float:
    """Convert catheter French size to millimeters.
    
    1 French (Fr) = 1/3 mm ≈ 0.333 mm.
    
    Args:
        fr: Catheter size in French (e.g., 5.0, 6.0, 7.0). Must be positive.
        
    Returns:
        float: Catheter outer diameter in millimeters.
        
    Raises:
        ValueError: If fr <= 0.
    """
    if fr <= 0:
        raise ValueError(f"Catheter French size must be positive, got: {fr}")
    return float(fr) * 0.333


def calculate_mm_per_pixel(
    catheter_fr: Union[int, float],
    catheter_diameter_px: Union[int, float],
) -> float:
    """Calculate the physical scaling factor (mm per pixel) using a known catheter reference.
    
    Formula:
        mm_per_pixel = (catheter_fr * 0.333) / catheter_diameter_px
        
    Args:
        catheter_fr: Catheter size in French (e.g. 5, 6). Must be positive.
        catheter_diameter_px: Measured catheter diameter in image pixels. Must be positive.
        
    Returns:
        float: Physical scale in millimeters per pixel.
        
    Raises:
        ValueError: If catheter_fr <= 0 or catheter_diameter_px <= 0.
    """
    if catheter_fr <= 0:
        raise ValueError(f"Catheter French size must be positive, got: {catheter_fr}")
    if catheter_diameter_px <= 0:
        raise ValueError(f"Catheter diameter in pixels must be positive, got: {catheter_diameter_px}")
    
    catheter_diameter_mm = fr_to_mm(catheter_fr)
    return catheter_diameter_mm / float(catheter_diameter_px)


def pixels_to_mm(pixels: Union[int, float], mm_per_pixel: float) -> float:
    """Convert pixel dimension to physical millimeters using scaling factor.
    
    Args:
        pixels: Distance or length in pixels. Must be non-negative.
        mm_per_pixel: Physical scaling factor in mm/px. Must be positive.
        
    Returns:
        float: Physical length in millimeters.
        
    Raises:
        ValueError: If pixels < 0 or mm_per_pixel <= 0.
    """
    if pixels < 0:
        raise ValueError(f"Pixel dimension cannot be negative, got: {pixels}")
    if mm_per_pixel <= 0:
        raise ValueError(f"Scaling factor mm_per_pixel must be positive, got: {mm_per_pixel}")
        
    return float(pixels) * float(mm_per_pixel)
