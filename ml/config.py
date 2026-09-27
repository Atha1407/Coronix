"""Centralized configuration for the StenoTrace ML/CV pipeline.

All pipeline parameters, CADICA taxonomy constants, thresholds, and
disclaimer texts are maintained here to avoid hard-coded magic values.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ==============================================================================
# CADICA Severity Taxonomy (Jiménez-Partinen et al., 2024)
# ==============================================================================
# CADICA defines exactly 7 discrete stenosis categories.
# Continuous / decimal stenosis percentages must NEVER be invented or displayed.
CADICA_CATEGORIES: List[str] = [
    "<20%",
    "20-50%",
    "50-70%",
    "70-90%",
    "90-98%",
    "99%",
    "100%",
]

CADICA_ID_TO_LABEL: Dict[str, str] = {
    "p0_20": "<20%",
    "p20_50": "20-50%",
    "p50_70": "50-70%",
    "p70_90": "70-90%",
    "p90_98": "90-98%",
    "p99": "99%",
    "p100": "100%",
}

LABEL_TO_CADICA_ID: Dict[str, str] = {v: k for k, v in CADICA_ID_TO_LABEL.items()}

# Color codes for visual badges and UI rendering
SEVERITY_COLORS: Dict[str, str] = {
    "<20%": "#22c55e",    # Green (minimal)
    "20-50%": "#84cc16",   # Light green / lime (mild)
    "50-70%": "#eab308",   # Amber / yellow (moderate)
    "70-90%": "#f97316",   # Orange (significant)
    "90-98%": "#ef4444",   # Red (severe)
    "99%": "#b91c1c",      # Dark red (critical/subtotal)
    "100%": "#7f1d1d",     # Deep maroon (total occlusion)
}


# ==============================================================================
# Mandatory Medical Disclaimer & Model Metadata
# ==============================================================================
MEDICAL_DISCLAIMER: str = (
    "This tool is an AI-assisted research/hackathon prototype for demonstration "
    "purposes only. It is not a medical device, does not provide a diagnosis, "
    "and must not be used for clinical decision-making."
)

MODEL_VERSION_LABEL: str = (
    "Hackathon CV Pipeline v0.1 - classical CV, not a clinically validated model"
)


# ==============================================================================
# Preprocessing Configuration
# ==============================================================================
@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters for angiogram image preprocessing."""
    # Standard CADICA frame resolution is 512x512
    target_size: Tuple[int, int] = (512, 512)
    resize_enabled: bool = False  # Keep native resolution by default
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # Light denoising
    denoise_enabled: bool = True
    denoise_method: str = "gaussian"  # 'gaussian', 'bilateral', or 'median'
    gaussian_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.8
    bilateral_d: int = 5
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 25.0


# ==============================================================================
# ROI Extraction Configuration
# ==============================================================================
@dataclass(frozen=True)
class ROIConfig:
    """Parameters for user A/B point selection and corridor extraction."""
    # Analysis corridor width perpendicular to the A-B vector
    default_corridor_width: int = 40
    min_corridor_width: int = 10
    max_corridor_width: int = 120
    
    # Minimum allowed Euclidean distance between Point A and Point B
    min_point_distance_px: float = 12.0
    
    # Boundary padding margin around cropped ROI
    padding_margin_px: int = 4


# ==============================================================================
# Vessel Enhancement Configuration
# ==============================================================================
@dataclass(frozen=True)
class VesselEnhancementConfig:
    """Parameters for classical CV vessel filtering."""
    method: str = "frangi"  # 'frangi', 'tophat', or 'multiscale'
    frangi_scale_range: Tuple[float, float] = (1.0, 4.0)
    frangi_scale_step: float = 0.5
    frangi_beta: float = 0.5
    frangi_c: float = 15.0
    tophat_kernel_size: Tuple[int, int] = (9, 9)


# ==============================================================================
# Lesion Detection Configuration
# ==============================================================================
@dataclass(frozen=True)
class LesionDetectionConfig:
    """Parameters for candidate lesion localization along vessel corridor."""
    # Threshold ratio for detecting lumen narrowing relative to reference width
    narrowing_threshold: float = 0.20
    # Minimum lesion window length in pixels along corridor
    min_lesion_length_px: int = 8
    # Search window step along the centerline
    sampling_step_px: int = 2


# ==============================================================================
# Fallback / Demo Configuration
# ==============================================================================
@dataclass(frozen=True)
class FallbackConfig:
    """Safe fallback defaults when CV extraction produces weak signals."""
    default_segment_label: str = "Selected coronary segment"
    default_severity_category: str = "50-70%"
    default_cadica_id: str = "p50_70"
    default_confidence: float = 0.65
    confidence_level: str = "Medium"
