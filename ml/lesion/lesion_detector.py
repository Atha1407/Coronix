"""Candidate lesion detection interface and data models.

Responsible for:
- Detecting candidate narrowing/lesion regions along the vessel corridor
- Returning structured lesion information (bounding boxes, relative position, confidence)
- Providing a clean, pluggable interface for future algorithm refinement
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ml.config import LesionDetectionConfig
from ml.roi.roi_extractor import ROIExtractionResult


@dataclass
class LesionDetectionResult:
    """Structured output of the lesion detection module.

    Attributes:
        lesion_detected: Boolean flag indicating if a candidate narrowing was identified.
        roi_bbox: [x, y, w, h] bounding box in rectified ROI coordinate space.
        full_image_bbox: [x, y, w, h] bounding box in original angiogram image coordinate space.
        relative_position: Normalized location along the A-B segment (0.0 = near A, 1.0 = near B).
        measured_narrowing_ratio: Estimated diameter ratio (used internally strictly for CADICA binning).
        confidence: Confidence score (0.0 to 1.0).
        segment_label: Anatomical or positional label (e.g., 'Selected coronary segment').
    """
    lesion_detected: bool
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
    full_image_bbox: Optional[Tuple[int, int, int, int]] = None
    relative_position: float = 0.5
    measured_narrowing_ratio: float = 0.0
    confidence: float = 0.5
    segment_label: str = "Selected coronary segment"


def detect_lesion(
    enhanced_roi: np.ndarray,
    roi_result: Optional[ROIExtractionResult] = None,
    config: Optional[LesionDetectionConfig] = None
) -> LesionDetectionResult:
    """Detect candidate narrowing or lesion region along the enhanced vessel corridor.

    NOTE: This is an initial interface stub designed for modular step-by-step implementation.
    The detailed cross-sectional profile analysis will be implemented in subsequent steps.

    Args:
        enhanced_roi: 2D enhanced vessel map inside the corridor strip.
        roi_result: Metadata from the ROI extraction step (contains transformation matrix and A/B coordinates).
        config: Optional LesionDetectionConfig override.

    Returns:
        LesionDetectionResult: Structured detection metadata.
    """
    cfg = config or LesionDetectionConfig()

    if enhanced_roi is None or enhanced_roi.size == 0:
        return LesionDetectionResult(
            lesion_detected=False,
            confidence=0.0,
            segment_label="Selected coronary segment",
        )

    h, w = enhanced_roi.shape[:2]

    # Interface baseline: default candidate region at midpoint of corridor
    mid_x = w // 2
    box_w = max(cfg.min_lesion_length_px, int(w * 0.25))
    box_h = max(8, int(h * 0.6))
    roi_bbox = (max(0, mid_x - box_w // 2), max(0, (h - box_h) // 2), box_w, box_h)

    # Calculate full image bounding box if roi_result is available
    full_image_bbox = None
    if roi_result is not None:
        full_image_bbox = roi_result.bounding_box

    return LesionDetectionResult(
        lesion_detected=True,
        roi_bbox=roi_bbox,
        full_image_bbox=full_image_bbox,
        relative_position=0.5,
        measured_narrowing_ratio=0.55,  # Interface placeholder for CADICA category assignment
        confidence=0.75,
        segment_label="Selected coronary segment",
    )
