"""
ML Interface for Coronix Angiography Lesion Analysis.

This module provides a clean, decoupled interface between the OpenCV/DICOM
preprocessing pipeline and the downstream machine learning lesion detection model.
The ML team can plug their PyTorch / ONNX / TensorFlow model directly into this interface
by implementing BaseLesionModel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


@dataclass
class MLModelInput:
    """
    Standardized payload prepared by the OpenCV pipeline for the ML model.
    """
    # 1. Standardized enhanced image (typically 512x512 uint8)
    preprocessed_image: np.ndarray

    # 2. Corridor-isolated full-frame image (non-corridor pixels masked to 0)
    roi_image: np.ndarray

    # 3. Tight cropped patch of the corridor bounding box
    cropped_roi: np.ndarray

    # 4. Binary mask of the corridor (255 inside corridor, 0 outside)
    roi_mask: np.ndarray

    # 5. Bounding box enclosing the corridor [x, y, width, height] in preprocessed coordinates
    roi_bbox: List[int]

    # 6. Standardized segment endpoints (x, y)
    point_a: Tuple[int, int]
    point_b: Tuple[int, int]

    # 7. Original viewer coordinates before resizing
    point_a_orig: Tuple[int, int]
    point_b_orig: Tuple[int, int]

    # 8. Coordinate transformation metadata
    original_shape: Tuple[int, int]  # (height, width)
    scale_factors: Dict[str, float]  # {'scale_x': sx, 'scale_y': sy, 'inv_scale_x': ..., 'inv_scale_y': ...}

    # 9. Additional clinical parameters
    catheter_size: Optional[float] = None
    corridor_width: int = 40
    dicom_metadata: Optional[Dict[str, Any]] = None


@dataclass
class MLModelOutput:
    """
    Standardized inference results returned by the ML model.
    """
    segment_label: str
    lesion_detected: bool
    lesion_bbox: Optional[List[float]] = None  # [x, y, w, h] in original image coordinates
    severity_category: Optional[str] = None    # e.g., "70-90%", "<50%", "50-70%"
    confidence: float = 0.0                    # 0.0 to 1.0
    catheter_size: Optional[float] = None
    vessel_mask: Optional[str] = None
    raw_predictions: Dict[str, Any] = field(default_factory=dict)


class BaseLesionModel(ABC):
    """
    Abstract base class for all lesion analysis models.
    The ML team should inherit from this class to integrate their model.
    """

    @abstractmethod
    def predict(self, model_input: MLModelInput) -> MLModelOutput:
        """
        Run lesion prediction on preprocessed OpenCV input.

        Parameters:
            model_input: Clean MLModelInput prepared by OpenCV pipeline

        Returns:
            MLModelOutput with lesion detection, bounding box, and severity
        """
        pass


class MockLesionModel(BaseLesionModel):
    """
    Default mock/baseline implementation of the lesion model.
    Used for development and testing until the ML team integrates the trained model.
    """

    def predict(self, model_input: MLModelInput) -> MLModelOutput:
        pt_a = model_input.point_a_orig
        pt_b = model_input.point_b_orig

        # Calculate a reasonable lesion bounding box around the midpoint of A and B
        mid_x = (pt_a[0] + pt_b[0]) / 2.0
        mid_y = (pt_a[1] + pt_b[1]) / 2.0
        dist = float(np.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]))

        # Lesion box roughly 35% of corridor length or minimum 40px
        box_size = max(40.0, dist * 0.35)
        bbox = [
            round(mid_x - box_size / 2.0, 1),
            round(mid_y - box_size / 2.0, 1),
            round(box_size, 1),
            round(box_size, 1),
        ]

        # Determine segment label heuristic based on position or metadata
        segment = "LAD"
        if model_input.dicom_metadata and "SeriesDescription" in model_input.dicom_metadata:
            desc = str(model_input.dicom_metadata["SeriesDescription"]).upper()
            if "RCA" in desc:
                segment = "RCA"
            elif "LCX" in desc:
                segment = "LCx"

        return MLModelOutput(
            segment_label=segment,
            lesion_detected=True,
            lesion_bbox=bbox,
            severity_category="70-90%",
            confidence=0.89,
            catheter_size=model_input.catheter_size,
            raw_predictions={"corridor_length_px": dist}
        )


# Global model registry allowing teammates to swap the active model
_ACTIVE_MODEL: BaseLesionModel = MockLesionModel()


def get_lesion_model() -> BaseLesionModel:
    """Returns the current active lesion model instance."""
    return _ACTIVE_MODEL


def set_lesion_model(model: BaseLesionModel) -> None:
    """
    Registers a new lesion model (e.g. when the ML team loads their PyTorch weights).
    """
    global _ACTIVE_MODEL
    if not isinstance(model, BaseLesionModel):
        raise TypeError("Model must inherit from BaseLesionModel")
    _ACTIVE_MODEL = model
