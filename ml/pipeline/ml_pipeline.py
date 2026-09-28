"""Integrated ML pipeline combining A/B vessel validation, ROI corridor extraction,
trained CADICA Faster R-CNN inference, and physical catheter calibration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ml.config import MEDICAL_DISCLAIMER
from ml.lesion.config import PROJECT_ROOT
from ml.lesion.device import get_device
from ml.pipeline.inference_pipeline import (
    analyze_coronary_lesion,
    get_loaded_lesion_detector,
)
from ml.quantification.calibration import (
    calculate_mm_per_pixel,
    fr_to_mm,
    pixels_to_mm,
)
from ml.quantification.measurements import compute_physical_lesion_measurements
from ml.utils.image_utils import load_image


def analyze_image(
    image_path: Union[str, Path, np.ndarray],
    point_a: Tuple[Union[int, float], Union[int, float]],
    point_b: Tuple[Union[int, float], Union[int, float]],
    catheter_fr: Union[int, float],
    catheter_diameter_px: Union[int, float],
    confidence_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Execute complete coronary angiogram analysis with physical catheter calibration.

    Workflow:
    1. Validate calibration inputs and compute physical scale (mm/px).
    2. Load image and validate Point A and Point B on coronary vessel.
    3. Extract vessel corridor ROI.
    4. Run trained CADICA lesion detector (Faster R-CNN MobileNetV3 on CUDA).
    5. Map detection coordinates to original full image.
    6. Compute physical dimensions (width, height, length in mm).
    7. Return standardized clinical quantification dictionary.

    Args:
        image_path: Path to angiogram file (str/Path) or pre-loaded numpy image array.
        point_a: (x, y) start coordinate along vessel.
        point_b: (x, y) end coordinate along vessel.
        catheter_fr: Catheter size in French (e.g. 5.0, 6.0). Must be > 0.
        catheter_diameter_px: Measured catheter diameter in image pixels. Must be > 0.
        confidence_threshold: Detector confidence cutoff (default: 0.15).

    Returns:
        Dict[str, Any]: Standardized analysis results dictionary.
    """
    pa = [round(float(point_a[0]), 1), round(float(point_a[1]), 1)]
    pb = [round(float(point_b[0]), 1), round(float(point_b[1]), 1)]

    # 1. Catheter calibration calculation & validation
    try:
        cath_fr = float(catheter_fr)
        cath_px = float(catheter_diameter_px)
        cath_mm = fr_to_mm(cath_fr)
        mm_per_px = calculate_mm_per_pixel(cath_fr, cath_px)
    except Exception as exc:
        return {
            "valid_points": False,
            "point_a": pa,
            "point_b": pb,
            "catheter_fr": catheter_fr,
            "catheter_diameter_mm": None,
            "catheter_diameter_px": catheter_diameter_px,
            "mm_per_pixel": None,
            "lesion_detected": False,
            "bbox": None,
            "severity": None,
            "confidence": None,
            "lesion_width_mm": None,
            "lesion_height_mm": None,
            "lesion_length_mm": None,
            "image_derived_stenosis_percent": None,
            "error": f"Invalid catheter calibration input: {exc}",
        }

    # 2. Image loading and core pipeline analysis (includes vessel validation & ROI & model)
    try:
        analysis_res = analyze_coronary_lesion(
            image=image_path,
            point_a=(int(round(pa[0])), int(round(pa[1]))),
            point_b=(int(round(pb[0])), int(round(pb[1]))),
            confidence_threshold=confidence_threshold,
        )
    except Exception as exc:
        return {
            "valid_points": False,
            "point_a": pa,
            "point_b": pb,
            "catheter_fr": cath_fr,
            "catheter_diameter_mm": cath_mm,
            "catheter_diameter_px": cath_px,
            "mm_per_pixel": mm_per_px,
            "lesion_detected": False,
            "bbox": None,
            "severity": None,
            "confidence": None,
            "lesion_width_mm": None,
            "lesion_height_mm": None,
            "lesion_length_mm": None,
            "image_derived_stenosis_percent": None,
            "error": f"Pipeline analysis failed: {exc}",
        }

    # If vessel validation failed on Point A or Point B
    v_val = analysis_res.get("vessel_validation", {})
    a_ok = v_val.get("point_a_valid", False)
    b_ok = v_val.get("point_b_valid", False)
    valid_points = bool(a_ok and b_ok)

    if not valid_points or not analysis_res.get("success", False):
        return {
            "valid_points": False,
            "point_a": pa,
            "point_b": pb,
            "catheter_fr": cath_fr,
            "catheter_diameter_mm": cath_mm,
            "catheter_diameter_px": cath_px,
            "mm_per_pixel": mm_per_px,
            "lesion_detected": False,
            "bbox": None,
            "severity": None,
            "confidence": None,
            "lesion_width_mm": None,
            "lesion_height_mm": None,
            "lesion_length_mm": None,
            "image_derived_stenosis_percent": None,
            "error": analysis_res.get("message", "A/B vessel validation failed"),
        }

    # 3. Process detection output and physical measurements
    lesion_detected = analysis_res.get("lesion_detected", False)
    candidate_detected = analysis_res.get("candidate_detected", False)
    status_str = analysis_res.get("status")
    lesions = analysis_res.get("lesions", [])

    primary_bbox = None
    severity = None
    confidence = None
    w_mm = None
    h_mm = None
    len_mm = None

    if candidate_detected and len(lesions) > 0:
        top_det = lesions[0]
        primary_bbox = top_det["bbox"]  # [x, y, w, h] on original image
        severity = top_det["severity"]
        confidence = float(top_det["confidence"])

        # Compute physical dimensions in mm using mm_per_pixel
        measurements = compute_physical_lesion_measurements(
            bbox=primary_bbox,
            mm_per_pixel=mm_per_px,
            min_diameter_px=None,
            reference_diameter_px=None,
        )
        w_mm = measurements["lesion_width_mm"]
        h_mm = measurements["lesion_height_mm"]
        len_mm = measurements["lesion_length_mm"]

    return {
        "valid_points": True,
        "point_a": pa,
        "point_b": pb,
        "catheter_fr": cath_fr,
        "catheter_diameter_mm": round(cath_mm, 3),
        "catheter_diameter_px": cath_px,
        "mm_per_pixel": round(mm_per_px, 5),
        "lesion_detected": lesion_detected,
        "candidate_detected": candidate_detected,
        "status": status_str,
        "confidence_threshold": confidence_threshold,
        "bbox": primary_bbox,
        "severity": severity,
        "confidence": confidence,
        "lesion_width_mm": w_mm,
        "lesion_height_mm": h_mm,
        "lesion_length_mm": len_mm,
        "image_derived_stenosis_percent": None,  # Not invented; exact reference diameter not calibrated
        "error": None,
    }
