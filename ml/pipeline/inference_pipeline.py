"""Thin orchestrator for the complete StenoTrace ML/CV inference pipeline.

Pipeline flow:
Image -> Preprocessing -> A/B Vessel Validation -> A/B ROI Corridor Extraction
      -> Trained CADICA Faster R-CNN Lesion Detection -> Visualization Overlay.
"""

import base64
import io
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from ml.config import (
    MEDICAL_DISCLAIMER,
    MODEL_VERSION_LABEL,
    FallbackConfig,
    LesionDetectionConfig,
    PreprocessingConfig,
    ROIConfig,
    VesselEnhancementConfig,
)
from ml.lesion.config import PROJECT_ROOT
from ml.lesion.device import get_device
from ml.lesion.inference import load_model, predict
from ml.lesion.lesion_detector import detect_lesion
from ml.preprocessing.image_preprocessing import preprocess_image
from ml.roi.roi_extractor import extract_roi
from ml.severity.severity_mapper import map_severity
from ml.utils.image_utils import encode_image_to_base64, load_image
from ml.utils.visualization import draw_analysis_overlay
from ml.vessel.vessel_enhancement import enhance_vessels, validate_point_on_vessel

logger = logging.getLogger(__name__)

# Global singleton model cache
_MODEL_INSTANCE: Optional[torch.nn.Module] = None
_MODEL_DEVICE: Optional[torch.device] = None


def get_loaded_lesion_detector(device: Optional[torch.device] = None) -> torch.nn.Module:
    """Retrieve singleton trained Faster R-CNN lesion detector loaded on CUDA."""
    global _MODEL_INSTANCE, _MODEL_DEVICE
    if _MODEL_INSTANCE is None:
        target_device = device or get_device(verbose=False)
        chk_path = PROJECT_ROOT / "ml" / "models" / "best_lesion_detector.pth"
        if not chk_path.is_file():
            chk_path = PROJECT_ROOT / "ml" / "models" / "latest_lesion_detector.pth"
        if not chk_path.is_file():
            raise FileNotFoundError(f"Trained CADICA checkpoint not found at {chk_path}")

        _MODEL_INSTANCE = load_model(chk_path, device=target_device)
        _MODEL_DEVICE = target_device
        logger.info("Loaded trained CADICA lesion detector onto %s", target_device)

    return _MODEL_INSTANCE


def compute_box_corridor_overlap(
    box_xyxy: List[float],
    corridor_bbox_xywh: Tuple[int, int, int, int],
    corridor_polygon: Optional[np.ndarray] = None,
    padding: float = 10.0,
) -> float:
    """Compute geometric overlap area between predicted box and A-B corridor polygon."""
    bx1, by1, bx2, by2 = box_xyxy
    box_poly = np.array([
        [bx1, by1],
        [bx2, by1],
        [bx2, by2],
        [bx1, by2]
    ], dtype=np.float32)

    if corridor_polygon is not None and len(corridor_polygon) >= 4:
        c_poly = np.asarray(corridor_polygon, dtype=np.float32)
        inter_area, _ = cv2.intersectConvexConvex(box_poly, c_poly)
        if inter_area > 0.0:
            return float(inter_area)

        if padding > 0.0:
            padded_box = np.array([
                [bx1 - padding, by1 - padding],
                [bx2 + padding, by1 - padding],
                [bx2 + padding, by2 + padding],
                [bx1 - padding, by2 + padding]
            ], dtype=np.float32)
            padded_inter, _ = cv2.intersectConvexConvex(padded_box, c_poly)
            if padded_inter > 0.0:
                return float(padded_inter)
        return 0.0

    rx, ry, rw, rh = corridor_bbox_xywh
    rx1 = rx - padding
    ry1 = ry - padding
    rx2 = rx + rw + padding
    ry2 = ry + rh + padding

    overlap_x = max(0.0, min(bx2, rx2) - max(bx1, rx1))
    overlap_y = max(0.0, min(by2, ry2) - max(by1, ry1))
    return float(overlap_x * overlap_y)


def check_box_intersects_corridor(
    box_xyxy: List[float],
    corridor_bbox_xywh: Tuple[int, int, int, int],
    corridor_polygon: Optional[np.ndarray] = None,
    padding: float = 10.0,
) -> bool:
    """Check if a detected bounding box intersects or lies near the A-B vessel corridor."""
    return compute_box_corridor_overlap(box_xyxy, corridor_bbox_xywh, corridor_polygon, padding) > 0.0


def analyze_coronary_lesion(
    image: Union[str, bytes, np.ndarray],
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    corridor_width: Optional[int] = None,
    confidence_threshold: float = 0.15,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute end-to-end A/B guided coronary lesion detection analysis.

    Workflow:
    1. Decode & preprocess angiogram.
    2. Validate Point A is on/near a coronary vessel.
    3. Validate Point B is on/near a coronary vessel.
    4. Extract vessel corridor ROI between Point A and Point B.
    5. Run trained Faster R-CNN CADICA lesion detector on CUDA.
    6. Filter & map detected lesions overlapping the selected corridor.
    7. Generate visual overlay with A/B markers, corridor, and predicted lesion box.

    Returns:
        Dict[str, Any]: Standardized JSON response payload.
    """
    # 1. Load image
    try:
        raw_image = load_image(image)
    except Exception as exc:
        return {
            "success": False,
            "point_a": {"x": point_a[0], "y": point_a[1]},
            "point_b": {"x": point_b[0], "y": point_b[1]},
            "vessel_validation": {"point_a_valid": False, "point_b_valid": False},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": f"Invalid angiogram image: {exc}",
        }

    h, w = raw_image.shape[:2]
    ax, ay = int(round(point_a[0])), int(round(point_a[1]))
    bx, by = int(round(point_b[0])), int(round(point_b[1]))

    # Basic bounds check
    if ax < 0 or ax >= w or ay < 0 or ay >= h:
        return {
            "success": False,
            "point_a": {"x": ax, "y": ay},
            "point_b": {"x": bx, "y": by},
            "vessel_validation": {"point_a_valid": False, "point_b_valid": False},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": f"Point A ({ax}, {ay}) is outside image boundaries ({w}x{h}).",
        }

    if bx < 0 or bx >= w or by < 0 or by >= h:
        return {
            "success": False,
            "point_a": {"x": ax, "y": ay},
            "point_b": {"x": bx, "y": by},
            "vessel_validation": {"point_a_valid": True, "point_b_valid": False},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": f"Point B ({bx}, {by}) is outside image boundaries ({w}x{h}).",
        }

    # Minimum separation check
    dist_ab = np.hypot(bx - ax, by - ay)
    if dist_ab < 10.0:
        return {
            "success": False,
            "point_a": {"x": ax, "y": ay},
            "point_b": {"x": bx, "y": by},
            "vessel_validation": {"point_a_valid": True, "point_b_valid": False},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": "Point A and Point B are too close together. Select distinct vessel endpoints.",
        }

    # Preprocess image for vesselness validation & ROI
    preprocessed = preprocess_image(raw_image)

    # 2. Validate Point A on/near coronary vessel
    a_valid, a_score, a_reason = validate_point_on_vessel(preprocessed, (ax, ay))
    if not a_valid:
        return {
            "success": False,
            "point_a": {"x": ax, "y": ay},
            "point_b": {"x": bx, "y": by},
            "vessel_validation": {"point_a_valid": False, "point_b_valid": False},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": "Point A is not on a valid vessel region.",
        }

    # 3. Validate Point B on/near coronary vessel
    b_valid, b_score, b_reason = validate_point_on_vessel(preprocessed, (bx, by))
    if not b_valid:
        return {
            "success": False,
            "point_a": {"x": ax, "y": ay},
            "point_b": {"x": bx, "y": by},
            "vessel_validation": {"point_a_valid": True, "point_b_valid": False},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": "Point B is not on a valid vessel region.",
        }

    # 4. Extract vessel corridor ROI between Point A and Point B
    try:
        roi_result = extract_roi(
            preprocessed,
            point_a=(ax, ay),
            point_b=(bx, by),
            width=corridor_width,
        )
    except Exception as exc:
        return {
            "success": False,
            "point_a": {"x": ax, "y": ay},
            "point_b": {"x": bx, "y": by},
            "vessel_validation": {"point_a_valid": True, "point_b_valid": True},
            "roi": None,
            "lesion_detected": False,
            "lesions": [],
            "overlay_image": None,
            "message": f"ROI corridor extraction failed: {exc}",
        }

    rx, ry, rw, rh = roi_result.bounding_box

    # 5. Run trained CADICA lesion detector
    detector = model or get_loaded_lesion_detector(device=device)
    raw_detections = predict(
        image=raw_image,
        model=detector,
        device=device or _MODEL_DEVICE,
        confidence_threshold=min(0.01, confidence_threshold),
    )

    logger.info("=== CADICA A/B CORRIDOR DETECTION LOGGING ===")
    logger.info("Raw detection count (conf >= %.2f): %d", min(0.01, confidence_threshold), len(raw_detections))
    for idx, det in enumerate(raw_detections):
        logger.info(
            "  [Raw %d] bbox=%s, confidence=%.4f, severity=%s",
            idx, det["bbox"], det["confidence"], det["severity"]
        )
    corridor_poly_list = roi_result.corridor_polygon.tolist() if roi_result.corridor_polygon is not None else None
    logger.info("A/B corridor: A=(%d, %d), B=(%d, %d), corridor_polygon=%s", ax, ay, bx, by, corridor_poly_list)

    # 6. Filter detections that fall within or intersect the A-B vessel corridor
    matched_lesions: List[Dict[str, Any]] = []
    for idx, det in enumerate(raw_detections):
        box_xyxy = det["bbox"]
        overlap_area = compute_box_corridor_overlap(
            box_xyxy=box_xyxy,
            corridor_bbox_xywh=roi_result.bounding_box,
            corridor_polygon=roi_result.corridor_polygon,
            padding=10.0,
        )
        intersects = overlap_area > 0.0
        logger.info(
            "  [Det %d] intersects corridor: %s (overlap area=%.2f px^2)",
            idx, intersects, overlap_area
        )
        if intersects:
            lx1, ly1, lx2, ly2 = box_xyxy
            lx = int(round(lx1))
            ly = int(round(ly1))
            lw = int(round(lx2 - lx1))
            lh = int(round(ly2 - ly1))

            matched_lesions.append({
                "bbox": [lx, ly, lw, lh],
                "severity": det["severity"],
                "confidence": round(det["confidence"], 4),
                "overlap_area": overlap_area,
            })

    # Sort matched lesions by confidence descending
    matched_lesions.sort(key=lambda x: x["confidence"], reverse=True)
    candidate_detected = len(matched_lesions) > 0
    selected_detection = matched_lesions[0] if candidate_detected else None

    # Check against configured confidence threshold
    has_lesion = bool(candidate_detected and selected_detection["confidence"] >= confidence_threshold)
    status_str = (
        "Lesion Detected"
        if has_lesion
        else (
            f"Below configured confidence threshold ({int(round(confidence_threshold * 100))}%)"
            if candidate_detected
            else "Not Detected"
        )
    )
    logger.info("Selected detection in A/B corridor: %s (has_lesion=%s, status=%s)", selected_detection, has_lesion, status_str)

    # 7. Generate Visual Overlay
    primary_box = selected_detection["bbox"] if candidate_detected else None
    primary_sev = selected_detection["severity"] if candidate_detected else None
    primary_conf = selected_detection["confidence"] if candidate_detected else None

    annotated = draw_analysis_overlay(
        image=raw_image,
        point_a=(ax, ay),
        point_b=(bx, by),
        lesion_bbox=primary_box,
        severity_category=primary_sev,
        corridor_width=roi_result.corridor_width,
        corridor_polygon=roi_result.corridor_polygon,
        confidence=primary_conf,
        lesions=matched_lesions,
        include_disclaimer=True,
    )

    b64_str = encode_image_to_base64(annotated)
    overlay_data_uri = f"data:image/png;base64,{b64_str}"

    return {
        "success": True,
        "point_a": {"x": ax, "y": ay},
        "point_b": {"x": bx, "y": by},
        "vessel_validation": {
            "point_a_valid": True,
            "point_b_valid": True,
        },
        "roi": {
            "bbox": [rx, ry, rw, rh],
            "width": rw,
            "height": rh,
            "length": round(roi_result.length, 2),
            "corridor_width": roi_result.corridor_width,
        },
        "lesion_detected": has_lesion,
        "candidate_detected": candidate_detected,
        "status": status_str,
        "confidence_threshold": confidence_threshold,
        "primary_lesion": selected_detection,
        "lesions": matched_lesions,
        "overlay_image": overlay_data_uri,
        "message": status_str,
    }


# Backward-compatible run_pipeline wrapper for existing tests
@dataclass
class PipelineResult:
    """Consolidated output of the end-to-end coronary lesion analysis pipeline."""
    segment_label: str
    lesion_detected: bool
    lesion_bbox: Optional[Tuple[int, int, int, int]]
    severity_category: str
    cadica_id: str
    confidence: float
    relative_position: float
    overlay_image: Optional[np.ndarray]
    overlay_image_base64: Optional[str]
    disclaimer: str
    model_version: str
    is_fallback: bool = False


def run_pipeline(
    image: Union[str, bytes, np.ndarray],
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    corridor_width: Optional[int] = None,
    preprocessing_cfg: Optional[PreprocessingConfig] = None,
    roi_cfg: Optional[ROIConfig] = None,
    vessel_cfg: Optional[VesselEnhancementConfig] = None,
    lesion_cfg: Optional[LesionDetectionConfig] = None,
    include_base64: bool = True,
) -> PipelineResult:
    """Execute analysis pipeline with classical CV fallback."""
    fallback_cfg = FallbackConfig()

    try:
        raw_image = load_image(image)
        preprocessed = preprocess_image(raw_image, config=preprocessing_cfg)
        roi_result = extract_roi(
            preprocessed,
            point_a=point_a,
            point_b=point_b,
            width=corridor_width,
            config=roi_cfg,
        )
        enhanced_roi = enhance_vessels(roi_result.roi_image, config=vessel_cfg)
        lesion_result = detect_lesion(
            enhanced_roi,
            roi_result=roi_result,
            config=lesion_cfg,
        )
        severity_result = map_severity(lesion_result)
        annotated = draw_analysis_overlay(
            image=raw_image,
            point_a=roi_result.point_a,
            point_b=roi_result.point_b,
            lesion_bbox=lesion_result.full_image_bbox,
            severity_category=severity_result.category,
            cadica_id=severity_result.cadica_id,
            corridor_width=roi_result.corridor_width,
        )
        b64_overlay = encode_image_to_base64(annotated) if include_base64 else None

        return PipelineResult(
            segment_label=lesion_result.segment_label,
            lesion_detected=lesion_result.lesion_detected,
            lesion_bbox=lesion_result.full_image_bbox,
            severity_category=severity_result.category,
            cadica_id=severity_result.cadica_id,
            confidence=lesion_result.confidence,
            relative_position=lesion_result.relative_position,
            overlay_image=annotated,
            overlay_image_base64=b64_overlay,
            disclaimer=MEDICAL_DISCLAIMER,
            model_version=MODEL_VERSION_LABEL,
            is_fallback=False,
        )
    except Exception as exc:
        logger.warning("Pipeline encountered error; activating safe demo fallback: %s", exc)
        raw_image = load_image(image) if not isinstance(image, np.ndarray) else image
        annotated = draw_analysis_overlay(
            image=raw_image,
            point_a=point_a,
            point_b=point_b,
            severity_category=fallback_cfg.default_severity_category,
            cadica_id=fallback_cfg.default_cadica_id,
        )
        b64_overlay = encode_image_to_base64(annotated) if include_base64 else None

        return PipelineResult(
            segment_label=f"{fallback_cfg.default_segment_label} (Demo Fallback)",
            lesion_detected=True,
            lesion_bbox=None,
            severity_category=fallback_cfg.default_severity_category,
            cadica_id=fallback_cfg.default_cadica_id,
            confidence=fallback_cfg.default_confidence,
            relative_position=0.5,
            overlay_image=annotated,
            overlay_image_base64=b64_overlay,
            disclaimer=MEDICAL_DISCLAIMER,
            model_version=MODEL_VERSION_LABEL,
            is_fallback=True,
        )
