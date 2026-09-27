import sys
from pathlib import Path

# Add opencv directory to Python path so modules can be imported directly
OPENCV_DIR = Path(__file__).resolve().parent.parent / "opencv"
if str(OPENCV_DIR) not in sys.path:
    sys.path.insert(0, str(OPENCV_DIR))

import io
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union

from dicom_reader import read_dicom_file, DICOMProcessingError, frame_to_base64_png
from preprocessing import run_preprocessing_pipeline
from roi import extract_roi_corridor
from ml_interface import MLModelInput, MLModelOutput, get_lesion_model


def load_input_image_or_dicom(
    file_bytes: bytes,
    filename: str = "",
    frame_index: int = 0
) -> Tuple[np.ndarray, Dict[str, Any], int, str, str]:
    """
    Loads raw bytes from either a DICOM or standard image (PNG/JPG).

    Returns:
        frame_uint8: np.ndarray image
        metadata: Dict of metadata (DICOM tags or basic image dimensions)
        total_frames: int
        detected_type: 'dicom' or 'image'
        image_base64: Base64 data URI of the frame for frontend display
    """
    is_dicom = filename.lower().endswith(".dcm")

    # Try DICOM reading first if file extension suggests or if header starts with DICM
    if is_dicom or (len(file_bytes) > 132 and file_bytes[128:132] == b"DICM"):
        try:
            frame, meta, total_frames, b64 = read_dicom_file(file_bytes, frame_index=frame_index)
            return frame, meta, total_frames, "dicom", b64
        except DICOMProcessingError:
            if is_dicom:
                raise

    # Standard image loading via OpenCV
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image from provided file bytes")

    # If alpha channel present, convert to BGR
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    meta = {
        "Modality": "Angiogram (Image)",
        "Rows": h,
        "Columns": w,
        "NumberOfFrames": 1,
        "Filename": filename
    }
    b64 = frame_to_base64_png(img)
    return img, meta, 1, "image", b64


def prepare_ml_input_from_cv(
    frame_uint8: np.ndarray,
    point_a: Dict[str, float],
    point_b: Dict[str, float],
    catheter_size: Optional[float] = None,
    corridor_width: int = 40,
    dicom_metadata: Optional[Dict[str, Any]] = None,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[MLModelInput, Dict[str, Any]]:
    """
    Executes the OpenCV Preprocessing and Corridor ROI extraction pipeline
    and constructs the clean MLModelInput object.
    """
    orig_h, orig_w = frame_uint8.shape[:2]

    # 1. OpenCV Preprocessing Pipeline (Grayscale -> Resize -> Denoise -> CLAHE)
    prep = run_preprocessing_pipeline(frame_uint8, target_size=target_size)
    enhanced = prep["enhanced_image"]
    scale_x = prep["scale_factors"]["scale_x"]
    scale_y = prep["scale_factors"]["scale_y"]

    # 2. Map coordinates from original viewer resolution to standardized 512x512 space
    pt_a_orig = (int(round(point_a["x"])), int(round(point_a["y"])))
    pt_b_orig = (int(round(point_b["x"])), int(round(point_b["y"])))

    pt_a_std = (int(round(point_a["x"] * scale_x)), int(round(point_a["y"] * scale_y)))
    pt_b_std = (int(round(point_b["x"] * scale_x)), int(round(point_b["y"] * scale_y)))

    # Adjust corridor width proportionally
    avg_scale = (scale_x + scale_y) / 2.0
    width_std = max(10, int(round(corridor_width * avg_scale)))

    # 3. Corridor ROI extraction in standardized OpenCV space
    roi_data = extract_roi_corridor(
        image=enhanced,
        point_a=pt_a_std,
        point_b=pt_b_std,
        width=width_std
    )

    # 4. Construct ML input payload
    ml_input = MLModelInput(
        preprocessed_image=enhanced,
        roi_image=roi_data["roi_image"],
        cropped_roi=roi_data["cropped_roi"],
        roi_mask=roi_data["roi_mask"],
        roi_bbox=roi_data["roi_bbox"],
        point_a=pt_a_std,
        point_b=pt_b_std,
        point_a_orig=pt_a_orig,
        point_b_orig=pt_b_orig,
        original_shape=(orig_h, orig_w),
        scale_factors=prep["scale_factors"],
        catheter_size=catheter_size,
        corridor_width=corridor_width,
        dicom_metadata=dicom_metadata
    )

    cv_debug_info = {
        "roi_bbox_std": roi_data["roi_bbox"],
        "corridor_length_std": roi_data["corridor_length"],
        "point_a_std": pt_a_std,
        "point_b_std": pt_b_std,
        "target_size": target_size,
    }

    return ml_input, cv_debug_info


def process_coronary_analysis(
    file_bytes: bytes,
    point_a: Dict[str, float],
    point_b: Dict[str, float],
    catheter_size: Optional[float],
    filename: str = "",
    frame_index: int = 0
) -> Dict[str, Any]:
    """
    End-to-End Orchestrator:
    DICOM/Image Loading -> OpenCV Preprocessing -> Corridor ROI -> ML Model Interface -> Frontend Response
    """
    # 1. Load image/frame
    frame, metadata, total_frames, file_type, image_base64 = load_input_image_or_dicom(
        file_bytes=file_bytes,
        filename=filename,
        frame_index=frame_index
    )

    # 2. OpenCV Preprocessing & Corridor ROI Extraction
    ml_input, cv_debug = prepare_ml_input_from_cv(
        frame_uint8=frame,
        point_a=point_a,
        point_b=point_b,
        catheter_size=catheter_size,
        dicom_metadata=metadata
    )

    # 3. Hand off to ML Model via clean interface
    model = get_lesion_model()
    ml_output: MLModelOutput = model.predict(ml_input)

    # 4. Format clean output matching existing frontend API contract
    return {
        "segment_label": ml_output.segment_label,
        "lesion_detected": ml_output.lesion_detected,
        "lesion_bbox": ml_output.lesion_bbox,
        "severity_category": ml_output.severity_category,
        "confidence": ml_output.confidence,
        "catheter_size": ml_output.catheter_size,
        "file_type": file_type,
        "total_frames": total_frames,
        "frame_index": frame_index,
        "metadata": metadata,
        "cv_pipeline": {
            "scale_factors": ml_input.scale_factors,
            "original_shape": ml_input.original_shape,
            "roi_bbox_std": cv_debug["roi_bbox_std"]
        }
    }
