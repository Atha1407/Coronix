import io
import base64
import logging
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import cv2

logger = logging.getLogger(__name__)


class DICOMProcessingError(Exception):
    """Custom exception for DICOM parsing and frame extraction errors."""
    pass


def extract_dicom_metadata(ds: pydicom.Dataset) -> Dict[str, Any]:
    """
    Extract relevant DICOM metadata including modality, dimensions, 
    number of frames, and pixel spacing.
    """
    # Safe tag extraction helper
    def safe_get(tag: str, default=None):
        try:
            val = getattr(ds, tag, default)
            if val is None or val == "":
                return default
            # Handle pydicom MultiValue or PersonName or numbers
            if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                return [float(x) if isinstance(x, (int, float, pydicom.valuerep.DSfloat, pydicom.valuerep.IS)) else str(x) for x in val]
            if isinstance(val, (int, float, pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
                return float(val) if isinstance(val, (float, pydicom.valuerep.DSfloat)) else int(val)
            return str(val).strip()
        except Exception:
            return default

    # Number of frames
    num_frames = 1
    raw_frames = safe_get("NumberOfFrames", 1)
    try:
        num_frames = int(raw_frames)
    except (ValueError, TypeError):
        num_frames = 1

    metadata: Dict[str, Any] = {
        "Modality": safe_get("Modality", "Unknown"),
        "Rows": safe_get("Rows"),
        "Columns": safe_get("Columns"),
        "NumberOfFrames": num_frames,
        "PhotometricInterpretation": safe_get("PhotometricInterpretation", "MONOCHROME2"),
        "BitsAllocated": safe_get("BitsAllocated"),
        "BitsStored": safe_get("BitsStored"),
        "PixelSpacing": safe_get("PixelSpacing"),
        "ImagerPixelSpacing": safe_get("ImagerPixelSpacing"),
        "SliceThickness": safe_get("SliceThickness"),
        "PatientID": safe_get("PatientID", "ANONYMOUS"),
        "StudyDescription": safe_get("StudyDescription"),
        "SeriesDescription": safe_get("SeriesDescription"),
        "Manufacturer": safe_get("Manufacturer"),
        "WindowCenter": safe_get("WindowCenter"),
        "WindowWidth": safe_get("WindowWidth"),
        "RescaleSlope": safe_get("RescaleSlope", 1.0),
        "RescaleIntercept": safe_get("RescaleIntercept", 0.0),
    }

    # Normalize pixel spacing info
    pixel_spacing = metadata.get("PixelSpacing") or metadata.get("ImagerPixelSpacing")
    if pixel_spacing and isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2:
        metadata["PixelDimensionMm"] = {
            "row_spacing_mm": float(pixel_spacing[0]),
            "col_spacing_mm": float(pixel_spacing[1]),
        }
    else:
        metadata["PixelDimensionMm"] = None

    return metadata


def normalize_frame_to_uint8(
    frame: np.ndarray,
    ds: pydicom.Dataset
) -> np.ndarray:
    """
    Normalizes a DICOM raw frame into an 8-bit uint8 NumPy/OpenCV image (0-255).
    Handles:
    - Rescale slope & intercept
    - Windowing (WindowCenter & WindowWidth)
    - MONOCHROME1 inversion (so vessels remain dark on bright backgrounds or vice versa standardly)
    - Fallback min-max normalization
    """
    frame = frame.astype(np.float32)

    # 1. Rescale Slope / Intercept
    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        frame = frame * slope + intercept
    except Exception as e:
        logger.warning(f"Error applying rescale slope/intercept: {e}")

    # 2. Windowing (Window Center & Width)
    has_windowing = False
    try:
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)

        if wc is not None and ww is not None:
            # Handle multi-value windowing
            center = float(wc[0] if hasattr(wc, "__iter__") and not isinstance(wc, (str, bytes)) else wc)
            width = float(ww[0] if hasattr(ww, "__iter__") and not isinstance(ww, (str, bytes)) else ww)

            if width > 0:
                lower = center - 0.5 - (width - 1.0) / 2.0
                upper = center - 0.5 + (width - 1.0) / 2.0
                frame = np.clip(frame, lower, upper)
                if upper > lower:
                    normalized = ((frame - lower) / (upper - lower) * 255.0).astype(np.uint8)
                    has_windowing = True
    except Exception as e:
        logger.warning(f"Error applying windowing parameters: {e}")

    # 3. Fallback to Min-Max if windowing was unavailable or failed
    if not has_windowing:
        f_min = float(np.min(frame))
        f_max = float(np.max(frame))
        if f_max > f_min:
            normalized = (((frame - f_min) / (f_max - f_min)) * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros_like(frame, dtype=np.uint8)

    # 4. Handle MONOCHROME1 Photometric Interpretation
    # MONOCHROME1 means 0 is white and max value is black (needs inversion for standard view)
    try:
        photo_interp = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
        if photo_interp == "MONOCHROME1":
            normalized = 255 - normalized
    except Exception as e:
        logger.warning(f"Error checking PhotometricInterpretation: {e}")

    return normalized


def frame_to_base64_png(frame: np.ndarray) -> str:
    """
    Encodes a uint8 OpenCV/NumPy frame as a PNG base64 Data URI.
    """
    success, buffer = cv2.imencode(".png", frame)
    if not success:
        raise DICOMProcessingError("Failed to encode image frame to PNG format")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def read_dicom_file(
    file_source: Union[str, bytes, io.BytesIO],
    frame_index: int = 0
) -> Tuple[np.ndarray, Dict[str, Any], int, str]:
    """
    Reads a DICOM file from file path, bytes, or file-like buffer.

    Parameters:
        file_source: File path (str), raw bytes (bytes), or BytesIO stream
        frame_index: 0-indexed frame number to extract (defaults to 0)

    Returns:
        frame_uint8: np.ndarray of shape (H, W) or (H, W, 3) uint8
        metadata: dict containing extracted DICOM tags
        total_frames: total number of frames in the dataset
        image_base64: Base64 data URI string for frontend display
    """
    # 1. Parse DICOM dataset
    try:
        if isinstance(file_source, str):
            ds = pydicom.dcmread(file_source, force=True)
        elif isinstance(file_source, bytes):
            ds = pydicom.dcmread(io.BytesIO(file_source), force=True)
        else:
            ds = pydicom.dcmread(file_source, force=True)
    except InvalidDicomError as e:
        raise DICOMProcessingError(f"Invalid or corrupted DICOM file: {str(e)}")
    except Exception as e:
        raise DICOMProcessingError(f"Failed to parse DICOM file: {str(e)}")

    # 2. Check for pixel data existence
    if "PixelData" not in ds and not hasattr(ds, "pixel_array"):
        raise DICOMProcessingError("DICOM file contains no PixelData element")

    # 3. Decode pixel array
    try:
        pixel_array = ds.pixel_array
    except Exception as e:
        raise DICOMProcessingError(f"Failed to decode DICOM pixel data: {str(e)}")

    if pixel_array is None or pixel_array.size == 0:
        raise DICOMProcessingError("DICOM pixel array is empty")

    # 4. Handle single-frame vs multi-frame
    total_frames = 1
    raw_frame: np.ndarray

    if pixel_array.ndim == 2:
        # Standard single frame grayscale (H, W)
        total_frames = 1
        if frame_index != 0:
            raise DICOMProcessingError(f"Invalid frame index {frame_index}. DICOM has only 1 frame.")
        raw_frame = pixel_array

    elif pixel_array.ndim == 3:
        # Ambiguity check: (frames, H, W) vs (H, W, channels)
        samples_per_pixel = getattr(ds, "SamplesPerPixel", 1)
        photo_interp = str(getattr(ds, "PhotometricInterpretation", "")).upper()

        if samples_per_pixel == 3 or photo_interp in ["RGB", "YBR_FULL", "YBR_FULL_422"]:
            # Single frame color image
            total_frames = 1
            if frame_index != 0:
                raise DICOMProcessingError(f"Invalid frame index {frame_index}. DICOM has only 1 frame.")
            raw_frame = pixel_array
        else:
            # Multi-frame grayscale: (frames, H, W)
            total_frames = pixel_array.shape[0]
            if frame_index < 0 or frame_index >= total_frames:
                raise DICOMProcessingError(
                    f"Invalid frame index {frame_index}. Multi-frame DICOM contains {total_frames} frames (indices 0 to {total_frames - 1})."
                )
            raw_frame = pixel_array[frame_index]

    elif pixel_array.ndim == 4:
        # Multi-frame color: (frames, H, W, channels)
        total_frames = pixel_array.shape[0]
        if frame_index < 0 or frame_index >= total_frames:
            raise DICOMProcessingError(
                f"Invalid frame index {frame_index}. Multi-frame DICOM contains {total_frames} frames (indices 0 to {total_frames - 1})."
            )
        raw_frame = pixel_array[frame_index]

    else:
        raise DICOMProcessingError(f"Unsupported DICOM pixel dimension: {pixel_array.ndim}D")

    # 5. Extract metadata
    metadata = extract_dicom_metadata(ds)
    metadata["TotalFrames"] = total_frames
    metadata["CurrentFrameIndex"] = frame_index

    # 6. Normalize frame to uint8
    frame_uint8 = normalize_frame_to_uint8(raw_frame, ds)

    # If it's a 3-channel RGB image, convert to BGR for OpenCV
    if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 3:
        frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

    # 7. Generate base64 representation for frontend
    image_base64 = frame_to_base64_png(frame_uint8)

    return frame_uint8, metadata, total_frames, image_base64
