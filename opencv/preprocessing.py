import cv2
import numpy as np
from typing import Union, Tuple, Dict, Any
from pathlib import Path


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV preprocessing pipeline for coronary angiography frames:
    Input -> Grayscale -> Resize/standardize -> Mild denoising -> CLAHE enhancement

    Parameters:
        image_input: File path (str/Path) OR in-memory numpy array (from DICOM or decoded image)
        target_size: Standardization dimensions (width, height), default (512, 512)

    Returns:
        img: Original input image as numpy array
        enhanced: Preprocessed, normalized, and CLAHE contrast-enhanced 8-bit image
    """
    # 1. Load or acquire input image
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not load image from path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    if img.size == 0:
        raise ValueError("Input image array is empty")

    # 2. Grayscale conversion
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img[:, :, 0]
    elif len(img.shape) == 2:
        gray = img.copy()
    else:
        raise ValueError(f"Unexpected image dimensions: {img.shape}")

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 3. Resize / Standardize
    if target_size is not None and (gray.shape[1], gray.shape[0]) != target_size:
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = gray

    # 4. Mild Denoising (Gaussian blur 3x3)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    # 5. Local Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(blurred)

    return img, enhanced


def run_preprocessing_pipeline(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = (512, 512)
) -> Dict[str, Any]:
    """
    Runs the complete preprocessing pipeline and returns comprehensive metadata
    including coordinate scaling factors for mapping frontend coordinates.
    """
    original, enhanced = preprocess_image(image_input, target_size=target_size)

    orig_h, orig_w = original.shape[:2]
    target_w, target_h = target_size

    scale_x = target_w / float(orig_w) if orig_w > 0 else 1.0
    scale_y = target_h / float(orig_h) if orig_h > 0 else 1.0

    return {
        "original_image": original,
        "enhanced_image": enhanced,
        "original_shape": (orig_h, orig_w),
        "target_shape": (target_h, target_w),
        "scale_factors": {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "inv_scale_x": 1.0 / scale_x if scale_x != 0 else 1.0,
            "inv_scale_y": 1.0 / scale_y if scale_y != 0 else 1.0
        }
    }