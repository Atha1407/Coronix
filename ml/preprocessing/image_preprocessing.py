"""Angiogram image preprocessing module for StenoTrace.

Responsible for:
- STEP 1: Input validation (type, dimension, non-empty, finite checks)
- STEP 2: Grayscale conversion (handling grayscale, BGR, and RGB inputs)
- STEP 3: Intensity normalization (scaling reliably to 0–255 uint8 range)
- STEP 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- STEP 5: Light denoising (subtle Gaussian filtering preserving thin vessels)
- STEP 6: Optional resizing (disabled by default to preserve spatial A/B coordinates)

IMPORTANT:
This is an AI-assisted research/hackathon prototype and is NOT a clinically validated
medical system. Transformations must preserve vessel geometry without aggressive loss
of fine vascular details.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ml.config import PreprocessingConfig
from ml.utils.image_utils import load_image


def validate_input_image(
    image: Union[str, Path, bytes, np.ndarray]
) -> np.ndarray:
    """Validate input image type, dimensions, and finite values.

    Args:
        image: File path, byte stream, or loaded NumPy array.

    Returns:
        np.ndarray: Validated NumPy array.

    Raises:
        ValueError: If image is None, empty, has invalid dimensions, or non-finite values.
        TypeError: If input is of an unsupported type.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    # Ingest from path, bytes, or existing array
    if isinstance(image, (str, Path, bytes)):
        try:
            arr = load_image(image)
        except Exception as exc:
            raise ValueError(f"Failed to load image from input source: {exc}") from exc
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Input must be a numpy.ndarray, file path, or bytes, got {type(image)}.")

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}.")

    if arr.size == 0:
        raise ValueError("Input image is empty (size is 0).")

    if arr.ndim not in (2, 3):
        raise ValueError(f"Input image must be 2D or 3D, got {arr.ndim}D with shape {arr.shape}.")

    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Image has invalid dimensions: height={h}, width={w}.")

    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels not in (1, 3, 4):
            raise ValueError(
                f"3D image must have 1, 3, or 4 channels, got {channels} channels."
            )

    # Check for NaN or Inf in floating-point representations
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            raise ValueError("Input image contains NaN values.")
        if np.isinf(arr).any():
            raise ValueError("Input image contains infinite (Inf) values.")

    return arr


def convert_to_grayscale(
    image: np.ndarray,
    is_rgb: bool = False
) -> np.ndarray:
    """Convert input image into a single-channel 2D array.

    Grayscale Handling:
    - If input is 2D, it is returned directly.
    - If input is 3D with 1 channel, it is squeezed to 2D.
    - If input is 3D with 3 channels:
      - Default assumed color order is BGR (standard OpenCV format).
      - If `is_rgb=True` (e.g. from PIL/Matplotlib/Web uploads), cv2.COLOR_RGB2GRAY is used.
    - If input is 3D with 4 channels (BGRA / RGBA), alpha is stripped during conversion.

    Args:
        image: Validated 2D or 3D NumPy array.
        is_rgb: Flag indicating whether 3-channel input is RGB rather than BGR.

    Returns:
        np.ndarray: 2D single-channel image.
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        elif channels == 3:
            if is_rgb:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            if is_rgb:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unable to convert image of shape {image.shape} to grayscale.")


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale intensity to a predictable 0–255 uint8 range.

    Uses cv2.NORM_MINMAX to linearly rescale pixel intensities across the full
    8-bit dynamic range, preventing clipping and contrast loss.

    Args:
        image: Single-channel 2D image (uint8, uint16, float, etc.).

    Returns:
        np.ndarray: 2D uint8 image with intensity normalized to [0, 255].
    """
    # If the image is completely uniform (all pixels identical), avoid division by zero
    min_val, max_val = float(np.min(image)), float(np.max(image))
    if min_val == max_val:
        return np.full(image.shape, fill_value=int(np.clip(min_val, 0, 255)), dtype=np.uint8)

    normalized = cv2.normalize(
        image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )
    return normalized


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Enhance local fluoroscopy contrast via CLAHE.

    Contrast Limited Adaptive Histogram Equalization operates on local contextual
    tiles rather than the global histogram. This reveals faint coronary vessels
    embedded within uneven fluoroscopic illumination while limiting noise over-amplification.

    Args:
        image: Single-channel uint8 grayscale image.
        clip_limit: Threshold for contrast limiting (default 2.0).
        tile_grid_size: Dimensions of local contextual grid tiles (default 8x8).

    Returns:
        np.ndarray: CLAHE contrast-enhanced 2D uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_light_denoising(
    image: np.ndarray,
    method: str = "gaussian",
    kernel_size: Tuple[int, int] = (3, 3),
    sigma: float = 0.8,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 25.0,
    bilateral_sigma_space: float = 25.0,
) -> np.ndarray:
    """Apply subtle spatial smoothing to mitigate fluoroscopy quantum mottle.

    Uses a small kernel (default 3x3) to remove fine sensor noise while preserving
    thin coronary vessel margins and stenotic transitions.

    Args:
        image: Single-channel uint8 grayscale image.
        method: Denoising algorithm: 'gaussian', 'bilateral', or 'median'.
        kernel_size: Spatial filter kernel size (e.g. (3, 3)).
        sigma: Standard deviation for Gaussian kernel.
        bilateral_d: Diameter of pixel neighborhood for bilateral filtering.
        bilateral_sigma_color: Filter sigma in color space for bilateral filtering.
        bilateral_sigma_space: Filter sigma in coordinate space for bilateral filtering.

    Returns:
        np.ndarray: Denoised 2D uint8 image.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, kernel_size, sigma)
    elif method == "bilateral":
        return cv2.bilateralFilter(
            image,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        )
    elif method == "median":
        ksize = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        return cv2.medianBlur(image, ksize)
    else:
        return image


def apply_optional_resize(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """Optionally resize the image to a standardized resolution.

    Disabled by default in the MVP to guarantee user A/B coordinates correspond
    directly to pixel indices on the original angiogram frame.

    Args:
        image: Single-channel 2D image.
        target_size: (width, height) desired dimensions.

    Returns:
        np.ndarray: Resized 2D image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: Union[str, Path, bytes, np.ndarray],
    config: Optional[PreprocessingConfig] = None,
    is_rgb: bool = False
) -> np.ndarray:
    """Execute the full classical preprocessing pipeline on an angiogram image.

    Pipeline execution order:
    1. Input Validation -> verifies type, dimensions, non-empty, finite values.
    2. Grayscale Conversion -> converts BGR/RGB to 2D single-channel.
    3. Intensity Normalization -> rescales dynamic range to [0, 255] uint8.
    4. CLAHE -> enhances local vessel-to-background contrast.
    5. Light Denoising -> suppresses high-frequency noise with small 3x3 kernel.
    6. Optional Resizing -> preserved at original resolution unless explicitly enabled.

    Args:
        image: Input image (file path, raw bytes, or NumPy ndarray).
        config: Optional PreprocessingConfig override.
        is_rgb: Set to True if color NumPy array is in RGB order instead of BGR.

    Returns:
        np.ndarray: Cleaned, contrast-enhanced, 2-dimensional uint8 grayscale image.

    Raises:
        ValueError: If input is None, empty, or has invalid dimensions.
        TypeError: If input is not a supported type.
    """
    cfg = config or PreprocessingConfig()

    # Step 1: Input Validation
    validated = validate_input_image(image)

    # Step 2: Grayscale Conversion
    gray = convert_to_grayscale(validated, is_rgb=is_rgb)

    # Step 3: Intensity Normalization to 0-255 uint8
    normalized = normalize_intensity(gray)

    # Step 4: CLAHE Contrast Enhancement
    clahe_enhanced = apply_clahe(
        normalized,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid_size=cfg.clahe_tile_grid_size,
    )

    # Step 5: Light Denoising (3x3 Gaussian smoothing)
    if cfg.denoise_enabled:
        denoised = apply_light_denoising(
            clahe_enhanced,
            method=cfg.denoise_method,
            kernel_size=cfg.gaussian_kernel_size,
            sigma=cfg.gaussian_sigma,
            bilateral_d=cfg.bilateral_d,
            bilateral_sigma_color=cfg.bilateral_sigma_color,
            bilateral_sigma_space=cfg.bilateral_sigma_space,
        )
    else:
        denoised = clahe_enhanced

    # Step 6: Optional Resizing (disabled by default)
    if cfg.resize_enabled and cfg.target_size:
        final_image = apply_optional_resize(denoised, cfg.target_size)
    else:
        final_image = denoised

    return final_image
