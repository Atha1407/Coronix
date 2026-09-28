"""Vessel enhancement using classical computer vision filters.

Responsible for:
- Enhancing tubular vessel structures inside the ROI
- Classical CV only (morphological filtering, Frangi vesselness)
- Fast execution suitable for real-time <5s response
"""

from typing import Optional

import numpy as np

from ml.config import VesselEnhancementConfig


def apply_morphological_enhancement(
    roi: np.ndarray,
    kernel_size: tuple = (9, 9)
) -> np.ndarray:
    """Enhance tubular structures using morphological black-hat and top-hat operations.

    In invasive coronary angiography (ICA), vessels with radiopaque dye appear darker
    than background tissue. Black-hat extracts dark structures smaller than the kernel.

    Args:
        roi: 2D uint8 grayscale ROI image.
        kernel_size: Structuring element dimensions.

    Returns:
        np.ndarray: Enhanced 2D uint8 vessel contrast map.
    """
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        # Black-hat isolates dark tubular vessels on light background
        blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
        # Contrast stretch
        enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced.astype(np.uint8)
    except ImportError:
        return roi.copy()


def apply_frangi_enhancement(
    roi: np.ndarray,
    scale_range: tuple = (1.0, 4.0),
    scale_step: float = 0.5,
    beta: float = 0.5,
    c: float = 15.0
) -> np.ndarray:
    """Enhance vesselness using the multiscale Frangi vesselness filter.

    Computes eigenvalues of the Hessian matrix to highlight tubular structures.

    Args:
        roi: 2D uint8 grayscale ROI image.
        scale_range: (min_sigma, max_sigma) scales to evaluate.
        scale_step: Step size across scales.
        beta: Frangi correction constant.
        c: Frangi noise threshold constant.

    Returns:
        np.ndarray: Normalized 2D uint8 vesselness probability map.
    """
    try:
        from skimage.filters import frangi

        # Normalize ROI to [0, 1] float
        roi_float = roi.astype(np.float32) / 255.0
        sigmas = np.arange(scale_range[0], scale_range[1] + 1e-4, scale_step)
        vesselness = frangi(
            roi_float,
            sigmas=sigmas,
            beta=beta,
            gamma=c,
            black_ridges=True  # Vessels appear dark in ICA
        )
        norm_vesselness = (vesselness / (np.max(vesselness) + 1e-7) * 255.0)
        return np.clip(norm_vesselness, 0, 255).astype(np.uint8)
    except (ImportError, Exception):
        # Fall back to morphological enhancement if scikit-image is not installed
        return apply_morphological_enhancement(roi)


def enhance_vessels(
    roi: np.ndarray,
    config: Optional[VesselEnhancementConfig] = None
) -> np.ndarray:
    """Enhance vessel structures within the extracted ROI corridor.

    Args:
        roi: 2D uint8 grayscale ROI image.
        config: Optional VesselEnhancementConfig override.

    Returns:
        np.ndarray: Enhanced vessel feature map (uint8).
    """
    cfg = config or VesselEnhancementConfig()

    if roi is None or roi.size == 0:
        raise ValueError("Input ROI is empty.")

    # Ensure 2D uint8
    if roi.ndim == 3:
        roi = roi[:, :, 0]
    if roi.dtype != np.uint8:
        roi = np.clip(roi, 0, 255).astype(np.uint8)

    if cfg.method == "frangi":
        return apply_frangi_enhancement(
            roi,
            scale_range=cfg.frangi_scale_range,
            scale_step=cfg.frangi_scale_step,
            beta=cfg.frangi_beta,
            c=cfg.frangi_c,
        )
    elif cfg.method == "tophat":
        return apply_morphological_enhancement(
            roi,
            kernel_size=cfg.tophat_kernel_size,
        )
    else:
        return apply_morphological_enhancement(roi)


def validate_point_on_vessel(
    image: np.ndarray,
    point: tuple,
    search_radius: int = 16,
    vessel_threshold: float = 0.10,
) -> tuple:
    """Validate whether an (x, y) click coordinate is on or near a coronary vessel.

    Tolerates slight clicking inaccuracy by checking a search_radius around (x, y).
    Rejects clicks that are on empty background, chamber, collimator borders, or out-of-bounds regions.

    Args:
        image: 2D grayscale angiogram image (uint8).
        point: (x, y) coordinate tuple.
        search_radius: Radius in pixels around point to evaluate for vessel presence.
        vessel_threshold: Minimum vessel enhancement peak required in the neighborhood.

    Returns:
        tuple[bool, float, str]: (is_valid, vesselness_score, reason)
    """
    if image is None or image.size == 0:
        return False, 0.0, "Empty image provided"

    if image.ndim == 3:
        image = image[:, :, 0]

    h, w = image.shape[:2]
    x, y = int(round(point[0])), int(round(point[1]))

    # Bounds check
    if x < 0 or x >= w or y < 0 or y >= h:
        return False, 0.0, f"Coordinate ({x}, {y}) is outside image boundaries"

    # Compute multiscale Frangi vesselness
    from skimage.filters import frangi

    img_float = image.astype(np.float32) / 255.0
    vesselness = frangi(img_float, sigmas=[1.5, 2.5], black_ridges=True)

    # Evaluate neighborhood around click
    x1 = max(0, x - search_radius)
    x2 = min(w, x + search_radius + 1)
    y1 = max(0, y - search_radius)
    y2 = min(h, y + search_radius + 1)

    patch = vesselness[y1:y2, x1:x2]
    max_score = float(np.max(patch)) if patch.size > 0 else 0.0

    is_valid = max_score >= vessel_threshold
    reason = "Valid vessel region" if is_valid else "Clicked region does not contain a discernible vessel"

    return is_valid, max_score, reason
