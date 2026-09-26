import cv2
import numpy as np


def extract_vessel(enhanced, mask):
    """
    Extract a preliminary vessel mask inside the A-B ROI.
    """

    # 1. Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(enhanced)

    # 2. Angiographic vessels are generally darker than surroundings
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    # 3. Restrict detection to our A-B ROI
    binary = cv2.bitwise_and(
        binary,
        binary,
        mask=mask
    )

    # 4. Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # 5. Connect nearby vessel pixels
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return binary