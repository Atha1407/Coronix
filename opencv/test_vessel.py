import cv2

from preprocessing import preprocess_image
from roi import create_roi
from vessel_analysis import extract_vessel


# -------------------------
# 1. Preprocess
# -------------------------

original, enhanced = preprocess_image("sample.jpg")


# -------------------------
# 2. Temporary A/B points
# -------------------------

A = (88, 147)
B = (176, 73)


# -------------------------
# 3. Create ROI
# -------------------------

roi, mask = create_roi(
    enhanced,
    A,
    B,
    width=40
)


# -------------------------
# 4. Extract vessel
# -------------------------

vessel_mask = extract_vessel(
    enhanced,
    mask
)


# -------------------------
# 5. Save results
# -------------------------

cv2.imwrite(
    "vessel_mask.jpg",
    vessel_mask
)

print("Vessel analysis completed!")