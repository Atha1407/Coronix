"""Severity category mapper conforming strictly to CADICA's 7 discrete stenosis bins.

Responsible for:
- Mapping lesion analysis outputs to CADICA's 7 standardized categories:
  '<20%', '20-50%', '50-70%', '70-90%', '90-98%', '99%', '100%'
- Enforcing discrete clinical bins without outputting continuous or decimal stenosis percentages.
"""

from dataclasses import dataclass

from ml.config import (
    CADICA_CATEGORIES,
    CADICA_ID_TO_LABEL,
    LABEL_TO_CADICA_ID,
    SEVERITY_COLORS,
)
from ml.lesion.lesion_detector import LesionDetectionResult


@dataclass
class SeverityResult:
    """Structured severity output conforming to the CADICA taxonomy.

    Attributes:
        category: CADICA category string (e.g., '50-70%').
        cadica_id: CADICA dataset identifier (e.g., 'p50_70').
        color_hex: Hex code for UI badge display.
        description: Standard descriptive label.
    """
    category: str
    cadica_id: str
    color_hex: str
    description: str


SEVERITY_DESCRIPTIONS = {
    "<20%": "Minimal or no significant angiographic narrowing",
    "20-50%": "Mild stenosis",
    "50-70%": "Moderate stenosis",
    "70-90%": "Severe stenosis",
    "90-98%": "Critical stenosis",
    "99%": "Subtotal occlusion",
    "100%": "Total occlusion",
}


def map_severity(lesion_result: LesionDetectionResult) -> SeverityResult:
    """Map detected lesion measurements strictly to one of CADICA's 7 discrete categories.

    Rule-based mapping partitions narrowing ratios into CADICA's discrete clinical bins.
    Invented decimal percentages are strictly prohibited.

    Args:
        lesion_result: LesionDetectionResult containing detection flags and narrowing ratio.

    Returns:
        SeverityResult: Structured severity classification.
    """
    if not lesion_result.lesion_detected:
        cat = "<20%"
        cadica_id = "p0_20"
        return SeverityResult(
            category=cat,
            cadica_id=cadica_id,
            color_hex=SEVERITY_COLORS[cat],
            description=SEVERITY_DESCRIPTIONS[cat],
        )

    ratio = lesion_result.measured_narrowing_ratio

    if ratio < 0.20:
        cat = "<20%"
    elif ratio < 0.50:
        cat = "20-50%"
    elif ratio < 0.70:
        cat = "50-70%"
    elif ratio < 0.90:
        cat = "70-90%"
    elif ratio < 0.99:
        cat = "90-98%"
    elif ratio < 1.0:
        cat = "99%"
    else:
        cat = "100%"

    # Validate that category is strictly one of the 7 supported CADICA categories
    if cat not in CADICA_CATEGORIES:
        cat = "50-70%"

    cadica_id = LABEL_TO_CADICA_ID.get(cat, "p50_70")
    color = SEVERITY_COLORS.get(cat, "#eab308")
    desc = SEVERITY_DESCRIPTIONS.get(cat, "Stenosis")

    return SeverityResult(
        category=cat,
        cadica_id=cadica_id,
        color_hex=color,
        description=desc,
    )
