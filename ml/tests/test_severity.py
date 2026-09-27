"""Tests for CADICA 7-category severity mapper."""

import unittest

from ml.config import CADICA_CATEGORIES
from ml.lesion.lesion_detector import LesionDetectionResult
from ml.severity.severity_mapper import SeverityResult, map_severity


class TestSeverityMapper(unittest.TestCase):
    """Verify that severity mapper strictly complies with CADICA's 7 discrete categories."""

    def test_no_lesion_detected_maps_to_minimal(self):
        result = LesionDetectionResult(lesion_detected=False)
        severity = map_severity(result)
        self.assertIsInstance(severity, SeverityResult)
        self.assertEqual(severity.category, "<20%")
        self.assertEqual(severity.cadica_id, "p0_20")

    def test_cadica_taxonomy_bins(self):
        test_cases = [
            (0.10, "<20%", "p0_20"),
            (0.35, "20-50%", "p20_50"),
            (0.60, "50-70%", "p50_70"),
            (0.80, "70-90%", "p70_90"),
            (0.95, "90-98%", "p90_98"),
            (0.99, "99%", "p99"),
            (1.00, "100%", "p100"),
        ]

        for ratio, expected_label, expected_id in test_cases:
            with self.subTest(ratio=ratio):
                res = LesionDetectionResult(
                    lesion_detected=True,
                    measured_narrowing_ratio=ratio
                )
                severity = map_severity(res)
                self.assertEqual(severity.category, expected_label)
                self.assertEqual(severity.cadica_id, expected_id)
                self.assertIn(severity.category, CADICA_CATEGORIES)

    def test_never_returns_arbitrary_decimal_string(self):
        res = LesionDetectionResult(
            lesion_detected=True,
            measured_narrowing_ratio=0.6789
        )
        severity = map_severity(res)
        # Must be exact categorical string, never '67.89%'
        self.assertIn(severity.category, CADICA_CATEGORIES)
        self.assertNotIn("67", severity.category)


if __name__ == "__main__":
    unittest.main()
