"""Tests for lesion detection module interface."""

import unittest
import numpy as np

from ml.lesion.lesion_detector import LesionDetectionResult, detect_lesion


class TestLesionDetector(unittest.TestCase):
    """Verify lesion detector interface and return contracts."""

    def setUp(self):
        self.dummy_roi = np.full((40, 100), 128, dtype=np.uint8)

    def test_detect_lesion_returns_result_dataclass(self):
        result = detect_lesion(self.dummy_roi)
        self.assertIsInstance(result, LesionDetectionResult)
        self.assertTrue(result.lesion_detected)
        self.assertIsNotNone(result.roi_bbox)
        self.assertEqual(len(result.roi_bbox), 4)
        self.assertTrue(0.0 <= result.confidence <= 1.0)

    def test_empty_roi_returns_non_detection(self):
        empty_roi = np.array([])
        result = detect_lesion(empty_roi)
        self.assertFalse(result.lesion_detected)
        self.assertIsNone(result.roi_bbox)


if __name__ == "__main__":
    unittest.main()
