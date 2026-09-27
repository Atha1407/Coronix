"""Tests for vessel enhancement module."""

import unittest
import numpy as np

from ml.vessel.vessel_enhancement import (
    apply_frangi_enhancement,
    apply_morphological_enhancement,
    enhance_vessels,
)


class TestVesselEnhancement(unittest.TestCase):
    """Verify classical CV vessel enhancement filters."""

    def setUp(self):
        # 40x120 dummy corridor strip
        self.roi = np.full((40, 120), 128, dtype=np.uint8)
        # Add synthetic dark line down the middle (vessel with contrast agent)
        self.roi[18:22, :] = 40

    def test_morphological_enhancement(self):
        enhanced = apply_morphological_enhancement(self.roi)
        self.assertEqual(enhanced.shape, (40, 120))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_frangi_enhancement(self):
        enhanced = apply_frangi_enhancement(self.roi)
        self.assertEqual(enhanced.shape, (40, 120))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_enhance_vessels_wrapper(self):
        enhanced = enhance_vessels(self.roi)
        self.assertEqual(enhanced.shape, (40, 120))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_empty_roi_raises_error(self):
        empty_roi = np.array([])
        with self.assertRaises(ValueError):
            enhance_vessels(empty_roi)


if __name__ == "__main__":
    unittest.main()
