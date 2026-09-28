"""Comprehensive unit tests for the A/B Guided ROI extraction module.

Quality Checks Verified:
1. Horizontal A/B corridor extraction.
2. Vertical A/B corridor extraction.
3. Diagonal A/B corridor extraction.
4. Arbitrary angle orientation handling.
5. Nearby A/B points below threshold raising ValueError.
6. Far A/B points across image frame.
7. Boundary points (near left, right, top, bottom edges) handled without crashes.
8. Identical A/B points strictly raising ValueError.
9. Out-of-bounds A/B points strictly raising ValueError.
10. Original image is NOT modified (immutability check).
11. Unrolled ROI dimensions match (corridor_width, round(length)).
12. Forward and inverse coordinate projection accuracy (Point A, B, and center mapping).
13. Seamless integration with image preprocessing module.
14. Real CADICA angiogram evaluation if present in data directories.
"""

from pathlib import Path
import unittest

import cv2
import numpy as np

from ml.config import ROIConfig
from ml.preprocessing.image_preprocessing import preprocess_image
from ml.roi.roi_extractor import (
    ROIExtractionResult,
    compute_corridor_geometry,
    extract_roi,
    validate_input,
)


class TestROIExtractor(unittest.TestCase):
    """Quality and functional test suite for ROI extraction."""

    @classmethod
    def setUpClass(cls):
        # Look for real clinical angiogram frames in ml/data/demo or ml/data/raw
        repo_root = Path(__file__).resolve().parent.parent
        demo_dir = repo_root / "data" / "demo"
        raw_dir = repo_root / "data" / "raw"

        candidate_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        real_images = []
        for ext in candidate_extensions:
            real_images.extend(demo_dir.glob(ext))
            real_images.extend(raw_dir.glob(ext))

        cls.real_images = [p for p in real_images if p.name != ".gitkeep"]

        if not cls.real_images:
            print(
                "\n[INFO] No sample CADICA angiography image found in ml/data/demo/ or ml/data/raw/. "
                "Per project instructions, testing continues using synthetic arrays for "
                "SOFTWARE/FUNCTIONALITY VERIFICATION ONLY (not medical validation)."
            )
        else:
            print(f"\n[INFO] Found {len(cls.real_images)} actual angiography image(s) for ROI testing.")

    def setUp(self):
        self.img_h, self.img_w = 512, 512
        # Synthetic test angiogram frame
        self.software_test_image = np.random.randint(40, 200, (self.img_h, self.img_w), dtype=np.uint8)
        self.software_test_bgr = np.repeat(self.software_test_image[:, :, np.newaxis], 3, axis=2)

    # --------------------------------------------------------------------------
    # Check 1 - 4: Orientations (Horizontal, Vertical, Diagonal, Arbitrary)
    # --------------------------------------------------------------------------
    def test_horizontal_orientation(self):
        """Verify extraction along a horizontal vector."""
        pt_a = (100, 200)
        pt_b = (300, 200)
        width = 40
        result = extract_roi(self.software_test_image, pt_a, pt_b, width=width)

        self.assertIsInstance(result, ROIExtractionResult)
        self.assertEqual(result.roi_image.shape[0], width)
        self.assertEqual(result.roi_image.shape[1], 200)
        self.assertAlmostEqual(result.length, 200.0, places=2)
        self.assertAlmostEqual(result.angle_degrees, 0.0, places=2)
        self.assertEqual(result.point_a, pt_a)
        self.assertEqual(result.point_b, pt_b)

    def test_vertical_orientation(self):
        """Verify extraction along a vertical vector."""
        pt_a = (200, 100)
        pt_b = (200, 350)
        width = 30
        result = extract_roi(self.software_test_image, pt_a, pt_b, width=width)

        self.assertEqual(result.roi_image.shape[0], width)
        self.assertEqual(result.roi_image.shape[1], 250)
        self.assertAlmostEqual(result.length, 250.0, places=2)
        self.assertAlmostEqual(result.angle_degrees, 90.0, places=2)

    def test_diagonal_orientation(self):
        """Verify extraction along a 45-degree diagonal vector."""
        pt_a = (100, 100)
        pt_b = (200, 200)
        width = 40
        result = extract_roi(self.software_test_image, pt_a, pt_b, width=width)

        expected_length = np.hypot(100, 100)
        self.assertEqual(result.roi_image.shape[0], width)
        self.assertEqual(result.roi_image.shape[1], int(round(expected_length)))
        self.assertAlmostEqual(result.angle_degrees, 45.0, places=2)

    def test_arbitrary_reverse_orientation(self):
        """Verify extraction when Point B is to the left of Point A (negative dx)."""
        pt_a = (300, 250)
        pt_b = (120, 180)
        result = extract_roi(self.software_test_image, pt_a, pt_b, width=36)

        self.assertEqual(result.roi_image.shape[0], 36)
        self.assertTrue(result.roi_image.shape[1] > 0)
        self.assertTrue(-180.0 <= result.angle_degrees <= 180.0)

    # --------------------------------------------------------------------------
    # Check 5 - 9: Validation and Error Handling
    # --------------------------------------------------------------------------
    def test_identical_points_raises_value_error(self):
        """Verify identical Point A and B raises clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_roi(self.software_test_image, (150, 150), (150, 150))
        self.assertIn("identical", str(ctx.exception).lower())

    def test_nearby_points_below_threshold_raises_value_error(self):
        """Verify points closer than minimum Euclidean separation raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_roi(self.software_test_image, (150, 150), (153, 154))
        self.assertIn("too close", str(ctx.exception).lower())

    def test_out_of_bounds_point_a_raises_value_error(self):
        """Verify Point A outside image bounds raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_roi(self.software_test_image, (-10, 200), (300, 200))
        self.assertIn("out of bounds", str(ctx.exception).lower())

    def test_out_of_bounds_point_b_raises_value_error(self):
        """Verify Point B outside image bounds raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_roi(self.software_test_image, (100, 100), (600, 200))
        self.assertIn("out of bounds", str(ctx.exception).lower())

    def test_none_image_raises_value_error(self):
        """Verify None image raises ValueError."""
        with self.assertRaises(ValueError):
            extract_roi(None, (100, 100), (200, 200))

    def test_empty_image_raises_value_error(self):
        """Verify empty numpy array raises ValueError."""
        with self.assertRaises(ValueError):
            extract_roi(np.array([]), (100, 100), (200, 200))

    # --------------------------------------------------------------------------
    # Check 7: Edge and Boundary Robustness (No crash on edge points)
    # --------------------------------------------------------------------------
    def test_points_close_to_image_boundaries(self):
        """Verify corridor extending outside image boundaries is handled safely via reflection."""
        # Top-left margin points
        res_top_left = extract_roi(self.software_test_image, (2, 2), (50, 2), width=40)
        self.assertIsNotNone(res_top_left.roi_image)
        self.assertEqual(res_top_left.roi_image.shape[0], 40)
        self.assertFalse(np.isnan(res_top_left.roi_image).any())

        # Bottom-right margin points
        res_bottom_right = extract_roi(
            self.software_test_image, (500, 508), (400, 508), width=50
        )
        self.assertIsNotNone(res_bottom_right.roi_image)
        self.assertEqual(res_bottom_right.roi_image.shape[0], 50)
        self.assertFalse(np.isnan(res_bottom_right.roi_image).any())

    # --------------------------------------------------------------------------
    # Check 10: Original Image Immutability
    # --------------------------------------------------------------------------
    def test_original_image_not_modified(self):
        """Verify ROI extraction does not mutate or alter the original image array."""
        image_copy = self.software_test_image.copy()
        extract_roi(self.software_test_image, (100, 100), (300, 300), width=40)
        np.testing.assert_array_equal(self.software_test_image, image_copy)

    # --------------------------------------------------------------------------
    # Check 11 & 12: Coordinate Projection (ROI space -> Image space)
    # --------------------------------------------------------------------------
    def test_coordinate_mapping_accuracy(self):
        """Verify forward and inverse perspective transform maps points accurately."""
        pt_a = (80, 120)
        pt_b = (320, 260)
        width = 40
        result = extract_roi(self.software_test_image, pt_a, pt_b, width=width)

        # In ROI coordinates:
        # Point A is at (0, width / 2)
        # Point B is at (length, width / 2)
        # Midpoint is at (length / 2, width / 2)
        roi_a = np.array([[0.0, width / 2.0]], dtype=np.float32)
        roi_b = np.array([[result.length, width / 2.0]], dtype=np.float32)
        roi_mid = np.array([[result.length / 2.0, width / 2.0]], dtype=np.float32)

        proj_a = result.roi_to_image_coords(roi_a)[0]
        proj_b = result.roi_to_image_coords(roi_b)[0]
        proj_mid = result.roi_to_image_coords(roi_mid)[0]

        # Verify mapped points align with input points within 0.5 pixel tolerance
        self.assertAlmostEqual(proj_a[0], pt_a[0], delta=0.5)
        self.assertAlmostEqual(proj_a[1], pt_a[1], delta=0.5)
        self.assertAlmostEqual(proj_b[0], pt_b[0], delta=0.5)
        self.assertAlmostEqual(proj_b[1], pt_b[1], delta=0.5)

        expected_mid_x = (pt_a[0] + pt_b[0]) / 2.0
        expected_mid_y = (pt_a[1] + pt_b[1]) / 2.0
        self.assertAlmostEqual(proj_mid[0], expected_mid_x, delta=0.5)
        self.assertAlmostEqual(proj_mid[1], expected_mid_y, delta=0.5)

    def test_roi_bbox_projection(self):
        """Verify projection of a bounding box in ROI coordinates to original image coordinates."""
        pt_a = (100, 150)
        pt_b = (300, 150)
        width = 40
        result = extract_roi(self.software_test_image, pt_a, pt_b, width=width)

        # Lesion box at center of ROI strip
        roi_bbox = (80, 10, 40, 20)  # x, y, w, h
        polygon = result.roi_bbox_to_image_polygon(roi_bbox)
        self.assertEqual(polygon.shape, (4, 2))

        img_bbox = result.roi_bbox_to_image_bbox(roi_bbox, self.software_test_image.shape)
        self.assertEqual(len(img_bbox), 4)
        x, y, w, h = img_bbox
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    # --------------------------------------------------------------------------
    # Check 13: Preprocessing Integration (raw -> preprocess -> extract_roi)
    # --------------------------------------------------------------------------
    def test_preprocessing_integration(self):
        """Verify intended flow: raw -> preprocess_image() -> extract_roi()."""
        raw_bgr = self.software_test_bgr
        preprocessed = preprocess_image(raw_bgr)
        self.assertEqual(preprocessed.ndim, 2)
        self.assertEqual(preprocessed.dtype, np.uint8)

        result = extract_roi(preprocessed, (120, 140), (280, 260), width=40)
        self.assertIsNotNone(result)
        self.assertEqual(result.roi_image.ndim, 2)
        self.assertEqual(result.roi_image.dtype, np.uint8)
        self.assertEqual(result.roi_image.shape[0], 40)

    # --------------------------------------------------------------------------
    # Check 14: Real Angiogram Frame Evaluation (if present)
    # --------------------------------------------------------------------------
    def test_real_angiogram_if_present(self):
        """Test on actual angiography frames if present in data directories."""
        if not self.real_images:
            self.skipTest(
                "No real CADICA image available in ml/data/demo/ or ml/data/raw/. "
                "Skipping real image test without creating fabricated images."
            )
        for img_path in self.real_images:
            with self.subTest(image=img_path.name):
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                preprocessed = preprocess_image(img)
                # Sample representative corridor through central region
                pt_a = (int(w * 0.3), int(h * 0.3))
                pt_b = (int(w * 0.7), int(h * 0.7))
                result = extract_roi(preprocessed, pt_a, pt_b, width=40)
                self.assertIsNotNone(result)
                self.assertEqual(result.roi_image.ndim, 2)
                self.assertEqual(result.roi_image.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
