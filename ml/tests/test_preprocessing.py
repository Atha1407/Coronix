"""Comprehensive unit tests for the image preprocessing module.

Quality Checks Verified:
1. Output is not None.
2. Output is a NumPy array.
3. Output is 2-dimensional.
4. Output dtype is uint8.
5. Pixel values are within [0, 255].
6. Image height and width are preserved by default.
7. No NaN values exist.
8. No infinite values exist.
9. Function works with 2D grayscale input.
10. Function works with 3D BGR input.
11. Function works with 3D RGB input.
12. Function handles invalid inputs cleanly (None, empty array, 1D, 4D, NaN, Inf).

NOTE:
If real CADICA angiogram images are placed in ml/data/demo/ or ml/data/raw/,
the test automatically evaluates them. Otherwise, it explicitly reports that no sample
angiography image is available and runs software-level functionality tests using synthetic arrays.
"""

from pathlib import Path
import unittest

import numpy as np

from ml.config import PreprocessingConfig
from ml.preprocessing.image_preprocessing import (
    apply_clahe,
    apply_light_denoising,
    apply_optional_resize,
    convert_to_grayscale,
    normalize_intensity,
    preprocess_image,
    validate_input_image,
)


class TestImagePreprocessing(unittest.TestCase):
    """Quality and functional test suite for angiogram preprocessing."""

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
            print(f"\n[INFO] Found {len(cls.real_images)} actual angiography image(s) for testing.")

    def setUp(self):
        # Synthetic test arrays labeled strictly for software testing
        self.synth_h, self.synth_w = 480, 640
        self.software_test_gray = np.random.randint(
            40, 210, (self.synth_h, self.synth_w), dtype=np.uint8
        )
        self.software_test_bgr = np.repeat(
            self.software_test_gray[:, :, np.newaxis], 3, axis=2
        )
        self.software_test_rgb = self.software_test_bgr.copy()

    # --------------------------------------------------------------------------
    # Check 1 - 8: Output validity, type, shape, dtype, ranges, finite values
    # --------------------------------------------------------------------------
    def test_output_properties_and_dimensions_preserved(self):
        """Verify output is non-None, ndarray, 2D, uint8, [0, 255], and preserves dimensions."""
        output = preprocess_image(self.software_test_gray)

        # 1. Output is not None
        self.assertIsNotNone(output)

        # 2. Output is a NumPy array
        self.assertIsInstance(output, np.ndarray)

        # 3. Output is 2-dimensional
        self.assertEqual(output.ndim, 2)

        # 4. Output dtype is uint8
        self.assertEqual(output.dtype, np.uint8)

        # 5. Pixel values are between 0 and 255
        self.assertGreaterEqual(int(output.min()), 0)
        self.assertLessEqual(int(output.max()), 255)

        # 6. Image height and width are preserved by default
        self.assertEqual(output.shape, (self.synth_h, self.synth_w))

        # 7. No NaN values exist
        self.assertFalse(np.isnan(output).any())

        # 8. No infinite values exist
        self.assertFalse(np.isinf(output).any())

    # --------------------------------------------------------------------------
    # Check 9 - 11: Grayscale, BGR, RGB inputs
    # --------------------------------------------------------------------------
    def test_grayscale_input_handling(self):
        """Verify pipeline handles 2D grayscale input."""
        output = preprocess_image(self.software_test_gray)
        self.assertEqual(output.shape, (self.synth_h, self.synth_w))
        self.assertEqual(output.dtype, np.uint8)

    def test_bgr_input_handling(self):
        """Verify pipeline converts 3D BGR input to 2D uint8."""
        output = preprocess_image(self.software_test_bgr, is_rgb=False)
        self.assertEqual(output.shape, (self.synth_h, self.synth_w))
        self.assertEqual(output.dtype, np.uint8)

    def test_rgb_input_handling(self):
        """Verify pipeline converts 3D RGB input to 2D uint8."""
        output = preprocess_image(self.software_test_rgb, is_rgb=True)
        self.assertEqual(output.shape, (self.synth_h, self.synth_w))
        self.assertEqual(output.dtype, np.uint8)

    # --------------------------------------------------------------------------
    # Check 11: Invalid input error handling
    # --------------------------------------------------------------------------
    def test_invalid_input_none_raises_value_error(self):
        """Verify None input raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_image(None)

    def test_invalid_input_empty_array_raises_value_error(self):
        """Verify empty numpy array raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_image(np.array([]))

    def test_invalid_input_1d_array_raises_value_error(self):
        """Verify 1D array raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_image(np.array([1, 2, 3, 4]))

    def test_invalid_input_4d_array_raises_value_error(self):
        """Verify 4D array raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((1, 512, 512, 3), dtype=np.uint8))

    def test_invalid_input_type_raises_type_error(self):
        """Verify unsupported types raise TypeError."""
        with self.assertRaises(TypeError):
            validate_input_image(12345)

    def test_invalid_input_nan_raises_value_error(self):
        """Verify floating-point arrays with NaNs raise ValueError."""
        nan_array = np.full((100, 100), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_input_image(nan_array)

    def test_invalid_input_inf_raises_value_error(self):
        """Verify floating-point arrays with Inf raise ValueError."""
        inf_array = np.full((100, 100), np.inf, dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_input_image(inf_array)

    # --------------------------------------------------------------------------
    # Step-by-Step Individual Unit Checks
    # --------------------------------------------------------------------------
    def test_normalization_dynamic_range(self):
        """Verify intensity normalization scales narrow dynamic range to 0-255."""
        narrow = np.full((100, 100), 50, dtype=np.uint8)
        narrow[40:60, 40:60] = 70
        norm = normalize_intensity(narrow)
        self.assertEqual(norm.dtype, np.uint8)
        self.assertEqual(int(norm.min()), 0)
        self.assertEqual(int(norm.max()), 255)

    def test_clahe_contrast_enhancement(self):
        """Verify CLAHE produces valid 2D uint8 output."""
        clahe_out = apply_clahe(self.software_test_gray, clip_limit=2.0, tile_grid_size=(8, 8))
        self.assertEqual(clahe_out.shape, self.software_test_gray.shape)
        self.assertEqual(clahe_out.dtype, np.uint8)

    def test_light_denoising_preserves_shape(self):
        """Verify light 3x3 Gaussian denoising preserves spatial dimensions."""
        denoised = apply_light_denoising(
            self.software_test_gray, method="gaussian", kernel_size=(3, 3), sigma=0.8
        )
        self.assertEqual(denoised.shape, self.software_test_gray.shape)
        self.assertEqual(denoised.dtype, np.uint8)

    def test_optional_resize_only_when_enabled(self):
        """Verify resizing is only triggered when explicitly enabled in config."""
        # By default, disabled:
        default_cfg = PreprocessingConfig()
        out_default = preprocess_image(self.software_test_gray, config=default_cfg)
        self.assertEqual(out_default.shape, (self.synth_h, self.synth_w))

        # Explicitly enabled with custom size:
        resize_cfg = PreprocessingConfig(resize_enabled=True, target_size=(256, 256))
        out_resized = preprocess_image(self.software_test_gray, config=resize_cfg)
        self.assertEqual(out_resized.shape, (256, 256))

    # --------------------------------------------------------------------------
    # Real Image Test (if available in ml/data/demo or ml/data/raw)
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
                result = preprocess_image(str(img_path))
                self.assertIsNotNone(result)
                self.assertEqual(result.ndim, 2)
                self.assertEqual(result.dtype, np.uint8)
                self.assertFalse(np.isnan(result).any())


if __name__ == "__main__":
    unittest.main()
