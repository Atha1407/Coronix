"""Tests for end-to-end pipeline execution and fallback."""

import unittest
import numpy as np

from ml.pipeline.inference_pipeline import PipelineResult, run_pipeline


class TestInferencePipeline(unittest.TestCase):
    """Verify end-to-end orchestration and fallback behavior."""

    def setUp(self):
        self.image = np.full((512, 512), 128, dtype=np.uint8)
        # Add synthetic vessel line
        self.image[150:160, 50:350] = 50

    def test_run_pipeline_success(self):
        result = run_pipeline(
            self.image,
            point_a=(50, 155),
            point_b=(350, 155),
            include_base64=True,
        )
        self.assertIsInstance(result, PipelineResult)
        self.assertFalse(result.is_fallback)
        self.assertTrue(result.lesion_detected)
        self.assertIsNotNone(result.overlay_image_base64)
        self.assertEqual(result.severity_category, "50-70%")
        self.assertEqual(result.cadica_id, "p50_70")

    def test_run_pipeline_fallback_on_invalid_points(self):
        # Invalid points (same coordinates) triggers fallback mode rather than crash
        result = run_pipeline(
            self.image,
            point_a=(100, 100),
            point_b=(100, 100),
            include_base64=True,
        )
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.severity_category, "50-70%")


if __name__ == "__main__":
    unittest.main()
