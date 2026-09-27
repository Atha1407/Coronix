"""Unit tests for the CADICA dataset inspection and parsing utility."""

import tempfile
import unittest
from pathlib import Path

from ml.data.inspect_cadica import (
    BoundingBox,
    inspect_cadica_dataset,
    parse_annotation_line,
)


class TestCADICAInspection(unittest.TestCase):
    """Verify CADICA dataset structure scanner and parser."""

    def test_parse_annotation_line_comma_separated(self):
        line = "120.5, 230.0, 45.0, 32.0, p50_70"
        box = parse_annotation_line(line)
        self.assertIsNotNone(box)
        self.assertEqual(box.x, 120.5)
        self.assertEqual(box.y, 230.0)
        self.assertEqual(box.w, 45.0)
        self.assertEqual(box.h, 32.0)
        self.assertEqual(box.category, "p50_70")
        self.assertTrue(box.is_valid)

    def test_parse_annotation_line_space_separated(self):
        line = "100 200 40 30 p70_90"
        box = parse_annotation_line(line)
        self.assertIsNotNone(box)
        self.assertEqual(box.category, "p70_90")
        self.assertTrue(box.is_valid)

    def test_parse_annotation_invalid_negative_dimension(self):
        line = "100, 200, -10, 30, p50_70"
        box = parse_annotation_line(line)
        self.assertIsNotNone(box)
        self.assertFalse(box.is_valid)
        self.assertIn("Non-positive", box.error_msg)

    def test_inspect_cadica_nonexistent_dir(self):
        nonexistent = Path("nonexistent_dir_12345")
        report = inspect_cadica_dataset(nonexistent)
        self.assertFalse(report.is_present)

    def test_inspect_cadica_mock_structure(self):
        """Verify parser on a minimal synthetic CADICA directory tree."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            selected = root / "selectedVideos" / "p1" / "v1"
            gt_dir = selected / "groundtruth"
            input_dir = selected / "input"
            gt_dir.mkdir(parents=True)
            input_dir.mkdir(parents=True)

            # Create mock metadata
            (root / "metadata.xlsx").touch()

            # Create mock frame annotation
            annot_file = gt_dir / "p1_v1_00001.txt"
            annot_file.write_text("100, 150, 40, 30, p50_70\n200, 250, 50, 35, p90_98\n", encoding="utf-8")

            report = inspect_cadica_dataset(root)
            self.assertTrue(report.is_present)
            self.assertTrue(report.has_metadata_excel)
            self.assertTrue(report.selected_videos_dir_present)
            self.assertEqual(report.total_patients, 1)
            self.assertEqual(report.total_videos, 1)
            self.assertEqual(report.total_lesions_count, 2)
            self.assertEqual(report.category_distribution["p50_70"], 1)
            self.assertEqual(report.category_distribution["p90_98"], 1)


if __name__ == "__main__":
    unittest.main()
