"""Comprehensive test suite for CADICA dataset parsing, splitting, and PyTorch dataset loading."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.data.cadica_dataset import CADICA_CATEGORY_TO_LABEL_ID, CADICADataset
from ml.data.cadica_parser import (
    VALID_CADICA_CATEGORIES,
    CADICALesionRecord,
    parse_cadica_dataset,
    validate_bounding_box,
)
from ml.data.prepare_cadica import perform_patient_level_split
from ml.data.transforms import get_cadica_transforms
from ml.data.visualize_annotations import generate_sample_visualizations


class TestCADICADataset(unittest.TestCase):
    """Quality and functional tests for CADICA parsing, splitting, dataset loading, and transforms."""

    @classmethod
    def setUpClass(cls):
        cls.raw_cadica_dir = Path("ml/data/raw/CADICA")
        cls.processed_cadica_dir = Path("ml/data/processed/cadica")
        cls.has_raw_cadica = cls.raw_cadica_dir.is_dir()
        cls.has_processed_cadica = cls.processed_cadica_dir.is_dir() and (cls.processed_cadica_dir / "manifest.csv").is_file()

    def test_01_dataset_path_discovery(self):
        """Verify discovery of raw and processed CADICA directories."""
        self.assertTrue(self.has_raw_cadica, "Raw CADICA directory not found at ml/data/raw/CADICA")
        self.assertTrue((self.raw_cadica_dir / "selectedVideos").is_dir())
        self.assertTrue((self.raw_cadica_dir / "metadata.xlsx").is_file())

    def test_02_annotation_parsing_and_counts(self):
        """Verify parser extracts exact known CADICA figures (6126 keyframes, 6161 lesions)."""
        if not self.has_raw_cadica:
            self.skipTest("Raw CADICA not present.")

        records, warnings = parse_cadica_dataset(self.raw_cadica_dir)
        self.assertEqual(len(records), 8291, "Expected 8,291 manifest records (including multi-lesion rows)")

        unique_frames = set(r.frame_id for r in records)
        self.assertEqual(len(unique_frames), 6126, "Expected exactly 6,126 unique keyframes")

        lesion_recs = [r for r in records if r.is_lesion]
        non_lesion_recs = [r for r in records if not r.is_lesion]
        self.assertEqual(len(lesion_recs), 6161, "Expected exactly 6,161 lesion annotations")
        self.assertEqual(len(non_lesion_recs), 2130, "Expected exactly 2,130 non-lesion frames")

    def test_03_bounding_box_validation(self):
        """Verify spatial bounding box boundary validator."""
        # Valid box
        is_valid, msg = validate_bounding_box(10, 20, 100, 80, 512, 512)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

        # Out-of-bounds box (x + w > 512)
        is_valid, msg = validate_bounding_box(500, 20, 30, 80, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("x + width", msg)

        # Negative coordinate
        is_valid, msg = validate_bounding_box(-5, 20, 30, 80, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("x (-5) < 0", msg)

        # Non-positive dimension
        is_valid, msg = validate_bounding_box(10, 20, 0, 80, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("width (0) <= 0", msg)

    def test_04_severity_category_validation(self):
        """Verify severity category set matches CADICA 7 categories."""
        expected = {"p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"}
        self.assertEqual(VALID_CADICA_CATEGORIES, expected)
        self.assertEqual(len(CADICA_CATEGORY_TO_LABEL_ID), 7)

    def test_05_multiple_lesions_in_one_image(self):
        """Verify that multi-lesion images preserve all distinct lesion boxes."""
        if not self.has_raw_cadica:
            self.skipTest("Raw CADICA not present.")

        records, _ = parse_cadica_dataset(self.raw_cadica_dir)
        frame_to_records = {}
        for r in records:
            frame_to_records.setdefault(r.frame_id, []).append(r)

        multi_lesion_frames = [f_id for f_id, recs in frame_to_records.items() if len(recs) > 1]
        self.assertGreater(len(multi_lesion_frames), 0)

        # Check a known multi-lesion frame e.g. in p10 or p12
        sample_multi = frame_to_records[multi_lesion_frames[0]]
        self.assertGreaterEqual(len(sample_multi), 2)
        # Verify each lesion box has distinct coordinates or valid dimensions
        for r in sample_multi:
            self.assertTrue(r.is_lesion)
            self.assertIsNotNone(r.bbox_x)
            self.assertGreater(r.bbox_width, 0)

    def test_06_non_lesion_image_handling(self):
        """Verify non-lesion frames have empty bbox coordinates and is_lesion=False."""
        if not self.has_raw_cadica:
            self.skipTest("Raw CADICA not present.")

        records, _ = parse_cadica_dataset(self.raw_cadica_dir)
        non_lesions = [r for r in records if not r.is_lesion]
        self.assertEqual(len(non_lesions), 2130)
        for nl in non_lesions[:50]:
            self.assertIsNone(nl.bbox_x)
            self.assertIsNone(nl.bbox_y)
            self.assertIsNone(nl.bbox_width)
            self.assertIsNone(nl.bbox_height)
            self.assertIsNone(nl.severity_category)
            self.assertFalse(nl.is_lesion)

    def test_07_patient_level_split_and_zero_overlap(self):
        """Verify strict patient-level split with zero patient leakage across train/val/test."""
        if not self.has_raw_cadica:
            self.skipTest("Raw CADICA not present.")

        records, _ = parse_cadica_dataset(self.raw_cadica_dir)
        patient_to_split, train_pts, val_pts, test_pts = perform_patient_level_split(records, seed=42)

        self.assertEqual(len(train_pts), 30)
        self.assertEqual(len(val_pts), 6)
        self.assertEqual(len(test_pts), 6)
        self.assertEqual(len(train_pts) + len(val_pts) + len(test_pts), 42)

        # Strict set intersection check
        self.assertEqual(len(train_pts.intersection(val_pts)), 0, "Data leakage between train and val!")
        self.assertEqual(len(train_pts.intersection(test_pts)), 0, "Data leakage between train and test!")
        self.assertEqual(len(val_pts.intersection(test_pts)), 0, "Data leakage between val and test!")

        # Verify rare class representation across all splits
        p100_in_train = any(r.severity_category == "p100" for r in records if r.patient_id in train_pts and r.is_lesion)
        p100_in_val = any(r.severity_category == "p100" for r in records if r.patient_id in val_pts and r.is_lesion)
        p100_in_test = any(r.severity_category == "p100" for r in records if r.patient_id in test_pts and r.is_lesion)
        self.assertTrue(p100_in_train, "p100 missing from train split!")
        self.assertTrue(p100_in_val, "p100 missing from val split!")
        self.assertTrue(p100_in_test, "p100 missing from test split!")

    def test_08_manifest_generation_files(self):
        """Verify generated CSV manifests and class distribution JSON exist and are valid."""
        if not self.has_processed_cadica:
            self.skipTest("Processed CADICA files not generated yet.")

        manifest_file = self.processed_cadica_dir / "manifest.csv"
        train_file = self.processed_cadica_dir / "train.csv"
        val_file = self.processed_cadica_dir / "val.csv"
        test_file = self.processed_cadica_dir / "test.csv"
        json_file = self.processed_cadica_dir / "class_distribution.json"

        self.assertTrue(manifest_file.is_file())
        self.assertTrue(train_file.is_file())
        self.assertTrue(val_file.is_file())
        self.assertTrue(test_file.is_file())
        self.assertTrue(json_file.is_file())

        # Validate JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total"]["unique_frames"], 6126)
        self.assertEqual(data["total"]["lesion_annotations"], 6161)
        self.assertEqual(data["total"]["non_lesion_frames"], 2130)

    def test_09_pytorch_dataset_loading(self):
        """Verify CADICADataset loads images and builds torchvision-style targets."""
        if not self.has_processed_cadica:
            self.skipTest("Processed CADICA manifest not available.")

        manifest_path = self.processed_cadica_dir / "manifest.csv"
        ds = CADICADataset(
            manifest_path=manifest_path,
            include_non_lesions=True,
            color_mode="RGB"
        )
        self.assertEqual(len(ds), 6126)

        # Test sample lesion item
        img, target = ds[0]
        self.assertIsNotNone(img)
        self.assertIn("boxes", target)
        self.assertIn("labels", target)
        self.assertIn("patient_id", target)
        self.assertIn("frame_id", target)

    def test_10_target_dictionary_structure(self):
        """Verify target dictionary keys and shapes conform to torchvision object detector contract."""
        if not self.has_processed_cadica:
            self.skipTest("Processed CADICA manifest not available.")

        train_path = self.processed_cadica_dir / "train.csv"
        ds_train = CADICADataset(manifest_path=train_path, color_mode="RGB")

        # Find a lesion item and a non-lesion item
        lesion_idx = next(i for i, f in enumerate(ds_train.frames) if f["is_lesion"])
        non_lesion_idx = next(i for i, f in enumerate(ds_train.frames) if not f["is_lesion"])

        # 1. Lesion item
        img, target = ds_train[lesion_idx]
        self.assertGreater(len(target["boxes"]), 0)
        self.assertEqual(target["boxes"].shape[-1], 4)
        self.assertEqual(len(target["labels"]), len(target["boxes"]))
        self.assertTrue(target["is_lesion"])

        # 2. Non-lesion item (empty target tensors)
        img_nl, target_nl = ds_train[non_lesion_idx]
        self.assertEqual(len(target_nl["boxes"]), 0)
        self.assertEqual(len(target_nl["labels"]), 0)
        self.assertFalse(target_nl["is_lesion"])

    def test_11_transforms_pipeline(self):
        """Verify data transformation and conservative augmentation pipeline."""
        transforms = get_cadica_transforms(is_train=True, target_size=(512, 512), enable_flips=True)

        # Mock image and target
        dummy_img = np.full((512, 512, 3), 128, dtype=np.uint8)
        dummy_target = {
            "boxes": np.array([[100.0, 150.0, 160.0, 210.0]], dtype=np.float32),
            "labels": np.array([3], dtype=np.int64)
        }

        trans_img, trans_target = transforms(dummy_img, dummy_target)
        self.assertIsNotNone(trans_img)
        self.assertEqual(trans_target["boxes"].shape, (1, 4))

    def test_12_visualization_generation(self):
        """Verify that sample visualization files were generated properly."""
        if not self.has_processed_cadica:
            self.skipTest("Processed CADICA files not available.")

        vis_dir = self.processed_cadica_dir / "visualizations"
        self.assertTrue(vis_dir.is_dir())
        rendered = list(vis_dir.glob("*.png"))
        self.assertGreaterEqual(len(rendered), 8, "Expected at least 8 sample visualizations")


if __name__ == "__main__":
    unittest.main()
