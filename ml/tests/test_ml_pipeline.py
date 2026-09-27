"""Fast unit tests for catheter calibration, physical measurements, and ML pipeline."""

import pytest
import numpy as np

from ml.quantification.calibration import (
    calculate_mm_per_pixel,
    fr_to_mm,
    pixels_to_mm,
)
from ml.quantification.measurements import (
    calculate_image_derived_stenosis_percent,
    compute_bbox_dimensions_px,
    compute_physical_lesion_measurements,
)
from ml.pipeline.ml_pipeline import analyze_image


class TestCatheterCalibration:
    """Verify French-to-mm and physical scale factor calculations."""

    def test_fr_to_mm_conversion(self):
        # 6 Fr = 6 * 0.333 = 1.998 mm
        assert pytest.approx(fr_to_mm(6.0), 0.001) == 1.998
        # 5 Fr = 5 * 0.333 = 1.665 mm
        assert pytest.approx(fr_to_mm(5.0), 0.001) == 1.665
        # 7 Fr = 7 * 0.333 = 2.331 mm
        assert pytest.approx(fr_to_mm(7.0), 0.001) == 2.331

    def test_calculate_mm_per_pixel(self):
        # 6 Fr (1.998 mm) over 20 pixels -> 0.0999 mm/px
        mm_px = calculate_mm_per_pixel(catheter_fr=6.0, catheter_diameter_px=20.0)
        assert pytest.approx(mm_px, 0.0001) == 0.0999

    def test_pixels_to_mm_conversion(self):
        mm_per_pixel = 0.1  # 0.1 mm/px
        assert pytest.approx(pixels_to_mm(50.0, mm_per_pixel), 0.01) == 5.0
        assert pytest.approx(pixels_to_mm(0.0, mm_per_pixel), 0.01) == 0.0

    def test_invalid_calibration_inputs(self):
        with pytest.raises(ValueError):
            fr_to_mm(0)
        with pytest.raises(ValueError):
            fr_to_mm(-5.0)
        with pytest.raises(ValueError):
            calculate_mm_per_pixel(catheter_fr=0, catheter_diameter_px=20)
        with pytest.raises(ValueError):
            calculate_mm_per_pixel(catheter_fr=6.0, catheter_diameter_px=0)
        with pytest.raises(ValueError):
            calculate_mm_per_pixel(catheter_fr=6.0, catheter_diameter_px=-10)
        with pytest.raises(ValueError):
            pixels_to_mm(-5.0, 0.1)
        with pytest.raises(ValueError):
            pixels_to_mm(10.0, 0)


class TestPhysicalMeasurements:
    """Verify lesion bounding box dimensions and image-derived stenosis formula."""

    def test_bbox_dimensions_px(self):
        bbox = (100, 150, 40, 30)
        dims = compute_bbox_dimensions_px(bbox)
        assert dims["width_px"] == 40.0
        assert dims["height_px"] == 30.0
        # diagonal length = sqrt(40^2 + 30^2) = 50.0
        assert dims["length_px"] == 50.0

    def test_stenosis_formula_calculation(self):
        # MLD = 2.0, RVD = 10.0 -> (1 - 2/10) * 100 = 80.0%
        stenosis = calculate_image_derived_stenosis_percent(min_diameter_px=2.0, reference_diameter_px=10.0)
        assert stenosis == 80.0

        # Normal vessel (MLD == RVD) -> 0%
        assert calculate_image_derived_stenosis_percent(min_diameter_px=10.0, reference_diameter_px=10.0) == 0.0

        # Total occlusion (MLD == 0) -> 100%
        assert calculate_image_derived_stenosis_percent(min_diameter_px=0.0, reference_diameter_px=10.0) == 100.0

    def test_stenosis_clamping_and_safeguards(self):
        # Ectatic / aneurysm (MLD > RVD) clamps to 0.0%
        assert calculate_image_derived_stenosis_percent(min_diameter_px=15.0, reference_diameter_px=10.0) == 0.0

        # Invalid or missing inputs return None
        assert calculate_image_derived_stenosis_percent(None, 10.0) is None
        assert calculate_image_derived_stenosis_percent(5.0, None) is None
        assert calculate_image_derived_stenosis_percent(5.0, 0.0) is None
        assert calculate_image_derived_stenosis_percent(5.0, -10.0) is None

    def test_compute_physical_lesion_measurements(self):
        bbox = (50, 50, 20, 20)
        mm_px = 0.1  # 0.1 mm/px
        meas = compute_physical_lesion_measurements(bbox=bbox, mm_per_pixel=mm_px)
        assert pytest.approx(meas["lesion_width_mm"], 0.01) == 2.0
        assert pytest.approx(meas["lesion_height_mm"], 0.01) == 2.0
        assert meas["image_derived_stenosis_percent"] is None  # Reference diameter not provided


class TestMLPipelineImportsAndStructure:
    """Verify pipeline module imports and output dictionary structure."""

    def test_pipeline_import_and_signature(self):
        import ml.pipeline.ml_pipeline as ml_pipe
        assert hasattr(ml_pipe, "analyze_image")
        assert callable(ml_pipe.analyze_image)

    def test_invalid_calibration_input_returns_structured_error(self):
        # Synthetic 512x512 image
        img = np.full((512, 512), 128, dtype=np.uint8)
        res = analyze_image(
            image_path=img,
            point_a=(100, 100),
            point_b=(200, 200),
            catheter_fr=-1.0,  # Invalid Fr
            catheter_diameter_px=20.0,
        )

        assert isinstance(res, dict)
        assert res["valid_points"] is False
        assert "error" in res
        assert res["error"] is not None
        assert "catheter" in res["error"].lower()
        # Verify all required schema keys are present
        required_keys = [
            "valid_points", "point_a", "point_b", "catheter_fr",
            "catheter_diameter_mm", "catheter_diameter_px", "mm_per_pixel",
            "lesion_detected", "bbox", "severity", "confidence",
            "lesion_width_mm", "lesion_height_mm", "lesion_length_mm",
            "image_derived_stenosis_percent", "error"
        ]
        for key in required_keys:
            assert key in res

    def test_invalid_vessel_point_returns_structured_result(self):
        # Blank image where points are on uniform background (fails vesselness)
        img = np.full((512, 512), 128, dtype=np.uint8)
        res = analyze_image(
            image_path=img,
            point_a=(50, 50),
            point_b=(150, 150),
            catheter_fr=6.0,
            catheter_diameter_px=20.0,
        )

        assert isinstance(res, dict)
        assert res["valid_points"] is False
        assert res["lesion_detected"] is False
        assert res["error"] is not None
