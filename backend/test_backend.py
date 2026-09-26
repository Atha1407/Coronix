import io
import os
import sys
from pathlib import Path

# Ensure paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opencv"))

from fastapi.testclient import TestClient
from main import app
from test_dicom import create_synthetic_dicom

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "MockLesionModel" in data["active_ml_model"]
    print("Health check PASSED!")


def test_dicom_upload_endpoint():
    dicom_file = "test_endpoint_dicom.dcm"
    create_synthetic_dicom(dicom_file, frames=3, rows=128, cols=128)

    with open(dicom_file, "rb") as f:
        file_bytes = f.read()

    # 1. Test Frame 0 Upload
    response = client.post(
        "/api/dicom/upload",
        files={"file": ("test_endpoint_dicom.dcm", file_bytes, "application/dicom")},
        data={"frame_index": 0}
    )
    assert response.status_code == 200, f"Upload failed: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert data["total_frames"] == 3
    assert data["frame_index"] == 0
    assert data["image_base64"].startswith("data:image/png;base64,")
    assert data["metadata"]["Modality"] == "XA"
    print("DICOM Upload Frame 0 PASSED!")

    # 2. Test Frame 2 Upload
    response = client.post(
        "/api/dicom/upload",
        files={"file": ("test_endpoint_dicom.dcm", file_bytes, "application/dicom")},
        data={"frame_index": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["frame_index"] == 2
    print("DICOM Upload Frame 2 PASSED!")

    # 3. Test Invalid Frame index
    response = client.post(
        "/api/dicom/upload",
        files={"file": ("test_endpoint_dicom.dcm", file_bytes, "application/dicom")},
        data={"frame_index": 10}
    )
    assert response.status_code == 422
    print("Invalid frame index check PASSED!")

    # Clean up test file
    if os.path.exists(dicom_file):
        os.remove(dicom_file)


def test_analyze_endpoint():
    dicom_file = "test_analyze_dicom.dcm"
    create_synthetic_dicom(dicom_file, frames=1, rows=256, cols=256)

    with open(dicom_file, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/analyze",
        files={"file": ("test_analyze_dicom.dcm", file_bytes, "application/dicom")},
        data={
            "point_a_x": 40.0,
            "point_a_y": 40.0,
            "point_b_x": 180.0,
            "point_b_y": 180.0,
            "catheter_size": 6.0,
            "frame_index": 0
        }
    )
    assert response.status_code == 200, f"Analyze failed: {response.text}"
    result = response.json()
    assert result["lesion_detected"] is True
    assert result["lesion_bbox"] is not None
    assert len(result["lesion_bbox"]) == 4
    assert result["catheter_size"] == 6.0
    assert result["severity_category"] == "70-90%"
    assert result["confidence"] > 0.8
    assert "cv_pipeline" in result
    print("Full Analyze Pipeline PASSED!")

    if os.path.exists(dicom_file):
        os.remove(dicom_file)


if __name__ == "__main__":
    test_api_health()
    test_dicom_upload_endpoint()
    test_analyze_endpoint()
    print("\nALL BACKEND API TESTS COMPLETED SUCCESSFULLY!")
