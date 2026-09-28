import os
import io
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import cv2

from dicom_reader import read_dicom_file, DICOMProcessingError, extract_dicom_metadata


def create_synthetic_dicom(
    filepath: str,
    frames: int = 1,
    rows: int = 256,
    cols: int = 256,
    photometric: str = "MONOCHROME2"
):
    """Generates a valid DICOM file for testing."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "XA"  # X-Ray Angiography
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "123456"
    ds.Rows = rows
    ds.Columns = cols
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.25, 0.25]
    ds.ImagerPixelSpacing = [0.25, 0.25]
    ds.SliceThickness = 1.0
    ds.StudyDescription = "Coronary Angiogram Study"
    ds.SeriesDescription = "LAD View"
    ds.WindowCenter = 128
    ds.WindowWidth = 256
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    if frames > 1:
        ds.NumberOfFrames = frames
        # Synthetic vessel lines across frames
        data = np.zeros((frames, rows, cols), dtype=np.uint8)
        for i in range(frames):
            frame = np.full((rows, cols), 180, dtype=np.uint8)
            # Draw synthetic dark vessel moving across frame
            cv2.line(frame, (30 + i * 2, 40), (200, 200), 40, thickness=6)
            data[i] = frame
        ds.PixelData = data.tobytes()
    else:
        frame = np.full((rows, cols), 180, dtype=np.uint8)
        cv2.line(frame, (40, 40), (200, 200), 40, thickness=6)
        ds.PixelData = frame.tobytes()

    ds.save_as(filepath)
    print(f"Created synthetic DICOM: {filepath} ({frames} frame(s))")


def test_dicom_ingestion():
    single_path = "sample_single_frame.dcm"
    multi_path = "sample_multi_frame.dcm"

    create_synthetic_dicom(single_path, frames=1)
    create_synthetic_dicom(multi_path, frames=5)

    print("\n--- Testing Single-Frame DICOM ---")
    frame, meta, total_frames, b64 = read_dicom_file(single_path, frame_index=0)
    assert total_frames == 1, f"Expected 1 frame, got {total_frames}"
    assert frame.shape == (256, 256), f"Unexpected shape {frame.shape}"
    assert meta["Modality"] == "XA"
    assert meta["PixelDimensionMm"] == {"row_spacing_mm": 0.25, "col_spacing_mm": 0.25}
    assert b64.startswith("data:image/png;base64,")
    print("Single-frame test PASSED!")

    print("\n--- Testing Multi-Frame DICOM ---")
    frame_0, meta_0, total_frames_m, b64_0 = read_dicom_file(multi_path, frame_index=0)
    assert total_frames_m == 5, f"Expected 5 frames, got {total_frames_m}"
    frame_2, meta_2, _, b64_2 = read_dicom_file(multi_path, frame_index=2)
    assert meta_2["CurrentFrameIndex"] == 2
    assert not np.array_equal(frame_0, frame_2), "Frame 0 and Frame 2 should have moved synthetic vessel"
    print("Multi-frame test PASSED!")

    print("\n--- Testing Invalid Frame Index ---")
    try:
        read_dicom_file(multi_path, frame_index=99)
        assert False, "Should have raised DICOMProcessingError for index out of bounds"
    except DICOMProcessingError as e:
        print(f"Correctly caught out-of-bounds error: {e}")

    print("\n--- Testing Invalid File ---")
    try:
        read_dicom_file(b"This is not a dicom file at all")
        assert False, "Should have raised DICOMProcessingError for invalid file"
    except DICOMProcessingError as e:
        print(f"Correctly caught invalid DICOM error: {e}")

    print("\nALL DICOM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dicom_ingestion()
