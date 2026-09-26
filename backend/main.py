import sys
from pathlib import Path

# Add backend and opencv to sys.path
BASE_DIR = Path(__file__).resolve().parent
OPENCV_DIR = BASE_DIR.parent / "opencv"
for d in [str(BASE_DIR), str(OPENCV_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dicom_reader import read_dicom_file, DICOMProcessingError
from preprocessing import run_preprocessing_pipeline
from pipeline import load_input_image_or_dicom, process_coronary_analysis
from ml_interface import get_lesion_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coronix-backend")

app = FastAPI(
    title="Coronix Angiography Lesion Analysis API",
    description="Backend API for DICOM ingestion, OpenCV preprocessing, and ML model interface for Coronix",
    version="1.0.0"
)

# CORS setup for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Coronix AI-Assisted Angiography Backend",
        "version": "1.0.0"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "active_ml_model": get_lesion_model().__class__.__name__
    }


@app.post("/api/dicom/upload")
async def upload_dicom(
    file: UploadFile = File(...),
    frame_index: int = Form(0)
):
    """
    Ingest a DICOM file (.dcm), extract metadata, decode specified frame,
    normalize to 8-bit uint8, and return base64 PNG data for frontend display.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        frame_uint8, metadata, total_frames, image_base64 = read_dicom_file(
            file_bytes,
            frame_index=frame_index
        )

        return {
            "success": True,
            "filename": file.filename,
            "frame_index": frame_index,
            "total_frames": total_frames,
            "metadata": metadata,
            "image_base64": image_base64
        }

    except DICOMProcessingError as e:
        logger.error(f"DICOM processing error for {file.filename}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error processing DICOM {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error processing DICOM: {str(e)}")


@app.post("/api/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    frame_index: int = Form(0)
):
    """
    Runs the OpenCV preprocessing pipeline on an uploaded DICOM or standard image.
    Grayscale -> Resize/standardize (512x512) -> Mild Denoise -> CLAHE
    """
    try:
        file_bytes = await file.read()
        frame, meta, total_frames, file_type, raw_b64 = load_input_image_or_dicom(
            file_bytes=file_bytes,
            filename=file.filename or "",
            frame_index=frame_index
        )

        prep = run_preprocessing_pipeline(frame, target_size=(512, 512))
        from dicom_reader import frame_to_base64_png
        enhanced_b64 = frame_to_base64_png(prep["enhanced_image"])

        return {
            "success": True,
            "filename": file.filename,
            "file_type": file_type,
            "total_frames": total_frames,
            "frame_index": frame_index,
            "metadata": meta,
            "enhanced_image_base64": enhanced_b64,
            "scale_factors": prep["scale_factors"],
            "original_shape": prep["original_shape"]
        }
    except Exception as e:
        logger.exception(f"Error in preprocessing endpoint: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    point_a_x: float = Form(...),
    point_a_y: float = Form(...),
    point_b_x: float = Form(...),
    point_b_y: float = Form(...),
    catheter_size: Optional[float] = Form(None),
    frame_index: int = Form(0)
):
    """
    Full pipeline execution:
    1. DICOM / Image reading
    2. OpenCV preprocessing
    3. Corridor ROI extraction between A and B
    4. Pass clean ROI to ML Model Interface
    5. Return analysis results to frontend
    """
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        point_a = {"x": point_a_x, "y": point_a_y}
        point_b = {"x": point_b_x, "y": point_b_y}

        result = process_coronary_analysis(
            file_bytes=file_bytes,
            point_a=point_a,
            point_b=point_b,
            catheter_size=catheter_size,
            filename=file.filename or "",
            frame_index=frame_index
        )

        return result

    except DICOMProcessingError as e:
        logger.error(f"DICOM error during analysis: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
