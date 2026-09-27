import sys
from pathlib import Path

# Add backend, opencv, and project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OPENCV_DIR = BASE_DIR.parent / "opencv"
ML_DIR = BASE_DIR.parent / "ml"
for d in [str(ML_DIR), str(PROJECT_ROOT), str(BASE_DIR), str(OPENCV_DIR)]:
    if d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)

import io
import base64
import logging
import datetime
from typing import Optional, List, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dicom_reader import read_dicom_file, DICOMProcessingError
from preprocessing import run_preprocessing_pipeline
from pipeline import load_input_image_or_dicom, process_coronary_analysis
from ml.pipeline.ml_pipeline import analyze_image
from ml.config import CADICA_ID_TO_LABEL

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


@app.get("/api/v1/health")
@app.get("/api/health")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Coronix API",
        "version": "1.0.0",
        "active_ml_model": "CADICA Faster R-CNN (best_lesion_detector.pth)",
        "model": "CADICA Faster R-CNN (best_lesion_detector.pth)",
        "checkpoint": "ml/models/best_lesion_detector.pth"
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


@app.post("/api/v1/analyze")
@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    point_a_x: float = Form(...),
    point_a_y: float = Form(...),
    point_b_x: float = Form(...),
    point_b_y: float = Form(...),
    catheter_size: Optional[float] = Form(None),
    catheter_fr: Optional[float] = Form(None),
    catheter_diameter_px: Optional[float] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    frame_index: int = Form(0)
):
    """
    Live real ML analysis endpoint:
    Uses trained CADICA Faster R-CNN detector (ml/models/best_lesion_detector.pth)
    via ml.pipeline.ml_pipeline.analyze_image().
    MockLesionModel is completely removed/bypassed.
    """
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # Load frame (handles both DICOM and standard images PNG/JPG)
        frame, metadata, total_frames, file_type, raw_b64 = load_input_image_or_dicom(
            file_bytes=file_bytes,
            filename=file.filename or "",
            frame_index=frame_index
        )

        pt_a = (float(point_a_x), float(point_a_y))
        pt_b = (float(point_b_x), float(point_b_y))
        effective_cath_fr = catheter_fr if catheter_fr is not None else (catheter_size if catheter_size is not None else 6.0)
        effective_cath_px = catheter_diameter_px if catheter_diameter_px is not None else 24.0
        conf_thresh = confidence_threshold if confidence_threshold is not None else 0.15

        # Call existing real ML pipeline directly with best_lesion_detector.pth
        ml_result = analyze_image(
            image_path=frame,
            point_a=pt_a,
            point_b=pt_b,
            catheter_fr=effective_cath_fr,
            catheter_diameter_px=effective_cath_px,
            confidence_threshold=conf_thresh,
        )

        # Map CADICA severity id to label for display/PDF
        sev_id = ml_result.get("severity")
        sev_cat = CADICA_ID_TO_LABEL.get(sev_id, sev_id) if sev_id else None

        # Build unified response compatible with frontend and verification checks
        response = {
            "valid_points": ml_result.get("valid_points", True),
            "point_a": ml_result.get("point_a"),
            "point_b": ml_result.get("point_b"),
            "catheter_fr": ml_result.get("catheter_fr"),
            "catheter_size": ml_result.get("catheter_fr"),
            "catheter_diameter_mm": ml_result.get("catheter_diameter_mm"),
            "catheter_diameter_px": ml_result.get("catheter_diameter_px"),
            "mm_per_pixel": ml_result.get("mm_per_pixel"),
            "lesion_detected": ml_result.get("lesion_detected", False),
            "candidate_detected": ml_result.get("candidate_detected", False),
            "status": ml_result.get("status"),
            "confidence_threshold": conf_thresh,
            "bbox": ml_result.get("bbox"),
            "lesion_bbox": ml_result.get("bbox"),
            "severity": ml_result.get("severity"),
            "severity_category": sev_cat,
            "confidence": ml_result.get("confidence"),
            "segment_label": "LAD",
            "lesion_width_mm": ml_result.get("lesion_width_mm"),
            "lesion_height_mm": ml_result.get("lesion_height_mm"),
            "lesion_length_mm": ml_result.get("lesion_length_mm"),
            "image_derived_stenosis_percent": ml_result.get("image_derived_stenosis_percent"),
            "model_version": "CADICA Faster R-CNN (best_lesion_detector.pth)",
            "file_type": file_type,
            "total_frames": total_frames,
            "frame_index": frame_index,
            "metadata": metadata,
            "error": ml_result.get("error")
        }

        return response

    except DICOMProcessingError as e:
        logger.error(f"DICOM error during analysis: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

class PointModel(BaseModel):
    x: float
    y: float

class ReportRequest(BaseModel):
    title: Optional[str] = "CORONIX - Coronary Lesion Analysis Report"
    case_id: Optional[str] = None
    segment_label: Optional[str] = None
    pointA: Optional[PointModel] = None
    pointB: Optional[PointModel] = None
    catheterSize: Optional[float] = None
    catheter_unit: Optional[str] = "Fr"
    lesion_detected: Optional[bool] = None
    lesion_bbox: Optional[List[float]] = None
    lesion_location: Optional[str] = None
    severity_category: Optional[str] = None
    confidence: Optional[float] = None
    overlay_image_base64: Optional[str] = None
    short_finding: Optional[str] = None
    disclaimer: Optional[str] = None
    model_version: Optional[str] = None




# ---------------------------------------------------------------------------
# PDF Report endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/report")
async def generate_report_endpoint(payload: ReportRequest):
    """
    Generates a single-page PDF summarising the coronary lesion analysis.
    Returns application/pdf for direct browser download.
    """
    try:
        from fpdf import FPDF

        def _safe(text: str) -> str:
            """Strip characters outside latin-1 range so Helvetica won't crash."""
            return text.encode("latin-1", errors="replace").decode("latin-1")

        TEAL = (0, 139, 139)
        DARK = (30, 41, 59)
        MUTED = (100, 116, 139)
        WHITE = (255, 255, 255)
        LIGHT_BG = (241, 245, 249)
        RED_LIGHT = (254, 242, 242)
        RED_BORDER = (220, 38, 38)

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False)
        pdf.add_page()
        W = pdf.w  # 210 mm

        # ---- Header bar ----
        pdf.set_fill_color(*TEAL)
        pdf.rect(0, 0, W, 22, "F")
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(10, 5)
        pdf.cell(0, 12, "CORONIX", ln=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_xy(10, 13)
        pdf.cell(0, 6, "AI-Assisted Coronary Angiography Analysis", ln=0)
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        pdf.set_xy(W - 70, 13)
        pdf.cell(60, 6, ts, align="R")

        # ---- Title ----
        pdf.set_xy(10, 27)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*DARK)
        title = (payload.title or "Coronary Lesion Analysis Report").replace("\u2014", "-")
        pdf.cell(0, 9, title, ln=True)

        # ---- Case ID ----
        if payload.case_id:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*MUTED)
            pdf.cell(0, 6, f"Case ID: {payload.case_id}", ln=True)

        y_after_title = pdf.get_y() + 2

        # ---- Helper: draw a labelled field row ----
        def row(label: str, value: str, y: float) -> float:
            """Draws a labelled row. Returns new y."""
            pdf.set_xy(10, y)
            pdf.set_fill_color(*LIGHT_BG)
            pdf.set_draw_color(*LIGHT_BG)
            pdf.rect(10, y, W - 20, 8, "F")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*MUTED)
            pdf.set_xy(12, y + 1)
            pdf.cell(45, 6, _safe(label.upper()), ln=0)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*DARK)
            pdf.set_xy(58, y + 1)
            pdf.cell(W - 68, 6, _safe(str(value)), ln=0)
            return y + 10

        def section_header(text: str, y: float) -> float:
            pdf.set_xy(10, y)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*TEAL)
            pdf.cell(0, 8, text, ln=True)
            pdf.set_draw_color(*TEAL)
            pdf.line(10, pdf.get_y(), W - 10, pdf.get_y())
            return pdf.get_y() + 3

        # ---- Section 1: Segment & Detection ----
        y = section_header("1. LESION DETECTION", y_after_title)
        y = row("Segment", payload.segment_label or "N/A", y)
        detection_val = ("DETECTED" if payload.lesion_detected else "NOT DETECTED") if payload.lesion_detected is not None else "N/A"
        y = row("Lesion", detection_val, y)
        y = row("Severity Category", payload.severity_category or "N/A", y)
        conf_str = f"{round(payload.confidence * 100, 1)}%" if payload.confidence is not None else "N/A"
        y = row("Confidence", conf_str, y)
        if payload.lesion_bbox:
            b = payload.lesion_bbox
            bbox_str = f"x={b[0]:.1f}, y={b[1]:.1f}, w={b[2]:.1f}, h={b[3]:.1f} (pixels)"
        else:
            bbox_str = "N/A"
        y = row("Lesion Bounding Box", bbox_str, y)
        y = row("Lesion Location", payload.lesion_location or "N/A", y)
        y += 4

        # ---- Section 2: Calibration ----
        y = section_header("2. CATHETER CALIBRATION", y)
        cath_str = f"{payload.catheterSize} {payload.catheter_unit or 'Fr'}" if payload.catheterSize is not None else "Not provided"
        y = row("Catheter Size", cath_str, y)
        y += 4

        # ---- Section 3: Coordinates ----
        y = section_header("3. ANALYSIS COORDINATES", y)
        pt_a_str = f"x={payload.pointA.x:.0f}, y={payload.pointA.y:.0f}" if payload.pointA else "N/A"
        pt_b_str = f"x={payload.pointB.x:.0f}, y={payload.pointB.y:.0f}" if payload.pointB else "N/A"
        y = row("Point A", pt_a_str, y)
        y = row("Point B", pt_b_str, y)
        y += 4

        # ---- Section 4: Finding ----
        y = section_header("4. CLINICAL FINDING", y)
        finding = _safe(payload.short_finding or ("Lesion detected." if payload.lesion_detected else "No significant lesion detected."))
        pdf.set_xy(10, y)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK)
        pdf.multi_cell(W - 20, 6, finding)
        y = pdf.get_y() + 4

        # ---- Section 5: Model Info ----
        y = section_header("5. MODEL INFORMATION", y)
        y = row("Model Version", payload.model_version or "CORONIX-v1.0 (CADICA Faster R-CNN)", y)
        y += 4

        # ---- Overlay image (if provided) ----
        if payload.overlay_image_base64:
            try:
                img_section_y = y
                y = section_header("6. ANNOTATED IMAGE", y)
                b64_data = payload.overlay_image_base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                img_buf = io.BytesIO(img_bytes)
                # Save to a temp file for fpdf
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tf.write(img_bytes)
                    tmp_path = tf.name
                max_img_w = min(90, W - 20)
                pdf.image(tmp_path, x=10, y=y, w=max_img_w)
                os.unlink(tmp_path)
                y = pdf.get_y() + max_img_w + 4
            except Exception as img_err:
                logger.warning(f"Could not embed overlay image in PDF: {img_err}")
                y += 4

        # ---- Disclaimer box ----
        disclaimer_text = (
            payload.disclaimer
            or "RESEARCH / HACKATHON PROTOTYPE - NOT A MEDICAL DEVICE. "
               "This output is generated by an AI model for demonstration purposes only. "
               "It has NOT been clinically validated and MUST NOT be used for diagnosis, "
               "treatment planning, or any clinical decision-making. "
               "Always consult a qualified cardiologist."
        )
        # Reserve space at bottom of page
        disc_y = max(y + 4, 240)
        pdf.set_xy(10, disc_y)
        pdf.set_fill_color(*RED_LIGHT)
        pdf.set_draw_color(*RED_BORDER)
        pdf.rect(10, disc_y, W - 20, 24, "FD")
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*RED_BORDER)
        pdf.set_xy(13, disc_y + 2)
        pdf.cell(0, 5, "DISCLAIMER - FOR RESEARCH USE ONLY", ln=True)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(80, 20, 20)
        pdf.set_xy(13, disc_y + 7)
        pdf.multi_cell(W - 26, 5, disclaimer_text)

        # ---- Footer ----
        pdf.set_y(pdf.h - 10)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, f"Generated by CORONIX AI  |  {ts}  |  Page 1 of 1", align="C")

        # Serialise to bytes
        pdf_bytes = bytes(pdf.output())
        pdf_buffer = io.BytesIO(pdf_bytes)

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="coronix-report.pdf"'}
        )

    except Exception as e:
        logger.exception(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
