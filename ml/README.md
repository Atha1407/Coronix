# StenoTrace — ML/CV Module

> **Mandatory Medical Disclaimer:**  
> "This tool is an AI-assisted research/hackathon prototype for demonstration purposes only. It is not a medical device, does not provide a diagnosis, and must not be used for clinical decision-making."  
> **This is an AI-assisted research/hackathon prototype and is NOT a clinically validated medical system.**

---

## 1. Purpose of the ML/CV Module

The `ml/` package provides the classical computer vision and lightweight processing pipeline for **StenoTrace** — a two-click, AI-guided coronary lesion analysis system.

In typical clinical workflows, interobserver variability in visual stenosis assessment is high. StenoTrace allows an operator to select two reference points (**Point A** and **Point B**) along a vessel segment on an Invasive Coronary Angiography (ICA) frame. The ML/CV module extracts the corridor between these points, enhances the vessel lumen, localizes candidate narrowing, and maps the findings directly to CADICA's standardized discrete severity bins.

To meet the 5-hour hackathon constraint and ensure real-time responsiveness (<5s per analysis), the pipeline prioritizes:
- **Classical computer vision** over complex deep learning architectures.
- **Deterministic, lightweight processing** without heavy GPU requirements.
- **Fast, reproducible inference** with structured outputs.
- **Clean modularity** so individual components can be tested and upgraded independently.

---

## 2. StenoTrace ML Pipeline

The primary simplified pipeline executes in the following sequence:

```text
Angiogram image (PNG/JPG)
         ↓
   Preprocessing (Grayscale, Normalization, CLAHE, Denoising)
         ↓
   A/B Points Validation (Separation, bounds checking)
         ↓
   A/B ROI Extraction (Corridor orientation & rectified strip)
         ↓
   Vessel Enhancement (Black-hat morphology / Frangi vesselness)
         ↓
   Lesion Analysis (Cross-sectional narrowing localization)
         ↓
   CADICA Severity Mapping (Strict discrete 7-category taxonomy)
         ↓
   Visualization Overlay & Output Assembly (Bounding box, markers, base64)
```

---

## 3. Directory Structure

```text
ml/
├── README.md                    # Module documentation & architectural guidelines
├── requirements.txt             # Lightweight classical-CV dependencies
├── __init__.py                  # Package root
│
├── config.py                    # Centralized configurations, CADICA taxonomy, thresholds
│
├── preprocessing/
│   ├── __init__.py
│   └── image_preprocessing.py   # Grayscale, normalization, CLAHE, light denoising
│
├── roi/
│   ├── __init__.py
│   └── roi_extractor.py         # Point validation, A-B corridor extraction, affine rectification
│
├── vessel/
│   ├── __init__.py
│   └── vessel_enhancement.py    # Morphological black-hat and Frangi vessel filtering
│
├── lesion/
│   ├── __init__.py
│   └── lesion_detector.py       # Candidate narrowing detection interface & data models
│
├── severity/
│   ├── __init__.py
│   └── severity_mapper.py       # Maps narrowing strictly to CADICA's 7 discrete categories
│
├── pipeline/
│   ├── __init__.py
│   └── inference_pipeline.py    # Thin orchestrator coordinating end-to-end execution
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py           # Image loading, conversion, coordinate clamping, base64 encoding
│   └── visualization.py         # Canvas overlay rendering (markers, bounding boxes, labels)
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py    # Unit tests for preprocessing
│   ├── test_roi.py              # Unit tests for A-B corridor extraction
│   ├── test_vessel.py           # Unit tests for vessel filters
│   ├── test_lesion.py           # Unit tests for lesion detection interface
│   └── test_severity.py         # Unit tests for CADICA category conformity
│
└── data/
    ├── raw/                     # Original angiogram images (protected from git commit)
    ├── processed/               # Cached or preprocessed frames (protected from git commit)
    └── demo/                    # Pre-tested representative CADICA demonstration frames
```

---

## 4. Role of Each Module

| Module | Responsibility |
| :--- | :--- |
| `config.py` | Single source of truth for CADICA categories, color hex codes, CLAHE limits, corridor widths, and disclaimers. Avoids magic numbers. |
| `preprocessing/` | Converts uploaded images to normalized 8-bit grayscale, enhances local fluoroscopy contrast via CLAHE, and removes high-frequency quantum mottle. |
| `roi/` | Validates Points A and B (enforcing minimum separation and bounds), calculates corridor angle/length, and extracts an affine-rectified horizontal strip along the vessel. |
| `vessel/` | Applies classical CV filters (Black-Hat morphology or multi-scale Frangi Hessian vesselness) to highlight dark radiopaque contrast dye in the lumen. |
| `lesion/` | Detects candidate stenosis/narrowing along the rectified corridor and packages structured bounding boxes, relative coordinates, and confidence scores. |
| `severity/` | Enforces the strict CADICA 7-category taxonomy. Disallows decimal percentages and continuous output. |
| `pipeline/` | Thin orchestrator calling the individual modules in sequence; includes a guaranteed non-crashing demo fallback mode for live presentations. |
| `utils/` | Shared image I/O helpers, coordinate bounds validators, base64 string encoders, and visualization overlay drawing. |
| `tests/` | Standard `unittest` suite covering all modules with synthetic test arrays. |

---

## 5. Current Implementation Status

- [x] **Repository & Clean Modular Architecture**: Fully scaffolded with type hints and docstrings.
- [x] **Centralized Configuration (`config.py`)**: All CADICA taxonomy categories, thresholds, and disclaimer texts locked in.
- [x] **Lightweight Classical Dependencies**: Defined in `requirements.txt` (OpenCV, scikit-image, NumPy, SciPy, Pillow, Matplotlib).
- [x] **Preprocessing Pipeline**: Implemented with grayscale conversion, CLAHE contrast enhancement, and Gaussian/bilateral denoising.
- [x] **ROI Corridor Extraction**: Implemented with coordinate validation, boundary clamping, and affine strip rectification.
- [x] **Vessel Enhancement Filters**: Implemented with morphological black-hat and multi-scale Frangi filters.
- [x] **Severity Mapper**: Fully compliant with CADICA's 7 discrete stenosis bins.
- [x] **Inference Pipeline Orchestrator**: Thin runner with structured dataclass output and safe demo fallback.
- [x] **Data Directory Protection**: Added `.gitignore` to prevent raw medical images or DICOM binaries from entering source control.
- [x] **Initial Interfaces & Unit Tests**: All tests passing with synthetic frames.

---

## 6. CADICA Seven-Category Severity Taxonomy

StenoTrace strictly outputs one of the seven discrete stenosis categories defined by the CADICA dataset (Jiménez-Partinen et al., 2024, *Expert Systems*, DOI 10.1111/exsy.13708):

| CADICA ID | Stenosis Category | Clinical Description | UI Badge Color |
| :---: | :---: | :--- | :---: |
| `p0_20` | `<20%` | Minimal or no significant angiographic narrowing | Green (`#22c55e`) |
| `p20_50` | `20-50%` | Mild stenosis | Lime (`#84cc16`) |
| `p50_70` | `50-70%` | Moderate stenosis | Amber (`#eab308`) |
| `p70_90` | `70-90%` | Severe stenosis | Orange (`#f97316`) |
| `p90_98` | `90-98%` | Critical stenosis | Red (`#ef4444`) |
| `p99` | `99%` | Subtotal occlusion | Dark Red (`#b91c1c`) |
| `p100` | `100%` | Total occlusion | Deep Maroon (`#7f1d1d`) |

> **Rule:** Never generate or display an invented decimal stenosis percentage (e.g. `63.4%`). Accurate continuous percentages require real-world catheter scale calibration and reference vessel selection, which are out of scope for this hackathon build.

---

## 7. What is Intentionally NOT Implemented Yet

To preserve our 5-hour hackathon timeline and prevent scope creep:
- **No deep learning model training from scratch**: 5 hours is insufficient to train a generalized detector on fluoroscopy frames.
- **No heavy DL framework installations**: `torch`, `torchvision`, `tensorflow`, and `ultralytics` are omitted from `requirements.txt`.
- **No DICOM metadata extraction / PACS networking**: Direct DICOM ingestion is deferred to future work; the pipeline currently processes standard frame buffers (PNG/JPG).
- **No continuous / calibrated decimal stenosis percentages**: Clinical stenosis calibration requires physical catheter reference scaling.
- **No multi-lesion / full-tree vessel segmentation**: The focus is strictly on the operator-guided segment defined by Point A and Point B.

---

## 8. How the ML Module Will Connect to the Backend

The ML module is designed to integrate seamlessly into a FastAPI backend service:

```python
# Conceptual FastAPI endpoint integration
from fastapi import FastAPI, UploadFile, File, Form
from ml.pipeline.inference_pipeline import run_pipeline

app = FastAPI()

@app.post("/analyze")
async def analyze_segment(
    file: UploadFile = File(...),
    point_a_x: int = Form(...),
    point_a_y: int = Form(...),
    point_b_x: int = Form(...),
    point_b_y: int = Form(...)
):
    image_bytes = await file.read()
    
    result = run_pipeline(
        image=image_bytes,
        point_a=(point_a_x, point_a_y),
        point_b=(point_b_x, point_b_y),
    )
    
    return {
        "segment_label": result.segment_label,
        "lesion_detected": result.lesion_detected,
        "lesion_bbox": result.lesion_bbox,
        "severity_category": result.severity_category,
        "cadica_id": result.cadica_id,
        "confidence": result.confidence,
        "overlay_image_base64": result.overlay_image_base64,
        "disclaimer": result.disclaimer,
        "is_fallback": result.is_fallback
    }
```

The returned JSON directly supplies the React frontend with the visualization coordinates, severity badge, and base64 overlay, as well as providing all parameters required by the PDF report generator.
