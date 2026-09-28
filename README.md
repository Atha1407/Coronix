# CORONIX

## AI-Assisted Coronary Angiography Analysis System

CORONIX is a research and hackathon prototype for AI-assisted analysis of coronary angiography images.

The system combines computer vision, deep learning, vessel-region selection, lesion localization, severity mapping, image-based quantification, and a web-based interface into a unified analysis workflow.

> **Two clicks define the vessel. AI finds the lesion. Quantification explains it.**

CORONIX was developed by **Team GeekHacks** for **Synapse 1.0**, where the project received **1st Runner-Up**.

---

## Overview

Coronary angiography images contain important information about vessel structure and potential stenotic lesions. Extracting useful information from angiographic images can involve multiple manual steps.

CORONIX explores an AI-assisted workflow where a user can:

1. Upload a coronary angiography image or DICOM file.
2. Select two points defining a vessel segment of interest.
3. Process the selected region using computer vision techniques.
4. Run an object-detection model to identify potential lesions.
5. Map detected lesions to severity categories.
6. Perform image-based vessel measurements using catheter-based calibration.
7. View the analysis through a web-based interface.

The project is intended as an engineering and research prototype and is **not a clinically validated diagnostic system**.

---

## Key Features

- Coronary angiography image processing
- DICOM image handling
- Single-frame and multi-frame DICOM support
- Interactive vessel-region selection
- Region-of-interest extraction
- Vessel enhancement and preprocessing
- AI-based lesion detection
- Lesion severity mapping
- Confidence-based detection results
- Catheter-based pixel-to-millimeter calibration
- Image-derived vessel measurements
- Modular ML inference pipeline
- React-based frontend
- FastAPI backend
- Python-based computer vision pipeline
- Training and evaluation utilities for the CADICA dataset
- Automated testing across multiple project components

---

## System Architecture

```text
                         CORONIX
                            │
                            ▼
                ┌───────────────────────┐
                │     React Frontend    │
                │                       │
                │ Upload                │
                │ DICOM Viewer          │
                │ Vessel Selection      │
                │ Analysis              │
                │ Results               │
                └───────────┬───────────┘
                            │
                            │ API
                            ▼
                ┌───────────────────────┐
                │    FastAPI Backend    │
                │                       │
                │ Request Handling      │
                │ Pipeline Integration  │
                │ ML Interface          │
                └───────────┬───────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │              ML PIPELINE                │
        │                                         │
        │  Image Preprocessing                    │
        │           ↓                             │
        │  ROI Extraction                         │
        │           ↓                             │
        │  Vessel Processing                      │
        │           ↓                             │
        │  Lesion Detection                       │
        │           ↓                             │
        │  Severity Mapping                       │
        │           ↓                             │
        │  Quantification                         │
        └────────────────────┬────────────────────┘
                             │
                             ▼
                ┌───────────────────────┐
                │    Analysis Results   │
                │                       │
                │ Lesion Location       │
                │ Severity              │
                │ Confidence            │
                │ Measurements          │
                └───────────────────────┘
```

Repository Structure
```text
CORONIX/
│
├── backend/
│   ├── main.py
│   ├── ml_interface.py
│   ├── pipeline.py
│   └── test_backend.py
│
├── frontend/
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   │
│   ├── src/
│   │   ├── assets/
│   │   │   ├── hero.png
│   │   │   ├── react.svg
│   │   │   └── vite.svg
│   │   │
│   │   ├── components/
│   │   │   ├── AnalysisOverlay.jsx
│   │   │   ├── AngiogramViewer.jsx
│   │   │   ├── ConnectedDicom.jsx
│   │   │   ├── Disclaimer.jsx
│   │   │   ├── Navbar.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── StepProgress.jsx
│   │   │
│   │   ├── pages/
│   │   │   ├── About.jsx
│   │   │   ├── Analysis.jsx
│   │   │   ├── HowItWorks.jsx
│   │   │   ├── Results.jsx
│   │   │   └── Upload.jsx
│   │   │
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   └── dicomApi.js
│   │   │
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── .gitignore
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── README.md
│   └── vite.config.js
│
├── ml/
│   ├── lesion/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dataloader.py
│   │   ├── diagnose_nan.py
│   │   ├── device.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── lesion_detector.py
│   │   ├── model.py
│   │   ├── run_training.py
│   │   ├── save_final_artifacts.py
│   │   ├── smoke_test.py
│   │   ├── train.py
│   │   └── verify_inference.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── inference_pipeline.py
│   │   ├── ml_pipeline.py
│   │   └── run_demo.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_preprocessing.py
│   │   └── visualize_preprocessing.py
│   │
│   ├── quantification/
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   └── measurements.py
│   │
│   ├── roi/
│   │   ├── __init__.py
│   │   ├── roi_extractor.py
│   │   └── visualize_roi.py
│   │
│   ├── severity/
│   │   ├── __init__.py
│   │   └── severity_mapper.py
│   │
│   ├── vessel/
│   │   ├── __init__.py
│   │   └── vessel_enhancement.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── visualization.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_cadica_dataset.py
│   │   ├── test_inspect_cadica.py
│   │   ├── test_lesion.py
│   │   ├── test_lesion_model.py
│   │   ├── test_ml_pipeline.py
│   │   ├── test_pipeline.py
│   │   ├── test_preprocessing.py
│   │   ├── test_roi.py
│   │   ├── test_severity.py
│   │   └── test_vessel.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   ├── training_report.md
│   │   └── training_results.json
│   │
│   ├── requirements.txt
│   └── README.md
│
├── opencv/
│   ├── dicom_reader.py
│   ├── enhanced.jpg
│   ├── preprocessing.py
│   ├── roi.jpg
│   ├── roi.py
│   ├── roi_selection.jpg
│   ├── sample.jpg
│   ├── sample_multi_frame.dcm
│   ├── sample_single_frame.dcm
│   ├── test_dicom.py
│   ├── test_preprocessing.py
│   ├── test_roi.py
│   ├── test_vessel.py
│   ├── vessel_analysis.py
│   └── vessel_mask.jpg
│
├── sample_data/
│   ├── coronary_multi_frame.dcm
│   └── coronary_single_frame.dcm
│
├── .gitignore
├── README.md
└── requirements.txt
```

Machine Learning Pipeline
The ML component is organized as a modular pipeline:

```text
Input Angiography
        │
        ▼
Image Preprocessing
        │
        ▼
ROI Extraction
        │
        ▼
Vessel Processing
        │
        ▼
Selected Vessel Region
        │
        ▼
Lesion Detection
        │
        ▼
Severity Mapping
        │
        ▼
Quantification
        │
        ▼
Analysis Results
```

Each stage is separated into dedicated modules to simplify development, testing, experimentation, and future model improvements.

Lesion Detection
CORONIX uses a Faster R-CNN object detection architecture with a MobileNetV3-Large backbone and Feature Pyramid Network.

```text
Faster R-CNN
      │
      └── MobileNetV3-Large 320 FPN
```

The model is initialized using Torchvision pretrained weights and trained for coronary lesion detection using the CADICA dataset.

Model Configuration
| Parameter | Value |
| --- | --- |
| Architecture | Faster R-CNN |
| Backbone | MobileNetV3-Large |
| Feature Pyramid | FPN |
| Input Variant | 320 |
| Pretrained Weights | Torchvision COCO Default |
| Total Classes | 8 |
| Background Class | 0 |
| Lesion Classes | 7 |
| Training Epochs | 5 |
| Batch Size | 4 |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Weighted Sampling | Enabled |
| Target Device | NVIDIA CUDA GPU |

Severity Classification
The lesion detection model uses seven CADICA severity categories:

- p0_20
- p20_50
- p50_70
- p70_90
- p90_98
- p99
- p100

The model label mapping is:
| Class ID | Category |
| --- | --- |
| 0 | Background |
| 1 | p0_20 |
| 2 | p20_50 |
| 3 | p50_70 |
| 4 | p70_90 |
| 5 | p90_98 |
| 6 | p99 |
| 7 | p100 |

These categories correspond to the labeling scheme used for the CADICA training data.

Model Evaluation
The repository contains documented training and evaluation results.
The reported experiment was trained for five epochs, with the best validation F1 score occurring at epoch 4.

Validation Results
| Epoch | Train Loss | Validation Loss | Validation F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0891 | 0.0709 | 0.0394 | 0.0225 | 0.1585 |
| 2 | 0.0849 | 0.0733 | 0.0489 | 0.0305 | 0.1235 |
| 3 | 0.0809 | 0.0981 | 0.0562 | 0.0412 | 0.0884 |
| 4 | 0.0746 | 0.0992 | 0.0765 | 0.0591 | 0.1082 |
| 5 | 0.0689 | 0.1127 | 0.0638 | 0.0534 | 0.0793 |

Test Set Results
The documented test evaluation was performed on a patient-isolated test set containing:

- 908 evaluated images
- 466 ground-truth boxes
- 1118 predicted boxes

At IoU >= 0.5:
| Metric | Result |
| --- | --- |
| Detection Precision | 0.0215 |
| Detection Recall | 0.0515 |
| Detection F1 Score | 0.0303 |
| Mean Matched IoU | 0.6035 |

Per-Class Results
| Severity Class | Ground Truth | Predictions | True Positives | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| p0_20 | 195 | 508 | 16 | 0.0315 | 0.0821 | 0.0455 |
| p20_50 | 84 | 30 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p50_70 | 83 | 383 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p70_90 | 61 | 150 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p90_98 | 16 | 79 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p99 | 77 | 28 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p100 | 52 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |

The evaluation results are included for transparency and should be interpreted as experimental model-development results rather than clinical performance metrics.

Quantification
CORONIX includes image-based vessel measurement utilities using catheter-based calibration.
The calibration approach uses the known catheter diameter as a reference for converting pixel distances into millimeter measurements.

Pixel-to-Millimeter Calibration
The project uses:

`1 Fr = 0.333 mm`

Catheter diameter in millimeters:

`catheter_diameter_mm = catheter_size_Fr × 0.333`

Pixel-to-millimeter scale:

`mm_per_pixel = catheter_diameter_mm / catheter_diameter_pixels`

Measured distance:

`measurement_mm = measurement_pixels × mm_per_pixel`

This allows image-derived measurements to be expressed in physical units when an appropriate calibration reference is available.

Stenosis Estimation
Where a reference vessel diameter is available, stenosis can be represented using:

`stenosis_percentage = (1 - D_min / D_reference) × 100`

Where:

- `D_min` = minimum measured vessel diameter
- `D_reference` = reference vessel diameter

The resulting value is an image-derived estimate and is not a clinically validated measurement.

DICOM Processing
CORONIX includes DICOM processing utilities using pydicom.
The repository contains sample DICOM files for development and testing:

```text
sample_data/
├── coronary_multi_frame.dcm
└── coronary_single_frame.dcm
```

The OpenCV/DICOM components provide utilities for:

- Reading DICOM files
- Extracting image data
- Handling angiographic frames
- Image preprocessing
- ROI processing
- Vessel analysis

DICOM processing is separated from the machine-learning components to maintain a modular project structure.

Computer Vision
The OpenCV modules provide supporting image-processing functionality.
Key areas include:

- Image preprocessing
- Vessel enhancement
- ROI extraction
- Vessel analysis
- DICOM image reading
- Image visualization
- Vessel mask generation

Relevant implementation files are located under:

`opencv/`

and:

- `ml/preprocessing/`
- `ml/roi/`
- `ml/vessel/`
- `ml/utils/`

Backend
The backend is implemented using FastAPI.
Backend structure:

```text
backend/
├── main.py
├── ml_interface.py
├── pipeline.py
└── test_backend.py
```

The backend is responsible for application-level processing and integration between the frontend and machine-learning pipeline.
The primary responsibilities include:

- API request handling
- Image-analysis workflow integration
- ML pipeline communication
- Backend-side orchestration

Frontend
The frontend is implemented using:

- React
- Vite
- JavaScript
- CSS

The frontend provides the user-facing analysis workflow.

Main Pages
```text
Upload
   │
   ▼
Analysis
   │
   ▼
Results
```

Additional pages and components provide:

- Project information
- Workflow guidance
- Navigation
- DICOM interaction
- Image viewing
- Analysis overlays
- Progress tracking
- Result presentation
- Safety/disclaimer information

Important components include:

- AnalysisOverlay
- AngiogramViewer
- ConnectedDicom
- Disclaimer
- Navbar
- Sidebar
- StepProgress

Technology Stack
Frontend
- React
- Vite
- JavaScript
- CSS

Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

Machine Learning
- PyTorch
- Torchvision
- Faster R-CNN
- MobileNetV3-Large
- Feature Pyramid Network

Computer Vision
- OpenCV
- NumPy
- Pillow
- scikit-image

Medical Imaging
- pydicom
- DICOM

Testing
- pytest-based test modules
- Backend tests
- ML pipeline tests
- Model tests
- Preprocessing tests
- ROI tests
- Vessel-processing tests
- Severity tests

Installation
Prerequisites
Before running CORONIX locally, make sure the following are installed:

- Python 3.x
- Node.js
- npm
- Git

For ML training/inference with GPU acceleration:

- NVIDIA GPU
- Compatible CUDA environment
- Compatible PyTorch/Torchvision installation

Clone the Repository
```bash
git clone https://github.com/Atha1407/Coronix.git
cd Coronix
```

Backend Setup
Create a Python virtual environment:

```bash
python -m venv .venv
```

Windows:
```cmd
.venv\Scripts\activate
```

Linux / macOS:
```bash
source .venv/bin/activate
```

Install the root dependencies:

```bash
pip install -r requirements.txt
```

The root requirements include the primary backend and image-processing dependencies.

ML Environment Setup
Install the ML-specific dependencies:

```bash
pip install -r ml/requirements.txt
```

For GPU-based training or inference, install a PyTorch/Torchvision build compatible with the target CUDA environment.

Frontend Setup
Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The Vite development server will provide the local application URL in the terminal.

Running the Backend
From the repository root:

```bash
uvicorn backend.main:app --reload
```

The FastAPI backend will start in development mode.

Running the Full Application
Use two terminals.

Terminal 1 — Backend:
From the project root:

```bash
uvicorn backend.main:app --reload
```

Terminal 2 — Frontend:
```bash
cd frontend
npm run dev
```

The frontend communicates with the backend through the configured API services.

Testing
The repository contains tests for the backend, OpenCV utilities, ML pipeline, preprocessing, ROI extraction, vessel processing, severity mapping, and model components.
Run the test suite with:

```bash
pytest
```

Relevant test directories include:

- `backend/`
- `opencv/`
- `ml/tests/`

Training
The lesion-detection training implementation is located in:

`ml/lesion/`

Important training files include:

```text
ml/lesion/
├── dataloader.py
├── model.py
├── train.py
├── run_training.py
├── evaluate.py
├── inference.py
├── lesion_detector.py
└── verify_inference.py
```

The training pipeline supports:

- CADICA dataset loading
- Model initialization
- Weighted sampling
- Model training
- Validation
- Evaluation
- Inference
- Artifact generation

Training Artifacts
Training summaries are stored in:

```text
ml/models/
├── training_report.md
└── training_results.json
```

Large model files and generated prediction artifacts are intentionally excluded from version control.
Examples include:

- `ml/models/*.pth`
- `ml/models/predictions/`
- `ml/models/*.png`
- `ml/data/`
- `test_images/`

This keeps the repository focused on source code and lightweight experiment artifacts.

Dataset
The lesion-detection model was trained using the CADICA dataset.
The dataset itself is not included in this repository.
The repository contains the code required for:

- Dataset loading
- Dataset inspection
- Preprocessing
- Training
- Validation
- Evaluation

Users reproducing the ML experiments must obtain the CADICA dataset separately and comply with its applicable terms and conditions.

Data Management
Large datasets and model artifacts are excluded through `.gitignore`.
The repository ignores:

- `ml/data/`
- `ml/models/*.pth`
- `ml/models/predictions/`
- `ml/models/*.png`
- `test_images/`

This prevents large datasets, trained model weights, and generated prediction images from unnecessarily increasing repository size.

Application Workflow
The intended CORONIX workflow is:

```text
┌──────────────────────────┐
│ 1. Upload Angiography   │
│    Image / DICOM        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 2. View Angiogram        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 3. Select Vessel         │
│    Point A → Point B     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 4. Extract ROI           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 5. Image Preprocessing   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 6. AI Lesion Detection   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 7. Severity Mapping      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 8. Quantification        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 9. Display Results       │
└──────────────────────────┘
```

Engineering Design
CORONIX is organized into separate application and processing layers.

Frontend Layer
Responsible for:

- User interaction
- Image viewing
- Vessel selection
- Workflow navigation
- Result visualization

Backend Layer
Responsible for:

- API handling
- Application orchestration
- ML pipeline integration

ML Layer
Responsible for:

- Image preprocessing
- ROI extraction
- Vessel processing
- Lesion detection
- Severity mapping
- Quantification

OpenCV Layer
Responsible for:

- DICOM processing
- Image preprocessing
- ROI processing
- Vessel-related image operations
- Visualization

This separation allows individual components to be developed and tested independently.

Project Status
CORONIX is currently a research and hackathon prototype.
The repository contains:

- React frontend
- FastAPI backend
- Modular machine-learning pipeline
- CADICA-based training pipeline
- Lesion-detection implementation
- Image preprocessing utilities
- ROI processing
- Vessel processing
- Severity mapping
- Quantification utilities
- DICOM processing utilities
- Automated test modules
- Training and evaluation artifacts

The current ML results are experimental and should not be interpreted as clinical validation.

Limitations
Model Performance
The documented model evaluation currently shows low detection precision, recall, and F1 score on the held-out test set.
The current model should therefore be considered an experimental research implementation.

Dataset Limitations
The model was trained using the CADICA dataset. Dataset characteristics, class distribution, labeling, and acquisition conditions can influence model performance.

Generalization
Performance on angiographic data from different hospitals, imaging systems, acquisition protocols, or patient populations has not been established by this project.

Image Quality
Image quality, contrast, acquisition conditions, vessel visibility, and ROI selection can affect downstream analysis.

Quantification
Image-based measurements depend on calibration accuracy, image quality, vessel visibility, and the reliability of the selected/reference regions.

Clinical Validation
The system has not undergone clinical validation, prospective clinical testing, or regulatory certification.

Future Development
Potential future improvements include:

- Improved lesion-detection performance
- Larger and more diverse training datasets
- Better handling of class imbalance
- Improved vessel segmentation
- More robust vessel centerline extraction
- Improved quantitative measurement methods
- More extensive multi-frame analysis
- More comprehensive DICOM metadata handling
- Better confidence calibration
- External validation
- Cross-device evaluation
- Cross-hospital evaluation
- Additional model architectures
- Model comparison experiments
- Deployment-oriented infrastructure
- Expanded automated reporting capabilities

These items represent potential future development and are not necessarily implemented in the current repository.

Reproducibility
For reproducible ML experiments:

1. Use the documented Python dependencies.
2. Install the frontend dependencies.
3. Obtain the CADICA dataset separately.
4. Configure the dataset according to the ML data-loading configuration.
5. Run the provided training/evaluation scripts.
6. Record the Python, PyTorch, Torchvision, CUDA, and hardware environment.
7. Preserve the relevant dataset and training configuration.

Exact ML reproduction can depend on:

- Hardware
- CUDA version
- PyTorch/Torchvision version
- Random seeds
- Dataset preprocessing
- Dataset version
- Training configuration

Safety and Clinical Disclaimer
IMPORTANT — RESEARCH AND HACKATHON PROTOTYPE ONLY
CORONIX is an engineering and research prototype developed for experimentation and demonstration.
It is not a clinically validated medical device, has not been certified for medical diagnosis, and must not be used as the sole basis for clinical decision-making, diagnosis, treatment, or patient management.
AI predictions, severity categories, image-derived measurements, and other outputs should be treated as experimental results.
Any clinical use would require appropriate validation, regulatory review, and qualified professional oversight.

Hackathon
Synapse 1.0
CORONIX was developed as a hackathon project for Synapse 1.0.

Achievement
1st Runner-Up
The project was developed and presented by Team GeekHacks as an AI-assisted coronary angiography analysis prototype.

Team
GeekHacks
| Member |
| --- |
| Atharva Ingle |
| Anish Jabras |
| Ritesh Dudhbhate |
| Piyush Ahir |

Acknowledgements
CORONIX uses and builds upon open-source technologies and research resources including:

- PyTorch
- Torchvision
- FastAPI
- React
- Vite
- OpenCV
- NumPy
- Pillow
- pydicom
- CADICA dataset
- Other open-source Python and JavaScript libraries used within the project

We acknowledge the researchers, dataset creators, open-source developers, and maintainers whose work supported the development of this prototype.

Contributing
Contributions and technical suggestions are welcome.
To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes.
4. Run the relevant tests.
5. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
6. Push your branch:
   ```bash
   git push origin feature/your-feature
   ```
7. Open a pull request.

When contributing:

- Keep frontend, backend, and ML responsibilities separated.
- Do not commit datasets or trained model binaries.
- Add tests for significant processing changes.
- Document changes affecting model behavior.
- Do not include patient-identifiable or sensitive medical information.
- Keep generated artifacts out of source control.

Repository Hygiene
The repository intentionally excludes large or generated files such as:

- Datasets
- Trained model weights
- Generated prediction images
- Temporary test images
- Python cache files
- Virtual environments
- Operating-system-specific files

This keeps the Git repository focused on:

- Source code
- Configuration
- Tests
- Documentation
- Lightweight experiment artifacts

License
A LICENSE file is not currently included in this repository.
Until a license is added, the repository should not be assumed to grant permission to copy, modify, distribute, or commercially use the code beyond rights provided by applicable law.
If the project is intended for public reuse, an appropriate open-source license should be added as a separate LICENSE file.

Repository
GitHub:
https://github.com/Atha1407/Coronix

Project Identification
CORONIX
AI-Assisted Coronary Angiography Analysis System

Team: GeekHacks
Hackathon: Synapse 1.0
Achievement: 1st Runner-Up

Summary
CORONIX demonstrates an end-to-end engineering approach to AI-assisted coronary angiography analysis by integrating:

```text
Medical Imaging
      +
Computer Vision
      +
Deep Learning
      +
Interactive Vessel Selection
      +
Lesion Detection
      +
Severity Mapping
      +
Image Quantification
      +
Web Application
```

The project provides a modular foundation for experimenting with AI-assisted coronary angiography analysis while maintaining a clear separation between the user interface, backend services, computer-vision processing, machine-learning inference, and quantitative analysis.

CORONIX — Two clicks define the vessel. AI finds the lesion. Quantification explains it.
