# All-Weather-Advance-ANPR-system

A highly advanced Automatic Number Plate Recognition (ANPR) system engineered to function robustly in severely degraded environmental conditions. By integrating state-of-the-art Image Restoration networks (Restormer, ESRGAN, NAFNet) into a real-time YOLOv10/PaddleOCR pipeline, this system successfully recovers and reads license plates obscured by heavy rain, fog, motion blur, and defocus.

## Architecture Overview

\\\
Camera Feed (WebSockets/OpenCV)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Object Detection Layer (YOLOv10m)               │  ← Detects Vehicles & License Plates
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Degradation Assessment & Preprocessing          │  ← Determines if the crop is degraded
└──────────┬───────────────────────────────────────┘
           │ (Degraded Crop)
           ▼
┌──────────────────────────────────────────────────┐
│  Image Restoration Engine (PyTorch)              │  
│  - Heavy Rain: Restormer Deraining               │  ← Reconstructs structural details
│  - Low Resolution: ESRGAN Super-Resolution       │  
│  - Motion Blur: NAFNet / MB-TaylorFormer         │  
└──────────┬───────────────────────────────────────┘
           │ (Cleaned Crop)
           ▼
┌──────────────────────────────────────────────────┐
│  Optical Character Recognition (PaddleOCR)       │  ← Extracts Alpha-Numeric String
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Django Application Logic (core/views.py)        │  ← Rules Engine (e.g., Odd/Even bans)
└──────────────┬───────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│  Multi-Camera Web Dashboard (WebSockets)         │  ← Real-time UI and Forensic Search
└──────────────────────────────────────────────────┘
\\\

## System Output

The system generates both real-time telemetry and database logs containing cleaned images and extracted data:

| Camera ID | Original Image | Restored Image | Plate Number | Confidence | Violation Alert |
|---|---|---|---|---|---|
| CAM_01_TOLL | [Blurry/Rainy] | [Clean ESRGAN] | MH 12 AB 1234 | 94.5% | None |
| CAM_02_HWY  | [Defocused]    | [Clean Restormer] | DL 4C AB 0009 | 98.1% | ⚠️ Friday Ban |

## Directory Structure

\\\
├── anpr_dashboard/             # Django project settings and routing
├── core/                       # Main ANPR application and models
│   ├── ESRGAN/                 # Super-resolution architecture and weights
│   ├── NAFNet/                 # Non-linear Activation Free Network for deblurring
│   ├── Restormer/              # Transformer-based Image Restoration (Deraining/Defocus)
│   ├── consumers.py            # Django Channels WebSocket handlers for live feeds
│   ├── models.py               # Database schemas (VehicleRegistry, SightingLog)
│   ├── views.py                # Core vision pipeline and HTTP endpoints
│   └── templates/              # Dashboard UIs (multicamera, forensic search)
│
├── media/                      # Storage for raw and restored snapshot crops
├── yolov10m.pt                 # YOLOv10 Medium weights for high-speed detection
├── manage.py                   # Django CLI
└── README.md                   # Project documentation
\\\

## How to Run

### 1. Prerequisites
- Python 3.9+
- CUDA-enabled GPU (Highly recommended for real-time Restormer/YOLO execution)
- Redis (Required for Django Channels / WebSockets)

### 2. Install Dependencies

\\\ash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install core requirements
pip install django channels daphne torch torchvision opencv-python ultralytics paddlepaddle paddleocr scikit-image
\\\
*(Note: Ensure you download the pretrained .pth weights for Restormer and ESRGAN and place them in their respective core/*/pretrained_models/ directories).*

### 3. Run the Application

\\\ash
# Run database migrations
python manage.py makemigrations
python manage.py migrate

# Start the ASGI server (Daphne) for WebSockets
daphne -b 0.0.0.0 -p 8000 anpr_dashboard.asgi:application
\\\

### 4. Usage
1. Open http://localhost:8000/dashboard to access the Multi-Camera Live View.
2. The system automatically connects via WebSockets (ws://) to ingest simulated or real RTSP camera streams.
3. As vehicles pass, YOLOv10 extracts the plate. If it detects noise/rain, the frame is routed through the PyTorch Restoration Engine before hitting PaddleOCR.
4. Alerts are generated dynamically on the dashboard if a vehicle violates programmed logic (e.g., specific trailing digits banned on certain weekdays).
5. Use the Forensic Dashboard (/forensic) to search the SQLite database by color, type, or partial plate.

## Key Design Decisions

1. **Transformer-Based Restoration (Restormer)**: Traditional CNN filters fail to recover characters obscured by heavy rain streaks. We integrated Restormer because its Multi-Dconv Head Transposed Attention (MDTA) effectively models global context across the image to physically reconstruct occluded characters.
2. **Dynamic Degradation Routing**: Running 3 deep learning models (YOLO + Restormer + OCR) per frame is computationally expensive. The pipeline intelligently assesses frame quality—if a plate is clear, it bypasses the restoration networks entirely to maintain a high FPS on edge hardware.
3. **Asynchronous WebSockets**: Using Django Channels allows the backend CV loop to broadcast bounding boxes, restored images, and OCR text directly to the frontend without blocking the HTTP request-response cycle, achieving true real-time monitoring across 6 simultaneous cameras.
