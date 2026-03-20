# Kabaddi Video Processing Dashboard

## 1. What This Code Is About

This project is a full kabaddi video-understanding system with a live dashboard.

It takes a kabaddi raid video, detects and tracks players, maps them onto the court, builds interaction relationships, reasons about possible raid events, validates important touch events with a classifier, archives confirmed event windows, and serves all of that to a frontend dashboard.

In practical terms, the system does the following:

- reads a kabaddi match or raid video
- detects players with object detection
- tracks players across frames
- maps player locations from image space to court coordinates using homography
- generates interaction proposals such as player-player and player-line interactions
- reasons over those proposals using an AFGN-style factor graph representation
- confirms temporal events across a frame window
- exports confirmed windows for classifier validation
- serves live streams, state, reports, archived clips, and event details through FastAPI
- visualizes the whole pipeline in a React + Vite frontend

## 2. Modules, How They Work, and What Methods They Use

### Main Pipeline

#### [Court_code2.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/Court_code2.py)
This is the main orchestration script.

Responsibilities:
- loads the detection model
- opens the input video
- computes court homography
- runs frame-by-frame processing
- updates tracking and interaction state
- pushes live data into API queues
- archives confirmed events
- starts the FastAPI server through `uvicorn`

Main methods/ideas used:
- YOLO / RT-DETR based player detection
- tracking across frames
- homography-based court mapping
- temporal event confirmation over multiple frames
- live queue publishing for the frontend

### Live API

#### [api_server.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/api_server.py)
This is the backend service layer built with FastAPI.

Responsibilities:
- serves MJPEG live streams
- serves archived MP4 clips and payload JSON
- exposes current live state
- exposes confirmed event details
- supports frontend polling and media access

Methods used:
- FastAPI REST endpoints
- MJPEG streaming using `StreamingResponse`
- file-range MP4 serving
- CORS-enabled local frontend/backend development

### Temporal Event Logic

#### [temporal_events.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/temporal_events.py)
This module converts frame-level interaction proposals into temporally confirmed events.

Responsibilities:
- keeps active temporal candidates alive for a small gap
- fuses proposal confidence and factor confidence
- confirms events such as raider-defender contact or line-touch style events
- creates a classifier payload for downstream verification

Methods used:
- temporal window aggregation
- confidence fusion
- gap-tolerant candidate maintenance
- proposal-to-event promotion rules

### AFGN / Reasoning Layer

#### [kabaddi_afgn_reasoning.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/kabaddi_afgn_reasoning.py)
This is the symbolic / graph reasoning engine.

Responsibilities:
- consumes scene and interaction context
- reasons about raid actions
- estimates point-related action outcomes
- produces action summaries and accuracy metrics used in the frontend

Methods used:
- Active Factor Graph Network style reasoning
- factor-based contextual scoring
- rule-guided action interpretation

### Interaction Graph and Proposals

#### `interaction_graph.py`
Responsibilities:
- builds graph-level relational context between players and lines
- stores pair factors and line factors
- provides data for frontend graph visualization

Methods used:
- relational graph representation
- pairwise factor construction
- line-based contextual factors

#### `interaction_logic.py`
Responsibilities:
- processes tracked player states
- creates interaction proposals such as HHI and HLI

Methods used:
- proximity logic
- relative motion cues
- spatial relationship heuristics

### Tracking and Detection

#### `tracking_pipeline.py`
Responsibilities:
- runs detector inference
- updates tracks
- adds new tracks
- maintains track confidence and gallery information

Methods used:
- detector-assisted multi-object tracking
- optical flow support
- track state management

#### `video_stream.py`
Responsibilities:
- handles frame access from video input

### Raider-Specific Logic

#### `raider_logic.py`
Responsibilities:
- identifies / updates the raider
- computes raider-related stats for reasoning and interaction analysis

### Report Generation

#### [report_video.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/report_video.py)
Responsibilities:
- buffers frames around confirmed events
- exports confirmed report segments
- creates classifier-ready clips
- overlays event metadata on report frames

Methods used:
- temporal frame buffering
- clip extraction
- annotation overlays

### Confirmed Event Export

#### [dataset_exporter.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/dataset_exporter.py)
Responsibilities:
- writes confirmed event windows as MP4 + JSON
- archives event metadata, mat data, court coordinates, and pose data
- runs YOLO pose extraction for archived confirmed windows

Methods used:
- MP4 export
- JSON payload archiving
- YOLOv8 pose estimation for event clips
- bbox-to-track matching using IoU

### Confirmed Event Validation

#### [classifier_bridge.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/classifier_bridge.py)
Responsibilities:
- scores confirmed windows
- uses a learned touch classifier when available
- falls back to heuristic scoring for non-touch events

Methods used:
- learned classifier inference
- heuristic feature scoring
- probability-based validation

#### [touch_classifier_model.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/touch_classifier_model.py)
Defines the PyTorch touch-classification model.

#### [touch_classifier_inference.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/touch_classifier_inference.py)
Loads the trained classifier and runs inference on confirmed event frames.

Methods used:
- PyTorch inference
- frame preprocessing
- probability prediction for `valid`, `invalid`, and `uncertain`

### Frontend

#### `kabaddi-frontend/`
This is the React dashboard.

Responsibilities:
- shows live input and processed video
- shows 2D mat, graph, gallery, signals, scores, and confirmed events
- loads archived outputs when live feed is unavailable
- opens confirmed events in detail view
- renders pose overlays and a 3D confirmed-event scene

Important frontend modules:
- [src/App.jsx](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/kabaddi-frontend/src/App.jsx): main dashboard layout and state wiring
- `src/Graph2D.jsx`: graph visualization
- [src/CourtMat2D.jsx](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/kabaddi-frontend/src/CourtMat2D.jsx): frontend-generated court view
- `src/RaidReplay3D.jsx`: 3D confirmed-event visualization, pose scrubber, and mannequin rendering

Methods used:
- React state-driven UI
- Vite dev server
- Three.js 3D rendering
- archived YOLO pose playback
- MJPEG and MP4 playback

## 3. How To Download and Run the Code

### 3.1 Prerequisites

Install the following first:

- Python 3.10 or 3.11
- Node.js 18+ and npm
- Git

Recommended:
- NVIDIA GPU with CUDA-compatible PyTorch build if you want faster inference

### 3.2 Clone the Repository

```powershell
git clone <your-repo-url>
cd Kabaddi_video_processing
```

### 3.3 Create a Python Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3.4 Install Backend Dependencies

There is no single root `requirements.txt` for the active pipeline, so install these explicitly:

```powershell
pip install --upgrade pip
pip install opencv-python numpy ultralytics torch torchvision fastapi uvicorn
```

Depending on your environment, you may also want:

```powershell
pip install scikit-learn matplotlib
```

If you want GPU-enabled PyTorch, install `torch` and `torchvision` using the official PyTorch selector for your CUDA version instead of the default CPU wheel.

### 3.5 Install Frontend Dependencies

```powershell
cd kabaddi-frontend
npm install --cache .npm-cache
cd ..
```

Frontend dependencies used include:
- `react`
- `react-dom`
- `three`
- `vite`
- Tailwind/PostCSS tooling

### 3.6 Model and Asset Files You Need

Make sure these are available:

- the main detector model path expected by [Court_code2.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/Court_code2.py)
- `yolov8n-pose.pt` for confirmed-event pose extraction
- optional trained classifier checkpoint under:
  - `models/touch_classifier/best_model.pt`
- frontend mannequin model:
  - `kabaddi-frontend/public/man.glb`

### 3.7 Input Videos

Place your input videos in the locations expected by the code, for example under:

- `Videos/`
- `Videos/Cam1/`

Use the paths configured in [Court_code2.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/Court_code2.py).

### 3.8 Run the Backend

From the project root:

```powershell
cd D:\Codes\kabaddi\Phase-2\Kabaddi_video_processing
python Court_code2.py
```

What this does:
- starts the kabaddi processing pipeline
- starts the FastAPI server at `http://localhost:8000`
- begins writing outputs under `Videos/`

### 3.9 Run the Frontend

In a second terminal:

```powershell
cd D:\Codes\kabaddi\Phase-2\Kabaddi_video_processing\kabaddi-frontend
npm run dev
```

Then open:

- `http://localhost:5173`

Set backend URL in the UI to:

- `http://localhost:8000`

### 3.10 Production Frontend Build

```powershell
cd kabaddi-frontend
npm run build
```

### 3.11 Typical Run Order

1. Activate Python virtual environment
2. Start `Court_code2.py`
3. Start frontend with `npm run dev`
4. Open the dashboard
5. Inspect live state, confirmed events, archived outputs, and 3D event scene

## 4. Significance and Advantages of the Project

This project is valuable because it combines computer vision, temporal reasoning, validation, and visualization into one practical sports-analysis workflow.

### Key Significance

- turns raw kabaddi footage into structured, analyzable event data
- provides explainable reasoning instead of only black-box predictions
- gives both live monitoring and post-event review
- supports event validation rather than trusting one model blindly
- makes kabaddi-specific interactions computationally tractable

### Advantages

- **End-to-end pipeline**: detection, tracking, mapping, reasoning, validation, and visualization are connected
- **Kabaddi-specific reasoning**: not just generic object tracking; the system understands player-player and player-line interactions
- **Court-aware analysis**: homography maps image motion into real court positions
- **Temporal robustness**: events are confirmed across windows, not from a single noisy frame
- **Classifier-backed confirmation**: important touch events can be validated by a separate model
- **Archival support**: confirmed windows, mat coordinates, poses, and event metadata are stored for later analysis
- **Interactive frontend**: live dashboard, confirmed-event drilldown, pose overlays, 2D/3D views
- **Research-friendly structure**: individual modules can be replaced or improved independently

## Project Structure

- [Court_code2.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/Court_code2.py): main pipeline
- [api_server.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/api_server.py): live + archive API
- [temporal_events.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/temporal_events.py): temporal event manager
- [kabaddi_afgn_reasoning.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/kabaddi_afgn_reasoning.py): reasoning engine
- [report_video.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/report_video.py): report builder
- [dataset_exporter.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/dataset_exporter.py): event export + pose archive
- [classifier_bridge.py](/d:/Codes/kabaddi/Phase-2/Kabaddi_video_processing/classifier_bridge.py): validation bridge
- `kabaddi-frontend/`: dashboard frontend

## Notes

- If `vite` is not recognized, run `npm install` first inside `kabaddi-frontend`.
- If the frontend cannot reach the backend, it can still show last archived outputs.
- If newly added event fields do not appear in the UI, rerun the backend once so archives are regenerated.
