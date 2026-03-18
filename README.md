# Kabaddi Video Processing (Backend + Frontend)

This repository contains:

- `Court_code2.py` and the Python processing pipeline (writes outputs under `Videos/`)
- `api_server.py` (FastAPI) that serves live state/streams and archived outputs
- `kabaddi-frontend/` (Vite + React) dashboard UI

## Layout

- `Videos/`: input videos and generated outputs (tracked video, confirmed report, event clips, archives)
- `kabaddi-frontend/`: frontend code (kept separate from backend)

## Run (Dev)

1. Backend (API + processing)

```powershell
cd D:\Codes\kabaddi\Phase-2\Kabaddi_video_processing
python Court_code2.py
```

This starts the processing loop and also runs the FastAPI server on `http://localhost:8000`.
Keep this terminal running so the frontend can continue to fetch archived outputs after processing ends.

2. Frontend

```powershell
cd D:\Codes\kabaddi\Phase-2\Kabaddi_video_processing\kabaddi-frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`) and keep the backend URL set to `http://localhost:8000`.

