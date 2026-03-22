"""
FastAPI server for the Kabaddi video-processing loop.

`Court_code2.py` runs this app via uvicorn and injects live queues into:
- `app.state.frame_queue`: numpy BGR frames (combined dashboard frame)
- `app.state.state_queue`: dict state snapshots
- `app.state.input_queue`: numpy BGR frames (raw input frame)

This file intentionally keeps dependencies minimal so the Python pipeline can
publish live data without coupling to the React frontend.
"""

from __future__ import annotations

import asyncio
import json
import re
import os
import queue
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse


app = FastAPI(title="Kabaddi Live API", version="0.1.0")

# Dev-friendly defaults: Vite dev server runs on a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _drain_latest(q: "queue.Queue[Any]") -> Any | None:
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest


def _cors_headers(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cross-Origin-Resource-Policy": "cross-origin",
    }
    if extra:
        headers.update(extra)
    return headers


def _jpeg_bytes(bgr_frame, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG.")
    return bytes(buf)


def _combined_slices(frame_width: int) -> dict[str, tuple[int, int, int, int]]:
    """
    Compute crop rectangles from the combined frame written by Court_code2.py.

    Layout (x-axis):
      [ vis | court_mat (400px) | graph_panel (420px) ]
    Layout (y-axis):
      - vis: full height
      - mat: top-left of its panel (260px tall)
      - graph: top-left of its panel (320px tall)
    """
    court_w, graph_w = 400, 420
    vis_w = max(1, frame_width - (court_w + graph_w))

    return {
        "vis": (0, 0, vis_w, -1),  # full height
        "mat": (vis_w, 0, vis_w + court_w, 260),
        "graph": (vis_w + court_w, 0, vis_w + court_w + graph_w, 320),
    }


def _mjpeg_stream(get_frame: Callable[[], Any | None], fps_cap: float = 30.0):
    boundary = b"frame"
    min_dt = 1.0 / max(1.0, float(fps_cap))
    last_emit = 0.0

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        dt = now - last_emit
        if dt < min_dt:
            time.sleep(min_dt - dt)
        last_emit = time.time()

        try:
            jpg = _jpeg_bytes(frame)
        except Exception:
            # Avoid killing the stream if a single encode fails.
            continue

        headers = (
            b"--" + boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
        )
        yield headers + jpg + b"\r\n"


def _mjpeg_from_video_file(path: Path, fps_cap: float = 30.0, loop: bool = True):
    """
    MJPEG stream from a saved video file (MP4 etc.).

    This is a compatibility fallback when the browser can't decode the MP4 codec
    (or when the container isn't streamable). It trades seeking/controls for
    "always displays" reliability via <img>.
    """
    boundary = b"frame"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        # yield nothing; client will just hang until it disconnects.
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = float(src_fps) if src_fps and src_fps > 0 else float(fps_cap)
    fps = min(float(fps_cap), fps) if fps_cap else fps
    min_dt = 1.0 / max(1.0, fps)
    last_emit = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not loop:
                    time.sleep(0.2)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            now = time.time()
            dt = now - last_emit
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last_emit = time.time()

            try:
                jpg = _jpeg_bytes(frame, quality=80)
            except Exception:
                continue

            headers = (
                b"--" + boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            )
            yield headers + jpg + b"\r\n"
    finally:
        cap.release()


def _crop_combined_frame(bgr_frame, kind: str):
    """
    Crop `vis`/`mat`/`graph` from the combined dashboard frame produced by Court_code2.py.
    """
    rects = _combined_slices(int(bgr_frame.shape[1]))
    if kind not in rects:
        return bgr_frame
    x0, y0, x1, y1 = rects[kind]
    if y1 == -1:
        return bgr_frame[:, x0:x1]
    return bgr_frame[y0:y1, x0:x1]


def _mjpeg_from_combined_video_file(path: Path, kind: str, fps_cap: float = 30.0, loop: bool = True):
    """
    MJPEG stream from a saved *combined* video file (vis|mat|graph), cropped to a single panel.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = float(src_fps) if src_fps and src_fps > 0 else float(fps_cap)
    fps = min(float(fps_cap), fps) if fps_cap else fps
    min_dt = 1.0 / max(1.0, fps)
    last_emit = 0.0
    boundary = b"frame"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not loop:
                    time.sleep(0.2)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = _crop_combined_frame(frame, kind)

            now = time.time()
            dt = now - last_emit
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last_emit = time.time()

            try:
                jpg = _jpeg_bytes(frame, quality=80)
            except Exception:
                continue

            headers = (
                b"--" + boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            )
            yield headers + jpg + b"\r\n"
    finally:
        cap.release()


def _require_queue(name: str) -> "queue.Queue[Any]":
    q = getattr(app.state, name, None)
    if q is None:
        raise HTTPException(status_code=503, detail=f"Queue '{name}' not attached yet.")
    return q


@app.get("/api/health")
def health():
    # `live` means the processing loop attached queues into app.state.
    live = getattr(app.state, "state_queue", None) is not None and getattr(app.state, "frame_queue", None) is not None
    run_id = getattr(app.state, "run_id", None)

    processed = _latest_video("processed_*.mp4")
    report = _latest_video("confirmed_report_*.mp4")
    archive = processed is not None or report is not None or (_videos_dir() / "confirmed_events_latest.json").exists()

    return {"ok": True, "live": live, "archive": archive, "run_id": run_id}


@app.get("/api/state")
def get_state():
    q = _require_queue("state_queue")
    latest = _drain_latest(q)
    if latest is None:
        return Response(status_code=204)
    return JSONResponse(latest)


@app.get("/api/state/stream")
async def stream_state():
    q = _require_queue("state_queue")

    async def gen():
        while True:
            latest = _drain_latest(q)
            if latest is not None:
                payload = json.dumps(latest, ensure_ascii=False, separators=(",", ":"))
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/logs/stream")
async def stream_logs():
    """
    Server-Sent Events stream of backend console logs.

    Court_code2.py can attach `app.state.log_queue` with newline-delimited strings.
    """
    q = _require_queue("log_queue")

    async def gen():
        while True:
            latest = _drain_latest(q)
            if latest is not None:
                payload = json.dumps(latest, ensure_ascii=False, separators=(",", ":"))
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/frame.jpg")
def get_frame_jpg():
    q = _require_queue("frame_queue")
    latest = _drain_latest(q)
    if latest is None:
        return Response(status_code=204)
    return Response(content=_jpeg_bytes(latest), media_type="image/jpeg")


@app.get("/api/frame/stream")
def stream_frame():
    q = _require_queue("frame_queue")

    def get_latest():
        return _drain_latest(q)

    return StreamingResponse(
        _mjpeg_stream(get_latest, fps_cap=30.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/input/stream")
def stream_input():
    q = _require_queue("input_queue")

    def get_latest():
        return _drain_latest(q)

    return StreamingResponse(
        _mjpeg_stream(get_latest, fps_cap=30.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/vis/stream")
def stream_vis():
    q = _require_queue("frame_queue")

    def get_latest():
        latest = _drain_latest(q)
        if latest is None:
            return None
        rects = _combined_slices(int(latest.shape[1]))
        x0, y0, x1, y1 = rects["vis"]
        if y1 == -1:
            return latest[:, x0:x1]
        return latest[y0:y1, x0:x1]

    return StreamingResponse(
        _mjpeg_stream(get_latest, fps_cap=30.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/mat/stream")
def stream_mat():
    q = _require_queue("frame_queue")

    def get_latest():
        latest = _drain_latest(q)
        if latest is None:
            return None
        rects = _combined_slices(int(latest.shape[1]))
        x0, y0, x1, y1 = rects["mat"]
        return latest[y0:y1, x0:x1]

    return StreamingResponse(
        _mjpeg_stream(get_latest, fps_cap=20.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/graph/stream")
def stream_graph():
    q = _require_queue("frame_queue")

    def get_latest():
        latest = _drain_latest(q)
        if latest is None:
            return None
        rects = _combined_slices(int(latest.shape[1]))
        x0, y0, x1, y1 = rects["graph"]
        return latest[y0:y1, x0:x1]

    return StreamingResponse(
        _mjpeg_stream(get_latest, fps_cap=20.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _videos_dir() -> Path:
    # Expect `Videos/` next to this file (matches Court_code2.py usage).
    base = Path(__file__).resolve().parent
    return base / "Videos"


def _classifier_dataset_dir() -> Path:
    return _videos_dir() / "classifier_dataset"


def _build_clip_id(event_type: str, frame: int, subject: Any, obj: Any) -> str:
    return f"{event_type}_f{int(frame):05d}_s{subject}_o{obj}"


def _parse_event_id(event_id: str) -> tuple[str, int, str, str]:
    """
    event_id format (from frontend): "{type}|{frame}|{subject}|{object}"
    """
    parts = str(event_id).split("|")
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="Invalid event id format.")
    event_type = parts[0].strip()
    try:
        frame = int(parts[1])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid frame in event id.") from exc
    subject = parts[2].strip()
    obj = parts[3].strip()
    if not event_type:
        raise HTTPException(status_code=400, detail="Missing event type in event id.")
    return event_type, frame, subject, obj


_CLIP_RE = re.compile(r"^(?P<event_type>.+)_f(?P<frame>\d{5})_s(?P<subject>.+)_o(?P<object>.+)$")


def _paths_for_clip_id(clip_id: str) -> tuple[Path, Path]:
    """
    Resolve clip (.mp4) and payload (.json) paths for a given clip_id.
    """
    clip_id = str(clip_id).strip()
    if "/" in clip_id or "\\" in clip_id or ".." in clip_id:
        raise HTTPException(status_code=400, detail="Invalid clip id.")
    match = _CLIP_RE.match(clip_id)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid clip id format.")

    event_type = match.group("event_type")
    base = _classifier_dataset_dir() / event_type
    clip_path = base / f"{clip_id}.mp4"
    payload_path = base / f"{clip_id}.json"
    return clip_path, payload_path


def _latest_video(glob_pattern: str) -> Optional[Path]:
    videos_dir = _videos_dir()
    candidates = list(videos_dir.glob(glob_pattern))
    if not candidates:
        return None

    def _is_probably_playable_mp4(path: Path) -> bool:
        """
        Heuristic filter so the frontend doesn't try to play:
        - 0-byte / incomplete outputs
        - MP4 files encoded as `mp4v` (MPEG-4 Part 2), which many browsers reject

        We prefer H.264 (`avc1`). This is a heuristic (not a full MP4 parser),
        but is enough to avoid the common "No video with supported format" issue.
        """
        try:
            st = path.stat()
        except OSError:
            return False
        if st.st_size < 4096:
            return False
        # OpenCV frequently writes MP4 with `moov` atom at the end (non-faststart),
        # so codec identifiers (e.g. `avc1`) may appear in the tail rather than the head.
        try:
            with path.open("rb") as f:
                head = f.read(256 * 1024)
                tail_len = min(st.st_size, 256 * 1024)
                if tail_len:
                    f.seek(max(0, st.st_size - tail_len))
                    tail = f.read(tail_len)
                else:
                    tail = b""
        except OSError:
            return False
        blob = head + tail
        if b"ftyp" not in blob:
            return False
        # Prefer (and require) H.264 for browser playback.
        # If we don't see `avc1`, treat it as non-playable and skip it.
        return b"avc1" in blob

    # newest-first but skip obviously broken/unplayable ones
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.suffix.lower() == ".mp4" and _is_probably_playable_mp4(p):
            return p
    return None


def _range_file_response(request: Request, path: Path, media_type: str):
    """
    Serve a file with HTTP Range support.

    This is important for MP4s produced by OpenCV where the `moov` atom is often
    written at the end of the file. Browsers will use Range requests to fetch
    metadata from the tail and begin playback without downloading the full file.
    """
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'inline; filename="{path.name}"',
    }
    headers.update(_cors_headers())

    if not range_header:
        # Let Starlette handle streaming; include Accept-Ranges regardless.
        return FileResponse(str(path), media_type=media_type, headers=headers)

    m = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not m:
        return FileResponse(str(path), media_type=media_type, headers=headers)

    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        return Response(status_code=416, headers=_cors_headers({"Content-Range": f"bytes */{file_size}"}))

    if start_s == "":
        # Suffix bytes: "-N" => last N bytes.
        length = int(end_s)
        start = max(0, file_size - length)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1

    if start >= file_size:
        return Response(status_code=416, headers=_cors_headers({"Content-Range": f"bytes */{file_size}"}))

    end = min(end, file_size - 1)
    content_length = (end - start) + 1

    def iterfile(chunk_size: int = 1024 * 1024):
        with path.open("rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers.update(
        {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
        }
    )
    return StreamingResponse(iterfile(), status_code=206, media_type=media_type, headers=headers)


@app.get("/api/videos/latest")
def latest_videos():
    videos_dir = _videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)

    processed = (
        _latest_video("processed_sequence_latest.mp4")
        or _latest_video("processed_sequence_*.mp4")
        or _latest_video("processed_*.mp4")
    )
    report = (
        _latest_video("confirmed_report_sequence_latest.mp4")
        or _latest_video("confirmed_report_sequence_*.mp4")
        or _latest_video("confirmed_report_*.mp4")
    )
    return {
        "processed": processed.name if processed else None,
        "report": report.name if report else None,
    }


@app.get("/api/archive/events")
def archive_events():
    """
    Returns the last saved confirmed-events log, if available.
    Written by Court_code2.py to Videos/confirmed_events_latest.json.
    """
    path = _videos_dir() / "confirmed_events_latest.json"
    if not path.exists():
        return {"events": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not read archive events.") from exc
    events = payload.get("events", [])
    if not isinstance(events, list):
        events = []
    return {
        "events": events,
        "raid_summaries": payload.get("raid_summaries", []),
        "team_scores": payload.get("team_scores", {}),
        "raid_label": payload.get("raid_label"),
        "raid_index": payload.get("raid_index"),
        "attacking_team": payload.get("attacking_team"),
    }


@app.get("/api/videos/file/{filename}")
def get_video_file(filename: str, request: Request):
    videos_dir = _videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Basic path traversal protection.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = videos_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    if path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    return _range_file_response(request, path, media_type="video/mp4")


@app.get("/api/videos/mjpeg/{filename}")
def get_video_mjpeg(filename: str):
    videos_dir = _videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = videos_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return StreamingResponse(
        _mjpeg_from_video_file(path, fps_cap=30.0, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_cors_headers({"Cache-Control": "no-store"}),
    )


@app.get("/api/videos/mjpeg/vis/{filename}")
def get_video_mjpeg_vis(filename: str):
    videos_dir = _videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = videos_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return StreamingResponse(
        _mjpeg_from_combined_video_file(path, "vis", fps_cap=30.0, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_cors_headers({"Cache-Control": "no-store"}),
    )


@app.get("/api/videos/mjpeg/mat/{filename}")
def get_video_mjpeg_mat(filename: str):
    videos_dir = _videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = videos_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return StreamingResponse(
        _mjpeg_from_combined_video_file(path, "mat", fps_cap=30.0, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_cors_headers({"Cache-Control": "no-store"}),
    )


@app.get("/api/events/details/{event_id}")
def event_details(event_id: str):
    """
    Fetch the exported clip + payload for a confirmed event (if available).
    These are written by `ConfirmedWindowDatasetExporter` under:
      Videos/classifier_dataset/<EVENT_TYPE>/<CLIP_ID>.{mp4,json}
    """
    event_type, frame, subject, obj = _parse_event_id(event_id)
    clip_id = _build_clip_id(event_type, frame, subject, obj)
    clip_path, payload_path = _paths_for_clip_id(clip_id)

    payload = None
    if payload_path.exists():
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None

    # Best-effort: also attach archived mat/court-coordinate window data (if present).
    archive_event = None
    court_meta = None
    try:
        archive_path = _videos_dir() / "confirmed_events_latest.json"
        if archive_path.exists():
            archive_payload = json.loads(archive_path.read_text(encoding="utf-8"))
            court_meta = archive_payload.get("court_meta")
            events = archive_payload.get("events", [])
            if isinstance(events, list):
                for ev in events:
                    try:
                        if str(ev.get("type")) != str(event_type):
                            continue
                        if int(ev.get("frame", -1)) != int(frame):
                            continue
                        if str(ev.get("subject")) != str(subject):
                            continue
                        if str(ev.get("object")) != str(obj):
                            continue
                        archive_event = ev
                        break
                    except Exception:
                        continue
    except Exception:
        archive_event = None

    return {
        "event_id": event_id,
        "clip_id": clip_id,
        "clip_available": clip_path.exists(),
        "payload_available": payload is not None,
        "clip_url": f"/api/events/clip/{clip_id}" if clip_path.exists() else None,
        "payload_url": f"/api/events/payload/{clip_id}" if payload_path.exists() else None,
        "payload": payload,
        "archive_event": archive_event,
        "court_meta": court_meta,
    }


@app.get("/api/events/clip/{clip_id}")
def get_event_clip(clip_id: str, request: Request):
    clip_path, _ = _paths_for_clip_id(clip_id)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return _range_file_response(request, clip_path, media_type="video/mp4")


@app.get("/api/events/clip_mjpeg/{clip_id}")
def get_event_clip_mjpeg(clip_id: str):
    clip_path, _ = _paths_for_clip_id(clip_id)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return StreamingResponse(
        _mjpeg_from_video_file(clip_path, fps_cap=30.0, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_cors_headers({"Cache-Control": "no-store"}),
    )


@app.get("/api/events/clip_mjpeg/vis/{clip_id}")
def get_event_clip_mjpeg_vis(clip_id: str):
    clip_path, _ = _paths_for_clip_id(clip_id)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return StreamingResponse(
        _mjpeg_from_combined_video_file(clip_path, "vis", fps_cap=30.0, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_cors_headers({"Cache-Control": "no-store"}),
    )


@app.get("/api/events/clip_mjpeg/mat/{clip_id}")
def get_event_clip_mjpeg_mat(clip_id: str):
    clip_path, _ = _paths_for_clip_id(clip_id)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return StreamingResponse(
        _mjpeg_from_combined_video_file(clip_path, "mat", fps_cap=30.0, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_cors_headers({"Cache-Control": "no-store"}),
    )


@app.get("/api/events/payload/{clip_id}")
def get_event_payload(clip_id: str):
    _, payload_path = _paths_for_clip_id(clip_id)
    if not payload_path.exists():
        raise HTTPException(status_code=404, detail="Payload not found.")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not read payload.") from exc
    return JSONResponse(payload)


if __name__ == "__main__":
    # Allows running the API without the processing loop:
    #   python api_server.py
    # Recommended (equivalent):
    #   uvicorn api_server:app --host 0.0.0.0 --port 8000
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, log_level="info")
