import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import os
import json
import hashlib
import sys
import glob
import re
from classifier_bridge import ConfirmedWindowClassifierBridge
from dataset_exporter import ConfirmedWindowDatasetExporter
# from kabaddi_afgn_reasoning import KabaddiAFGNEngine  # (LEGACY - Replaced by GNN)
from interaction_graph import InteractionProposalEngine, ActiveFactorGraphNetwork, render_graph_panel
from interaction_logic import build_player_states, process_interactions
from raider_logic import collect_raider_stats, assign_raider
from report_video import ConfirmedInteractionReportBuilder
from temporal_events import TemporalInteractionCandidateManager
from tracking_pipeline import apply_optical_flow, run_yolo_detection, update_tracks, add_new_tracks, render_gallery
from video_stream import VideoStream

#COMMIT CHECK 25/3
# ======================================================
# PERFORMANCE: THREADED VIDEO READER
# ======================================================

VIDEO_GLOB = os.path.join("Videos", "Cam1", "raid*.mp4")
FALLBACK_VIDEO_PATH = os.path.join("Videos", "Cam1", "raid1.mp4")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path):
    """Resolve relative paths against this script's directory."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

def raid_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", stem)
    if m:
        return (0, int(m.group(1)), stem.lower())
    return (1, stem.lower())


def discover_raid_video_paths():
    pattern = resolve_path(VIDEO_GLOB)
    paths = [os.path.abspath(p) for p in glob.glob(pattern)]
    if not paths:
        fallback = resolve_path(FALLBACK_VIDEO_PATH)
        if os.path.exists(fallback):
            paths = [os.path.abspath(fallback)]
    return sorted(paths, key=raid_sort_key)


RAID_VIDEO_PATHS = discover_raid_video_paths()
if not RAID_VIDEO_PATHS:
    raise FileNotFoundError(
        f"No raid videos found. Expected files matching: {resolve_path(VIDEO_GLOB)}"
    )

SEQUENCE_RUN_ID = hashlib.md5("|".join(RAID_VIDEO_PATHS).encode("utf-8")).hexdigest()[:8]
CURRENT_VIDEO_STEM = os.path.splitext(os.path.basename(RAID_VIDEO_PATHS[0]))[0]

# ======================================================
# CONFIG (RESTORED)
# ======================================================

DISPLAY_SCALE = 0.5
FPS_DELAY = 1
# Set to True only if you want OpenCV windows on the backend machine.
# For the web dashboard flow, keep this False to run headless.
SHOW_BACKEND_WINDOWS = False
CONF_THRESH = 0.4
SMOOTH_ALPHA = 0.25 
MAX_PLAYERS = 8
MAX_POINTS_PER_RAID = 4
MAX_AGE=200
MODEL1 = "models/yolov8n.pt"
cursor_court_pos = None
LINE_MARGIN = 0.6 
proposal_engine = InteractionProposalEngine()  # interaction_graph.InteractionProposalEngine
graph_engine = ActiveFactorGraphNetwork(top_k=4)  # interaction_graph.ActiveFactorGraphNetwork

# --- NEW GNN INFERENCE ENGINE ---
from afgn_gnn.inference import AFGNEngineInference
# Loads the weights we just trained on your real videos!
action_engine = AFGNEngineInference(model_path="afgn_gnn/model_weights_real.pt", device="cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------

candidate_manager = TemporalInteractionCandidateManager()  # temporal_events.TemporalInteractionCandidateManager
report_builder = ConfirmedInteractionReportBuilder(max_buffer_frames=300)  # report_video.ConfirmedInteractionReportBuilder
classifier_bridge = ConfirmedWindowClassifierBridge()  # classifier_bridge.ConfirmedWindowClassifierBridge
dataset_exporter = ConfirmedWindowDatasetExporter(os.path.join(BASE_DIR, "Videos", "classifier_dataset"), fps=30.0)  # dataset_exporter.ConfirmedWindowDatasetExporter

# IMAGE-SPACE COURT LINES
lines = {
    "baulk": [(606, 483), (1771, 1078)],
    "bonus": [(745, 471), (1918, 960)],
    "middle": [(55, 486), (0, 575)],
    "end_back": [(885, 471), (1918, 763)],
    "end_left": [(58, 490), (885, 473)],
    "end_right": [(1833, 1076), (1916, 1041)],
    "lobby_left": [(45, 525), (921, 493)],
    "lobby_right": [(690, 1076), (1915, 821)],
}

lines2 = {
    "baulk": [(606, 291), (378, 128)],
    "bonus": [(678, 269), (427, 123)],
    "middle": [(215, 410), (213, 144)],
    "end_back": [(847, 268), (462, 112)],
    "end_left": [(215, 132), (462, 114)],
    "end_right": [(224, 512), (847, 268)],
    "lobby_left": [(213, 143), (477, 120)],
    "lobby_right": [(223, 407), (773, 237)],
}


def select_line_coordinates(video_path):
    normalized_path = video_path.replace("\\", "/").lower()
    if "/cam2/" in normalized_path:
        return lines2, "Cam2"
    return lines, "Cam1"

# ======================================================
# MATH & HOMOGRAPHY
# ======================================================
def line_eq(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.array([y2 - y1, x1 - x2, x2*y1 - x1*y2], dtype=np.float64)

def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1*b2 - a2*b1
    if abs(d) < 1e-6: return None
    return [(b1*c2 - b2*c1) / d, (a2*c1 - a1*c2) / d]

COURT_W, COURT_H = 400, 260
GRAPH_W, GRAPH_H = 420, 320
# Backend-rendered mat background color.
# (Frontend renders its own transparent SVG mat; this is for the backend MJPEG stream.)
MAT_TRANSPARENT_BG = False


def build_raid_geometry(video_path):
    active_lines, active_camera = select_line_coordinates(video_path)
    print(f"Using {active_camera} court coordinates for {os.path.basename(video_path)}")

    L = {k: line_eq(*v) for k, v in active_lines.items()}
    img_pts = np.array([intersect(L[a], L[b]) for a, b in [
        ("end_back", "end_left"), ("end_back", "end_right"),
        ("middle", "end_left"), ("middle", "end_right")
    ]], dtype=np.float32)

    court_pts = np.array([[0, 6.5], [10, 6.5], [0, 0], [10, 0]], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)

    def court_to_pixel(x, y):
        px = int(x / 10 * COURT_W)
        py = int((6.5 - y) / 6.5 * COURT_H)
        return px, py

    mat_base = (
        np.zeros((COURT_H, COURT_W, 3), dtype=np.uint8)
        if MAT_TRANSPARENT_BG
        else (np.ones((COURT_H, COURT_W, 3), dtype=np.uint8) * 255)
    )

    ink = (220, 220, 220) if MAT_TRANSPARENT_BG else (0, 0, 0)
    label_ink = (200, 200, 200) if MAT_TRANSPARENT_BG else (60, 60, 60)
    cv2.rectangle(mat_base, court_to_pixel(0, 0), court_to_pixel(10, 6.5), ink, 2)
    for y, name in [(3.75, "baulk"), (4.75, "bonus")]:
        cv2.line(mat_base, court_to_pixel(0, y), court_to_pixel(10, y), ink, 1)
        cv2.putText(
            mat_base,
            name,
            (8, court_to_pixel(0, y)[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            label_ink,
            1,
        )
    for x in [0.75, 9.25]:
        cv2.line(mat_base, court_to_pixel(x, 0), court_to_pixel(x, 6.5), ink, 1)

    court_meta = {
        "court_w": 10.0,
        "court_h": 6.5,
        "baulk_y": 3.75,
        "bonus_y": 4.75,
        "lobby_left_x": 0.75,
        "lobby_right_x": 9.25,
        "camera": active_camera,
        "homography": H.tolist() if hasattr(H, "tolist") else None,
    }
    return active_camera, H, mat_base, court_to_pixel, court_meta


def combine_video_files(input_paths, output_path, fps=30.0):
    valid_paths = [str(p) for p in input_paths if p and os.path.exists(p)]
    if not valid_paths:
        return None

    first_cap = cv2.VideoCapture(valid_paths[0])
    if not first_cap.isOpened():
        first_cap.release()
        return None

    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = float(first_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    first_cap.release()

    if width <= 0 or height <= 0:
        return None

    target_fps = src_fps if src_fps > 1.0 else float(fps)

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    def _open_writer(path, writer_fps, size):
        candidates = [
            ("msmf", getattr(cv2, "CAP_MSMF", 0), "H264"),
            ("msmf", getattr(cv2, "CAP_MSMF", 0), "avc1"),
            ("any", 0, "mp4v"),
        ]
        for _, api, tag in candidates:
            try:
                writer = cv2.VideoWriter(
                    path,
                    api,
                    cv2.VideoWriter_fourcc(*tag),
                    writer_fps,
                    size,
                )
            except Exception:
                continue
            if writer.isOpened():
                return writer
            writer.release()
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), writer_fps, size)
        return writer if writer.isOpened() else None

    writer = _open_writer(output_path, target_fps, (width, height))
    if writer is None:
        return None

    try:
        for path in valid_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                continue
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
            cap.release()
    finally:
        writer.release()

    return output_path if os.path.exists(output_path) else None

# ======================================================
# MAIN LOOP
# ======================================================

# ======================================================
# LIVE STREAMING API (FASTAPI)
# ======================================================
import time
import queue
import threading

LIVE_FRAME_QUEUE = queue.Queue(maxsize=5)
LIVE_STATE_QUEUE = queue.Queue(maxsize=5)
LIVE_INPUT_QUEUE = queue.Queue(maxsize=5)
LIVE_LOG_QUEUE = queue.Queue(maxsize=300)
_API_THREAD = None
_USER_QUIT = False
_USER_RESTART = False

try:
    import msvcrt  # Windows-only (non-blocking keypress)
except Exception:
    msvcrt = None


def _log_enqueue(line: str, stream: str = "stdout"):
    payload = {
        "t": time.time(),
        "stream": stream,
        "line": str(line),
    }
    if LIVE_LOG_QUEUE.full():
        try:
            LIVE_LOG_QUEUE.get_nowait()
        except queue.Empty:
            pass
    try:
        LIVE_LOG_QUEUE.put_nowait(payload)
    except queue.Full:
        pass


class _TeeStream:
    def __init__(self, inner, stream_name: str):
        self._inner = inner
        self._name = stream_name
        self._buf = ""

    def write(self, s):
        try:
            self._inner.write(s)
        except Exception:
            pass

        try:
            self._buf += str(s)
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    _log_enqueue(line, stream=self._name)
        except Exception:
            # Never break the pipeline because of logging.
            pass

    def flush(self):
        try:
            if self._buf.strip():
                _log_enqueue(self._buf, stream=self._name)
            self._buf = ""
        except Exception:
            pass
        try:
            self._inner.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._inner.isatty())
        except Exception:
            return False

def start_api_server():
    import uvicorn
    import api_server
    api_server.app.state.frame_queue = LIVE_FRAME_QUEUE
    api_server.app.state.state_queue = LIVE_STATE_QUEUE
    api_server.app.state.input_queue = LIVE_INPUT_QUEUE
    api_server.app.state.log_queue = LIVE_LOG_QUEUE
    api_server.app.state.run_id = SEQUENCE_RUN_ID
    api_server.app.state.video_stem = CURRENT_VIDEO_STEM
    uvicorn.run(api_server.app, host="0.0.0.0", port=8000, log_level="error")

# Keep the API server running while the main script is alive (during processing and
# while we wait at the "Type q to quit" prompt). When the main script exits, the
# daemon server thread will stop automatically.
_API_THREAD = threading.Thread(target=start_api_server, daemon=True)
_API_THREAD.start()

# Mirror stdout/stderr into LIVE_LOG_QUEUE so the frontend can visualize backend output.
try:
    sys.stdout = _TeeStream(sys.stdout, "stdout")
    sys.stderr = _TeeStream(sys.stderr, "stderr")
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    MODEL1="models/yolov8n.pt"
print("Device used: ",device)
model = YOLO(MODEL1).to(device)
# model = RTDETR("rtdetr-l.pt").to(device)

def process_single_raid(video_path, raid_index, team_scores, raid_summaries):
    global CURRENT_VIDEO_STEM, _USER_QUIT, _USER_RESTART
    VIDEO_PATH = resolve_path(video_path)
    path_hash = hashlib.md5(VIDEO_PATH.encode()).hexdigest()[:8]
    video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    CURRENT_VIDEO_STEM = video_stem
    raid_label = f"raid{int(raid_index)}"
    attacking_team = "A" if int(raid_index) % 2 == 1 else "B"
    defending_team = "B" if attacking_team == "A" else "A"
    ACTIVE_CAMERA, H, mat_base, court_to_pixel, COURT_META = build_raid_geometry(VIDEO_PATH)

    try:
        import api_server as _api_server
        _api_server.app.state.run_id = SEQUENCE_RUN_ID
        _api_server.app.state.video_stem = video_stem
    except Exception:
        pass

    print("\n" + "=" * 90)
    print(f"[RAID] Starting {raid_label} | Video: {os.path.basename(VIDEO_PATH)} | Attack: Team {attacking_team}")
    print(f"[RAID] Cumulative score entering raid: Team A {team_scores.get('A', 0)} - Team B {team_scores.get('B', 0)}")

    vs = VideoStream(VIDEO_PATH).start()  # video_stream.VideoStream.start
    prev_gray = None
    NEXT_ID = 0
    GALLERY = {}
    frame_idx = 0
    last_action_results = None

    # ======================================================
    # RAIDER IDENTIFICATION (Single-side logic)
    # ======================================================
    RAIDER_ID = None
    RAIDER_STATS = {}
    RAID_ASSIGNMENT_DONE = False
    RAIDER_EXITED = False
    RAIDER_MISSING_FRAMES = 0
    RAIDER_CONV_ACCUM = {}
    MIN_FRAMES_FOR_DECISION = 40

    BAULK_Y = 3.75
    ASSIGN_FRAME = 70   # assign raider after this frame

    # Cursor Tracking Logic
    cursor_court_pos = None

    def mouse_tracker(event, x, y, flags, param):
        nonlocal cursor_court_pos
        if event == cv2.EVENT_MOUSEMOVE:
            # Map scaled window coordinates back to original video size
            orig_x, orig_y = x / DISPLAY_SCALE, y / DISPLAY_SCALE
            pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, H)[0][0]
            cursor_court_pos = (mapped[0], mapped[1])

    if SHOW_BACKEND_WINDOWS:
        cv2.namedWindow("Video (Integrated)")
        cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
        cv2.namedWindow("Half Court (2D)")
        cv2.namedWindow("Interaction Graph")
        cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
    # ---------------------------------------


    output_filename = os.path.join(BASE_DIR, "Videos", f"processed_{video_stem}_{path_hash}.mp4")
    report_output_filename = os.path.join(BASE_DIR, "Videos", f"confirmed_report_{video_stem}_{path_hash}.mp4")


    def _even(n: int) -> int:
        n = int(n)
        return n if n % 2 == 0 else n + 1


    vis_w = _even(int(1920 * DISPLAY_SCALE))
    vis_h = _even(int(1080 * DISPLAY_SCALE))
    canvas_w = _even(vis_w + COURT_W + GRAPH_W)
    canvas_h = _even(max(vis_h, COURT_H, GRAPH_H))

    if os.path.exists(output_filename):
        os.remove(output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Prefer browser-friendly MP4 by explicitly selecting MSMF (Windows) for H.264.
    def _open_video_writer(path, fps, size):
        # Try H.264 via MSMF first to avoid FFmpeg's OpenH264 DLL dependency.
        candidates = [
            ("msmf", getattr(cv2, "CAP_MSMF", 0), "H264"),
            ("msmf", getattr(cv2, "CAP_MSMF", 0), "avc1"),
            ("any", 0, "mp4v"),
        ]
        for _, api, tag in candidates:
            try:
                writer = cv2.VideoWriter(
                    path,
                    api,
                    cv2.VideoWriter_fourcc(*tag),
                    fps,
                    size,
                )
            except Exception:
                continue
            if writer.isOpened():
                return writer
            writer.release()
        return None

    out = _open_video_writer(output_filename, 30.0, (canvas_w, canvas_h))
    if out is None:
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (canvas_w, canvas_h))
    if out is not None and not out.isOpened():
        # Avoid producing a 0-byte/unplayable file that the frontend will try to play.
        try:
            out.release()
        except Exception:
            pass
        out = None
        print(f"[WARN] Could not open VideoWriter for: {output_filename} (will not record processed video)")
    print(f"Recording to: {output_filename}")

    def mouse_tracker(event, x, y, flags, param):
        nonlocal cursor_court_pos

        if event == cv2.EVENT_MOUSEMOVE:

            # Convert displayed coordinates back to original frame scale
            orig_x = x / DISPLAY_SCALE
            orig_y = y / DISPLAY_SCALE

            pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)

            mapped = cv2.perspectiveTransform(pt, H)

            cursor_court_pos = (mapped[0][0][0], mapped[0][0][1])

    def log_event(event_type, player_id, frame_idx):
        EVENT_LOG.append({
            "frame": frame_idx,
            "player": player_id,
            "type": event_type,
            "investigation_until": frame_idx + INTERACTION_WINDOW
        })


    def log_confirmed_event(event):
        CONFIRMED_EVENT_LOG.append(event)


    def apply_classifier_results(confirmed_log, classifier_results):
        if not classifier_results:
            return
        result_map = {result["event_key"]: result for result in classifier_results}
        for event in confirmed_log:
            event_key = (event["type"], event["frame"], event["subject"], event["object"])
            classifier_result = result_map.get(event_key)
            if classifier_result is None:
                continue
            event["classifier_result"] = classifier_result
            event["classifier_valid_prob"] = classifier_result["probabilities"].get("valid", 0.0)
            event["classifier_label"] = classifier_result["predicted_label"]
            event["guaranteed_by_classifier"] = classifier_result["guaranteed"]

    # ======================================================
    # PHASE 2: ACTION RECOGNITION & SCORING
    # ======================================================

    # Event and interaction tracking (restored)
    EVENT_LOG = []
    CONFIRMED_EVENT_LOG = []
    touch_confirmed = False
    INTERACTION_WINDOW = 20  # frames for future investigation
    INTERACTION_COUNT = 0

    # -------------------------------------------------------------------
    # Per-frame court coordinate history (for confirmed-event archiving)
    # -------------------------------------------------------------------
    COURT_META = {
        "court_w": 10.0,
        "court_h": 6.5,
        "baulk_y": 3.75,
        "bonus_y": 4.75,
        "lobby_left_x": 0.75,
        "lobby_right_x": 9.25,
        "camera": ACTIVE_CAMERA,
        # Planar homography used to map screen -> court coords (3x3). Useful for future
        # camera-perspective matching in the 3D replay (best-effort without full intrinsics).
        "homography": H.tolist() if hasattr(H, "tolist") else None,
    }
    _FRAME_HISTORY_KEEP = 2000
    _FRAME_GALLERY_HISTORY = {}
    _CONFIRMED_EVENT_MAT_ARCHIVE = {}

    def _extract_dominant_palette(frame_bgr, bbox, k=4):
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                return []
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return []
            sample = cv2.resize(
                crop,
                (
                    max(12, min(32, crop.shape[1])),
                    max(12, min(32, crop.shape[0])),
                ),
            )
            pixels = sample.reshape((-1, 3)).astype(np.float32)
            if pixels.shape[0] < 8:
                return []
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
            _compactness, labels, centers = cv2.kmeans(
                pixels,
                int(k),
                None,
                criteria,
                2,
                cv2.KMEANS_PP_CENTERS,
            )
            counts = np.bincount(labels.flatten(), minlength=int(k))
            order = np.argsort(-counts)
            palette = []
            for idx in order[: int(k)]:
                bgr = centers[int(idx)]
                palette.append([
                    int(np.clip(round(float(bgr[2])), 0, 255)),
                    int(np.clip(round(float(bgr[1])), 0, 255)),
                    int(np.clip(round(float(bgr[0])), 0, 255)),
                ])
            return palette
        except Exception:
            return []

    def _snapshot_frame_for_mat(frame_idx, gallery, raider_id):
        players = []
        try:
            for pid, data in gallery.items():
                m_pos = data.get("display_pos")
                if m_pos is None:
                    continue
                bb = data.get("last_bbox", (0, 0, 0, 0))
                players.append({
                    "id": int(pid),
                    "visible": bool(data.get("age", 0) == 0),
                    "court_pos": [float(m_pos[0]), float(m_pos[1])],
                    "bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                })
            players.sort(key=lambda x: x["id"])
        except Exception:
            players = []
        return {
            "frame": int(frame_idx),
            "raider_id": int(raider_id) if raider_id is not None else None,
            "players": players,
        }

    def _event_key(event):
        return (
            str(event.get("type")),
            int(event.get("frame", -1)),
            str(event.get("subject")),
            str(event.get("object")),
        )

    def _archive_event_mat_window(event):
        """
        Store the per-frame court coordinate history for the event's temporal window.
        This is merged into the on-disk archive JSON at the end (to keep live objects small).
        """
        try:
            window_start = int(event.get("window_start", event.get("frame", 0)))
            window_end = int(event.get("window_end", event.get("frame", window_start)))
            mat_window = []
            for fi in range(window_start, window_end + 1):
                snap = _FRAME_GALLERY_HISTORY.get(int(fi))
                if snap is None:
                    continue
                mat_window.append(snap)
            _CONFIRMED_EVENT_MAT_ARCHIVE[_event_key(event)] = mat_window
        except Exception:
            pass

    MIDDLE_Y = 0.0
    BAULK_Y = 3.75
    BONUS_Y = 4.75
    END_LINE_Y = 6.5

    LINE_MARGIN = 0.25
    LOBBY_LEFT = 0.75
    LOBBY_RIGHT = 9.25

    paused = False
    last_vis = None
    last_mat = None
    TOTAL_POINTS = 0
    CURRENT_RAID_POINTS = 0
    last_local_totals = {"attacker": 0, "defender": 0}
    raid_summary = {
        "raid_label": raid_label,
        "raid_index": int(raid_index),
        "video_name": os.path.basename(VIDEO_PATH),
        "attacking_team": attacking_team,
        "defending_team": defending_team,
        "raider_id": None,
        "attacking_points_awarded": 0,
        "score_breakdown": [],
        "actions": [],
    }
    action_seen = set()
    classifier_touch_awards = set()

    while vs.running():
        # Optional: allow quitting anytime (Windows terminals).
        if msvcrt is not None:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"q", b"Q"):
                        _USER_QUIT = True
                        break
                    if ch in (b"n", b"N"):
                        _USER_RESTART = True
                        break
            except Exception:
                pass
        if not paused:
            frame = vs.read()
            if frame is None: continue
            frame_idx += 1
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        mat = mat_base.copy()
    

        if cursor_court_pos is not None:
            cx, cy = cursor_court_pos

            # Only draw if inside court bounds
            if -1 < cx < 11 and -1 < cy < 7.5:
                mx, my = court_to_pixel(cx, cy)

                cv2.circle(mat, (mx, my), 7, (0, 255, 255), -1)
                cv2.circle(mat, (mx, my), 9, (0, 0, 0), 1)

        # Display current scores on mat
        score_ink = (230, 230, 230) if MAT_TRANSPARENT_BG else (0, 0, 0)
        cv2.putText(mat, f"Total Points: {TOTAL_POINTS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_ink, 2)
        cv2.putText(mat, f"Current Raid: {CURRENT_RAID_POINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_ink, 2)

        # 1. OPTICAL FLOW MOTION COMPENSATION
        if prev_gray is not None:
            apply_optical_flow(prev_gray, gray, GALLERY)  # tracking_pipeline.apply_optical_flow

        # 2. YOLO DETECTION
        detections = run_yolo_detection(model, frame, device, CONF_THRESH)  # tracking_pipeline.run_yolo_detection

        # 3. TRACK PREDICTION & MATCHING
        matched_tracks, matched_dets = update_tracks(  # tracking_pipeline.update_tracks
            GALLERY,
            detections,
            gray,
            vis,
            frame_idx,
            RAID_ASSIGNMENT_DONE,
            RAIDER_ID,
        )

        # 4. NEW TRACKS & AGING
        NEXT_ID = add_new_tracks(GALLERY, detections, matched_dets, NEXT_ID, MAX_PLAYERS)  # tracking_pipeline.add_new_tracks

        # 5. MAT RENDERING & DIRECTION ARROWS
        render_gallery(GALLERY, matched_tracks, vis, mat, H, court_to_pixel, LINE_MARGIN, SMOOTH_ALPHA, RAIDER_ID, MAX_AGE)  # tracking_pipeline.render_gallery

        # Cache per-frame court coords so confirmed events can later include the exact
        # window's mat data (and the frontend can re-render it on demand).
        try:
            _FRAME_GALLERY_HISTORY[int(frame_idx)] = _snapshot_frame_for_mat(frame_idx, GALLERY, RAIDER_ID)
            if len(_FRAME_GALLERY_HISTORY) > (_FRAME_HISTORY_KEEP + 50):
                cutoff = int(frame_idx) - int(_FRAME_HISTORY_KEEP)
                for k in list(_FRAME_GALLERY_HISTORY.keys()):
                    if int(k) < cutoff:
                        del _FRAME_GALLERY_HISTORY[k]
        except Exception:
            pass

        if RAID_ASSIGNMENT_DONE and RAIDER_ID is not None:
            raider_track = GALLERY.get(RAIDER_ID)
            raider_missing = (
                raider_track is None
                or raider_track.get("age", 0) > 15
                or (
                    raider_track.get("display_pos") is None
                    and raider_track.get("miss_streak", 0) > 10
                )
            )
            if raider_missing:
                RAIDER_MISSING_FRAMES += 1
            else:
                RAIDER_MISSING_FRAMES = 0

            if RAIDER_MISSING_FRAMES >= 25 and not RAIDER_EXITED:
                RAIDER_EXITED = True
                print(f"RAIDER EXIT LOCK ENGAGED at frame {frame_idx} for ID {RAIDER_ID}")

        

        # ======================================================
        # RAIDER STATS COLLECTION (FIRST PHASE)
        # ======================================================

        if not RAID_ASSIGNMENT_DONE and not RAIDER_EXITED and frame_idx < ASSIGN_FRAME:
            collect_raider_stats(GALLERY, RAIDER_STATS, frame_idx, BAULK_Y)  # raider_logic.collect_raider_stats



   
        # =====================================================
        # REVISED RAIDER ASSIGNMENT (Safety-first logic)
        # ======================================================

        if not RAID_ASSIGNMENT_DONE and not RAIDER_EXITED and frame_idx >= ASSIGN_FRAME:
            best_id, raid_done, ASSIGN_FRAME = assign_raider(  # raider_logic.assign_raider
                GALLERY,
                RAIDER_STATS,
                frame_idx,
                ASSIGN_FRAME,
                BAULK_Y,
                BONUS_Y,
            )
            if raid_done:
                RAIDER_ID = best_id
                raid_summary["raider_id"] = int(RAIDER_ID) if RAIDER_ID is not None else None
                RAID_ASSIGNMENT_DONE = True
                print(f"RAIDER IDENTIFIED (Defense-Half Model): {RAIDER_ID}")

        # ======================================================
        # MODULE-2: INTERACTION LOGIC & PROPOSAL LAYER
        # ======================================================
   
    
        player_states, active_players = build_player_states(GALLERY)  # interaction_logic.build_player_states
        effective_raid_assignment = RAID_ASSIGNMENT_DONE and not RAIDER_EXITED
        interaction_candidates, touch_confirmed = process_interactions(  # interaction_logic.process_interactions
            frame_idx,
            GALLERY,
            player_states,
            active_players,
            RAIDER_ID,
            effective_raid_assignment,
            proposal_engine,
            BONUS_Y,
            BAULK_Y,
            END_LINE_Y,
            LINE_MARGIN,
            LOBBY_LEFT,
            LOBBY_RIGHT,
            touch_confirmed,
            log_event,
        )

        # ------------------------------------------------------
        # EVENT WINDOW CLEANUP / FUTURE INVESTIGATION
        # ------------------------------------------------------

        for event in EVENT_LOG:
            if frame_idx <= event["investigation_until"]:
                # This event is still active for analysis
                pass


        # Optional Debug Visualization
        for cand in interaction_candidates:
            p1, p2 = cand["pair"]

            # SAFETY CHECK: Only draw if both players are currently "Active"
            if p1 in player_states and p2 in player_states:
                mx1, my1 = court_to_pixel(*player_states[p1]["pos"])
                mx2, my2 = court_to_pixel(*player_states[p2]["pos"])

                cv2.line(mat, (mx1, my1), (mx2, my2), (0, 0, 255), 2)

        prev_gray = gray.copy()

        # --- DEBUG BLOCK: Print EVERYTHING in the Gallery ---
        # --- DETAILED DEBUG: Every Parameter + First 5 Color Values ---
     
    
        
        frame_proposals = proposal_engine.finalize_frame_proposals()  # interaction_graph.InteractionProposalEngine.finalize_frame_proposals
        scene_graph = None
        if effective_raid_assignment and proposal_engine.candidate_proposals:
            scene_graph = graph_engine.build_graph(  # interaction_graph.ActiveFactorGraphNetwork.build_graph
                proposal_engine.candidate_proposals,
                GALLERY,
                RAIDER_ID
            )
        confirmed_events = candidate_manager.update(  # temporal_events.TemporalInteractionCandidateManager.update
            frame_idx,
            frame_proposals,
            player_states,
            RAIDER_ID,
            scene_graph,
        )
        for confirmed_event in confirmed_events:
            try:
                confirmed_event['raid_label'] = raid_label
                confirmed_event['raid_index'] = raid_index
                confirmed_event['attacking_team'] = attacking_team
                confirmed_event['video_name'] = os.path.basename(VIDEO_PATH)
            except Exception:
                pass
            _archive_event_mat_window(confirmed_event)
            log_confirmed_event(confirmed_event)
    
        if frame_idx % 30 == 0:
            print(f"\n" + "█"*80)
            print(f" LOG FOR FRAME {frame_idx:05d} | TOTAL INTERACTION PROPOSALS: {len(proposal_engine.candidate_proposals)}")
            print("█"*80)
        
       

            # 3. Existing Gallery Debug (Condensed)
            print(f"\n [GALLERY] ACTIVE TRACKS: {len(GALLERY)}")
            for pid, data in GALLERY.items():
                # 1. Physics & State
                state = data["kf"].statePost.flatten() # [x, y, vx, vy]
            
                # 2. Extract first 5 color values from the 512-dim histogram
                color_sample = data["feat"][:5] 
            
                # 3. Court Coordinates
                m_pos = data["display_pos"] if data["display_pos"] else (0.0, 0.0)
            
                # 4. Bounding Box Dimensions
                bb = data["last_bbox"]
                w, h = (bb[2]-bb[0]), (bb[3]-bb[1])
                x=data['age']
                print(f"PLAYER ID: {pid:02d} | Status: {'[VISIBLE]' if data['age']==0 else f'[LOST (Age:{x})]'}")
                print(f" ├─ PHYSICS: Screen_Pos({state[0]:.0f}, {state[1]:.0f}) | Velocity({state[2]:.2f}, {state[3]:.2f})")
                print(f" ├─ COURT:   Width: {m_pos[0]:.2f}m, Depth: {m_pos[1]:.2f}m")
                print(f" ├─ COLORS:  First 5 of 512 bins: {color_sample}") 
                print(f" ├─ CAMERA:  Optical Flow: {len(data['flow_pts']) if data['flow_pts'] is not None else 0} tracking points")
                print(f" └─ VISUAL:  BBox Height: {h}px | BBox Width: {w}px")
                print("-" * 70)

            # 1. Print Human-Human Interaction (HHI) Triplets
            # ======================================================
            # FORMAL INTERACTION TRIPLET PRINTING <S, I, O>
            # ======================================================
    
            # 1. Print Human-Human Interaction (HHI) Triplets
            hhi_proposals = [p for p in proposal_engine.candidate_proposals if p["type"] == "HHI"]
            if hhi_proposals:
                print(f" [HHI] PLAYER-TO-PLAYER TRIPLETS:")
                for p in hhi_proposals:
                    # Replace IDs with "RAIDER" if they match
                    sub_label = "RAIDER" if p['S'] == RAIDER_ID else f"ID_{p['S']}"
                    obj_label = "RAIDER" if p['O'] == RAIDER_ID else f"ID_{p['O']}"

                    INTERACTION_COUNT+=1
                
                    # Format: [Frame] <Subject, Interaction, Object>
                    triplet = f"<{sub_label}, {p['I']}, {obj_label}>"
                    # Change this line inside your HHI loop:
                    print(f"  ├─ Frame {p['frame']:05d} | {triplet:30} | Rel_Vel: {p['features']['rel_vel']:.2f} | Dist: {p['features']['dist']:.2f}m")

            # 2. Print Human-Line Interaction (HLI) Triplets
            hli_proposals = [p for p in proposal_engine.candidate_proposals if p["type"] == "HLI" and p["features"]["dist"] < 0.5]
            if hli_proposals:
                print(f"\n [HLI] PLAYER-TO-LINE TRIPLETS (Active Proximity):")
                for p in hli_proposals:
                    # Replace Subject ID with "RAIDER" if it matches
                    sub_label = "RAIDER" if p['S'] == RAIDER_ID else f"ID_{p['S']}"
                    INTERACTION_COUNT+=1
                    # Format: [Frame] <Subject, Interaction, Object>
                    triplet = f"<{sub_label}, {p['I']}, {p['O']}>"
                    status = "[TOUCHING]" if p["features"]["active"] else "[NEAR]"
                    print(f"  ├─ Frame {p['frame']:05d} | {triplet:30} | Status: {status:10} | Dist: {p['features']['dist']:.2f}m")


            recent_confirmed = [event for event in CONFIRMED_EVENT_LOG if frame_idx - event["frame"] <= 30]
            if recent_confirmed:
                print(f"\n [CONFIRMED EVENTS] TEMPORAL WINDOW OUTPUT:")
                for event in recent_confirmed:
                    review_tag = " | Visual Review Window" if event["requires_visual_confirmation"] else ""
                    print(
                        f"  â”œâ”€ Frame {event['frame']:05d} | {event['type']} "
                        f"| Window: {event['window_start']:05d}-{event['window_end']:05d} "
                        f"| Conf: {event['confidence']:.2f}"
                        f"| Factor: {event.get('factor_confidence', 0):.2f}{review_tag}"
                    )

            if effective_raid_assignment and proposal_engine.candidate_proposals:
                # Build the Graph and Encode Features
                if scene_graph is None:
                    scene_graph = graph_engine.build_graph(  # interaction_graph.ActiveFactorGraphNetwork.build_graph
                        proposal_engine.candidate_proposals,
                        GALLERY,
                        RAIDER_ID
                    )
            
                print(f"\n [GRAPH] DYNAMIC INTERACTION GRAPH CONSTRUCTED")
                print(f"  ├─ Active Nodes: {graph_engine.active_nodes}")
                print(f"  └─ Perspective Adjacency Matrix (Top 2x2 Sample):")

                if graph_engine.adjacency_matrix.size > 0:
                    print(f"     {graph_engine.adjacency_matrix[:2, :2]}")

                # --- GNN LIVE REASONING INFERENCE ---
                # The GNN returns pure prob-based events
                gnn_out = action_engine.process_frame(scene_graph, RAIDER_ID)
                
                # We adapt the GNN output to match the expected legacy format 
                # so the rest of the frontend and scoring can render successfully.
                action_results = {
                    "actions": gnn_out["emitted_events"],
                    "total_points": {"attacker": 0, "defender": 0},
                    "raid_ended": gnn_out["raid_ended"],
                    "points_scored": 0
                }
                
                # Simple rule bridging: If tackle emitted, defense scores. If returned, raider scores touches.
                touch_count = sum(1 for e in gnn_out["emitted_events"] if e["type"] == "RAIDER_DEFENDER_CONTACT")
                if touch_count > 0 and any(e["type"] == "RAIDER_RETURNED_MIDDLE" for e in gnn_out["emitted_events"]):
                    action_results["total_points"]["attacker"] = touch_count
                if any(e["type"] == "DEFENDER_TACKLE" for e in gnn_out["emitted_events"]):
                    action_results["total_points"]["defender"] = 1
                
                last_action_results = action_results
                # --------------------------------------------------
            
                total_score = action_results['total_points']
                attacker_total = int(total_score.get('attacker', 0))
                defender_total = int(total_score.get('defender', 0))
                attacker_delta = max(0, attacker_total - last_local_totals['attacker'])
                defender_delta = max(0, defender_total - last_local_totals['defender'])
                TOTAL_POINTS = attacker_total - defender_total
                CURRENT_RAID_POINTS = TOTAL_POINTS

                if attacker_delta:
                    remaining_attacker_points = max(
                        0,
                        int(MAX_POINTS_PER_RAID) - int(raid_summary.get("attacking_points_awarded", 0)),
                    )
                    awarded_attacker_delta = min(int(attacker_delta), remaining_attacker_points)
                    if awarded_attacker_delta > 0:
                        team_scores[attacking_team] = int(team_scores.get(attacking_team, 0)) + awarded_attacker_delta
                        raid_summary["attacking_points_awarded"] = int(
                            raid_summary.get("attacking_points_awarded", 0)
                        ) + awarded_attacker_delta
                        raid_summary['score_breakdown'].append({
                            'frame': int(frame_idx),
                            'team': attacking_team,
                            'delta': int(awarded_attacker_delta),
                            'reason': 'attacker_points',
                        })
                if defender_delta:
                    team_scores[defending_team] = int(team_scores.get(defending_team, 0)) + defender_delta
                    raid_summary['score_breakdown'].append({
                        'frame': int(frame_idx),
                        'team': defending_team,
                        'delta': int(defender_delta),
                        'reason': 'defender_points',
                    })

                last_local_totals['attacker'] = attacker_total
                last_local_totals['defender'] = defender_total
            
                print(f"\n [ACTIONS] RECOGNIZED ACTIONS THIS FRAME:")
                for action in action_results["actions"]:
                    conf = action.get("confidence", 0)
                    action_key = (int(frame_idx), str(action.get("type")), str(action.get("description")))
                    if action_key not in action_seen:
                        action_seen.add(action_key)
                        raid_summary["actions"].append({
                            "frame": int(frame_idx),
                            "type": str(action.get("type")),
                            "description": str(action.get("description")),
                            "points": int(action.get("points", 0)),
                            "confidence": float(conf),
                            "highlight": bool(int(action.get("points", 0)) > 0),
                        })
                    print(f"  🎯 {action['type']}: {action.get('description', '')} | Points: {action.get('points', 0)} | Conf: {conf:.2f}")
                print(f"  └─ Frame Points: {action_results['points_scored']} | Total Points: {action_results['total_points']}")
                print(
                    f"     Scoreboard A/D: {total_score['attacker']}/{total_score['defender']} "
                    f"| Net: {TOTAL_POINTS}"
                )
                if action_results.get('confidence_scores'):
                    avg_conf = np.mean(action_results['confidence_scores']) if action_results['confidence_scores'] else 0
                    print(f"     Average Confidence: {avg_conf:.2f}")

                # Display accuracy metrics every 100 frames
                if frame_idx % 100 == 0 and 'accuracy_metrics' in action_results:
                    metrics = action_results['accuracy_metrics']
                    print(f"\n [ACCURACY] Estimated: {metrics['estimated_accuracy']:.1%} | High Conf: {metrics['high_confidence_rate']:.1%} | Total Actions: {metrics['total_actions']}")

            proposal_engine.reset_proposals()

        recent_graph_events = [event for event in CONFIRMED_EVENT_LOG if frame_idx - event["frame"] <= 15]
        graph_panel = render_graph_panel(  # interaction_graph.render_graph_panel
            scene_graph,
            width=GRAPH_W,
            height=GRAPH_H,
            frame_idx=frame_idx,
            recent_events=recent_graph_events,
        )
  
        # Display current scores on mat
        score_ink = (230, 230, 230) if MAT_TRANSPARENT_BG else (0, 0, 0)
        cv2.putText(mat, f"Total Points: {TOTAL_POINTS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_ink, 2)
        cv2.putText(mat, f"Current Raid: {CURRENT_RAID_POINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_ink, 2)

        # Add frame ID to video
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        vis_render = cv2.resize(vis, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
        combined_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        combined_frame[:vis_h, :vis_w] = vis_render
        combined_frame[:COURT_H, vis_w:vis_w+COURT_W] = mat
        combined_frame[:GRAPH_H, vis_w+COURT_W:vis_w+COURT_W+GRAPH_W] = graph_panel

        # --- PUSH LIVE DATA TO REACT API ---
        if LIVE_INPUT_QUEUE.full():
            try: LIVE_INPUT_QUEUE.get_nowait()
            except queue.Empty: pass
        LIVE_INPUT_QUEUE.put(frame.copy())

        if LIVE_FRAME_QUEUE.full():
            try: LIVE_FRAME_QUEUE.get_nowait()
            except queue.Empty: pass
        LIVE_FRAME_QUEUE.put(combined_frame.copy())

        # --- STRUCTURED SNAPSHOTS FOR FRONTEND (non-terminal UI) ---
        gallery_snapshot = []
        try:
            for pid, data in GALLERY.items():
                state = data["kf"].statePost.flatten().tolist()  # [x, y, vx, vy]
                m_pos = data.get("display_pos") if data.get("display_pos") else (0.0, 0.0)
                bb = data.get("last_bbox", (0, 0, 0, 0))
                feat = data.get("feat")
                hsv_bins5 = None
                try:
                    if feat is not None:
                        # First 5 bins of the 512-D HSV histogram (for quick UI diagnostics).
                        hsv_bins5 = [float(x) for x in feat[:5]]
                except Exception:
                    hsv_bins5 = None

                flow_pts = data.get("flow_pts")
                flow_count = 0
                try:
                    flow_count = int(len(flow_pts)) if flow_pts is not None else 0
                except Exception:
                    flow_count = 0
                dominant_colors = _extract_dominant_palette(frame, bb, k=4)
                gallery_snapshot.append({
                    "id": int(pid),
                    "visible": bool(data.get("age", 0) == 0),
                    "age": int(data.get("age", 0)),
                    "screen_pos": [float(state[0]), float(state[1])],
                    "velocity": [float(state[2]), float(state[3])],
                    "court_pos": [float(m_pos[0]), float(m_pos[1])],
                    "bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                    "hsv_bins5": hsv_bins5,
                    "dominant_colors": dominant_colors,
                    "flow_points": flow_count,
                })
            gallery_snapshot.sort(key=lambda x: x["id"])
        except Exception:
            gallery_snapshot = []

        hhi_live = []
        hli_live = []
        try:
            for p in proposal_engine.candidate_proposals[-120:]:
                if p.get("type") == "HHI":
                    feats = p.get("features", {}) or {}
                    hhi_live.append({
                        "frame": int(p.get("frame", frame_idx)),
                        "S": int(p.get("S")) if p.get("S") is not None else None,
                        "I": str(p.get("I")),
                        "O": int(p.get("O")) if p.get("O") is not None else None,
                        "dist": float(feats.get("dist", 0.0)),
                        "rel_vel": float(feats.get("rel_vel", 0.0)),
                    })
                elif p.get("type") == "HLI":
                    feats = p.get("features", {}) or {}
                    hli_live.append({
                        "frame": int(p.get("frame", frame_idx)),
                        "S": int(p.get("S")) if p.get("S") is not None else None,
                        "I": str(p.get("I")),
                        "O": str(p.get("O")),
                        "dist": float(feats.get("dist", 0.0)),
                        "active": bool(feats.get("active", False)),
                    })
        except Exception:
            hhi_live, hli_live = [], []

        action_summary = None
        try:
            if last_action_results is not None:
                action_summary = {
                    "points_scored": int(last_action_results.get("points_scored", 0)),
                    "total_points": last_action_results.get("total_points", {}),
                    "raid_ended": bool(last_action_results.get("raid_ended", False)),
                    "accuracy_metrics": last_action_results.get("accuracy_metrics", {}),
                    "actions": last_action_results.get("actions", [])[:10],
                }
        except Exception:
            action_summary = None

        live_state = {
            "raider_id": RAIDER_ID,
            "score_attacker": int(team_scores.get("A", 0)),
            "score_defender": int(team_scores.get("B", 0)),
            "raid_label": raid_label,
            "raid_index": int(raid_index),
            "video_name": os.path.basename(VIDEO_PATH),
            "attacking_team": attacking_team,
            "defending_team": defending_team,
            "frame_idx": frame_idx,
            "gallery": gallery_snapshot,
            "hhi": hhi_live[-25:],
            "hli": hli_live[-25:],
            "action_summary": action_summary,
            "graph": None,
            "current_raid": {
                "raid_label": raid_label,
                "raid_index": int(raid_index),
                "video_name": os.path.basename(VIDEO_PATH),
                "attacking_team": attacking_team,
                "defending_team": defending_team,
                "raider_id": int(RAIDER_ID) if RAIDER_ID is not None else None,
                "points_this_raid": int(min(MAX_POINTS_PER_RAID, raid_summary.get("attacking_points_awarded", 0))),
                "team_scores": {"A": int(team_scores.get("A", 0)), "B": int(team_scores.get("B", 0))},
                "score_breakdown": raid_summary["score_breakdown"][-20:],
                "actions": raid_summary["actions"][-20:],
            },
            "raid_summaries": (raid_summaries + [{
                "raid_label": raid_label,
                "raid_index": int(raid_index),
                "video_name": os.path.basename(VIDEO_PATH),
                "attacking_team": attacking_team,
                "defending_team": defending_team,
                "raider_id": int(RAIDER_ID) if RAIDER_ID is not None else None,
                "points_this_raid": int(min(MAX_POINTS_PER_RAID, raid_summary.get("attacking_points_awarded", 0))),
                "team_scores": {"A": int(team_scores.get("A", 0)), "B": int(team_scores.get("B", 0))},
                "score_breakdown": raid_summary["score_breakdown"][-20:],
                "actions": raid_summary["actions"][-20:],
                "status": "live",
            }])[-16:],
            "events": [
                {
                    "id": f"{e.get('type')}|{e.get('frame')}|{e.get('subject')}|{e.get('object')}",
                    "raid_label": raid_label,
                    "raid_index": int(raid_index),
                    "frame": e.get("frame"),
                    "type": e.get("type"),
                    "subject": e.get("subject"),
                    "object": e.get("object"),
                    "conf": round(float(e.get("confidence", 0.0)), 3),
                    "factor_conf": round(float(e.get("factor_confidence", 0.0)), 3),
                    "requires_visual_confirmation": bool(e.get("requires_visual_confirmation", False)),
                    "classifier_label": e.get("classifier_label"),
                    "classifier_valid_prob": round(float(e.get("classifier_valid_prob", 0.0)), 3) if e.get("classifier_valid_prob") is not None else None,
                    "guaranteed_by_classifier": bool(e.get("guaranteed_by_classifier", False)),
                }
                for e in CONFIRMED_EVENT_LOG[-5:]  # Send latest 5 events
            ]
        }

        # Compact graph payload for frontend rendering (nodes + edges).
        # This avoids shipping a pre-rendered image and lets React render a 3D graph.
        if scene_graph and scene_graph.get("nodes"):
            graph_nodes = []
            for node in scene_graph.get("nodes", []):
                spatial = node.get("spatial")
                if spatial is None:
                    continue
                graph_nodes.append(
                    {
                        "id": int(node.get("id")),
                        "kind": "player",
                        "role": node.get("role"),
                        "pos": [float(spatial[0]), float(spatial[1])],
                        "track_confidence": float(node.get("track_confidence", 0.0)),
                        "visibility_confidence": float(node.get("visibility_confidence", 0.0)),
                    }
                )

            # Add static line nodes so line factors can connect to something.
            line_nodes = [
                {"id": "LINE_BAULK", "kind": "line", "label": "BAULK", "pos": [5.0, 3.75]},
                {"id": "LINE_BONUS", "kind": "line", "label": "BONUS", "pos": [5.0, 4.75]},
                {"id": "LINE_END", "kind": "line", "label": "END_LINE", "pos": [5.0, 6.5]},
            ]
            graph_nodes.extend(line_nodes)

            graph_edges = []
            for factor in scene_graph.get("pair_factors", []):
                nodes = factor.get("nodes", [])
                if len(nodes) != 2:
                    continue
                features = factor.get("features", {}) or {}
                dist = float(features.get("distance", 0.0))
                proximity = max(0.0, 1.0 - dist / 1.2)
                approach = float(features.get("approach_score", 0.0))
                strength = float(0.5 * proximity + 0.5 * approach)
                graph_edges.append(
                    {
                        "source": int(nodes[0]),
                        "target": int(nodes[1]),
                        "kind": "pair",
                        "type": factor.get("type"),
                        "distance": dist,
                        "weight": max(0.0, min(1.0, strength)),
                    }
                )

            for factor in scene_graph.get("line_factors", []):
                nodes = factor.get("nodes", [])
                if not nodes:
                    continue
                line_name = factor.get("line")
                line_id = {
                    "BAULK": "LINE_BAULK",
                    "BONUS": "LINE_BONUS",
                    "END_LINE": "LINE_END",
                }.get(line_name)
                if line_id is None:
                    continue
                active = bool((factor.get("features", {}) or {}).get("active"))
                graph_edges.append(
                    {
                        "source": int(nodes[0]),
                        "target": line_id,
                        "kind": "line",
                        "type": line_name,
                        "weight": 1.0 if active else 0.35,
                        "active": active,
                    }
                )

            live_state["graph"] = {
                "nodes": graph_nodes,
                "edges": graph_edges,
                "meta": {
                    "best_contact_score": float(scene_graph.get("global_context", {}).get("best_contact_score", 0.0)),
                    "best_containment_score": float(scene_graph.get("global_context", {}).get("best_containment_score", 0.0)),
                    "visible_defenders": int(scene_graph.get("global_context", {}).get("visible_defenders", 0)),
                },
            }
        if LIVE_STATE_QUEUE.full():
            try: LIVE_STATE_QUEUE.get_nowait()
            except queue.Empty: pass
        LIVE_STATE_QUEUE.put(live_state)
        # -----------------------------------

        report_builder.add_frame(frame_idx, combined_frame)  # report_video.ConfirmedInteractionReportBuilder.add_frame
        report_builder.capture_events(confirmed_events)  # report_video.ConfirmedInteractionReportBuilder.capture_events
        if report_builder.has_classifier_inputs():
            classifier_inputs = report_builder.consume_classifier_inputs()  # report_video.ConfirmedInteractionReportBuilder.consume_classifier_inputs
            for item in classifier_inputs:
                try:
                    event = item.get("event", {}) or {}
                    payload = item.get("payload")
                    if not isinstance(payload, dict):
                        payload = {}
                        item["payload"] = payload
                    payload["mat_window"] = _CONFIRMED_EVENT_MAT_ARCHIVE.get(_event_key(event), [])
                    payload["court_meta"] = COURT_META
                except Exception:
                    pass
            exported_windows = dataset_exporter.export_batch(classifier_inputs)  # dataset_exporter.ConfirmedWindowDatasetExporter.export_batch
            classifier_results = classifier_bridge.score_batch(classifier_inputs)  # classifier_bridge.ConfirmedWindowClassifierBridge.score_batch
            apply_classifier_results(CONFIRMED_EVENT_LOG, classifier_results)
            for event in CONFIRMED_EVENT_LOG:
                try:
                    if str(event.get("type")) != "CONFIRMED_RAIDER_DEFENDER_CONTACT":
                        continue
                    if str(event.get("classifier_label")) != "valid":
                        continue
                    event_key = _event_key(event)
                    if event_key in classifier_touch_awards:
                        continue
                    remaining_attacker_points = max(
                        0,
                        int(MAX_POINTS_PER_RAID) - int(raid_summary.get("attacking_points_awarded", 0)),
                    )
                    if remaining_attacker_points <= 0:
                        classifier_touch_awards.add(event_key)
                        continue
                    classifier_touch_awards.add(event_key)
                    team_scores[attacking_team] = int(team_scores.get(attacking_team, 0)) + 1
                    raid_summary["attacking_points_awarded"] = int(
                        raid_summary.get("attacking_points_awarded", 0)
                    ) + 1
                    raid_summary["score_breakdown"].append({
                        "frame": int(event.get("frame", frame_idx)),
                        "team": attacking_team,
                        "delta": 1,
                        "reason": "valid_touch",
                    })
                    raid_summary["actions"].append({
                        "frame": int(event.get("frame", frame_idx)),
                        "type": "VALIDATED_TOUCH_POINT",
                        "description": f"Classifier validated touch for Team {attacking_team}",
                        "points": 1,
                        "confidence": float(event.get("classifier_valid_prob", 0.0) or 0.0),
                        "highlight": True,
                    })
                except Exception:
                    continue
            if exported_windows:
                print(f"[DATASET] Exported {len(exported_windows)} confirmed window(s) to Videos/classifier_dataset")

   
    
        if out is not None:
            out.write(combined_frame)

        if SHOW_BACKEND_WINDOWS:
            cv2.imshow("Video (Integrated)", cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Half Court (2D)", mat)
            cv2.imshow("Interaction Graph", graph_panel)

            key = cv2.waitKey(0 if paused else FPS_DELAY)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('p'):
                paused = not paused
        else:
            # Keep a tiny sleep to avoid burning CPU when running headless.
            time.sleep(max(0.0, float(FPS_DELAY)) / 1000.0)

    print("Total number of Interactions: ", INTERACTION_COUNT)
    
    if out is not None:
        out.release()
    if report_builder.has_segments():
        if os.path.exists(report_output_filename):
            os.remove(report_output_filename)
        wrote_report = report_builder.write_video(report_output_filename, 30.0, (canvas_w, canvas_h))  # report_video.ConfirmedInteractionReportBuilder.write_video
        if wrote_report:
            print(f"Confirmed interaction report saved to: {report_output_filename}")
        else:
            print("Confirmed interaction report could not be written.")
    else:
        print("No confirmed interaction windows captured for report video.")
    cv2.destroyAllWindows()

    # Persist confirmed events so the frontend can show results without reprocessing.
    try:
        archive_dir = os.path.join(BASE_DIR, "Videos")
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, f"confirmed_events_{path_hash}.json")
        latest_path = os.path.join(archive_dir, "confirmed_events_latest.json")
        events_out = []
        try:
            for ev in CONFIRMED_EVENT_LOG:
                row = dict(ev) if isinstance(ev, dict) else {"event": ev}
                row.setdefault("raid_label", raid_label)
                row.setdefault("raid_index", int(raid_index))
                row.setdefault("attacking_team", attacking_team)
                row["mat_window"] = _CONFIRMED_EVENT_MAT_ARCHIVE.get(_event_key(row), [])
                events_out.append(row)
        except Exception:
            events_out = list(CONFIRMED_EVENT_LOG)
        payload = {
            "video_path": VIDEO_PATH,
            "path_hash": path_hash,
            "raid_label": raid_label,
            "raid_index": int(raid_index),
            "attacking_team": attacking_team,
            "processed_video": os.path.basename(output_filename),
            "report_video": os.path.basename(report_output_filename),
            "team_scores": {"A": int(team_scores.get("A", 0)), "B": int(team_scores.get("B", 0))},
            "raid_summaries": (raid_summaries + [{
                "raid_label": raid_label,
                "raid_index": int(raid_index),
                "video_name": os.path.basename(VIDEO_PATH),
                "attacking_team": attacking_team,
                "defending_team": defending_team,
                "raider_id": int(raid_summary.get("raider_id")) if raid_summary.get("raider_id") is not None else None,
                "points_this_raid": int(min(MAX_POINTS_PER_RAID, raid_summary.get("attacking_points_awarded", 0))),
                "team_scores": {"A": int(team_scores.get("A", 0)), "B": int(team_scores.get("B", 0))},
                "score_breakdown": raid_summary["score_breakdown"][-40:],
                "actions": raid_summary["actions"][-40:],
                "status": "completed",
            }])[-16:],
            "court_meta": COURT_META,
            "events": events_out,
        }
        with open(archive_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[ARCHIVE] Wrote confirmed events: {archive_path}")
    except Exception as _exc:
        print(f"[ARCHIVE] Could not write confirmed events archive: {_exc}")


    completed_summary = {
        "raid_label": raid_label,
        "raid_index": int(raid_index),
        "video_name": os.path.basename(VIDEO_PATH),
        "attacking_team": attacking_team,
        "defending_team": defending_team,
        "raider_id": int(raid_summary.get("raider_id")) if raid_summary.get("raider_id") is not None else None,
        "points_this_raid": int(min(MAX_POINTS_PER_RAID, raid_summary.get("attacking_points_awarded", 0))),
        "team_scores": {"A": int(team_scores.get("A", 0)), "B": int(team_scores.get("B", 0))},
        "score_breakdown": raid_summary["score_breakdown"][-40:],
        "actions": raid_summary["actions"][-40:],
        "confirmed_event_count": int(len(CONFIRMED_EVENT_LOG)),
        "status": "completed",
        "processed_video": os.path.basename(output_filename) if out is not None else None,
        "report_video": os.path.basename(report_output_filename) if report_builder.has_segments() else None,
    }
    return completed_summary


TEAM_SCORES = {"A": 0, "B": 0}
RAID_SUMMARIES = []
for _raid_index, _video_path in enumerate(RAID_VIDEO_PATHS, start=1):
    if _USER_QUIT or _USER_RESTART:
        break
    _completed_summary = process_single_raid(_video_path, _raid_index, TEAM_SCORES, RAID_SUMMARIES)
    if _completed_summary is not None:
        RAID_SUMMARIES.append(_completed_summary)
    if _USER_QUIT or _USER_RESTART:
        break

try:
    _videos_dir = os.path.join(BASE_DIR, "Videos")
    os.makedirs(_videos_dir, exist_ok=True)
    _combined_processed = os.path.join(_videos_dir, f"processed_sequence_{SEQUENCE_RUN_ID}.mp4")
    _combined_report = os.path.join(_videos_dir, f"confirmed_report_sequence_{SEQUENCE_RUN_ID}.mp4")
    _combined_processed_latest = os.path.join(_videos_dir, "processed_sequence_latest.mp4")
    _combined_report_latest = os.path.join(_videos_dir, "confirmed_report_sequence_latest.mp4")

    _processed_parts = [
        os.path.join(_videos_dir, item["processed_video"])
        for item in RAID_SUMMARIES
        if item.get("processed_video")
    ]
    _report_parts = [
        os.path.join(_videos_dir, item["report_video"])
        for item in RAID_SUMMARIES
        if item.get("report_video")
    ]

    _combined_processed_path = combine_video_files(_processed_parts, _combined_processed, fps=30.0)
    if _combined_processed_path:
        combine_video_files([_combined_processed_path], _combined_processed_latest, fps=30.0)
        print(f"[ARCHIVE] Combined processed sequence saved to: {_combined_processed_path}")

    _combined_report_path = combine_video_files(_report_parts, _combined_report, fps=30.0)
    if _combined_report_path:
        combine_video_files([_combined_report_path], _combined_report_latest, fps=30.0)
        print(f"[ARCHIVE] Combined confirmed report sequence saved to: {_combined_report_path}")
except Exception as _exc:
    print(f"[ARCHIVE] Could not combine sequence videos: {_exc}")

# After processing ends, detach the live queues so the frontend switches to
# archive mode, but keep the API server running to serve videos + event logs.
try:
    import api_server as _api_server

    _api_server.app.state.frame_queue = None
    _api_server.app.state.state_queue = None
    _api_server.app.state.input_queue = None
except Exception as _exc:
    print(f"[ARCHIVE] Could not detach live queues from API server: {_exc}")

# Keep the backend alive so the frontend can keep playing the final outputs.
try:
    if (not _USER_QUIT) and sys.stdin is not None and sys.stdin.isatty():
        print("\nBackend is now in ARCHIVE mode (API still running on http://localhost:8000).")
        while True:
            cmd = input("Type q to quit, n to restart: ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd in ("n", "new", "restart", "r"):
                _USER_RESTART = True
                break
except (KeyboardInterrupt, EOFError):
    pass

if _USER_QUIT:
    print("[QUIT] User requested quit during processing.")

if _USER_RESTART:
    # Full process restart (useful for iterative runs without manually re-launching Python).
    try:
        print("[RESTART] Restarting backend...")
    except Exception:
        pass
    os.execv(sys.executable, [sys.executable] + sys.argv)
