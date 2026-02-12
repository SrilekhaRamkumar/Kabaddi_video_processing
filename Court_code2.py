import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from threading import Thread
from queue import Queue
import time
import torch
import os
import hashlib
# ======================================================
# PERFORMANCE: THREADED VIDEO READER
# ======================================================

VIDEO_PATH = "Videos/raid1.mp4"

class VideoStream:
    def __init__(self, path, queue_size=5):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)
        
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.001)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

    def running(self):
        return not self.stopped or not self.queue.empty()

# ======================================================
# CONFIG (RESTORED)
# ======================================================

DISPLAY_SCALE = 0.5
FPS_DELAY = 1
CONF_THRESH = 0.4
SMOOTH_ALPHA = 0.25 
MAX_PLAYERS = 8
MAX_AGE=200
LINE_MARGIN = 0.6 

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

L = {k: line_eq(*v) for k, v in lines.items()}
img_pts = np.array([intersect(L[a], L[b]) for a, b in [
    ("end_back", "end_left"), ("end_back", "end_right"),
    ("middle", "end_left"), ("middle", "end_right")
]], dtype=np.float32)

court_pts = np.array([[0, 6.5], [10, 6.5], [0, 0], [10, 0]], dtype=np.float32)
H, _ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)

COURT_W, COURT_H = 400, 260
mat_base = np.ones((COURT_H, COURT_W, 3), dtype=np.uint8) * 235

def court_to_pixel(x, y):
    px = int(x / 10 * COURT_W)
    py = int((6.5 - y) / 6.5 * COURT_H)
    return px, py

# Draw static mat labels/lines
cv2.rectangle(mat_base, court_to_pixel(0, 0), court_to_pixel(10, 6.5), (0, 0, 0), 2)
for y, name in [(3.75, "baulk"), (4.75, "bonus")]:
    cv2.line(mat_base, court_to_pixel(0, y), court_to_pixel(10, y), (0, 0, 0), 1)
    cv2.putText(mat_base, name, (8, court_to_pixel(0, y)[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
for x in [0.75, 9.25]:
    cv2.line(mat_base, court_to_pixel(x, 0), court_to_pixel(x, 6.5), (0, 0, 0), 1)

# ======================================================
# TRACKING HELPERS
# ======================================================
def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf

def extract_embedding(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def draw_3d_bbox(img, x1, y1, x2, y2, depth=12):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1+depth, y1-depth), (x2+depth, y2-depth), (0, 255, 0), 2)
    for p in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        cv2.line(img, p, (p[0]+depth, p[1]-depth), (0, 255, 0), 2)

# ======================================================
# MAIN LOOP
# ======================================================
# To this:

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device used: ",device)
model = YOLO("yolov8n.pt").to(device)

vs = VideoStream(VIDEO_PATH).start()
prev_gray = None
NEXT_ID = 0
GALLERY = {}
frame_idx = 0

cv2.namedWindow("Video (Integrated)")
cv2.namedWindow("Half Court (2D)")

# Cursor Tracking Logic ---
cursor_court_pos = None

def mouse_tracker(event, x, y, flags, param):
    global cursor_court_pos
    if event == cv2.EVENT_MOUSEMOVE:
        # Map scaled window coordinates back to original video size
        orig_x, orig_y = x / DISPLAY_SCALE, y / DISPLAY_SCALE
        pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        cursor_court_pos = (mapped[0], mapped[1])

cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
# ---------------------------------------

# 1. Generate unique path for the output
path_hash = hashlib.md5(VIDEO_PATH.encode()).hexdigest()[:8]
output_filename = f"Videos/processed_{path_hash}.mp4"

# 2. Define canvas dimensions regardless of recording status
vis_w = int(1920 * DISPLAY_SCALE)
vis_h = int(1080 * DISPLAY_SCALE)
canvas_w = vis_w + COURT_W 
canvas_h = max(vis_h, COURT_H)

# 3. Check if file exists - Only initialize 'out' if it does NOT exist
if os.path.exists(output_filename):
    print(f"Playback only: {output_filename} already exists. Skipping recording.")
    out = None 
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (canvas_w, canvas_h))
    print(f"Recording to: {output_filename}")

# The loop 'while vs.running():' will now run for both cases.


while vs.running():
    frame = vs.read()
    if frame is None: continue
    frame_idx += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    mat = mat_base.copy()

    # Draw court lines on video
    for (p1, p2) in lines.values():
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # 1. OPTICAL FLOW MOTION COMPENSATION
    if prev_gray is not None:
        for pid, data in GALLERY.items():
            if data["flow_pts"] is not None:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, data["flow_pts"], None, winSize=(15, 15), maxLevel=2)
                if new_pts is not None:
                    good_new = new_pts[status == 1]
                    good_old = data["flow_pts"][status == 1]
                    if len(good_new) > 5:
                        dx, dy = np.mean(good_new - good_old, axis=0)
                        data["kf"].statePost[0] += dx
                        data["kf"].statePost[1] += dy
                        data["flow_pts"] = good_new.reshape(-1, 1, 2)

    # 2. YOLO DETECTION
    results = model(frame, device=device, verbose=False)[0]
    detections = []
    for box in results.boxes:
        if int(box.cls[0]) != 0 or float(box.conf[0]) < CONF_THRESH: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        emb = extract_embedding(frame, (x1, y1, x2, y2))
        if emb is not None: detections.append({"bbox": (x1, y1, x2, y2), "foot": ((x1+x2)//2, y2), "emb": emb})

    # 3. TRACK PREDICTION & MATCHING
    track_ids = list(GALLERY.keys())
    predictions = [GALLERY[pid]["kf"].predict() for pid in track_ids]
    
    matched_tracks, matched_dets = set(), set()
    if predictions and detections:
        cost_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            px, py = pred[0][0], pred[1][0]
            for j, det in enumerate(detections):
                fx, fy = det["foot"]
                cost_matrix[i, j] = 0.7 * (np.sqrt((px-fx)**2 + (py-fy)**2)/200) + 0.3 * (1 - cosine(det["emb"], GALLERY[track_ids[i]]["feat"]))
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1.2:
                pid, det = track_ids[r], detections[c]
                GALLERY[pid]["kf"].correct(np.array([[np.float32(det["foot"][0])], [np.float32(det["foot"][1])]]))
                GALLERY[pid]["feat"] = 0.8 * GALLERY[pid]["feat"] + 0.2 * det["emb"]
                GALLERY[pid]["age"] = 0
                GALLERY[pid]["last_bbox"] = det["bbox"]
                
                # RESTORED: Foot Ellipse and Labels
                fx, fy = det["foot"]
                x1, y1, x2, y2 = det["bbox"]
                draw_3d_bbox(vis, x1, y1, x2, y2)
                
                ew, eh = int((x2-x1)*0.7), int((x2-x1)*0.22)
                cv2.ellipse(vis, ((fx, fy), (ew, eh), 0), (255, 0, 0), 2)
                cv2.putText(vis, f"ID {pid}", (x1, max(30, y1-12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.circle(vis, (fx, fy), 5, (255, 0, 0), -1)

                # PERFORMANCE TWEAK: Only refresh flow points periodically
                if frame_idx % 4 == 0:
                    pts = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=8, qualityLevel=0.3, minDistance=5)
                    if pts is not None:
                        pts[:, :, 0] += x1; pts[:, :, 1] += y1
                        GALLERY[pid]["flow_pts"] = pts
                
                matched_tracks.add(pid); matched_dets.add(c)

    # 4. NEW TRACKS & AGING
    for j, det in enumerate(detections):
        if j not in matched_dets and len(GALLERY) < MAX_PLAYERS:
            GALLERY[NEXT_ID] = {"feat": det["emb"], "kf": create_kalman(*det["foot"]), "age": 0, "display_pos": None, "flow_pts": None, "last_bbox": det["bbox"]}
            NEXT_ID += 1

    # 5. MAT RENDERING & DIRECTION ARROWS
    dead = [pid for pid, d in GALLERY.items() if pid not in matched_tracks and d["age"] > MAX_AGE]
    for pid in dead: del GALLERY[pid]
    
    for pid, data in GALLERY.items():
        if pid not in matched_tracks: data["age"] += 1
        
        pred = data["kf"].statePost
        px, py, vx, vy = pred[0][0], pred[1][0], pred[2][0], pred[3][0]
        
        # RESTORED: Directional Arrows
        angle = np.degrees(np.arctan2(vy, vx))
        ex = int(px + 40 * np.cos(np.radians(angle)))
        ey = int(py + 40 * np.sin(np.radians(angle)))
        cv2.arrowedLine(vis, (int(px), int(py)), (ex, ey), (0, 255, 255), 2)

        # Homography Mapping
        mapped = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)[0][0]
        cx, cy = mapped
        if LINE_MARGIN < cx < 10 - LINE_MARGIN and LINE_MARGIN < cy < 6.5 - LINE_MARGIN:
            if data["display_pos"] is None: data["display_pos"] = (cx, cy)
            else:
                ox, oy = data["display_pos"]
                data["display_pos"] = (ox + SMOOTH_ALPHA*(cx-ox), oy + SMOOTH_ALPHA*(cy-oy))
            
            mx, my = court_to_pixel(*data["display_pos"])
            
            # RESTORED: Dynamic Dot Sizing & Box
            bh = data["last_bbox"][3] - data["last_bbox"][1]
            scale = np.clip((bh - 80) / 220, 0, 1)
            dot_rad = int(5 + scale * 8)
            
            cv2.circle(mat, (mx, my), dot_rad, (255, 0, 0), -1)
            cv2.rectangle(mat, (mx-20, my-20), (mx+20, my+20), (0, 0, 0), 1)
            cv2.putText(mat, f"{pid}", (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- ADD THIS: Render live cursor dot on mat ---
    if cursor_court_pos is not None:
        cx, cy = cursor_court_pos
        # Only draw if within reasonable court bounds
        if -1 < cx < 11 and -1 < cy < 7.5:
            mx, my = court_to_pixel(cx, cy)
            cv2.circle(mat, (mx, my), 7, (0, 255, 255), -1) # Yellow solid dot
            cv2.circle(mat, (mx, my), 9, (0, 0, 0), 1)      # Black outline
    # -----------------------------------------------

    prev_gray = gray.copy()


    # Print Debug Info every 30 frames
    if frame_idx % 30 == 0:
        print(f"\n--- Frame: {frame_idx} | Active Players: {len(GALLERY)} ---")
        for pid, data in GALLERY.items():
            # 1. Access Kalman State (x, y, vx, vy)
            state = data["kf"].statePost.flatten()
            
            # 2. Access SGM (Spatial-Global Matching / Appearance Feature) 
            # Printing only the first 5 elements of the 512-dim histogram for brevity
            feature_sample = data["feat"][:5] 
            
            print(f"ID {pid:2} | Pos: ({state[0]:4.1f}, {state[1]:4.1f}) | "
                f"V: ({state[2]:4.1f}, {state[3]:4.1f}) | "
                f"Age: {data['age']:3} | SGM Sample: {feature_sample}")
            
  
    vis_render = cv2.resize(vis, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    combined_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    combined_frame[:vis_h, :vis_w] = vis_render
    combined_frame[:COURT_H, vis_w:vis_w+COURT_W] = mat
    
    # Write to file
    if out is not None:
        out.write(combined_frame)
    cv2.imshow("Video (Integrated)", cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Half Court (2D)", mat)
    
    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'): break


if 'out' in locals():
    out.release()
cv2.destroyAllWindows()