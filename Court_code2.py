import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
from scipy.optimize import linear_sum_assignment
from threading import Thread
from queue import Queue
import time
import torch
import os
import hashlib

#COMMIT CHECK
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
MODEL1 = "yolo26m.pt"
cursor_court_pos = None
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

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    MODEL1="yolov8n.pt"
print("Device used: ",device)
model = YOLO(MODEL1).to(device)
# model = RTDETR("rtdetr-l.pt").to(device)

vs = VideoStream(VIDEO_PATH).start()
prev_gray = None
NEXT_ID = 0
GALLERY = {}
frame_idx = 0

# ======================================================
# RAIDER IDENTIFICATION (Single-side logic)
# ======================================================
RAIDER_ID = None
RAIDER_STATS = {}
RAID_ASSIGNMENT_DONE = False
RAIDER_CONV_ACCUM = {}
MIN_FRAMES_FOR_DECISION = 40

BAULK_Y = 3.75
ASSIGN_FRAME = 70   # assign raider after this frame

# Cursor Tracking Logic
cursor_court_pos = None

def mouse_tracker(event, x, y, flags, param):
    global cursor_court_pos
    if event == cv2.EVENT_MOUSEMOVE:
        # Map scaled window coordinates back to original video size
        orig_x, orig_y = x / DISPLAY_SCALE, y / DISPLAY_SCALE
        pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        cursor_court_pos = (mapped[0], mapped[1])

cv2.namedWindow("Video (Integrated)")
cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
cv2.namedWindow("Half Court (2D)")



cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
# ---------------------------------------


path_hash = hashlib.md5(VIDEO_PATH.encode()).hexdigest()[:8]
output_filename = f"Videos/processed_{path_hash}.mp4"


vis_w = int(1920 * DISPLAY_SCALE)
vis_h = int(1080 * DISPLAY_SCALE)
canvas_w = vis_w + COURT_W 
canvas_h = max(vis_h, COURT_H)

if os.path.exists(output_filename):
    print(f"Playback only: {output_filename} already exists. Skipping recording.")
    out = None 
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (canvas_w, canvas_h))
    print(f"Recording to: {output_filename}")

def mouse_tracker(event, x, y, flags, param):
    global cursor_court_pos

    if event == cv2.EVENT_MOUSEMOVE:

        # Convert displayed coordinates back to original frame scale
        orig_x = x / DISPLAY_SCALE
        orig_y = y / DISPLAY_SCALE

        pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)

        mapped = cv2.perspectiveTransform(pt, H)

        cursor_court_pos = (mapped[0][0][0], mapped[0][0][1])


while vs.running():
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
                if RAID_ASSIGNMENT_DONE and pid == RAIDER_ID:
                    cv2.putText(
                        vis,
                        "RAIDER",
                        (x1 + 120, max(30, y1 - 12)),  # beside ID text
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),   # yellow for distinction
                        3
                    )


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
            

            bh = data["last_bbox"][3] - data["last_bbox"][1]
            scale = np.clip((bh - 80) / 220, 0, 1)
            dot_rad = int(5 + scale * 8)
            
            cv2.circle(mat, (mx, my), dot_rad, (255, 0, 0), -1)
            if pid == RAIDER_ID:
                cv2.circle(mat, (mx, my), dot_rad + 6, (0, 0, 255), 3)

            cv2.rectangle(mat, (mx-20, my-20), (mx+20, my+20), (0, 0, 0), 1)
            cv2.putText(mat, f"{pid}", (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        

    # ======================================================
    # RAIDER STATS COLLECTION (FIRST PHASE)
    # ======================================================

    if not RAID_ASSIGNMENT_DONE and frame_idx < ASSIGN_FRAME:

        for pid, data in GALLERY.items():

            if data["display_pos"] is None:
                continue

            cx, cy = data["display_pos"]

            if pid not in RAIDER_STATS:
                RAIDER_STATS[pid] = {
                    "first_seen": frame_idx,
                    "min_y": cy,
                    "max_y": cy,
                    "vy_list": [],
                    "behind_baulk_frames": 0,
                    "frames": 0
                }

            rec = RAIDER_STATS[pid]

            rec["min_y"] = min(rec["min_y"], cy)
            rec["max_y"] = max(rec["max_y"], cy)
            rec["frames"] += 1

            # Depth-based defender prior
            if cy > BAULK_Y:
                rec["behind_baulk_frames"] += 1

            state = data["kf"].statePost.flatten()
            vy = state[3]
            rec["vy_list"].append(vy)



   
    # =====================================================
    # REVISED RAIDER ASSIGNMENT (Safety-first logic)
    # ======================================================

    if not RAID_ASSIGNMENT_DONE and frame_idx >= ASSIGN_FRAME:

        best_score = -1e9
        best_id = None

        visible_players = [pid for pid, d in GALLERY.items()
                        if d["display_pos"] is not None and d["age"] == 0]

        if len(visible_players) < 3:
            ASSIGN_FRAME += 10
        else:
            for pid in visible_players:

                rec = RAIDER_STATS.get(pid, None)
                if rec is None or rec["frames"] < 20:
                    continue


                # -------------------------------------------------
                # ELIMINATE PLAYERS MOSTLY BEHIND BONUS LINE
                # -------------------------------------------------

                BONUS_Y = 2.75  # already defined court bonus line depth

                behind_bonus_frames = 0

                # Count how many frames player stayed deep behind bonus
                min_depth = rec["max_y"]

                # If player’s minimum depth is always high,
                # it means he never came forward (strong defender signal)
                if rec["min_y"] > BONUS_Y:
                    continue

                # Strong elimination rule:
                # If most observed frames are behind bonus, remove
                deep_ratio = sum(1 for y in [rec["max_y"]] if y > BONUS_Y) / rec["frames"]

                if rec["min_y"] > BONUS_Y - 0.3:
                    continue


                xi, yi = GALLERY[pid]["display_pos"]
                state_i = GALLERY[pid]["kf"].statePost.flatten()
                vxi, vyi = state_i[2], state_i[3]

                # -----------------------------
                # 1. DEPTH DOMINANCE SCORE
                # -----------------------------
                depths = [GALLERY[o]["display_pos"][1]
                        for o in visible_players if o != pid]

                if len(depths) == 0:
                    continue

                depth_rank = sum(yi > d for d in depths)

                # -----------------------------
                # 2. CONVERGENCE SCORE
                # -----------------------------
                convergence_count = 0
                close_players = 0

                for other in visible_players:
                    if other == pid:
                        continue

                    xj, yj = GALLERY[other]["display_pos"]
                    state_j = GALLERY[other]["kf"].statePost.flatten()
                    vxj, vyj = state_j[2], state_j[3]

                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

                    if dist < 2.5:
                        close_players += 1

                    dir_vec = np.array([dx/dist, dy/dist])
                    vel_vec = np.array([vxj, vyj])

                    if np.dot(vel_vec, dir_vec) > 0:
                        convergence_count += 1

                # -----------------------------
                # 3. MOVEMENT UNIQUENESS
                # Raider usually has higher speed magnitude
                # -----------------------------
                speed = np.sqrt(vxi*vxi + vyi*vyi)

                # -----------------------------
                # 4. ENTRY PRIOR (appeared later boost)
                # -----------------------------
                entry_prior = rec["first_seen"] / frame_idx

                # -----------------------------
                # FINAL SCORE
                # -----------------------------
                score = (
                    depth_rank * 4.0 +
                    convergence_count * 3.5 +
                    close_players * 2.0 +
                    speed * 1.5 +
                    entry_prior * 5.0
                )

                if score > best_score:
                    best_score = score
                    best_id = pid

            if best_id is not None:
                RAIDER_ID = best_id
                RAID_ASSIGNMENT_DONE = True
                print(f"🔥 RAIDER IDENTIFIED (Defense-Half Model): {RAIDER_ID}")

    # ======================================================
    # MODULE-2: INTERACTION PROPOSAL LAYER
    # ======================================================

    active_players = []
    player_states = {}

    for pid, data in GALLERY.items():
        if data["age"] == 0 and data["display_pos"] is not None:
            active_players.append(pid)
            state = data["kf"].statePost.flatten()
            player_states[pid] = {
                "pos": data["display_pos"],          # (cx, cy) in meters
                "vel": (state[2], state[3])         # (vx, vy) in pixels/frame
            }

    interaction_candidates = []

    DIST_THRESH = 1.0  # meters

    if RAIDER_ID is not None and RAID_ASSIGNMENT_DONE:

        if RAIDER_ID in player_states:

            xi, yi = player_states[RAIDER_ID]["pos"]
            vxi, vyi = player_states[RAIDER_ID]["vel"]

            for pid, pdata in player_states.items():

                if pid == RAIDER_ID:
                    continue

                xj, yj = pdata["pos"]
                vxj, vyj = pdata["vel"]

                d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                v_rel = np.sqrt((vxi - vxj)**2 + (vyi - vyj)**2)

                if d < DIST_THRESH:
                    interaction_candidates.append({
                        "pair": (RAIDER_ID, pid),
                        "distance": d,
                        "v_rel": v_rel
                    })


    # Optional Debug Visualization
    for cand in interaction_candidates:
        p1 = cand["pair"][0]
        p2 = cand["pair"][1]

        mx1, my1 = court_to_pixel(*player_states[p1]["pos"])
        mx2, my2 = court_to_pixel(*player_states[p2]["pos"])

        cv2.line(mat, (mx1, my1), (mx2, my2), (0, 0, 255), 2)

    prev_gray = gray.copy()

    # --- DEBUG BLOCK: Print EVERYTHING in the Gallery ---
    # --- DETAILED DEBUG: Every Parameter + First 5 Color Values ---
    if frame_idx % 30 == 0:
        print(f"\n" + "█"*70)
        print(f" LOG FOR FRAME {frame_idx:05d} | TOTAL MEMORY ENTRIES: {len(GALLERY)}")
        print("█"*70)
        
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
  
    vis_render = cv2.resize(vis, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    combined_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    combined_frame[:vis_h, :vis_w] = vis_render
    combined_frame[:COURT_H, vis_w:vis_w+COURT_W] = mat
    
    if out is not None:
        out.write(combined_frame)
    cv2.imshow("Video (Integrated)", cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Half Court (2D)", mat)
    
    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'): break


if 'out' in locals():
    out.release()
cv2.destroyAllWindows()