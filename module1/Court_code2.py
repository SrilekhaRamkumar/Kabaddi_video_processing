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
from action_recognition import ActionRecognitionEngine

#COMMIT CHECK 8/3
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
    
class InteractionProposalEngine:
    """Encodes atomic actions into <S, I, O> triplets with strict per-frame uniqueness."""
    def __init__(self):
        self.candidate_proposals = []
        self._frame_cache = {} # Key: (frame, type, S, O) -> value: proposal_dict

    def reset_proposals(self):
        """Clears accumulated triplets for the 30-frame reasoning window."""
        self.candidate_proposals = []
        self._frame_cache = {}

    def _add_unique_proposal(self, frame_idx, proposal):
        """Maintains only the highest-confidence (closest) triplet per frame."""
        # Create a unique key for this specific interaction event
        key = (frame_idx, proposal["type"], proposal["S"], proposal["O"])
        
        if key not in self._frame_cache:
            # First time seeing this interaction in this frame
            self._frame_cache[key] = proposal
        else:
            # If it already exists, only update if the new one is 'closer' (higher contact chance)
            if proposal["features"]["dist"] < self._frame_cache[key]["features"]["dist"]:
                self._frame_cache[key] = proposal

    def finalize_frame_proposals(self):
        """Converts the unique frame cache into the main proposal list for the Graph Engine."""
        # This is called once per frame after all detections are processed
        for proposal in self._frame_cache.values():
            self.candidate_proposals.append(proposal)
        # Clear cache for the next frame while keeping the candidate_proposals history
        self._frame_cache = {}

    def encode_hhi(self, frame_idx, raider_id, defender_id, r_pos, d_pos, r_vel, d_vel, r_feat, d_feat):
        dist = np.sqrt((r_pos[0]-d_pos[0])**2 + (r_pos[1]-d_pos[1])**2)
        proposal = {
            "frame": frame_idx,
            "type": "HHI",
            "S": raider_id, "O": defender_id, "I": "POTENTIAL_CONTACT",
            "features": {
                "dist": dist, 
                "rel_vel": np.linalg.norm(np.array(r_vel)-np.array(d_vel)),
                "mask": [d_pos[0]-r_pos[0], d_pos[1]-r_pos[1]], 
                "emb": (0.5*r_feat + 0.5*d_feat).tolist()
            }
        }
        self._add_unique_proposal(frame_idx, proposal)

    def encode_hli(self, frame_idx, player_id, line_name, p_pos, line_y):
        dist_to_line = abs(p_pos[1] - line_y)
        proposal = {
            "frame": frame_idx,
            "type": "HLI",
            "S": player_id, "O": line_name, "I": "LINE_PROXIMITY",
            "features": {"dist": dist_to_line, "active": dist_to_line < 0.25}
        }
        self._add_unique_proposal(frame_idx, proposal)

class ActiveFactorGraphNetwork:
    """Constructs a structured graph from triplets using AFGN and Lee et al. methods, extended with factor nodes for third-order interactions."""
    def __init__(self, top_k=4):
        self.top_k = top_k
        self.active_nodes = []
        self.adjacency_matrix = None
        self.factor_nodes = []  # For third-order interactions

    def build_graph(self, proposals, gallery, raider_id):
        # 1. AFGN: Calculate Influence Weights for Active Selection
        # Initialize only for players currently in the gallery
        influence = {pid: 0.0 for pid in gallery if gallery[pid]["age"] == 0}
        
        for p in proposals:
            if p["type"] == "HHI":
                # Enhanced influence weighting: distance, relative velocity, and feature similarity
                dist_weight = 1.0 / (p["features"]["dist"] + 1e-6)
                vel_weight = p["features"]["rel_vel"]
                # Feature similarity using dot product of embedding halves (simplified)
                emb = np.array(p["features"]["emb"])
                half = len(emb) // 2
                feat_sim = np.dot(emb[:half], emb[half:]) / (np.linalg.norm(emb[:half]) * np.linalg.norm(emb[half:]) + 1e-6)
                weight = dist_weight * vel_weight * feat_sim
                
                # SAFETY CHECK: Only update influence if the player is still 'Active'
                if p["S"] in influence:
                    influence[p["S"]] += weight
                if p["O"] in influence:
                    influence[p["O"]] += weight
                    
        # 2. AFGN: Top-k Pruning (Ensures Raider is always included)
        sorted_ids = sorted(influence.keys(), key=lambda x: influence[x], reverse=True)
        self.active_nodes = [pid for pid in sorted_ids if pid != raider_id][:self.top_k - 1]
        self.active_nodes.append(raider_id) # The 'Subject' is always active

        # 3. Lee et al: Construct Adjacency Matrix using Diagonal Ratios
        n = len(self.active_nodes)
        self.adjacency_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                id_i, id_j = self.active_nodes[i], self.active_nodes[j]
                
                # Get Bounding Box Diagonals for perspective correction
                bb_i, bb_j = gallery[id_i]["last_bbox"], gallery[id_j]["last_bbox"]
                d_i = np.sqrt((bb_i[2]-bb_i[0])**2 + (bb_i[3]-bb_i[1])**2)
                d_j = np.sqrt((bb_j[2]-bb_j[0])**2 + (bb_j[3]-bb_j[1])**2)
                
                # Diagonal Ratio Formula: min(di, dj) / max(di, dj)
                r_ij = min(d_i, d_j) / max(d_i, d_j)
                self.adjacency_matrix[i, j] = r_ij

        # 4. AFGN: Factor Node Construction for Third-Order Interactions
        self.factor_nodes = []
        from itertools import combinations
        for triplet in combinations(self.active_nodes, 3):  # Third-order triplets
            pid1, pid2, pid3 = triplet
            # Extract features for the triplet, with safety checks
            pos1 = gallery[pid1]["display_pos"]
            pos2 = gallery[pid2]["display_pos"]
            pos3 = gallery[pid3]["display_pos"]
            if pos1 is None or pos2 is None or pos3 is None:
                continue
            pos1 = np.array(pos1)
            pos2 = np.array(pos2)
            pos3 = np.array(pos3)
            feat1 = np.array(gallery[pid1]["feat"])
            feat2 = np.array(gallery[pid2]["feat"])
            feat3 = np.array(gallery[pid3]["feat"])
            dist12 = np.linalg.norm(pos1 - pos2)
            dist13 = np.linalg.norm(pos1 - pos3)
            dist23 = np.linalg.norm(pos2 - pos3)
            # Aggregate embeddings and distances
            factor_feat = {
                "triplet": triplet,
                "features": {
                    "distances": [dist12, dist13, dist23],
                    "embeddings": np.mean([feat1, feat2, feat3], axis=0).tolist()
                }
            }
            self.factor_nodes.append(factor_feat)

        return self.package_features(gallery)

    def package_features(self, gallery):
        """Encodes Spatiotemporal features for the Factor Graph."""
        graph_data = {"nodes": [], "edges": self.adjacency_matrix.tolist(), "factor_nodes": self.factor_nodes}
        for pid in self.active_nodes:
            state = gallery[pid]["kf"].statePost.flatten()
            node_feat = {
                "id": pid,
                "role": "RAIDER" if pid == RAIDER_ID else "DEFENDER",
                "motion": [float(state[2]), float(state[3])], # Velocity (vx, vy)
                "visual": gallery[pid]["feat"].tolist(),     # HSV Embedding
                "spatial": gallery[pid]["display_pos"]        # Normalized Mat (x, y)
            }
            graph_data["nodes"].append(node_feat)
        return graph_data

# ======================================================
# CONFIG (RESTORED)
# ======================================================

DISPLAY_SCALE = 0.5
FPS_DELAY = 1
CONF_THRESH = 0.4
SMOOTH_ALPHA = 0.25 
MAX_PLAYERS = 8
MAX_AGE=200
MODEL1 = "yolov8n.pt"
cursor_court_pos = None
LINE_MARGIN = 0.6 
proposal_engine = InteractionProposalEngine()
graph_engine = ActiveFactorGraphNetwork(top_k=4)
action_engine = ActionRecognitionEngine()

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

def log_event(event_type, player_id, frame_idx):
    EVENT_LOG.append({
        "frame": frame_idx,
        "player": player_id,
        "type": event_type,
        "investigation_until": frame_idx + INTERACTION_WINDOW
    })

# ======================================================
# PHASE 2: ACTION RECOGNITION & SCORING
# ======================================================

# Event and interaction tracking (restored)
EVENT_LOG = []
touch_confirmed = False
INTERACTION_WINDOW = 20  # frames for future investigation
INTERACTION_COUNT = 0

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

while vs.running():
    global TOTAL_POINTS, CURRENT_RAID_POINTS
    if 'TOTAL_POINTS' not in globals():
        TOTAL_POINTS = 0
        CURRENT_RAID_POINTS = 0
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
    cv2.putText(mat, f"Total Points: {TOTAL_POINTS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(mat, f"Current Raid: {CURRENT_RAID_POINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
        else:
            data["display_pos"]=None

        

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

                #BONUS_Y = 2.75  # already defined court bonus line depth

                behind_bonus_frames = 0

                # Count how many frames player stayed deep behind bonus
                min_depth = rec["max_y"]

                # If player’s minimum depth is always high,
                # it means he never came forward (strong defender signal)
                if rec["min_y"] > BAULK_Y-1:
                    continue

                avg_vy = np.mean(rec["vy_list"]) if rec["vy_list"] else 0
                if abs(avg_vy) < 0.05: # Threshold for 'stationary'
                    continue

                # ADDITIONAL SAFETY: Eliminate players who appeared deep in the court (close to the end line) early on.
                

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
                print(f"RAIDER IDENTIFIED (Defense-Half Model): {RAIDER_ID}")

    # ======================================================
    # MODULE-2: INTERACTION LOGIC & PROPOSAL LAYER
    # ======================================================
   
    
    # NEW: Define player_states for existing features and HHI/HLI encoding
    player_states = {}
    active_players = [pid for pid, d in GALLERY.items() if d["age"] == 0 and d["display_pos"] is not None]
    
    for pid in active_players:
        state = GALLERY[pid]["kf"].statePost.flatten()
        player_states[pid] = {
            "pos": GALLERY[pid]["display_pos"],
            "vel": (state[2], state[3]),
            "feat": GALLERY[pid]["feat"]
        }
    
    interaction_candidates = [] # For your existing line-drawing feature

    if RAID_ASSIGNMENT_DONE and RAIDER_ID in GALLERY:
        r_data = GALLERY[RAIDER_ID]
        if r_data["display_pos"]:
            r_pos = r_data["display_pos"]
            r_vel = (r_data["kf"].statePost[2][0], r_data["kf"].statePost[3][0])

            # B. Generate Human-Human (HHI) Proposals
            for pid in active_players:
                if pid == RAIDER_ID: continue
                
                d_pos = player_states[pid]["pos"]
                d_vel = player_states[pid]["vel"]
                d_feat = player_states[pid]["feat"]
                
                dist = np.sqrt((r_pos[0]-d_pos[0])**2 + (r_pos[1]-d_pos[1])**2)
                
                # Geometric Gating: Propose within 1.5m
                if dist < 1.5:
                    proposal_engine.encode_hhi(frame_idx,RAIDER_ID, pid, r_pos, d_pos, r_vel, d_vel, r_data["feat"], d_feat)
                    
                    # Populate candidates for your existing line drawing feature
                    interaction_candidates.append({"pair": (RAIDER_ID, pid), "distance": dist})
                    
                    if not touch_confirmed and dist < 1.0:
                        touch_confirmed = True
                        log_event("RAIDER_DEFENDER_CONTACT", RAIDER_ID, frame_idx)

            # C. Generate Human-Line (HLI) Proposals
            proposal_engine.encode_hli(frame_idx,RAIDER_ID, "BONUS", r_pos, BONUS_Y)
            proposal_engine.encode_hli(frame_idx,RAIDER_ID, "BAULK", r_pos, BAULK_Y)
            
            for pid, pdata in GALLERY.items():
                if pdata["display_pos"]:
                    proposal_engine.encode_hli(frame_idx,pid, "END_LINE", pdata["display_pos"], END_LINE_Y)

    # ------------------------------------------------------
    # DEFENDER TOUCHING END LINE
    # ------------------------------------------------------

    for pid, pdata in player_states.items():

        if RAIDER_ID is not None and pid != RAIDER_ID:

            px, py = pdata["pos"]

            if abs(py - END_LINE_Y) < LINE_MARGIN:
                log_event("DEFENDER_ENDLINE_TOUCH", pid, frame_idx)

    # ------------------------------------------------------
    # RAIDER TOUCHING BONUS / BAULK
    # ------------------------------------------------------

    if RAID_ASSIGNMENT_DONE and RAIDER_ID in player_states:

        rx, ry = player_states[RAIDER_ID]["pos"]

        if abs(ry - BONUS_Y) < LINE_MARGIN:
            log_event("RAIDER_BONUS_TOUCH", RAIDER_ID, frame_idx)

        if abs(ry - BAULK_Y) < LINE_MARGIN:
            log_event("RAIDER_BAULK_TOUCH", RAIDER_ID, frame_idx)
    
    # ------------------------------------------------------
    # LOBBY ENTRY RESTRICTION (BEFORE TOUCH)
    # ------------------------------------------------------

    for pid, pdata in player_states.items():

        px, py = pdata["pos"]

        if not touch_confirmed:

            if px < LOBBY_LEFT or px > LOBBY_RIGHT:
                log_event("ILLEGAL_LOBBY_ENTRY_BEFORE_TOUCH", pid, frame_idx)
    
    # ------------------------------------------------------
    # RAIDER RETURN TO MIDDLE (AFTER TOUCH)
    # ------------------------------------------------------

    if touch_confirmed and RAIDER_ID in player_states:

        rx, ry = player_states[RAIDER_ID]["pos"]

        if ry < 0.8:
            log_event("RAIDER_RETURNED_MIDDLE", RAIDER_ID, frame_idx)

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
     
    
        
    proposal_engine.finalize_frame_proposals()
    
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


        if RAID_ASSIGNMENT_DONE and proposal_engine.candidate_proposals:
            # Build the Graph and Encode Features
            scene_graph = graph_engine.build_graph(
                proposal_engine.candidate_proposals, 
                GALLERY, 
                RAIDER_ID
            )
            
            print(f"\n [GRAPH] DYNAMIC INTERACTION GRAPH CONSTRUCTED")
            print(f"  ├─ Active Nodes: {graph_engine.active_nodes}")
            print(f"  └─ Perspective Adjacency Matrix (Top 2x2 Sample):")

            if graph_engine.adjacency_matrix.size > 0:
                print(f"     {graph_engine.adjacency_matrix[:2, :2]}")

            # PHASE 2: Action Recognition and Point Calculation
            action_results = action_engine.process_frame_actions(
                scene_graph, 
                proposal_engine.candidate_proposals, 
                RAIDER_ID, 
                frame_idx
            )
            
            # Update global scores
            TOTAL_POINTS = action_results['total_points']
            CURRENT_RAID_POINTS += action_results['points_scored']
            
            if action_results.get('raid_ended', False):
                CURRENT_RAID_POINTS = 0  # Reset for new raid
            
            print(f"\n [ACTIONS] RECOGNIZED ACTIONS THIS FRAME:")
            for action in action_results["actions"]:
                conf = action.get("confidence", 0)
                print(f"  ├─ {action['type']}: {action['description']} | Points: {action.get('points', 0)} | Conf: {conf:.2f}")
            print(f"  └─ Frame Points: {action_results['points_scored']} | Total Points: {action_results['total_points']}")
            if action_results.get('confidence_scores'):
                avg_conf = np.mean(action_results['confidence_scores']) if action_results['confidence_scores'] else 0
                print(f"     Average Confidence: {avg_conf:.2f}")

            # Display accuracy metrics every 100 frames
            if frame_idx % 100 == 0 and 'accuracy_metrics' in action_results:
                metrics = action_results['accuracy_metrics']
                print(f"\n [ACCURACY] Estimated: {metrics['estimated_accuracy']:.1%} | High Conf: {metrics['high_confidence_rate']:.1%} | Total Actions: {metrics['total_actions']}")

        proposal_engine.reset_proposals()
  
    # Display current scores on mat
    cv2.putText(mat, f"Total Points: {TOTAL_POINTS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(mat, f"Current Raid: {CURRENT_RAID_POINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Add frame ID to video
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    vis_render = cv2.resize(vis, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    combined_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    combined_frame[:vis_h, :vis_w] = vis_render
    combined_frame[:COURT_H, vis_w:vis_w+COURT_W] = mat

   
    
    if out is not None:
        out.write(combined_frame)
    cv2.imshow("Video (Integrated)", cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Half Court (2D)", mat)
    
    key = cv2.waitKey(0 if paused else FPS_DELAY)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):
        paused = not paused

print("Total number of Interactions: ", INTERACTION_COUNT)
if out is not None:
    out.release()
cv2.destroyAllWindows()