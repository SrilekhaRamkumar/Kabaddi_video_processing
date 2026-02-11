import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ======================================================
# CONFIG
# ======================================================
VIDEO_PATH = "D:/Codes/kabaddi/Phase-2/Videos/raid1.mp4"
DISPLAY_SCALE = 0.5
FPS_DELAY = 1
CONF_THRESH = 0.4
SMOOTH_ALPHA = 0.25   # lower = smoother
MAX_PLAYERS = 8
LINE_MARGIN = 0.6   # meters

# ======================================================
# IMAGE-SPACE COURT LINES
# ======================================================
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
# LINE MATH
# ======================================================
def line_eq(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.array([y2 - y1, x1 - x2, x2*y1 - x1*y2], dtype=np.float64)

def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1*b2 - a2*b1
    if abs(d) < 1e-6:
        return None
    x = (b1*c2 - b2*c1) / d
    y = (a2*c1 - a1*c2) / d
    return [x, y]

# ======================================================
# HOMOGRAPHY (IMAGE → COURT)
# ======================================================
L = {k: line_eq(*v) for k, v in lines.items()}

img_pts = []
for a, b in [
    ("end_back", "end_left"),
    ("end_back", "end_right"),
    ("middle", "end_left"),
    ("middle", "end_right")
]:
    pt = intersect(L[a], L[b])
    if pt is None:
        raise RuntimeError(f"Intersection failed for {a} & {b}")
    img_pts.append(pt)

img_pts = np.array(img_pts, dtype=np.float32)

court_pts = np.array([
    [0, 6.5],
    [10, 6.5],
    [0, 0],
    [10, 0],
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, court_pts)

# ======================================================
# HALF COURT MAT
# ======================================================
COURT_W, COURT_H = 400, 260
mat_base = np.ones((COURT_H, COURT_W, 3), dtype=np.uint8) * 235

def court_to_pixel(x, y):
    px = int(x / 10 * COURT_W)
    py = int((6.5 - y) / 6.5 * COURT_H)
    return px, py

# Draw mat lines ONCE
cv2.rectangle(mat_base, court_to_pixel(0, 0),
              court_to_pixel(10, 6.5), (0, 0, 0), 2)

for y, name in [(3.75, "baulk"), (4.75, "bonus")]:
    cv2.line(mat_base, court_to_pixel(0, y),
             court_to_pixel(10, y), (0, 0, 0), 1)
    cv2.putText(mat_base, name, (8, court_to_pixel(0, y)[1]-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

for x in [0.75, 9.25]:
    cv2.line(mat_base, court_to_pixel(x, 0),
             court_to_pixel(x, 6.5), (0, 0, 0), 1)

# ======================================================
# YOLO PERSON DETECTOR (CPU FOR STABILITY)
# ======================================================
model = YOLO("yolov8n.pt")
model.to("cpu")

# ======================================================
# MOUSE
# ======================================================
mouse_pt = None
def mouse_cb(event, x, y, flags, param):
    global mouse_pt
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pt = (int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE))

cv2.namedWindow("Video (Integrated)")
cv2.namedWindow("Half Court (2D)")
cv2.setMouseCallback("Video (Integrated)", mouse_cb)

# ======================================================
# 3D BBOX DRAW
# ======================================================
def draw_3d_bbox(img, x1, y1, x2, y2, depth=12):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1+depth, y1-depth),
                  (x2+depth, y2-depth), (0, 255, 0), 2)
    for p in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        cv2.line(img, p, (p[0]+depth, p[1]-depth), (0, 255, 0), 2)



# ======================================================
# TRACKER STATE
# ======================================================

NEXT_ID = 0
GALLERY = {}   # pid -> state

MAX_AGE = 120
MAX_PLAYERS = 8


# ------------------------------------------------------
def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)

    return kf


# ------------------------------------------------------
def extract_embedding(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)

    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


# ------------------------------------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


# ======================================================
# VIDEO LOOP
# ======================================================

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vis = frame.copy()
    mat = mat_base.copy()

    # Draw court lines
    for (p1, p2) in lines.values():
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # --------------------------------------------------
    # STEP 1: Collect detections
    # --------------------------------------------------

    results = model(frame, device="cpu", verbose=False)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) != 0 or float(box.conf[0]) < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        fx = (x1 + x2) // 2
        fy = y2

        emb = extract_embedding(frame, (x1, y1, x2, y2))
        if emb is None:
            continue

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "foot": (fx, fy),
            "emb": emb
        })

    # --------------------------------------------------
    # STEP 2: Predict all tracks
    # --------------------------------------------------

    track_ids = list(GALLERY.keys())
    predictions = []

    for pid in track_ids:
        pred = GALLERY[pid]["kf"].predict()
        px, py = pred[0][0], pred[1][0]
        predictions.append((px, py))

    # --------------------------------------------------
    # STEP 3: Hungarian Matching
    # --------------------------------------------------

    matched_tracks = set()
    matched_dets = set()

    if len(predictions) > 0 and len(detections) > 0:

        cost_matrix = np.zeros((len(predictions), len(detections)))

        for i, (px, py) in enumerate(predictions):
            for j, det in enumerate(detections):

                fx, fy = det["foot"]

                # spatial cost
                dist = np.sqrt((px - fx)**2 + (py - fy)**2)
                spatial_cost = dist / 200

                # appearance cost
                sim = cosine(det["emb"], GALLERY[track_ids[i]]["feat"])
                appearance_cost = 1 - sim

                cost_matrix[i, j] = 0.7 * spatial_cost + 0.3 * appearance_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):

            if cost_matrix[r, c] > 1.2:
                continue

            pid = track_ids[r]
            det = detections[c]

            fx, fy = det["foot"]

            # Kalman correction
            measurement = np.array([[np.float32(fx)], [np.float32(fy)]])
            GALLERY[pid]["kf"].correct(measurement)

            # update appearance
            GALLERY[pid]["feat"] = (
                0.8 * GALLERY[pid]["feat"] + 0.2 * det["emb"]
            )

            GALLERY[pid]["age"] = 0
            # --------------------------
            # DRAW ON VIDEO
            # --------------------------

            x1, y1, x2, y2 = det["bbox"]
            fx, fy = det["foot"]

            # 3D Bounding Box
            draw_3d_bbox(vis, x1, y1, x2, y2)

            # Ellipse under feet (bigger)
            ellipse_width  = int((x2 - x1) * 0.7)
            ellipse_height = int((x2 - x1) * 0.22)

            cv2.ellipse(
                vis,
                ((fx, fy), (ellipse_width, ellipse_height), 0),
                (255, 0, 0),   # blue
                2
            )

            # Big ID label
            cv2.putText(
                vis,
                f"ID {pid}",
                (x1, max(30, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,          # bigger text
                (0, 0, 255),  # red
                3,
                cv2.LINE_AA
            )

            # Foot contact point
            cv2.circle(
                vis,
                (fx, fy),
                5,
                (255, 0, 0),
                -1
            )
            



            matched_tracks.add(pid)
            matched_dets.add(c)

    # --------------------------------------------------
    # STEP 4: Create new tracks
    # --------------------------------------------------

    for j, det in enumerate(detections):
        if j in matched_dets:
            continue

        if len(GALLERY) >= MAX_PLAYERS:
            continue

        fx, fy = det["foot"]

        pid = NEXT_ID
        NEXT_ID += 1

        GALLERY[pid] = {
            "feat": det["emb"],
            "kf": create_kalman(fx, fy),
            "age": 0,
            "display_pos": None
        }

        matched_tracks.add(pid)

    # --------------------------------------------------
    # STEP 5: Age unmatched tracks
    # --------------------------------------------------

    dead = []

    for pid in GALLERY:
        if pid not in matched_tracks:
            GALLERY[pid]["age"] += 1
        if GALLERY[pid]["age"] > MAX_AGE:
            dead.append(pid)

    for pid in dead:
        del GALLERY[pid]

    # --------------------------------------------------
    # STEP 6: Draw players
    # --------------------------------------------------

    for pid, data in GALLERY.items():

        pred = data["kf"].predict()
        px, py = pred[0][0], pred[1][0]

        vx = pred[2][0]
        vy = pred[3][0]

        angle = np.degrees(np.arctan2(vy, vx))

        data["direction"] = angle



        mapped = cv2.perspectiveTransform(
            np.array([[[px, py]]], dtype=np.float32), H
        )[0][0]

        cx, cy = mapped

        if not (
            LINE_MARGIN < cx < 10 - LINE_MARGIN and
            LINE_MARGIN < cy < 6.5 - LINE_MARGIN
        ):
            continue

        # Smooth
        if data["display_pos"] is None:
            data["display_pos"] = (cx, cy)
        else:
            ox, oy = data["display_pos"]
            nx = ox + SMOOTH_ALPHA * (cx - ox)
            ny = oy + SMOOTH_ALPHA * (cy - oy)
            data["display_pos"] = (nx, ny)

        sx, sy = data["display_pos"]
        mx, my = court_to_pixel(sx, sy)
        bbox_h = y2 - y1

        # Normalize height (tune these values based on your video)
        min_h = 80
        max_h = 300

        scale = (bbox_h - min_h) / (max_h - min_h)
        scale = max(0, min(1, scale))  # clamp 0-1

        dot_radius = int(5 + scale * 8)   # between 5 and 13


        cv2.circle(mat, (mx, my), dot_radius, (255, 0, 0), -1)


        BOX = 20
        cv2.rectangle(
            mat,
            (mx - BOX, my - BOX),
            (mx + BOX, my + BOX),
            (0, 0, 0),
            1
        )

        cv2.putText(
            mat,
            f"{pid}",
            (mx + 6, my - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        arrow_length = 40

        end_x = int(px + arrow_length * np.cos(np.radians(angle)))
        end_y = int(py + arrow_length * np.sin(np.radians(angle)))

        cv2.arrowedLine(
            vis,
            (int(px), int(py)),
            (end_x, end_y),
            (0, 255, 255),
            2
        )

    vis_small = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Video (Integrated)", vis_small)
    cv2.imshow("Half Court (2D)", mat)

    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
