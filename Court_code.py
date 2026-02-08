import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================
# CONFIG
# ======================================================
VIDEO_PATH = "D:/Codes/kabaddi/Phase-2/Videos/raid1.mp4"
DISPLAY_SCALE = 0.5
FPS_DELAY = 1
CONF_THRESH = 0.4

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
# SGM: CONFIDENCE-ANCHORED GALLERY (FINAL)
# ======================================================

NEXT_ID = 0
GALLERY = {}   # pid -> state

# -------- Parameters --------
MATCH_THRESH = 0.55
MAX_AGE = 120         # frames to keep missing players
DECAY = 0.995         # confidence decay

# ------------------------------------------------------
def extract_embedding(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)

    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0,1,2], None,
        [8,8,8], [0,180,0,256,0,256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()

# ------------------------------------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)

# ------------------------------------------------------
def sgm_assign(embedding,used_ids):
    global NEXT_ID

    best_id = None
    best_sim = -1

    for pid, data in GALLERY.items():
        if pid in used_ids:
            continue
        sim = cosine(embedding, data["feat"])
        if sim > best_sim:
            best_sim = sim
            best_id = pid


    if best_sim > MATCH_THRESH:
        GALLERY[best_id]["feat"] = (
            0.8 * GALLERY[best_id]["feat"] + 0.2 * embedding
        )
        GALLERY[best_id]["score"] = min(1.0, GALLERY[best_id]["score"] + 0.1)
        GALLERY[best_id]["age"] = 0
        return best_id

    pid = NEXT_ID
    NEXT_ID += 1

    GALLERY[pid] = {
        "feat": embedding,
        "score": 1.0,
        "age": 0,
        "pos": None,     # last confident court coord (cx, cy)
        "seen": True
    }
    return pid

# ------------------------------------------------------
def sgm_step():
    dead = []
    for pid, data in GALLERY.items():
        data["age"] += 1
        data["score"] *= DECAY

        if data["age"] > MAX_AGE and data["pos"] is None:
            dead.append(pid)

    for pid in dead:
        del GALLERY[pid]

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

    used_ids = set()


    # mark all players as not seen initially
    for pid in GALLERY:
        GALLERY[pid]["seen"] = False

    # Draw court lines on VIDEO
    for (p1, p2) in lines.values():
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # Player detection
    results = model(frame, device="cpu", verbose=False)[0]

    for box in results.boxes:

        if int(box.cls[0]) != 0 or float(box.conf[0]) < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        emb = extract_embedding(frame, (x1,y1,x2,y2))
        if emb is None:
            continue

        pid = sgm_assign(emb, used_ids)
        used_ids.add(pid)


        # Foot point
        fx = (x1 + x2) // 2
        fy = y2

        mapped = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype=np.float32), H)[0][0]

        cx, cy = mapped
        if 0 <= cx <= 10 and 0 <= cy <= 6.5:
            draw_3d_bbox(vis, x1, y1, x2, y2)
            ellipse_width  = int((x2 - x1) * 0.7)
            ellipse_height = int((x2 - x1) * 0.22)


            cv2.ellipse(
                vis,
                ((fx, fy), (ellipse_width, ellipse_height), 0),
                (255, 0, 0),   # blue
                2
    )
            cv2.putText(
            vis,
            f"ID {pid}",
            (x1, max(30, y1 - 12)),   # keep text inside frame
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,                     # 🔹 BIGGER TEXT
            (0, 0, 255),             # 🔹 RED COLOR (BGR)
            3,                       # 🔹 THICK
            cv2.LINE_AA
            )
            # draw foot contact point (small circle on ground)
            cv2.circle(
                vis,
                (fx, fy),
                5,              # radius
                (255, 0, 0),    # BLUE (BGR)
                -1              # filled
            )

            


            # --- update SGM state ---
            prev = GALLERY[pid]["pos"]
            if prev is not None:
                vx = cx - prev[0]
                vy = cy - prev[1]
                GALLERY[pid]["vel"] = (vx, vy)

            GALLERY[pid]["pos"] = (cx, cy)
            GALLERY[pid]["seen"] = True

            mx, my = court_to_pixel(cx, cy)
            cv2.circle(mat, (mx, my), 6, (255, 0, 0), -1)

            BOX = 14
            cv2.rectangle(
                mat,
                (mx - BOX, my - BOX),
                (mx + BOX, my + BOX),
                (0, 0, 0),
                1
            )

            # draw ID near dot
            cv2.putText(
                mat,
                f"{pid}",
                (mx + 6, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0), 1)

            

    # ======================================================
    # MISSING PLAYER BELIEF (STATIC, RADIUS-BASED)
    # ======================================================
    RADIUS = 0.6  # meters

    for pid, data in GALLERY.items():
        if data["seen"]:
            continue
        if data["pos"] is None:
            continue

        cx, cy = data["pos"]

        # tiny jitter inside belief radius (optional)
        jx = np.random.uniform(-RADIUS/4, RADIUS/4)
        jy = np.random.uniform(-RADIUS/4, RADIUS/4)

        px = max(0, min(10, cx + jx))
        py = max(0, min(6.5, cy + jy))

        mx, my = court_to_pixel(px, py)

        # faded dot for missing player
        cv2.circle(mat, (mx, my), 5, (170, 170, 170), -1)

       

        BOX = 14
        cv2.rectangle(
            mat,
            (mx - BOX, my - BOX),
            (mx + BOX, my + BOX),
            (120, 120, 120),
            1
        )


        cv2.putText(
        mat,
        f"{pid}",
        (mx + 6, my - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0,0,0),
        1
        )





    sgm_step()

    # Cursor mapping
    if mouse_pt is not None:
        mapped = cv2.perspectiveTransform(
            np.array([[mouse_pt]], dtype=np.float32), H
        )[0][0]

        cx, cy = mapped
        if 0 <= cx <= 10 and 0 <= cy <= 6.5:
            cv2.circle(vis, mouse_pt, 5, (255, 0, 0), -1)
            mx, my = court_to_pixel(cx, cy)
            cv2.circle(mat, (mx, my), 5, (0, 0, 255), -1)

    vis_small = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Video (Integrated)", vis_small)
    cv2.imshow("Half Court (2D)", mat)

    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
