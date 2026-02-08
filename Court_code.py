import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================
# CONFIG
# ======================================================
VIDEO_PATH = "Videos/raid1.mp4"
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

img_pts = np.array([
    intersect(L["end_back"], L["end_left"]),
    intersect(L["end_back"], L["end_right"]),
    intersect(L["middle"], L["end_left"]),
    intersect(L["middle"], L["end_right"]),
], dtype=np.float32)

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
# VIDEO LOOP
# ======================================================
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vis = frame.copy()
    mat = mat_base.copy()

    # Draw court lines on VIDEO
    for (p1, p2) in lines.values():
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # Player detection
    results = model(frame, device="cpu", verbose=False)[0]

    for box in results.boxes:
        if int(box.cls[0]) != 0 or float(box.conf[0]) < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Foot point
        fx = (x1 + x2) // 2
        fy = y2

        # Project to court
        mapped = cv2.perspectiveTransform(
            np.array([[[fx, fy]]], dtype=np.float32), H
        )[0][0]

        cx, cy = mapped

        if 0 <= cx <= 10 and 0 <= cy <= 6.5:
            draw_3d_bbox(vis, x1, y1, x2, y2)
            mx, my = court_to_pixel(cx, cy)
            cv2.circle(mat, (mx, my), 5, (255, 0, 0), -1)

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
