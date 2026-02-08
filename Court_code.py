import cv2
import numpy as np

# ======================================================
# CONFIG
# ======================================================
VIDEO_PATH = "Videos/raid1.mp4"
DISPLAY_SCALE = 0.5
FPS_DELAY = 30

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

colors = {
    k: (100, 100, 255) for k in lines
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
# HOMOGRAPHY SETUP
# ======================================================
L = {k: line_eq(*v) for k, v in lines.items()}

img_pts = np.array([
    intersect(L["end_back"],  L["end_left"]),    # (0,0)
    intersect(L["end_back"],  L["end_right"]),   # (10,0)
    intersect(L["middle"],    L["end_left"]),    # (0,6.5)
    intersect(L["middle"],    L["end_right"]),   # (10,6.5)
], dtype=np.float32)

court_pts = np.array([
    [0, 6.5],    # end_back ∩ end_left
    [10, 6.5],   # end_back ∩ end_right
    [0, 0],      # middle ∩ end_left
    [10, 0]      # middle ∩ end_right
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, court_pts)

# ======================================================
# HALF COURT MAT (GROUND TRUTH)
# ======================================================
COURT_W, COURT_H = 400, 260
mat_base = np.ones((COURT_H, COURT_W, 3), dtype=np.uint8) * 235

def court_to_pixel(x, y):
    px = int(x / 10 * COURT_W)
    py = int((6.5 - y) / 6.5 * COURT_H)
    return px, py

# Outer boundary
cv2.rectangle(
    mat_base,
    court_to_pixel(0, 0),
    court_to_pixel(10, 6.5),
    (0, 0, 0),
    2
)

# ===== Single correct lines =====
court_lines = [
    (3.75, "baulk"),
    (4.75, "bonus"),
]

for y, name in court_lines:
    p1 = court_to_pixel(0, y)
    p2 = court_to_pixel(10, y)
    cv2.line(mat_base, p1, p2, (0, 0, 0), 1)
    cv2.putText(
        mat_base, name,
        (10, max(p1[1]-5, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1
    )

# Lobby lines
for x in [0.75, 9.25]:
    p1 = court_to_pixel(x, 0)
    p2 = court_to_pixel(x, 6.5)
    cv2.line(mat_base, p1, p2, (0, 0, 0), 1)
    lx = min(max(p1[0]+3, 5), COURT_W-60)
    cv2.putText(
        mat_base, "lobby",
        (lx, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1
    )

# ======================================================
# MOUSE
# ======================================================
mouse_pt = None
def mouse_cb(event, x, y, flags, param):
    global mouse_pt
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pt = (int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE))

cv2.namedWindow("Video (Calibrated)")
cv2.namedWindow("Half Court (Corrected)")
cv2.setMouseCallback("Video (Calibrated)", mouse_cb)

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

    # --- draw lines + labels on video ---
    for name, (p1, p2) in lines.items():
        cv2.line(vis, p1, p2, colors[name], 2)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        cv2.putText(
            vis, name,
            (mx+4, my-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1
        )

    # --- cursor mapping ---
    if mouse_pt is not None:
        pt = np.array([[mouse_pt]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        cx, cy = mapped

        if 0 <= cx <= 10 and 0 <= cy <= 6.5:
            cv2.circle(vis, mouse_pt, 6, (0, 0, 255), -1)
            mx, my = court_to_pixel(cx, cy)
            cv2.circle(mat, (mx, my), 6, (0, 0, 255), -1)

    vis_small = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Video (Calibrated)", vis_small)
    cv2.imshow("Half Court (Corrected)", mat)

    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
