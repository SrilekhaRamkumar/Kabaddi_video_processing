import cv2

VIDEO_PATH = "Videos/raid1.mp4"
DISPLAY_SCALE = 0.6  # change if needed

# Read first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame")

h, w = frame.shape[:2]
frame_small = cv2.resize(
    frame,
    (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)),
    interpolation=cv2.INTER_AREA
)

# Mouse callback
def mouse_event(event, x, y, flags, param):
    # Convert back to original resolution
    ox = int(x / DISPLAY_SCALE)
    oy = int(y / DISPLAY_SCALE)

    if event == cv2.EVENT_MOUSEMOVE:
        img = frame_small.copy()
        cv2.putText(
            img,
            f"x={ox}, y={oy}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("Move Mouse - Press Left Click to Select", img)

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({ox}, {oy})")

# Window
cv2.namedWindow("Move Mouse - Press Left Click to Select")
cv2.setMouseCallback("Move Mouse - Press Left Click to Select", mouse_event)
cv2.imshow("Move Mouse - Press Left Click to Select", frame_small)

cv2.waitKey(0)
cv2.destroyAllWindows()
