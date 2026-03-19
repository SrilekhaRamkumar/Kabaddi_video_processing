import cv2
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Open video (0 for webcam or give file path)
cap = cv2.VideoCapture("Videos/Cam1/raid1.mp4")

frame_count = 0
max_frames = 200  # process first few frames (change/remove if needed)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count >= max_frames:
        break

    results = model(frame)

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints = r.keypoints.xy.cpu().numpy()

        for box, kpts in zip(boxes, keypoints):
            x1, y1, x2, y2 = map(int, box)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Keypoints
            for (x, y) in kpts:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

            # Skeleton (bones)
            skeleton = [
                (5, 7), (7, 9),
                (6, 8), (8, 10),
                (5, 6),
                (5, 11), (6, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16),
                (11, 12)
            ]

            for i, j in skeleton:
                if i < len(kpts) and j < len(kpts):
                    xi, yi = kpts[i]
                    xj, yj = kpts[j]
                    cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Pose Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()