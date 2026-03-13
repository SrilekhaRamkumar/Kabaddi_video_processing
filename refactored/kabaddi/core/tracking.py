"""
Tracking Pipeline Module
Handles YOLO detection, Kalman filtering, optical flow, and track management.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


def extract_embedding(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


def draw_3d_bbox(img, x1, y1, x2, y2, depth=12):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1 + depth, y1 - depth), (x2 + depth, y2 - depth), (0, 255, 0), 2)
    for point in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        cv2.line(img, point, (point[0] + depth, point[1] - depth), (0, 255, 0), 2)


def apply_optical_flow(prev_gray, gray, gallery):
    for _, data in gallery.items():
        if data["flow_pts"] is None:
            continue
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            data["flow_pts"],
            None,
            winSize=(15, 15),
            maxLevel=2,
        )
        if new_pts is None:
            continue
        good_new = new_pts[status == 1]
        good_old = data["flow_pts"][status == 1]
        if len(good_new) > 5:
            dx, dy = np.mean(good_new - good_old, axis=0)
            data["kf"].statePost[0] += dx
            data["kf"].statePost[1] += dy
            data["flow_pts"] = good_new.reshape(-1, 1, 2)


def run_yolo_detection(model, frame, device, conf_thresh):
    results = model(frame, device=device, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if int(box.cls[0]) != 0 or conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        emb = extract_embedding(frame, (x1, y1, x2, y2))
        if emb is None:
            continue
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "foot": ((x1 + x2) // 2, y2),
            "emb": emb,
            "conf": conf,
            "height": max(1, y2 - y1),
            "width": max(1, x2 - x1),
        })
    return detections


def update_tracks(gallery, detections, gray, vis, frame_idx, raid_assignment_done, raider_id):
    track_ids = list(gallery.keys())
    predictions = [gallery[pid]["kf"].predict() for pid in track_ids]
    matched_tracks, matched_dets = set(), set()

    if predictions and detections:
        cost_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            px, py = pred[0][0], pred[1][0]
            last_bbox = gallery[track_ids[i]]["last_bbox"]
            prev_h = max(1, last_bbox[3] - last_bbox[1])
            prev_w = max(1, last_bbox[2] - last_bbox[0])
            for j, det in enumerate(detections):
                fx, fy = det["foot"]
                spatial_cost = np.sqrt((px - fx) ** 2 + (py - fy) ** 2) / 220
                appearance_cost = 1 - cosine(det["emb"], gallery[track_ids[i]]["feat"])
                size_cost = abs(det["height"] - prev_h) / max(det["height"], prev_h)
                aspect_cost = abs((det["width"] / det["height"]) - (prev_w / prev_h))
                cost_matrix[i, j] = (
                    0.55 * spatial_cost
                    + 0.25 * appearance_cost
                    + 0.15 * size_cost
                    + 0.05 * min(1.0, aspect_cost)
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 1.15:
                continue

            pid, det = track_ids[r], detections[c]
            prev_bbox = gallery[pid]["last_bbox"]
            smoothed_bbox = (
                int(0.7 * prev_bbox[0] + 0.3 * det["bbox"][0]),
                int(0.7 * prev_bbox[1] + 0.3 * det["bbox"][1]),
                int(0.7 * prev_bbox[2] + 0.3 * det["bbox"][2]),
                int(0.7 * prev_bbox[3] + 0.3 * det["bbox"][3]),
            )
            gallery[pid]["kf"].correct(np.array([[np.float32(det["foot"][0])], [np.float32(det["foot"][1])]]))
            gallery[pid]["feat"] = 0.8 * gallery[pid]["feat"] + 0.2 * det["emb"]
            gallery[pid]["age"] = 0
            gallery[pid]["miss_streak"] = 0
            gallery[pid]["hits"] += 1
            gallery[pid]["last_bbox"] = smoothed_bbox
            gallery[pid]["last_foot"] = det["foot"]
            gallery[pid]["detection_confidence"] = det["conf"]
            gallery[pid]["visibility_confidence"] = _clamp(0.6 * gallery[pid]["visibility_confidence"] + 0.4 * det["conf"])
            gallery[pid]["track_confidence"] = _clamp(
                0.45 * gallery[pid]["visibility_confidence"]
                + 0.35 * _clamp(1.0 - cost_matrix[r, c] / 1.15)
                + 0.20 * _clamp(gallery[pid]["hits"] / 8.0)
            )

            fx, fy = det["foot"]
            x1, y1, x2, y2 = smoothed_bbox
            draw_3d_bbox(vis, x1, y1, x2, y2)

            ew, eh = int((x2 - x1) * 0.7), int((x2 - x1) * 0.22)
            cv2.ellipse(vis, ((fx, fy), (ew, eh), 0), (255, 0, 0), 2)
            cv2.putText(vis, f"ID {pid}", (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if raid_assignment_done and pid == raider_id:
                cv2.putText(vis, "RAIDER", (x1 + 120, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            cv2.circle(vis, (fx, fy), 5, (255, 0, 0), -1)

            if frame_idx % 4 == 0:
                pts = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=8, qualityLevel=0.3, minDistance=5)
                if pts is not None:
                    pts[:, :, 0] += x1
                    pts[:, :, 1] += y1
                    gallery[pid]["flow_pts"] = pts

            matched_tracks.add(pid)
            matched_dets.add(c)

    return matched_tracks, matched_dets


def add_new_tracks(gallery, detections, matched_dets, next_id, max_players):
    for j, det in enumerate(detections):
        if j in matched_dets or len(gallery) >= max_players:
            continue
        gallery[next_id] = {
            "feat": det["emb"],
            "kf": create_kalman(*det["foot"]),
            "age": 0,
            "display_pos": None,
            "flow_pts": None,
            "last_bbox": det["bbox"],
            "last_foot": det["foot"],
            "hits": 1,
            "miss_streak": 0,
            "detection_confidence": det["conf"],
            "visibility_confidence": det["conf"],
            "track_confidence": det["conf"],
        }
        next_id += 1
    return next_id


def render_gallery(gallery, matched_tracks, vis, mat, homography, court_to_pixel, line_margin, smooth_alpha, raider_id, max_age):
    dead = [pid for pid, data in gallery.items() if pid not in matched_tracks and data["age"] > max_age]
    for pid in dead:
        del gallery[pid]

    for pid, data in gallery.items():
        if pid not in matched_tracks:
            data["age"] += 1
            data["miss_streak"] = data.get("miss_streak", 0) + 1
            data["visibility_confidence"] = _clamp(data.get("visibility_confidence", 0.0) * 0.92)
            data["track_confidence"] = _clamp(data.get("track_confidence", 0.0) * 0.95)
        else:
            data["miss_streak"] = 0

        pred = data["kf"].statePost
        px, py, vx, vy = pred[0][0], pred[1][0], pred[2][0], pred[3][0]
        angle = np.degrees(np.arctan2(vy, vx))
        ex = int(px + 40 * np.cos(np.radians(angle)))
        ey = int(py + 40 * np.sin(np.radians(angle)))
        cv2.arrowedLine(vis, (int(px), int(py)), (ex, ey), (0, 255, 255), 2)

        mapped = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), homography)[0][0]
        cx, cy = mapped

        if line_margin < cx < 10 - line_margin and line_margin < cy < 6.5 - line_margin:
            adaptive_alpha = smooth_alpha * (0.75 + 0.5 * data.get("track_confidence", 0.5))
            if data["display_pos"] is None:
                data["display_pos"] = (cx, cy)
            else:
                ox, oy = data["display_pos"]
                data["display_pos"] = (ox + adaptive_alpha * (cx - ox), oy + adaptive_alpha * (cy - oy))

            mx, my = court_to_pixel(*data["display_pos"])
            bh = data["last_bbox"][3] - data["last_bbox"][1]
            scale = np.clip((bh - 80) / 220, 0, 1)
            dot_rad = int(5 + scale * 8)

            cv2.circle(mat, (mx, my), dot_rad, (255, 0, 0), -1)
            if pid == raider_id:
                cv2.circle(mat, (mx, my), dot_rad + 6, (0, 0, 255), 3)

            cv2.rectangle(mat, (mx - 20, my - 20), (mx + 20, my + 20), (0, 0, 0), 1)
            cv2.putText(mat, f"{pid}", (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            if data.get("miss_streak", 0) > 2:
                data["display_pos"] = None

