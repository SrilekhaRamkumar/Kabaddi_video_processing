import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


_NMS_FALLBACK_WARNED = False


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


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def _foot_distance(foot_a, foot_b):
    return float(np.sqrt((foot_a[0] - foot_b[0]) ** 2 + (foot_a[1] - foot_b[1]) ** 2))


def _predict_bbox_from_track(track):
    x1, y1, x2, y2 = track["last_bbox"]
    state = track["kf"].statePost.flatten()
    dx = int(state[2])
    dy = int(state[3])
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _ambiguous_track_ids(gallery, track_ids):
    ambiguous = set()
    predicted = {pid: _predict_bbox_from_track(gallery[pid]) for pid in track_ids}
    feet = {pid: gallery[pid].get("last_foot") for pid in track_ids}
    for i, pid_a in enumerate(track_ids):
        for pid_b in track_ids[i + 1:]:
            foot_a = feet.get(pid_a)
            foot_b = feet.get(pid_b)
            proximity = 999.0 if foot_a is None or foot_b is None else _foot_distance(foot_a, foot_b)
            overlap = bbox_iou(predicted[pid_a], predicted[pid_b])
            if proximity < 90 or overlap > 0.18:
                ambiguous.add(pid_a)
                ambiguous.add(pid_b)
    return ambiguous


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
    global _NMS_FALLBACK_WARNED
    try:
        results = model(frame, device=device, verbose=False)[0]
    except NotImplementedError as exc:
        error_text = str(exc)
        if device == "cuda" and "torchvision::nms" in error_text:
            if not _NMS_FALLBACK_WARNED:
                print("CUDA NMS is unavailable in this environment. Falling back to CPU detection.")
                _NMS_FALLBACK_WARNED = True
            results = model(frame, device="cpu", verbose=False)[0]
        else:
            raise
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
    ambiguous_tracks = _ambiguous_track_ids(gallery, track_ids)

    if predictions and detections:
        cost_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            pid = track_ids[i]
            px, py = pred[0][0], pred[1][0]
            track = gallery[pid]
            last_bbox = track["last_bbox"]
            prev_h = max(1, last_bbox[3] - last_bbox[1])
            prev_w = max(1, last_bbox[2] - last_bbox[0])
            is_ambiguous = pid in ambiguous_tracks
            appearance_anchor = track.get("appearance_anchor", track["feat"])
            protected_until = track.get("protected_until", -1)
            velocity = track["kf"].statePost.flatten()[2:4]
            for j, det in enumerate(detections):
                fx, fy = det["foot"]
                spatial_cost = np.sqrt((px - fx) ** 2 + (py - fy) ** 2) / 220
                appearance_cost = 1 - cosine(det["emb"], track["feat"])
                anchor_cost = 1 - cosine(det["emb"], appearance_anchor)
                size_cost = abs(det["height"] - prev_h) / max(det["height"], prev_h)
                aspect_cost = abs((det["width"] / det["height"]) - (prev_w / prev_h))
                det_dx = fx - track.get("last_foot", det["foot"])[0]
                det_dy = fy - track.get("last_foot", det["foot"])[1]
                motion_consistency = np.sqrt((velocity[0] - det_dx) ** 2 + (velocity[1] - det_dy) ** 2) / 120.0
                overlap_cost = 1.0 - min(1.0, bbox_iou(last_bbox, det["bbox"]) * 2.0)

                if is_ambiguous:
                    total_cost = (
                        0.32 * spatial_cost
                        + 0.28 * appearance_cost
                        + 0.20 * anchor_cost
                        + 0.10 * motion_consistency
                        + 0.05 * size_cost
                        + 0.05 * min(1.0, aspect_cost)
                    )
                else:
                    total_cost = (
                        0.48 * spatial_cost
                        + 0.22 * appearance_cost
                        + 0.10 * anchor_cost
                        + 0.10 * overlap_cost
                        + 0.07 * size_cost
                        + 0.03 * min(1.0, aspect_cost)
                    )

                if pid == raider_id and raid_assignment_done:
                    total_cost += 0.22 * anchor_cost
                    if is_ambiguous:
                        total_cost += 0.10 * motion_consistency
                    if frame_idx <= protected_until:
                        total_cost += 0.12 * max(0.0, appearance_cost - 0.18)

                cost_matrix[i, j] = total_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            pid = track_ids[r]
            track = gallery[pid]
            threshold = 0.95 if pid in ambiguous_tracks else 1.15
            if pid == raider_id and raid_assignment_done:
                threshold = min(threshold, 0.82 if pid in ambiguous_tracks else 0.98)
            if cost_matrix[r, c] >= threshold:
                continue

            det = detections[c]
            appearance_cost = 1 - cosine(det["emb"], track["feat"])
            anchor_cost = 1 - cosine(det["emb"], track.get("appearance_anchor", track["feat"]))
            if pid == raider_id and raid_assignment_done and pid in ambiguous_tracks and max(appearance_cost, anchor_cost) > 0.42:
                # Prefer temporary track loss over swapping the raider identity in contact/occlusion.
                continue

            prev_bbox = track["last_bbox"]
            smoothed_bbox = (
                int(0.7 * prev_bbox[0] + 0.3 * det["bbox"][0]),
                int(0.7 * prev_bbox[1] + 0.3 * det["bbox"][1]),
                int(0.7 * prev_bbox[2] + 0.3 * det["bbox"][2]),
                int(0.7 * prev_bbox[3] + 0.3 * det["bbox"][3]),
            )
            gallery[pid]["kf"].correct(np.array([[np.float32(det["foot"][0])], [np.float32(det["foot"][1])]]))
            feature_blend = 0.10 if pid in ambiguous_tracks else 0.20
            anchor_blend = 0.04 if pid in ambiguous_tracks else 0.08
            gallery[pid]["feat"] = (1.0 - feature_blend) * gallery[pid]["feat"] + feature_blend * det["emb"]
            gallery[pid]["appearance_anchor"] = (
                (1.0 - anchor_blend) * gallery[pid].get("appearance_anchor", gallery[pid]["feat"])
                + anchor_blend * det["emb"]
            )
            gallery[pid]["age"] = 0
            gallery[pid]["miss_streak"] = 0
            gallery[pid]["hits"] += 1
            gallery[pid]["last_bbox"] = smoothed_bbox
            gallery[pid]["last_foot"] = det["foot"]
            gallery[pid]["detection_confidence"] = det["conf"]
            gallery[pid]["occlusion_lock"] = pid in ambiguous_tracks
            if pid in ambiguous_tracks:
                gallery[pid]["occlusion_count"] = gallery[pid].get("occlusion_count", 0) + 1
            else:
                gallery[pid]["occlusion_count"] = 0
            if pid == raider_id and raid_assignment_done and pid in ambiguous_tracks:
                gallery[pid]["protected_until"] = frame_idx + 8
            gallery[pid]["visibility_confidence"] = _clamp(0.65 * gallery[pid]["visibility_confidence"] + 0.35 * det["conf"])
            gallery[pid]["track_confidence"] = _clamp(
                0.40 * gallery[pid]["visibility_confidence"]
                + 0.35 * _clamp(1.0 - cost_matrix[r, c] / max(threshold, 1e-6))
                + 0.15 * _clamp(gallery[pid]["hits"] / 8.0)
                + 0.10 * _clamp(1.0 - anchor_cost)
            )

            fx, fy = det["foot"]
            x1, y1, x2, y2 = smoothed_bbox
            draw_3d_bbox(vis, x1, y1, x2, y2)

            ew, eh = int((x2 - x1) * 0.7), int((x2 - x1) * 0.22)
            cv2.ellipse(vis, ((fx, fy), (ew, eh), 0), (255, 0, 0), 2)
            cv2.putText(vis, f"ID {pid}", (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if raid_assignment_done and pid == raider_id:
                cv2.putText(vis, "RAIDER", (x1 + 120, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            if pid in ambiguous_tracks:
                cv2.putText(vis, "LOCK", (x1, max(55, y1 - 38)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
        duplicate_candidate = False
        for data in gallery.values():
            last_foot = data.get("last_foot")
            if last_foot is None:
                continue
            if data.get("miss_streak", 0) <= 3 and _foot_distance(last_foot, det["foot"]) < 70:
                duplicate_candidate = True
                break
            if data.get("protected_until", -1) >= 0 and bbox_iou(data.get("last_bbox", det["bbox"]), det["bbox"]) > 0.25:
                duplicate_candidate = True
                break
        if duplicate_candidate:
            continue
        gallery[next_id] = {
            "feat": det["emb"],
            "appearance_anchor": det["emb"].copy(),
            "kf": create_kalman(*det["foot"]),
            "age": 0,
            "display_pos": None,
            "flow_pts": None,
            "last_bbox": det["bbox"],
            "last_foot": det["foot"],
            "hits": 1,
            "miss_streak": 0,
            "occlusion_count": 0,
            "occlusion_lock": False,
            "protected_until": -1,
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
            data["occlusion_lock"] = False
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
