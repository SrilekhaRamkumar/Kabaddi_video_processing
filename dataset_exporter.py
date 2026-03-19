import csv
import json
import os
from typing import List

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


POSE_SKELETON_EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]
POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
_POSE_MODEL = None
_POSE_MODEL_ERROR = None


class ConfirmedWindowDatasetExporter:
    """Exports confirmed interaction windows as clips plus metadata for labeling."""

    def __init__(self, root_dir, fps=30.0):
        self.root_dir = root_dir
        self.fps = fps
        self.manifest_path = os.path.join(root_dir, "manifest.csv")
        self.exported_keys = set()
        os.makedirs(self.root_dir, exist_ok=True)
        self._ensure_manifest()
        self._load_existing_manifest_keys()

    def export_batch(self, classifier_inputs: List[dict]):
        exported = []
        for item in classifier_inputs:
            result = self.export_window(item)
            if result is not None:
                exported.append(result)
        return exported

    def export_window(self, classifier_input):
        event = classifier_input["event"]
        event_key = (event["type"], event["frame"], event["subject"], event["object"])
        clip_id = self._clip_id(event)
        clip_dir = os.path.join(self.root_dir, event["type"])
        os.makedirs(clip_dir, exist_ok=True)

        clip_path = os.path.join(clip_dir, f"{clip_id}.mp4")
        payload_path = os.path.join(clip_dir, f"{clip_id}.json")
        frames = classifier_input.get("frames", [])
        if not frames:
            return None

        if event_key not in self.exported_keys or not os.path.exists(clip_path):
            self._write_clip(clip_path, frames)
        self._write_payload(payload_path, classifier_input)
        if event_key not in self.exported_keys:
            self._append_manifest_row(event, clip_id, clip_path, payload_path, len(frames))
            self.exported_keys.add(event_key)

        return {
            "event_key": event_key,
            "clip_id": clip_id,
            "clip_path": clip_path,
            "payload_path": payload_path,
        }

    def _clip_id(self, event):
        return (
            f"{event['type']}_f{event['frame']:05d}"
            f"_s{event['subject']}_o{event['object']}"
        )

    def _write_clip(self, clip_path, frames):
        height, width = frames[0].shape[:2]

        # H.264 encoders often require even dimensions. If the incoming frames are odd-sized,
        # pad them so we can still write a browser-playable MP4.
        target_w = int(width) + (int(width) % 2)
        target_h = int(height) + (int(height) % 2)

        def _pad_to_target(bgr):
            h, w = bgr.shape[:2]
            if (w, h) == (target_w, target_h):
                return bgr
            out = bgr[:target_h, :target_w].copy()
            oh, ow = out.shape[:2]
            pad_r = max(0, target_w - ow)
            pad_b = max(0, target_h - oh)
            if pad_r or pad_b:
                out = cv2.copyMakeBorder(out, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return out

        # Prefer a browser-friendly codec when available (H.264 in MP4).
        # On Windows, force MSMF to avoid FFmpeg's OpenH264 DLL dependency.
        writer = None
        candidates = [
            ("msmf", getattr(cv2, "CAP_MSMF", 0), "H264"),
            ("msmf", getattr(cv2, "CAP_MSMF", 0), "avc1"),
            ("any", 0, "mp4v"),
        ]
        for _, api, fourcc in candidates:
            candidate = cv2.VideoWriter(
                clip_path,
                api,
                cv2.VideoWriter_fourcc(*fourcc),
                self.fps,
                (target_w, target_h),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()

        if writer is None:
            writer = cv2.VideoWriter(
                clip_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (target_w, target_h),
            )
            if not writer.isOpened():
                try:
                    writer.release()
                except Exception:
                    pass
                return
        for frame in frames:
            writer.write(_pad_to_target(frame))
        writer.release()

    def _write_payload(self, payload_path, classifier_input):
        event = classifier_input["event"]
        payload = classifier_input.get("payload", {})
        if isinstance(payload, dict):
            pose_window = self._build_pose_window(classifier_input)
            if pose_window:
                payload["pose_window"] = pose_window
                payload["pose_meta"] = {
                    "model": "yolov8n-pose",
                    "keypoint_names": POSE_KEYPOINT_NAMES,
                    "skeleton_edges": POSE_SKELETON_EDGES,
                }
        data = {
            "event": {
                "type": event["type"],
                "event_family": event.get("event_family", self._event_family(event)),
                "frame": event["frame"],
                "window_start": event["window_start"],
                "window_end": event["window_end"],
                "core_window_start": event.get("core_window_start"),
                "core_window_end": event.get("core_window_end"),
                "subject": event["subject"],
                "object": event["object"],
                "line_name": event.get("line_name"),
                "confidence": event.get("confidence", 0.0),
                "factor_confidence": event.get("factor_confidence", 0.0),
                "requires_visual_confirmation": event.get("requires_visual_confirmation", False),
            },
            "payload": payload,
            "label": "",
            "notes": "",
        }
        with open(payload_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _build_pose_window(self, classifier_input):
        frames = classifier_input.get("frames", []) or []
        payload = classifier_input.get("payload", {}) or {}
        mat_window = payload.get("mat_window", []) if isinstance(payload, dict) else []
        window_frames = payload.get("window_frames", []) if isinstance(payload, dict) else []
        if not frames:
            return []

        model = self._load_pose_model()
        if model is None:
            return []

        pose_window = []
        try:
            results = model(frames, verbose=False)
        except Exception as exc:
            print(f"[POSE] Could not estimate poses for confirmed window: {exc}")
            return []

        for idx, frame in enumerate(frames):
            result = results[idx] if idx < len(results) else None
            snapshot = mat_window[idx] if idx < len(mat_window) and isinstance(mat_window[idx], dict) else {}
            tracked_players = snapshot.get("players", []) if isinstance(snapshot, dict) else []
            detections = self._extract_pose_detections(result)
            matched_players = self._match_pose_detections(tracked_players, detections)
            source_frame = None
            if idx < len(window_frames):
                try:
                    source_frame = int(window_frames[idx])
                except (TypeError, ValueError):
                    source_frame = window_frames[idx]
            elif isinstance(snapshot, dict):
                source_frame = snapshot.get("frame")
            pose_window.append({
                "clip_index": idx,
                "frame": source_frame,
                "players": matched_players,
                "detections": detections,
            })
        return pose_window

    def _load_pose_model(self):
        global _POSE_MODEL, _POSE_MODEL_ERROR
        if _POSE_MODEL is not None:
            return _POSE_MODEL
        if _POSE_MODEL_ERROR is not None:
            return None
        if YOLO is None:
            _POSE_MODEL_ERROR = "ultralytics not installed"
            print("[POSE] ultralytics is not installed; skipping pose archive.")
            return None
        try:
            _POSE_MODEL = YOLO("yolov8n-pose.pt")
            return _POSE_MODEL
        except Exception as exc:
            _POSE_MODEL_ERROR = exc
            print(f"[POSE] Could not load YOLO pose model: {exc}")
            return None

    def _extract_pose_detections(self, result):
        if result is None or getattr(result, "boxes", None) is None or getattr(result, "keypoints", None) is None:
            return []
        try:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy() if getattr(result.boxes, "conf", None) is not None else None
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = (
                result.keypoints.conf.cpu().numpy()
                if getattr(result.keypoints, "conf", None) is not None
                else None
            )
        except Exception:
            return []

        detections = []
        for det_idx, box in enumerate(boxes):
            kp_xy = keypoints_xy[det_idx] if det_idx < len(keypoints_xy) else []
            kp_conf = keypoints_conf[det_idx] if keypoints_conf is not None and det_idx < len(keypoints_conf) else None
            detections.append({
                "bbox": [float(v) for v in box.tolist()],
                "score": float(scores[det_idx]) if scores is not None and det_idx < len(scores) else None,
                "keypoints": self._serialize_keypoints(kp_xy, kp_conf),
            })
        return detections

    def _serialize_keypoints(self, kp_xy, kp_conf):
        serialized = []
        for kp_idx, xy in enumerate(kp_xy):
            x = float(xy[0]) if len(xy) > 0 else None
            y = float(xy[1]) if len(xy) > 1 else None
            conf = float(kp_conf[kp_idx]) if kp_conf is not None and kp_idx < len(kp_conf) else None
            serialized.append({
                "name": POSE_KEYPOINT_NAMES[kp_idx] if kp_idx < len(POSE_KEYPOINT_NAMES) else f"kp_{kp_idx}",
                "x": x,
                "y": y,
                "confidence": conf,
            })
        return serialized

    def _match_pose_detections(self, tracked_players, detections):
        if not tracked_players or not detections:
            return []
        remaining = list(range(len(detections)))
        matched = []
        for player in tracked_players:
            player_box = player.get("bbox") if isinstance(player, dict) else None
            player_id = player.get("id") if isinstance(player, dict) else None
            if player_box is None or player_id is None:
                continue
            best_idx = None
            best_score = -1.0
            for det_idx in remaining:
                det = detections[det_idx]
                iou = self._bbox_iou(player_box, det.get("bbox"))
                if iou > best_score:
                    best_score = iou
                    best_idx = det_idx
            if best_idx is None:
                continue
            det = detections[best_idx]
            remaining.remove(best_idx)
            matched.append({
                "id": player_id,
                "bbox": player_box,
                "court_pos": player.get("court_pos"),
                "visible": player.get("visible"),
                "pose_score": det.get("score"),
                "keypoints": det.get("keypoints", []),
            })
        return matched

    def _bbox_iou(self, a, b):
        if not a or not b or len(a) < 4 or len(b) < 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 1e-6:
            return 0.0
        return inter / denom

    def _ensure_manifest(self):
        if os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "clip_id",
                "event_type",
                "event_family",
                "frame",
                "window_start",
                "window_end",
                "subject",
                "object",
                "line_name",
                "confidence",
                "factor_confidence",
                "num_frames",
                "clip_path",
                "payload_path",
                "label",
                "notes",
            ])

    def _load_existing_manifest_keys(self):
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (
                    row["event_type"],
                    int(row["frame"]),
                    self._parse_int(row["subject"]),
                    self._parse_int(row["object"]),
                )
                self.exported_keys.add(key)

    def _append_manifest_row(self, event, clip_id, clip_path, payload_path, num_frames):
        with open(self.manifest_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                clip_id,
                event["type"],
                event.get("event_family", self._event_family(event)),
                event["frame"],
                event["window_start"],
                event["window_end"],
                event["subject"],
                event["object"],
                event.get("line_name", ""),
                event.get("confidence", 0.0),
                event.get("factor_confidence", 0.0),
                num_frames,
                clip_path,
                payload_path,
                "",
                "",
            ])

    def _parse_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def _event_family(self, event):
        if event.get("line_name"):
            return "HLI"
        if isinstance(event.get("object"), str):
            return "HLI"
        return "HHI"
