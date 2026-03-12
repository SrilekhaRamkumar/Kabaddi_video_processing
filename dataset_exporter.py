import csv
import json
import os
from typing import List

import cv2


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
        if event_key in self.exported_keys:
            return None

        clip_id = self._clip_id(event)
        clip_dir = os.path.join(self.root_dir, event["type"])
        os.makedirs(clip_dir, exist_ok=True)

        clip_path = os.path.join(clip_dir, f"{clip_id}.mp4")
        payload_path = os.path.join(clip_dir, f"{clip_id}.json")
        frames = classifier_input.get("frames", [])
        if not frames:
            return None

        self._write_clip(clip_path, frames)
        self._write_payload(payload_path, classifier_input)
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
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

    def _write_payload(self, payload_path, classifier_input):
        event = classifier_input["event"]
        payload = classifier_input.get("payload", {})
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
