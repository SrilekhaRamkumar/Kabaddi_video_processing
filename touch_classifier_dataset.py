import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ConfirmedWindowTouchDataset(Dataset):
    """
    Loads exported confirmed windows from the manifest and converts each clip
    into a fixed-length tensor for binary touch classification.
    """

    LABEL_MAP = {
        "valid_touch": 1,
        "touch": 1,
        "positive": 1,
        "no_touch": 0,
        "invalid": 0,
        "negative": 0,
    }

    def __init__(
        self,
        root_dir,
        split="train",
        num_frames=12,
        image_size=224,
        include_hli=False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.include_hli = include_hli
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        frames = self._read_video(sample["clip_path"])
        clip_tensor = self._prepare_clip(frames)
        return {
            "clip": clip_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "clip_id": sample["clip_id"],
            "event_type": sample["event_type"],
        }

    def _load_samples(self):
        manifest_path = self.root_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        split_path = self.root_dir / f"{self.split}_clips.txt"
        split_ids = None
        if split_path.exists():
            split_ids = {
                line.strip()
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

        samples = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            for line in handle:
                values = self._parse_csv_line(line.strip())
                if len(values) != len(header):
                    continue
                row = dict(zip(header, values))
                clip_id = row["clip_id"]
                if split_ids is not None and clip_id not in split_ids:
                    continue

                label = self._load_label(Path(row["payload_path"]))
                if label is None:
                    continue

                event_family = row.get("event_family", "")
                if not self.include_hli and event_family == "HLI":
                    continue

                clip_path = Path(row["clip_path"])
                if not clip_path.exists():
                    continue

                samples.append({
                    "clip_id": clip_id,
                    "clip_path": clip_path,
                    "event_type": row["event_type"],
                    "label": label,
                })

        if not samples:
            raise RuntimeError(
                f"No labeled samples found in {manifest_path}. "
                "Add labels to payload json files or create split text files."
            )
        return samples

    def _load_label(self, payload_path):
        if not payload_path.exists():
            return None
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_label = str(payload.get("label", "")).strip().lower()
        if raw_label not in self.LABEL_MAP:
            return None
        return self.LABEL_MAP[raw_label]

    def _read_video(self, clip_path):
        capture = cv2.VideoCapture(str(clip_path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        capture.release()
        if not frames:
            raise RuntimeError(f"Could not read frames from {clip_path}")
        return frames

    def _prepare_clip(self, frames):
        indices = self._sample_indices(len(frames))
        processed = []
        for idx in indices:
            frame = frames[idx]
            resized = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
            processed.append(np.transpose(normalized, (2, 0, 1)))
        clip = np.stack(processed, axis=0)
        return torch.tensor(clip, dtype=torch.float32)

    def _sample_indices(self, num_available):
        if num_available >= self.num_frames:
            return np.linspace(0, num_available - 1, self.num_frames).round().astype(int).tolist()
        indices = list(range(num_available))
        while len(indices) < self.num_frames:
            indices.append(indices[-1])
        return indices

    def _parse_csv_line(self, line):
        values = []
        current = []
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                values.append("".join(current))
                current = []
            else:
                current.append(char)
        values.append("".join(current))
        return values
