"""
Touch Classifier Inference

Load a trained touch classifier model and run inference on video clips.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from kabaddi.classifier.model import TouchWindowClassifier


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["no_touch", "valid_touch"]


class TouchClassifierInference:
    def __init__(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.config = checkpoint.get("config", {})
        self.num_frames = int(self.config.get("num_frames", 12))
        self.image_size = int(self.config.get("image_size", 224))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TouchWindowClassifier(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_video(self, clip_path):
        frames = self._read_video(clip_path)
        return self.predict_frames(frames)

    @torch.no_grad()
    def predict_frames(self, frames):
        clip = self._prepare_clip(frames).unsqueeze(0).to(self.device)
        logits = self.model(clip)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_index = int(np.argmax(probs))
        return {
            "predicted_label": CLASS_NAMES[label_index],
            "probabilities": {
                CLASS_NAMES[idx]: float(probs[idx])
                for idx in range(len(CLASS_NAMES))
            },
        }

    def _read_video(self, clip_path):
        capture = cv2.VideoCapture(str(clip_path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
        return torch.tensor(np.stack(processed, axis=0), dtype=torch.float32)

    def _sample_indices(self, num_available):
        if num_available >= self.num_frames:
            return np.linspace(0, num_available - 1, self.num_frames).round().astype(int).tolist()
        indices = list(range(num_available))
        while len(indices) < self.num_frames:
            indices.append(indices[-1])
        return indices

