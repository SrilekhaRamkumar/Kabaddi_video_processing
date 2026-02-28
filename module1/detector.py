# detector.py

from ultralytics import YOLO
import torch

class PlayerDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device used:", self.device)
        self.model = YOLO(model_path).to(self.device)

    def detect(self, frame, conf_thresh):
        results = self.model(frame, device=self.device, verbose=False)[0]
        return results
