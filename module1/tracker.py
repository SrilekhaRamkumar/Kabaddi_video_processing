# tracker.py

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from tracking_utils import create_kalman, cosine

class MultiTracker:
    def __init__(self, max_players):
        self.gallery = {}
        self.next_id = 0
        self.max_players = max_players

    def predict(self):
        return {pid: data["kf"].predict() for pid,data in self.gallery.items()}

    def add_track(self, foot, emb, bbox):
        self.gallery[self.next_id] = {
            "feat": emb,
            "kf": create_kalman(*foot),
            "age": 0,
            "display_pos": None,
            "flow_pts": None,
            "last_bbox": bbox
        }
        self.next_id += 1
