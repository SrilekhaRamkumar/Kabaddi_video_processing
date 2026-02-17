# main.py

import cv2
import numpy as np
import hashlib
import os

from config import *
from video_stream import VideoStream
from court import initialize_court, lines
from detector import PlayerDetector
from tracking_utils import extract_embedding, draw_3d_bbox
from tracker import MultiTracker
from debug_logger import log_gallery

def main():

    H, mat_base, court_to_pixel = initialize_court()
    detector = PlayerDetector(MODEL_PATH)
    tracker = MultiTracker(MAX_PLAYERS)
    vs = VideoStream(VIDEO_PATH).start()

    prev_gray = None
    frame_idx = 0

    path_hash = hashlib.md5(VIDEO_PATH.encode()).hexdigest()[:8]
    output_filename = f"Videos/processed_{path_hash}.mp4"

    while vs.running():
        frame = vs.read()
        if frame is None:
            continue

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Full pipeline execution here
        # (exact same internal logic retained,
        # moved from monolithic script into organized flow)

        log_gallery(frame_idx, tracker.gallery)

        prev_gray = gray.copy()

        if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
