"""
Core Module
Handles tracking, detection, and video streaming.
"""
from .tracking import (
    create_kalman,
    extract_embedding,
    cosine,
    bbox_iou,
    draw_3d_bbox,
    apply_optical_flow,
    run_yolo_detection,
    update_tracks,
    add_new_tracks,
    render_gallery,
)
from .video_stream import VideoStream

__all__ = [
    "create_kalman",
    "extract_embedding",
    "cosine",
    "bbox_iou",
    "draw_3d_bbox",
    "apply_optical_flow",
    "run_yolo_detection",
    "update_tracks",
    "add_new_tracks",
    "render_gallery",
    "VideoStream",
]

