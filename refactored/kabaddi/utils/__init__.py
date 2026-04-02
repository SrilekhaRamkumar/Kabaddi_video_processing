"""
Utils Module
Provides geometry utilities and constants.
"""
from .constants import *
from .geometry import (
    line_eq,
    intersect,
    compute_homography,
    create_court_mat,
    court_to_pixel,
    select_line_coordinates,
)

__all__ = [
    "line_eq",
    "intersect",
    "compute_homography",
    "create_court_mat",
    "court_to_pixel",
    "select_line_coordinates",
    "COURT_WIDTH",
    "COURT_HEIGHT",
    "BAULK_LINE_Y",
    "BONUS_LINE_Y",
    "MID_LINE_Y",
    "END_LINE_Y",
    "LOBBY_LEFT_X",
    "LOBBY_RIGHT_X",
    "COURT_MAT_WIDTH",
    "COURT_MAT_HEIGHT",
    "MAX_TRACK_AGE",
    "TRACK_CONFIDENCE_THRESHOLD",
    "VISIBILITY_CONFIDENCE_THRESHOLD",
    "INTERACTION_DISTANCE_THRESHOLD",
    "TOUCH_DISTANCE_THRESHOLD",
    "LINE_MARGIN",
    "TEMPORAL_MAX_GAP",
    "TEMPORAL_PRE_CONTEXT",
    "TEMPORAL_POST_CONTEXT",
    "RAIDER_ASSIGN_FRAME",
    "MIN_RAIDER_STATS_FRAMES",
    "HSV_BINS",
    "MAX_BUFFER_FRAMES",
    "MIN_ACTION_CONFIDENCE",
    "MIN_CONTACT_CONFIDENCE",
    "MIN_LINE_TOUCH_CONFIDENCE",
    "MIN_TACKLE_CONFIDENCE",
    "COLOR_RAIDER",
    "COLOR_DEFENDER",
    "COLOR_TRACK",
    "COLOR_INTERACTION",
    "COLOR_TEXT",
    "COLOR_BACKGROUND",
]

