"""
Constants Module
Defines court dimensions, thresholds, and configuration values.
"""

# Court dimensions (in meters)
COURT_WIDTH = 10.0
COURT_HEIGHT = 6.5
BAULK_LINE_Y = 3.75
BONUS_LINE_Y = 4.75
MID_LINE_Y = 0.0
END_LINE_Y = 6.5
LOBBY_LEFT_X = 0.75
LOBBY_RIGHT_X = 9.25

# Visualization dimensions (in pixels)
COURT_MAT_WIDTH = 400
COURT_MAT_HEIGHT = 260

# Tracking thresholds
MAX_TRACK_AGE = 15
TRACK_CONFIDENCE_THRESHOLD = 0.5
VISIBILITY_CONFIDENCE_THRESHOLD = 0.6

# Interaction thresholds
INTERACTION_DISTANCE_THRESHOLD = 1.5
TOUCH_DISTANCE_THRESHOLD = 1.0
LINE_MARGIN = 0.35

# Temporal validation
TEMPORAL_MAX_GAP = 3
TEMPORAL_PRE_CONTEXT = 10
TEMPORAL_POST_CONTEXT = 10

# Raider assignment
RAIDER_ASSIGN_FRAME = 60
MIN_RAIDER_STATS_FRAMES = 20

# HSV embedding
HSV_BINS = (8, 8, 8)

# Video buffer
MAX_BUFFER_FRAMES = 240

# Confidence thresholds
MIN_ACTION_CONFIDENCE = 0.58
MIN_CONTACT_CONFIDENCE = 0.58
MIN_LINE_TOUCH_CONFIDENCE = 0.62
MIN_TACKLE_CONFIDENCE = 0.68

# Colors for visualization (BGR)
COLOR_RAIDER = (0, 255, 0)  # Green
COLOR_DEFENDER = (255, 0, 0)  # Blue
COLOR_TRACK = (255, 255, 0)  # Cyan
COLOR_INTERACTION = (0, 255, 255)  # Yellow
COLOR_TEXT = (255, 255, 255)  # White
COLOR_BACKGROUND = (0, 0, 0)  # Black

