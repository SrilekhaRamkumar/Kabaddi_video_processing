"""
Kabaddi API Module

FastAPI server for live streaming and archived event playback.
"""

from kabaddi.api.server import app, attach_queues, set_run_id

__all__ = ["app", "attach_queues", "set_run_id"]

