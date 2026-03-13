"""
Report Video Builder Module
Generates event-specific video clips with annotations.
"""
from collections import deque

import cv2
import numpy as np


class ConfirmedInteractionReportBuilder:
    def __init__(self, max_buffer_frames=240):
        self.frame_buffer = deque(maxlen=max_buffer_frames)
        self.segments = []
        self.pending_events = []
        self.captured_event_keys = set()

    def add_frame(self, frame_idx, frame):
        self.frame_buffer.append((frame_idx, frame.copy()))
        self._flush_pending_events(frame_idx)

    def capture_events(self, events):
        if not events:
            return

        for event in events:
            event_key = (event["type"], event["frame"], event["subject"], event["object"])
            if event_key in self.captured_event_keys:
                continue
            self.captured_event_keys.add(event_key)
            self.pending_events.append(event)

    def has_segments(self):
        return bool(self.segments or self.pending_events)

    def write_video(self, output_path, fps, frame_size):
        if not self.segments:
            if self.pending_events and self.frame_buffer:
                self._flush_pending_events(self.frame_buffer[-1][0], force=True)
            if not self.segments:
                return False

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )

        for idx, segment in enumerate(self.segments, start=1):
            event = segment["event"]
            title_card = self._build_title_card(frame_size, event, idx, len(self.segments))
            for _ in range(int(fps)):
                writer.write(title_card)

            for frame in segment["frames"]:
                writer.write(frame)

        writer.release()
        return True

    def _flush_pending_events(self, current_frame_idx, force=False):
        if not self.pending_events:
            return

        frame_map = {frame_idx: frame for frame_idx, frame in self.frame_buffer}
        remaining = []
        for event in self.pending_events:
            if not force and current_frame_idx < event["window_end"]:
                remaining.append(event)
                continue

            segment_frames = []
            for frame_idx in range(event["window_start"], event["window_end"] + 1):
                frame = frame_map.get(frame_idx)
                if frame is None:
                    continue
                segment_frames.append(self._annotate_frame(frame, event, frame_idx))

            if segment_frames:
                self.segments.append({
                    "event": event,
                    "frames": segment_frames,
                })

        self.pending_events = remaining

    def _annotate_frame(self, frame, event, frame_idx):
        annotated = frame.copy()
        overlay = annotated.copy()
        cv2.rectangle(overlay, (18, 18), (annotated.shape[1] - 18, 118), (25, 25, 25), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        cv2.putText(
            annotated,
            f"Confirmed: {event['type']}",
            (32, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Window: {event['window_start']} - {event['window_end']}",
            (32, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Core: {event.get('core_window_start', event['window_start'])} - {event.get('core_window_end', event['window_end'])}",
            (32, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
        )
        cv2.putText(
            annotated,
            f"Frame: {frame_idx} | Conf: {event['confidence']:.2f} | Factor: {event.get('factor_confidence', 0.0):.2f}",
            (32, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
        )
        return annotated

    def _build_title_card(self, frame_size, event, segment_idx, total_segments):
        width, height = frame_size
        card = np.zeros((height, width, 3), dtype=np.uint8)
        card[:] = (18, 18, 18)
        cv2.putText(
            card,
            "Kabaddi Confirmed Interaction Report",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            card,
            f"Segment {segment_idx}/{total_segments}: {event['type']}",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            card,
            f"Window {event['window_start']} - {event['window_end']} | Core {event.get('core_window_start', event['window_start'])} - {event.get('core_window_end', event['window_end'])}",
            (40, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 255, 180),
            2,
        )
        cv2.putText(
            card,
            f"Confidence {event['confidence']:.2f} | Factor {event.get('factor_confidence', 0.0):.2f}",
            (32, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 255, 180),
            2,
        )
        return card

