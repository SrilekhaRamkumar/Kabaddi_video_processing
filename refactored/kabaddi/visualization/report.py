from collections import deque

import cv2
import numpy as np


class ConfirmedInteractionReportBuilder:
    def __init__(self, max_buffer_frames=240):
        self.frame_buffer = deque(maxlen=max_buffer_frames)
        self.segments = []
        self.pending_events = []
        self.captured_event_keys = set()
        self.classifier_inputs = []

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

    def has_classifier_inputs(self):
        return bool(self.classifier_inputs)

    def consume_classifier_inputs(self):
        inputs = list(self.classifier_inputs)
        self.classifier_inputs.clear()
        return inputs

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

            classifier_card = self._build_classifier_card(frame_size, event)
            if classifier_card is not None:
                for _ in range(max(10, int(fps * 0.75))):
                    writer.write(classifier_card)

            for frame_idx, frame in segment["raw_frames"]:
                writer.write(self._annotate_frame(frame, event, frame_idx))

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

            raw_frames = []
            classifier_frames = []
            for frame_idx in range(event["window_start"], event["window_end"] + 1):
                frame = frame_map.get(frame_idx)
                if frame is None:
                    continue
                classifier_frames.append(frame.copy())
                raw_frames.append((frame_idx, frame.copy()))

            if raw_frames:
                self.segments.append({
                    "event": event,
                    "raw_frames": raw_frames,
                })
                self.classifier_inputs.append({
                    "event": event,
                    "frames": classifier_frames,
                    "payload": event.get("classifier_payload", {}),
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
        family = event.get("event_family", "")
        line_name = event.get("line_name")
        if family or line_name:
            family_text = f"Family: {family}" if family else ""
            if line_name:
                family_text = f"{family_text} | Line: {line_name}" if family_text else f"Line: {line_name}"
            cv2.putText(
                annotated,
                family_text,
                (32, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 220, 255),
                2,
            )
            core_y = 134
            metrics_y = 160
            classifier_y = 186
        else:
            core_y = 108
            metrics_y = 134
            classifier_y = 160
        cv2.putText(
            annotated,
            f"Core: {event.get('core_window_start', event['window_start'])} - {event.get('core_window_end', event['window_end'])}",
            (32, core_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
        )
        cv2.putText(
            annotated,
            f"Frame: {frame_idx} | Conf: {event['confidence']:.2f} | Factor: {event.get('factor_confidence', 0.0):.2f}",
            (32, metrics_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
        )
        classifier_result = event.get("classifier_result")
        if classifier_result:
            valid_prob = classifier_result["probabilities"].get("valid", 0.0)
            cv2.putText(
                annotated,
                f"Model: {classifier_result.get('model_name', 'classifier')} | {classifier_result['predicted_label']} | Valid {valid_prob:.2f}",
                (32, classifier_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (120, 220, 255) if classifier_result.get("guaranteed") else (210, 210, 210),
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
        family = event.get("event_family", "")
        line_name = event.get("line_name")
        if family or line_name:
            family_text = f"Family {family}" if family else ""
            if line_name:
                family_text = f"{family_text} | Line {line_name}" if family_text else f"Line {line_name}"
            cv2.putText(
                card,
                family_text,
                (40, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (180, 220, 255),
                2,
            )
            classifier_y = 255
        else:
            classifier_y = 220
        classifier_result = event.get("classifier_result")
        if classifier_result:
            valid_prob = classifier_result["probabilities"].get("valid", 0.0)
            cv2.putText(
                card,
                f"Classifier {classifier_result['predicted_label']} | Valid {valid_prob:.2f}",
                (40, classifier_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (120, 220, 255) if classifier_result.get("guaranteed") else (210, 210, 210),
                2,
            )
        return card

    def _build_classifier_card(self, frame_size, event):
        classifier_result = event.get("classifier_result")
        if not classifier_result:
            return None

        width, height = frame_size
        card = np.zeros((height, width, 3), dtype=np.uint8)
        card[:] = (12, 20, 28)

        valid_prob = classifier_result["probabilities"].get("valid", 0.0)
        invalid_prob = classifier_result["probabilities"].get("invalid", 0.0)
        uncertain_prob = classifier_result["probabilities"].get("uncertain", 0.0)

        cv2.putText(
            card,
            "Visual Touch Confirmation",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (120, 220, 255),
            2,
        )
        cv2.putText(
            card,
            f"Model: {classifier_result.get('model_name', 'classifier')}",
            (40, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            card,
            f"Decision: {classifier_result['predicted_label']}",
            (40, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 255, 200) if classifier_result.get("guaranteed") else (255, 220, 120),
            2,
        )
        cv2.putText(
            card,
            f"Valid {valid_prob:.2f} | Invalid {invalid_prob:.2f} | Uncertain {uncertain_prob:.2f}",
            (40, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 255, 200),
            2,
        )
        cv2.putText(
            card,
            "This review is shown before any scoring interpretation.",
            (40, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 210, 210),
            2,
        )
        return card
