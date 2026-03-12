import numpy as np


class ConfirmedWindowClassifierBridge:
    """
    Lightweight bridge between confirmed temporal windows and a future trained model.

    The current implementation is heuristic and deterministic, but it exposes a stable
    API that can later be replaced with a trained classifier without touching the rest
    of the pipeline.
    """

    def __init__(self):
        self.history = []

    def score_batch(self, classifier_inputs):
        results = []
        for item in classifier_inputs:
            results.append(self.score_window(item))
        return results

    def score_window(self, classifier_input):
        event = classifier_input["event"]
        frames = classifier_input.get("frames", [])
        payload = classifier_input.get("payload", {})
        features = self._featurize(payload, frames, event)
        probabilities = self._score_probabilities(event["type"], features)
        predicted_label = max(probabilities, key=probabilities.get) if probabilities else "uncertain"
        result = {
            "event_key": (event["type"], event["frame"], event["subject"], event["object"]),
            "event_type": event["type"],
            "predicted_label": predicted_label,
            "probabilities": probabilities,
            "features": features,
            "guaranteed": probabilities.get("valid", 0.0) >= 0.82 and predicted_label == "valid",
            "model_name": "heuristic_window_bridge_v1",
        }
        self.history.append(result)
        return result

    def _featurize(self, payload, frames, event):
        aggregates = payload.get("aggregates", {})
        temporal_trace = payload.get("temporal_trace", [])
        graph_snapshot = payload.get("graph_snapshot", {})
        global_context = graph_snapshot.get("global_context", {})

        mean_motion = 0.0
        mean_intensity = 0.0
        if frames:
            gray_means = []
            frame_diffs = []
            prev = None
            for frame in frames:
                gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
                gray_means.append(float(np.mean(gray)))
                if prev is not None:
                    frame_diffs.append(float(np.mean(np.abs(gray - prev))))
                prev = gray
            mean_intensity = float(np.mean(gray_means)) if gray_means else 0.0
            mean_motion = float(np.mean(frame_diffs)) if frame_diffs else 0.0

        trace_contact = [step["best_contact_score"] for step in temporal_trace]
        trace_containment = [step["best_containment_score"] for step in temporal_trace]

        return {
            "avg_proposal_confidence": float(aggregates.get("avg_proposal_confidence", 0.0)),
            "avg_factor_confidence": float(aggregates.get("avg_factor_confidence", 0.0)),
            "peak_window_pair_score": float(aggregates.get("peak_window_pair_score", 0.0)),
            "peak_window_line_score": float(aggregates.get("peak_window_line_score", 0.0)),
            "peak_window_containment": float(aggregates.get("peak_window_containment", 0.0)),
            "visible_defenders": float(aggregates.get("visible_defenders", 0.0)),
            "trace_contact_mean": float(np.mean(trace_contact)) if trace_contact else 0.0,
            "trace_containment_mean": float(np.mean(trace_containment)) if trace_containment else 0.0,
            "trace_contact_peak": float(np.max(trace_contact)) if trace_contact else 0.0,
            "trace_containment_peak": float(np.max(trace_containment)) if trace_containment else 0.0,
            "raider_to_endline": float(global_context.get("raider_to_endline", 0.0)),
            "mean_motion": mean_motion,
            "mean_intensity": mean_intensity,
            "window_length": float(len(payload.get("window_frames", []))),
            "core_length": float(len(payload.get("core_frames", []))),
            "requires_visual_confirmation": 1.0 if event.get("requires_visual_confirmation") else 0.0,
        }

    def _score_probabilities(self, event_type, features):
        if event_type == "CONFIRMED_RAIDER_DEFENDER_CONTACT":
            valid = (
                0.28 * features["avg_proposal_confidence"]
                + 0.22 * features["avg_factor_confidence"]
                + 0.22 * features["peak_window_pair_score"]
                + 0.18 * features["trace_contact_peak"]
                + 0.10 * min(1.0, features["mean_motion"] / 12.0)
            )
        elif event_type in {"CONFIRMED_RAIDER_BONUS_TOUCH", "CONFIRMED_RAIDER_BAULK_TOUCH"}:
            valid = (
                0.28 * features["avg_proposal_confidence"]
                + 0.25 * features["avg_factor_confidence"]
                + 0.27 * features["peak_window_line_score"]
                + 0.10 * min(1.0, features["visible_defenders"] / 7.0)
                + 0.10 * max(0.0, 1.0 - features["raider_to_endline"] / 6.5)
            )
        elif event_type == "CONFIRMED_DEFENDER_ENDLINE_TOUCH":
            valid = (
                0.30 * features["avg_proposal_confidence"]
                + 0.30 * features["avg_factor_confidence"]
                + 0.25 * features["peak_window_line_score"]
                + 0.15 * min(1.0, features["window_length"] / 10.0)
            )
        else:
            valid = 0.5 * features["avg_proposal_confidence"] + 0.5 * features["avg_factor_confidence"]

        valid = float(np.clip(valid, 0.0, 1.0))
        invalid = float(np.clip(1.0 - valid, 0.0, 1.0))
        uncertain = float(np.clip(1.0 - abs(valid - 0.5) * 2.0, 0.0, 1.0))
        total = valid + invalid + uncertain + 1e-6
        return {
            "valid": valid / total,
            "invalid": invalid / total,
            "uncertain": uncertain / total,
        }
