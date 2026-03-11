import numpy as np


class TemporalInteractionCandidateManager:
    def __init__(self, max_gap=3, pre_context=10, post_context=10):
        self.max_gap = max_gap
        self.pre_context = pre_context
        self.post_context = post_context
        self.active_candidates = {}
        self.confirmed_events = []

    def update(self, frame_idx, frame_proposals, player_states, raider_id, scene_graph=None):
        confirmed_now = []
        seen_keys = set()
        pair_factor_map = self._pair_factor_map(scene_graph)
        line_factor_map = self._line_factor_map(scene_graph)

        for proposal in frame_proposals:
            key = self._candidate_key(proposal)
            seen_keys.add(key)
            confidence = self._proposal_confidence(proposal, player_states, raider_id)
            factor_confidence = self._factor_confidence(proposal, pair_factor_map, line_factor_map)

            candidate = self.active_candidates.get(key)
            if candidate is None:
                candidate = {
                    "key": key,
                    "type": proposal["type"],
                    "subject": proposal["S"],
                    "object": proposal["O"],
                    "interaction": proposal["I"],
                    "start_frame": frame_idx,
                    "last_frame": frame_idx,
                    "frames": [],
                    "confidences": [],
                    "factor_confidences": [],
                    "peak_confidence": 0.0,
                    "confirmed": False,
                }
                self.active_candidates[key] = candidate

            candidate["last_frame"] = frame_idx
            candidate["frames"].append(frame_idx)
            candidate["confidences"].append(confidence)
            candidate["factor_confidences"].append(factor_confidence)
            combined_conf = 0.65 * confidence + 0.35 * factor_confidence
            candidate["peak_confidence"] = max(candidate["peak_confidence"], combined_conf)

            event = self._try_confirm(candidate, proposal, raider_id)
            if event is not None:
                confirmed_now.append(event)

        stale_keys = []
        for key, candidate in self.active_candidates.items():
            if key in seen_keys:
                continue
            if frame_idx - candidate["last_frame"] > self.max_gap:
                stale_keys.append(key)

        for key in stale_keys:
            del self.active_candidates[key]

        self.confirmed_events.extend(confirmed_now)
        return confirmed_now

    def _pair_factor_map(self, scene_graph):
        if not scene_graph:
            return {}
        pair_factors = scene_graph.get("pair_factors", [])
        return {
            tuple(pair_factor["nodes"]): pair_factor
            for pair_factor in pair_factors
            if pair_factor.get("type") == "RAIDER_DEFENDER_PAIR"
        }

    def _line_factor_map(self, scene_graph):
        if not scene_graph:
            return {}
        line_factors = scene_graph.get("line_factors", [])
        return {
            (line_factor["nodes"][0], line_factor["line"]): line_factor
            for line_factor in line_factors
            if line_factor.get("type") == "PLAYER_LINE_FACTOR" and line_factor.get("nodes")
        }

    def _candidate_key(self, proposal):
        return (proposal["type"], proposal["S"], proposal["O"])

    def _proposal_confidence(self, proposal, player_states, raider_id):
        if proposal["type"] == "HHI":
            dist_conf = max(0.0, 1.0 - proposal["features"]["dist"] / 1.5)
            rel_vel = proposal["features"]["rel_vel"]
            vel_conf = min(1.0, rel_vel / 2.0)
            subject_conf = player_states.get(proposal["S"], {}).get("track_confidence", 0.5)
            object_conf = player_states.get(proposal["O"], {}).get("track_confidence", 0.5)
            role_boost = 0.1 if proposal["S"] == raider_id else 0.0
            return min(1.0, 0.5 * dist_conf + 0.25 * vel_conf + 0.25 * (subject_conf + object_conf) / 2.0 + role_boost)

        dist_conf = max(0.0, 1.0 - proposal["features"]["dist"] / 0.35)
        subject_conf = player_states.get(proposal["S"], {}).get("track_confidence", 0.5)
        active_boost = 0.15 if proposal["features"].get("active") else 0.0
        return min(1.0, 0.7 * dist_conf + 0.3 * subject_conf + active_boost)

    def _factor_confidence(self, proposal, pair_factor_map, line_factor_map):
        if proposal["type"] == "HHI":
            pair_factor = pair_factor_map.get((proposal["S"], proposal["O"]))
            if pair_factor is None:
                return 0.5
            features = pair_factor["features"]
            proximity = max(0.0, 1.0 - features["distance"] / 1.2)
            rel_vel = min(1.0, features["relative_velocity"] / 1.8)
            approach = min(1.0, features["approach_score"])
            adjacency = min(1.0, features["adjacency"])
            return min(1.0, 0.4 * proximity + 0.2 * rel_vel + 0.2 * approach + 0.2 * adjacency)

        line_factor = line_factor_map.get((proposal["S"], proposal["O"]))
        if line_factor is None:
            return 0.5
        features = line_factor["features"]
        distance_conf = max(0.0, 1.0 - features["distance"] / 0.35)
        active_boost = 0.2 if features["active"] else 0.0
        return min(1.0, 0.65 * distance_conf + 0.35 * features["track_confidence"] + active_boost)

    def _try_confirm(self, candidate, proposal, raider_id):
        if candidate["confirmed"]:
            return None

        frame_count = len(candidate["frames"])
        avg_conf = float(np.mean(candidate["confidences"])) if candidate["confidences"] else 0.0
        avg_factor_conf = float(np.mean(candidate["factor_confidences"])) if candidate["factor_confidences"] else 0.0
        fused_conf = 0.6 * avg_conf + 0.4 * avg_factor_conf

        if proposal["type"] == "HHI" and proposal["S"] == raider_id:
            if frame_count >= 2 and fused_conf >= 0.58:
                candidate["confirmed"] = True
                return self._build_event(candidate, "CONFIRMED_RAIDER_DEFENDER_CONTACT", fused_conf, True, avg_factor_conf)

        if proposal["type"] == "HLI" and proposal["S"] == raider_id and proposal["O"] == "BONUS":
            if frame_count >= 2 and fused_conf >= 0.62 and proposal["features"]["active"]:
                candidate["confirmed"] = True
                return self._build_event(candidate, "CONFIRMED_RAIDER_BONUS_TOUCH", fused_conf, False, avg_factor_conf)

        if proposal["type"] == "HLI" and proposal["S"] == raider_id and proposal["O"] == "BAULK":
            if frame_count >= 2 and fused_conf >= 0.62 and proposal["features"]["active"]:
                candidate["confirmed"] = True
                return self._build_event(candidate, "CONFIRMED_RAIDER_BAULK_TOUCH", fused_conf, False, avg_factor_conf)

        if proposal["type"] == "HLI" and proposal["O"] == "END_LINE" and proposal["S"] != raider_id:
            if frame_count >= 2 and fused_conf >= 0.62 and proposal["features"]["active"]:
                candidate["confirmed"] = True
                return self._build_event(candidate, "CONFIRMED_DEFENDER_ENDLINE_TOUCH", fused_conf, False, avg_factor_conf)

        return None

    def _build_event(self, candidate, event_type, confidence, requires_visual_confirmation, factor_confidence):
        return {
            "type": event_type,
            "frame": candidate["last_frame"],
            "window_start": max(1, candidate["start_frame"] - self.pre_context),
            "window_end": candidate["last_frame"] + self.post_context,
            "core_window_start": candidate["start_frame"],
            "core_window_end": candidate["last_frame"],
            "subject": candidate["subject"],
            "object": candidate["object"],
            "confidence": confidence,
            "factor_confidence": factor_confidence,
            "requires_visual_confirmation": requires_visual_confirmation,
        }
