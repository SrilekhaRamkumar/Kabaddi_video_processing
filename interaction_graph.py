import numpy as np
from itertools import combinations
import cv2


class InteractionProposalEngine:
    """Encodes atomic actions into <S, I, O> triplets with strict per-frame uniqueness."""

    def __init__(self):
        self.candidate_proposals = []
        self._frame_cache = {}

    def reset_proposals(self):
        self.candidate_proposals = []
        self._frame_cache = {}

    def _add_unique_proposal(self, frame_idx, proposal):
        key = (frame_idx, proposal["type"], proposal["S"], proposal["O"])
        if key not in self._frame_cache:
            self._frame_cache[key] = proposal
        elif proposal["features"]["dist"] < self._frame_cache[key]["features"]["dist"]:
            self._frame_cache[key] = proposal

    def finalize_frame_proposals(self):
        frame_proposals = list(self._frame_cache.values())
        for proposal in frame_proposals:
            self.candidate_proposals.append(proposal)
        self._frame_cache = {}
        return frame_proposals

    def encode_hhi(self, frame_idx, raider_id, defender_id, r_pos, d_pos, r_vel, d_vel, r_feat, d_feat):
        dist = np.sqrt((r_pos[0] - d_pos[0]) ** 2 + (r_pos[1] - d_pos[1]) ** 2)
        proposal = {
            "frame": frame_idx,
            "type": "HHI",
            "S": raider_id,
            "O": defender_id,
            "I": "POTENTIAL_CONTACT",
            "features": {
                "dist": dist,
                "rel_vel": np.linalg.norm(np.array(r_vel) - np.array(d_vel)),
                "mask": [d_pos[0] - r_pos[0], d_pos[1] - r_pos[1]],
                "emb": (0.5 * r_feat + 0.5 * d_feat).tolist(),
            },
        }
        self._add_unique_proposal(frame_idx, proposal)

    def encode_hli(self, frame_idx, player_id, line_name, p_pos, line_y):
        dist_to_line = abs(p_pos[1] - line_y)
        proposal = {
            "frame": frame_idx,
            "type": "HLI",
            "S": player_id,
            "O": line_name,
            "I": "LINE_PROXIMITY",
            "features": {"dist": dist_to_line, "active": dist_to_line < 0.25},
        }
        self._add_unique_proposal(frame_idx, proposal)


class ActiveFactorGraphNetwork:
    """Constructs a structured graph from triplets using AFGN and Lee et al. methods."""

    def __init__(self, top_k=4):
        self.top_k = top_k
        self.active_nodes = []
        self.adjacency_matrix = None
        self.factor_nodes = []
        self.pair_factors = []
        self.line_factors = []
        self.full_nodes = []
        self.full_adjacency_matrix = None
        self.full_pair_factors = []
        self.full_line_factors = []
        self.full_factor_nodes = []
        self.global_context = {}

    def build_graph(self, proposals, gallery, raider_id):
        visible_ids = [
            pid for pid, data in gallery.items()
            if data["age"] == 0 and data.get("display_pos") is not None
        ]
        influence = {pid: 0.0 for pid in visible_ids}
        pair_proposals = {}
        line_proposals = {}
        raider_pos = gallery.get(raider_id, {}).get("display_pos") if raider_id in gallery else None

        for proposal in proposals:
            if proposal["type"] == "HHI":
                dist_weight = 1.0 / (proposal["features"]["dist"] + 1e-6)
                vel_weight = proposal["features"]["rel_vel"]
                emb = np.array(proposal["features"]["emb"])
                half = len(emb) // 2
                feat_sim = np.dot(emb[:half], emb[half:]) / (
                    np.linalg.norm(emb[:half]) * np.linalg.norm(emb[half:]) + 1e-6
                )
                subject_boost = 1.35 if proposal["S"] == raider_id else 1.0
                object_boost = 1.2 if proposal["O"] == raider_id else 1.0
                weight = dist_weight * max(0.25, vel_weight) * max(0.1, feat_sim + 0.25)

                if proposal["S"] in influence:
                    influence[proposal["S"]] += subject_boost * weight
                if proposal["O"] in influence:
                    influence[proposal["O"]] += object_boost * weight

                pair_key = (proposal["S"], proposal["O"])
                stored = pair_proposals.get(pair_key)
                if stored is None or proposal["features"]["dist"] < stored["features"]["dist"]:
                    pair_proposals[pair_key] = proposal
            elif proposal["type"] == "HLI":
                line_key = (proposal["S"], proposal["O"])
                stored = line_proposals.get(line_key)
                if stored is None or proposal["features"]["dist"] < stored["features"]["dist"]:
                    line_proposals[line_key] = proposal

        if raider_pos is not None:
            for pid in visible_ids:
                if pid == raider_id:
                    influence[pid] += 5.0
                    continue
                player_pos = gallery[pid]["display_pos"]
                dist_to_raider = np.linalg.norm(np.array(player_pos) - np.array(raider_pos))
                raider_dist_score = max(0.0, 1.0 - dist_to_raider / 3.0)
                state = gallery[pid]["kf"].statePost.flatten()
                approach_vec = np.array(raider_pos) - np.array(player_pos)
                norm = np.linalg.norm(approach_vec) + 1e-6
                closing_speed = max(0.0, np.dot(np.array(state[2:4]), approach_vec / norm))
                influence[pid] += 1.8 * raider_dist_score + 0.8 * min(1.0, closing_speed / 1.5)

        for (subject_id, line_name), proposal in line_proposals.items():
            if subject_id not in influence:
                continue
            line_bonus = 0.0
            if subject_id == raider_id and line_name in {"BAULK", "BONUS"}:
                line_bonus = 1.25
            elif line_name == "END_LINE":
                line_bonus = 0.45
            proximity = max(0.0, 1.0 - proposal["features"]["dist"] / 0.8)
            influence[subject_id] += line_bonus * proximity

        sorted_ids = sorted(influence.keys(), key=lambda pid: influence[pid], reverse=True)
        self.full_nodes = sorted_ids
        self.active_nodes = [pid for pid in sorted_ids if pid != raider_id][: max(0, self.top_k - 1)]
        if raider_id is not None and raider_id in gallery:
            self.active_nodes.append(raider_id)
        self.active_nodes = list(dict.fromkeys(self.active_nodes))

        self.full_adjacency_matrix = self._build_adjacency_matrix(self.full_nodes, gallery)
        self.adjacency_matrix = self._build_adjacency_matrix(self.active_nodes, gallery)
        self.full_pair_factors = self._build_pair_factors(self.full_nodes, pair_proposals, gallery, raider_id, self.full_adjacency_matrix)
        self.pair_factors = [factor for factor in self.full_pair_factors if set(factor["nodes"]).issubset(set(self.active_nodes))]
        self.full_line_factors = self._build_line_factors(self.full_nodes, line_proposals, gallery, raider_id)
        self.line_factors = [factor for factor in self.full_line_factors if factor["nodes"][0] in self.active_nodes]
        self.full_factor_nodes = self._build_typed_factor_nodes(self.full_nodes, gallery, raider_id)
        self.factor_nodes = [factor for factor in self.full_factor_nodes if set(factor["triplet"]).issubset(set(self.active_nodes))]
        self.global_context = self._build_global_context(gallery, raider_id, influence, self.full_pair_factors, self.full_factor_nodes)

        return self.package_features(gallery, raider_id)

    def _build_adjacency_matrix(self, node_ids, gallery):
        n = len(node_ids)
        matrix = np.zeros((n, n))
        if n == 0:
            return matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                id_i, id_j = node_ids[i], node_ids[j]
                bb_i, bb_j = gallery[id_i]["last_bbox"], gallery[id_j]["last_bbox"]
                d_i = np.sqrt((bb_i[2] - bb_i[0]) ** 2 + (bb_i[3] - bb_i[1]) ** 2)
                d_j = np.sqrt((bb_j[2] - bb_j[0]) ** 2 + (bb_j[3] - bb_j[1]) ** 2)
                matrix[i, j] = min(d_i, d_j) / max(d_i, d_j)
        return matrix

    def _build_pair_factors(self, node_ids, pair_proposals, gallery, raider_id, adjacency_matrix):
        node_index = {pid: idx for idx, pid in enumerate(node_ids)}
        factors = []
        for subject_id, object_id in pair_proposals:
            if subject_id not in node_index or object_id not in node_index:
                continue
            proposal = pair_proposals[(subject_id, object_id)]
            subject_pos = gallery[subject_id]["display_pos"]
            object_pos = gallery[object_id]["display_pos"]
            if subject_pos is None or object_pos is None:
                continue

            subject_motion = gallery[subject_id]["kf"].statePost.flatten()[2:4]
            object_motion = gallery[object_id]["kf"].statePost.flatten()[2:4]
            relative_vector = np.array(object_pos) - np.array(subject_pos)
            relative_dist = np.linalg.norm(relative_vector) + 1e-6
            approach_score = max(
                0.0,
                np.dot((np.array(subject_motion) - np.array(object_motion)), relative_vector / relative_dist),
            )
            if subject_id == raider_id:
                factor_type = "RAIDER_DEFENDER_PAIR"
            elif object_id == raider_id:
                factor_type = "DEFENDER_RAIDER_PAIR"
            else:
                factor_type = "DEFENDER_DEFENDER_PAIR"

            factors.append({
                "type": factor_type,
                "nodes": [subject_id, object_id],
                "features": {
                    "distance": float(proposal["features"]["dist"]),
                    "relative_velocity": float(proposal["features"]["rel_vel"]),
                    "approach_score": float(min(1.0, approach_score / 2.0)),
                    "adjacency": float(adjacency_matrix[node_index[subject_id], node_index[object_id]]),
                    "track_confidence": float(
                        0.5 * (
                            gallery[subject_id].get("track_confidence", 0.0)
                            + gallery[object_id].get("track_confidence", 0.0)
                        )
                    ),
                    "raider_involved": bool(subject_id == raider_id or object_id == raider_id),
                },
            })
        return factors

    def _build_line_factors(self, node_ids, line_proposals, gallery, raider_id):
        active_set = set(node_ids)
        factors = []
        for subject_id, line_name in line_proposals:
            if subject_id not in active_set:
                continue
            proposal = line_proposals[(subject_id, line_name)]
            factor_type = "RAIDER_LINE_FACTOR" if subject_id == raider_id else "DEFENDER_LINE_FACTOR"
            factors.append({
                "type": factor_type,
                "nodes": [subject_id],
                "line": line_name,
                "features": {
                    "distance": float(proposal["features"]["dist"]),
                    "active": bool(proposal["features"]["active"]),
                    "track_confidence": float(gallery[subject_id].get("track_confidence", 0.0)),
                },
            })
        return factors

    def _build_typed_factor_nodes(self, node_ids, gallery, raider_id):
        active_set = set(node_ids)
        factors = []
        if raider_id not in active_set:
            return factors

        defender_ids = [pid for pid in node_ids if pid != raider_id]
        for defender_id in defender_ids:
            raider_pos = gallery[raider_id]["display_pos"]
            defender_pos = gallery[defender_id]["display_pos"]
            if raider_pos is None or defender_pos is None:
                continue
            for line_name, line_value in (("BAULK", 3.75), ("BONUS", 4.75), ("END_LINE", 6.5)):
                centroid = np.mean([np.array(raider_pos), np.array(defender_pos)], axis=0)
                line_gap = abs(float(centroid[1]) - line_value)
                if line_gap > 1.4:
                    continue
                factors.append({
                    "triplet": (raider_id, defender_id, line_name),
                    "type": "RAIDER_DEFENDER_LINE",
                    "features": {
                        "line": line_name,
                        "line_gap": line_gap,
                        "distance": float(np.linalg.norm(np.array(raider_pos) - np.array(defender_pos))),
                        "centroid": centroid.tolist(),
                    },
                })

        for triplet in combinations(defender_ids, 2):
            pid1, pid2 = triplet
            pos_r = gallery[raider_id]["display_pos"]
            pos1 = gallery[pid1]["display_pos"]
            pos2 = gallery[pid2]["display_pos"]
            if pos_r is None or pos1 is None or pos2 is None:
                continue

            pos_r = np.array(pos_r)
            pos1 = np.array(pos1)
            pos2 = np.array(pos2)
            d1 = np.linalg.norm(pos_r - pos1)
            d2 = np.linalg.norm(pos_r - pos2)
            dd = np.linalg.norm(pos1 - pos2)
            centroid = np.mean([pos_r, pos1, pos2], axis=0)
            spread = np.mean([
                np.linalg.norm(pos_r - centroid),
                np.linalg.norm(pos1 - centroid),
                np.linalg.norm(pos2 - centroid),
            ])
            compactness = max(0.0, 1.0 - np.mean([d1, d2, dd]) / 1.8)
            if compactness < 0.15:
                continue
            factors.append({
                "triplet": (raider_id, pid1, pid2),
                "type": "THIRD_ORDER_PRESSURE",
                "features": {
                    "distances": [float(d1), float(d2), float(dd)],
                    "spread": float(spread),
                    "compactness": float(compactness),
                    "centroid": centroid.tolist(),
                },
            })

            angle = self._triplet_enclosure_angle(pos_r, pos1, pos2)
            if angle > 0.35:
                factors.append({
                    "triplet": (raider_id, pid1, pid2),
                    "type": "DEFENDER_CONTAINMENT",
                    "features": {
                        "angle": float(angle),
                        "spread": float(spread),
                        "distances": [float(d1), float(d2), float(dd)],
                    },
                })
        return factors

    def _triplet_enclosure_angle(self, raider_pos, pos1, pos2):
        v1 = pos1 - raider_pos
        v2 = pos2 - raider_pos
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosine_value = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cosine_value) / np.pi)

    def _build_global_context(self, gallery, raider_id, influence, pair_factors, factor_nodes):
        visible_nodes = [
            pid for pid, data in gallery.items()
            if data["age"] == 0 and data.get("display_pos") is not None
        ]
        raider_pos = gallery.get(raider_id, {}).get("display_pos") if raider_id in gallery else None
        defenders = [pid for pid in visible_nodes if pid != raider_id]
        depths = [gallery[pid]["display_pos"][1] for pid in defenders if gallery[pid]["display_pos"] is not None]
        widths = [gallery[pid]["display_pos"][0] for pid in defenders if gallery[pid]["display_pos"] is not None]
        containment_scores = [
            factor["features"].get("angle", 0.0)
            for factor in factor_nodes
            if factor.get("type") == "DEFENDER_CONTAINMENT"
        ]
        contact_scores = [
            max(0.0, 1.0 - factor["features"]["distance"] / 1.2)
            for factor in pair_factors
            if factor["features"].get("raider_involved")
        ]
        return {
            "visible_players": len(visible_nodes),
            "visible_defenders": len(defenders),
            "raider_in_active_graph": raider_id in self.active_nodes,
            "max_influence": float(max(influence.values()) if influence else 0.0),
            "mean_defender_depth": float(np.mean(depths)) if depths else 0.0,
            "defender_width_span": float(max(widths) - min(widths)) if len(widths) >= 2 else 0.0,
            "best_contact_score": float(max(contact_scores) if contact_scores else 0.0),
            "best_containment_score": float(max(containment_scores) if containment_scores else 0.0),
            "raider_to_endline": float(max(0.0, 6.5 - raider_pos[1])) if raider_pos is not None else 0.0,
        }

    def package_features(self, gallery, raider_id):
        graph_data = {
            "nodes": [],
            "edges": self.adjacency_matrix.tolist(),
            "factor_nodes": self.factor_nodes,
            "pair_factors": self.pair_factors,
            "line_factors": self.line_factors,
            "active_node_ids": list(self.active_nodes),
            "full_nodes": [],
            "full_edges": self.full_adjacency_matrix.tolist() if self.full_adjacency_matrix is not None else [],
            "full_pair_factors": self.full_pair_factors,
            "full_line_factors": self.full_line_factors,
            "full_factor_nodes": self.full_factor_nodes,
            "global_context": self.global_context,
        }
        for pid in self.active_nodes:
            state = gallery[pid]["kf"].statePost.flatten()
            graph_data["nodes"].append({
                "id": pid,
                "role": "RAIDER" if pid == raider_id else "DEFENDER",
                "motion": [float(state[2]), float(state[3])],
                "visual": gallery[pid]["feat"].tolist(),
                "spatial": gallery[pid]["display_pos"],
                "track_confidence": gallery[pid].get("track_confidence", 0.0),
                "visibility_confidence": gallery[pid].get("visibility_confidence", 0.0),
            })
        for pid in self.full_nodes:
            state = gallery[pid]["kf"].statePost.flatten()
            graph_data["full_nodes"].append({
                "id": pid,
                "role": "RAIDER" if pid == raider_id else "DEFENDER",
                "motion": [float(state[2]), float(state[3])],
                "visual": gallery[pid]["feat"].tolist(),
                "spatial": gallery[pid]["display_pos"],
                "track_confidence": gallery[pid].get("track_confidence", 0.0),
                "visibility_confidence": gallery[pid].get("visibility_confidence", 0.0),
            })
        return graph_data


def render_graph_panel(scene_graph, width=420, height=320, frame_idx=None, recent_events=None):
    """Render the current factor graph as a compact court-grounded panel."""
    panel = np.full((height, width, 3), 242, dtype=np.uint8)

    cv2.putText(
        panel,
        "Interaction Graph",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
    )
    if frame_idx is not None:
        cv2.putText(
            panel,
            f"Frame {frame_idx}",
            (width - 110, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 60, 60),
            1,
        )

    if not scene_graph or not scene_graph.get("nodes"):
        cv2.putText(
            panel,
            "No active graph",
            (14, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (90, 90, 90),
            2,
        )
        return panel

    margin_x = 24
    margin_top = 64
    margin_bottom = 22
    court_w = width - 2 * margin_x
    court_h = height - margin_top - margin_bottom - 54

    def to_panel_point(pos):
        x, y = pos
        px = int(margin_x + (x / 10.0) * court_w)
        py = int(margin_top + ((6.5 - y) / 6.5) * court_h)
        return px, py

    line_y = {
        "BAULK": 3.75,
        "BONUS": 4.75,
        "END_LINE": 6.5,
    }

    cv2.rectangle(panel, (margin_x, margin_top), (margin_x + court_w, margin_top + court_h), (40, 40, 40), 2)
    for line_name, y in line_y.items():
        py = to_panel_point((0, y))[1]
        color = (160, 160, 160) if line_name != "BONUS" else (120, 150, 180)
        cv2.line(panel, (margin_x, py), (margin_x + court_w, py), color, 1)
        cv2.putText(
            panel,
            line_name,
            (margin_x + 6, max(margin_top + 14, py - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    node_map = {node["id"]: node for node in scene_graph["nodes"] if node.get("spatial") is not None}
    node_points = {pid: to_panel_point(node["spatial"]) for pid, node in node_map.items()}
    global_context = scene_graph.get("global_context", {})
    recent_events = recent_events or []
    latest_event = recent_events[-1] if recent_events else None
    highlight_nodes = set()
    if latest_event:
        if isinstance(latest_event.get("subject"), int):
            highlight_nodes.add(latest_event["subject"])
        if isinstance(latest_event.get("object"), int):
            highlight_nodes.add(latest_event["object"])

    cv2.rectangle(panel, (12, 32), (width - 12, 58), (230, 230, 230), -1)
    cv2.rectangle(panel, (12, 32), (width - 12, 58), (180, 180, 180), 1)
    info_text = (
        f"Active {len(scene_graph.get('nodes', []))}/{len(scene_graph.get('full_nodes', []))}  "
        f"Pairs {len(scene_graph.get('pair_factors', []))}  "
        f"Lines {len(scene_graph.get('line_factors', []))}  "
        f"Higher {len(scene_graph.get('factor_nodes', []))}"
    )
    cv2.putText(panel, info_text, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55, 55, 55), 1)

    cv2.putText(
        panel,
        f"Containment {global_context.get('best_containment_score', 0.0):.2f}  Contact {global_context.get('best_contact_score', 0.0):.2f}",
        (14, height - 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (40, 40, 40),
        1,
    )
    if latest_event:
        event_label = latest_event["type"].replace("CONFIRMED_", "")
        cv2.putText(panel, f"Recent: {event_label}", (14, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 90, 170), 1)
    else:
        cv2.putText(panel, "Recent: none", (14, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

    for factor in scene_graph.get("pair_factors", []):
        nodes = factor.get("nodes", [])
        if len(nodes) != 2:
            continue
        src, dst = nodes
        if src not in node_points or dst not in node_points:
            continue
        p1 = node_points[src]
        p2 = node_points[dst]
        proximity = max(0.0, 1.0 - factor["features"]["distance"] / 1.2)
        strength = 0.5 * proximity + 0.5 * factor["features"]["approach_score"]
        if factor.get("type") in {"RAIDER_DEFENDER_PAIR", "DEFENDER_RAIDER_PAIR"}:
            color = (40, int(120 + 120 * strength), 230)
        else:
            color = (140, 140, 140)
        thickness = 1 + int(2 * min(1.0, strength))
        cv2.line(panel, p1, p2, color, thickness)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        label = f"{factor['features']['distance']:.2f}m"
        cv2.putText(panel, label, (mid_x + 4, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    for factor in scene_graph.get("factor_nodes", []):
        triplet = factor.get("triplet", ())
        participant_ids = [pid for pid in triplet if isinstance(pid, int)]
        if not participant_ids or any(pid not in node_map for pid in participant_ids):
            continue
        centroid = np.mean([np.array(node_map[pid]["spatial"]) for pid in participant_ids], axis=0)
        fx, fy = to_panel_point(centroid)
        diamond = np.array([
            [fx, fy - 7],
            [fx + 7, fy],
            [fx, fy + 7],
            [fx - 7, fy],
        ], dtype=np.int32)
        factor_color = (0, 140, 255) if factor.get("type") == "THIRD_ORDER_PRESSURE" else (80, 90, 200)
        cv2.polylines(panel, [diamond], isClosed=True, color=factor_color, thickness=2)
        for pid in participant_ids:
            cv2.line(panel, (fx, fy), node_points[pid], (150, 190, 230), 1)
        factor_label = "P" if factor.get("type") == "THIRD_ORDER_PRESSURE" else "C"
        cv2.putText(panel, factor_label, (fx + 9, fy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, factor_color, 1)

    for factor in scene_graph.get("line_factors", []):
        nodes = factor.get("nodes", [])
        if not nodes:
            continue
        subject_id = nodes[0]
        line_name = factor.get("line")
        if subject_id not in node_points or line_name not in line_y:
            continue
        px, _ = node_points[subject_id]
        py = to_panel_point((0, line_y[line_name]))[1]
        color = (0, 170, 0) if factor["features"].get("active") else (110, 170, 110)
        cv2.line(panel, node_points[subject_id], (px, py), color, 1)
        cv2.putText(panel, line_name[0], (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

    for node in scene_graph["nodes"]:
        pos = node.get("spatial")
        if pos is None:
            continue
        px, py = node_points[node["id"]]
        is_raider = node.get("role") == "RAIDER"
        track_conf = float(node.get("track_confidence", 0.0))
        radius = 10 if is_raider else 8
        fill = (0, 70, 220) if is_raider else (210, 80, 50)
        outline = (0, 255, 255) if node["id"] in highlight_nodes else (0, 0, 0)
        cv2.circle(panel, (px, py), radius, fill, -1)
        cv2.circle(panel, (px, py), radius + 2, outline, 2 if node["id"] in highlight_nodes else 1)
        conf_bar = int(18 * max(0.0, min(1.0, track_conf)))
        cv2.rectangle(panel, (px - 9, py + radius + 4), (px - 9 + conf_bar, py + radius + 8), (0, 180, 0), -1)
        cv2.rectangle(panel, (px - 9, py + radius + 4), (px + 9, py + radius + 8), (50, 50, 50), 1)
        label = f"R{node['id']}" if is_raider else f"D{node['id']}"
        cv2.putText(panel, label, (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

    legend_y = height - 10
    cv2.putText(panel, "Blue: raider pair  Gray: defender pair", (12, legend_y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
    cv2.putText(panel, "Green: line factor  P/C: pressure/containment", (12, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    return panel
