"""
Interaction Graph Module
Handles interaction proposal encoding and factor graph construction.
"""
import numpy as np
from itertools import combinations


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

    def build_graph(self, proposals, gallery, raider_id):
        influence = {pid: 0.0 for pid in gallery if gallery[pid]["age"] == 0}
        pair_proposals = {}
        line_proposals = {}

        for proposal in proposals:
            if proposal["type"] == "HHI":
                dist_weight = 1.0 / (proposal["features"]["dist"] + 1e-6)
                vel_weight = proposal["features"]["rel_vel"]
                emb = np.array(proposal["features"]["emb"])
                half = len(emb) // 2
                feat_sim = np.dot(emb[:half], emb[half:]) / (
                    np.linalg.norm(emb[:half]) * np.linalg.norm(emb[half:]) + 1e-6
                )
                weight = dist_weight * vel_weight * feat_sim

                if proposal["S"] in influence:
                    influence[proposal["S"]] += weight
                if proposal["O"] in influence:
                    influence[proposal["O"]] += weight

                pair_key = (proposal["S"], proposal["O"])
                stored = pair_proposals.get(pair_key)
                if stored is None or proposal["features"]["dist"] < stored["features"]["dist"]:
                    pair_proposals[pair_key] = proposal
            elif proposal["type"] == "HLI":
                line_key = (proposal["S"], proposal["O"])
                stored = line_proposals.get(line_key)
                if stored is None or proposal["features"]["dist"] < stored["features"]["dist"]:
                    line_proposals[line_key] = proposal

        sorted_ids = sorted(influence.keys(), key=lambda pid: influence[pid], reverse=True)
        self.active_nodes = [pid for pid in sorted_ids if pid != raider_id][: self.top_k - 1]
        self.active_nodes.append(raider_id)

        n = len(self.active_nodes)
        self.adjacency_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                id_i, id_j = self.active_nodes[i], self.active_nodes[j]
                bb_i, bb_j = gallery[id_i]["last_bbox"], gallery[id_j]["last_bbox"]
                d_i = np.sqrt((bb_i[2] - bb_i[0]) ** 2 + (bb_i[3] - bb_i[1]) ** 2)
                d_j = np.sqrt((bb_j[2] - bb_j[0]) ** 2 + (bb_j[3] - bb_j[1]) ** 2)
                self.adjacency_matrix[i, j] = min(d_i, d_j) / max(d_i, d_j)

        self.pair_factors = []
        for subject_id, object_id in pair_proposals:
            if subject_id not in self.active_nodes or object_id not in self.active_nodes:
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
            self.pair_factors.append({
                "type": "RAIDER_DEFENDER_PAIR",
                "nodes": [subject_id, object_id],
                "features": {
                    "distance": float(proposal["features"]["dist"]),
                    "relative_velocity": float(proposal["features"]["rel_vel"]),
                    "approach_score": float(min(1.0, approach_score / 2.0)),
                    "adjacency": float(self.adjacency_matrix[self.active_nodes.index(subject_id), self.active_nodes.index(object_id)]),
                    "track_confidence": float(
                        0.5 * (
                            gallery[subject_id].get("track_confidence", 0.0)
                            + gallery[object_id].get("track_confidence", 0.0)
                        )
                    ),
                },
            })

        self.line_factors = []
        for subject_id, line_name in line_proposals:
            if subject_id not in self.active_nodes:
                continue
            proposal = line_proposals[(subject_id, line_name)]
            self.line_factors.append({
                "type": "PLAYER_LINE_FACTOR",
                "nodes": [subject_id],
                "line": line_name,
                "features": {
                    "distance": float(proposal["features"]["dist"]),
                    "active": bool(proposal["features"]["active"]),
                    "track_confidence": float(gallery[subject_id].get("track_confidence", 0.0)),
                },
            })

        self.factor_nodes = []
        for triplet in combinations(self.active_nodes, 3):
            pid1, pid2, pid3 = triplet
            pos1 = gallery[pid1]["display_pos"]
            pos2 = gallery[pid2]["display_pos"]
            pos3 = gallery[pid3]["display_pos"]
            if pos1 is None or pos2 is None or pos3 is None:
                continue

            pos1 = np.array(pos1)
            pos2 = np.array(pos2)
            pos3 = np.array(pos3)
            feat1 = np.array(gallery[pid1]["feat"])
            feat2 = np.array(gallery[pid2]["feat"])
            feat3 = np.array(gallery[pid3]["feat"])
            centroid = np.mean([pos1, pos2, pos3], axis=0)
            spread = np.mean([
                np.linalg.norm(pos1 - centroid),
                np.linalg.norm(pos2 - centroid),
                np.linalg.norm(pos3 - centroid),
            ])
            self.factor_nodes.append({
                "triplet": triplet,
                "type": "THIRD_ORDER_PRESSURE",
                "features": {
                    "distances": [
                        np.linalg.norm(pos1 - pos2),
                        np.linalg.norm(pos1 - pos3),
                        np.linalg.norm(pos2 - pos3),
                    ],
                    "spread": float(spread),
                    "embeddings": np.mean([feat1, feat2, feat3], axis=0).tolist(),
                },
            })

        return self.package_features(gallery, raider_id)

    def package_features(self, gallery, raider_id):
        graph_data = {
            "nodes": [],
            "edges": self.adjacency_matrix.tolist(),
            "factor_nodes": self.factor_nodes,
            "pair_factors": self.pair_factors,
            "line_factors": self.line_factors,
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
        return graph_data

