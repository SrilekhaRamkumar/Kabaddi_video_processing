import torch
import numpy as np
from collections import deque
from afgn_gnn.model import KabaddiAFGN
from afgn_gnn.data_pipeline import KabaddiGraphBuilder, collate_temporal_graphs

class AFGNEngineInference:
    """
    Replaces the rule-based KabaddiAFGNEngine with a learnable Neural Engine.
    Operates smoothly over temporal windows inside a single raid.
    """
    def __init__(self, model_path, window_size=5, device=torch.device('cpu')):
        self.device = device
        # Use the upgraded architecture dimensions!
        self.model = KabaddiAFGN(hidden_dim=128).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.builder = KabaddiGraphBuilder(self.device)
        self.window_size = window_size
        self.scene_graph_buffer = deque(maxlen=window_size)
        
        # Temporal smoothing buffers for outputs
        self.prob_buffer = {
            "contact": deque(maxlen=3), # Max pooling over last 3 frames
            "tackle": deque(maxlen=5),  # Moving avg over 5 frames
            "return": deque(maxlen=3),
            "bonus": deque(maxlen=3),
            "raid_end": deque(maxlen=2)
        }
        
    def _temporal_smooth(self, key, current_prob, method="max"):
        self.prob_buffer[key].append(current_prob)
        if method == "max":
            try:
                # Handle dictionary of node contacts or single float
                if isinstance(current_prob, dict):
                    smoothed = {}
                    for k in current_prob.keys():
                        smoothed[k] = max([b.get(k, 0.0) for b in self.prob_buffer[key] if isinstance(b, dict)])
                    return smoothed
                return max(self.prob_buffer[key])
            except Exception:
                return current_prob
        elif method == "avg":
            if isinstance(current_prob, dict):
                return current_prob # Simplified due to changing nodes
            return sum(self.prob_buffer[key]) / len(self.prob_buffer[key])

    @torch.no_grad()
    def process_frame(self, scene_graph, raider_id):
        """
        Takes the current frame's scene graph and returns event probabilities.
        """
        self.scene_graph_buffer.append(scene_graph)
        
        if len(self.scene_graph_buffer) < 2:
            return self._empty_result()
            
        pyg_seq = self.builder.process_sequence(list(self.scene_graph_buffer), raider_id)
        
        # We need a batch axis: List[Data] -> List[Batch of 1 element]
        batch_seq = [collate_temporal_graphs([[g]])[0].to(self.device) for g in pyg_seq]
        
        outputs = self.model(batch_seq)
        
        # Map node contact probs back to IDs
        # The data builder mapped raider to 0, and ordered defenders.
        # We can extract the node idx map from the pyg graph node_idx and node ids from scene_graph
        # For inference, just rely on the latest frame's defenders.
        latest_sg = self.scene_graph_buffer[-1]
        nodes = latest_sg.get("full_nodes", latest_sg.get("nodes", []))
        active_defenders = [n["id"] for n in nodes if n["id"] != raider_id and n.get("spatial") is not None]
        
        # Hacky retrieval of mapping based on sorting IDs (matches Data builder logic)
        sorted_defs = sorted(list(set(active_defenders)))
        contact_probs = {}
        for i, d_id in enumerate(sorted_defs):
            if i + 1 < 8:
                prob = outputs["p_contact"][0, i + 1].item()
                contact_probs[d_id] = prob
                
        # Raw probabilities
        p_tackle = outputs["p_tackle"][0].item()
        p_return = outputs["p_return"][0].item()
        p_bonus = outputs["p_bonus"][0].item()
        p_raid_end = outputs["p_raid_end"][0].item()
        
        # Apply Temporal Smoothing
        smooth_contact = self._temporal_smooth("contact", contact_probs, method="max")
        smooth_tackle = self._temporal_smooth("tackle", p_tackle, method="avg")
        smooth_return = self._temporal_smooth("return", p_return, method="max")
        smooth_bonus = self._temporal_smooth("bonus", p_bonus, method="max")
        smooth_raid_end = self._temporal_smooth("raid_end", p_raid_end, method="max")
        
        return self._apply_thresholds({
            "contact": smooth_contact,
            "tackle": smooth_tackle,
            "return": smooth_return,
            "bonus": smooth_bonus,
            "raid_end": smooth_raid_end
        })

    def _apply_thresholds(self, probs):
        """
        Thresholds are ONLY applied at the very end of inference for game logic integration.
        Model outputs pure probabilities.
        """
        actions = []
        
        # Evaluate nodes over threshold
        for d_id, prob in probs["contact"].items():
            if prob > 0.65:
                actions.append({
                    "type": "RAIDER_DEFENDER_CONTACT",
                    "description": f"Contact with Defender {d_id}",
                    "points": 1,
                    "confidence": prob,
                    "defender_id": d_id
                })
                
        if probs["tackle"] > 0.70:
            actions.append({
                "type": "DEFENDER_TACKLE",
                "description": "Defender tackle",
                "points": 1,
                "confidence": probs["tackle"]
            })
            
        if probs["return"] > 0.80:
            actions.append({
                "type": "RAIDER_RETURNED_MIDDLE",
                "description": "Returned to midline",
                "points": 0,
                "confidence": probs["return"]
            })
            
        if probs["bonus"] > 0.75:
            actions.append({
                "type": "RAIDER_BONUS_TOUCH",
                "description": "Bonus touch attempted",
                "points": 1,
                "confidence": probs["bonus"]
            })

        return {
            "raw_probabilities": probs,
            "emitted_events": actions,
            "raid_ended": probs["raid_end"] > 0.85
        }

    def _empty_result(self):
        return {
            "raw_probabilities": {"contact": {}, "tackle": 0.0, "return": 0.0, "bonus": 0.0, "raid_end": 0.0},
            "emitted_events": [],
            "raid_ended": False
        }
