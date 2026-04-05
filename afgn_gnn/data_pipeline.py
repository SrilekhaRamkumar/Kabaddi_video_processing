import torch
import numpy as np
from torch_geometric.data import Data, Batch

class KabaddiGraphBuilder:
    """
    Enhanced data pipeline with richer features:
    - 10-dim node features (role, pos, velocity, visibility, distance-to-baulk, distance-to-bonus)
    - 6-dim edge features (distance, rel_vel, approach, adjacency, angle, closing_speed)
    - 5-dim global features (defenders_ratio, containment, contact_score, raider_depth, width_span)
    """
    def __init__(self, use_device=torch.device("cpu")):
        self.device = use_device
        self.prev_positions = {}  # Track positions for velocity computation
    
    def process_sequence(self, scene_graph_seq, raider_id):
        unique_defenders = set()
        for sg in scene_graph_seq:
            for node in sg.get("full_nodes", sg.get("nodes", [])):
                if node["id"] != raider_id and node.get("spatial") is not None:
                    unique_defenders.add(node["id"])
                    
        defender_list = sorted(list(unique_defenders))
        id_to_idx = {raider_id: 0}
        for i, d_id in enumerate(defender_list):
            if i + 1 < 8:
                id_to_idx[d_id] = i + 1
        
        # Track positions across frames for velocity estimation
        position_history = {}
                
        pyg_graphs = []
        for t, sg in enumerate(scene_graph_seq):
            pyg_graphs.append(self._to_pyg_data(sg, raider_id, id_to_idx, position_history, t))
            
        return pyg_graphs
        
    def _to_pyg_data(self, scene_graph, raider_id, id_to_idx, position_history, timestep):
        nodes = scene_graph.get("full_nodes", scene_graph.get("nodes", []))
        
        node_features = []
        node_indices = []
        id_to_graph_idx = {}
        
        # Global context
        global_ctx = scene_graph.get("global_context", {})
        n_defenders = float(len([n for n in nodes if n["id"] != raider_id and n.get("spatial") is not None]))
        containment = float(global_ctx.get("best_containment_score", 0.0))
        contact_score = float(global_ctx.get("best_contact_score", 0.0))
        
        # Find raider position for relative features
        raider_pos = None
        for n in nodes:
            if n["id"] == raider_id:
                raider_pos = np.array(n.get("spatial", [5.0, 3.0]))
                break
        if raider_pos is None:
            raider_pos = np.array([5.0, 3.0])
        
        raider_depth = raider_pos[1] / 6.5  # Normalized distance into defender half
        
        # Compute defender width span
        def_positions = []
        for n in nodes:
            if n["id"] != raider_id and n.get("spatial") is not None:
                def_positions.append(n["spatial"][0])
        width_span = (max(def_positions) - min(def_positions)) / 10.0 if len(def_positions) >= 2 else 0.0
        
        # 5-dim global: [defenders_ratio, containment, contact_score, raider_depth, width_span]
        u = torch.tensor([[n_defenders / 7.0, containment, contact_score, raider_depth, width_span]], dtype=torch.float)
        
        BAULK_Y = 3.75
        BONUS_Y = 4.75
        
        for graph_idx, node in enumerate(nodes):
            nid = node["id"]
            if nid not in id_to_idx:
                continue
                
            spatial = np.array(node.get("spatial", [0.0, 0.0]))
            
            # Compute velocity from position history
            if nid in position_history and timestep > 0:
                prev_pos = position_history[nid]
                velocity = spatial - prev_pos
            else:
                motion = node.get("motion", [0.0, 0.0])
                if motion is None or (isinstance(motion, np.ndarray) and motion.size == 0):
                    velocity = np.array([0.0, 0.0])
                elif isinstance(motion, (int, float)):
                    velocity = np.array([float(motion), 0.0])
                else:
                    velocity = np.array(motion[:2]) if len(motion) >= 2 else np.array([0.0, 0.0])
            
            position_history[nid] = spatial.copy()
            
            visibility = float(node.get("track_confidence", 1.0))
            is_raider = 1.0 if nid == raider_id else 0.0
            
            # Distance to court lines (normalized)
            dist_to_baulk = abs(spatial[1] - BAULK_Y) / 6.5
            dist_to_bonus = abs(spatial[1] - BONUS_Y) / 6.5
            
            # Speed magnitude
            speed = float(np.linalg.norm(velocity))
            
            # 10-dim: [is_raider, x, y, vx, vy, visibility, dist_baulk, dist_bonus, speed, dist_to_raider]
            dist_to_raider = float(np.linalg.norm(spatial - raider_pos)) / 10.0 if not is_raider else 0.0
            
            feat = [
                is_raider,
                spatial[0] / 10.0, spatial[1] / 6.5,   # Normalized positions
                velocity[0] / 3.0, velocity[1] / 3.0,   # Normalized velocity
                visibility,
                dist_to_baulk,
                dist_to_bonus,
                speed / 3.0,
                dist_to_raider
            ]
            node_features.append(feat)
            node_indices.append(id_to_idx[nid])
            id_to_graph_idx[nid] = graph_idx
            
        x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.zeros((1, 10), dtype=torch.float)
        node_idx_tensor = torch.tensor(node_indices, dtype=torch.long) if node_indices else torch.zeros(1, dtype=torch.long)
        
        # Build Edges with richer features
        edge_indices = []
        edge_attrs = []
        
        pair_factors = scene_graph.get("full_pair_factors", scene_graph.get("pair_factors", []))
        for factor in pair_factors:
            subj, obj = factor.get("nodes", (None, None))
            if subj not in id_to_graph_idx or obj not in id_to_graph_idx:
                continue
                
            features = factor.get("features", {})
            dist = features.get("distance", 2.0)
            rel_vel = features.get("relative_velocity", 0.0)
            approach = features.get("approach_score", 0.0)
            adj = features.get("adjacency", 0.0)
            
            # New: Compute angle between players (relative to court orientation)
            s_node = next((n for n in nodes if n["id"] == subj), None)
            o_node = next((n for n in nodes if n["id"] == obj), None)
            angle = 0.0
            closing_speed = 0.0
            if s_node and o_node:
                s_pos = np.array(s_node.get("spatial", [0, 0]))
                o_pos = np.array(o_node.get("spatial", [0, 0]))
                diff = o_pos - s_pos
                angle = float(np.arctan2(diff[1], diff[0]) / np.pi)  # Normalized to [-1, 1]
                # Closing speed proxy
                closing_speed = max(0.0, approach * rel_vel) / 3.0
            
            # 6-dim edge: [dist, rel_vel, approach, adjacency, angle, closing_speed]
            e_feat = [dist / 5.0, rel_vel / 3.0, approach, adj, angle, closing_speed]
            
            idx_s = id_to_graph_idx[subj]
            idx_o = id_to_graph_idx[obj]
            
            edge_indices.append([idx_s, idx_o])
            edge_attrs.append(e_feat)
            edge_indices.append([idx_o, idx_s])
            edge_attrs.append(e_feat)
            
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)
            
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
        data.node_idx = node_idx_tensor
        return data

def collate_temporal_graphs(graph_seq_list):
    batch_size = len(graph_seq_list)
    T = len(graph_seq_list[0])
    
    batched_seq = []
    for t in range(T):
        graphs_at_t = [graph_seq_list[b][t] for b in range(batch_size)]
        batched = Batch.from_data_list(graphs_at_t)
        batched_seq.append(batched)
    
    return batched_seq
