import json
import os
from pathlib import Path

class SceneGraphRecorder:
    """
    Hooks into the live tracking pipeline to save real scene_graphs 
    for training the GNN offline.
    """
    def __init__(self, output_dir="afgn_gnn/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_raid_id = None
        self.current_sequence = []
        self.raid_counter = 0
        
    def start_new_raid(self, raid_id=None):
        """Called when a new raid begins."""
        self._save_current_sequence()
        self.current_raid_id = raid_id if raid_id else f"raid_{self.raid_counter:04d}"
        self.current_sequence = []
        self.raid_counter += 1
        
    def record_frame(self, scene_graph, raider_id, frame_idx):
        """Called every frame right before the old AFGN reasoning runs."""
        # Clean up any non-serializable objects if necessary
        clean_graph = self._clean_for_json(scene_graph)
        
        frame_data = {
            "frame_idx": frame_idx,
            "raider_id": raider_id,
            "scene_graph": clean_graph
        }
        self.current_sequence.append(frame_data)
        
    def end_raid(self, labels=None):
        """
        Called when the raid ends.
        labels: Option dict defining if it was a tackle, return, touch, etc.
        """
        self._save_current_sequence(labels)
        self.current_sequence = []

    def _save_current_sequence(self, labels=None):
        if not self.current_sequence:
            return
            
        file_path = self.output_dir / f"{self.current_raid_id}.json"
        
        payload = {
            "raid_id": self.current_raid_id,
            "frames": self.current_sequence,
            "labels": labels if labels else {}
        }
        
        with open(file_path, 'w') as f:
            json.dump(payload, f, indent=2)
            
        print(f"[DatasetRecorder] Saved {len(self.current_sequence)} frames to {file_path}")

    def _clean_for_json(self, obj):
        """Ensures numpy arrays and float32s are JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self._clean_for_json(v) for v in obj]
        elif hasattr(obj, "tolist"): # Numpy arrays
            return obj.tolist()
        elif hasattr(obj, "item"): # Numpy scalars
            return obj.item()
        return obj
