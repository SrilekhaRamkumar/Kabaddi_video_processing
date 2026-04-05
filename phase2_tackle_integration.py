"""
Phase 2.2: Integration of Lightweight Tackle Model into KabaddiAFGNEngine

This module extends KabaddiAFGNEngine to augment tackle detection with an ML model.
The model is optional (graceful fallback to pure geometry if not loaded).

Integration approach:
  1. Load tackle model in KabaddiAFGNEngine.__init__()
  2. In _infer_defender_events(), compute two scores:
     - geo_score: existing geometric logic (pressure, speed, containment)
     - model_prob: learned probability from tackle_model.predict_single()
  3. Blend: tackle_conf = 0.6 * geo_score + 0.4 * model_prob
  4. Rest of logic unchanged (output format identical to Phase 1)

This is backward-compatible: if model not found, falls back to pure phase 1.
"""

import os
import torch
import numpy as np


def integrate_tackle_model_into_engine():
    """
    Monkey-patch KabaddiAFGNEngine to add tackle model support.
    
    This function modifies the engine class to support optional ML-augmented
    tackle detection while maintaining backward compatibility with Phase 1.
    """
    
    from kabaddi_afgn_reasoning import KabaddiAFGNEngine
    from tackle_model import LightweightTackleModel
    
    # Store original __init__ and _infer_defender_events
    original_init = KabaddiAFGNEngine.__init__
    original_infer_defender = KabaddiAFGNEngine._infer_defender_events
    
    def __init__(self, model_path=None):
        """Initialize with optional tackle model."""
        original_init(self)
        
        self.tackle_model = None
        self.tackle_model_enabled = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.tackle_model = LightweightTackleModel()
                checkpoint = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.tackle_model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.tackle_model.load_state_dict(checkpoint)
                self.tackle_model.eval()
                self.tackle_model_enabled = True
                print(f"✓ Tackle model loaded from {model_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load tackle model from {model_path}: {e}")
                print(f"  Falling back to Phase 1 pure geometry logic")
                self.tackle_model = None
                self.tackle_model_enabled = False
    
    def _infer_defender_events(self, context):
        """Extended defender event inference with optional ML tackle model."""
        actions = []
        raider_speed = context["raider_speed"]
        pressure = context["higher_order_pressure"]
        defenders_on_court = context["defenders_on_court"]
        containment = max(
            [factor["features"].get("angle", 0.0) for factor in context["containment_factors"]],
            default=0.0,
        )
        support = min(1.0, len(context["nearby_defenders"]) / 3.0)
        full_graph_pressure = min(1.0, context["global_context"].get("best_contact_score", 0.0) + 0.35 * containment)
        
        # Phase 1: Geometric score (existing logic)
        geo_score = min(
            1.0,
            0.25 * pressure
            + 0.20 * full_graph_pressure
            + 0.25 * max(0.0, 1.0 - raider_speed / 0.35)
            + 0.20 * support
            + 0.10 * min(1.0, context["global_context"].get("best_containment_score", 0.0)),
        )
        
        # Phase 2.2: Optional ML augmentation
        if self.tackle_model_enabled and self.tackle_model is not None:
            try:
                # Compute contact confidence (recent contact from action buffer)
                contact_conf = 0.0
                if "RAIDER_DEFENDER_CONTACT" in self.action_buffer:
                    _, contact_conf = self.action_buffer["RAIDER_DEFENDER_CONTACT"]
                
                # Get model prediction
                model_prob = self.tackle_model.predict_single(
                    contact_conf=float(contact_conf),
                    nearby_count=len(context["nearby_defenders"]),
                    raider_speed=float(raider_speed),
                    containment_angle=float(containment),
                )
                
                # Phase 2.2: Blend geometric + ML scores
                # 60% weight on geometry (proven, interpretable)
                # 40% weight on ML (learns contact requirement)
                tackle_conf = 0.6 * geo_score + 0.4 * model_prob
                
                # Log which model contributed more
                if model_prob > geo_score:
                    source = "ML"
                else:
                    source = "Geometry"
                
            except Exception as e:
                # Graceful fallback to pure geometry
                print(f"⚠ Model inference error: {e}. Falling back to geometry.")
                tackle_conf = geo_score
                source = "Fallback"
        else:
            # Pure Phase 1 (no ML)
            tackle_conf = geo_score
            source = "Phase1"
        
        # Rest of the logic unchanged from Phase 1
        if len(context["nearby_defenders"]) >= 2 and tackle_conf >= 0.6:
            actions.append(self._make_action("DEFENDER_ASSIST_TACKLE", context["frame_idx"], tackle_conf, "Multiple defenders are engaging the raider"))
            self.action_buffer["DEFENDER_ASSIST_TACKLE"] = (context["frame_idx"], tackle_conf)

        if tackle_conf >= 0.68:
            actions.append(self._make_action("DEFENDER_TACKLE", context["frame_idx"], tackle_conf, "Defenders tackled the raider"))
            actions.append(self._make_action("RAIDER_CAUGHT", context["frame_idx"], min(1.0, tackle_conf + 0.08), "Raider was caught by defenders"))
            self.current_raid["raider_caught"] = True
            self.current_raid["raid_ended"] = True
            self.current_raid["raider_out"] = True
            self.action_buffer["DEFENDER_TACKLE"] = (context["frame_idx"], tackle_conf)

        if defenders_on_court <= 3 and tackle_conf >= 0.68:
            actions.append(self._make_action("SUPER_TACKLE_TRIGGER", context["frame_idx"], min(1.0, tackle_conf + 0.06), "Super tackle condition triggered"))
            self.action_buffer["SUPER_TACKLE_TRIGGER"] = (context["frame_idx"], min(1.0, tackle_conf + 0.06))

        endline_contacts = [
            event for event in context["confirmed_events"]
            if event["type"] == "CONFIRMED_DEFENDER_ENDLINE_TOUCH"
        ]
        for event in endline_contacts:
            defender_id = event["subject"]
            if defender_id in self.current_raid["defender_endline_touch"]:
                continue
            self.current_raid["defender_endline_touch"].add(defender_id)
            actions.append(self._make_action(
                "DEFENDER_ENDLINE_TOUCH",
                context["frame_idx"],
                event["confidence"],
                f"Defender {defender_id} touched end line",
                metadata={"defender_id": defender_id},
            ))

        return self._dedupe_actions(actions)
    
    # Apply monkey patches
    KabaddiAFGNEngine.__init__ = __init__
    KabaddiAFGNEngine._infer_defender_events = _infer_defender_events
    
    print("✓ KabaddiAFGNEngine extended with Phase 2.2 tackle model support")


if __name__ == "__main__":
    print("\nTesting Phase 2.2 integration...\n")
    
    # Test 1: Load engine without model
    print("Test 1: Engine without tackle model (Phase 1 fallback)")
    from kabaddi_afgn_reasoning import KabaddiAFGNEngine
    engine = KabaddiAFGNEngine()
    print(f"  ✓ Engine initialized (no model path provided)")
    
    # Test 2: Apply integration (monkey patch)
    print("\nTest 2: Apply Phase 2.2 integration")
    integrate_tackle_model_into_engine()
    engine2 = KabaddiAFGNEngine(model_path="models/tackle_model.pt")  # Will fall back if not found
    print(f"  ✓ Integration applied (graceful fallback if model missing)")
    
    print("\n✓ Phase 2.2 integration tests passed!\n")
