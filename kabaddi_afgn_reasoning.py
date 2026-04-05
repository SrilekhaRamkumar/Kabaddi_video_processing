import numpy as np


class KabaddiAFGNEngine:
    def __init__(self):
        self.total_points = {"attacker": 0, "defender": 0}
        self.empty_raid_count = 0
        self.do_or_die_next = False
        self.touched_defenders = set()
        self.confirmed_action_history = []
        self.current_raid = self._new_raid_state()
        self.accuracy_stats = {
            "total_actions": 0,
            "high_confidence_actions": 0,
            "factor_consistency": 0.0,
        }
        # Phase 1.5: Temporal hysteresis buffer for action denoising
        self.action_buffer = {}  # {action_type: (frame_idx, confidence)}
        self.current_frame_idx = 0

    def _new_raid_state(self):
        return {
            "raid_active": False,
            "entered_court": False,
            "baulk_crossed": False,
            "bonus_crossed": False,
            "bonus_touch": False,
            "touch_occurred": False,
            "returned_middle": False,
            "raider_caught": False,
            "raid_ended": False,
            "raider_out": False,
            "lobby_activated": False,
            "raider_illegal_lobby": False,
            "defender_endline_touch": set(),
            "touched_defenders": set(),
            "revivals_pending": 0,
            "defender_points_awarded": False,
            "attacker_points_awarded": False,
            "bonus_points_awarded": False,
            "all_out_awarded": False,
            "empty_raid_recorded": False,
            "step_out_recorded": False,
            "super_tackle_awarded": False,
            "super_raid_awarded": False,
            "emitted_action_keys": set(),
        }

    def process_frame_actions(self, scene_graph, proposals, confirmed_events, raider_id, frame_idx, gallery):
        self.current_frame_idx = frame_idx  # Track current frame for hysteresis
        if raider_id is None or not scene_graph.get("nodes"):
            return self._empty_result()

        node_map = {node["id"]: node for node in scene_graph["nodes"]}
        raider_node = node_map.get(raider_id)
        if raider_node is None or raider_node["spatial"] is None:
            return self._empty_result()

        inferred_actions = []
        factor_scores = []

        context = self._build_context(scene_graph, proposals, confirmed_events, raider_id, frame_idx, gallery, node_map, raider_node)
        inferred_actions.extend(self._infer_raid_progress_events(context))
        inferred_actions.extend(self._infer_contact_events(context))
        inferred_actions.extend(self._infer_defender_events(context))
        inferred_actions.extend(self._infer_boundary_events(context))

        scored_actions = self._apply_rules(inferred_actions, context)
        self._update_accuracy(scored_actions, factor_scores + [a["confidence"] for a in inferred_actions])

        return {
            "actions": scored_actions,
            "points_scored": sum(action.get("points", 0) for action in scored_actions),
            "total_points": dict(self.total_points),
            "raid_ended": self.current_raid["raid_ended"],
            "confidence_scores": [action.get("confidence", 0.0) for action in scored_actions],
            "accuracy_metrics": self.get_accuracy_metrics(),
        }

    def _empty_result(self):
        return {
            "actions": [],
            "points_scored": 0,
            "total_points": dict(self.total_points),
            "raid_ended": self.current_raid["raid_ended"],
            "confidence_scores": [],
            "accuracy_metrics": self.get_accuracy_metrics(),
        }

    def _build_context(self, scene_graph, proposals, confirmed_events, raider_id, frame_idx, gallery, node_map, raider_node):
        raider_pos = np.array(raider_node["spatial"])
        raider_speed = np.linalg.norm(np.array(raider_node["motion"]))
        active_defenders = [node for node in scene_graph["nodes"] if node["id"] != raider_id and node["spatial"] is not None]
        full_nodes = scene_graph.get("full_nodes", scene_graph.get("nodes", []))
        defenders = [node for node in full_nodes if node["id"] != raider_id and node["spatial"] is not None]
        pair_factors = scene_graph.get("pair_factors", [])
        line_factors = scene_graph.get("line_factors", [])
        third_order_factors = scene_graph.get("factor_nodes", [])
        full_pair_factors = scene_graph.get("full_pair_factors", pair_factors)
        full_line_factors = scene_graph.get("full_line_factors", line_factors)
        full_factor_nodes = scene_graph.get("full_factor_nodes", third_order_factors)
        global_context = scene_graph.get("global_context", {})

        pairwise_contacts = []
        for pair_factor in full_pair_factors:
            if pair_factor["type"] not in {"RAIDER_DEFENDER_PAIR", "DEFENDER_RAIDER_PAIR"}:
                continue
            subject_id, object_id = pair_factor["nodes"]
            defender_id = object_id if subject_id == raider_id else subject_id
            if raider_id not in (subject_id, object_id):
                continue
            defender_node = next((node for node in defenders if node["id"] == defender_id), None)
            if defender_node is None or defender_node["spatial"] is None:
                continue
            pair_score = self._pair_factor_score(pair_factor)
            pairwise_contacts.append((defender_id, pair_factor, pair_score))

        if not pairwise_contacts:
            for proposal in proposals:
                if proposal["type"] == "HHI" and proposal["S"] == raider_id:
                    defender_node = node_map.get(proposal["O"])
                    if defender_node is None or defender_node["spatial"] is None:
                        continue
                    pair_score = self._pairwise_contact_score(raider_node, defender_node, proposal)
                    pairwise_contacts.append((proposal["O"], proposal, pair_score))

        line_factor_map = {}
        for line_factor in full_line_factors:
            if line_factor["nodes"]:
                line_factor_map[(line_factor["nodes"][0], line_factor["line"])] = line_factor

        nearby_defenders = [
            node["id"]
            for node in defenders
            if np.linalg.norm(np.array(node["spatial"]) - raider_pos) < 1.1
        ]
        containment_factors = [
            factor for factor in full_factor_nodes
            if factor.get("type") == "DEFENDER_CONTAINMENT" and raider_id in factor.get("triplet", ())
        ]
        line_triplet_factors = [
            factor for factor in full_factor_nodes
            if factor.get("type") == "RAIDER_DEFENDER_LINE" and factor.get("triplet", (None,))[0] == raider_id
        ]

        return {
            "frame_idx": frame_idx,
            "raider_id": raider_id,
            "raider_node": raider_node,
            "raider_pos": raider_pos,
            "raider_speed": raider_speed,
            "defenders": defenders,
            "active_defenders": active_defenders,
            "gallery": gallery,
            "confirmed_events": confirmed_events,
            "pairwise_contacts": pairwise_contacts,
            "nearby_defenders": nearby_defenders,
            "defenders_on_court": max(0, len(defenders)),
            "higher_order_pressure": max(
                self._higher_order_pressure(raider_pos, defenders),
                self._graph_pressure_score(full_factor_nodes, raider_id),
            ),
            "line_factor_map": line_factor_map,
            "containment_factors": containment_factors,
            "line_triplet_factors": line_triplet_factors,
            "global_context": global_context,
            "full_pair_factors": full_pair_factors,
            "full_factor_nodes": full_factor_nodes,
        }

    def _infer_raid_progress_events(self, context):
        actions = []
        _, ry = context["raider_pos"]

        entered_conf = self._unary_line_score(ry, 0.0, inverse=False, scale=0.8)
        actions.append(self._make_action("RAIDER_ENTERED_COURT", context["frame_idx"], entered_conf, "Raider entered defending half"))
        self.current_raid["entered_court"] = True
        self.current_raid["raid_active"] = True

        baulk_factor = context["line_factor_map"].get((context["raider_id"], "BAULK"))
        if ry >= 3.55:
            conf = self._line_touch_confidence(context["confirmed_events"], "CONFIRMED_RAIDER_BAULK_TOUCH", fallback=0.65 + min(0.25, (ry - 3.55) * 0.25))
            conf = max(conf, self._line_factor_score(baulk_factor, fallback=0.0))
            actions.append(self._make_action("RAIDER_CROSSED_BAULK_LINE", context["frame_idx"], conf, "Raider crossed baulk line"))
            self.current_raid["baulk_crossed"] = True

        bonus_factor = context["line_factor_map"].get((context["raider_id"], "BONUS"))
        if ry >= 4.55:
            conf = self._line_factor_score(bonus_factor, fallback=0.62 + min(0.25, (ry - 4.55) * 0.25))
            actions.append(self._make_action("RAIDER_CROSSED_BONUS_LINE", context["frame_idx"], conf, "Raider crossed bonus line"))
            self.current_raid["bonus_crossed"] = True

        bonus_conf = self._line_touch_confidence(context["confirmed_events"], "CONFIRMED_RAIDER_BONUS_TOUCH", fallback=0.0)
        bonus_conf = max(bonus_conf, self._line_factor_score(bonus_factor, fallback=0.0))
        if bonus_conf > 0.0:
            actions.append(self._make_action("RAIDER_BONUS_TOUCH", context["frame_idx"], bonus_conf, "Raider touched bonus line"))
            self.current_raid["bonus_touch"] = True

        if ry < 0.8 and self.current_raid["raid_active"]:
            return_conf = min(1.0, 0.7 + max(0.0, (0.8 - ry)) * 0.3)
            actions.append(self._make_action("RAIDER_RETURNED_MIDDLE", context["frame_idx"], return_conf, "Raider returned to middle"))
            self.current_raid["returned_middle"] = True

        return self._dedupe_actions(actions)

    def _infer_contact_events(self, context):
        actions = []
        confirmed_contact_events = [event for event in context["confirmed_events"] if event["type"] == "CONFIRMED_RAIDER_DEFENDER_CONTACT"]
        for defender_id, proposal, pair_score in context["pairwise_contacts"]:
            temporal_conf = self._temporal_event_confidence(confirmed_contact_events, defender_id)
            contact_conf = 0.65 * pair_score + 0.35 * temporal_conf
            # Phase 1.3: Soften threshold from 0.58 to 0.52 with decay for marginal contacts
            if contact_conf < 0.52:
                continue
            # Apply confidence decay for sub-0.60 contacts (soft rejection)
            if 0.52 <= contact_conf < 0.60:
                contact_conf = contact_conf * (1.0 + 0.2 * (contact_conf - 0.52))

            actions.append(self._make_action(
                "RAIDER_DEFENDER_CONTACT",
                context["frame_idx"],
                contact_conf,
                f"Raider made contact with defender {defender_id}",
                metadata={"defender_id": defender_id},
            ))
            self.current_raid["touch_occurred"] = True
            self.current_raid["lobby_activated"] = True
            self.current_raid["touched_defenders"].add(defender_id)
            self.touched_defenders.add(defender_id)
            # Phase 1.5: Update action buffer for temporal hysteresis
            self.action_buffer["RAIDER_DEFENDER_CONTACT"] = (context["frame_idx"], contact_conf)

        if len(self.current_raid["touched_defenders"]) >= 2:
            multi_conf = min(1.0, 0.62 + 0.1 * len(self.current_raid["touched_defenders"]))
            actions.append(self._make_action(
                "RAIDER_MULTIPLE_DEFENDER_TOUCH",
                context["frame_idx"],
                multi_conf,
                "Raider touched multiple defenders",
                metadata={"touch_count": len(self.current_raid["touched_defenders"])},
            ))

        if self.current_raid["lobby_activated"]:
            lobby_conf = 0.85
            # Phase 1.5: Boost if persistent over recent frames
            if self._has_recent_action("RAIDER_DEFENDER_CONTACT", lookback=2):
                lobby_conf = min(1.0, lobby_conf + 0.08)
            actions.append(self._make_action("LOBBY_ACTIVATED", context["frame_idx"], lobby_conf, "Lobby activated after touch"))

        return self._dedupe_actions(actions)

    def _infer_defender_events(self, context):
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
        tackle_conf = min(
            1.0,
            0.25 * pressure
            + 0.20 * full_graph_pressure
            + 0.25 * max(0.0, 1.0 - raider_speed / 0.35)
            + 0.20 * support
            + 0.10 * min(1.0, context["global_context"].get("best_containment_score", 0.0)),
        )

        if len(context["nearby_defenders"]) >= 2 and tackle_conf >= 0.6:
            actions.append(self._make_action("DEFENDER_ASSIST_TACKLE", context["frame_idx"], tackle_conf, "Multiple defenders are engaging the raider"))
            # Phase 1.5: Update buffer for tackle momentum
            self.action_buffer["DEFENDER_ASSIST_TACKLE"] = (context["frame_idx"], tackle_conf)

        if tackle_conf >= 0.68:
            actions.append(self._make_action("DEFENDER_TACKLE", context["frame_idx"], tackle_conf, "Defenders tackled the raider"))
            actions.append(self._make_action("RAIDER_CAUGHT", context["frame_idx"], min(1.0, tackle_conf + 0.08), "Raider was caught by defenders"))
            self.current_raid["raider_caught"] = True
            self.current_raid["raid_ended"] = True
            self.current_raid["raider_out"] = True
            # Phase 1.5: Update action buffer
            self.action_buffer["DEFENDER_TACKLE"] = (context["frame_idx"], tackle_conf)

        if defenders_on_court <= 3 and tackle_conf >= 0.68:
            actions.append(self._make_action("SUPER_TACKLE_TRIGGER", context["frame_idx"], min(1.0, tackle_conf + 0.06), "Super tackle condition triggered"))
            # Phase 1.5: Update action buffer
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

    def _infer_boundary_events(self, context):
        actions = []
        rx, ry = context["raider_pos"]

        if rx < 0.75 or rx > 9.25:
            lobby_conf = min(1.0, 0.55 + min(abs(rx - 0.75), abs(rx - 9.25)) * 0.4)
            actions.append(self._make_action("RAIDER_LOBBY_ENTRY", context["frame_idx"], lobby_conf, "Raider entered lobby region"))
            if not self.current_raid["lobby_activated"]:
                actions.append(self._make_action("RAIDER_ILLEGAL_LOBBY_ENTRY", context["frame_idx"], min(1.0, lobby_conf + 0.12), "Illegal raider lobby entry before touch"))
                self.current_raid["raider_illegal_lobby"] = True

        if rx < 0.0 or rx > 10.0 or ry < 0.0 or ry > 6.5:
            out_conf = 0.85
            actions.append(self._make_action("OUT_OF_BOUNDS", context["frame_idx"], out_conf, "Raider moved out of bounds"))
            if not self.current_raid["touch_occurred"]:
                actions.append(self._make_action("RAIDER_STEP_OUT", context["frame_idx"], out_conf, "Raider stepped out without a valid touch"))
            else:
                actions.append(self._make_action("RAIDER_SELF_OUT", context["frame_idx"], out_conf - 0.05, "Raider self-out after contact"))
            self.current_raid["raid_ended"] = True
            self.current_raid["raider_out"] = True

        return self._dedupe_actions(actions)

    def _apply_rules(self, actions, context):
        finalized = []
        consistency_scores = self._consistency_scores(context)
        actions = self._apply_consistency_to_actions(actions, consistency_scores)
        actions_by_type = {action["type"]: action for action in self._dedupe_actions(actions)}
        touch_count = len(self.current_raid["touched_defenders"])

        for action in actions_by_type.values():
            event_key = self._action_dedupe_key(action)
            if event_key in self.current_raid["emitted_action_keys"]:
                continue
            self.current_raid["emitted_action_keys"].add(event_key)
            finalized.append(action)
            self.confirmed_action_history.append(action)

        if (
            self.current_raid["touch_occurred"]
            and self.current_raid["returned_middle"]
            and not self.current_raid["attacker_points_awarded"]
        ):
            points = touch_count
            if points > 0:
                finalized.append(self._score_action("REVIVAL", context["frame_idx"], 0.88, f"Revive {points} players", points, "attacker"))
                self.total_points["attacker"] += points
                self.current_raid["revivals_pending"] += points
                self.current_raid["attacker_points_awarded"] = True

        if (
            self.current_raid["bonus_touch"]
            and context["defenders_on_court"] >= 6
            and self.current_raid["returned_middle"]
            and not self.current_raid["bonus_points_awarded"]
        ):
            finalized.append(self._score_action("RAIDER_BONUS_TOUCH", context["frame_idx"], 0.82, "Bonus point awarded", 1, "attacker"))
            self.total_points["attacker"] += 1
            self.current_raid["bonus_points_awarded"] = True

        if self.current_raid["raider_caught"] and not self.current_raid["defender_points_awarded"]:
            if context["defenders_on_court"] <= 3:
                finalized.append(self._score_action("SUPER_TACKLE_TRIGGER", context["frame_idx"], 0.84, "Super tackle awarded", 2, "defender"))
                self.total_points["defender"] += 2
                self.current_raid["super_tackle_awarded"] = True
            else:
                finalized.append(self._score_action("DEFENDER_TACKLE", context["frame_idx"], 0.8, "Tackle point awarded", 1, "defender"))
                self.total_points["defender"] += 1
            self.current_raid["defender_points_awarded"] = True

        if (
            touch_count >= 3
            and self.current_raid["returned_middle"]
            and not self.current_raid["super_raid_awarded"]
        ):
            finalized.append(self._make_action("SUPER_RAID_TRIGGER", context["frame_idx"], 0.86, "Super raid triggered", metadata={"touch_count": touch_count}))
            self.current_raid["super_raid_awarded"] = True

        if (
            self.current_raid["returned_middle"]
            and not self.current_raid["touch_occurred"]
            and not self.current_raid["bonus_touch"]
            and not self.current_raid["empty_raid_recorded"]
        ):
            finalized.append(self._make_action("RAIDER_EMPTY_RAID", context["frame_idx"], 0.82, "Raid completed without a touch"))
            self.empty_raid_count += 1
            self.current_raid["empty_raid_recorded"] = True
            if self.empty_raid_count >= 2:
                finalized.append(self._make_action("DO_OR_DIE_RAID", context["frame_idx"], 0.8, "Next raid is do-or-die"))
                self.do_or_die_next = True

        if self.current_raid["raider_illegal_lobby"] and not self.current_raid["step_out_recorded"]:
            finalized.append(self._score_action("RAIDER_ILLEGAL_LOBBY_ENTRY", context["frame_idx"], 0.81, "Illegal lobby entry point to defenders", 1, "defender"))
            self.total_points["defender"] += 1
            self.current_raid["step_out_recorded"] = True
            self.current_raid["raid_ended"] = True

        if "RAIDER_STEP_OUT" in actions_by_type and not self.current_raid["step_out_recorded"]:
            finalized.append(self._score_action("RAIDER_STEP_OUT", context["frame_idx"], 0.84, "Raider stepped out, defenders score", 1, "defender"))
            self.total_points["defender"] += 1
            self.current_raid["step_out_recorded"] = True

        if "DEFENDER_ENDLINE_TOUCH" in actions_by_type and not self.current_raid["touch_occurred"]:
            finalized.append(self._score_action("DEFENDER_STEP_OUT", context["frame_idx"], 0.76, "Defender step-out awards raider point", 1, "attacker"))
            self.total_points["attacker"] += 1

        if context["defenders_on_court"] == 0 and not self.current_raid["all_out_awarded"]:
            finalized.append(self._score_action("ALL_OUT_TRIGGER", context["frame_idx"], 0.9, "All out triggered", 2, "attacker"))
            self.total_points["attacker"] += 2
            self.current_raid["all_out_awarded"] = True

        finalized = self._dedupe_actions(finalized)
        if self.current_raid["raid_ended"] and self.current_raid["returned_middle"]:
            self._reset_raid_state()
        elif self.current_raid["raid_ended"] and self.current_raid["raider_out"]:
            self._reset_raid_state()
        return finalized

    def _consistency_scores(self, context):
        """Phase 1.4: Per-action consistency scoring with additive penalties and floors.
        Prevents cascading false negatives from one weak factor poisoning all actions.
        """
        scores = {
            "contact_legality": 1.0,
            "bonus_legality": 1.0,
            "tackle_legality": 1.0,
            "return_legality": 1.0,
            "lobby_legality": 1.0,
        }

        # Contact legality: only consider pairwise contact strength, not other factors
        if not context["pairwise_contacts"]:
            scores["contact_legality"] = 0.75  # Degrade gracefully, don't multiply
        elif max(score for _, _, score in context["pairwise_contacts"]) < 0.55:
            scores["contact_legality"] = 0.85  # Less aggressive penalty
        else:
            scores["contact_legality"] = 1.0  # Good contact quality

        # Bonus legality: only check defender visibility
        if context["global_context"].get("visible_defenders", 0) < 6:
            scores["bonus_legality"] = 0.80  # Additive penalty instead of ×0.7
        else:
            scores["bonus_legality"] = 1.0

        # Tackle legality: requires BOTH geometric pressure AND contact presence
        # Stream A: geometric (defender proximity/containment)
        geometric_score = 1.0
        if len(context["nearby_defenders"]) < 2:
            geometric_score *= 0.85
        if context["raider_speed"] > 0.45:
            geometric_score *= 0.88
        if context["global_context"].get("best_containment_score", 0.0) < 0.18:
            geometric_score *= 0.90
        
        # Stream B: contact requirement (only trust tackle if contact was seen)
        has_recent_contact = self._has_recent_action("RAIDER_DEFENDER_CONTACT", lookback=3)
        contact_stream = 1.0 if has_recent_contact else 0.75
        
        # Blend: 60% geometric + 40% contact presence
        scores["tackle_legality"] = 0.6 * geometric_score + 0.4 * contact_stream

        # Return legality: only if some contact/bonus happened
        if not self.current_raid["touch_occurred"] and not self.current_raid["bonus_touch"]:
            scores["return_legality"] = 0.80
        else:
            scores["return_legality"] = 1.0

        # Lobby legality: only check raider position, account for legitimacy of lobby access
        rx = context["raider_pos"][0]
        if 0.75 <= rx <= 9.25:
            scores["lobby_legality"] = 0.75  # In main court, lobby is risky
        elif self.current_raid["touch_occurred"]:
            scores["lobby_legality"] = 0.95  # Post-contact lobby is legitimate
        else:
            scores["lobby_legality"] = 0.70  # Pre-contact lobby is illegal

        return scores

    def _apply_consistency_to_actions(self, actions, consistency_scores):
        """Phase 1.1: Apply consistency scores with additive penalties and minimum floor.
        Prevents cascading false negatives from multiplicative penalties.
        """
        adjusted = []
        for action in actions:
            confidence = action["confidence"]
            
            # Determine which consistency factor applies
            applied_factor = 1.0
            if action["type"] in {"RAIDER_DEFENDER_CONTACT", "RAIDER_MULTIPLE_DEFENDER_TOUCH", "LOBBY_ACTIVATED"}:
                applied_factor = consistency_scores["contact_legality"]
            elif action["type"] in {"RAIDER_CROSSED_BONUS_LINE", "RAIDER_BONUS_TOUCH"}:
                applied_factor = consistency_scores["bonus_legality"]
            elif action["type"] in {"DEFENDER_ASSIST_TACKLE", "DEFENDER_TACKLE", "RAIDER_CAUGHT", "SUPER_TACKLE_TRIGGER"}:
                applied_factor = consistency_scores["tackle_legality"]
            elif action["type"] == "RAIDER_RETURNED_MIDDLE":
                applied_factor = consistency_scores["return_legality"]
            elif action["type"] in {"RAIDER_LOBBY_ENTRY", "RAIDER_ILLEGAL_LOBBY_ENTRY"}:
                applied_factor = consistency_scores["lobby_legality"]
            
            # Phase 1.1: Replace multiplicative with floor-based logic
            # If legality factor < 1.0, use it as a confidence floor: don't drop below (original * factor)
            # This prevents cascading multipliers like 0.62 * 0.65 * 0.82 * 0.75
            if applied_factor < 1.0:
                confidence_floor = confidence * applied_factor
                confidence = max(confidence_floor, confidence - 0.10)  # Don't degrade more than 10 percentage points
            
            action = dict(action)
            action["confidence"] = float(np.clip(confidence, 0.0, 1.0))
            action["consistency"] = {
                "contact": consistency_scores["contact_legality"],
                "bonus": consistency_scores["bonus_legality"],
                "tackle": consistency_scores["tackle_legality"],
                "return": consistency_scores["return_legality"],
                "lobby": consistency_scores["lobby_legality"],
            }
            adjusted.append(action)
        return adjusted

    def _reset_raid_state(self):
        self.current_raid = self._new_raid_state()
        self.touched_defenders.clear()

    def _has_recent_action(self, action_type, lookback=3):
        """Phase 1.5: Check if action was seen within last N frames for temporal hysteresis."""
        if action_type not in self.action_buffer:
            return False
        frame_idx, _ = self.action_buffer[action_type]
        return (self.current_frame_idx - frame_idx) <= lookback

    def _make_action(self, action_type, frame_idx, confidence, description, metadata=None):
        action = {
            "type": action_type,
            "frame": frame_idx,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "description": description,
            "points": 0,
        }
        if metadata:
            action.update(metadata)
        return action

    def _score_action(self, action_type, frame_idx, confidence, description, points, side):
        return {
            "type": action_type,
            "frame": frame_idx,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "description": description,
            "points": points,
            "side": side,
        }

    def _dedupe_actions(self, actions):
        deduped = {}
        for action in actions:
            key = self._action_dedupe_key(action)
            current = deduped.get(key)
            if current is None or action["confidence"] > current["confidence"]:
                deduped[key] = action
        return list(deduped.values())

    def _action_dedupe_key(self, action):
        return (
            action["type"],
            action.get("defender_id"),
            action.get("touch_count"),
            action.get("side"),
        )

    def _pairwise_contact_score(self, raider_node, defender_node, proposal):
        dist = proposal["features"]["dist"]
        rel_vel = proposal["features"]["rel_vel"]
        visibility = 0.5 * (raider_node.get("track_confidence", 0.5) + defender_node.get("track_confidence", 0.5))
        motion_align = min(1.0, rel_vel / 1.8)
        proximity = max(0.0, 1.0 - dist / 1.2)
        return float(np.clip(0.5 * proximity + 0.25 * motion_align + 0.25 * visibility, 0.0, 1.0))

    def _pair_factor_score(self, pair_factor):
        features = pair_factor["features"]
        proximity = max(0.0, 1.0 - features["distance"] / 1.2)
        rel_vel = min(1.0, features["relative_velocity"] / 1.8)
        approach = min(1.0, features["approach_score"])
        adjacency = min(1.0, features["adjacency"])
        visibility = min(1.0, features["track_confidence"])
        return float(np.clip(0.35 * proximity + 0.2 * rel_vel + 0.2 * approach + 0.1 * adjacency + 0.15 * visibility, 0.0, 1.0))

    def _higher_order_pressure(self, raider_pos, defenders):
        if not defenders:
            return 0.0
        close = []
        for defender in defenders:
            pos = np.array(defender["spatial"])
            dist = np.linalg.norm(pos - raider_pos)
            if dist < 1.4:
                close.append(1.0 - dist / 1.4)
        if not close:
            return 0.0
        return float(np.clip(np.mean(close) + 0.12 * max(0, len(close) - 1), 0.0, 1.0))

    def _graph_pressure_score(self, factor_nodes, raider_id):
        scores = []
        for factor in factor_nodes:
            triplet = factor.get("triplet", ())
            if raider_id not in triplet:
                continue
            factor_type = factor.get("type")
            if factor_type == "THIRD_ORDER_PRESSURE":
                distances = factor["features"].get("distances", [])
                spread = factor["features"].get("spread", 1.0)
                if not distances:
                    continue
                compactness = factor["features"].get("compactness", max(0.0, 1.0 - np.mean(distances) / 1.8))
                spread_score = max(0.0, 1.0 - spread / 0.9)
                scores.append(0.6 * compactness + 0.4 * spread_score)
            elif factor_type == "DEFENDER_CONTAINMENT":
                angle = factor["features"].get("angle", 0.0)
                spread = factor["features"].get("spread", 1.0)
                scores.append(0.7 * angle + 0.3 * max(0.0, 1.0 - spread / 1.0))
        return float(np.clip(max(scores) if scores else 0.0, 0.0, 1.0))

    def _line_touch_confidence(self, confirmed_events, event_type, fallback=0.0):
        matches = [event["confidence"] for event in confirmed_events if event["type"] == event_type]
        return max(matches) if matches else fallback

    def _line_factor_score(self, line_factor, fallback=0.0):
        if line_factor is None:
            return fallback
        features = line_factor["features"]
        # Phase 1.2: Relax distance threshold from 0.35 to 0.50 (accounts for tracking noise)
        distance_score = max(0.0, 1.0 - features["distance"] / 0.50)
        active_bonus = 0.15 if features["active"] else 0.0
        confidence = 0.65 * distance_score + 0.35 * features["track_confidence"] + active_bonus
        return float(np.clip(max(fallback, confidence), 0.0, 1.0))

    def _temporal_event_confidence(self, confirmed_events, object_id):
        matches = [event["confidence"] for event in confirmed_events if event["object"] == object_id]
        return max(matches) if matches else 0.0

    def _unary_line_score(self, value, line_value, inverse=False, scale=1.0):
        delta = (line_value - value) if inverse else (value - line_value)
        return float(np.clip(0.55 + delta * 0.1 * scale, 0.0, 1.0))

    def get_accuracy_metrics(self):
        total = self.accuracy_stats["total_actions"]
        if total == 0:
            return {
                "estimated_accuracy": 0.0,
                "high_confidence_rate": 0.0,
                "total_actions": 0,
                "factor_consistency": 0.0,
            }
        high_conf = self.accuracy_stats["high_confidence_actions"] / total
        estimated = min(0.88, 0.52 + 0.25 * high_conf + 0.10 * self.accuracy_stats["factor_consistency"])
        return {
            "estimated_accuracy": estimated,
            "high_confidence_rate": high_conf,
            "total_actions": total,
            "factor_consistency": self.accuracy_stats["factor_consistency"],
        }

    def _update_accuracy(self, actions, factor_scores):
        for action in actions:
            self.accuracy_stats["total_actions"] += 1
            if action.get("confidence", 0.0) >= 0.7:
                self.accuracy_stats["high_confidence_actions"] += 1
        if factor_scores:
            self.accuracy_stats["factor_consistency"] = float(np.clip(np.mean(factor_scores), 0.0, 1.0))
