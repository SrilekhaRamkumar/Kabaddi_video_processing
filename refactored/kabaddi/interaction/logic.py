import numpy as np


def build_player_states(gallery):
    player_states = {}
    active_players = [pid for pid, data in gallery.items() if data["age"] == 0 and data["display_pos"] is not None]

    for pid in active_players:
        state = gallery[pid]["kf"].statePost.flatten()
        player_states[pid] = {
            "pos": gallery[pid]["display_pos"],
            "vel": (state[2], state[3]),
            "feat": gallery[pid]["feat"],
            "track_confidence": gallery[pid].get("track_confidence", 0.0),
            "visibility_confidence": gallery[pid].get("visibility_confidence", 0.0),
        }

    return player_states, active_players


def process_interactions(
    frame_idx,
    gallery,
    player_states,
    active_players,
    raider_id,
    raid_assignment_done,
    proposal_engine,
    bonus_y,
    baulk_y,
    end_line_y,
    line_margin,
    lobby_left,
    lobby_right,
    touch_confirmed,
    log_event,
):
    interaction_candidates = []

    if raid_assignment_done and raider_id in gallery:
        raider_data = gallery[raider_id]
        if raider_data["display_pos"]:
            r_pos = raider_data["display_pos"]
            r_vel = (raider_data["kf"].statePost[2][0], raider_data["kf"].statePost[3][0])

            for pid in active_players:
                if pid == raider_id:
                    continue
                defender_pos = player_states[pid]["pos"]
                defender_vel = player_states[pid]["vel"]
                defender_feat = player_states[pid]["feat"]
                dist = np.sqrt((r_pos[0] - defender_pos[0]) ** 2 + (r_pos[1] - defender_pos[1]) ** 2)

                if dist < 1.5:
                    proposal_engine.encode_hhi(
                        frame_idx,
                        raider_id,
                        pid,
                        r_pos,
                        defender_pos,
                        r_vel,
                        defender_vel,
                        raider_data["feat"],
                        defender_feat,
                    )
                    interaction_candidates.append({"pair": (raider_id, pid), "distance": dist})

                    if not touch_confirmed and dist < 1.0:
                        touch_confirmed = True
                        log_event("RAIDER_DEFENDER_CONTACT", raider_id, frame_idx)

            proposal_engine.encode_hli(frame_idx, raider_id, "BONUS", r_pos, bonus_y)
            proposal_engine.encode_hli(frame_idx, raider_id, "BAULK", r_pos, baulk_y)

            for pid, pdata in gallery.items():
                if pdata["display_pos"]:
                    proposal_engine.encode_hli(frame_idx, pid, "END_LINE", pdata["display_pos"], end_line_y)

    for pid, pdata in player_states.items():
        if raider_id is not None and pid != raider_id:
            _, py = pdata["pos"]
            if abs(py - end_line_y) < line_margin:
                log_event("DEFENDER_ENDLINE_TOUCH", pid, frame_idx)

    if raid_assignment_done and raider_id in player_states:
        _, ry = player_states[raider_id]["pos"]
        if abs(ry - bonus_y) < line_margin:
            log_event("RAIDER_BONUS_TOUCH", raider_id, frame_idx)
        if abs(ry - baulk_y) < line_margin:
            log_event("RAIDER_BAULK_TOUCH", raider_id, frame_idx)

    for pid, pdata in player_states.items():
        px, _ = pdata["pos"]
        if not touch_confirmed and (px < lobby_left or px > lobby_right):
            log_event("ILLEGAL_LOBBY_ENTRY_BEFORE_TOUCH", pid, frame_idx)

    if touch_confirmed and raider_id in player_states:
        _, ry = player_states[raider_id]["pos"]
        if ry < 0.8:
            log_event("RAIDER_RETURNED_MIDDLE", raider_id, frame_idx)

    return interaction_candidates, touch_confirmed
