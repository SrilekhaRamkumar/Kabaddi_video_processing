import numpy as np


def collect_raider_stats(gallery, raider_stats, frame_idx, baulk_y):
    for pid, data in gallery.items():
        if data["display_pos"] is None:
            continue

        _, cy = data["display_pos"]
        if pid not in raider_stats:
            raider_stats[pid] = {
                "first_seen": frame_idx,
                "min_y": cy,
                "max_y": cy,
                "vy_list": [],
                "behind_baulk_frames": 0,
                "frames": 0,
            }

        record = raider_stats[pid]
        record["min_y"] = min(record["min_y"], cy)
        record["max_y"] = max(record["max_y"], cy)
        record["frames"] += 1

        if cy > baulk_y:
            record["behind_baulk_frames"] += 1

        state = data["kf"].statePost.flatten()
        record["vy_list"].append(state[3])


def assign_raider(gallery, raider_stats, frame_idx, assign_frame, baulk_y, bonus_y):
    visible_players = [pid for pid, data in gallery.items() if data["display_pos"] is not None and data["age"] == 0]
    if len(visible_players) < 3:
        return None, False, assign_frame + 10

    best_score = -1e9
    best_id = None

    for pid in visible_players:
        record = raider_stats.get(pid)
        if record is None or record["frames"] < 20:
            continue
        if record["min_y"] > baulk_y - 1:
            continue

        avg_vy = np.mean(record["vy_list"]) if record["vy_list"] else 0
        if abs(avg_vy) < 0.05:
            continue
        if record["min_y"] > bonus_y - 0.3:
            continue

        xi, yi = gallery[pid]["display_pos"]
        state_i = gallery[pid]["kf"].statePost.flatten()
        vxi, vyi = state_i[2], state_i[3]
        depths = [gallery[other]["display_pos"][1] for other in visible_players if other != pid]
        if not depths:
            continue

        depth_rank = sum(yi > depth for depth in depths)
        convergence_count = 0
        close_players = 0

        for other in visible_players:
            if other == pid:
                continue
            xj, yj = gallery[other]["display_pos"]
            state_j = gallery[other]["kf"].statePost.flatten()
            vxj, vyj = state_j[2], state_j[3]

            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            if dist < 2.5:
                close_players += 1

            dir_vec = np.array([dx / dist, dy / dist])
            vel_vec = np.array([vxj, vyj])
            if np.dot(vel_vec, dir_vec) > 0:
                convergence_count += 1

        speed = np.sqrt(vxi * vxi + vyi * vyi)
        entry_prior = record["first_seen"] / frame_idx
        score = depth_rank * 4.0 + convergence_count * 3.5 + close_players * 2.0 + speed * 1.5 + entry_prior * 5.0

        if score > best_score:
            best_score = score
            best_id = pid

    return best_id, best_id is not None, assign_frame
