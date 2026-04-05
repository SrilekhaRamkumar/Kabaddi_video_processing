from pathlib import Path

import cv2
import numpy as np

from classifier_bridge import ConfirmedWindowClassifierBridge
from interaction_graph import InteractionProposalEngine
from raider_logic import assign_raider
from temporal_events import TemporalInteractionCandidateManager
from touch_classifier_inference import IMAGENET_MEAN, IMAGENET_STD
from tracking_pipeline import create_kalman


CANVAS_W = 1280
CANVAS_H = 720
FPS = 30
FRAME_COUNT = 220
MARGIN = 70
SEED = 7

OUT_DIR = Path(__file__).resolve().parent / "debug_outputs"
OUT_DIR.mkdir(exist_ok=True)
PNG_PATH = OUT_DIR / "kalman_filter_visualization.png"
MP4_PATH = OUT_DIR / "kalman_filter_visualization.mp4"
GALLERY_PNG_PATH = OUT_DIR / "gallery_visualization.png"
MAT_PNG_PATH = OUT_DIR / "empty_kabaddi_mat.png"
RAIDER_PNG_PATH = OUT_DIR / "raider_score_visualization.png"
INTERACTION_PNG_PATH = OUT_DIR / "interaction_triplets_visualization.png"
CONF_EVENT_PNG_PATH = OUT_DIR / "confirmed_event_confidence_visualization.png"
TOUCH_VIS_PNG_PATH = OUT_DIR / "touch_classifier_visualization.png"


def world_to_canvas(pt):
    x = int(np.clip(pt[0], 0, 10) / 10.0 * (CANVAS_W - 2 * MARGIN) + MARGIN)
    y = int((1.0 - np.clip(pt[1], 0, 1)) * (CANVAS_H - 2 * MARGIN) + MARGIN)
    return x, y


def build_ground_truth(frame_idx):
    t = frame_idx / 22.0
    x = 1.0 + 0.03 * frame_idx + 0.6 * np.sin(t * 0.85)
    y = 0.48 + 0.16 * np.sin(t * 1.45) + 0.05 * np.cos(t * 0.7)
    return np.array([x, y], dtype=np.float32)


def noisy_measurement(point, rng):
    noise = rng.normal(0, [0.18, 0.045], size=2).astype(np.float32)
    return point + noise


def draw_legend(canvas):
    items = [
        ("Ground Truth", (0, 200, 0)),
        ("Measurement", (0, 140, 255)),
        ("Prediction", (210, 210, 210)),
        ("Kalman Estimate", (255, 0, 255)),
    ]
    x = 24
    y = 30
    for label, color in items:
      cv2.line(canvas, (x, y), (x + 26, y), color, 3, cv2.LINE_AA)
      cv2.putText(
          canvas,
          label,
          (x + 36, y + 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          color,
          2,
          cv2.LINE_AA,
      )
      x += 250


def draw_path(canvas, points, color, thickness):
    if len(points) < 2:
        return
    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], color, thickness, cv2.LINE_AA)


def draw_point(canvas, pt, color, radius=6):
    cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, pt, radius + 2, color, 1, cv2.LINE_AA)


def draw_state_panel(canvas, frame_idx, gt, meas, pred, estimate, velocity, missed):
    panel = canvas.copy()
    cv2.rectangle(panel, (CANVAS_W - 390, 24), (CANVAS_W - 24, 220), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (CANVAS_W - 390, 24), (CANVAS_W - 24, 220), (90, 90, 90), 1)

    rows = [
        f"Frame: {frame_idx:03d}",
        f"Measurement: {'missed' if missed else f'({meas[0]:.2f}, {meas[1]:.2f})'}",
        f"Prediction: ({pred[0]:.2f}, {pred[1]:.2f})",
        f"Estimate: ({estimate[0]:.2f}, {estimate[1]:.2f})",
        f"Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f})",
        f"Ground Truth: ({gt[0]:.2f}, {gt[1]:.2f})",
    ]

    cv2.putText(
        canvas,
        "Kalman State",
        (CANVAS_W - 360, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = 86
    for row in rows:
        cv2.putText(
            canvas,
            row,
            (CANVAS_W - 360, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 24


def make_mock_gallery():
    players = [
        {
            "pid": 4,
            "bbox": (260, 180, 340, 410),
            "foot": (300, 410),
            "display_pos": (4.8, 3.4),
            "det_conf": 0.93,
            "vis_conf": 0.91,
            "track_conf": 0.95,
            "age": 0,
            "hits": 14,
            "miss_streak": 0,
            "occlusion_lock": False,
            "protected_until": 128,
        },
        {
            "pid": 7,
            "bbox": (540, 210, 620, 430),
            "foot": (580, 430),
            "display_pos": (5.6, 2.7),
            "det_conf": 0.79,
            "vis_conf": 0.73,
            "track_conf": 0.76,
            "age": 2,
            "hits": 9,
            "miss_streak": 1,
            "occlusion_lock": True,
            "protected_until": -1,
        },
        {
            "pid": 0,
            "bbox": (820, 230, 900, 445),
            "foot": (860, 445),
            "display_pos": (2.1, 1.9),
            "det_conf": 0.68,
            "vis_conf": 0.64,
            "track_conf": 0.61,
            "age": 5,
            "hits": 6,
            "miss_streak": 2,
            "occlusion_lock": False,
            "protected_until": -1,
        },
    ]

    gallery = {}
    for idx, player in enumerate(players):
        kf = create_kalman(*player["foot"])
        kf.statePost = np.array(
            [
                [player["foot"][0]],
                [player["foot"][1]],
                [(-2.8 + idx * 2.1)],
                [(1.1 - idx * 0.7)],
            ],
            dtype=np.float32,
        )
        gallery[player["pid"]] = {
            "feat": np.zeros(8, dtype=np.float32),
            "appearance_anchor": np.zeros(8, dtype=np.float32),
            "kf": kf,
            "age": player["age"],
            "display_pos": player["display_pos"],
            "flow_pts": None,
            "last_bbox": player["bbox"],
            "last_foot": player["foot"],
            "hits": player["hits"],
            "miss_streak": player["miss_streak"],
            "occlusion_count": 2 if player["occlusion_lock"] else 0,
            "occlusion_lock": player["occlusion_lock"],
            "protected_until": player["protected_until"],
            "detection_confidence": player["det_conf"],
            "visibility_confidence": player["vis_conf"],
            "track_confidence": player["track_conf"],
        }
    return gallery


def draw_gallery_visualization(gallery, output_path):
    canvas = np.full((820, 1400, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Player Gallery Visualization",
        (28, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Mock snapshot of the gallery entries created by tracking_pipeline.py",
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    court_x1, court_y1, court_x2, court_y2 = 900, 120, 1340, 430
    cv2.rectangle(canvas, (court_x1, court_y1), (court_x2, court_y2), (25, 25, 25), 2)
    cv2.line(canvas, (court_x1, (court_y1 + court_y2) // 2), (court_x2, (court_y1 + court_y2) // 2), (80, 80, 80), 1)
    cv2.putText(canvas, "Court Display Positions", (930, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 45, 45), 2, cv2.LINE_AA)

    colors = [(0, 130, 255), (255, 0, 170), (0, 180, 80), (120, 80, 255)]
    card_w, card_h = 400, 190
    start_x, start_y = 28, 120
    gap_y = 20

    for idx, (pid, data) in enumerate(sorted(gallery.items())):
        x = start_x
        y = start_y + idx * (card_h + gap_y)
        color = colors[idx % len(colors)]
        cv2.rectangle(canvas, (x, y), (x + card_w, y + card_h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + card_w, y + card_h), (220, 220, 220), 2)
        cv2.rectangle(canvas, (x, y), (x + 12, y + card_h), color, -1)

        bbox = data["last_bbox"]
        foot = data["last_foot"]
        state = data["kf"].statePost.flatten()
        rows = [
            f"ID {pid}",
            f"bbox: {bbox}",
            f"foot: {foot}",
            f"statePost: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.2f}, {state[3]:.2f})",
            f"display_pos: {data['display_pos']}",
            f"age/hits/miss: {data['age']} / {data['hits']} / {data['miss_streak']}",
            f"det/vis/track: {data['detection_confidence']:.2f} / {data['visibility_confidence']:.2f} / {data['track_confidence']:.2f}",
            f"occlusion_lock: {data['occlusion_lock']}   protected_until: {data['protected_until']}",
        ]
        yy = y + 28
        for ridx, row in enumerate(rows):
            font = 0.78 if ridx == 0 else 0.54
            thickness = 2 if ridx == 0 else 1
            cv2.putText(canvas, row, (x + 24, yy), cv2.FONT_HERSHEY_SIMPLEX, font, (35, 35, 35), thickness, cv2.LINE_AA)
            yy += 22 if ridx == 0 else 20

        mini_x1, mini_y1, mini_x2, mini_y2 = x + 270, y + 22, x + 380, y + 132
        cv2.rectangle(canvas, (mini_x1, mini_y1), (mini_x2, mini_y2), (235, 235, 235), -1)
        cv2.rectangle(canvas, (mini_x1, mini_y1), (mini_x2, mini_y2), (190, 190, 190), 1)
        bx1, by1, bx2, by2 = bbox
        norm_x1 = mini_x1 + int((bx1 / 1280.0) * (mini_x2 - mini_x1))
        norm_y1 = mini_y1 + int((by1 / 720.0) * (mini_y2 - mini_y1))
        norm_x2 = mini_x1 + int((bx2 / 1280.0) * (mini_x2 - mini_x1))
        norm_y2 = mini_y1 + int((by2 / 720.0) * (mini_y2 - mini_y1))
        cv2.rectangle(canvas, (norm_x1, norm_y1), (norm_x2, norm_y2), color, 2)
        foot_x = mini_x1 + int((foot[0] / 1280.0) * (mini_x2 - mini_x1))
        foot_y = mini_y1 + int((foot[1] / 720.0) * (mini_y2 - mini_y1))
        cv2.circle(canvas, (foot_x, foot_y), 4, color, -1, cv2.LINE_AA)

        court_pos = data["display_pos"]
        if court_pos is not None:
            cx = court_x1 + int((court_pos[0] / 10.0) * (court_x2 - court_x1))
            cy = court_y2 - int((court_pos[1] / 6.5) * (court_y2 - court_y1))
            cv2.circle(canvas, (cx, cy), 11, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(pid), (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
            vel_tip = (int(cx + state[2] * 10), int(cy - state[3] * 10))
            cv2.arrowedLine(canvas, (cx, cy), vel_tip, color, 2, cv2.LINE_AA, tipLength=0.2)

    cv2.imwrite(str(output_path), canvas)


def draw_empty_kabaddi_mat(output_path):
    canvas = np.full((760, 1280, 3), 255, dtype=np.uint8)
    x1, y1, x2, y2 = 140, 120, 1140, 640
    ink = (20, 20, 20)

    cv2.putText(
        canvas,
        "Empty Kabaddi Mat",
        (40, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        ink,
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(canvas, (x1, y1), (x2, y2), ink, 3)

    court_w = x2 - x1
    court_h = y2 - y1

    def y_from_ratio(ratio):
        return int(y2 - ratio * court_h)

    def x_from_ratio(ratio):
        return int(x1 + ratio * court_w)

    baulk_y = y_from_ratio(3.75 / 6.5)
    bonus_y = y_from_ratio(4.75 / 6.5)
    mid_y = y_from_ratio(0.5)
    lobby_left_x = x_from_ratio(0.75 / 10.0)
    lobby_right_x = x_from_ratio(9.25 / 10.0)

    cv2.line(canvas, (x1, baulk_y), (x2, baulk_y), ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (x1, bonus_y), (x2, bonus_y), ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (x1, mid_y), (x2, mid_y), ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (lobby_left_x, y1), (lobby_left_x, y2), ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (lobby_right_x, y1), (lobby_right_x, y2), ink, 2, cv2.LINE_AA)

    labels = [
        ("Bonus", x1 + 14, bonus_y - 10),
        ("Baulk", x1 + 14, baulk_y - 10),
        ("Mid", x1 + 14, mid_y - 10),
        ("Lobby", lobby_left_x + 10, y1 + 24),
        ("Lobby", lobby_right_x - 70, y1 + 24),
    ]
    for text, tx, ty in labels:
        cv2.putText(
            canvas,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            ink,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def make_mock_raider_inputs():
    frame_idx = 120
    assign_frame = 60
    baulk_y = 3.75
    bonus_y = 4.75
    gallery = {
        4: {
            "display_pos": (5.4, 2.1),
            "age": 0,
            "kf": create_kalman(0, 0),
        },
        7: {
            "display_pos": (6.8, 3.1),
            "age": 0,
            "kf": create_kalman(0, 0),
        },
        0: {
            "display_pos": (3.2, 2.8),
            "age": 0,
            "kf": create_kalman(0, 0),
        },
    }
    gallery[4]["kf"].statePost = np.array([[0], [0], [0.32], [0.22]], dtype=np.float32)
    gallery[7]["kf"].statePost = np.array([[0], [0], [-0.08], [0.12]], dtype=np.float32)
    gallery[0]["kf"].statePost = np.array([[0], [0], [0.04], [0.01]], dtype=np.float32)

    raider_stats = {
        4: {
            "first_seen": 6,
            "min_y": 2.1,
            "max_y": 4.8,
            "vy_list": [0.24, 0.21, 0.25, 0.19, 0.23],
            "behind_baulk_frames": 34,
            "frames": 52,
        },
        7: {
            "first_seen": 22,
            "min_y": 3.0,
            "max_y": 4.2,
            "vy_list": [0.10, 0.12, 0.08, 0.09],
            "behind_baulk_frames": 19,
            "frames": 44,
        },
        0: {
            "first_seen": 15,
            "min_y": 3.3,
            "max_y": 3.8,
            "vy_list": [0.01, 0.03, 0.02, 0.01],
            "behind_baulk_frames": 11,
            "frames": 38,
        },
    }
    return gallery, raider_stats, frame_idx, assign_frame, baulk_y, bonus_y


def compute_raider_score_breakdown(gallery, raider_stats, frame_idx, baulk_y, bonus_y):
    visible_players = [pid for pid, data in gallery.items() if data["display_pos"] is not None and data["age"] == 0]
    rows = []
    for pid in visible_players:
        record = raider_stats.get(pid)
        if record is None:
            continue
        row = {
            "pid": pid,
            "eligible": True,
            "reason": "",
            "score": None,
        }
        if record["frames"] < 20:
            row["eligible"] = False
            row["reason"] = "frames < 20"
            rows.append(row)
            continue
        if record["min_y"] > baulk_y - 1:
            row["eligible"] = False
            row["reason"] = "never entered deep enough"
            rows.append(row)
            continue

        avg_vy = float(np.mean(record["vy_list"])) if record["vy_list"] else 0.0
        if abs(avg_vy) < 0.05:
            row["eligible"] = False
            row["reason"] = "|avg_vy| < 0.05"
            rows.append(row)
            continue
        if record["min_y"] > bonus_y - 0.3:
            row["eligible"] = False
            row["reason"] = "not before bonus threshold"
            rows.append(row)
            continue

        xi, yi = gallery[pid]["display_pos"]
        state_i = gallery[pid]["kf"].statePost.flatten()
        vxi, vyi = float(state_i[2]), float(state_i[3])
        depths = [gallery[other]["display_pos"][1] for other in visible_players if other != pid]
        if not depths:
            row["eligible"] = False
            row["reason"] = "no comparison players"
            rows.append(row)
            continue

        depth_rank = sum(yi > depth for depth in depths)
        convergence_count = 0
        close_players = 0
        for other in visible_players:
            if other == pid:
                continue
            xj, yj = gallery[other]["display_pos"]
            state_j = gallery[other]["kf"].statePost.flatten()
            vxj, vyj = float(state_j[2]), float(state_j[3])
            dx = xi - xj
            dy = yi - yj
            dist = float(np.sqrt(dx * dx + dy * dy) + 1e-6)
            if dist < 2.5:
                close_players += 1
            dir_vec = np.array([dx / dist, dy / dist], dtype=np.float32)
            vel_vec = np.array([vxj, vyj], dtype=np.float32)
            if float(np.dot(vel_vec, dir_vec)) > 0:
                convergence_count += 1

        speed = float(np.sqrt(vxi * vxi + vyi * vyi))
        entry_prior = float(record["first_seen"] / frame_idx)
        score_depth = depth_rank * 4.0
        score_convergence = convergence_count * 3.5
        score_close = close_players * 2.0
        score_speed = speed * 1.5
        score_entry = entry_prior * 5.0
        total = score_depth + score_convergence + score_close + score_speed + score_entry

        row.update({
            "avg_vy": avg_vy,
            "depth_rank": depth_rank,
            "convergence_count": convergence_count,
            "close_players": close_players,
            "speed": speed,
            "entry_prior": entry_prior,
            "score_depth": score_depth,
            "score_convergence": score_convergence,
            "score_close": score_close,
            "score_speed": score_speed,
            "score_entry": score_entry,
            "score": total,
        })
        rows.append(row)
    return rows


def draw_raider_score_visualization(output_path):
    gallery, raider_stats, frame_idx, assign_frame, baulk_y, bonus_y = make_mock_raider_inputs()
    best_id, _, _ = assign_raider(gallery, raider_stats, frame_idx, assign_frame, baulk_y, bonus_y)
    rows = compute_raider_score_breakdown(gallery, raider_stats, frame_idx, baulk_y, bonus_y)

    canvas = np.full((860, 1400, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Raider Score Visualization", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Standalone view of the scoring formula used in raider_logic.assign_raider()",
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    panel_x1, panel_y1, panel_x2, panel_y2 = 930, 120, 1360, 430
    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), 2)
    cv2.putText(canvas, "Court Context", (1060, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 45, 45), 2, cv2.LINE_AA)
    cv2.line(canvas, (panel_x1, panel_y1 + 130), (panel_x2, panel_y1 + 130), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Baulk", (panel_x1 + 10, panel_y1 + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.line(canvas, (panel_x1, panel_y1 + 70), (panel_x2, panel_y1 + 70), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Bonus", (panel_x1 + 10, panel_y1 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 2, cv2.LINE_AA)

    colors = [(0, 140, 255), (180, 0, 255), (0, 170, 70)]
    base_y = 120
    max_score = max((row["score"] or 0.0) for row in rows) if rows else 1.0

    for idx, row in enumerate(rows):
        y = base_y + idx * 210
        color = colors[idx % len(colors)]
        cv2.rectangle(canvas, (28, y), (870, y + 180), (255, 255, 255), -1)
        cv2.rectangle(canvas, (28, y), (870, y + 180), (220, 220, 220), 2)
        cv2.rectangle(canvas, (28, y), (40, y + 180), color, -1)

        header = f"ID {row['pid']}"
        if row["pid"] == best_id:
            header += "  <- selected as raider"
        cv2.putText(canvas, header, (58, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (35, 35, 35), 2, cv2.LINE_AA)

        if not row["eligible"]:
            cv2.putText(canvas, f"Rejected: {row['reason']}", (58, y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 180), 2, cv2.LINE_AA)
            continue

        cv2.putText(
            canvas,
            f"depth_rank*4={row['score_depth']:.2f} | convergence*3.5={row['score_convergence']:.2f} | close*2={row['score_close']:.2f}",
            (58, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"speed*1.5={row['score_speed']:.2f} | entry_prior*5={row['score_entry']:.2f} | avg_vy={row['avg_vy']:.2f}",
            (58, y + 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

        bar_x1, bar_y1, bar_x2, bar_y2 = 58, y + 125, 810, y + 152
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (238, 238, 238), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (205, 205, 205), 1)
        fill_w = int((row["score"] / max_score) * (bar_x2 - bar_x1))
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), color, -1)
        cv2.putText(
            canvas,
            f"total score = {row['score']:.2f}",
            (bar_x1, y + 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )

        gx, gy = gallery[row["pid"]]["display_pos"]
        cx = panel_x1 + int((gx / 10.0) * (panel_x2 - panel_x1))
        cy = panel_y2 - int((gy / 6.5) * (panel_y2 - panel_y1))
        state = gallery[row["pid"]]["kf"].statePost.flatten()
        cv2.circle(canvas, (cx, cy), 12, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(row["pid"]), (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
        vel_tip = (int(cx + state[2] * 55), int(cy - state[3] * 55))
        cv2.arrowedLine(canvas, (cx, cy), vel_tip, color, 2, cv2.LINE_AA, tipLength=0.2)

    cv2.imwrite(str(output_path), canvas)


def make_mock_interaction_proposals():
    engine = InteractionProposalEngine()
    frame_idx = 96
    raider_id = 4
    defender_positions = {
        4: (5.2, 2.6),
        7: (5.8, 2.9),
        0: (4.4, 3.1),
    }
    velocities = {
        4: (0.32, 0.18),
        7: (-0.12, -0.08),
        0: (0.06, -0.11),
    }
    dummy_feat = np.ones(16, dtype=np.float32)

    engine.encode_hhi(
        frame_idx,
        raider_id,
        7,
        defender_positions[4],
        defender_positions[7],
        velocities[4],
        velocities[7],
        dummy_feat,
        dummy_feat * 0.9,
    )
    engine.encode_hhi(
        frame_idx,
        raider_id,
        0,
        defender_positions[4],
        defender_positions[0],
        velocities[4],
        velocities[0],
        dummy_feat,
        dummy_feat * 1.05,
    )
    engine.encode_hli(frame_idx, raider_id, "BAULK", defender_positions[4], 3.75)
    engine.encode_hli(frame_idx, 7, "BONUS", defender_positions[7], 4.75)
    engine.encode_hli(frame_idx, 0, "END_LINE", defender_positions[0], 6.5)
    proposals = engine.finalize_frame_proposals()
    return proposals, defender_positions, raider_id, frame_idx


def draw_interaction_triplets_visualization(output_path):
    proposals, positions, raider_id, frame_idx = make_mock_interaction_proposals()
    canvas = np.full((860, 1480, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Interaction Proposal Visualization", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Standalone view of HHI and HLI triplets created by InteractionProposalEngine",
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    court_x1, court_y1, court_x2, court_y2 = 900, 120, 1425, 470
    cv2.rectangle(canvas, (court_x1, court_y1), (court_x2, court_y2), (30, 30, 30), 2)
    cv2.putText(canvas, "Court Context", (1085, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 45, 45), 2, cv2.LINE_AA)

    def court_px(x, y):
        px = court_x1 + int((x / 10.0) * (court_x2 - court_x1))
        py = court_y2 - int((y / 6.5) * (court_y2 - court_y1))
        return px, py

    for line_name, yv in [("BONUS", 4.75), ("BAULK", 3.75), ("END_LINE", 6.5)]:
        py = court_y2 - int((yv / 6.5) * (court_y2 - court_y1))
        cv2.line(canvas, (court_x1, py), (court_x2, py), (130, 130, 130), 1 if line_name != "BAULK" else 2, cv2.LINE_AA)
        cv2.putText(canvas, line_name, (court_x1 + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (85, 85, 85), 2, cv2.LINE_AA)

    colors = {4: (0, 140, 255), 7: (190, 0, 255), 0: (0, 170, 70)}
    for pid, pos in positions.items():
        px, py = court_px(*pos)
        cv2.circle(canvas, (px, py), 12, colors[pid], -1, cv2.LINE_AA)
        label = f"R{pid}" if pid == raider_id else f"D{pid}"
        cv2.putText(canvas, label, (px + 12, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)

    hhi = [p for p in proposals if p["type"] == "HHI"]
    hli = [p for p in proposals if p["type"] == "HLI"]

    for proposal in hhi:
        s = proposal["S"]
        o = proposal["O"]
        p1 = court_px(*positions[s])
        p2 = court_px(*positions[o])
        cv2.line(canvas, p1, p2, (80, 80, 80), 2, cv2.LINE_AA)

    section_y = 120
    cv2.putText(canvas, f"Frame {frame_idx:05d}", (34, section_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (45, 45, 45), 2, cv2.LINE_AA)

    card_x1, card_x2 = 28, 840
    card_h = 118
    gap = 16

    for idx, proposal in enumerate(hhi):
        y1 = section_y + idx * (card_h + gap)
        y2 = y1 + card_h
        cv2.rectangle(canvas, (card_x1, y1), (card_x2, y2), (255, 255, 255), -1)
        cv2.rectangle(canvas, (card_x1, y1), (card_x2, y2), (220, 220, 220), 2)
        cv2.rectangle(canvas, (card_x1, y1), (card_x1 + 12, y2), (0, 140, 255), -1)
        triplet = f"<R{proposal['S']}, {proposal['I']}, D{proposal['O']}>"
        cv2.putText(canvas, "HHI", (56, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, triplet, (56, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (55, 55, 55), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"dist={proposal['features']['dist']:.2f} m   rel_vel={proposal['features']['rel_vel']:.2f}",
            (56, y1 + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (85, 85, 85),
            1,
            cv2.LINE_AA,
        )

    hli_base_y = section_y + max(0, len(hhi)) * (card_h + gap) + 36
    cv2.putText(canvas, "HLI Triplets", (34, hli_base_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (45, 45, 45), 2, cv2.LINE_AA)

    for idx, proposal in enumerate(hli):
        y1 = hli_base_y + idx * (card_h + gap)
        y2 = y1 + card_h
        cv2.rectangle(canvas, (card_x1, y1), (card_x2, y2), (255, 255, 255), -1)
        cv2.rectangle(canvas, (card_x1, y1), (card_x2, y2), (220, 220, 220), 2)
        cv2.rectangle(canvas, (card_x1, y1), (card_x1 + 12, y2), (180, 0, 255), -1)
        actor_label = f"R{proposal['S']}" if proposal["S"] == raider_id else f"D{proposal['S']}"
        triplet = f"<{actor_label}, {proposal['I']}, {proposal['O']}>"
        status = "ACTIVE" if proposal["features"]["active"] else "INACTIVE"
        cv2.putText(canvas, "HLI", (56, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, triplet, (56, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (55, 55, 55), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"dist={proposal['features']['dist']:.2f} m   status={status}",
            (56, y1 + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (85, 85, 85),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def build_mock_confirmed_event_inputs():
    proposal = {
        "frame": 128,
        "type": "HHI",
        "S": 4,
        "O": 7,
        "I": "POTENTIAL_CONTACT",
        "features": {
            "dist": 0.58,
            "rel_vel": 0.92,
            "mask": [0.3, 0.1],
            "emb": [0.4] * 32,
        },
    }
    player_states = {
        4: {"track_confidence": 0.91},
        7: {"track_confidence": 0.79},
    }
    scene_graph = {
        "pair_factors": [
            {
                "type": "RAIDER_DEFENDER_PAIR",
                "nodes": [4, 7],
                "features": {
                    "distance": 0.58,
                    "relative_velocity": 0.92,
                    "approach_score": 0.74,
                    "adjacency": 0.66,
                    "track_confidence": 0.85,
                    "raider_involved": True,
                },
            }
        ],
        "line_factors": [],
        "global_context": {
            "best_contact_score": 0.72,
            "best_containment_score": 0.46,
            "visible_defenders": 3,
        },
    }
    candidate = {
        "type": "HHI",
        "subject": 4,
        "object": 7,
        "start_frame": 126,
        "last_frame": 128,
        "frames": [126, 127, 128],
        "confidences": [0.63, 0.71, 0.77],
        "factor_confidences": [0.55, 0.61, 0.69],
    }
    return proposal, player_states, scene_graph, candidate


def draw_confirmed_event_confidence_visualization(output_path):
    manager = TemporalInteractionCandidateManager()
    proposal, player_states, scene_graph, candidate = build_mock_confirmed_event_inputs()
    pair_factor_map = manager._pair_factor_map(scene_graph)
    line_factor_map = manager._line_factor_map(scene_graph)
    proposal_conf = manager._proposal_confidence(proposal, player_states, raider_id=4)
    factor_conf = manager._factor_confidence(proposal, pair_factor_map, line_factor_map)

    avg_conf = float(np.mean(candidate["confidences"]))
    avg_factor_conf = float(np.mean(candidate["factor_confidences"]))
    fused_conf = 0.6 * avg_conf + 0.4 * avg_factor_conf
    threshold = 0.58
    frame_count = len(candidate["frames"])
    confirmed = frame_count >= 2 and fused_conf >= threshold

    canvas = np.full((880, 1480, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Confirmed Event Confidence Visualization", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Standalone view of how TemporalInteractionCandidateManager confirms an HHI event",
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    x1, x2 = 28, 860
    y = 120
    blocks = [
        ("1. Proposal Confidence", [
            f"dist_conf = max(0, 1 - dist/1.5)  -> dist={proposal['features']['dist']:.2f}",
            f"vel_conf = min(1, rel_vel/2.0)    -> rel_vel={proposal['features']['rel_vel']:.2f}",
            f"track_conf avg = ({player_states[4]['track_confidence']:.2f} + {player_states[7]['track_confidence']:.2f}) / 2",
            "formula = 0.5*dist_conf + 0.25*vel_conf + 0.25*track_avg + role_boost",
            f"proposal_confidence = {proposal_conf:.3f}",
        ]),
        ("2. Factor Confidence", [
            f"pair distance = {scene_graph['pair_factors'][0]['features']['distance']:.2f}",
            f"relative_velocity = {scene_graph['pair_factors'][0]['features']['relative_velocity']:.2f}",
            f"approach_score = {scene_graph['pair_factors'][0]['features']['approach_score']:.2f}",
            f"adjacency = {scene_graph['pair_factors'][0]['features']['adjacency']:.2f}",
            "formula = 0.4*proximity + 0.2*rel_vel + 0.2*approach + 0.2*adjacency",
            f"factor_confidence = {factor_conf:.3f}",
        ]),
        ("3. Temporal Aggregation", [
            f"proposal confidences over frames = {candidate['confidences']}",
            f"factor confidences over frames   = {candidate['factor_confidences']}",
            f"avg_proposal_confidence = {avg_conf:.3f}",
            f"avg_factor_confidence   = {avg_factor_conf:.3f}",
            "fused_confidence = 0.6 * avg_proposal + 0.4 * avg_factor",
            f"fused_confidence = {fused_conf:.3f}",
        ]),
        ("4. Confirmation Rule", [
            "For HHI with raider as subject:",
            "confirm if frame_count >= 2 and fused_confidence >= 0.58",
            f"frame_count = {frame_count}",
            f"threshold = {threshold:.2f}",
            f"decision = {'CONFIRMED_RAIDER_DEFENDER_CONTACT' if confirmed else 'NOT CONFIRMED'}",
        ]),
    ]

    for title, rows in blocks:
        h = 150 if len(rows) <= 5 else 172
        cv2.rectangle(canvas, (x1, y), (x2, y + h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x1, y), (x2, y + h), (220, 220, 220), 2)
        cv2.putText(canvas, title, (x1 + 18, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
        yy = y + 58
        for row in rows:
            cv2.putText(canvas, row, (x1 + 18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
            yy += 22
        y += h + 18

    panel_x1, panel_y1, panel_x2, panel_y2 = 920, 120, 1430, 420
    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (220, 220, 220), 2)
    cv2.putText(canvas, "Fused Confidence Gauge", (1060, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2, cv2.LINE_AA)

    bar_x1, bar_y1, bar_x2, bar_y2 = 980, 230, 1380, 270
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (235, 235, 235), -1)
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (180, 180, 180), 1)
    threshold_x = bar_x1 + int(threshold * (bar_x2 - bar_x1))
    fill_x = bar_x1 + int(np.clip(fused_conf, 0.0, 1.0) * (bar_x2 - bar_x1))
    cv2.rectangle(canvas, (bar_x1, bar_y1), (fill_x, bar_y2), (0, 150, 255), -1)
    cv2.line(canvas, (threshold_x, bar_y1 - 18), (threshold_x, bar_y2 + 18), (0, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(canvas, "threshold", (threshold_x - 34, bar_y1 - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{fused_conf:.3f}", (fill_x - 18, bar_y2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)
    verdict_color = (0, 140, 60) if confirmed else (0, 0, 200)
    cv2.putText(
        canvas,
        "Confirmed" if confirmed else "Rejected",
        (1100, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        verdict_color,
        3,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), canvas)


def build_mock_touch_clip(num_frames=18, frame_size=(360, 640)):
    height, width = frame_size
    frames = []
    for idx in range(num_frames):
        frame = np.full((height, width, 3), 245, dtype=np.uint8)

        cv2.rectangle(frame, (0, 0), (width, int(height * 0.55)), (236, 242, 250), -1)
        cv2.rectangle(frame, (0, int(height * 0.55)), (width, height), (177, 205, 225), -1)
        cv2.line(frame, (0, int(height * 0.78)), (width, int(height * 0.78)), (255, 255, 255), 2, cv2.LINE_AA)

        raider_x = int(170 + idx * 10)
        defender_x = int(410 - idx * 4)
        raider_y = int(height * 0.68 + 8 * np.sin(idx * 0.55))
        defender_y = int(height * 0.68 + 6 * np.cos(idx * 0.4))
        overlap = abs(raider_x - defender_x) < 52

        raider_color = (60, 170, 255)
        defender_color = (150, 70, 255) if overlap else (90, 90, 90)

        draw_player_silhouette(frame, (raider_x, raider_y), 1.0, raider_color, lean=-0.18)
        draw_player_silhouette(frame, (defender_x, defender_y), 1.0, defender_color, lean=0.22)

        if overlap:
            cv2.circle(frame, (raider_x + 26, raider_y - 78), 14, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(frame, (raider_x + 8, raider_y - 34), (defender_x - 8, defender_y - 46), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "contact window", (width - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"frame {idx:02d}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(frame, "Raider", (raider_x - 28, raider_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(frame, "Defender", (defender_x - 40, defender_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 2, cv2.LINE_AA)
        frames.append(frame)
    return frames


def draw_player_silhouette(frame, foot, scale, color, lean=0.0):
    fx, fy = foot
    head_center = (int(fx + lean * 10 * scale), int(fy - 108 * scale))
    torso_top = (int(fx + lean * 8 * scale), int(fy - 86 * scale))
    torso_mid = (int(fx + lean * 16 * scale), int(fy - 52 * scale))
    hip = (int(fx + lean * 10 * scale), int(fy - 20 * scale))
    left_hand = (int(fx - 24 * scale + lean * 8 * scale), int(fy - 58 * scale))
    right_hand = (int(fx + 30 * scale + lean * 12 * scale), int(fy - 46 * scale))
    left_knee = (int(fx - 12 * scale), int(fy + 10 * scale))
    right_knee = (int(fx + 15 * scale), int(fy + 8 * scale))
    left_foot = (int(fx - 14 * scale), int(fy + 34 * scale))
    right_foot = (int(fx + 18 * scale), int(fy + 34 * scale))

    cv2.circle(frame, head_center, int(15 * scale), color, -1, cv2.LINE_AA)
    cv2.line(frame, head_center, torso_top, color, 7, cv2.LINE_AA)
    cv2.line(frame, torso_top, torso_mid, color, 10, cv2.LINE_AA)
    cv2.line(frame, torso_mid, hip, color, 9, cv2.LINE_AA)
    cv2.line(frame, torso_top, left_hand, color, 7, cv2.LINE_AA)
    cv2.line(frame, torso_top, right_hand, color, 7, cv2.LINE_AA)
    cv2.line(frame, hip, left_knee, color, 8, cv2.LINE_AA)
    cv2.line(frame, hip, right_knee, color, 8, cv2.LINE_AA)
    cv2.line(frame, left_knee, left_foot, color, 7, cv2.LINE_AA)
    cv2.line(frame, right_knee, right_foot, color, 7, cv2.LINE_AA)


def format_prob(prob):
    return f"{prob:.3f}"


def draw_touch_classifier_visualization(output_path):
    frames = build_mock_touch_clip()
    bridge = ConfirmedWindowClassifierBridge()
    event = {
        "type": "CONFIRMED_RAIDER_DEFENDER_CONTACT",
        "frame": 214,
        "subject": 4,
        "object": 7,
        "requires_visual_confirmation": True,
    }
    payload = {
        "aggregates": {
            "avg_proposal_confidence": 0.72,
            "avg_factor_confidence": 0.67,
            "peak_window_pair_score": 0.86,
            "peak_window_line_score": 0.18,
            "peak_window_containment": 0.12,
            "visible_defenders": 3,
        },
        "temporal_trace": [
            {"best_contact_score": 0.44, "best_containment_score": 0.11},
            {"best_contact_score": 0.58, "best_containment_score": 0.15},
            {"best_contact_score": 0.73, "best_containment_score": 0.18},
            {"best_contact_score": 0.89, "best_containment_score": 0.24},
        ],
        "graph_snapshot": {"global_context": {"raider_to_endline": 2.1}},
        "window_frames": list(range(len(frames))),
        "core_frames": list(range(5, 13)),
    }

    if bridge.touch_inference is not None:
        prediction = bridge.touch_inference.predict_frames([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
        sample_indices = bridge.touch_inference._sample_indices(len(frames))
        image_size = bridge.touch_inference.image_size
    else:
        prediction = {
            "predicted_label": "valid_touch",
            "probabilities": {"no_touch": 0.21, "valid_touch": 0.79},
        }
        sample_indices = np.linspace(0, len(frames) - 1, 12).round().astype(int).tolist()
        image_size = 224

    result = bridge.score_window({"event": event, "frames": frames, "payload": payload})

    canvas = np.full((980, 1600, 3), 247, dtype=np.uint8)
    cv2.putText(canvas, "Touch Visual-Classifier Confirmation", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "How CONFIRMED_RAIDER_DEFENDER_CONTACT gets a visual vote from the trained touch-classifier model",
        (28, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    tl_x1, tl_y1, tl_x2, tl_y2 = 28, 120, 1020, 430
    cv2.rectangle(canvas, (tl_x1, tl_y1), (tl_x2, tl_y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (tl_x1, tl_y1), (tl_x2, tl_y2), (220, 220, 220), 2)
    cv2.putText(canvas, "1. Input Window Sent To The Model", (tl_x1 + 18, tl_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(frames)} raw frames -> sampled down to {len(sample_indices)} frames", (tl_x1 + 18, tl_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 80, 80), 1, cv2.LINE_AA)

    thumb_w, thumb_h = 150, 84
    sx = tl_x1 + 18
    sy = tl_y1 + 88
    for idx, frame_idx in enumerate(sample_indices[:6]):
        thumb = cv2.resize(frames[frame_idx], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x = sx + idx * (thumb_w + 10)
        canvas[sy:sy + thumb_h, x:x + thumb_w] = thumb
        cv2.rectangle(canvas, (x, sy), (x + thumb_w, sy + thumb_h), (205, 205, 205), 1)
        cv2.putText(canvas, f"f{frame_idx}", (x + 8, sy + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
    sy2 = sy + thumb_h + 40
    for idx, frame_idx in enumerate(sample_indices[6:12]):
        thumb = cv2.resize(frames[frame_idx], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x = sx + idx * (thumb_w + 10)
        canvas[sy2:sy2 + thumb_h, x:x + thumb_w] = thumb
        cv2.rectangle(canvas, (x, sy2), (x + thumb_w, sy2 + thumb_h), (205, 205, 205), 1)
        cv2.putText(canvas, f"f{frame_idx}", (x + 8, sy2 + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)

    pipe_x1, pipe_y1, pipe_x2, pipe_y2 = 1050, 120, 1568, 430
    cv2.rectangle(canvas, (pipe_x1, pipe_y1), (pipe_x2, pipe_y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (pipe_x1, pipe_y1), (pipe_x2, pipe_y2), (220, 220, 220), 2)
    cv2.putText(canvas, "2. Preprocessing + Model Path", (pipe_x1 + 18, pipe_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)

    raw_preview = frames[sample_indices[len(sample_indices) // 2]]
    raw_small = cv2.resize(raw_preview, (160, 110), interpolation=cv2.INTER_AREA)
    resized_small = cv2.resize(cv2.cvtColor(raw_preview, cv2.COLOR_BGR2RGB), (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    resized_small = cv2.resize(resized_small, (110, 110), interpolation=cv2.INTER_AREA)
    norm_rgb = cv2.cvtColor(raw_preview, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm_rgb = (norm_rgb - IMAGENET_MEAN) / IMAGENET_STD
    norm_vis = ((np.clip(norm_rgb, -2.5, 2.5) + 2.5) / 5.0 * 255.0).astype(np.uint8)
    norm_vis = cv2.resize(norm_vis, (110, 110), interpolation=cv2.INTER_AREA)

    canvas[pipe_y1 + 72:pipe_y1 + 182, pipe_x1 + 18:pipe_x1 + 178] = raw_small
    canvas[pipe_y1 + 72:pipe_y1 + 182, pipe_x1 + 205:pipe_x1 + 315] = resized_small
    canvas[pipe_y1 + 72:pipe_y1 + 182, pipe_x1 + 352:pipe_x1 + 462] = norm_vis
    cv2.rectangle(canvas, (pipe_x1 + 18, pipe_y1 + 72), (pipe_x1 + 178, pipe_y1 + 182), (205, 205, 205), 1)
    cv2.rectangle(canvas, (pipe_x1 + 205, pipe_y1 + 72), (pipe_x1 + 315, pipe_y1 + 182), (205, 205, 205), 1)
    cv2.rectangle(canvas, (pipe_x1 + 352, pipe_y1 + 72), (pipe_x1 + 462, pipe_y1 + 182), (205, 205, 205), 1)
    cv2.putText(canvas, "RGB frame", (pipe_x1 + 52, pipe_y1 + 204), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"resize {image_size}x{image_size}", (pipe_x1 + 190, pipe_y1 + 204), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, "normalize", (pipe_x1 + 375, pipe_y1 + 204), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (pipe_x1 + 180, pipe_y1 + 126), (pipe_x1 + 198, pipe_y1 + 126), (110, 110, 110), 2, cv2.LINE_AA, tipLength=0.28)
    cv2.arrowedLine(canvas, (pipe_x1 + 327, pipe_y1 + 126), (pipe_x1 + 345, pipe_y1 + 126), (110, 110, 110), 2, cv2.LINE_AA, tipLength=0.28)

    steps = [
        "sample_indices = np.linspace(...) or pad to num_frames",
        "clip tensor shape = [T, C, H, W]",
        "per-frame encoder = ResNet18 backbone",
        "temporal average pooling across T frames",
        "MLP head -> logits -> softmax over [no_touch, valid_touch]",
    ]
    y = pipe_y1 + 248
    for row in steps:
        cv2.putText(canvas, row, (pipe_x1 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (70, 70, 70), 1, cv2.LINE_AA)
        y += 24

    score_x1, score_y1, score_x2, score_y2 = 28, 468, 760, 920
    cv2.rectangle(canvas, (score_x1, score_y1), (score_x2, score_y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (score_x1, score_y1), (score_x2, score_y2), (220, 220, 220), 2)
    cv2.putText(canvas, "3. Model Output And Bridge Mapping", (score_x1 + 18, score_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)

    raw_probs = prediction["probabilities"]
    bridge_probs = result["probabilities"]
    bars = [
        ("no_touch", float(raw_probs.get("no_touch", 0.0)), (70, 70, 70)),
        ("valid_touch", float(raw_probs.get("valid_touch", 0.0)), (0, 145, 255)),
        ("bridge.invalid", float(bridge_probs.get("invalid", 0.0)), (0, 80, 200)),
        ("bridge.valid", float(bridge_probs.get("valid", 0.0)), (0, 165, 80)),
        ("bridge.uncertain", float(bridge_probs.get("uncertain", 0.0)), (180, 0, 200)),
    ]
    bar_x1, bar_x2 = score_x1 + 190, score_x2 - 32
    y = score_y1 + 92
    for label, value, color in bars:
        cv2.putText(canvas, label, (score_x1 + 22, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (55, 55, 55), 2 if "valid" in label else 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x1, y - 16), (bar_x2, y + 10), (240, 240, 240), -1)
        cv2.rectangle(canvas, (bar_x1, y - 16), (bar_x2, y + 10), (210, 210, 210), 1)
        fill_x = bar_x1 + int(np.clip(value, 0.0, 1.0) * (bar_x2 - bar_x1))
        cv2.rectangle(canvas, (bar_x1, y - 16), (fill_x, y + 10), color, -1)
        cv2.putText(canvas, format_prob(value), (bar_x2 - 72, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 2, cv2.LINE_AA)
        y += 56

    mapping_rows = [
        f"raw predicted_label = {prediction['predicted_label']}",
        "bridge mapping: valid_touch -> valid, no_touch -> invalid",
        "uncertain = 1 - max(valid_touch, no_touch)",
        f"bridge predicted_label = {result['predicted_label']}",
        f"guaranteed flag = {result['guaranteed']}",
        f"model_name = {result['model_name']}",
    ]
    y += 4
    for row in mapping_rows:
        cv2.putText(canvas, row, (score_x1 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 70, 70), 1, cv2.LINE_AA)
        y += 28

    decision_x1, decision_y1, decision_x2, decision_y2 = 790, 468, 1568, 920
    cv2.rectangle(canvas, (decision_x1, decision_y1), (decision_x2, decision_y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (decision_x1, decision_y1), (decision_x2, decision_y2), (220, 220, 220), 2)
    cv2.putText(canvas, "4. Final Visual Confirmation Decision", (decision_x1 + 18, decision_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)

    explanation_rows = [
        "The temporal pipeline proposes a candidate contact window first.",
        "Only then the visual model sees the clip and votes from appearance + motion.",
        "If valid >= invalid, the bridge returns predicted_label = valid.",
        "If valid >= 0.82 as well, the bridge marks it as guaranteed.",
        "This keeps visual confirmation separate from the court-logic proposal stage.",
    ]
    y = decision_y1 + 86
    for row in explanation_rows:
        cv2.putText(canvas, row, (decision_x1 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
        y += 28

    result_color = (0, 160, 80) if result["predicted_label"] == "valid" else (0, 0, 200)
    certainty = "Guaranteed Visual Touch" if result["guaranteed"] else ("Visual Touch Supported" if result["predicted_label"] == "valid" else "Visual Touch Rejected")
    cv2.rectangle(canvas, (decision_x1 + 22, decision_y1 + 250), (decision_x2 - 22, decision_y1 + 390), (245, 245, 245), -1)
    cv2.rectangle(canvas, (decision_x1 + 22, decision_y1 + 250), (decision_x2 - 22, decision_y1 + 390), (220, 220, 220), 2)
    cv2.putText(canvas, certainty, (decision_x1 + 48, decision_y1 + 310), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Event: {event['type']}   Subject: ID {event['subject']}   Object: ID {event['object']}",
        (decision_x1 + 48, decision_y1 + 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (55, 55, 55),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Decision rule used in bridge: valid >= invalid   |   guaranteed: valid >= 0.82",
        (decision_x1 + 48, decision_y1 + 384),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), canvas)


def main():
    rng = np.random.default_rng(SEED)
    kf = create_kalman(1.0, 0.5)
    gt_path = []
    meas_path = []
    pred_path = []
    est_path = []
    last_frame = None

    writer = cv2.VideoWriter(
        str(MP4_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (CANVAS_W, CANVAS_H),
    )

    for frame_idx in range(FRAME_COUNT):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 245, dtype=np.uint8)
        cv2.putText(
            canvas,
            "Kalman Filter Visualization Using tracking_pipeline.create_kalman()",
            (24, CANVAS_H - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )

        gt = build_ground_truth(frame_idx)
        measurement = noisy_measurement(gt, rng)
        missed = frame_idx % 37 in (12, 13, 14)

        prediction = kf.predict().flatten()
        pred_xy = np.array([prediction[0], prediction[1]], dtype=np.float32)

        if not missed:
            kf.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))
            meas_path.append(world_to_canvas(measurement))

        state = kf.statePost.flatten()
        estimate = np.array([state[0], state[1]], dtype=np.float32)
        velocity = np.array([state[2], state[3]], dtype=np.float32)

        gt_path.append(world_to_canvas(gt))
        pred_path.append(world_to_canvas(pred_xy))
        est_path.append(world_to_canvas(estimate))

        draw_legend(canvas)
        draw_path(canvas, gt_path, (0, 200, 0), 2)
        draw_path(canvas, meas_path, (0, 140, 255), 1)
        draw_path(canvas, pred_path, (210, 210, 210), 1)
        draw_path(canvas, est_path, (255, 0, 255), 3)

        draw_point(canvas, gt_path[-1], (0, 200, 0), 6)
        draw_point(canvas, pred_path[-1], (210, 210, 210), 5)
        draw_point(canvas, est_path[-1], (255, 0, 255), 7)
        if not missed:
            draw_point(canvas, meas_path[-1], (0, 140, 255), 5)

        est_xy = est_path[-1]
        vel_tip = (int(est_xy[0] + velocity[0] * 18), int(est_xy[1] - velocity[1] * 90))
        cv2.arrowedLine(canvas, est_xy, vel_tip, (120, 0, 180), 2, cv2.LINE_AA, tipLength=0.18)

        draw_state_panel(canvas, frame_idx, gt, measurement, pred_xy, estimate, velocity, missed)

        if missed:
            cv2.putText(
                canvas,
                "Measurement dropped -> prediction only",
                (24, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 200),
                2,
                cv2.LINE_AA,
            )

        writer.write(canvas)
        last_frame = canvas

    writer.release()

    if last_frame is not None:
        cv2.imwrite(str(PNG_PATH), last_frame)

    gallery = make_mock_gallery()
    draw_gallery_visualization(gallery, GALLERY_PNG_PATH)
    draw_empty_kabaddi_mat(MAT_PNG_PATH)
    draw_raider_score_visualization(RAIDER_PNG_PATH)
    draw_interaction_triplets_visualization(INTERACTION_PNG_PATH)
    draw_confirmed_event_confidence_visualization(CONF_EVENT_PNG_PATH)
    draw_touch_classifier_visualization(TOUCH_VIS_PNG_PATH)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved MP4: {MP4_PATH}")
    print(f"Saved gallery PNG: {GALLERY_PNG_PATH}")
    print(f"Saved mat PNG: {MAT_PNG_PATH}")
    print(f"Saved raider PNG: {RAIDER_PNG_PATH}")
    print(f"Saved interaction PNG: {INTERACTION_PNG_PATH}")
    print(f"Saved confirmed-event PNG: {CONF_EVENT_PNG_PATH}")
    print(f"Saved touch-classifier PNG: {TOUCH_VIS_PNG_PATH}")


if __name__ == "__main__":
    main()
