import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import os
import hashlib
from classifier_bridge import ConfirmedWindowClassifierBridge
from dataset_exporter import ConfirmedWindowDatasetExporter
from kabaddi_afgn_reasoning import KabaddiAFGNEngine
from interaction_graph import InteractionProposalEngine, ActiveFactorGraphNetwork, render_graph_panel
from interaction_logic import build_player_states, process_interactions
from raider_logic import collect_raider_stats, assign_raider
from report_video import ConfirmedInteractionReportBuilder
from temporal_events import TemporalInteractionCandidateManager
from tracking_pipeline import apply_optical_flow, run_yolo_detection, update_tracks, add_new_tracks, render_gallery
from video_stream import VideoStream

#COMMIT CHECK 12/3
# ======================================================
# PERFORMANCE: THREADED VIDEO READER
# ======================================================

VIDEO_PATH = "Videos/Cam1/raid1.mp4"

# ======================================================
# CONFIG (RESTORED)
# ======================================================

DISPLAY_SCALE = 0.5
FPS_DELAY = 1
CONF_THRESH = 0.4
SMOOTH_ALPHA = 0.25 
MAX_PLAYERS = 8
MAX_AGE=200
MODEL1 = "models/yolov8n.pt"
cursor_court_pos = None
LINE_MARGIN = 0.6 
proposal_engine = InteractionProposalEngine()
graph_engine = ActiveFactorGraphNetwork(top_k=4)
action_engine = KabaddiAFGNEngine()
candidate_manager = TemporalInteractionCandidateManager()
report_builder = ConfirmedInteractionReportBuilder(max_buffer_frames=300)
classifier_bridge = ConfirmedWindowClassifierBridge()
dataset_exporter = ConfirmedWindowDatasetExporter(os.path.join("Videos", "classifier_dataset"), fps=30.0)

# IMAGE-SPACE COURT LINES
lines = {
    "baulk": [(606, 483), (1771, 1078)],
    "bonus": [(745, 471), (1918, 960)],
    "middle": [(55, 486), (0, 575)],
    "end_back": [(885, 471), (1918, 763)],
    "end_left": [(58, 490), (885, 473)],
    "end_right": [(1833, 1076), (1916, 1041)],
    "lobby_left": [(45, 525), (921, 493)],
    "lobby_right": [(690, 1076), (1915, 821)],
}

lines2 = {
    "baulk": [(606, 291), (378, 128)],
    "bonus": [(678, 269), (427, 123)],
    "middle": [(215, 410), (213, 144)],
    "end_back": [(847, 268), (462, 112)],
    "end_left": [(215, 132), (462, 114)],
    "end_right": [(224, 512), (847, 268)],
    "lobby_left": [(213, 143), (477, 120)],
    "lobby_right": [(223, 407), (773, 237)],
}


def select_line_coordinates(video_path):
    normalized_path = video_path.replace("\\", "/").lower()
    if "/cam2/" in normalized_path:
        return lines2, "Cam2"
    return lines, "Cam1"

# ======================================================
# MATH & HOMOGRAPHY
# ======================================================
def line_eq(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.array([y2 - y1, x1 - x2, x2*y1 - x1*y2], dtype=np.float64)

def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1*b2 - a2*b1
    if abs(d) < 1e-6: return None
    return [(b1*c2 - b2*c1) / d, (a2*c1 - a1*c2) / d]

ACTIVE_LINES, ACTIVE_CAMERA = select_line_coordinates(VIDEO_PATH)
print(f"Using {ACTIVE_CAMERA} court coordinates for {os.path.basename(VIDEO_PATH)}")

L = {k: line_eq(*v) for k, v in ACTIVE_LINES.items()}
img_pts = np.array([intersect(L[a], L[b]) for a, b in [
    ("end_back", "end_left"), ("end_back", "end_right"),
    ("middle", "end_left"), ("middle", "end_right")
]], dtype=np.float32)

court_pts = np.array([[0, 6.5], [10, 6.5], [0, 0], [10, 0]], dtype=np.float32)
H, _ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)

COURT_W, COURT_H = 400, 260
GRAPH_W, GRAPH_H = 420, 320
mat_base = np.ones((COURT_H, COURT_W, 3), dtype=np.uint8) * 235

def court_to_pixel(x, y):
    px = int(x / 10 * COURT_W)
    py = int((6.5 - y) / 6.5 * COURT_H)
    return px, py

# Draw static mat labels/lines
cv2.rectangle(mat_base, court_to_pixel(0, 0), court_to_pixel(10, 6.5), (0, 0, 0), 2)
for y, name in [(3.75, "baulk"), (4.75, "bonus")]:
    cv2.line(mat_base, court_to_pixel(0, y), court_to_pixel(10, y), (0, 0, 0), 1)
    cv2.putText(mat_base, name, (8, court_to_pixel(0, y)[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
for x in [0.75, 9.25]:
    cv2.line(mat_base, court_to_pixel(x, 0), court_to_pixel(x, 6.5), (0, 0, 0), 1)

# ======================================================
# MAIN LOOP
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    MODEL1="models/yolov8n.pt"
print("Device used: ",device)
model = YOLO(MODEL1).to(device)
# model = RTDETR("rtdetr-l.pt").to(device)

vs = VideoStream(VIDEO_PATH).start()
prev_gray = None
NEXT_ID = 0
GALLERY = {}
frame_idx = 0

# ======================================================
# RAIDER IDENTIFICATION (Single-side logic)
# ======================================================
RAIDER_ID = None
RAIDER_STATS = {}
RAID_ASSIGNMENT_DONE = False
RAIDER_EXITED = False
RAIDER_MISSING_FRAMES = 0
RAIDER_CONV_ACCUM = {}
MIN_FRAMES_FOR_DECISION = 40

BAULK_Y = 3.75
ASSIGN_FRAME = 70   # assign raider after this frame

# Cursor Tracking Logic
cursor_court_pos = None

def mouse_tracker(event, x, y, flags, param):
    global cursor_court_pos
    if event == cv2.EVENT_MOUSEMOVE:
        # Map scaled window coordinates back to original video size
        orig_x, orig_y = x / DISPLAY_SCALE, y / DISPLAY_SCALE
        pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        cursor_court_pos = (mapped[0], mapped[1])

cv2.namedWindow("Video (Integrated)")
cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
cv2.namedWindow("Half Court (2D)")



cv2.setMouseCallback("Video (Integrated)", mouse_tracker)
# ---------------------------------------


path_hash = hashlib.md5(VIDEO_PATH.encode()).hexdigest()[:8]
video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
output_filename = f"Videos/processed_{video_stem}_{path_hash}.mp4"
report_output_filename = f"Videos/confirmed_report_{video_stem}_{path_hash}.mp4"


vis_w = int(1920 * DISPLAY_SCALE)
vis_h = int(1080 * DISPLAY_SCALE)
canvas_w = vis_w + COURT_W + GRAPH_W
canvas_h = max(vis_h, COURT_H, GRAPH_H)

if os.path.exists(output_filename):
    os.remove(output_filename)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (canvas_w, canvas_h))
print(f"Recording to: {output_filename}")

def mouse_tracker(event, x, y, flags, param):
    global cursor_court_pos

    if event == cv2.EVENT_MOUSEMOVE:

        # Convert displayed coordinates back to original frame scale
        orig_x = x / DISPLAY_SCALE
        orig_y = y / DISPLAY_SCALE

        pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)

        mapped = cv2.perspectiveTransform(pt, H)

        cursor_court_pos = (mapped[0][0][0], mapped[0][0][1])

def log_event(event_type, player_id, frame_idx):
    EVENT_LOG.append({
        "frame": frame_idx,
        "player": player_id,
        "type": event_type,
        "investigation_until": frame_idx + INTERACTION_WINDOW
    })


def log_confirmed_event(event):
    CONFIRMED_EVENT_LOG.append(event)


def apply_classifier_results(confirmed_log, classifier_results):
    if not classifier_results:
        return
    result_map = {result["event_key"]: result for result in classifier_results}
    for event in confirmed_log:
        event_key = (event["type"], event["frame"], event["subject"], event["object"])
        classifier_result = result_map.get(event_key)
        if classifier_result is None:
            continue
        event["classifier_result"] = classifier_result
        event["classifier_valid_prob"] = classifier_result["probabilities"].get("valid", 0.0)
        event["classifier_label"] = classifier_result["predicted_label"]
        event["guaranteed_by_classifier"] = classifier_result["guaranteed"]

# ======================================================
# PHASE 2: ACTION RECOGNITION & SCORING
# ======================================================

# Event and interaction tracking (restored)
EVENT_LOG = []
CONFIRMED_EVENT_LOG = []
touch_confirmed = False
INTERACTION_WINDOW = 20  # frames for future investigation
INTERACTION_COUNT = 0

MIDDLE_Y = 0.0
BAULK_Y = 3.75
BONUS_Y = 4.75
END_LINE_Y = 6.5

LINE_MARGIN = 0.25
LOBBY_LEFT = 0.75
LOBBY_RIGHT = 9.25

paused = False
last_vis = None
last_mat = None

while vs.running():
    global TOTAL_POINTS, CURRENT_RAID_POINTS
    if 'TOTAL_POINTS' not in globals():
        TOTAL_POINTS = 0
        CURRENT_RAID_POINTS = 0
    if not paused:
        frame = vs.read()
        if frame is None: continue
        frame_idx += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    mat = mat_base.copy()
    

    if cursor_court_pos is not None:
        cx, cy = cursor_court_pos

        # Only draw if inside court bounds
        if -1 < cx < 11 and -1 < cy < 7.5:
            mx, my = court_to_pixel(cx, cy)

            cv2.circle(mat, (mx, my), 7, (0, 255, 255), -1)
            cv2.circle(mat, (mx, my), 9, (0, 0, 0), 1)

    # Display current scores on mat
    cv2.putText(mat, f"Total Points: {TOTAL_POINTS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(mat, f"Current Raid: {CURRENT_RAID_POINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 1. OPTICAL FLOW MOTION COMPENSATION
    if prev_gray is not None:
        apply_optical_flow(prev_gray, gray, GALLERY)

    # 2. YOLO DETECTION
    detections = run_yolo_detection(model, frame, device, CONF_THRESH)

    # 3. TRACK PREDICTION & MATCHING
    matched_tracks, matched_dets = update_tracks(
        GALLERY,
        detections,
        gray,
        vis,
        frame_idx,
        RAID_ASSIGNMENT_DONE,
        RAIDER_ID,
    )

    # 4. NEW TRACKS & AGING
    NEXT_ID = add_new_tracks(GALLERY, detections, matched_dets, NEXT_ID, MAX_PLAYERS)

    # 5. MAT RENDERING & DIRECTION ARROWS
    render_gallery(GALLERY, matched_tracks, vis, mat, H, court_to_pixel, LINE_MARGIN, SMOOTH_ALPHA, RAIDER_ID, MAX_AGE)

    if RAID_ASSIGNMENT_DONE and RAIDER_ID is not None:
        raider_track = GALLERY.get(RAIDER_ID)
        raider_missing = (
            raider_track is None
            or raider_track.get("age", 0) > 15
            or (
                raider_track.get("display_pos") is None
                and raider_track.get("miss_streak", 0) > 10
            )
        )
        if raider_missing:
            RAIDER_MISSING_FRAMES += 1
        else:
            RAIDER_MISSING_FRAMES = 0

        if RAIDER_MISSING_FRAMES >= 25 and not RAIDER_EXITED:
            RAIDER_EXITED = True
            print(f"RAIDER EXIT LOCK ENGAGED at frame {frame_idx} for ID {RAIDER_ID}")

        

    # ======================================================
    # RAIDER STATS COLLECTION (FIRST PHASE)
    # ======================================================

    if not RAID_ASSIGNMENT_DONE and not RAIDER_EXITED and frame_idx < ASSIGN_FRAME:
        collect_raider_stats(GALLERY, RAIDER_STATS, frame_idx, BAULK_Y)



   
    # =====================================================
    # REVISED RAIDER ASSIGNMENT (Safety-first logic)
    # ======================================================

    if not RAID_ASSIGNMENT_DONE and not RAIDER_EXITED and frame_idx >= ASSIGN_FRAME:
        best_id, raid_done, ASSIGN_FRAME = assign_raider(
            GALLERY,
            RAIDER_STATS,
            frame_idx,
            ASSIGN_FRAME,
            BAULK_Y,
            BONUS_Y,
        )
        if raid_done:
            RAIDER_ID = best_id
            RAID_ASSIGNMENT_DONE = True
            print(f"RAIDER IDENTIFIED (Defense-Half Model): {RAIDER_ID}")

    # ======================================================
    # MODULE-2: INTERACTION LOGIC & PROPOSAL LAYER
    # ======================================================
   
    
    player_states, active_players = build_player_states(GALLERY)
    effective_raid_assignment = RAID_ASSIGNMENT_DONE and not RAIDER_EXITED
    interaction_candidates, touch_confirmed = process_interactions(
        frame_idx,
        GALLERY,
        player_states,
        active_players,
        RAIDER_ID,
        effective_raid_assignment,
        proposal_engine,
        BONUS_Y,
        BAULK_Y,
        END_LINE_Y,
        LINE_MARGIN,
        LOBBY_LEFT,
        LOBBY_RIGHT,
        touch_confirmed,
        log_event,
    )

    # ------------------------------------------------------
    # EVENT WINDOW CLEANUP / FUTURE INVESTIGATION
    # ------------------------------------------------------

    for event in EVENT_LOG:
        if frame_idx <= event["investigation_until"]:
            # This event is still active for analysis
            pass


    # Optional Debug Visualization
    for cand in interaction_candidates:
        p1, p2 = cand["pair"]

        # SAFETY CHECK: Only draw if both players are currently "Active"
        if p1 in player_states and p2 in player_states:
            mx1, my1 = court_to_pixel(*player_states[p1]["pos"])
            mx2, my2 = court_to_pixel(*player_states[p2]["pos"])

            cv2.line(mat, (mx1, my1), (mx2, my2), (0, 0, 255), 2)

    prev_gray = gray.copy()

    # --- DEBUG BLOCK: Print EVERYTHING in the Gallery ---
    # --- DETAILED DEBUG: Every Parameter + First 5 Color Values ---
     
    
        
    frame_proposals = proposal_engine.finalize_frame_proposals()
    scene_graph = None
    if effective_raid_assignment and proposal_engine.candidate_proposals:
        scene_graph = graph_engine.build_graph(
            proposal_engine.candidate_proposals,
            GALLERY,
            RAIDER_ID
        )
    confirmed_events = candidate_manager.update(
        frame_idx,
        frame_proposals,
        player_states,
        RAIDER_ID,
        scene_graph,
    )
    for confirmed_event in confirmed_events:
        log_confirmed_event(confirmed_event)
    
    if frame_idx % 30 == 0:
        print(f"\n" + "█"*80)
        print(f" LOG FOR FRAME {frame_idx:05d} | TOTAL INTERACTION PROPOSALS: {len(proposal_engine.candidate_proposals)}")
        print("█"*80)
        
       

        # 3. Existing Gallery Debug (Condensed)
        print(f"\n [GALLERY] ACTIVE TRACKS: {len(GALLERY)}")
        for pid, data in GALLERY.items():
            # 1. Physics & State
            state = data["kf"].statePost.flatten() # [x, y, vx, vy]
            
            # 2. Extract first 5 color values from the 512-dim histogram
            color_sample = data["feat"][:5] 
            
            # 3. Court Coordinates
            m_pos = data["display_pos"] if data["display_pos"] else (0.0, 0.0)
            
            # 4. Bounding Box Dimensions
            bb = data["last_bbox"]
            w, h = (bb[2]-bb[0]), (bb[3]-bb[1])
            x=data['age']
            print(f"PLAYER ID: {pid:02d} | Status: {'[VISIBLE]' if data['age']==0 else f'[LOST (Age:{x})]'}")
            print(f" ├─ PHYSICS: Screen_Pos({state[0]:.0f}, {state[1]:.0f}) | Velocity({state[2]:.2f}, {state[3]:.2f})")
            print(f" ├─ COURT:   Width: {m_pos[0]:.2f}m, Depth: {m_pos[1]:.2f}m")
            print(f" ├─ COLORS:  First 5 of 512 bins: {color_sample}") 
            print(f" ├─ CAMERA:  Optical Flow: {len(data['flow_pts']) if data['flow_pts'] is not None else 0} tracking points")
            print(f" └─ VISUAL:  BBox Height: {h}px | BBox Width: {w}px")
            print("-" * 70)

        # 1. Print Human-Human Interaction (HHI) Triplets
        # ======================================================
        # FORMAL INTERACTION TRIPLET PRINTING <S, I, O>
        # ======================================================
    
        # 1. Print Human-Human Interaction (HHI) Triplets
        hhi_proposals = [p for p in proposal_engine.candidate_proposals if p["type"] == "HHI"]
        if hhi_proposals:
            print(f" [HHI] PLAYER-TO-PLAYER TRIPLETS:")
            for p in hhi_proposals:
                # Replace IDs with "RAIDER" if they match
                sub_label = "RAIDER" if p['S'] == RAIDER_ID else f"ID_{p['S']}"
                obj_label = "RAIDER" if p['O'] == RAIDER_ID else f"ID_{p['O']}"

                INTERACTION_COUNT+=1
                
                # Format: [Frame] <Subject, Interaction, Object>
                triplet = f"<{sub_label}, {p['I']}, {obj_label}>"
                # Change this line inside your HHI loop:
                print(f"  ├─ Frame {p['frame']:05d} | {triplet:30} | Rel_Vel: {p['features']['rel_vel']:.2f} | Dist: {p['features']['dist']:.2f}m")

        # 2. Print Human-Line Interaction (HLI) Triplets
        hli_proposals = [p for p in proposal_engine.candidate_proposals if p["type"] == "HLI" and p["features"]["dist"] < 0.5]
        if hli_proposals:
            print(f"\n [HLI] PLAYER-TO-LINE TRIPLETS (Active Proximity):")
            for p in hli_proposals:
                # Replace Subject ID with "RAIDER" if it matches
                sub_label = "RAIDER" if p['S'] == RAIDER_ID else f"ID_{p['S']}"
                INTERACTION_COUNT+=1
                # Format: [Frame] <Subject, Interaction, Object>
                triplet = f"<{sub_label}, {p['I']}, {p['O']}>"
                status = "[TOUCHING]" if p["features"]["active"] else "[NEAR]"
                print(f"  ├─ Frame {p['frame']:05d} | {triplet:30} | Status: {status:10} | Dist: {p['features']['dist']:.2f}m")


        recent_confirmed = [event for event in CONFIRMED_EVENT_LOG if frame_idx - event["frame"] <= 30]
        if recent_confirmed:
            print(f"\n [CONFIRMED EVENTS] TEMPORAL WINDOW OUTPUT:")
            for event in recent_confirmed:
                review_tag = " | Visual Review Window" if event["requires_visual_confirmation"] else ""
                print(
                    f"  â”œâ”€ Frame {event['frame']:05d} | {event['type']} "
                    f"| Window: {event['window_start']:05d}-{event['window_end']:05d} "
                    f"| Conf: {event['confidence']:.2f}"
                    f"| Factor: {event.get('factor_confidence', 0):.2f}{review_tag}"
                )

        if effective_raid_assignment and proposal_engine.candidate_proposals:
            # Build the Graph and Encode Features
            if scene_graph is None:
                scene_graph = graph_engine.build_graph(
                    proposal_engine.candidate_proposals,
                    GALLERY,
                    RAIDER_ID
                )
            
            print(f"\n [GRAPH] DYNAMIC INTERACTION GRAPH CONSTRUCTED")
            print(f"  ├─ Active Nodes: {graph_engine.active_nodes}")
            print(f"  └─ Perspective Adjacency Matrix (Top 2x2 Sample):")

            if graph_engine.adjacency_matrix.size > 0:
                print(f"     {graph_engine.adjacency_matrix[:2, :2]}")

            # PHASE 2: Action Recognition and Point Calculation
            action_results = action_engine.process_frame_actions(
                scene_graph, 
                proposal_engine.candidate_proposals, 
                CONFIRMED_EVENT_LOG,
                RAIDER_ID, 
                frame_idx,
                GALLERY,
            )
            
            # Update global scores
            total_score = action_results['total_points']
            TOTAL_POINTS = total_score['attacker'] - total_score['defender']
            CURRENT_RAID_POINTS += action_results['points_scored']
            
            if action_results.get('raid_ended', False):
                CURRENT_RAID_POINTS = 0  # Reset for new raid
            
            print(f"\n [ACTIONS] RECOGNIZED ACTIONS THIS FRAME:")
            for action in action_results["actions"]:
                conf = action.get("confidence", 0)
                print(f"  ├─ {action['type']}: {action['description']} | Points: {action.get('points', 0)} | Conf: {conf:.2f}")
            print(f"  └─ Frame Points: {action_results['points_scored']} | Total Points: {action_results['total_points']}")
            print(
                f"     Scoreboard A/D: {total_score['attacker']}/{total_score['defender']} "
                f"| Net: {TOTAL_POINTS}"
            )
            if action_results.get('confidence_scores'):
                avg_conf = np.mean(action_results['confidence_scores']) if action_results['confidence_scores'] else 0
                print(f"     Average Confidence: {avg_conf:.2f}")

            # Display accuracy metrics every 100 frames
            if frame_idx % 100 == 0 and 'accuracy_metrics' in action_results:
                metrics = action_results['accuracy_metrics']
                print(f"\n [ACCURACY] Estimated: {metrics['estimated_accuracy']:.1%} | High Conf: {metrics['high_confidence_rate']:.1%} | Total Actions: {metrics['total_actions']}")

        proposal_engine.reset_proposals()

    recent_graph_events = [event for event in CONFIRMED_EVENT_LOG if frame_idx - event["frame"] <= 15]
    graph_panel = render_graph_panel(
        scene_graph,
        width=GRAPH_W,
        height=GRAPH_H,
        frame_idx=frame_idx,
        recent_events=recent_graph_events,
    )
  
    # Display current scores on mat
    cv2.putText(mat, f"Total Points: {TOTAL_POINTS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(mat, f"Current Raid: {CURRENT_RAID_POINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Add frame ID to video
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    vis_render = cv2.resize(vis, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    combined_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    combined_frame[:vis_h, :vis_w] = vis_render
    combined_frame[:COURT_H, vis_w:vis_w+COURT_W] = mat
    combined_frame[:GRAPH_H, vis_w+COURT_W:vis_w+COURT_W+GRAPH_W] = graph_panel

    report_builder.add_frame(frame_idx, combined_frame)
    report_builder.capture_events(confirmed_events)
    if report_builder.has_classifier_inputs():
        classifier_inputs = report_builder.consume_classifier_inputs()
        exported_windows = dataset_exporter.export_batch(classifier_inputs)
        classifier_results = classifier_bridge.score_batch(classifier_inputs)
        apply_classifier_results(CONFIRMED_EVENT_LOG, classifier_results)
        if exported_windows:
            print(f"[DATASET] Exported {len(exported_windows)} confirmed window(s) to Videos/classifier_dataset")

   
    
    if out is not None:
        out.write(combined_frame)
    cv2.imshow("Video (Integrated)", cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Half Court (2D)", mat)
    cv2.imshow("Interaction Graph", graph_panel)
    
    key = cv2.waitKey(0 if paused else FPS_DELAY)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):
        paused = not paused

print("Total number of Interactions: ", INTERACTION_COUNT)
if out is not None:
    out.release()
if report_builder.has_segments():
    if os.path.exists(report_output_filename):
        os.remove(report_output_filename)
    wrote_report = report_builder.write_video(report_output_filename, 30.0, (canvas_w, canvas_h))
    if wrote_report:
        print(f"Confirmed interaction report saved to: {report_output_filename}")
    else:
        print("Confirmed interaction report could not be written.")
else:
    print("No confirmed interaction windows captured for report video.")
cv2.destroyAllWindows()
