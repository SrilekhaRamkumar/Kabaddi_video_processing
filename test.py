import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================
# CONFIGURATION
# ======================================================
VIDEO_PATH = "Videos/raid1.mp4"
MODEL_PATH = "yolo26m.pt"
CONF_THRESH = 0.4

# Boundary Margins (in meters)
SIDE_MARGIN = 1# Ignore if within 0.5m of left/right lobby
END_MARGIN = 0.5   # Ignore if within 0.5m of the back end-line

# ======================================================
# HOMOGRAPHY SETUP (Using your project's logic)
# ======================================================
def get_homography_matrix():
    # Corner points from your provided image-space coordinates
    img_pts = np.array([
        [885, 471],   # Back-Left corner
        [1918, 763],  # Back-Right corner
        [58, 490],    # Mid-Left corner
        [0, 575]      # Mid-Right corner
    ], dtype=np.float32)

    # Real world coordinates (Width 10m, Depth 6.5m)
    # Midline is y=0, End-line is y=6.5
    court_pts = np.array([[0, 6.5], [10, 6.5], [0, 0], [10, 0]], dtype=np.float32)
    
    H, _ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)
    return H

# ======================================================
# FILTERED RAIDER ANALYSIS
# ======================================================
def identify_raider_with_margins(frame, model, H):
    results = model(frame, verbose=False)[0]
    output_img = frame.copy()
    raider_count = 0

    for box in results.boxes:
        if int(box.cls[0]) != 0: continue 
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        foot_pt = np.array([[[ (x1 + x2) / 2, y2 ]]], dtype=np.float32)
        
        # Map to court meters
        court_pt = cv2.perspectiveTransform(foot_pt, H)[0][0]
        cx, cy = court_pt 

        # --- REFINED SPATIAL FILTER ---
        
        # 1. Check if player is "Inside" the playable area (accounting for margins)
        is_not_on_sideline = (SIDE_MARGIN <= cx <= 10 - SIDE_MARGIN)
        is_not_on_endline = (cy <= 6.5 - END_MARGIN)
        
        # 2. Check if player is near the Midline (Potential Raider zone)
        is_near_midline = (-0.2 <= cy <= 1.2)

        if is_not_on_sideline and is_not_on_endline and is_near_midline:
            raider_count += 1
            # Highlight as Raider (Gold)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 215, 255), 4)
            cv2.putText(output_img, "RAIDER CANDIDATE", (x1, y1-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        else:
            # Highlight as Defender or Out-of-Bounds (Blue)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = "DEFENDER" if is_not_on_sideline and is_not_on_endline else "OUT/BORDER"
            cv2.putText(output_img, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return output_img, raider_count

# ======================================================
# EXECUTION
# ======================================================
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    H_matrix = get_homography_matrix()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, frame = cap.read()
    cap.release()

    if success:
        result_img, count = identify_raider_with_margins(frame, model, H_matrix)
        print(f"Analysis Complete: Found {count} player(s) in valid raider starting positions.")
        
        display_img = cv2.resize(result_img, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Filtered First Frame Analysis", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()