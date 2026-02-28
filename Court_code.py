import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from threading import Thread
from queue import Queue
import time
import torch
import os
import hashlib

# ======================================================
# VIDEO THREADING
# ======================================================

VIDEO_PATH = "Videos/raid1.mp4"

class VideoStream:
    def __init__(self, path, queue_size=5):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update)
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.001)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

    def running(self):
        return not self.stopped or not self.queue.empty()

# ======================================================
# CONFIG
# ======================================================

DISPLAY_SCALE = 0.5
CONF_THRESH = 0.4
MAX_PLAYERS = 8
MAX_AGE = 200
SMOOTH_ALPHA = 0.25
LINE_MARGIN = 0.6
MODEL_PATH = "yolo26m.pt"

BAULK_Y = 3.75
ASSIGN_FRAME = 70
MIN_FRAMES_FOR_DECISION = 40

# ======================================================
# COURT + HOMOGRAPHY
# ======================================================

lines = {
    "baulk": [(606, 483), (1771, 1078)],
    "bonus": [(745, 471), (1918, 960)],
    "middle": [(55, 486), (0, 575)],
    "end_back": [(885, 471), (1918, 763)],
    "end_left": [(58, 490), (885, 473)],
    "end_right": [(1833, 1076), (1916, 1041)],
}

def line_eq(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return np.array([y2-y1, x1-x2, x2*y1-x1*y2], dtype=np.float64)

def intersect(l1,l2):
    a1,b1,c1 = l1; a2,b2,c2 = l2
    d = a1*b2 - a2*b1
    if abs(d)<1e-6: return None
    return [(b1*c2-b2*c1)/d, (a2*c1-a1*c2)/d]

L = {k: line_eq(*v) for k,v in lines.items()}
img_pts = np.array([intersect(L[a],L[b]) for a,b in [
    ("end_back","end_left"),
    ("end_back","end_right"),
    ("middle","end_left"),
    ("middle","end_right")
]], dtype=np.float32)

court_pts = np.array([[0,6.5],[10,6.5],[0,0],[10,0]], dtype=np.float32)
H,_ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)

COURT_W,COURT_H = 400,260
mat_base = np.ones((COURT_H,COURT_W,3),dtype=np.uint8)*235

def court_to_pixel(x,y):
    px=int(x/10*COURT_W)
    py=int((6.5-y)/6.5*COURT_H)
    return px,py

cv2.rectangle(mat_base,court_to_pixel(0,0),
              court_to_pixel(10,6.5),(0,0,0),2)

# ======================================================
# TRACKING HELPERS
# ======================================================

def create_kalman(x,y):
    kf=cv2.KalmanFilter(4,2)
    kf.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kf.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kf.processNoiseCov=np.eye(4,dtype=np.float32)*0.03
    kf.measurementNoiseCov=np.eye(2,dtype=np.float32)*0.5
    kf.errorCovPost=np.eye(4,dtype=np.float32)
    kf.statePost=np.array([[x],[y],[0],[0]],np.float32)
    return kf

def extract_embedding(frame,box):
    x1,y1,x2,y2=box
    crop=frame[max(0,y1):y2,max(0,x1):x2]
    if crop.size==0: return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6)

# ======================================================
# INIT
# ======================================================

device="cuda" if torch.cuda.is_available() else "cpu"
print("Device:",device)
model=YOLO(MODEL_PATH).to(device)

vs=VideoStream(VIDEO_PATH).start()
NEXT_ID=0
GALLERY={}
frame_idx=0
prev_gray=None

# RAIDER VARIABLES
RAIDER_ID=None
RAID_ASSIGNMENT_DONE=False
RAIDER_CONV_ACCUM={}
RAIDER_STATS={}

# ======================================================
# MAIN LOOP
# ======================================================

while vs.running():

    frame=vs.read()
    if frame is None: continue
    frame_idx+=1
    vis=frame.copy()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mat=mat_base.copy()

    # DETECTION
    results=model(frame, device=device, verbose=False)[0]
    detections=[]
    for box in results.boxes:
        if int(box.cls[0])!=0 or float(box.conf[0])<CONF_THRESH: continue
        x1,y1,x2,y2=map(int,box.xyxy[0])
        emb=extract_embedding(frame,(x1,y1,x2,y2))
        if emb is not None:
            detections.append({"bbox":(x1,y1,x2,y2),
                               "foot":((x1+x2)//2,y2),
                               "emb":emb})

    # TRACK MATCHING
    track_ids=list(GALLERY.keys())
    predictions=[GALLERY[pid]["kf"].predict() for pid in track_ids]

    matched_tracks=set()
    matched_dets=set()

    if predictions and detections:
        cost=np.zeros((len(predictions),len(detections)))

        for i,pred in enumerate(predictions):
            px,py=pred[0][0],pred[1][0]
            for j,det in enumerate(detections):
                fx,fy=det["foot"]
                cost[i,j]=np.sqrt((px-fx)**2+(py-fy)**2)/200 \
                          +0.3*(1-cosine(det["emb"],GALLERY[track_ids[i]]["feat"]))

        r_ind,c_ind=linear_sum_assignment(cost)

        for r,c in zip(r_ind,c_ind):
            if cost[r,c]<1.2:
                pid=track_ids[r]
                det=detections[c]

                GALLERY[pid]["kf"].correct(
                    np.array([[np.float32(det["foot"][0])],
                              [np.float32(det["foot"][1])]])
                )

                GALLERY[pid]["feat"]=0.8*GALLERY[pid]["feat"]+0.2*det["emb"]
                GALLERY[pid]["age"]=0
                GALLERY[pid]["last_bbox"]=det["bbox"]

                matched_tracks.add(pid)
                matched_dets.add(c)

    # NEW TRACKS
    for j,det in enumerate(detections):
        if j not in matched_dets and len(GALLERY)<MAX_PLAYERS:
            GALLERY[NEXT_ID]={
                "feat":det["emb"],
                "kf":create_kalman(*det["foot"]),
                "age":0,
                "display_pos":None,
                "last_bbox":det["bbox"]
            }
            NEXT_ID+=1

    # AGE
    for pid in list(GALLERY.keys()):
        if pid not in matched_tracks:
            GALLERY[pid]["age"]+=1
        if GALLERY[pid]["age"]>MAX_AGE:
            del GALLERY[pid]

    # COURT PROJECTION + SMOOTHING
    active=[]
    for pid,data in GALLERY.items():

        state=data["kf"].statePost.flatten()
        px,py=state[0],state[1]

        mapped=cv2.perspectiveTransform(
            np.array([[[px,py]]],dtype=np.float32),H)[0][0]

        cx,cy=mapped

        if data["display_pos"] is None:
            data["display_pos"]=(cx,cy)
        else:
            ox,oy=data["display_pos"]
            data["display_pos"]=(ox+SMOOTH_ALPHA*(cx-ox),
                                 oy+SMOOTH_ALPHA*(cy-oy))

        if data["age"]==0:
            active.append(pid)

    # ======================================================
    # RAIDER IDENTIFICATION
    # ======================================================

    if not RAID_ASSIGNMENT_DONE:

        # remove defenders behind baulk initially
        filtered=[]
        for pid in active:
            cx,cy=GALLERY[pid]["display_pos"]
            if frame_idx<60 and cy>BAULK_Y:
                continue
            filtered.append(pid)

        # convergence model (defenders moving toward target)
        for i in filtered:
            xi,yi=GALLERY[i]["display_pos"]
            if i not in RAIDER_CONV_ACCUM:
                RAIDER_CONV_ACCUM[i]=0

            for j in filtered:
                if i==j: continue
                xj,yj=GALLERY[j]["display_pos"]
                vj=GALLERY[j]["kf"].statePost.flatten()[2:4]

                dx,dy=xi-xj,yi-yj
                dist=np.sqrt(dx*dx+dy*dy)+1e-6
                dir_vec=np.array([dx/dist,dy/dist])

                approach=np.dot(vj,dir_vec)

                if approach>0:
                    RAIDER_CONV_ACCUM[i]+=approach

        if frame_idx>=ASSIGN_FRAME:

            best_id=None
            best_score=-1

            for pid,score in RAIDER_CONV_ACCUM.items():
                if score>best_score:
                    best_score=score
                    best_id=pid

            RAIDER_ID=best_id
            RAID_ASSIGNMENT_DONE=True
            print("🔥 RAIDER:",RAIDER_ID)

    # ======================================================
    # RENDERING
    # ======================================================

    for pid,data in GALLERY.items():

        state=data["kf"].statePost.flatten()
        px,py=state[0],state[1]

        cx,cy=data["display_pos"]
        mx,my=court_to_pixel(cx,cy)

        cv2.circle(mat,(mx,my),6,(255,0,0),-1)

        if RAID_ASSIGNMENT_DONE and pid==RAIDER_ID:
            cv2.circle(mat,(mx,my),12,(0,0,255),2)

        x1,y1,x2,y2=data["last_bbox"]
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(vis,f"ID {pid}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

        if RAID_ASSIGNMENT_DONE and pid==RAIDER_ID:
            cv2.putText(vis,"RAIDER",
                        (x1+100,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

    # ======================================================
    # INTERACTION (ONLY RAIDER)
    # ======================================================

    if RAID_ASSIGNMENT_DONE and RAIDER_ID in GALLERY:

        r_pos=GALLERY[RAIDER_ID]["display_pos"]

        for pid,data in GALLERY.items():
            if pid==RAIDER_ID: continue

            d=np.linalg.norm(
                np.array(r_pos)-np.array(data["display_pos"])
            )

            if d<1.0:
                mx1,my1=court_to_pixel(*r_pos)
                mx2,my2=court_to_pixel(*data["display_pos"])
                cv2.line(mat,(mx1,my1),(mx2,my2),(0,0,255),2)

    cv2.imshow("Video (Integrated)",
               cv2.resize(vis,None,fx=DISPLAY_SCALE,fy=DISPLAY_SCALE))
    cv2.imshow("Half Court (2D)",mat)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cv2.destroyAllWindows()
