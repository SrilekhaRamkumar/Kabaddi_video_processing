# tracking_utils.py

import cv2
import numpy as np

def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
    return kf

def extract_embedding(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def cosine(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-6)

def draw_3d_bbox(img,x1,y1,x2,y2,depth=12):
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1+depth,y1-depth),(x2+depth,y2-depth),(0,255,0),2)
    for p in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]:
        cv2.line(img,p,(p[0]+depth,p[1]-depth),(0,255,0),2)
