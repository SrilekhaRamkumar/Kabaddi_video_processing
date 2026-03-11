"""
Geometry Utilities Module
Handles homography, line equations, and coordinate transformations.
"""
import cv2
import numpy as np


def line_eq(p1, p2):
    """Calculate line equation coefficients from two points."""
    x1, y1 = p1
    x2, y2 = p2
    return np.array([y2 - y1, x1 - x2, x2*y1 - x1*y2], dtype=np.float64)


def intersect(l1, l2):
    """Find intersection point of two lines."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1*b2 - a2*b1
    if abs(d) < 1e-6:
        return None
    return [(b1*c2 - b2*c1) / d, (a2*c1 - a1*c2) / d]


def compute_homography(lines):
    """
    Compute homography matrix from court lines.
    
    Args:
        lines: Dictionary of court line endpoints
        
    Returns:
        Homography matrix H
    """
    L = {k: line_eq(*v) for k, v in lines.items()}
    img_pts = np.array([intersect(L[a], L[b]) for a, b in [
        ("end_back", "end_left"), ("end_back", "end_right"),
        ("middle", "end_left"), ("middle", "end_right")
    ]], dtype=np.float32)
    
    court_pts = np.array([[0, 6.5], [10, 6.5], [0, 0], [10, 0]], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)
    return H


def create_court_mat(court_w=400, court_h=260):
    """
    Create a blank court mat with lines and labels.
    
    Args:
        court_w: Court width in pixels
        court_h: Court height in pixels
        
    Returns:
        Court mat image
    """
    mat_base = np.ones((court_h, court_w, 3), dtype=np.uint8) * 235
    
    def court_to_pixel(x, y):
        px = int(x / 10 * court_w)
        py = int((6.5 - y) / 6.5 * court_h)
        return px, py
    
    # Draw court boundary
    cv2.rectangle(mat_base, court_to_pixel(0, 0), court_to_pixel(10, 6.5), (0, 0, 0), 2)
    
    # Draw baulk and bonus lines
    for y, name in [(3.75, "baulk"), (4.75, "bonus")]:
        cv2.line(mat_base, court_to_pixel(0, y), court_to_pixel(10, y), (0, 0, 0), 1)
        cv2.putText(mat_base, name, (8, court_to_pixel(0, y)[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    
    # Draw lobby lines
    for x in [0.75, 9.25]:
        cv2.line(mat_base, court_to_pixel(x, 0), court_to_pixel(x, 6.5), (0, 0, 0), 1)
    
    return mat_base, court_to_pixel


def court_to_pixel(x, y, court_w=400, court_h=260):
    """Convert court coordinates to pixel coordinates."""
    px = int(x / 10 * court_w)
    py = int((6.5 - y) / 6.5 * court_h)
    return px, py

