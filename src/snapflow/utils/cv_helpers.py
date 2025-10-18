"""OpenCV helper functions"""

import cv2
import numpy as np


def bgr_to_rgb(img):
    """Convert BGR image to RGB"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    """Convert RGB image to BGR"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_landmarks(image, points, color=(0, 255, 0), radius=2):
    """Draw landmark points on image"""
    for pt in points:
        cv2.circle(image, tuple(pt), radius, color, -1)
    return image


def create_eye_mask(image_shape, eye_points):
    """Create mask for eye region"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(eye_points) > 0:
        eye_hull = cv2.convexHull(eye_points)
        cv2.fillConvexPoly(mask, eye_hull, 255)
    return mask
