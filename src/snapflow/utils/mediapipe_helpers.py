"""MediaPipe helper functions"""

import numpy as np
import math


# MediaPipe Face Mesh landmark indices
LEFT_EYE_LANDMARKS = [
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    7,
]
RIGHT_EYE_LANDMARKS = [
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
]

# Simplified eye landmarks for EAR calculation
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # 6 points for EAR
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # 6 points for EAR

# Eye anchor points for scaling
LEFT_EYE_ANCHOR = 133
RIGHT_EYE_ANCHOR = 362


def extract_eye_landmarks(landmarks, image_width, image_height, use_full_eye=True):
    """
    Extract eye landmark points from MediaPipe results

    Args:
        landmarks: MediaPipe face landmarks
        image_width: Image width in pixels
        image_height: Image height in pixels
        use_full_eye: If True, use full eye contour; if False, use 6-point EAR landmarks

    Returns:
        tuple: (left_eye_points, right_eye_points) as numpy arrays
    """
    if use_full_eye:
        left_indices = LEFT_EYE_LANDMARKS
        right_indices = RIGHT_EYE_LANDMARKS
    else:
        left_indices = LEFT_EYE_EAR
        right_indices = RIGHT_EYE_EAR

    def extract_points(indices):
        return np.array(
            [
                [int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)]
                for i in indices
            ],
            dtype=np.int32,
        )

    left_eye_pts = extract_points(left_indices)
    right_eye_pts = extract_points(right_indices)

    return left_eye_pts, right_eye_pts


def extract_eye_anchors(landmarks, image_width, image_height):
    """Extract eye anchor points for scaling effects"""
    left_anchor = np.array(
        [
            int(landmarks[LEFT_EYE_ANCHOR].x * image_width),
            int(landmarks[LEFT_EYE_ANCHOR].y * image_height),
        ]
    )
    right_anchor = np.array(
        [
            int(landmarks[RIGHT_EYE_ANCHOR].x * image_width),
            int(landmarks[RIGHT_EYE_ANCHOR].y * image_height),
        ]
    )

    return left_anchor, right_anchor


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) from 6 landmark points

    Args:
        eye_points: Array of 6 (x, y) points in order: [outer_corner, top1, top2, inner_corner, bottom2, bottom1]

    Returns:
        float: EAR value (typically 0.2-0.4 for open eyes, <0.2 for closed)
    """
    if len(eye_points) != 6:
        return 0.0

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # Vertical distances
    vertical_1 = euclidean_distance(eye_points[1], eye_points[5])  # top1 to bottom1
    vertical_2 = euclidean_distance(eye_points[2], eye_points[4])  # top2 to bottom2

    # Horizontal distance
    horizontal = euclidean_distance(
        eye_points[0], eye_points[3]
    )  # outer to inner corner

    if horizontal == 0:
        return 0.0

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def normalize_landmarks(landmarks, image_width, image_height):
    """Convert normalized MediaPipe landmarks to pixel coordinates"""
    return np.array(
        [
            [int(landmark.x * image_width), int(landmark.y * image_height)]
            for landmark in landmarks
        ],
        dtype=np.int32,
    )
