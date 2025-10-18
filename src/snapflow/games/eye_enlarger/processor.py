"""Eye enlarger effect processor"""

import cv2
import numpy as np
import mediapipe as mp
from ...core.processor import Processor
from ...core.registry import register_game
from ...core.store import NoOpStore


class EyeEnlargerProcessor(Processor):
    """Processor that enlarges eyes using triangle-based warping"""

    def __init__(self, scale_factor=1.4):
        """
        Initialize eye enlarger

        Args:
            scale_factor: Factor by which to enlarge eyes (1.0 = no change, >1.0 = larger)
        """
        self.scale_factor = scale_factor
        self.mp_face = None

        # MediaPipe Face Mesh landmark indices for eyes
        self.LEFT_EYE_LANDMARKS = [
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
        self.RIGHT_EYE_LANDMARKS = [
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
        self.LEFT_EYE_ANCHOR = 133
        self.RIGHT_EYE_ANCHOR = 362

    def open(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame, state):
        """
        Process frame to enlarge eyes

        Args:
            frame: Frame dict with 'image', 'frame_id', 'timestamp'
            state: Shared state dict

        Returns:
            Result dict with processed image
        """
        if not self.mp_face:
            return {"image": frame["image"]}

        img = frame["image"]
        h, w = img.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)

        img_out = img.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract eye landmarks and anchors
            left_eye, left_anchor, right_eye, right_anchor = self._get_eye_landmarks(
                landmarks, w, h
            )

            if left_eye is not None and right_eye is not None:
                # Apply scaling to both eyes using triangle-based warping
                img_out = self._scale_eye_region(
                    img_out, left_eye, left_anchor, self.scale_factor
                )
                img_out = self._scale_eye_region(
                    img_out, right_eye, right_anchor, self.scale_factor
                )

        return {"image": img_out}

    def _get_eye_landmarks(self, landmarks, w, h):
        """Extract eye landmarks and anchor points"""

        def extract_points(indices):
            return np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
                dtype=np.int32,
            )

        left_eye_pts = extract_points(self.LEFT_EYE_LANDMARKS)
        right_eye_pts = extract_points(self.RIGHT_EYE_LANDMARKS)
        left_anchor = np.array(
            [
                int(landmarks[self.LEFT_EYE_ANCHOR].x * w),
                int(landmarks[self.LEFT_EYE_ANCHOR].y * h),
            ]
        )
        right_anchor = np.array(
            [
                int(landmarks[self.RIGHT_EYE_ANCHOR].x * w),
                int(landmarks[self.RIGHT_EYE_ANCHOR].y * h),
            ]
        )

        return left_eye_pts, left_anchor, right_eye_pts, right_anchor

    def _scale_eye_region(self, image, eye_points, anchor, scale_factor):
        """
        Scale eye region using triangle-by-triangle warping (proper implementation)

        Args:
            image: Input image
            eye_points: Eye landmark points
            anchor: Anchor point for scaling
            scale_factor: Scaling factor

        Returns:
            Image with scaled eye region
        """
        if len(eye_points) == 0:
            return image

        # Create mask for eye region
        mask = np.zeros_like(image)
        eye_hull = cv2.convexHull(eye_points)
        cv2.fillConvexPoly(mask, eye_hull, (255, 255, 255))

        # Create scaled points around anchor
        scaled_points = []
        for pt in eye_points:
            vector = pt - anchor
            scaled = anchor + (vector * scale_factor)
            scaled_points.append(scaled)
        scaled_points = np.array(scaled_points, dtype=np.int32)

        # Triangle-by-triangle warping (this is the key fix!)
        warped = image.copy()

        for i in range(len(eye_points)):
            # Create triangles: current point, anchor, next point
            src_triangle = np.array(
                [eye_points[i], anchor, eye_points[(i + 1) % len(eye_points)]],
                dtype=np.float32,
            )

            dst_triangle = np.array(
                [scaled_points[i], anchor, scaled_points[(i + 1) % len(eye_points)]],
                dtype=np.float32,
            )

            # Get affine transformation matrix for this triangle
            mat = cv2.getAffineTransform(src_triangle, dst_triangle)

            # Warp the entire image using this transformation
            triangle_warped = cv2.warpAffine(
                image, mat, (image.shape[1], image.shape[0])
            )

            # Create mask for this specific triangle
            triangle_mask = np.zeros_like(image)
            cv2.fillConvexPoly(
                triangle_mask, dst_triangle.astype(np.int32), (255, 255, 255)
            )

            # Replace pixels where triangle mask is white
            warped = np.where(triangle_mask == 255, triangle_warped, warped)

        return warped

    def close(self):
        """Clean up MediaPipe resources"""
        if self.mp_face:
            self.mp_face.close()


# Register the eye enlarger effect
@register_game("eye_enlarger")
def make_eye_enlarger():
    """Factory function to create eye enlarger effect"""
    processors = [EyeEnlargerProcessor(scale_factor=1.4)]
    store = NoOpStore()  # No metrics needed for this effect
    return processors, store


@register_game("eye_enlarger_extreme")
def make_eye_enlarger_extreme():
    """Factory function to create extreme eye enlarger effect"""
    processors = [EyeEnlargerProcessor(scale_factor=2.0)]
    store = NoOpStore()
    return processors, store
