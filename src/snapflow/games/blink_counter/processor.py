"""Blink counter game processor"""

import cv2
import numpy as np
import mediapipe as mp
from ...core.processor import Processor
from ...core.registry import register_game
from ...core.store import SimpleStore
from ...utils.mediapipe_helpers import extract_eye_landmarks, calculate_ear


class BlinkGameProcessor(Processor):
    """Processor that counts blinks using Eye Aspect Ratio (EAR)"""

    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        """
        Initialize blink detector

        Args:
            ear_threshold: EAR threshold below which eye is considered closed
            consecutive_frames: Number of consecutive frames to confirm blink
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        # Blink detection state
        self.frame_counter = 0
        self.total_blinks = 0
        self.last_blink_frame = -999

        # MediaPipe setup
        self.mp_face = None

    def open(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame, state):
        """
        Process frame to detect blinks and update score

        Args:
            frame: Frame dict with 'image', 'frame_id', 'timestamp'
            state: Shared state dict

        Returns:
            Result dict with processed image, events, and metrics
        """
        if not self.mp_face:
            return {"image": frame["image"]}

        img = frame["image"]
        h, w = img.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)

        events = []
        metrics = {}
        img_out = img.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract eye landmarks (6 points each for EAR calculation)
            left_eye, right_eye = extract_eye_landmarks(
                landmarks, w, h, use_full_eye=False
            )

            # Calculate Eye Aspect Ratio for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            metrics["ear"] = avg_ear
            metrics["left_ear"] = left_ear
            metrics["right_ear"] = right_ear

            # Blink detection logic
            if avg_ear < self.ear_threshold:
                self.frame_counter += 1
            else:
                # Eye opened - check if we had enough consecutive closed frames
                if self.frame_counter >= self.consecutive_frames:
                    # Avoid double counting - ensure cooldown between blinks
                    if frame["frame_id"] - self.last_blink_frame > 5:
                        events.append("blink")
                        self.total_blinks += 1
                        self.last_blink_frame = frame["frame_id"]
                self.frame_counter = 0

            # Draw eye landmarks for visualization
            for pt in left_eye:
                cv2.circle(img_out, tuple(pt), 2, (0, 255, 0), -1)
            for pt in right_eye:
                cv2.circle(img_out, tuple(pt), 2, (0, 255, 0), -1)

        # Draw UI elements
        cv2.putText(
            img_out,
            f"Blinks: {self.total_blinks}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        if metrics.get("ear"):
            cv2.putText(
                img_out,
                f'EAR: {metrics["ear"]:.3f}',
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Show status
            status = "CLOSED" if metrics["ear"] < self.ear_threshold else "OPEN"
            color = (0, 0, 255) if status == "CLOSED" else (0, 255, 0)
            cv2.putText(
                img_out,
                f"Eyes: {status}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        return {
            "image": img_out,
            "events": events,
            "metrics": metrics,
            "state": {"score": self.total_blinks, "blinks": self.total_blinks},
        }

    def close(self):
        """Clean up MediaPipe resources"""
        if self.mp_face:
            self.mp_face.close()


# Register the blink counter game
@register_game("blink_counter")
def make_blink_game():
    """Factory function to create blink counter game components"""
    processors = [BlinkGameProcessor()]
    store = SimpleStore()
    return processors, store


@register_game("blink_counter_hard")
def make_blink_game_hard():
    """Factory function to create harder blink counter game"""
    processors = [BlinkGameProcessor(ear_threshold=0.20, consecutive_frames=2)]
    store = SimpleStore()
    return processors, store
