"""Ghost faces effect processor - shows past expressions as ghosts"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from ...core.processor import Processor
from ...core.registry import register_game
from ...core.store import NoOpStore


class FaceGhost:
    """Represents a ghost face from the past"""

    def __init__(self, landmarks, timestamp, frame_id, image_width, image_height):
        self.landmarks = landmarks
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.image_width = image_width
        self.image_height = image_height

        # Convert landmarks to pixel coordinates
        self.pixel_landmarks = self._convert_to_pixels()

    def _convert_to_pixels(self):
        """Convert normalized landmarks to pixel coordinates"""
        pixels = []
        for landmark in self.landmarks:
            x = int(landmark.x * self.image_width)
            y = int(landmark.y * self.image_height)
            pixels.append((x, y))
        return pixels

    def get_age_seconds(self, current_time):
        """Get how old this ghost is in seconds"""
        return current_time - self.timestamp

    def get_opacity_and_color(self, current_time, max_age=3.0):
        """Get opacity and color based on age"""
        age = self.get_age_seconds(current_time)
        if age > max_age:
            return 0, (128, 128, 128)  # Too old, invisible

        # Calculate opacity (1.0 = newest, 0.0 = oldest)
        opacity = 1.0 - (age / max_age)

        # Color gradient based on age
        if age < 0.5:
            # Recent: bright green
            color = (0, 255, 0)
        elif age < 1.0:
            # Medium: blue
            color = (255, 100, 0)
        elif age < 1.5:
            # Older: purple
            color = (255, 0, 255)
        elif age < 2.0:
            # Very old: red
            color = (0, 0, 255)
        else:
            # Ancient: yellow
            color = (0, 255, 255)

        return opacity, color


class GhostFacesProcessor(Processor):
    """Processor that shows past facial expressions as ghosts"""

    def __init__(self, max_ghosts=50, ghost_lifetime=3.0, capture_interval=3):
        """
        Initialize ghost faces effect

        Args:
            max_ghosts: Maximum number of ghost faces to keep
            ghost_lifetime: How long ghosts survive (seconds)
            capture_interval: Frames between ghost captures (lower = more ghosts)
        """
        self.max_ghosts = max_ghosts
        self.ghost_lifetime = ghost_lifetime
        self.capture_interval = capture_interval

        # Storage for ghost faces
        self.ghost_faces = deque(maxlen=max_ghosts)
        self.last_capture_frame = 0

        # MediaPipe setup
        self.mp_face = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        # Face mesh drawing specifications
        self.face_connections = None

        # Visual settings
        self.show_current_landmarks = True
        self.show_ghost_connections = True
        self.show_face_outline_only = False

    def open(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

    def process(self, frame, state):
        """
        Process frame to show ghost faces effect

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
        current_time = time.time()

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)

        # Create output image
        img_out = img.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Capture new ghost if enough frames have passed
            if frame["frame_id"] - self.last_capture_frame >= self.capture_interval:
                ghost = FaceGhost(landmarks, current_time, frame["frame_id"], w, h)
                self.ghost_faces.append(ghost)
                self.last_capture_frame = frame["frame_id"]

            # Draw current face landmarks (if enabled)
            if self.show_current_landmarks:
                self._draw_current_face(img_out, results.multi_face_landmarks[0], w, h)

        # Clean up old ghosts and draw remaining ones
        self._cleanup_old_ghosts(current_time)
        self._draw_ghost_faces(img_out, current_time)

        # Draw UI
        self._draw_ui(img_out, current_time)

        return {"image": img_out}

    def _draw_current_face(self, image, face_landmarks, width, height):
        """Draw current face landmarks"""
        # Draw face mesh connections
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.face_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

        # Draw key landmark points
        key_landmarks = [
            # Eyes
            33,
            133,
            160,
            158,
            144,
            153,  # Left eye
            362,
            263,
            387,
            385,
            373,
            380,  # Right eye
            # Nose
            1,
            2,
            5,
            4,
            6,
            19,
            94,
            125,
            141,
            235,
            236,
            3,
            51,
            48,
            115,
            131,
            134,
            102,
            49,
            220,
            305,
            292,
            281,
            284,
            275,
            # Mouth
            61,
            84,
            17,
            314,
            405,
            320,
            307,
            375,
            321,
            308,
            324,
            318,
            # Face outline
            10,
            151,
            9,
            8,
            139,
            71,
            68,
            54,
            103,
            67,
            109,
            10,
        ]

        for idx in key_landmarks:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1)

    def _cleanup_old_ghosts(self, current_time):
        """Remove ghosts that are too old"""
        # Remove from front of deque while they're too old
        while (
            self.ghost_faces
            and self.ghost_faces[0].get_age_seconds(current_time) > self.ghost_lifetime
        ):
            self.ghost_faces.popleft()

    def _draw_ghost_faces(self, image, current_time):
        """Draw all ghost faces with appropriate opacity and colors"""
        for ghost in self.ghost_faces:
            opacity, color = ghost.get_opacity_and_color(
                current_time, self.ghost_lifetime
            )

            if opacity > 0.1:  # Only draw visible ghosts
                self._draw_single_ghost(image, ghost, opacity, color)

    def _draw_single_ghost(self, image, ghost, opacity, color):
        """Draw a single ghost face"""
        # Create overlay for blending
        overlay = image.copy()

        if self.show_face_outline_only:
            # Draw only face outline for cleaner look
            face_outline_indices = [
                10,
                338,
                297,
                332,
                284,
                251,
                389,
                356,
                454,
                323,
                361,
                288,
                397,
                365,
                379,
                378,
                400,
                377,
                152,
                148,
                176,
                149,
                150,
                136,
                172,
                58,
                132,
                93,
                234,
                127,
                162,
                21,
                54,
                103,
                67,
                109,
            ]

            outline_points = []
            for idx in face_outline_indices:
                if idx < len(ghost.pixel_landmarks):
                    outline_points.append(ghost.pixel_landmarks[idx])

            if len(outline_points) > 2:
                outline_points = np.array(outline_points, dtype=np.int32)
                cv2.polylines(overlay, [outline_points], True, color, 2)

        else:
            # Draw key facial feature points
            key_indices = [
                # Eyes
                33,
                133,
                160,
                158,
                144,
                153,  # Left eye
                362,
                263,
                387,
                385,
                373,
                380,  # Right eye
                # Eyebrows
                70,
                63,
                105,
                66,
                107,
                55,
                65,
                52,
                53,
                46,  # Left eyebrow
                285,
                295,
                334,
                296,
                336,
                285,
                295,
                282,
                283,
                276,  # Right eyebrow
                # Nose
                1,
                2,
                5,
                4,
                6,
                19,
                94,
                125,
                141,
                235,
                236,
                3,
                51,
                48,
                115,
                131,
                134,
                102,
                49,
                220,
                305,
                292,
                281,
                284,
                275,
                # Mouth
                61,
                84,
                17,
                314,
                405,
                320,
                307,
                375,
                321,
                308,
                324,
                318,
                78,
                95,
                88,
                178,
                87,
                14,
                317,
                402,
                318,
                324,
                # Chin
                18,
                175,
                199,
                428,
                262,
                369,
                396,
                175,
                199,
            ]

            # Draw points
            for idx in key_indices:
                if idx < len(ghost.pixel_landmarks):
                    point = ghost.pixel_landmarks[idx]
                    cv2.circle(overlay, point, 2, color, -1)

            # Draw connections for major features
            self._draw_ghost_connections(overlay, ghost, color)

        # Blend overlay with main image
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    def _draw_ghost_connections(self, overlay, ghost, color):
        """Draw connections between ghost landmark points"""
        # Eye connections
        left_eye = [33, 160, 158, 133, 153, 144, 33]
        right_eye = [362, 385, 387, 263, 373, 380, 362]

        # Mouth outline
        mouth_outer = [
            61,
            84,
            17,
            314,
            405,
            320,
            307,
            375,
            321,
            308,
            324,
            318,
            78,
            95,
            88,
            178,
            87,
            14,
            317,
            402,
            318,
            324,
            308,
            61,
        ]

        # Nose outline
        nose = [
            1,
            2,
            5,
            4,
            6,
            168,
            8,
            9,
            10,
            151,
            195,
            197,
            196,
            3,
            51,
            48,
            115,
            131,
            134,
            102,
            49,
            220,
            305,
            292,
            281,
            284,
            275,
            1,
        ]

        connection_groups = [left_eye, right_eye, mouth_outer, nose]

        for group in connection_groups:
            for i in range(len(group) - 1):
                if group[i] < len(ghost.pixel_landmarks) and group[i + 1] < len(
                    ghost.pixel_landmarks
                ):
                    pt1 = ghost.pixel_landmarks[group[i]]
                    pt2 = ghost.pixel_landmarks[group[i + 1]]
                    cv2.line(overlay, pt1, pt2, color, 1)

    def _draw_ui(self, image, current_time):
        """Draw user interface elements"""
        # Ghost count and info
        cv2.putText(
            image,
            f"Ghost Faces: {len(self.ghost_faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Age legend
        legend_y = 60
        legend_colors = [
            (0.5, (0, 255, 0), "0.5s ago"),
            (1.0, (255, 100, 0), "1.0s ago"),
            (1.5, (255, 0, 255), "1.5s ago"),
            (2.0, (0, 0, 255), "2.0s ago"),
            (2.5, (0, 255, 255), "2.5s ago"),
        ]

        cv2.putText(
            image,
            "Ghost Timeline:",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        for i, (age, color, label) in enumerate(legend_colors):
            y_pos = legend_y + 20 + (i * 15)
            cv2.circle(image, (20, y_pos), 5, color, -1)
            cv2.putText(
                image,
                label,
                (35, y_pos + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # Instructions
        instructions = [
            "Your past expressions appear as colored ghosts",
            "Recent = Green, Older = Blue/Purple/Red/Yellow",
            "Ghosts fade away after 3 seconds",
        ]

        start_y = image.shape[0] - 60
        for i, instruction in enumerate(instructions):
            cv2.putText(
                image,
                instruction,
                (10, start_y + i * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    def close(self):
        """Clean up MediaPipe resources"""
        if self.mp_face:
            self.mp_face.close()


@register_game("ghost_faces")
def make_ghost_faces():
    """Factory function to create ghost faces effect"""
    processors = [
        GhostFacesProcessor(max_ghosts=50, ghost_lifetime=3.0, capture_interval=3)
    ]
    store = NoOpStore()
    return processors, store


@register_game("ghost_faces_dense")
def make_ghost_faces_dense():
    """Factory function to create dense ghost faces effect"""
    processors = [
        GhostFacesProcessor(max_ghosts=100, ghost_lifetime=5.0, capture_interval=1)
    ]
    store = NoOpStore()
    return processors, store


@register_game("ghost_faces_sparse")
def make_ghost_faces_sparse():
    """Factory function to create sparse ghost faces effect"""
    processors = [
        GhostFacesProcessor(max_ghosts=20, ghost_lifetime=2.0, capture_interval=10)
    ]
    store = NoOpStore()
    return processors, store


@register_game("ghost_faces_outline")
def make_ghost_faces_outline():
    """Factory function to create outline-only ghost faces effect"""
    processor = GhostFacesProcessor(
        max_ghosts=30, ghost_lifetime=3.0, capture_interval=5
    )
    processor.show_face_outline_only = True
    processor.show_current_landmarks = False
    processors = [processor]
    store = NoOpStore()
    return processors, store
