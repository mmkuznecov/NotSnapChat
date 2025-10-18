"""Head tilt cannon game processor"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
from ...core.processor import Processor
from ...core.registry import register_game
from ...core.store import SimpleStore
from ...utils.mediapipe_helpers import calculate_ear


class GameObject:
    """Base class for game objects"""

    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.active = True

    def update(self):
        """Update object state"""
        pass

    def draw(self, image):
        """Draw object on image"""
        if self.active:
            cv2.circle(image, (int(self.x), int(self.y)), self.radius, self.color, -1)

    def get_bounds(self):
        """Get object bounds for collision detection"""
        return (
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius,
        )


class FallingObject(GameObject):
    """Red falling circle object"""

    def __init__(self, x, y, speed=2):
        super().__init__(x, y, radius=15, color=(0, 0, 255))  # Red
        self.speed = speed

    def update(self):
        """Move downward"""
        self.y += self.speed


class Bullet(GameObject):
    """Green bullet object"""

    def __init__(self, x, y, speed=8):
        super().__init__(x, y, radius=5, color=(0, 255, 0))  # Green
        self.speed = speed

    def update(self):
        """Move upward"""
        self.y -= self.speed


class Cannon:
    """Blue cannon rectangle"""

    def __init__(self, x, y, width=60, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 0, 0)  # Blue
        self.center_x = x + width // 2

    def update_position(self, new_x, screen_width):
        """Update cannon position with screen bounds checking"""
        self.x = max(0, min(new_x - self.width // 2, screen_width - self.width))
        self.center_x = self.x + self.width // 2

    def draw(self, image):
        """Draw cannon"""
        cv2.rectangle(
            image,
            (int(self.x), int(self.y)),
            (int(self.x + self.width), int(self.y + self.height)),
            self.color,
            -1,
        )

    def get_bullet_spawn_pos(self):
        """Get position where bullets should spawn"""
        return (self.center_x, self.y)


class HeadTiltCannonGameProcessor(Processor):
    """Head tilt controlled cannon game with blink shooting"""

    def __init__(self, lives=3, spawn_rate=0.02, neutral_threshold=15):
        """
        Initialize cannon game

        Args:
            lives: Number of lives (falling objects that can pass)
            spawn_rate: Probability of spawning falling object per frame
            neutral_threshold: Head tilt threshold for neutral position
        """
        self.lives = lives
        self.max_lives = lives
        self.score = 0
        self.spawn_rate = spawn_rate
        self.neutral_threshold = neutral_threshold

        # Game objects
        self.falling_objects = []
        self.bullets = []
        self.cannon = None

        # MediaPipe setup
        self.mp_face = None

        # Eye landmarks for EAR calculation
        self.LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

        # Head tilt landmarks
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 362

        # Blink detection
        self.ear_threshold = 0.25
        self.consecutive_frames = 2
        self.frame_counter = 0
        self.last_shot_frame = -999
        self.shot_cooldown = 10  # Frames between shots

        # Game state
        self.game_over = False
        self.screen_width = 640
        self.screen_height = 480

    def open(self):
        """Initialize MediaPipe and game objects"""
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize cannon at bottom center
        cannon_y = self.screen_height - 40
        self.cannon = Cannon(self.screen_width // 2 - 30, cannon_y)

    def process(self, frame, state):
        """
        Process frame for head tilt cannon game

        Args:
            frame: Frame dict with 'image', 'frame_id', 'timestamp'
            state: Shared state dict

        Returns:
            Result dict with processed image, events, and metrics
        """
        if not self.mp_face:
            return {"image": frame["image"]}

        img = frame["image"]
        self.screen_height, self.screen_width = img.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)

        events = []
        metrics = {}
        img_out = img.copy()

        # Reset game if game over
        if self.game_over:
            if frame["frame_id"] % 180 == 0:  # Reset after 6 seconds at 30fps
                self._reset_game()

        if results.multi_face_landmarks and not self.game_over:
            landmarks = results.multi_face_landmarks[0].landmark

            # Head tilt detection for cannon control
            tilt_angle = self._get_head_tilt_angle(landmarks)
            if tilt_angle is not None:
                self._update_cannon_position(tilt_angle)
                metrics["head_tilt_angle"] = tilt_angle

            # Blink detection for shooting
            shot_fired = self._detect_blink_shot(landmarks, frame["frame_id"])
            if shot_fired:
                events.append("shot")
                self._fire_bullet()

        # Game mechanics (only if not game over)
        if not self.game_over:
            self._spawn_falling_objects()
            self._update_game_objects()
            self._check_collisions()
            self._check_game_over()

        # Render everything
        self._draw_game(img_out)

        metrics.update(
            {"lives": self.lives, "score": self.score, "game_over": self.game_over}
        )

        return {
            "image": img_out,
            "events": events,
            "metrics": metrics,
            "state": {
                "score": self.score,
                "lives": self.lives,
                "game_over": self.game_over,
            },
        }

    def _get_head_tilt_angle(self, landmarks):
        """Calculate head tilt angle using eye corner landmarks"""
        w, h = self.screen_width, self.screen_height

        # Extract eye corner coordinates
        left_eye_outer = landmarks[self.LEFT_EYE_OUTER]
        right_eye_outer = landmarks[self.RIGHT_EYE_OUTER]

        # Convert to pixel coordinates
        left_point = np.array([left_eye_outer.x * w, left_eye_outer.y * h])
        right_point = np.array([right_eye_outer.x * w, right_eye_outer.y * h])

        # Calculate angle
        dx = right_point[0] - left_point[0]
        dy = right_point[1] - left_point[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _update_cannon_position(self, tilt_angle):
        """Update cannon position based on head tilt"""
        # Map tilt angle to cannon position
        # Negative angle = left tilt = move left
        # Positive angle = right tilt = move right

        # Clamp angle to reasonable range
        max_angle = 45
        clamped_angle = max(-max_angle, min(max_angle, tilt_angle))

        # Map angle to screen position
        # Center position when angle is 0
        center_x = self.screen_width // 2
        max_offset = self.screen_width // 3  # Max movement range

        offset = (clamped_angle / max_angle) * max_offset
        new_x = center_x + offset

        self.cannon.update_position(new_x, self.screen_width)

    def _detect_blink_shot(self, landmarks, frame_id):
        """Detect blink for shooting bullets"""
        w, h = self.screen_width, self.screen_height

        # Extract eye landmarks for EAR calculation
        def get_eye_points(indices):
            return np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices]
            )

        left_eye = get_eye_points(self.LEFT_EYE_EAR)
        right_eye = get_eye_points(self.RIGHT_EYE_EAR)

        # Calculate EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Blink detection logic
        shot_fired = False
        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if (
                self.frame_counter >= self.consecutive_frames
                and frame_id - self.last_shot_frame > self.shot_cooldown
            ):
                shot_fired = True
                self.last_shot_frame = frame_id
            self.frame_counter = 0

        return shot_fired

    def _fire_bullet(self):
        """Create a new bullet at cannon position"""
        bullet_x, bullet_y = self.cannon.get_bullet_spawn_pos()
        bullet = Bullet(bullet_x, bullet_y)
        self.bullets.append(bullet)

    def _spawn_falling_objects(self):
        """Randomly spawn falling objects from top"""
        if random.random() < self.spawn_rate:
            x = random.randint(15, self.screen_width - 15)
            falling_obj = FallingObject(x, 0)
            self.falling_objects.append(falling_obj)

    def _update_game_objects(self):
        """Update positions of all game objects"""
        # Update falling objects
        for obj in self.falling_objects[:]:
            obj.update()
            if obj.y > self.screen_height + obj.radius:
                self.falling_objects.remove(obj)
                self.lives -= 1  # Lost a life

        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.y < -bullet.radius:
                self.bullets.remove(bullet)

    def _check_collisions(self):
        """Check collisions between bullets and falling objects"""
        for bullet in self.bullets[:]:
            for obj in self.falling_objects[:]:
                if self._objects_collide(bullet, obj):
                    # Hit! Remove both objects and increase score
                    self.bullets.remove(bullet)
                    self.falling_objects.remove(obj)
                    self.score += 1
                    break

    def _objects_collide(self, obj1, obj2):
        """Check if two circular objects collide"""
        distance = math.sqrt((obj1.x - obj2.x) ** 2 + (obj1.y - obj2.y) ** 2)
        return distance < (obj1.radius + obj2.radius)

    def _check_game_over(self):
        """Check if game is over (no lives left)"""
        if self.lives <= 0:
            self.game_over = True

    def _reset_game(self):
        """Reset game to initial state"""
        self.lives = self.max_lives
        self.score = 0
        self.falling_objects.clear()
        self.bullets.clear()
        self.game_over = False

        # Reset cannon position
        cannon_y = self.screen_height - 40
        self.cannon = Cannon(self.screen_width // 2 - 30, cannon_y)

    def _draw_game(self, image):
        """Draw all game elements"""
        # Draw cannon
        if self.cannon:
            self.cannon.draw(image)

        # Draw falling objects
        for obj in self.falling_objects:
            obj.draw(image)

        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(image)

        # Draw UI
        self._draw_ui(image)

    def _draw_ui(self, image):
        """Draw game UI (score, lives, instructions)"""
        # Score
        cv2.putText(
            image,
            f"Score: {self.score}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Lives
        cv2.putText(
            image,
            f"Lives: {self.lives}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Instructions
        if not self.game_over:
            cv2.putText(
                image,
                "Tilt head to move, blink to shoot",
                (10, image.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "Red circles = targets, Blue = cannon, Green = bullets",
                (10, image.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Game over screen
        if self.game_over:
            # Semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(
                overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # Game over text
            cv2.putText(
                image,
                "GAME OVER",
                (image.shape[1] // 2 - 100, image.shape[0] // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                image,
                f"Final Score: {self.score}",
                (image.shape[1] // 2 - 120, image.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                "Restarting in 6 seconds...",
                (image.shape[1] // 2 - 150, image.shape[0] // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    def close(self):
        """Clean up MediaPipe resources"""
        if self.mp_face:
            self.mp_face.close()


# Register the cannon game
@register_game("cannon_game")
def make_cannon_game():
    """Factory function to create cannon game"""
    processors = [HeadTiltCannonGameProcessor(lives=3, spawn_rate=0.02)]
    store = SimpleStore()
    return processors, store


@register_game("cannon_game_hard")
def make_cannon_game_hard():
    """Factory function to create harder cannon game"""
    processors = [HeadTiltCannonGameProcessor(lives=1, spawn_rate=0.035)]
    store = SimpleStore()
    return processors, store


@register_game("cannon_game_easy")
def make_cannon_game_easy():
    """Factory function to create easier cannon game"""
    processors = [HeadTiltCannonGameProcessor(lives=5, spawn_rate=0.015)]
    store = SimpleStore()
    return processors, store
